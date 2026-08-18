"""Streaming GLP training with native post_process implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import json
import logging
import math
import os
import random
from pathlib import Path
import shutil
from typing import Iterable, Iterator, List, Optional

try:
    from baukit import TraceDict
except ImportError:
    TraceDict = None

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from .activation_stream import MemmapReader, MemmapWriter, RunningMoments, iter_stream_texts, parse_storage_dtype
from .paths import get_data_root
from .glp import GLP, _canonicalize_normalization_method


LOGGER = logging.getLogger(__name__)
CHECKPOINT_FILES = ("config.yaml", "rep_statistics.pt", "final.safetensors")


@dataclass
class StreamTrainConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    layer: int = 7
    layer_prefix: str = "model.layers"
    retain: str = "output"
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    storage_dtype: str = "bfloat16"

    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: Optional[str] = "sample-10BT"
    dataset_split: str = "train"
    text_field: str = "text"
    max_documents: Optional[int] = 50_000

    max_length: int = 2048
    token_idx: str = "all"
    seed: Optional[int] = 42
    drop_bos: bool = True
    padding_side: str = "right"
    document_batch_size: int = 16
    forward_batch_size: int = 1

    stream_chunk_size: int = 1_000_000
    num_epochs: int = 1
    total_steps: int = 244
    batch_size: int = 4096
    learning_rate: float = 5e-5
    normalization_method: str = "gaussian"
    noise_sampling_method: str = "uniform"
    u_sampling_method: str = "uniform"
    ot_chunk_size: int = 256
    gradient_clipping_threshold: float = 1.0
    grad_accum: int = 1
    log_every_n_steps: int = 10
    tail_variance_proportion: float = 0.05
    split: bool = False
    split_proportion: float = 0.1
    warmup_ratio: float = 0.01
    initial_factor: float = 0.01
    final_factor: float = 0.1
    use_bf16: bool = True
    shuffle: bool = True
    init_ckpt: Optional[str] = None
    load_opt: bool = False
    scheduler_type: str = "cosine"

    save_root: str = str(get_data_root() / "GLP")
    run_name: str = "glp-stream"
    checkpoint_token_step: int = 100_000_000
    denoiser_layers: int = 3
    d_model_mult: int = 2
    d_mlp_mult: int = 4
    phase_switch: bool = False
    offload_device: str = "cpu"
    # subspace_k removed; training operates in full activation space

    wandb: bool = False
    wandb_project: str = "glp"
    cache_dataset: bool = False


class _FineWebLike:
    def __init__(self, cfg: StreamTrainConfig):
        self.dataset_name = cfg.dataset_name
        self.dataset_config = cfg.dataset_config
        self.split = cfg.dataset_split
        self.text_field = cfg.text_field
        self.streaming = True
        self.max_documents = cfg.max_documents


class ActDataset(Dataset):
    def __init__(self, reader: MemmapReader):
        self.reader = reader

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, idx: int) -> dict:
        latents = torch.tensor(self.reader[idx])[None, :]
        latents = latents.view(torch.bfloat16) if latents.dtype == torch.int16 else latents
        return {"activations": latents.float()}


class ActivationCollator:
    def __init__(self, normalizer):
        self.normalizer = normalizer

    @torch.no_grad()
    def __call__(self, rows: list[dict]) -> dict:
        latents = torch.stack([row["activations"] for row in rows], dim=0)
        # Return raw latents; GLP.forward() handles: project → normalize → flow
        return {"latents": latents}


def _batch_items(items: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    if batch_size < 1:
        raise ValueError("document_batch_size must be >= 1")
    bucket: list[str] = []
    for item in items:
        bucket.append(" ".join(item.split()))
        if len(bucket) >= batch_size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested (%s) but unavailable; falling back to CPU.", requested_device)
        return "cpu"
    return requested_device


def _load_model_and_tokenizer(cfg: StreamTrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=_parse_torch_dtype(cfg.torch_dtype),
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(_resolve_device(cfg.device))
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
        if model.generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def _build_tracedict_config(cfg: StreamTrainConfig) -> dict:
    return {
        "layer_prefix": cfg.layer_prefix,
        "layers": [cfg.layer],
        "retain": cfg.retain,
    }


@torch.no_grad()
def _save_acts(
    hf_model,
    hf_tokenizer,
    text: list[str],
    tracedict_config: dict,
    padding_side: str,
    token_idx: str,
    batch_size: int,
    max_length: int,
):
    if TraceDict is None:
        raise ImportError(
            "Missing dependency 'baukit'. Install with: pip install git+https://github.com/davidbau/baukit"
        )
    cfg = dict(tracedict_config)
    retain_attr = cfg.pop("retain")
    if retain_attr not in {"input", "output"}:
        raise ValueError(f"Unsupported retain={retain_attr}")
    cfg[f"retain_{retain_attr}"] = True
    layer_prefix = cfg.pop("layer_prefix", None)
    if layer_prefix is not None:
        cfg["layers"] = [f"{layer_prefix}.{layer}" for layer in cfg["layers"]]
    if hf_tokenizer.padding_side != padding_side:
        hf_tokenizer.padding_side = padding_side

    outputs = []
    for start in tqdm(range(0, len(text), batch_size), leave=False, desc="Extract acts"):
        minibatch = hf_tokenizer(
            text[start : start + batch_size],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        minibatch = {k: v.to(hf_model.device) for k, v in minibatch.items()}
        max_positions = getattr(hf_model.config, "max_position_embeddings", max_length)
        if minibatch["input_ids"].shape[1] > max_positions:
            minibatch = {k: v[:, :max_positions] for k, v in minibatch.items()}
        vocab_size = getattr(hf_model.config, "vocab_size", None)
        if vocab_size is not None:
            minibatch["input_ids"] = minibatch["input_ids"].clamp(0, vocab_size - 1)
        with TraceDict(hf_model, **cfg) as miniret:
            hf_model(**minibatch, use_cache=False)
        retained = [getattr(miniret[layer], retain_attr) for layer in cfg["layers"]]
        retained = [value[0] if isinstance(value, tuple) else value for value in retained]
        retained = torch.stack(retained)
        retained = einops.rearrange(retained, "l b s d -> b l s d")
        if token_idx == "last":
            last_token_idx = -1 if padding_side == "left" else (minibatch["attention_mask"].sum(dim=1) - 1)
            retained = retained[torch.arange(retained.shape[0]), :, last_token_idx, :].detach().cpu()
        elif token_idx == "all":
            retained = retained.detach().cpu()
        else:
            raise NotImplementedError(f"Unsupported token_idx={token_idx}")
        outputs.append(retained)

    if token_idx == "all" and len(outputs) > 1:
        max_seq_len = max(t.shape[2] for t in outputs)
        if any(t.shape[2] != max_seq_len for t in outputs):
            outputs = [
                F.pad(t, (0, 0, 0, max_seq_len - t.shape[2])) if t.shape[2] != max_seq_len else t
                for t in outputs
            ]
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0)


def _sample_random_token_per_document(
    activations: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    drop_bos: bool,
    rng: np.random.Generator,
) -> torch.Tensor:
    vectors = activations[:, 0, :, :]
    picks = []
    for idx in range(vectors.shape[0]):
        valid_len = int(attention_mask[idx].sum().item())
        start = 1 if drop_bos else 0
        if valid_len <= start:
            continue
        token_index = int(rng.integers(start, valid_len))
        picks.append(vectors[idx, token_index, :])
    if not picks:
        return torch.empty((0, vectors.shape[-1]), dtype=vectors.dtype)
    return torch.stack(picks, dim=0)


def _flatten_layer_activations(activations: torch.Tensor, *, drop_bos: bool) -> torch.Tensor:
    if activations.ndim == 4:
        vectors = activations[:, 0, :, :]
        if drop_bos and vectors.shape[1] > 0:
            vectors = vectors[:, 1:, :]
        return vectors.reshape(-1, vectors.shape[-1])
    if activations.ndim == 3:
        return activations[:, 0, :].reshape(-1, activations.shape[-1])
    raise ValueError(f"Unexpected activation tensor rank: {activations.ndim}")


def _extract_activation_vectors(
    *,
    hf_model,
    hf_tokenizer,
    text_batch: list[str],
    cfg: StreamTrainConfig,
    tracedict_config: dict,
    rng: np.random.Generator,
) -> torch.Tensor:
    save_token_idx = "all" if cfg.token_idx == "random_doc" else cfg.token_idx
    activations = _save_acts(
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        text=text_batch,
        tracedict_config=tracedict_config,
        padding_side=cfg.padding_side,
        token_idx=save_token_idx,
        batch_size=cfg.forward_batch_size,
        max_length=cfg.max_length,
    )
    if cfg.token_idx == "random_doc":
        tokenized = hf_tokenizer(
            text_batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=cfg.max_length,
        )
        return _sample_random_token_per_document(
            activations,
            tokenized["attention_mask"],
            drop_bos=cfg.drop_bos,
            rng=rng,
        )
    return _flatten_layer_activations(activations, drop_bos=cfg.drop_bos)


def _to_storage_array(vectors: torch.Tensor, storage_dtype: str) -> np.ndarray:
    if storage_dtype == "bfloat16":
        return vectors.to(torch.bfloat16).view(torch.int16).cpu().numpy()
    if storage_dtype == "float16":
        return vectors.to(torch.float16).cpu().numpy()
    if storage_dtype == "float32":
        return vectors.to(torch.float32).cpu().numpy()
    raise ValueError(f"Unsupported storage_dtype: {storage_dtype}")


def _write_vectors_to_memmap(writer: MemmapWriter, vectors: torch.Tensor, storage_dtype: str) -> int:
    if vectors.numel() == 0 or vectors.shape[0] == 0:
        return 0
    storage_vectors = _to_storage_array(vectors, storage_dtype)
    for row in storage_vectors:
        writer.write(np.ascontiguousarray(row))
    return int(storage_vectors.shape[0])


def _load_activation_dataset(dataset_path: str):
    path = Path(dataset_path)
    dtype = np.dtype((path / "dtype.txt").read_text().strip().replace("np.", ""))
    return ConcatDataset([ActDataset(reader=MemmapReader(path, dtype))])

# Subspace/PCA building removed — training GLP operates on full activation space.
# The legacy `_build_subspace_from_memmap` function was intentionally removed
# to avoid any PCA/subspace preprocessing during GLP training.

def _get_activation_dataloader(dataset, batch_size: int, normalizer, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=ActivationCollator(normalizer),
        num_workers=0,
        pin_memory=False,
    )


def _linear_scheduler(step, max_steps, initial_factor, final_factor):
    alpha = step / max_steps
    return alpha * final_factor + (1 - alpha) * initial_factor


def _cosine_scheduler(step, max_steps, initial_factor, final_factor):
    alpha = step / max_steps
    cosine_out = 0.5 * (1 + math.cos(math.pi * alpha))
    return final_factor + (initial_factor - final_factor) * cosine_out


def _cosine_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    if step < warmup_steps:
        return _linear_scheduler(step, max(warmup_steps, 1), initial_factor, 1.0)
    if step >= max_steps:
        return final_factor
    return _cosine_scheduler(step - warmup_steps, max(max_steps - warmup_steps, 1), 1.0, final_factor)


def _normalization_requires_stats(method: str) -> bool:
    return _canonicalize_normalization_method(method) != "log_norm"


def _quantile_percent_from_method(method: str) -> float | None:
    method = _canonicalize_normalization_method(method)
    if method.startswith("quantile_"):
        return float(method.split("_", 1)[1])
    return None


def _is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def _get_module_device(module: torch.nn.Module) -> str:
    for param in module.parameters():
        return str(param.device)
    for buffer in module.buffers():
        return str(buffer.device)
    return "cpu"


def _move_model_to_device(module: torch.nn.Module, target_device: str, module_name: str) -> None:
    current_device = _get_module_device(module)
    if current_device == str(target_device):
        return
    LOGGER.info("Moving %s from %s to %s.", module_name, current_device, target_device)
    module.to(target_device)


def _cleanup_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _split_hf_checkpoint_ref(checkpoint_ref: str) -> tuple[str, str]:
    parts = checkpoint_ref.strip("/").split("/")
    if len(parts) <= 2:
        return checkpoint_ref, ""
    return "/".join(parts[:2]), "/".join(parts[2:])


def _resolve_init_checkpoint(init_ckpt: Optional[str]) -> Optional[Path]:
    if not init_ckpt:
        return None
    resolved = Path(init_ckpt).expanduser()
    if resolved.exists():
        return resolved
    from huggingface_hub import snapshot_download

    repo_id, subfolder = _split_hf_checkpoint_ref(init_ckpt)
    allow_patterns = [f"{subfolder}/{name}" if subfolder else name for name in CHECKPOINT_FILES]
    resolved = Path(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns))
    return resolved / subfolder if subfolder else resolved


def _setup_glp_model(hidden_size: int, cfg: StreamTrainConfig) -> GLP:
    # Denoiser operates on full activation dimension (no subspace projection).
    denoiser_dim = hidden_size
    return GLP(
        normalizer_config={
            "rep_statistic": "",
            "d_input": hidden_size,
            "normalization_method": cfg.normalization_method,
        },
        denoiser_config={
            "d_input": denoiser_dim,
            "d_model": cfg.d_model_mult * denoiser_dim,
            "d_mlp": cfg.d_mlp_mult * denoiser_dim,
            "n_layers": cfg.denoiser_layers,
            "multi_layer_n_layers": None,
        },
        noise_sampling_method=cfg.noise_sampling_method,
        u_sampling_method=cfg.u_sampling_method,
        ot_chunk_size=cfg.ot_chunk_size,
        tracedict_config=_build_tracedict_config(cfg),
    )


def _setup_init_glp_model(init_dir: Path, hidden_size: int, cfg: StreamTrainConfig) -> GLP:
    config = yaml.safe_load((init_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    glp_kwargs = dict(config.get("glp_kwargs", {}))
    if not glp_kwargs:
        raise ValueError(f"Checkpoint config at {init_dir / 'config.yaml'} does not contain glp_kwargs.")
    normalizer_config = dict(glp_kwargs.get("normalizer_config", {}))
    rep_stats_path = init_dir / "rep_statistics.pt"
    normalizer_config["rep_statistic"] = str(rep_stats_path) if rep_stats_path.exists() else ""
    normalizer_config["d_input"] = hidden_size
    normalizer_config["normalization_method"] = normalizer_config.get(
        "normalization_method",
        cfg.normalization_method,
    )
    return GLP(
        normalizer_config=normalizer_config,
        denoiser_config=glp_kwargs["denoiser_config"],
        noise_sampling_method=glp_kwargs.get("noise_sampling_method", cfg.noise_sampling_method),
        u_sampling_method=glp_kwargs.get("u_sampling_method", cfg.u_sampling_method),
        ot_chunk_size=glp_kwargs.get("ot_chunk_size", cfg.ot_chunk_size),
        tracedict_config=_build_tracedict_config(cfg),
    )


def _format_token_count(token_count: int) -> str:
    if token_count >= 1_000_000 and token_count % 1_000_000 == 0:
        return f"{token_count // 1_000_000}M"
    if token_count >= 1_000 and token_count % 1_000 == 0:
        return f"{token_count // 1_000}K"
    return str(token_count)


def _save_glp_checkpoint(
    *,
    save_dir: Path,
    cfg: StreamTrainConfig,
    glp_model: GLP,
    hidden_size: int,
    normalization_method: str,
    global_step: int,
    total_tokens_collected: int,
    checkpoint_name: str = "final",
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    # Store checkpoints using the standard GLP filename so load_glp() can
    # resolve both milestone and final checkpoints consistently.
    glp_model.save_pretrained(path=save_dir, name="final")
    if _normalization_requires_stats(normalization_method):
        torch.save(
            {
                "mean": glp_model.normalizer.mean.cpu(),
                "var": glp_model.normalizer.var.cpu(),
                "normalization_method": normalization_method,
            },
            save_dir / "rep_statistics.pt",
        )

    denoiser_model = glp_model.denoiser.model
    config_dict = {
        "model_name": cfg.model_name,
        "glp_kwargs": {
            "normalizer_config": {
                "rep_statistic": "rep_statistics.pt",
                "d_input": hidden_size,
                "normalization_method": normalization_method,
            },
            "denoiser_config": {
                "d_input": denoiser_model.d_input,
                "d_model": denoiser_model.d_model,
                "d_mlp": denoiser_model.d_mlp,
                "n_layers": denoiser_model.n_layers,
                "multi_layer_n_layers": denoiser_model.multi_layer_n_layers,
                "split": getattr(denoiser_model, "split", False),
                "split_tail_indices": list(getattr(denoiser_model, "split_tail_indices", [])),
            },
            "noise_sampling_method": glp_model.noise_sampling_method,
            "u_sampling_method": glp_model.u_sampling_method,
            "ot_chunk_size": glp_model.ot_chunk_size,
            "tracedict_config": glp_model.tracedict_config,
        },
        "stream_train_config": asdict(cfg),
    }
    with open(save_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_dict, handle, sort_keys=False)


def _linear_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    """Linear scheduler with warmup phase."""
    if step < warmup_steps:
        alpha = step / max(warmup_steps, 1)
        return initial_factor + (1.0 - initial_factor) * alpha
    else:
        alpha = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 1.0 + (final_factor - 1.0) * alpha


def stream_train(cfg: StreamTrainConfig) -> dict:
    # Setup seed for reproducibility
    if cfg.seed > 0:
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    
    cfg.device = _resolve_device(cfg.device)
    normalization_method = _canonicalize_normalization_method(cfg.normalization_method)
    
    # Calculate total training steps based on epochs
    steps_per_epoch = max(1, cfg.total_steps)
    total_training_steps = steps_per_epoch * cfg.num_epochs

    phase_switch = bool(cfg.phase_switch)
    if phase_switch and not _is_cuda_device(cfg.device):
        LOGGER.warning("Phase switching disabled because training device is %s.", cfg.device)
        phase_switch = False
    if phase_switch and cfg.offload_device == cfg.device:
        LOGGER.warning("Phase switching disabled because offload_device matches device.")
        phase_switch = False

    hf_model, hf_tokenizer = _load_model_and_tokenizer(cfg)
    hidden_size = int(hf_model.config.hidden_size)
    init_dir = _resolve_init_checkpoint(cfg.init_ckpt)
    if init_dir is not None:
        glp_model = _setup_init_glp_model(init_dir, hidden_size, cfg).to(cfg.device)
        glp_model.load_pretrained(init_dir, name="final")
    else:
        glp_model = _setup_glp_model(hidden_size, cfg).to(cfg.device)

    optimizer = None
    scheduler = None

    def create_optimizer_and_scheduler():
        opt = torch.optim.AdamW(glp_model.parameters(), lr=cfg.learning_rate)
        warmup_steps = int(cfg.warmup_ratio * total_training_steps)
        
        # Setup scheduler based on type
        if cfg.scheduler_type == "linear":
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=partial(
                    _linear_scheduler_with_warmup,
                    warmup_steps=warmup_steps,
                    max_steps=total_training_steps,
                    initial_factor=cfg.initial_factor,
                    final_factor=cfg.final_factor,
                ),
            )
        else:  # cosine
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=partial(
                    _cosine_scheduler_with_warmup,
                    warmup_steps=warmup_steps,
                    max_steps=total_training_steps,
                    initial_factor=cfg.initial_factor,
                    final_factor=cfg.final_factor,
                ),
            )
        return opt, sched

    wandb_run = None
    if cfg.wandb:
        import wandb  # type: ignore

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg))

    use_stats = _normalization_requires_stats(normalization_method)
    use_gaussian_stats = normalization_method == "gaussian"
    use_rmsnorm_stats = normalization_method == "rmsnorm"
    use_iqr_stats = normalization_method == "iqr"
    quantile_percent = _quantile_percent_from_method(normalization_method)
    use_quantile_stats = quantile_percent is not None

    stats = RunningMoments(hidden_size) if (use_gaussian_stats or use_quantile_stats) else None
    second_moment_sum = np.zeros(hidden_size, dtype=np.float64) if use_rmsnorm_stats else None
    second_moment_count = 0
    iqr_q25 = np.zeros(hidden_size, dtype=np.float64) if use_iqr_stats else None
    iqr_median = np.zeros(hidden_size, dtype=np.float64) if use_iqr_stats else None
    iqr_q75 = np.zeros(hidden_size, dtype=np.float64) if use_iqr_stats else None
    iqr_count = 0
    quantile_scale = np.ones(hidden_size, dtype=np.float64) if use_quantile_stats else None
    quantile_count = 0

    tmp_dir = Path(get_data_root()) / f"tmp_stream_{cfg.run_name}"
    
    # Create seed-aware RNG for epoch-level reproducibility
    if cfg.seed > 0:
        rng = np.random.default_rng(cfg.seed)
    else:
        rng = np.random.default_rng()
    
    tracedict_config = _build_tracedict_config(cfg)
    use_autocast = bool(cfg.use_bf16 and _is_cuda_device(cfg.device))

    global_step = 0
    total_tokens_collected = 0
    next_checkpoint_target = cfg.checkpoint_token_step if cfg.checkpoint_token_step else float("inf")
    progress = tqdm(total=total_training_steps, desc="Streaming GLP")

    try:
        for epoch in range(cfg.num_epochs):
            if global_step >= total_training_steps:
                break
                
            if cfg.seed > 0:
                import random
                random.seed(cfg.seed)
                np.random.seed(cfg.seed)
                torch.manual_seed(cfg.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(cfg.seed)
            
            # Reset batch iterator for each epoch
            batch_iterator = _batch_items(iter_stream_texts(_FineWebLike(cfg)), cfg.document_batch_size)
            epoch_step = 0
            
            while epoch_step < steps_per_epoch and global_step < total_training_steps:
                if phase_switch:
                    _move_model_to_device(glp_model, cfg.offload_device, "GLP model")
                    _move_model_to_device(hf_model, cfg.device, "Extractor model")
                    _cleanup_cuda_cache()

                need_collect = True
                if cfg.cache_dataset and tmp_dir.exists() and (tmp_dir / "dtype.txt").exists():
                    need_collect = False

                if need_collect:
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir)
                    tmp_dir.mkdir(parents=True, exist_ok=True)

                    stream_storage_dtype = cfg.storage_dtype
                    file_size = cfg.stream_chunk_size * hidden_size
                    np_dtype, dtype_label = parse_storage_dtype(stream_storage_dtype)
                    writer = MemmapWriter(output_dir=tmp_dir, file_size=file_size, dtype=np_dtype)
                    (tmp_dir / "dtype.txt").write_text(dtype_label)

                    vectors_written = 0
                    while vectors_written < cfg.stream_chunk_size and epoch_step < steps_per_epoch and global_step < total_training_steps:
                        text_batch = next(batch_iterator, None)
                        if not text_batch:
                            LOGGER.warning("Streaming dataset exhausted.")
                            break
                        vectors = _extract_activation_vectors(
                            hf_model=hf_model,
                            hf_tokenizer=hf_tokenizer,
                            text_batch=text_batch,
                            cfg=cfg,
                            tracedict_config=tracedict_config,
                            rng=rng,
                        )
                        if vectors.numel() == 0:
                            continue
                        remaining = cfg.stream_chunk_size - vectors_written
                        vectors = vectors[:remaining]
                        if vectors.numel() == 0:
                            continue

                        if use_stats:
                            vectors_np = vectors.detach().float().cpu().numpy().astype(np.float64, copy=False)
                            if stats is not None:
                                stats.update(vectors_np)
                            if use_rmsnorm_stats and second_moment_sum is not None:
                                second_moment_sum += np.square(vectors_np).sum(axis=0)
                                second_moment_count += vectors_np.shape[0]
                            if use_iqr_stats and iqr_q25 is not None and iqr_median is not None and iqr_q75 is not None:
                                batch_count = vectors_np.shape[0]
                                chunk_q25 = np.percentile(vectors_np, 25, axis=0)
                                chunk_median = np.percentile(vectors_np, 50, axis=0)
                                chunk_q75 = np.percentile(vectors_np, 75, axis=0)
                                if iqr_count == 0:
                                    iqr_q25 = chunk_q25
                                    iqr_median = chunk_median
                                    iqr_q75 = chunk_q75
                                else:
                                    total = iqr_count + batch_count
                                    iqr_q25 = (iqr_q25 * iqr_count + chunk_q25 * batch_count) / total
                                    iqr_median = (iqr_median * iqr_count + chunk_median * batch_count) / total
                                    iqr_q75 = (iqr_q75 * iqr_count + chunk_q75 * batch_count) / total
                                iqr_count += batch_count
                            if use_quantile_stats and quantile_scale is not None and quantile_percent is not None and stats is not None:
                                centered_vectors = vectors_np - stats.mean
                                chunk_q = np.percentile(np.abs(centered_vectors), quantile_percent, axis=0)
                                batch_count = vectors_np.shape[0]
                                if quantile_count == 0:
                                    quantile_scale = chunk_q
                                else:
                                    total = quantile_count + batch_count
                                    quantile_scale = (quantile_scale * quantile_count + chunk_q * batch_count) / total
                                quantile_count += batch_count

                        written = _write_vectors_to_memmap(writer, vectors, cfg.storage_dtype)
                        vectors_written += written

                    writer.flush()
                    if vectors_written == 0:
                        break

                    normalizer_device = glp_model.normalizer.mean.device
                    if use_gaussian_stats and stats is not None:
                        mean, var = stats.finalize()
                        glp_model.normalizer.mean = torch.tensor(mean, dtype=torch.float32, device=normalizer_device)
                        glp_model.normalizer.var = torch.tensor(var, dtype=torch.float32, device=normalizer_device)
                    elif use_rmsnorm_stats and second_moment_sum is not None:
                        rms_sq = np.maximum(second_moment_sum / max(second_moment_count, 1), 1e-8)
                        glp_model.normalizer.mean = torch.zeros(hidden_size, dtype=torch.float32, device=normalizer_device)
                        glp_model.normalizer.var = torch.tensor(rms_sq, dtype=torch.float32, device=normalizer_device)
                    elif use_iqr_stats and iqr_median is not None and iqr_q25 is not None and iqr_q75 is not None:
                        iqr = np.maximum(iqr_q75 - iqr_q25, 1e-6)
                        glp_model.normalizer.mean = torch.tensor(iqr_median, dtype=torch.float32, device=normalizer_device)
                        glp_model.normalizer.var = torch.tensor(iqr * iqr, dtype=torch.float32, device=normalizer_device)
                    elif use_quantile_stats and quantile_scale is not None and stats is not None:
                        scale = np.maximum(quantile_scale, 1e-6)
                        glp_model.normalizer.mean = torch.tensor(stats.mean, dtype=torch.float32, device=normalizer_device)
                        glp_model.normalizer.var = torch.tensor(scale * scale, dtype=torch.float32, device=normalizer_device)

                    if optimizer is None:
                        if cfg.split and not glp_model.denoiser.model.split:
                            if use_stats:
                                split_tail_indices = glp_model.configure_split_output_from_normalizer(
                                    proportion=cfg.split_proportion
                                )
                                LOGGER.info("Configured split output with %d dims", len(split_tail_indices))
                else:
                    _temp_ds = _load_activation_dataset(str(tmp_dir))
                    vectors_written = len(_temp_ds)
                    if vectors_written == 0:
                        break

                if phase_switch:
                    _move_model_to_device(hf_model, cfg.offload_device, "Extractor model")
                    _move_model_to_device(glp_model, cfg.device, "GLP model")
                    _cleanup_cuda_cache()
                    
                if optimizer is None:
                    LOGGER.info("Creating optimizer and scheduler") 
                    optimizer, scheduler = create_optimizer_and_scheduler()

                train_dataset = _load_activation_dataset(str(tmp_dir))
                train_dataloader = _get_activation_dataloader(
                    dataset=train_dataset,
                    batch_size=cfg.batch_size,
                    normalizer=glp_model.normalizer,
                    shuffle=cfg.shuffle,
                )

                glp_model.train()
                accumulated_loss = 0.0
                for batch_idx, batch in enumerate(train_dataloader):
                    if epoch_step >= steps_per_epoch or global_step >= total_training_steps:
                        break
                    batch = {k: v.to(cfg.device) if v is not None else None for k, v in batch.items()}
                    loss_kwargs = {
                        "tail_variance_proportion": cfg.tail_variance_proportion,
                    }
                    with torch.autocast(
                        device_type="cuda" if use_autocast else "cpu",
                        dtype=torch.bfloat16,
                        enabled=use_autocast,
                    ):
                        outputs = glp_model(
                            **batch,
                            global_step=global_step,
                            total_steps=total_training_steps,
                            loss_kwargs=loss_kwargs,
                        )
                        loss = outputs.loss
                    
                    # Gradient accumulation
                    loss = loss / cfg.grad_accum
                    loss.backward()
                    accumulated_loss += loss.item()

                    # Step after gradient accumulation
                    if (batch_idx + 1) % cfg.grad_accum == 0:
                        max_grad_norm = (
                            cfg.gradient_clipping_threshold
                            if cfg.gradient_clipping_threshold > 0.0
                            else float("inf")
                        )
                        grad_norm = torch.nn.utils.clip_grad_norm_(glp_model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1
                        epoch_step += 1
                        progress.update(1)
                        accumulated_loss = 0.0
                        progress.set_description(
                            f"Epoch {epoch + 1}/{cfg.num_epochs} | "
                            f"Step {global_step}/{total_training_steps} | "
                            f"Loss: {loss.item():.4f}"
                        )
                        
                        if wandb_run and global_step % max(1, cfg.log_every_n_steps) == 0:
                            wandb_run.log(
                                {
                                    "train/loss": loss.item(),
                                    "train/loss_rel": outputs.loss_rel.item(),
                                    "train/loss_early": float(outputs.loss_early),
                                    "train/loss_mid": float(outputs.loss_mid),
                                    "train/loss_late": float(outputs.loss_late),
                                    "train/loss_raw": outputs.loss_raw.item(),
                                    "train/grad_norm": float(grad_norm.detach().float().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                                    "train/cos_sim": outputs.cos_sim.item(),
                                    "train/target_norm": outputs.tgt_norm.item(),
                                    "train/latent_pre_l2": outputs.latent_pre_l2.item(),
                                    "train/latent_post_l2": outputs.latent_post_l2.item(),
                                    "train/latent_pre_l1": outputs.latent_pre_l1.item(),
                                    "train/latent_post_l1": outputs.latent_post_l1.item(),
                                    "train/batch_mean": outputs.batch_mean.item(),
                                    "train/batch_var_max": outputs.batch_var.item(),
                                    "train/global_mean": outputs.global_mean.item(),
                                    "train/global_var_max": outputs.global_var.item(),
                                    "train/tail_weight_mean": outputs.tail_weight_mean.item(),
                                    "train/tail_weight_max": outputs.tail_weight_max.item(),
                                    "train/tail_weighted_mse": outputs.tail_weighted_mse.item(),
                                    "train/tail_region_mse": outputs.tail_region_mse.item(),
                                    "train/non_tail_region_mse": outputs.non_tail_region_mse.item(),
                                    "train/lr_adamw": scheduler.get_last_lr()[0],
                                },
                                step=global_step,
                            )
                        
                        # step-based checkpointing disabled; rely on token-step or epoch checkpoints

                total_tokens_collected += vectors_written

                # Token-based checkpointing: save as soon as the threshold is crossed.
                while total_tokens_collected >= next_checkpoint_target:
                    checkpoint_name = _format_token_count(int(next_checkpoint_target))
                    milestone_dir = Path(cfg.save_root) / cfg.run_name / checkpoint_name
                    print(f"Saving checkpoint at {milestone_dir} after collecting {total_tokens_collected} tokens...")
                    _save_glp_checkpoint(
                        save_dir=milestone_dir,
                        cfg=cfg,
                        glp_model=glp_model,
                        hidden_size=hidden_size,
                        normalization_method=normalization_method,
                        global_step=global_step,
                        total_tokens_collected=total_tokens_collected,
                        checkpoint_name=checkpoint_name,
                    )
                    next_checkpoint_target += cfg.checkpoint_token_step

    finally:
        progress.close()
        if wandb_run:
            wandb_run.finish()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    out_dir = Path(cfg.save_root) / cfg.run_name
    _save_glp_checkpoint(
        save_dir=out_dir,
        cfg=cfg,
        glp_model=glp_model,
        hidden_size=hidden_size,
        normalization_method=normalization_method,
        global_step=global_step,
        total_tokens_collected=total_tokens_collected,
        checkpoint_name="final",
    )
    summary = {
        "output_dir": str(out_dir),
        "steps": global_step,
        "tokens_collected": total_tokens_collected,
        "layer": cfg.layer,
        "normalization_method": normalization_method,
        "noise_sampling_method": cfg.noise_sampling_method,
        "u_sampling_method": cfg.u_sampling_method,
    }
    with open(out_dir / "stream_train_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Stream training complete: %s", summary)
    return summary
