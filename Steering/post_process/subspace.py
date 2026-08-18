"""Subspace GLP — concept A→B flow matching in a centered PCA subspace.

Training pipeline
-----------------
Phase 0  build_subspace()          — stack A/B, center, SVD → fixed P
Phase 1  train_subspace_glp()      — full training, reuses GLP(d_input=k) unchanged

Inference
---------
SubspaceGLP.steer(h_A) → h_B      — project → ODE → restore null space
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from .glp import (
    GLP,
    _canonicalize_noise_sampling_method,
    _canonicalize_normalization_method,
    _canonicalize_u_sampling_method,
    _sample_training_u,
    load_glp,
    match_noise_to_latents_ot,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class SubspaceGLPConfig:
    model_name: str = "google/gemma-2-2b-it"
    layer: int = 14
    hook_point: str = "post"
    pooling: str = "last"
    device: str = "cuda:0"
    model_dtype: str = "bfloat16"

    dataset_name: str = "refusal_cast_responses"
    n_samples: Optional[int] = None
    prompt_batch_size: int = 16
    act_cache_dir: str = "data/subspace_raw_activations"
    reextract_acts: bool = False

    subspace_k: int = 64
    weight: str = "svd"
    max_weight: float = 30.0

    # Synthetic augmentation factor. Repeat each paired activation `aug` times.
    aug: int = 1

    total_steps: int = 10
    num_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    loss: str = "mse"
    warmup_ratio: float = 0.05
    initial_factor: float = 0.01
    final_factor: float = 0.1
    gradient_clip: float = 1.0
    grad_accum: int = 1
    noise_sampling_method: str = "sot"
    u_sampling_method: str = "logit_normal"
    ot_chunk_size: int = 256
    normalization_method: str = "gaussian"
    use_bf16: bool = True
    shuffle: bool = True
    log_every_n_steps: int = 50
    seed: Optional[int] = None
    scheduler_type: str = "cosine"

    denoiser_n_layers: int = 3
    d_model_mult: int = 4
    d_mlp_mult: int = 8
    init_ckpt: Optional[str] = None

    save_root: str = "."
    run_name: str = "subspace-glp"
    checkpoint_token_step: int = 100000

    wandb: bool = True
    wandb_project: str = "flow"


def build_subspace(
    acts_A: torch.Tensor,
    acts_B: torch.Tensor,
    k: int,
    weight_mode: str = "svd",
    max_weight: float = 30.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build a centered PCA subspace from stacked concept activations."""
    acts_A = acts_A.float()
    acts_B = acts_B.float()

    def _squeeze(acts: torch.Tensor) -> torch.Tensor:
        if acts.ndim == 3 and acts.shape[1] == 1:
            return acts.squeeze(1)
        if acts.ndim != 2:
            raise ValueError(f"Expected 2D activations or [N, 1, D] tensors, got shape {tuple(acts.shape)}")
        return acts

    acts_A = _squeeze(acts_A)
    acts_B = _squeeze(acts_B)

    if acts_A.shape[0] != acts_B.shape[0]:
        n_pairs = min(acts_A.shape[0], acts_B.shape[0])
        LOGGER.info(
            "Unequal activation counts detected in build_subspace: %d vs %d; truncating to %d pairs.",
            acts_A.shape[0],
            acts_B.shape[0],
            n_pairs,
        )
        acts_A = acts_A[:n_pairs]
        acts_B = acts_B[:n_pairs]

    stack_AB = torch.cat([acts_A, acts_B], dim=0)
    mean_P = stack_AB.mean(0)
    stack_centered = stack_AB - mean_P

    _, S, Vt = torch.linalg.svd(stack_centered, full_matrices=False)
    k = min(k, S.shape[0])
    P = Vt[:k].T.contiguous()

    h_A_sub = (acts_A - mean_P) @ P
    h_B_sub = (acts_B - mean_P) @ P

    h_sub_stack = torch.cat([h_A_sub, h_B_sub], dim=0)
    sub_mean = h_sub_stack.mean(0)
    sub_std = h_sub_stack.std(0, unbiased=False).clamp_min(1e-8)
    weight_mode_lower = str(weight_mode).strip().lower()
    if weight_mode_lower == "var":
        raw_weights = h_sub_stack.std(0, unbiased=False)
    elif weight_mode_lower == "none":
        raw_weights = torch.ones(k, dtype=h_sub_stack.dtype, device=h_sub_stack.device)
    else:
        raw_weights = S[:k]
        
    # Normalize weights so the average weight is 1.0. This makes it invariant
    # to dataset size (S grows with sqrt(N)) and scale.
    normalized_weights = raw_weights / raw_weights.mean().clamp_min(1e-8)
    
    # Clamp to avoid extreme dominance by PC1
    weights = normalized_weights.clamp_min(0.1).clamp_max(float(max_weight))
    
    # Log stats BEFORE clamping so we can see the true distribution
    weight_min = float(normalized_weights.min().item())
    weight_max = float(normalized_weights.max().item())
    weight_mean = float(normalized_weights.mean().item())

    denom = S.pow(2).sum().clamp_min(1e-8)
    info_preserved_pct = 100.0 * S[:k].pow(2).sum().item() / denom.item()

    LOGGER.info(
        "Subspace built: k=%d, top-S=%.4f, bottom-S=%.4f, explained_var=%.2f%%",
        k,
        S[0].item(),
        S[k - 1].item(),
        info_preserved_pct,
    )
    
    # Pack the pre-clamp weight stats so train_subspace_glp can log them
    setattr(weights, "weight_min", weight_min)
    setattr(weights, "weight_max", weight_max)
    setattr(weights, "weight_mean", weight_mean)

    return P, mean_P, h_A_sub, h_B_sub, sub_mean, sub_std, weights, info_preserved_pct


def collect_concept_pair_activations(
    model_name: str,
    dataset_name: str,
    layer: int,
    hook_point: str = "post",
    pooling: str = "last",
    device: str = "cuda:0",
    model_dtype: str = "bfloat16",
    n_samples: Optional[int] = None,
    prompt_batch_size: int = 16,
    act_cache_dir: str = "data/subspace_raw_activations",
    reextract_acts: bool = False,
    augmentation: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract paired dense residual activations via TransformerLens."""
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = Path(act_cache_dir)
    if not cache_root.is_absolute():
        cache_root = repo_root / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_payload = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "layer": int(layer),
        "hook_point": str(hook_point),
        "pooling": str(pooling),
        "model_dtype": str(model_dtype),
        "n_samples": None if n_samples is None else int(n_samples),
        "augmentation": int(augmentation),
    }
    cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_stub = f"{dataset_name}_layer{layer}_{hook_point}_{pooling}_{cache_key}".replace("/", "_")
    cache_path = cache_root / f"{cache_stub}.pt"

    if cache_path.exists() and not reextract_acts:
        LOGGER.info("Loading cached raw activations from %s", cache_path)
        cached = torch.load(cache_path, map_location="cpu")
        h_A = cached.get("h_A")
        h_B = cached.get("h_B")
        if isinstance(h_A, torch.Tensor) and isinstance(h_B, torch.Tensor):
            LOGGER.info(
                "Loaded cached activations: A=%s, B=%s",
                tuple(h_A.shape),
                tuple(h_B.shape),
            )
            return h_A.float(), h_B.float()
        LOGGER.warning("Cache file %s is invalid. Re-extracting activations.", cache_path)

    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    from ..data.loader import DataLoader as SteDataLoader
    from ..utils import collect_dense_activations

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(model_dtype, torch.bfloat16)

    LOGGER.info("Loading model %s for activation extraction...", model_name)
    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        dtype=torch_dtype,
    )
    tl_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    loader = SteDataLoader()
    data = loader.load(
        dataset_name,
        n_samples=n_samples,
        apply_chat_template=True,
        tokenizer=tokenizer,
        augmentation=augmentation,
    )
    cfg = loader.get_config(dataset_name)

    target_texts = [d[cfg.target_key] for d in data if d.get(cfg.target_key)]
    contrast_texts = [d[cfg.contrast_key] for d in data if d.get(cfg.contrast_key)]
    n = min(len(target_texts), len(contrast_texts))
    target_texts, contrast_texts = target_texts[:n], contrast_texts[:n]

    def _preview_text(text: str, limit: int = 500) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[:limit].rstrip() + "..."

    LOGGER.info("Collecting activations for %d pairs (layer=%d, hook=%s)...", n, layer, hook_point)
    if n > 0:
        LOGGER.info(
            "Example extracted prompt pair:\nA: %s\nB: %s",
            _preview_text(target_texts[0]),
            _preview_text(contrast_texts[0]),
        )

    def _collect(texts: List[str]) -> torch.Tensor:
        result = collect_dense_activations(
            tl_model,
            texts,
            layers=[layer],
            hook_point=hook_point,
            batch_size=prompt_batch_size,
            pooling=pooling,
            device=device,
            tokenizer=tokenizer,
        )
        return result[layer].cpu().float()

    # For Flow Matching, we typically want to flow FROM the negative/undesired concept (Contrast) 
    # TO the positive/desired concept (Target). So A=Contrast, B=Target.
    h_B = _collect(target_texts)
    h_A = _collect(contrast_texts)

    payload = {
        "meta": cache_payload,
        "h_A": h_A.cpu().float(),
        "h_B": h_B.cpu().float(),
    }
    torch.save(payload, cache_path)
    LOGGER.info("Saved raw activation cache to %s", cache_path)

    del tl_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return h_A, h_B


def _make_glp(k: int, cfg: SubspaceGLPConfig) -> GLP:
    d_model = cfg.d_model_mult * k
    d_mlp = cfg.d_mlp_mult * k
    return GLP(
        normalizer_config={
            "rep_statistic": "",
            "d_input": k,
            "normalization_method": cfg.normalization_method,
        },
        denoiser_config={
            "d_input": k,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "n_layers": cfg.denoiser_n_layers,
        },
        noise_sampling_method=cfg.noise_sampling_method,
        u_sampling_method=cfg.u_sampling_method,
        ot_chunk_size=cfg.ot_chunk_size,
    )


def _setup_seed(seed: Optional[int]) -> int:
    """Seed Python, NumPy, and Torch for deterministic epoch replay."""
    effective_seed = int(seed if seed is not None else 0)
    torch.manual_seed(effective_seed)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed


def _cosine_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    if step < warmup_steps:
        alpha = step / max(warmup_steps, 1)
        return initial_factor + (1.0 - initial_factor) * alpha
    alpha = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine_out = 0.5 * (1 + math.cos(math.pi * alpha))
    return final_factor + (initial_factor - final_factor) * cosine_out


def _linear_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    if step < warmup_steps:
        alpha = step / max(warmup_steps, 1)
        return initial_factor + (1.0 - initial_factor) * alpha
    alpha = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return 1.0 + (final_factor - 1.0) * alpha


def _save_subspace_glp_checkpoint(
    save_dir: Path,
    cfg: SubspaceGLPConfig,
    glp: GLP,
    P: torch.Tensor,
    mean_P: torch.Tensor,
    sub_mean: torch.Tensor,
    sub_std: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    checkpoint_name: str = "final",
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    glp.save_pretrained(save_dir, name=checkpoint_name)

    # Make each checkpoint self-contained for standalone loading (e.g. Hub uploads).
    _save_subspace_static_artifacts(
        save_dir,
        cfg,
        glp,
        P=P,
        mean_P=mean_P,
        sub_mean=sub_mean,
        sub_std=sub_std,
        weights=weights,
        k=int(k),
    )

    LOGGER.info("Checkpoint %s saved to %s", checkpoint_name, save_dir)


def _format_token_count(n: int) -> str:
    """Format an integer count into human-friendly suffixes (k, M, B).

    Examples: 1000 -> '1k', 10000000 -> '10M'
    """
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    if n >= 1_000_000:
        return f"{int(round(n / 1_000_000))}M"
    if n >= 1_000:
        return f"{int(round(n / 1_000))}k"
    return str(n)


def _save_subspace_static_artifacts(
    save_dir: Path,
    cfg: SubspaceGLPConfig,
    glp: GLP,
    P: torch.Tensor,
    mean_P: torch.Tensor,
    sub_mean: torch.Tensor,
    sub_std: torch.Tensor,
    weights: torch.Tensor,
    k: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(P, save_dir / "subspace.pt")
    torch.save(mean_P, save_dir / "mean_P.pt")
    torch.save(weights, save_dir / "weights.pt")
    torch.save(
        {
            "mean": sub_mean.cpu(),
            "var": (sub_std ** 2).cpu(),
            "normalization_method": cfg.normalization_method,
        },
        save_dir / "rep_statistics.pt",
    )

    config_dict = {
        "subspace_glp_config": asdict(cfg),
        "subspace_k": k,
        "layer": cfg.layer,
        "hook_point": cfg.hook_point,
        "model_name": cfg.model_name,
        "glp_kwargs": {
            "normalizer_config": {
                "rep_statistic": "rep_statistics.pt",
                "d_input": k,
                "normalization_method": cfg.normalization_method,
            },
            "denoiser_config": {
                "d_input": k,
                "d_model": cfg.d_model_mult * k,
                "d_mlp": cfg.d_mlp_mult * k,
                "n_layers": cfg.denoiser_n_layers,
            },
            "noise_sampling_method": cfg.noise_sampling_method,
            "u_sampling_method": cfg.u_sampling_method,
            "ot_chunk_size": cfg.ot_chunk_size,
        },
        "subspace_artifacts": {
            "P": "subspace.pt",
            "mean_P": "mean_P.pt",
            "weights": "weights.pt",
            "rep_statistics": "rep_statistics.pt",
        },
    }
    with open(save_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)


def _sample_epoch_batch_indices(n_items: int, batch_size: int, shuffle: bool, device: torch.device | str) -> torch.Tensor:
    if n_items <= 0:
        raise ValueError("Cannot sample from an empty dataset.")
    if not shuffle:
        return torch.arange(batch_size, device=device) % n_items
    return torch.randint(0, n_items, (batch_size,), device=device)


def _compute_batch_norm_stats(h_batch: torch.Tensor, normalization_method: str) -> Tuple[torch.Tensor, torch.Tensor]:
    method = _canonicalize_normalization_method(normalization_method)
    h = h_batch.float()
    if method == "gaussian":
        return h.mean(0), h.var(0, unbiased=False).clamp_min(1e-8)
    if method == "rmsnorm":
        return torch.zeros(h.shape[1], device=h.device), h.pow(2).mean(0).clamp_min(1e-8)
    return h.mean(0), h.var(0, unbiased=False).clamp_min(1e-8)


def train_subspace_glp(cfg: SubspaceGLPConfig) -> dict:
    """Full training pipeline (Phase 0 -> Phase 1)."""
    save_dir = Path(cfg.save_root) / cfg.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    base_seed = _setup_seed(cfg.seed)

    h_A, h_B = collect_concept_pair_activations(
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
        layer=cfg.layer,
        hook_point=cfg.hook_point,
        pooling=cfg.pooling,
        device=cfg.device,
        model_dtype=cfg.model_dtype,
        n_samples=cfg.n_samples,
        prompt_batch_size=cfg.prompt_batch_size,
        act_cache_dir=cfg.act_cache_dir,
        reextract_acts=cfg.reextract_acts,
        augmentation=cfg.aug,
    )

    P, mean_P, h_A_sub, h_B_sub, sub_mean, sub_std, weights, subspace_info_preserved_pct = build_subspace(
        h_A,
        h_B,
        cfg.subspace_k,
        weight_mode=cfg.weight,
        max_weight=cfg.max_weight,
    )
    k = P.shape[1]

    train_device = cfg.device
    if cfg.init_ckpt:
        LOGGER.info("Loading initial checkpoint from %s", cfg.init_ckpt)
        glp = load_glp(cfg.init_ckpt, device=train_device, checkpoint="final")
        loaded_k = int(glp.denoiser.model.d_input)
        if loaded_k != int(k):
            raise ValueError(
                f"Checkpoint d_input mismatch: loaded {loaded_k}, expected subspace_k {k}. "
                "Use a checkpoint trained with the same subspace dimension."
            )
    else:
        glp = _make_glp(k, cfg).to(train_device)

    normalizer_mean = sub_mean.to(train_device)
    normalizer_var = sub_std.pow(2).to(train_device)
    glp.normalizer.mean = normalizer_mean
    glp.normalizer.var = normalizer_var

    norm_method = _canonicalize_normalization_method(cfg.normalization_method)
    sub_mean = sub_mean.to(train_device)
    sub_std = sub_std.to(train_device)
    weights = weights.to(train_device)

    if norm_method == "gaussian":
        h_A_norm = (h_A_sub.to(train_device) - sub_mean) / sub_std
        h_B_norm = (h_B_sub.to(train_device) - sub_mean) / sub_std
    elif norm_method == "rmsnorm":
        h_A_norm = h_A_sub.to(train_device) / sub_std
        h_B_norm = h_B_sub.to(train_device) / sub_std
    else:
        h_A_norm = h_A_sub.to(train_device)
        h_B_norm = h_B_sub.to(train_device)

    optimizer = torch.optim.AdamW(glp.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_batch_steps = max(1, cfg.total_steps) * max(1, cfg.num_epochs)
    warmup_steps = max(1, int(cfg.warmup_ratio * total_batch_steps))

    if cfg.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                _cosine_scheduler_with_warmup,
                warmup_steps=warmup_steps,
                max_steps=total_batch_steps,
                initial_factor=cfg.initial_factor,
                final_factor=cfg.final_factor,
            ),
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(
                _linear_scheduler_with_warmup,
                warmup_steps=warmup_steps,
                max_steps=total_batch_steps,
                initial_factor=cfg.initial_factor,
                final_factor=cfg.final_factor,
            ),
        )

    noise_method = _canonicalize_noise_sampling_method(cfg.noise_sampling_method)
    u_method = _canonicalize_u_sampling_method(cfg.u_sampling_method)
    use_autocast = cfg.use_bf16 and cfg.device.startswith("cuda")

    wandb_run = None
    if cfg.wandb:
        import wandb

        wandb_name = cfg.run_name if str(cfg.run_name).startswith("subspace_") else f"subspace_{cfg.run_name}"
        wandb_run = wandb.init(project=cfg.wandb_project, name=wandb_name, config=asdict(cfg))
        wandb_run.log(
            {
                "subspace/k": float(k),
                "subspace/info_preserved_pct": float(subspace_info_preserved_pct),
                "subspace/min_weight": getattr(weights, "weight_min", float(weights.min().item())),
                "subspace/max_weight": getattr(weights, "weight_max", float(weights.max().item())),
                "subspace/mean_weight": getattr(weights, "weight_mean", float(weights.float().mean().item())),
            },
            step=0,
        )

    glp.train()
    glp.scheduler.set_timesteps(glp.scheduler.config.num_train_timesteps)

    global_step = 0
    optimizer_step = 0
    pending_accum_steps = 0
    processed_tokens = 0
    next_checkpoint_token = max(1, int(cfg.checkpoint_token_step)) if cfg.checkpoint_token_step > 0 else None
    progress_bar = tqdm(total=total_batch_steps, desc="Subspace GLP Training")

    try:
        for epoch in range(max(1, cfg.num_epochs)):
            _setup_seed(base_seed)
            epoch_loss = 0.0
            epoch_cos_sim = 0.0
            epoch_batches = 0

            for _ in range(max(1, cfg.total_steps)):
                indices = _sample_epoch_batch_indices(h_A_norm.shape[0], cfg.batch_size, cfg.shuffle, train_device)
                h_A_batch = h_A_norm.index_select(0, indices).to(train_device)
                h_B_batch = h_B_norm.index_select(0, indices).to(train_device)

                if noise_method in {"sot", "sinkhorn"}:
                    h_A_batch = match_noise_to_latents_ot(
                        h_B_batch.unsqueeze(1),
                        h_A_batch.unsqueeze(1),
                        method=noise_method,
                        chunk_size=cfg.ot_chunk_size,
                    ).squeeze(1)

                u = _sample_training_u(h_B_batch.unsqueeze(1), u_sampling_method=u_method)
                t = u.to(train_device)
                z_t = (1 - t[:, None]) * h_A_batch + t[:, None] * h_B_batch
                v_target_norm = h_B_batch - h_A_batch

                num_ts = glp.scheduler.config.num_train_timesteps
                timesteps = (u * num_ts).long().clamp(0, num_ts - 1).to(train_device)

                with torch.autocast(
                    device_type="cuda" if use_autocast else "cpu",
                    dtype=torch.bfloat16,
                    enabled=use_autocast,
                ):
                    v_pred_norm = glp.denoiser.model(z_t, timesteps)
                    diff = v_pred_norm.float() - v_target_norm.float()
                    sq_err = diff.pow(2)
                    if str(cfg.loss).strip().lower() == "huber":
                        elem_loss = F.huber_loss(
                            v_pred_norm.float(),
                            v_target_norm.float(),
                            delta=1.0,
                            reduction="none",
                        )
                    else:
                        elem_loss = sq_err
                    loss = (weights.unsqueeze(0) * elem_loss).mean()

                loss = loss / cfg.grad_accum
                loss.backward()
                pending_accum_steps += 1
                epoch_loss += loss.item()
                epoch_batches += 1
                processed_tokens += int(h_B_batch.shape[0])
                cos_sim = F.cosine_similarity(v_pred_norm.float(), v_target_norm.float(), dim=-1).mean().item()
                epoch_cos_sim += cos_sim
                global_step += 1
                progress_bar.update(1)

                loss_unreduced = sq_err.mean(dim=1)
                mask_early = u < 0.3
                mask_mid = (u >= 0.3) & (u <= 0.7)
                mask_late = u > 0.7
                loss_early = loss_unreduced[mask_early].mean().item() if mask_early.any() else 0.0
                loss_mid = loss_unreduced[mask_mid].mean().item() if mask_mid.any() else 0.0
                loss_late = loss_unreduced[mask_late].mean().item() if mask_late.any() else 0.0

                if pending_accum_steps >= cfg.grad_accum:
                    if cfg.gradient_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(glp.parameters(), cfg.gradient_clip)
                    else:
                        grad_norm = 0.0

                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    optimizer_step += 1
                    pending_accum_steps = 0

                    if global_step % cfg.log_every_n_steps == 0:
                        avg_loss = epoch_loss / max(epoch_batches, 1)
                        avg_cos = epoch_cos_sim / max(epoch_batches, 1)
                        batch_mean = h_B_batch.float().mean()
                        batch_var_max = _compute_batch_norm_stats(h_B_batch, cfg.normalization_method)[1].max()
                        global_mean = glp.normalizer.mean.float().mean()
                        global_var_max = glp.normalizer.var.float().max()
                        progress_bar.set_description(
                            f"Epoch {epoch + 1}/{cfg.num_epochs} | Step {global_step}/{total_batch_steps} | "
                            f"Loss: {avg_loss:.5f} | Raw: {float(sq_err.mean().item()):.5f} | Cos: {avg_cos:.4f}"
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "subspace/loss": avg_loss,
                                    "subspace/loss_early": float(loss_early),
                                    "subspace/loss_mid": float(loss_mid),
                                    "subspace/loss_late": float(loss_late),
                                    "subspace/loss_raw": float(sq_err.mean().item()),
                                    "subspace/cos_sim": avg_cos,
                                    "subspace/batch_mean": float(batch_mean.item()),
                                    "subspace/batch_var_max": float(batch_var_max.item()),
                                    "subspace/global_mean": float(global_mean.item()),
                                    "subspace/global_var_max": float(global_var_max.item()),
                                    "subspace/grad_norm": float(grad_norm),
                                    "subspace/lr_adamw": scheduler.get_last_lr()[0],
                                    "subspace/epoch": epoch,
                                    "subspace/k": float(k),
                                    "subspace/info_preserved_pct": float(subspace_info_preserved_pct),
                                    "subspace/min_weight": getattr(weights, "weight_min", float(weights.min().item())),
                                    "subspace/max_weight": getattr(weights, "weight_max", float(weights.max().item())),
                                    "subspace/mean_weight": getattr(weights, "weight_mean", float(weights.float().mean().item())),
                                },
                                step=global_step,
                            )

                if next_checkpoint_token is not None and processed_tokens >= next_checkpoint_token:
                    # Format a human-readable label (e.g. 10M) and save into a per-checkpoint subfolder
                    label = _format_token_count(processed_tokens)
                    ckpt_dir = save_dir / "checkpoints" / label
                    _save_subspace_glp_checkpoint(
                        ckpt_dir,
                        cfg,
                        glp,
                        P=P,
                        mean_P=mean_P,
                        sub_mean=sub_mean,
                        sub_std=sub_std,
                        weights=weights,
                        k=k,
                        checkpoint_name="final",
                    )
                    next_checkpoint_token += max(1, int(cfg.checkpoint_token_step))

            if pending_accum_steps > 0:
                if cfg.gradient_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(glp.parameters(), cfg.gradient_clip)
                else:
                    grad_norm = 0.0

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                optimizer_step += 1
                pending_accum_steps = 0

    finally:
        progress_bar.close()
        if wandb_run is not None:
            wandb_run.finish()

    # Final model checkpoint: place in checkpoints/final to avoid polluting save root
    final_ckpt_dir = save_dir / "checkpoints" / "final"
    _save_subspace_glp_checkpoint(
        final_ckpt_dir,
        cfg,
        glp,
        P=P,
        mean_P=mean_P,
        sub_mean=sub_mean,
        sub_std=sub_std,
        weights=weights,
        k=k,
        checkpoint_name="final",
    )

    summary = {
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "total_steps_per_epoch": cfg.total_steps,
        "num_epochs": cfg.num_epochs,
        "total_batch_steps": total_batch_steps,
        "processed_tokens": processed_tokens,
        "save_dir": str(save_dir),
        "final_checkpoint_dir": str(final_ckpt_dir),
        "k": k,
        "subspace_info_preserved_pct": float(subspace_info_preserved_pct),
    }
    LOGGER.info(
        "Training complete. Final checkpoint saved to %s. Batch steps: %d/%d",
        final_ckpt_dir,
        global_step,
        total_batch_steps,
    )
    return summary


class SubspaceGLP:
    """Wrap a trained GLP with centered PCA subspace project/restore logic."""

    def __init__(
        self,
        glp: GLP,
        P: torch.Tensor,
        mean_P: torch.Tensor,
        sub_mean: torch.Tensor,
        sub_std: torch.Tensor,
        weights: torch.Tensor,
        device: str = "cuda:0",
    ):
        self.glp = glp.eval()
        self.P = P.to(device)
        self.mean_P = mean_P.to(device)
        self.sub_mean = sub_mean.to(device)
        self.sub_std = sub_std.to(device)
        self.weights = weights.to(device)
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint_dir: str,
        device: str = "cuda:0",
        checkpoint: str = "final",
        local_files_only: Optional[bool] = None,
    ) -> "SubspaceGLP":
        ckpt = Path(checkpoint_dir)
        glp = load_glp(
            str(ckpt),
            device=device,
            checkpoint=checkpoint,
            local_files_only=local_files_only,
        )

        if not ckpt.exists():
            from huggingface_hub import snapshot_download

            allow_patterns = [
                "subspace.pt",
                "mean_P.pt",
                "weights.pt",
                "config.yaml",
                "rep_statistics.pt",
                "final.safetensors",
            ]
            if checkpoint != "final":
                allow_patterns.extend(
                    [
                        f"{checkpoint}.safetensors",
                        f"{checkpoint}/subspace.pt",
                        f"{checkpoint}/mean_P.pt",
                        f"{checkpoint}/weights.pt",
                        f"{checkpoint}/config.yaml",
                        f"{checkpoint}/rep_statistics.pt",
                        f"{checkpoint}/final.safetensors",
                    ]
                )

            download_kwargs = {"repo_id": checkpoint_dir, "allow_patterns": allow_patterns}
            if local_files_only is True:
                local_dir = snapshot_download(local_files_only=True, **download_kwargs)
            elif local_files_only is False:
                local_dir = snapshot_download(local_files_only=False, **download_kwargs)
            else:
                local_dir = snapshot_download(**download_kwargs)
            ckpt = Path(local_dir)

        def _find_file(candidates: List[Path]) -> Path:
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"Missing required subspace artifact: {[str(p) for p in candidates]}")

        # Resolve checkpoint subfolder if specified
        resolved_ckpt = ckpt / checkpoint if (ckpt / checkpoint).is_dir() else ckpt

        p_candidates = [resolved_ckpt / "subspace.pt", ckpt / "subspace.pt"]
        mean_candidates = [resolved_ckpt / "mean_P.pt", ckpt / "mean_P.pt"]
        rep_candidates = [resolved_ckpt / "rep_statistics.pt", ckpt / "rep_statistics.pt"]
        weight_candidates = [resolved_ckpt / "weights.pt", ckpt / "weights.pt"]

        P = torch.load(_find_file(p_candidates), map_location=device)
        mean_P = torch.load(_find_file(mean_candidates), map_location=device)

        # Load weights (required)
        weights = torch.load(_find_file(weight_candidates), map_location=device)

        # Load normalization stats from rep_statistics.pt (required)
        rep = torch.load(_find_file(rep_candidates), map_location=device)
        rep_mean = rep.get("mean")
        rep_var = rep.get("var")
        if rep_mean is None or rep_var is None:
            raise FileNotFoundError(f"rep_statistics.pt missing mean/var in {ckpt}")
        sub_mean = rep_mean
        sub_std = rep_var.sqrt()

        return cls(glp, P, mean_P, sub_mean, sub_std, weights, device=device)

    @torch.no_grad()
    def project(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if h.ndim == 1:
            h = h.unsqueeze(0)
        h = h.to(self.device).float()
        h_sub = (h - self.mean_P) @ self.P
        h_norm = (h_sub - self.sub_mean) / (self.sub_std + 1e-8)
        h_null = h - (h - self.mean_P) @ self.P @ self.P.T - self.mean_P
        return h_norm, h_null

    @torch.no_grad()
    def unproject(self, h_norm: torch.Tensor, h_null: torch.Tensor) -> torch.Tensor:
        h_sub = h_norm * self.sub_std + self.sub_mean
        return h_sub @ self.P.T + self.mean_P + h_null

    @torch.no_grad()
    def steer(self, h_A: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        squeeze = h_A.ndim == 1
        if squeeze:
            h_A = h_A.unsqueeze(0)

        h_norm, h_null = self.project(h_A)
        z = h_norm  
        dt = 1.0 / n_steps
        self.glp.denoiser.eval()
        num_ts = self.glp.scheduler.config.num_train_timesteps

        for i in range(n_steps):
            t_cur = torch.full((z.shape[0],), i * dt, device=self.device)
            t_nxt = torch.full((z.shape[0],), (i + 1) * dt, device=self.device)
            ts_cur = (t_cur * num_ts).long().clamp(0, num_ts - 1)
            ts_nxt = (t_nxt * num_ts).long().clamp(0, num_ts - 1)

            v1 = self.glp.denoiser.model(z, ts_cur)
            z_euler = z + dt * v1
            v2 = self.glp.denoiser.model(z_euler, ts_nxt)
            z = z + dt * 0.5 * (v1 + v2)

        h_B = self.unproject(z, h_null)
        return h_B.squeeze(0) if squeeze else h_B
