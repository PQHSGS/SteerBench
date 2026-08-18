"""Train concept classifier for GLP classifier guidance using Steering contrastive datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from tqdm import tqdm
import yaml

from transformer_lens import HookedTransformer

from ..data import DataLoader as SteeringDataLoader
from ..utils import get_hook_name, get_resid_acts
from .classifier import ConceptClassifier


@dataclass
class ClassifierTrainConfig:
    # Model + hook extraction
    model_name: str = "google/gemma-2-2b-it"
    layer: int = 14
    hook_point: str = "pre"
    position: Union[str, int] = "last"
    sequence_pooling: str = "last"  # one of: last, mean
    device: str = "cuda"
    model_dtype: str = "bfloat16"

    # Data (reuses Steering dataset registry)
    dataset_name: str = "sycophancy"
    n_train: Optional[int] = None
    prompt_batch_size: int = 16

    # Classifier architecture
    d_model: int = 256
    d_mlp: int = 512
    n_layers: int = 4
    t_embed_dim: int = 128

    # Training
    seed: int = 42
    num_epochs: int = 5
    train_batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    val_ratio: float = 0.1

    # Diffusion-style noising during classifier training
    noise_rate: float = 0.5
    timestep_min: float = 0.0
    timestep_max: float = 1.0

    # Output
    save_root: str = "./GLP"
    run_name: str = "classifier-stream"

    # wandb
    wandb_enabled: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: str = "glp"
    wandb_run_name: Optional[str] = None
    wandb_log_every_steps: int = 10



def _init_wandb(cfg: ClassifierTrainConfig):
    if not cfg.wandb_enabled:
        return None

    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "wandb tracking is enabled but wandb is not installed. Install it with: pip install wandb"
        ) from exc

    return wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=cfg.wandb_run_name or cfg.run_name,
        config=asdict(cfg),
    )



def _parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported model_dtype: {name}")



def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _build_text_label_pairs(cfg: ClassifierTrainConfig, model: HookedTransformer) -> Tuple[List[str], List[int], str]:
    loader = SteeringDataLoader()
    raw = loader.load(
        dataset_name=cfg.dataset_name,
        n_samples=cfg.n_train,
        format=True,
        apply_chat_template=True,
        tokenizer=model.tokenizer,
    )
    ds_cfg = loader.get_config(cfg.dataset_name)

    target_key = ds_cfg.target_key
    contrast_key = ds_cfg.contrast_key
    if contrast_key is None:
        raise ValueError(
            f"Dataset '{cfg.dataset_name}' does not expose contrast_key and cannot be used for classifier training"
        )

    texts: List[str] = []
    labels: List[int] = []

    for row in raw:
        target = row.get(target_key)
        contrast = row.get(contrast_key)

        if target:
            texts.append(str(target))
            labels.append(1)
        if contrast:
            texts.append(str(contrast))
            labels.append(0)

    if not texts:
        raise RuntimeError(f"No usable target/contrast text pairs found for dataset '{cfg.dataset_name}'")

    dataset_path = str(ds_cfg.path) if hasattr(ds_cfg, "path") else cfg.dataset_name
    return texts, labels, dataset_path



def _pool_acts(acts: torch.Tensor, cfg: ClassifierTrainConfig) -> torch.Tensor:
    selected = get_resid_acts(acts, cfg.position)
    if selected.ndim == 2:
        return selected

    if selected.ndim != 3:
        raise ValueError(f"Unexpected activation shape after position selection: {tuple(selected.shape)}")

    if cfg.sequence_pooling == "last":
        return selected[:, -1, :]
    if cfg.sequence_pooling == "mean":
        return selected.mean(dim=1)

    raise ValueError("sequence_pooling must be one of: last, mean")



def _collect_contrastive_activations(
    cfg: ClassifierTrainConfig,
    model: HookedTransformer,
    texts: Sequence[str],
    labels: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    hook_name = get_hook_name(cfg.layer, cfg.hook_point)

    all_acts: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for i in tqdm(range(0, len(texts), cfg.prompt_batch_size), desc="Collect classifier activations"):
        batch_texts = list(texts[i : i + cfg.prompt_batch_size])
        batch_labels = labels[i : i + cfg.prompt_batch_size]

        toks = model.to_tokens(batch_texts)
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=[hook_name], return_type=None)
            acts = cache[hook_name]
            pooled = _pool_acts(acts, cfg).detach().cpu()

        del cache, acts

        all_acts.append(pooled)
        all_labels.append(torch.tensor(batch_labels, dtype=torch.long))

    x = torch.cat(all_acts, dim=0)
    y = torch.cat(all_labels, dim=0)
    return x, y



def _noisy_batch(x0: torch.Tensor, cfg: ClassifierTrainConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = x0.shape[0]
    t = torch.empty(bsz, device=x0.device).uniform_(cfg.timestep_min, cfg.timestep_max)
    t = t.clamp(min=0.0, max=1.0)

    noise = torch.randn_like(x0)
    z_t = (1.0 - t[:, None]) * x0 + t[:, None] * noise
    return z_t, t



def _evaluate(model: ConceptClassifier, x_val: torch.Tensor, y_val: torch.Tensor, cfg: ClassifierTrainConfig) -> dict:
    model.eval()
    with torch.no_grad():
        z_t, t = _noisy_batch(x_val, cfg)
        logits = model(z_t, t)
        loss = F.binary_cross_entropy_with_logits(logits, y_val.float())
        pred = (torch.sigmoid(logits) >= 0.5).long()
        acc = (pred == y_val).float().mean()
    return {"val/loss": float(loss.item()), "val/acc": float(acc.item())}



def train_classifier(cfg: ClassifierTrainConfig) -> dict:
    _set_seed(cfg.seed)

    wandb_run = _init_wandb(cfg)

    model = HookedTransformer.from_pretrained(
        cfg.model_name,
        device=cfg.device,
        dtype=_parse_dtype(cfg.model_dtype),
    )

    texts, labels, dataset_path = _build_text_label_pairs(cfg, model)
    x, y = _collect_contrastive_activations(cfg, model, texts, labels)

    # deterministic split
    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]

    val_count = int(max(1, round(cfg.val_ratio * x.shape[0])))
    if val_count >= x.shape[0]:
        val_count = max(1, x.shape[0] // 10)

    x_val, y_val = x[:val_count].to(cfg.device), y[:val_count].to(cfg.device)
    x_train, y_train = x[val_count:], y[val_count:]

    if x_train.shape[0] < 2:
        raise RuntimeError("Not enough training samples after split; reduce val_ratio or increase n_train")

    train_loader = TorchDataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.train_batch_size,
        shuffle=True,
        drop_last=False,
    )

    clf = ConceptClassifier(
        d_input=int(x.shape[-1]),
        d_model=cfg.d_model,
        d_mlp=cfg.d_mlp,
        n_layers=cfg.n_layers,
        t_embed_dim=cfg.t_embed_dim,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(clf.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.save_root) / cfg.run_name
    final_dir = out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    progress = tqdm(total=cfg.num_epochs * len(train_loader), desc="Train classifier")

    best_val_loss = float("inf")
    best_metrics = {}

    try:
        for epoch in range(cfg.num_epochs):
            clf.train()
            for (x0, yb) in train_loader:
                x0 = x0.to(cfg.device)
                yb = yb.to(cfg.device).float()

                z_t, t = _noisy_batch(x0, cfg)
                logits = clf(z_t, t)
                loss = F.binary_cross_entropy_with_logits(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf.parameters(), cfg.grad_clip)
                optimizer.step()

                global_step += 1
                progress.update(1)
                progress.set_postfix({"loss": float(loss.item())})

                if wandb_run and global_step % max(1, cfg.wandb_log_every_steps) == 0:
                    with torch.no_grad():
                        pred = (torch.sigmoid(logits) >= 0.5).float()
                        train_acc = (pred == yb).float().mean().item()
                    wandb_run.log(
                        {
                            "train/step": global_step,
                            "train/loss": float(loss.item()),
                            "train/acc": float(train_acc),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                        },
                        step=global_step,
                    )

            val_metrics = _evaluate(clf, x_val, y_val, cfg)
            val_loss = float(val_metrics["val/loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = dict(val_metrics)
                final_dir.mkdir(parents=True, exist_ok=True)
                clf.save_pretrained(final_dir, name="classifier")

            if wandb_run:
                wandb_run.log(
                    {
                        "val/epoch": epoch + 1,
                        **val_metrics,
                    },
                    step=global_step,
                )

    finally:
        progress.close()
        if wandb_run is not None:
            wandb_run.finish()

    if not final_dir.exists():
        final_dir.mkdir(parents=True, exist_ok=True)
        clf.save_pretrained(final_dir, name="classifier")

    classifier_kwargs = {
        "d_input": int(x.shape[-1]),
        "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp,
        "n_layers": cfg.n_layers,
        "t_embed_dim": cfg.t_embed_dim,
    }

    classifier_config_payload = {
        "classifier_kwargs": classifier_kwargs,
        "classifier_train_config": asdict(cfg),
    }
    with open(final_dir / "classifier_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(classifier_config_payload, handle, sort_keys=False)

    # If GLP config exists, append classifier config for one-folder bundles.
    glp_cfg_path = final_dir / "config.yaml"
    if glp_cfg_path.exists():
        with open(glp_cfg_path, "r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle) or {}
        existing["classifier_config"] = classifier_kwargs
        with open(glp_cfg_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(existing, handle, sort_keys=False)

    summary = {
        "output_dir": str(final_dir),
        "dataset_name": cfg.dataset_name,
        "dataset_path": dataset_path,
        "num_examples": int(x.shape[0]),
        "num_train": int(x_train.shape[0]),
        "num_val": int(x_val.shape[0]),
        "best_val_loss": float(best_val_loss),
        "best_metrics": best_metrics,
        "layer": cfg.layer,
        "hook_point": cfg.hook_point,
        "position": cfg.position,
    }

    with open(out_dir / "train_classifier_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
