"""Classifier module for diffusion classifier guidance in post-process GLP."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class SinusoidalPosEmb(nn.Module):
    """High-frequency sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            x = x[None]
        x = x.flatten()

        half_dim = self.dim // 2
        if half_dim <= 0:
            raise ValueError("SinusoidalPosEmb requires dim >= 2")

        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb_scale)
        emb = x[:, None].float() * emb[None, :]
        out = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.dim % 2 == 1:
            out = F.pad(out, (0, 1))

        return out


class ClassifierMLPBlock(nn.Module):
    """Residual MLP block with full FiLM conditioning from timestep features."""

    def __init__(self, d_model: int, d_mlp: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.up_proj = nn.Linear(d_model, d_mlp)
        self.gate_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)
        self.act = nn.SiLU()

        self.scale_proj = nn.Linear(cond_dim, d_model)
        self.shift_proj = nn.Linear(cond_dim, d_model)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        resid = x
        x = self.norm(x)

        scale = self.scale_proj(cond_emb)
        shift = self.shift_proj(cond_emb)
        x = (1.0 + scale) * x + shift

        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.act(gate) * up
        x = self.down_proj(x)

        return x + resid


class ConceptClassifier(nn.Module):
    """Binary classifier p_phi(y=1 | z_t, t) for classifier guidance."""

    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        d_mlp: int = 512,
        n_layers: int = 4,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        self.d_input = int(d_input)
        self.d_model = int(d_model)
        self.d_mlp = int(d_mlp)
        self.n_layers = int(n_layers)
        self.t_embed_dim = int(t_embed_dim)

        self.time_embed = SinusoidalPosEmb(self.t_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.t_embed_dim, self.t_embed_dim),
            nn.SiLU(),
            nn.Linear(self.t_embed_dim, self.t_embed_dim),
            nn.SiLU(),
        )

        self.in_proj = nn.Linear(self.d_input, self.d_model)
        self.layers = nn.ModuleList(
            [
                ClassifierMLPBlock(
                    d_model=self.d_model,
                    d_mlp=self.d_mlp,
                    cond_dim=self.t_embed_dim,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, 1)

    def _expand_timestep(self, t: torch.Tensor, bsz: int, seq: int = 1) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.flatten()

        if t.shape[0] == 1:
            return t.repeat(bsz * seq)
        if t.shape[0] == bsz:
            return t.repeat_interleave(seq) if seq > 1 else t
        if t.shape[0] == bsz * seq:
            return t

        raise ValueError(
            f"Classifier timestep shape mismatch: got {tuple(t.shape)}, expected "
            f"[1], [batch], or [batch*seq] with batch={bsz}, seq={seq}"
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return binary logits.

        Args:
            z_t: [B, D] or [B, S, D]
            t: [B], [B*S], or scalar

        Returns:
            logits: [B] for 2D input, [B, S] for 3D input
        """
        if z_t.ndim == 2:
            bsz, _ = z_t.shape
            seq = 1
            z_flat = z_t
        elif z_t.ndim == 3:
            bsz, seq, dim = z_t.shape
            z_flat = z_t.reshape(bsz * seq, dim)
        else:
            raise ValueError(f"Expected z_t with shape [B, D] or [B, S, D], got {tuple(z_t.shape)}")

        t_flat = self._expand_timestep(t.to(device=z_flat.device), bsz=bsz, seq=seq)
        t_freq = self.time_embed(t_flat)
        cond_emb = self.cond_mlp(t_freq.to(dtype=z_flat.dtype))

        x = self.in_proj(z_flat)
        for layer in self.layers:
            x = layer(x, cond_emb)

        x = self.out_norm(x)
        logits = self.out_proj(x).squeeze(-1)

        if z_t.ndim == 3:
            logits = logits.view(bsz, seq)

        return logits

    def log_prob(self, z_t: torch.Tensor, t: torch.Tensor, negative: bool = False) -> torch.Tensor:
        logits = self.forward(z_t, t)
        return F.logsigmoid(-logits) if negative else F.logsigmoid(logits)

    def save_pretrained(self, path: str | Path, name: str = "classifier") -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), out / f"{name}.safetensors")

    def load_pretrained(self, path: str | Path, name: str = "classifier") -> None:
        in_dir = Path(path)
        self.load_state_dict(load_file(in_dir / f"{name}.safetensors"))


DEFAULT_CLASSIFIER_KWARGS = {
    "d_model": 256,
    "d_mlp": 512,
    "n_layers": 4,
    "t_embed_dim": 128,
}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def _resolve_classifier_dir(weights_folder: str, checkpoint: str) -> Path:
    resolved = Path(weights_folder).expanduser()
    if not resolved.exists():
        resolved = Path(snapshot_download(repo_id=weights_folder))

    checkpoint_dir = resolved / checkpoint
    if checkpoint != "final" and checkpoint_dir.is_dir():
        return checkpoint_dir
    return resolved


def _resolve_classifier_kwargs(folder: Path) -> dict:
    classifier_cfg = _read_yaml(folder / "classifier_config.yaml")
    config_yaml = _read_yaml(folder / "config.yaml")

    kwargs = {}

    if "classifier_kwargs" in classifier_cfg:
        kwargs.update(classifier_cfg["classifier_kwargs"] or {})
    if "classifier_config" in config_yaml and isinstance(config_yaml["classifier_config"], dict):
        kwargs.update(config_yaml["classifier_config"])
    if "classifier_kwargs" in config_yaml and isinstance(config_yaml["classifier_kwargs"], dict):
        kwargs.update(config_yaml["classifier_kwargs"])

    rep_stats = folder / "rep_statistics.pt"
    if "d_input" not in kwargs and rep_stats.exists():
        payload = torch.load(rep_stats, map_location="cpu")
        mean = payload.get("mean")
        if mean is not None:
            kwargs["d_input"] = int(mean.shape[-1])

    for key, value in DEFAULT_CLASSIFIER_KWARGS.items():
        kwargs.setdefault(key, value)

    if "d_input" not in kwargs:
        raise ValueError(
            "Could not infer classifier d_input. Provide classifier_config.yaml with classifier_kwargs.d_input "
            "or keep rep_statistics.pt next to classifier weights."
        )

    return kwargs


def load_classifier(weights_folder: str, device: str = "cuda:0", checkpoint: str = "final") -> ConceptClassifier:
    """Load classifier from local folder or Hugging Face repo."""
    folder = _resolve_classifier_dir(weights_folder, checkpoint)
    kwargs = _resolve_classifier_kwargs(folder)

    model = ConceptClassifier(**kwargs)
    model.load_pretrained(folder, name="classifier")
    model.to(device)
    model.eval()
    return model
