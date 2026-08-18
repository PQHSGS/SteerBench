"""Small flow-matching utilities for nonlinear steering methods."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    repeat_kv,
    rotate_half,
)


class FlowMLP(nn.Module):
    """Lightweight velocity field v_theta(x_t, t)."""

    def __init__(self, dim: int, hidden_dim: int = 512, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = dim + 1
        for _ in range(max(int(n_layers), 1)):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x, t.to(dtype=x.dtype, device=x.device)], dim=-1))


def sinusoidal_time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_dim=128):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t):
        emb = sinusoidal_time_embedding(t, self.freq_dim)
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


def _apply_rope_single(x, cos, sin, unsqueeze_dim=1):
    """Apply RoPE to a single tensor (cheaper than apply_rotary_pos_emb(x, x, ...))."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class FlowCrossAttention(nn.Module):
    """Cross-attention with Gemma2 GQA config. RoPE applied to both Q and K sides."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.softcap = config.attn_logit_softcapping
        hidden_size = config.hidden_size

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        self.q_norm = Gemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma2RotaryEmbedding(config=config)

    def forward(self, hidden_states, encoder_hidden_states,
                encoder_attention_mask=None, q_pos_offset=0,
                q_position_ids=None, kv_position_ids=None):
        bsz, q_len, _ = hidden_states.size()
        kv_len = encoder_hidden_states.size(1)

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)
        k = k.view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(k)

        if q_position_ids is not None:
            q_pos = q_position_ids
        else:
            q_pos = torch.arange(q_len, device=q.device).unsqueeze(0) + q_pos_offset
        if kv_position_ids is not None:
            kv_pos = kv_position_ids
        else:
            kv_pos = torch.arange(kv_len, device=k.device).unsqueeze(0)
        q_cos, q_sin = self.rotary_emb(q, q_pos)
        k_cos, k_sin = self.rotary_emb(k, kv_pos)
        q = _apply_rope_single(q, q_cos, q_sin)
        k = _apply_rope_single(k, k_cos, k_sin)

        k_exp = repeat_kv(k, self.num_kv_groups)
        v_exp = repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k_exp.transpose(2, 3)) * self.scaling

        if self.softcap is not None:
            attn_weights = attn_weights / self.softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.softcap

        if encoder_attention_mask is not None:
            mask = encoder_attention_mask[:, None, None, :]
            attn_weights = attn_weights.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v_exp)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out)



class FlowBlock(nn.Module):
    """FLAS block: TimeEmbed -> CrossAttn -> causal SelfAttn -> MLP."""

    def __init__(
        self,
        config,
        init_gate=0.1,
        layer_idx=0,
        disable_cross_attn=False,
        disable_self_attn=False,
        disable_mlp=False,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        rms_norm_eps = config.rms_norm_eps
        self.layer_idx = layer_idx
        self.disable_cross_attn = disable_cross_attn
        self.disable_self_attn = disable_self_attn
        self.disable_mlp = disable_mlp

        self.pre_cross_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_attn = FlowCrossAttention(config)
        self.post_cross_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        sa_config = type(config).from_dict(config.to_dict())
        sa_config._attn_implementation = "eager"
        self.pre_sa_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_sa_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Gemma2Attention(sa_config, layer_idx=layer_idx % 2)
        self.self_attn_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

        self.pre_mlp_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.post_mlp_norm = Gemma2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.act_fn = nn.GELU(approximate="tanh")
        self.mlp_gate = nn.Parameter(torch.full((hidden_size,), init_gate))

    def forward(
        self,
        h,
        concept_hidden,
        concept_mask=None,
        t_emb=None,
        self_attn_cache=None,
        position_embeddings=None,
        padding_mask=None,
        use_cache=False,
        q_pos_offset=0,
        activation_position_ids=None,
    ):
        if t_emb is not None:
            h = h + t_emb[:, None, :]

        if not self.disable_cross_attn:
            h_normed = self.pre_cross_norm(h)
            ca_delta = self.cross_attn(
                h_normed,
                concept_hidden,
                concept_mask,
                q_pos_offset=q_pos_offset,
                q_position_ids=activation_position_ids,
            )
            ca_delta = self.post_cross_norm(ca_delta)
            h = h + self.cross_gate * ca_delta

        if self.disable_self_attn:
            new_cache = self_attn_cache if use_cache else None
        else:
            h_normed_sa = self.pre_sa_norm(h)
            q_len = h.size(1)
            past_len = self_attn_cache.get_seq_length() if self_attn_cache is not None else 0
            kv_len = past_len + q_len

            mask_val = -1e4
            row_idx = torch.arange(q_len, device=h.device).unsqueeze(1)
            col_idx = torch.arange(kv_len, device=h.device).unsqueeze(0)
            causal_mask = torch.where(
                col_idx <= row_idx + past_len,
                torch.zeros(1, device=h.device, dtype=h.dtype),
                torch.full((1,), mask_val, device=h.device, dtype=h.dtype),
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            if padding_mask is not None:
                pm = padding_mask[:, :kv_len]
                pad_4d = (1.0 - pm[:, None, None, :].to(h.dtype)) * mask_val
                causal_mask = causal_mask + pad_4d

            sa_out = self.self_attn(
                h_normed_sa,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=self_attn_cache,
            )
            sa_delta = self.post_sa_norm(sa_out[0])
            new_cache = self_attn_cache if use_cache else None
            h = h + self.self_attn_gate * sa_delta

        if not self.disable_mlp:
            x = self.pre_mlp_norm(h)
            x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            x = self.post_mlp_norm(x)
            h = h + self.mlp_gate * x

        return h, new_cache


class FlowFunction(nn.Module):
    """Velocity field v_theta(h, t, concept) = blocks(h, t, c) - h_input."""

    def __init__(
        self,
        config,
        num_blocks=3,
        time_conditioned=True,
        layer_idx=0,
        disable_cross_attn=False,
        disable_self_attn=False,
        disable_mlp=False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_blocks = num_blocks
        self.time_conditioned = time_conditioned

        if time_conditioned:
            self.time_embed = TimeEmbedder(config.hidden_size)

        self.blocks = nn.ModuleList(
            [
                FlowBlock(
                    config,
                    layer_idx=layer_idx,
                    disable_cross_attn=disable_cross_attn,
                    disable_self_attn=disable_self_attn,
                    disable_mlp=disable_mlp,
                )
                for _ in range(num_blocks)
            ]
        )
        self.rotary_emb = Gemma2RotaryEmbedding(config=config)

    def forward(
        self,
        h,
        concept_hidden,
        concept_mask=None,
        t=None,
        self_attn_caches=None,
        use_cache=False,
        padding_mask=None,
        past_len=0,
        position_ids=None,
    ):
        h_input = h
        t_emb = self.time_embed(t) if self.time_conditioned and t is not None else None

        seq_len = h.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=h.device
            ).unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if self_attn_caches is not None:
                sc = self_attn_caches[i]
            elif use_cache:
                sc = DynamicCache()
            else:
                sc = None
            h, kv_cache = block(
                h,
                concept_hidden,
                concept_mask,
                t_emb=t_emb,
                self_attn_cache=sc,
                position_embeddings=position_embeddings,
                padding_mask=padding_mask,
                use_cache=use_cache,
                q_pos_offset=past_len,
                activation_position_ids=position_ids,
            )
            if use_cache:
                new_caches.append(kv_cache)

        return h - h_input, new_caches


class ConceptEncoder(nn.Module):
    """Frozen two-layer Gemma encoder for concept text."""

    def __init__(self, model_id="google/gemma-2-2b-it", num_layers=2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_id)
        full_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

        self.embed_tokens = full_model.model.embed_tokens
        self.layers = nn.ModuleList([full_model.model.layers[i] for i in range(num_layers)])
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm.load_state_dict(full_model.model.norm.state_dict())
        self.hidden_size = config.hidden_size
        self.rotary_emb = full_model.model.rotary_emb

        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_base_model(cls, base_model, num_layers=2):
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        cfg = base_model.config
        base = base_model.model
        self.embed_tokens = base.embed_tokens
        self.layers = nn.ModuleList(list(base.layers[:num_layers]))
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        self.hidden_size = cfg.hidden_size
        for p in self.parameters():
            p.requires_grad = False
        return self

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        if not hasattr(self.embed_tokens, "embed_scale"):
            h = h * (self.hidden_size ** 0.5)

        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(h, position_ids)

        min_val = torch.finfo(h.dtype).min
        causal = torch.triu(
            torch.full((seq_len, seq_len), min_val, device=h.device, dtype=h.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) * min_val
            mask_4d = (causal + pad_mask).clamp(min=min_val)
        else:
            mask_4d = causal.expand(bsz, -1, -1, -1)

        for layer in self.layers:
            out = layer(
                h,
                attention_mask=mask_4d,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            h = out[0] if isinstance(out, tuple) else out

        return self.norm(h)



def robust_stats(x: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    x64 = x.detach().to(torch.float64)
    median = x64.median(dim=0).values.to(torch.float32)
    q25 = torch.quantile(x64, 0.25, dim=0).to(torch.float32)
    q75 = torch.quantile(x64, 0.75, dim=0).to(torch.float32)
    iqr = (q75 - q25).clamp(min=float(eps))
    return {"median": median, "iqr": iqr}


def gaussian_stats(x: torch.Tensor, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    x64 = x.detach().to(torch.float64)
    mean = x64.mean(dim=0).to(torch.float32)
    std = x64.std(dim=0, unbiased=False).to(torch.float32).clamp(min=float(eps))
    return {"mean": mean, "std": std}


def normalize(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    # Support both robust (median/iqr) and gaussian (mean/std) stats.
    if "median" in stats and "iqr" in stats:
        return (x - stats["median"].to(x.device)) / stats["iqr"].to(x.device)
    if "mean" in stats and "std" in stats:
        return (x - stats["mean"].to(x.device)) / stats["std"].to(x.device)
    raise KeyError("Unsupported stats keys for normalize(); expected median/iqr or mean/std")


def denormalize(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "median" in stats and "iqr" in stats:
        return x * stats["iqr"].to(x.device) + stats["median"].to(x.device)
    if "mean" in stats and "std" in stats:
        return x * stats["std"].to(x.device) + stats["mean"].to(x.device)
    raise KeyError("Unsupported stats keys for denormalize(); expected median/iqr or mean/std")


def compute_svd_basis(data: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data for basis computation, got shape {tuple(data.shape)}")
    if dim is None or int(dim) <= 0:
        raise ValueError(f"Expected positive dim for basis computation, got {dim}")
    mean = data.mean(dim=0, keepdim=True)
    centered = data - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    k = min(int(dim), vh.shape[0])
    return vh[:k].detach().cpu().to(torch.float32), mean.detach().cpu().to(torch.float32)


def project_to_basis(
    x: torch.Tensor,
    basis: torch.Tensor | None,
    mean: torch.Tensor | None = None,
) -> torch.Tensor:
    if basis is None:
        return x
    basis = basis.to(device=x.device, dtype=x.dtype)
    if mean is not None:
        x = x - mean.to(device=x.device, dtype=x.dtype)
    return x @ basis.T


def unproject_from_basis(
    x: torch.Tensor,
    basis: torch.Tensor | None,
    mean: torch.Tensor | None = None,
) -> torch.Tensor:
    if basis is None:
        return x
    basis = basis.to(device=x.device, dtype=x.dtype)
    x = x @ basis
    if mean is not None:
        x = x + mean.to(device=x.device, dtype=x.dtype)
    return x


def train_flow_model(
    source: torch.Tensor,
    target: torch.Tensor,
    hidden_dim: int = 512,
    n_layers: int = 2,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cuda",
    loss_mode: str = "mse",
    max_weight: float | None = None,
    weighted: bool = True,
) -> Tuple[FlowMLP, Dict[str, float]]:
    """Train a rectified-flow velocity model from source to target."""
    if source.shape != target.shape:
        raise ValueError(f"source and target shape mismatch: {source.shape} vs {target.shape}")

    train_device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    x0 = source.detach().to(torch.float32).cpu()
    x1 = target.detach().to(torch.float32).cpu()
    model = FlowMLP(dim=x0.shape[-1], hidden_dim=hidden_dim, n_layers=n_layers).to(train_device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    n = x0.shape[0]
    bs = max(1, min(int(batch_size), n))
    last_loss = 0.0

    # Prepare optional per-dimension weights for MSE mode. Use variance of velocity (b1-b0)
    loss_mode = str(loss_mode).strip().lower()
    if loss_mode not in {"mse", "huber"}:
        raise ValueError(f"Unsupported loss_mode '{loss_mode}'. Expected 'mse' or 'huber'.")
    weight_vec = None
    if loss_mode == "mse" and weighted:
        # compute velocity statistics on CPU then move to train device as needed
        vel_all = (x1 - x0).to(torch.float32)
        var = vel_all.var(dim=0, unbiased=False)
        if max_weight is not None:
            var = torch.clamp(var, max=float(max_weight))
        # normalize weights to have mean 1 to keep loss scale stable
        denom = var.mean().item() if var.numel() > 0 else 1.0
        if denom == 0:
            denom = 1.0
        weight_vec = (var / float(denom)).to(train_device)

    for _ in tqdm(range(max(int(epochs), 1)), desc="Training"):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            b0 = x0[idx].to(train_device)
            b1 = x1[idx].to(train_device)
            t = torch.rand(b0.shape[0], device=train_device)
            xt = (1.0 - t[:, None]) * b0 + t[:, None] * b1
            velocity = b1 - b0
            pred = model(xt, t)
            if loss_mode == "huber":
                loss = F.huber_loss(pred, velocity, delta=1.0)
            else:
                # weighted MSE per-dimension
                # weight_vec shape: (d,), pred/velocity shape: (bs, d)
                w = weight_vec
                if w is None:
                    loss = F.mse_loss(pred, velocity)
                else:
                    loss = ((pred - velocity) ** 2) * w.unsqueeze(0)
                    loss = loss.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())

    model = model.cpu().eval()
    return model, {"train_loss": last_loss, "epochs": int(epochs), "n_samples": int(n)}


def solve_flow(
    model: FlowMLP,
    source: torch.Tensor,
    steps: int = 16,
    guidance: torch.Tensor | None = None,
    guidance_strength: float = 0.0,
    guidance_mode: str = "fixed",
) -> torch.Tensor:
    """Euler solve dx/dt = v_theta(x,t), optionally with simple guidance."""
    x = source
    n_steps = max(int(steps), 1)
    dt = 1.0 / n_steps
    guidance_mode = str(guidance_mode).strip().lower()
    if guidance_mode not in {"fixed", "off"}:
        raise ValueError(f"Unsupported guidance_mode '{guidance_mode}'. Expected 'fixed' or 'off'.")

    for i in range(n_steps):
        t = torch.full((x.shape[0],), i / n_steps, device=x.device, dtype=x.dtype)
        v = model(x, t)
        if guidance_mode != "off" and guidance is not None and guidance_strength:
            g = guidance.to(device=x.device, dtype=x.dtype)
            if g.ndim == 1:
                g = g.unsqueeze(0).expand_as(v)
            g = F.normalize(g, dim=-1)
            v = v + float(guidance_strength) * g
        x = x + dt * v
    return x


def flow_steer_activations(
    x: torch.Tensor,
    flow_model: FlowMLP,
    source_stats: Dict[str, torch.Tensor],
    target_stats: Dict[str, torch.Tensor],
    basis: torch.Tensor | None = None,
    basis_mean: torch.Tensor | None = None,
    basis_inv: torch.Tensor | None = None,
    train_space: str = "full",
    steps: int = 1,
    denoise_mode: str = "none",
    coeff: float = 1.0,
    guidance: torch.Tensor | None = None,
    guidance_strength: float = 0.0,
    guidance_mode: str = "off",
) -> torch.Tensor:
    """Shared transformation pipeline used by both training and inference.

    Projects activations to flow space, normalizes, solves flow (Euler),
    denormalizes, restores from flow space, and handles nullspace denoising.

    Differentiable through flow_model when steps=1 (training path).
    """
    train_space = str(train_space).strip().lower()
    denoise_mode = str(denoise_mode).strip().lower()

    dev = x.device
    dtype = x.dtype
    if basis is not None:
        basis = basis.to(device=dev, dtype=dtype)
    if basis_mean is not None:
        basis_mean = basis_mean.to(device=dev, dtype=dtype)
    if basis_inv is not None:
        basis_inv = basis_inv.to(device=dev, dtype=dtype)
    source_stats = {k: v.to(device=dev, dtype=dtype) if isinstance(v, torch.Tensor) else v for k, v in source_stats.items()}
    target_stats = {k: v.to(device=dev, dtype=dtype) if isinstance(v, torch.Tensor) else v for k, v in target_stats.items()}

    # 1. Project to flow space
    if basis is not None and train_space != "full":
        source_coords = project_to_basis(x, basis, basis_mean)
    else:
        source_coords = x

    # 2. Normalize
    source = normalize(source_coords, source_stats)

    # 3. Project guidance into flow space if present
    guidance_coords = None
    if guidance is not None:
        if basis is not None:
            guidance_coords = guidance @ basis.T
        else:
            guidance_coords = guidance

    # 4. ODE solve
    solved = solve_flow(
        flow_model, source, steps=steps,
        guidance=guidance_coords,
        guidance_strength=guidance_strength,
        guidance_mode=guidance_mode,
    )

    # 5. Denormalize
    target_coords = denormalize(solved, target_stats)

    # 6. Restore from flow space
    if basis is not None and train_space != "full":
        if basis_inv is not None:
            target_full = target_coords @ basis_inv.T
            if basis_mean is not None:
                target_full = target_full + basis_mean
        else:
            target_full = unproject_from_basis(target_coords, basis, basis_mean)
    else:
        target_full = target_coords

    # 7. Denoise modes
    if denoise_mode == "correction":
        correction = target_full - x
        if basis is not None:
            coords = project_to_basis(correction, basis, None)
            if basis_inv is not None:
                correction = coords @ basis_inv.T
            else:
                correction = unproject_from_basis(coords, basis, None)
        return x + float(coeff) * correction

    if denoise_mode == "proj" and basis is not None:
        # Project target to basis
        coords = project_to_basis(target_full, basis, basis_mean)
        if basis_inv is not None:
            target_proj = coords @ basis_inv.T
            if basis_mean is not None:
                target_proj = target_proj + basis_mean
        else:
            target_proj = unproject_from_basis(coords, basis, basis_mean)
        # Nullspace of original
        coords_x = project_to_basis(x, basis, basis_mean)
        if basis_inv is not None:
            x_proj = coords_x @ basis_inv.T
            if basis_mean is not None:
                x_proj = x_proj + basis_mean
        else:
            x_proj = unproject_from_basis(coords_x, basis, basis_mean)
        return target_proj + (x - x_proj)

    return target_full