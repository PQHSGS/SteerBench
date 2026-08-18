"""Transport-based steer models: ACT (with PCA-OT), CHARS, LinNEAS."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from ..base import BaseSteerModel
from ..logger import setup_logger
from ..utils import get_resid_acts, set_resid_acts

logger = setup_logger(__name__)


# =============================================================================
# Activation Transport Steer Model — includes PCA-OT
# =============================================================================

class ActivationTransportSteerModel(BaseSteerModel):
    """Apply Mean/Gaussian/Linear/PCA-OT Activation Transport maps."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        act_stats: Dict[int, Dict[str, torch.Tensor]],
        act_support: str = "q_all",
        act_std_eps: float = 1e-4,
        apply_all_layers: bool = True,
        hook_point: Union[str, List[str]] = "post",
        position: Union[str, int] = "all",
        **kwargs,
    ):
        super().__init__(
            model=model,
            layer=layer,
            steering_vector=steering_vector,
            hook_point=hook_point,
            position=position,
            **kwargs,
        )
        self.act_stats = {int(k): v for k, v in act_stats.items()}
        self.act_support = act_support
        self.act_std_eps = float(act_std_eps)
        self.apply_all_layers = apply_all_layers

        if apply_all_layers:
            n_layers = model.cfg.n_layers
            missing = [l for l in range(n_layers) if l not in self.act_stats]
            if missing:
                raise ValueError(
                    f"apply_all_layers=True requires act_stats for all {n_layers} layers, "
                    f"but missing: {missing}. "
                    f"Extract on all layers or set apply_all_layers=False."
                )

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        if not self.apply_all_layers:
            return super().setup_hooks(coeff)

        n_layers = self.model.cfg.n_layers
        anchor_layer = self.layer[0]
        default_coeff = coeff.get(anchor_layer, next(iter(coeff.values())))

        hook_handles = []
        for layer_idx in range(n_layers):
            layer_coeff = coeff.get(layer_idx, default_coeff)
            for hook_name in self.get_hook_name(layer=[layer_idx]):
                handle = self.model.add_hook(
                    hook_name,
                    partial(
                        self._apply_steering_hook,
                        hook_fn=self.hook_fn,
                        coeff=layer_coeff,
                        position=self.position,
                        steering_vector=self.steering_vector[layer_idx]
                    ),
                )
                hook_handles.append(handle)
        return hook_handles

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        **kwargs,
    ) -> torch.Tensor:
        layer = int(hook.name.split(".")[1])

        if self.apply_all_layers:
            if layer not in self.act_stats:
                raise KeyError(
                    f"Layer {layer} not found in act_stats (available: {sorted(self.act_stats.keys())}). "
                    f"Set apply_all_layers=False to steer only configured layers."
                )
        else:
            if layer not in self.act_stats and layer in self.layer:
                raise KeyError(
                    f"Layer {layer} is in steer config but not in act_stats. "
                    f"Available: {sorted(self.act_stats.keys())}."
                )
            elif layer not in self.act_stats:
                return resid

        raw_stats = self.act_stats[layer]
        stats = {k: (v.to(resid.device) if isinstance(v, torch.Tensor) else v) for k, v in raw_stats.items()}
        acts = get_resid_acts(resid, position)
        dtype = acts.dtype
        x = acts.to(torch.float32)

        if "pca_components" in stats:
            P = stats["pca_components"].to(torch.float32)
            A = stats["transport_matrix"].to(torch.float32)
            b = stats["transport_bias"].to(torch.float32)
            mu = stats["pooled_mean"].to(torch.float32)
            x_centered = x - mu
            z = x_centered @ P
            z_t = z @ A.T + b
            delta = (z_t - z) @ P.T
            transported = x + delta
        else:
            omega = stats.get("omega")
            beta = stats.get("beta")
            transported = omega.to(device=x.device, dtype=torch.float32) * x + beta.to(device=x.device, dtype=torch.float32)
            if self.act_support != "q_all":
                support_min = stats["support_min"].to(torch.float32)
                support_max = stats["support_max"].to(torch.float32)
                mask = ((x >= support_min) & (x <= support_max)).to(torch.float32)
                transported = mask * transported + (1.0 - mask) * x

        updated = (1.0 - float(coeff)) * x + float(coeff) * transported
        return set_resid_acts(resid, position, updated.to(dtype))


# =============================================================================
# CHARS Steer Model
# =============================================================================

class CHARSSteerModel(BaseSteerModel):
    """Concept Heterogeneity-aware Representation Steering (CHARS) Model."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        chars_centroids_A: Dict[int, torch.Tensor],
        chars_centroids_B: Dict[int, torch.Tensor],
        chars_coupling: Dict[int, torch.Tensor],
        chars_k: Dict[int, int],
        chars_components: Optional[Dict[int, torch.Tensor]] = None,
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
        chars_mode: str = "addition",
        chars_pct: bool = False,
        chars_pct_l: int = 4,
        chars_diag: bool = False,
        chars_clip_tail: bool = False,
        chars_clip_z: float = 3.0,
        chars_std_A: Optional[Dict[int, torch.Tensor]] = None,
        chars_std_B: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            layer=layer,
            steering_vector=steering_vector,
            hook_point=hook_point,
            position=position,
            chars_centroids_A=chars_centroids_A,
            chars_centroids_B=chars_centroids_B,
            chars_coupling=chars_coupling,
            chars_k=chars_k,
            chars_components=chars_components,
            chars_std_A=chars_std_A,
            chars_std_B=chars_std_B,
            **kwargs,
        )
        self.chars_mode = chars_mode
        self.chars_pct = chars_pct
        self.chars_pct_l = chars_pct_l
        self.chars_diag = chars_diag
        self.chars_clip_tail = chars_clip_tail
        self.chars_clip_z = chars_clip_z
        self.chars_tail_transform = kwargs.get("chars_tail_transform", "none")
        self.chars_pca_k = kwargs.get("chars_pca_k", 0)
        self.chars_P_concept = kwargs.get("chars_P_concept", None)
        self.chars_X_mean = kwargs.get("chars_X_mean", None)
        if self.chars_P_concept is not None:
            self.chars_P_concept = {int(k): v for k, v in self.chars_P_concept.items()}


    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(x.abs())

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        chars_centroids_A: torch.Tensor,
        chars_centroids_B: torch.Tensor,
        chars_coupling: torch.Tensor,
        chars_k: int,
        chars_components: Optional[torch.Tensor] = None,
        chars_std_A: Optional[torch.Tensor] = None,
        chars_std_B: Optional[torch.Tensor] = None,
        chars_P_concept: Optional[torch.Tensor] = None,
        chars_X_mean: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        device = resid.device
        dtype = resid.dtype

        acts = get_resid_acts(resid, position)
        orig_shape = acts.shape
        x = acts.reshape(-1, orig_shape[-1])

        if self.chars_clip_tail:
            x_mean = x.mean(dim=-1, keepdim=True)
            x_std = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
            z = (x - x_mean) / x_std
            z = torch.clamp(z, -self.chars_clip_z, self.chars_clip_z)
            x = x_mean + z * x_std

        # Subspace projection (concept-space K-means)
        project_subspace = (
            chars_P_concept is not None and 
            isinstance(chars_P_concept, torch.Tensor) and 
            chars_X_mean is not None and 
            isinstance(chars_X_mean, torch.Tensor)
        )
        if project_subspace:
            P = chars_P_concept.to(device=device, dtype=dtype)
            xm = chars_X_mean.to(device=device, dtype=dtype)
            centroids_A = chars_centroids_A.to(device=device, dtype=dtype)
            centroids_B = chars_centroids_B.to(device=device, dtype=dtype)
            coupling = chars_coupling.to(device=device, dtype=dtype)
            x_sub = (x - xm) @ P
            v_hat_sub = self._compute_chars_transport(
                x_sub, centroids_A, centroids_B, coupling, chars_k,
                device, dtype, steering_vector, chars_components, chars_std_A, chars_std_B,
            )
            v_hat = v_hat_sub @ P.T
        else:
            centroids_A = chars_centroids_A.to(device=device, dtype=dtype)
            centroids_B = chars_centroids_B.to(device=device, dtype=dtype)
            coupling = chars_coupling.to(device=device, dtype=dtype)
            v_hat = self._compute_chars_transport(
                x, centroids_A, centroids_B, coupling, chars_k,
                device, dtype, steering_vector, chars_components, chars_std_A, chars_std_B,
            )

        if self.norm:
            v_hat = v_hat / (v_hat.norm(p=2, dim=-1, keepdim=True) + 1e-12)

        if self.chars_mode == "ablation":
            v_hat_norm = v_hat / (v_hat.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            proj = torch.sum(x * v_hat_norm, dim=-1, keepdim=True)
            x_steered = x - coeff * proj * v_hat_norm
        else:
            x_steered = x + coeff * v_hat

        steered_acts = x_steered.reshape(orig_shape)
        return set_resid_acts(resid, position, steered_acts)

    def _compute_chars_transport(
        self,
        x: torch.Tensor,
        centroids_A: torch.Tensor,
        centroids_B: torch.Tensor,
        coupling: torch.Tensor,
        chars_k: int,
        device: torch.device,
        dtype: torch.dtype,
        steering_vector: torch.Tensor,
        chars_components: Optional[torch.Tensor] = None,
        chars_std_A: Optional[torch.Tensor] = None,
        chars_std_B: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply tail transform to activations and centroids before distance computation
        if self.chars_tail_transform == "symlog":
            x = self._symlog(x)
            centroids_A = self._symlog(centroids_A)
            centroids_B = self._symlog(centroids_B)

        diag_mode = (
            self.chars_diag and 
            chars_std_A is not None and 
            isinstance(chars_std_A, torch.Tensor) and 
            chars_std_B is not None and 
            isinstance(chars_std_B, torch.Tensor)
        )
        if diag_mode:
            std_A = chars_std_A.to(device=device, dtype=dtype)
            std_B = chars_std_B.to(device=device, dtype=dtype)
            rate = std_B.unsqueeze(0) / std_A.unsqueeze(1).clamp(min=1e-8)

        if chars_k == 1:
            if diag_mode:
                v_hat = (centroids_B[0] - centroids_A[0]) + (rate[0, 0] - 1.0) * (x - centroids_A[0].unsqueeze(0))
            else:
                v_hat = centroids_B[0] - centroids_A[0]
            v_hat = v_hat.unsqueeze(0).expand(x.shape[0], -1)
        else:
            dists = torch.sum((x.unsqueeze(1) - centroids_A.unsqueeze(0)) ** 2, dim=-1)
            dists_median = torch.median(dists, dim=-1).values.clamp(min=1e-8)
            kernel = torch.exp(-dists / (2.0 * dists_median.unsqueeze(-1)))

            if diag_mode:
                rate_minus_1 = rate - 1.0
                u_translate = torch.einsum("ij,ijd->id", coupling, centroids_B.unsqueeze(0) - centroids_A.unsqueeze(1))
                u_scale = torch.einsum("ij,ijd->id", coupling, rate_minus_1)
                s = coupling.sum(dim=-1)
                denom = torch.matmul(kernel, s)
                num_translate = torch.matmul(kernel, u_translate)
                x_minus_a = x.unsqueeze(1) - centroids_A.unsqueeze(0)
                num_scale = torch.sum(kernel.unsqueeze(-1) * u_scale.unsqueeze(0) * x_minus_a, dim=1)
                num = num_translate + num_scale
            else:
                diffs = centroids_B.unsqueeze(0) - centroids_A.unsqueeze(1)
                u = torch.einsum("ij,ijd->id", coupling, diffs)
                s = coupling.sum(dim=-1)
                denom = torch.matmul(kernel, s)
                num = torch.matmul(kernel, u)

            v_hat = num / (denom.unsqueeze(-1) + 1e-12)

        if self.chars_pct and chars_components is not None and isinstance(chars_components, torch.Tensor) and chars_k > 1:
            v_mean = steering_vector.to(device=device, dtype=dtype)
            v_centered = v_hat - v_mean.unsqueeze(0)
            components = chars_components.to(device=device, dtype=dtype)
            L = min(self.chars_pct_l, components.shape[0])
            components_L = components[:L]
            proj = torch.matmul(v_centered, components_L.T)
            v_projected = torch.matmul(proj, components_L)
            v_hat = v_mean.unsqueeze(0) + v_projected

        return v_hat


# =============================================================================
# LinNEAS Steer Model
# =============================================================================

class LinNEASSteerModel(BaseSteerModel):
    """Linearized Non-linear End-to-end Activation Steering (LinNEAS) model."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        linneas_w1: Dict[int, torch.Tensor],
        linneas_b1: Dict[int, torch.Tensor],
        apply_all_layers: bool = True,
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "all",
        **kwargs,
    ):
        super().__init__(
            model=model,
            layer=layer,
            steering_vector=steering_vector,
            hook_point=hook_point,
            position=position,
            **kwargs,
        )
        self.linneas_w1 = {int(k): v.to(device=self.device, dtype=self.model.cfg.dtype) for k, v in linneas_w1.items()}
        self.linneas_b1 = {int(k): v.to(device=self.device, dtype=self.model.cfg.dtype) for k, v in linneas_b1.items()}
        self.apply_all_layers = apply_all_layers
        self.n_layers = model.cfg.n_layers

        if apply_all_layers:
            missing = [l for l in range(self.n_layers) if l not in self.linneas_w1]
            if missing:
                raise ValueError(
                    f"apply_all_layers=True requires linneas_w1/b1 for all {self.n_layers} layers, "
                    f"but missing: {missing}. "
                    f"Extract on all layers or set apply_all_layers=False."
                )

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        hook,
        position: Union[str, int] = "all",
        **kwargs,
    ) -> torch.Tensor:
        layer_idx = int(hook.name.split(".")[1])

        if self.apply_all_layers:
            w = self.linneas_w1[layer_idx].to(device=resid.device, dtype=resid.dtype)
            b = self.linneas_b1[layer_idx].to(device=resid.device, dtype=resid.dtype)
        else:
            if layer_idx not in self.linneas_w1:
                raise KeyError(
                    f"Layer {layer_idx} not found in linneas_w1 (available: {sorted(self.linneas_w1.keys())}). "
                    f"Ensure steer config layer includes it, or set apply_all_layers=True."
                )
            w = self.linneas_w1[layer_idx].to(device=resid.device, dtype=resid.dtype)
            b = self.linneas_b1[layer_idx].to(device=resid.device, dtype=resid.dtype)

        h = get_resid_acts(resid, position)
        transported = h * w + b
        steered = coeff * transported + (1.0 - coeff) * h
        return set_resid_acts(resid, position, steered)

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        if not self.apply_all_layers:
            return super().setup_hooks(coeff)

        anchor_layer = self.layer[0]
        default_coeff = coeff.get(anchor_layer, next(iter(coeff.values())))

        hook_handles = []
        for layer_idx in range(self.n_layers):
            layer_coeff = coeff.get(layer_idx, default_coeff)
            for hook_name in self.get_hook_name(layer=[layer_idx]):
                handle = self.model.add_hook(
                    hook_name,
                    partial(
                        self._apply_steering_hook,
                        hook_fn=self.hook_fn,
                        coeff=layer_coeff,
                        position=self.position,
                    ),
                )
                hook_handles.append(handle)
        return hook_handles
