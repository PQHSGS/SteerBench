"""
Dense steering model implementations.
Contains steer models that apply dense steering vectors directly to residual streams.
"""

from typing import List, Optional, Dict, Union
from functools import partial
import torch
from ..base import BaseSteerModel
from ..logger import setup_logger
from ..utils import get_resid_acts, set_resid_acts

logger = setup_logger(__name__)

class DenseSteerModel(BaseSteerModel):
    """Wrapper for dense steering vector application."""

    def hook_fn(
        self, resid: torch.Tensor, coeff: float, steering_vector: torch.Tensor, position: Union[str,int], hook, **kwargs
    ) -> torch.Tensor:
        """Add steering vector while preserving residual norm."""
        if position == "last":
            resid[:, -1, :] = resid[:, -1, :] + coeff * steering_vector.to(dtype=resid.dtype)
        elif position == "all":
            resid += coeff*steering_vector.to(dtype=resid.dtype)
        else:
            resid[:,position,:] = resid[:, position, :] + coeff * steering_vector.to(dtype=resid.dtype)
        return resid


class RiemannianSteerModel(BaseSteerModel):
    """Riemannian Activation Steering via exponential map on product of spheres.

    Instead of linear addition (h + coeff·v), applies the Riemannian exponential
    map on the sphere S^{d-1}(r_k) at each sample's activation point, with
    Mahalanobis preconditioning from within-class covariance:
        1. Precondition: v' = M · v  (M = (Σ_w + λI)^{-1}, warps space by concept geometry)
        2. Map h to sphere: h_sphere = h / ||h|| · r_k
        3. Project v' onto tangent space: w = v' - (v'·h_sphere / r_k²) · h_sphere
        4. Exponential map: h_steered = cos(θ)·h_sphere + sin(θ)·(w/||w||)·r_k
           where θ = coeff · ||w|| / r_k
        5. Scale back: h_steered = h_steered · (||h|| / r_k)

    Mahalanobis metric makes movement fast along concept-separation axes
    (high variance in Σ_w) and slow along overlap axes (low variance).
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        riemannian_calpha: float = 1.0,
        riemannian_mahal: Optional[Dict[int, torch.Tensor]] = None,
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
        **kwargs,
    ):
        super().__init__(
            model=model, layer=layer, steering_vector=steering_vector,
            hook_point=hook_point, position=position, **kwargs,
        )
        self.riemannian_calpha = float(riemannian_calpha)
        self.riemannian_mahal = riemannian_mahal

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        **kwargs,
    ) -> torch.Tensor:
        acts = get_resid_acts(resid, position)
        h = acts.to(dtype=torch.float64)
        v = steering_vector.to(dtype=torch.float64, device=h.device)

        if self.riemannian_mahal is not None:
            layer_idx = int(hook.name.split(".")[1])
            M = self.riemannian_mahal.get(layer_idx)
            if M is not None:
                M = M.to(dtype=torch.float64, device=h.device)
                v = M @ v

        batch_size, D = h.shape
        h_norm = h.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)

        alpha_k = self.riemannian_calpha * h_norm / D
        r = torch.sqrt(alpha_k)

        h_sphere = h / h_norm * r

        v_radial = (h_sphere * (h_sphere * v).sum(dim=-1, keepdim=True)) / (r * r)
        w = v - v_radial
        w_norm = w.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)

        theta = coeff * w_norm / r
        w_hat = w / w_norm

        h_steered = (
            torch.cos(theta) * h_sphere + torch.sin(theta) * w_hat * r
        )

        h_steered = h_steered * (h_norm / r)

        updated = h_steered.to(dtype=acts.dtype)
        return set_resid_acts(resid, position, updated)


class SAEFreeSteerModel(DenseSteerModel):
    """
    SAE-FREE steering model with GT-style residual renormalization.

    GT Code/SAE-free/free_gsm.py behavior:
    1) add coeff * steering_vector
    2) preserve per-token residual norm after intervention
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
        saefree_norm_eps: float = 1e-8,
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
        self.saefree_norm_eps = saefree_norm_eps

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        **kwargs,
    ) -> torch.Tensor:
        vec = steering_vector.to(device=resid.device, dtype=resid.dtype)
        original = get_resid_acts(resid, position)
        steered = original + coeff * vec
        if self.norm:
            orig_norm = original.norm(p=2, dim=-1, keepdim=True).clamp(min=self.saefree_norm_eps)
            new_norm = steered.norm(p=2, dim=-1, keepdim=True).clamp(min=self.saefree_norm_eps)
            steered = steered * (orig_norm / new_norm)
        return set_resid_acts(resid, position, steered)

class ConditionalSteerModel(BaseSteerModel):
    """Wrapper for conditional activation steering (CAST)."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        position: Union[str,int] = 'last',
        conditional_vector: Optional[torch.Tensor] = None,
        conditional_threshold: Optional[List[float]] = None,
        conditional_layer: Optional[int] = None,
        verbose: bool = False,
        conditional_threshold_is: str = "smaller",
        conditional_pos: str = "mean",
        load_conditional_vector: Optional[str] = None,
        hook_point: str = "pre",
        apply_to_all_tokens: bool = True,
        norm: bool = True,
        **kwargs,
    ):
        super().__init__(model, layer, steering_vector, hook_point=hook_point, position=position, **kwargs)
        self.conditional_layer = conditional_layer
        self.verbose = verbose
        self.conditional_threshold_is = conditional_threshold_is
        self.conditional_pos = conditional_pos
        self.condition_triggered = None
        if apply_to_all_tokens: self.position = "all"
        self.norm = norm

        self.use_conditional = conditional_vector is not None
        if not self.use_conditional:
            self.threshold = None
            return
        cond = conditional_vector
        if load_conditional_vector is not None:
            cond = torch.load(load_conditional_vector).to(self.device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        self.conditional_vector = cond

        logger.info(f"Conditional vector shape: {cond.shape}")

        n_cond = cond.shape[0]
        if conditional_threshold is None:
            self.threshold = torch.zeros(n_cond, device=self.device)
        elif isinstance(conditional_threshold, (int, float)):
            self.threshold = torch.full((n_cond,), float(conditional_threshold), device=self.device)
        else:
            self.threshold = torch.tensor(conditional_threshold, device=self.device)

    def _conditional_hook(self, resid: torch.Tensor, condtional_vector, hook):
        """Check condition and set trigger flag."""
        if self.condition_triggered is not None:
             return resid

        h = resid.mean(dim=1) if self.conditional_pos == "mean" else resid[:, -1]
        c = condtional_vector.to(dtype=resid.dtype, device=resid.device)

        # Batched projection: Proj_c(h) = (h . c / |c|^2) * c
        hc = torch.einsum("bd,nd->bn", h, c)
        cc = (c * c).sum(dim=1)
        proj = (hc / (cc + 1e-8)).unsqueeze(-1) * c
        
        proj_tanh = torch.tanh(proj)
        h_norm = h.norm(dim=1, keepdim=True)
        proj_tanh_norm = proj_tanh.norm(dim=2) 

        sims = torch.einsum("bd,bnd->bn", h, proj_tanh) / (h_norm * proj_tanh_norm + 1e-8)

        if self.conditional_threshold_is == "smaller":
             triggered = (sims > self.threshold).any(dim=1)
        else:
             triggered = (sims < self.threshold).any(dim=1)
        
        self.condition_triggered = triggered.float().unsqueeze(1)

        if self.verbose:
            logger.debug(f"[COND] triggered={int(triggered.any().item())}, sims_range=[{sims.min().item():.3f}, {sims.max().item():.3f}]")
        
        if self._prompt_metadata is None:
            self._reset_prompt_metadata()
        self._prompt_metadata.setdefault("triggered", triggered.float().cpu().tolist())
        self._prompt_metadata.setdefault("sims", sims.max(dim=1).values.detach().cpu().tolist())
        return resid

    def _apply_ooi_normalization(self, h: torch.Tensor, orig_norm: torch.Tensor) -> torch.Tensor:
        """Apply out-of-input (OOI) preventive normalization."""
        new_norm = h.norm(dim=-1, keepdim=True)
        max_ratio = (new_norm / orig_norm).max().item()
        
        if max_ratio > 1 or torch.isnan(h).any() or torch.isinf(h).any():
            if self.verbose:
                logger.debug(f"[OOI] Normalizing. Max ratio: {max_ratio:.3f}")
            h = h * (orig_norm / (new_norm + 1e-8))
        return h

    def hook_fn(
        self, resid: torch.Tensor, coeff: float, steering_vector: torch.Tensor,position: Union[str,int], hook, **kwargs
    ) -> torch.Tensor:
        """Apply conditional steering."""
        orig_norm = resid.norm(dim=-1, keepdim=True)
        vec = steering_vector.to(dtype=resid.dtype)
        
        if self.condition_triggered is None:
            # No conditional - apply steering directly
            if position == "last":
                resid[:, -1, :] = resid[:, -1, :] + coeff * vec
            elif position == "all":
                resid += coeff*vec
            else:
                resid[:,position,:] =  resid[:, position, :] + coeff * vec
            return resid
        else:
            # Conditional steering with mask
            mask = self.condition_triggered
            if position == "last":
                resid[:, -1, :] = resid[:, -1, :] + coeff * mask.squeeze(-1) * vec
            elif position == "all":
                resid += coeff * vec
            else:
                resid[:,position,:] = resid[:, position, :] + coeff * mask.squeeze(-1) * vec
            
        if self.norm:
            resid = self._apply_ooi_normalization(resid, orig_norm)
        return resid

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        """Setup conditional and steering hooks."""
        hook_handles = []
        if self.use_conditional:
            self.condition_triggered = None
            cond_layer = [self.conditional_layer] if self.conditional_layer is not None else None
            for hook in self.get_hook_name(layer=cond_layer):
                hook_handles.append(self.model.add_hook(hook, partial(self._conditional_hook, condtional_vector=self.conditional_vector)))
        hook_handles.extend(super().setup_hooks(coeff))
        return hook_handles

    
class ManifoldSteerModel(BaseSteerModel):
    """
    Manifold Steering Model.

    Applies the intervention h' = h - alpha * (h @ r) * r,
    where r is the manifold-projected overthinking direction.
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "all",
        apply_all_layers: bool = True,
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
        self.apply_all_layers = apply_all_layers
        self.n_layers = model.cfg.n_layers

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        **kwargs,
    ) -> torch.Tensor:
        """Apply projection ablation on selected residual activations."""
        acts = get_resid_acts(resid, position)
        r = steering_vector.to(device=acts.device, dtype=acts.dtype)

        if acts.ndim == 2:
            proj = torch.einsum("bd,d->b", acts, r)
        elif acts.ndim == 3:
            proj = torch.einsum("bsd,d->bs", acts, r)
        else:
            raise ValueError(f"Unsupported activation shape for MANIFOLD steering: {acts.shape}")

        steered = acts - coeff * proj.unsqueeze(-1) * r
        return set_resid_acts(resid, position, steered)

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        """Install hooks across all layers by default, matching paper intervention."""
        if not self.apply_all_layers:
            return super().setup_hooks(coeff)

        if not self.layer:
            raise ValueError("MANIFOLD requires at least one layer in steer config.")

        anchor_layer = self.layer[0]
        anchor_vector = self.steering_vector[anchor_layer]
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
                        steering_vector=anchor_vector,
                        position="all",
                        shared_position="all",
                    ),
                )
                hook_handles.append(handle)
        hook_handles.extend(self._setup_post_process_hooks())

        return hook_handles


class ManifoldPaperSteerModel(BaseSteerModel):
    """
    Manifold paper steer model.

    Applies a projection-ablation style intervention using a provided
    manifold direction `r`. Accepts optional manifold basis/mean in
    `manifold_payload` to allow more advanced steer-time operations.
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        manifold_payload: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
        apply_all_layers: bool = True,
        **kwargs,
    ):
        super().__init__(model=model, layer=layer, steering_vector=steering_vector, hook_point=hook_point, position=position, **kwargs)
        self.manifold_payload = {} if manifold_payload is None else manifold_payload
        self.apply_all_layers = apply_all_layers
        self.n_layers = model.cfg.n_layers

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        **kwargs,
    ) -> torch.Tensor:
        acts = get_resid_acts(resid, position)
        r = steering_vector.to(device=acts.device, dtype=acts.dtype)

        if acts.ndim == 2:
            proj = torch.einsum("bd,d->b", acts, r)
            steered = acts - coeff * proj.unsqueeze(-1) * r
        elif acts.ndim == 3:
            proj = torch.einsum("bsd,d->bs", acts, r)
            steered = acts - coeff * proj.unsqueeze(-1) * r
        else:
            raise ValueError(f"Unsupported activation shape for MANIFOLD_PAPER steering: {acts.shape}")

        return set_resid_acts(resid, position, steered)

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        if not self.apply_all_layers:
            return super().setup_hooks(coeff)

        if not self.layer:
            raise ValueError("MANIFOLD_PAPER requires at least one anchor layer in steer config.")

        anchor_layer = self.layer[0]
        anchor_vector = self.steering_vector[anchor_layer]
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
                        steering_vector=anchor_vector,
                        position="all",
                        shared_position="all",
                    ),
                )
                hook_handles.append(handle)
        hook_handles.extend(self._setup_post_process_hooks())

        return hook_handles


class SphericalSteerModel(BaseSteerModel):
    """
    Spherical Steering model with norm-preserving geodesic rotation.

    Implements paper-style steering:
    - Build confidence from vMF-inspired prototype scores
    - Convert confidence to bounded steering strength
    - Apply SLERP-like rotation toward truthful prototype while preserving norm
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        spherical_kappa: float = 20.0,
        spherical_alpha: float = 0.7,
        spherical_beta: float = -0.15,
        spherical_use_vmf_gate: bool = True,
        **kwargs,
    ):
        super().__init__(model=model, layer=layer, steering_vector=steering_vector, **kwargs)
        self.spherical_kappa = float(spherical_kappa)
        self.spherical_alpha = float(spherical_alpha)
        self.spherical_beta = float(spherical_beta)
        self.spherical_use_vmf_gate = bool(spherical_use_vmf_gate)

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        spherical_mu_h: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        acts = get_resid_acts(resid, position)
        original_shape = acts.shape
        flat = acts.reshape(-1, acts.shape[-1]).to(dtype=torch.float32)

        mu_t = steering_vector.to(device=flat.device, dtype=torch.float32)
        mu_t = mu_t / mu_t.norm(p=2).clamp(min=1e-8)

        if spherical_mu_h is None:
            mu_h = -mu_t
        else:
            mu_h = spherical_mu_h.to(device=flat.device, dtype=torch.float32)
            mu_h = mu_h / mu_h.norm(p=2).clamp(min=1e-8)

        orig_norm = flat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        flat_hat = flat / orig_norm

        cos_t = torch.clamp(flat_hat @ mu_t, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        cos_h = torch.clamp(flat_hat @ mu_h, min=-1.0 + 1e-6, max=1.0 - 1e-6)

        if self.spherical_use_vmf_gate:
            logits = torch.stack(
                [self.spherical_kappa * cos_t, self.spherical_kappa * cos_h],
                dim=-1,
            )
            probs = torch.softmax(logits, dim=-1)
            p_t, p_h = probs[:, 0], probs[:, 1]
            delta = p_h - p_t

            t = torch.zeros_like(delta)
            active = delta > self.spherical_beta
            denom = max(1.0 - self.spherical_beta, 1e-8)
            t[active] = self.spherical_alpha * (delta[active] - self.spherical_beta) / denom
        else:
            t = torch.full_like(cos_t, self.spherical_alpha)

        t = torch.clamp(t * abs(float(coeff)), min=0.0, max=1.0)

        theta = torch.acos(cos_t)
        sin_theta = torch.sin(theta)
        safe = sin_theta > 1e-6

        flat_new_hat = flat_hat.clone()
        if safe.any():
            theta_new = (1.0 - t[safe]) * theta[safe]
            u = (
                flat_hat[safe] - cos_t[safe].unsqueeze(-1) * mu_t.unsqueeze(0)
            ) / sin_theta[safe].unsqueeze(-1)

            flat_new_hat[safe] = (
                torch.cos(theta_new).unsqueeze(-1) * mu_t.unsqueeze(0)
                + torch.sin(theta_new).unsqueeze(-1) * u
            )

        flat_new = flat_new_hat * orig_norm
        updated = flat_new.reshape(original_shape).to(dtype=acts.dtype)

        if self._prompt_metadata is None:
            self._reset_prompt_metadata()
        self._prompt_metadata.setdefault(
            "spherical_trigger_rate",
            float((t > 0).float().mean().item()),
        )
        self._prompt_metadata.setdefault(
            "spherical_mean_strength",
            float(t.mean().item()),
        )

        return set_resid_acts(resid, position, updated)
