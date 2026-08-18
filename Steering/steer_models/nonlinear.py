"""Steer models for transport and nonlinear dense methods."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from ..base import BaseSteerModel
from ..config import SteeringVector
from ..flow_utils import ConceptEncoder, FlowFunction, FlowMLP, denormalize, flow_steer_activations, normalize, project_to_basis, solve_flow, unproject_from_basis
from ..utils import get_resid_acts, set_resid_acts
from ..post_process.subspace import SubspaceGLP
from ..utils import get_resid_acts, set_resid_acts

class AngularSteerModel(BaseSteerModel):
    """Angular Steering by rotation in a learned 2D subspace."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_plane: torch.Tensor,
        steering_vector: Optional[Dict[int, torch.Tensor]] = None,
        target_angle: float = 90.0,
        adaptive_mode: int = 1,
        feature_direction: Optional[torch.Tensor] = None,
        selected_layer: Optional[List[int]] = None,
        apply_all_layers: bool = True,
        hook_point: str = "mid",
        **kwargs,
    ):
        if selected_layer is not None:
            if isinstance(selected_layer, list) and len(selected_layer) > 0 and isinstance(selected_layer[0], list):
                layer = selected_layer[0]
            else:
                layer = selected_layer

        vector_dict = steering_vector if steering_vector is not None else {l: steering_plane[0] for l in layer}
        super().__init__(model, layer, vector_dict, hook_point=hook_point, **kwargs)
        self.steering_plane = steering_plane
        self.target_angle = target_angle
        self.adaptive_mode = adaptive_mode
        self.feature_direction = feature_direction if feature_direction is not None else steering_plane[0]
        self.apply_all_layers = apply_all_layers
        self.n_layers = model.cfg.n_layers
        self._cache = {}

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        hook,
        position: Optional[Union[str, int]] = None,
        ln_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        h, device, dtype = resid, resid.device, resid.dtype
        cache_key = (device, dtype, self.target_angle)

        if cache_key not in self._cache:
            f1 = self.steering_plane[0].to(device, torch.float64)
            f2 = self.steering_plane[1].to(device, torch.float64)
            b1 = f1 / f1.norm()
            b2 = f2 - (f2 @ b1) * b1
            b2 = b2 / b2.norm()

            theta = torch.tensor(
                self.target_angle * torch.pi / 180.0,
                device=device,
                dtype=torch.float64,
            )
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)

            self._cache[cache_key] = (
                (torch.outer(b1, b1) + torch.outer(b2, b2)).to(dtype),
                (cos_t * b1 + sin_t * b2).to(dtype),
                f1.to(dtype),
            )

        proj, steer_vec, f1 = self._cache[cache_key]

        if ln_scale is not None:
            ln_scale = ln_scale.to(device=device, dtype=dtype)
            h_curr = h * ln_scale
        else:
            h_curr = h

        projected = h_curr @ proj
        delta = (h_curr - projected + projected.norm(dim=-1, keepdim=True) * steer_vec) - h_curr
        delta *= coeff

        if self.adaptive_mode == 1:
            delta.masked_fill_((h_curr @ f1).unsqueeze(-1) <= 0, 0)

        return h + (delta / (ln_scale + 1e-6) if ln_scale is not None else delta)

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        if not self.apply_all_layers:
            return super().setup_hooks(coeff)

        hook_handles, eff_coeff = [], list(coeff.values())[0]
        hooks = [f"blocks.{i}.ln2.hook_normalized" for i in range(self.n_layers)] + [
            f"blocks.{i}.ln1.hook_normalized" for i in range(1, self.n_layers)
        ]

        for name in hooks:
            block_idx, type_name = int(name.split(".")[1]), name.split(".")[2]
            try:
                ln_scale = getattr(self.model.blocks[block_idx], type_name).w
            except AttributeError:
                ln_scale = None
            hook_handles.append(
                self.model.add_hook(
                    name,
                    partial(
                        self._apply_steering_hook,
                        hook_fn=self.hook_fn,
                        coeff=eff_coeff,
                        ln_scale=ln_scale,
                        shared_position="all",
                    ),
                )
            )
        hook_handles.extend(self._setup_post_process_hooks())
        return hook_handles





class PIDSteerModel(BaseSteerModel):
    """PID dense steering application."""

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
        vec = steering_vector.to(device=acts.device, dtype=acts.dtype)
        updated = acts + float(coeff) * vec
        if self.norm:
            original_norm = acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            updated_norm = updated.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            updated = updated * (original_norm / updated_norm)
        return set_resid_acts(resid, position, updated)


class CurveballSteerModel(BaseSteerModel):
    """Curveball polynomial KPCA pre-image steering."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        curveball_models: Dict[int, Any],
        curveball_directions: Dict[int, torch.Tensor],
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
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
        self.curveball_models = {int(k): v for k, v in curveball_models.items()}
        self.curveball_directions = {int(k): v.detach().cpu() for k, v in curveball_directions.items()}

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
        acts = get_resid_acts(resid, position)
        original_shape = acts.shape
        flat = acts.detach().to(torch.float32).reshape(-1, original_shape[-1]).cpu()

        kpca = self.curveball_models[layer]
        z = torch.tensor(kpca.transform(flat.numpy()), dtype=torch.float32)
        direction = self.curveball_directions[layer].to(z)
        if self.norm:
            direction= F.normalize(direction, dim=0)
        z_target = z + float(coeff) * direction

        inv_current = torch.tensor(kpca.inverse_transform(z.numpy()), dtype=torch.float32)
        inv_target = torch.tensor(kpca.inverse_transform(z_target.numpy()), dtype=torch.float32)
        residual = flat - inv_current
        steered = (inv_target + residual).reshape(original_shape).to(device=acts.device, dtype=acts.dtype)
        return set_resid_acts(resid, position, steered)


class FlowSteerModel(BaseSteerModel):
    """Apply a learned flow map from source activations to target activations."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        flow_models: Dict[int, Dict[str, Any]],
        flow_steps: int = 16,
        flow_denoise_mode: str = "none",
        flow_guidance_strength: float = 0.0,
        flow_guidance_mode: str = "fixed",
        hook_point: Union[str, List[str]] = "post",
        position: Union[str, int] = "last",
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
        self.flow_payload = {int(k): v for k, v in flow_models.items()}
        self.flow_steps = int(flow_steps)
        self.flow_denoise_mode = str(flow_denoise_mode).strip().lower()
        if self.flow_denoise_mode not in {"none", "proj", "correction"}:
            raise ValueError("Unsupported flow_denoise_mode; expected one of 'none','proj','correction'")
        self.flow_guidance_strength = float(flow_guidance_strength)
        self.flow_guidance_mode = str(flow_guidance_mode)
        self._models: Dict[int, FlowMLP] = {}

    def _get_model(self, layer: int, device: torch.device) -> FlowMLP:
        if layer not in self._models:
            payload = self.flow_payload[layer]
            flow = FlowMLP(
                dim=int(payload["dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                n_layers=int(payload["n_layers"]),
            )
            flow.load_state_dict(payload["state_dict"])
            self._models[layer] = flow.eval()
        return self._models[layer].to(device)

    def _get_flow_basis(
        self,
        payload: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        basis = payload.get("flow_basis")
        if not isinstance(basis, torch.Tensor):
            return None, None, None

        basis = basis.to(device=device, dtype=dtype)
        mean = payload.get("flow_basis_mean")
        if isinstance(mean, torch.Tensor):
            mean = mean.to(device=device, dtype=dtype)
        else:
            mean = None

        basis_inv = payload.get("flow_basis_inv")
        if isinstance(basis_inv, torch.Tensor):
            basis_inv = basis_inv.to(device=device, dtype=dtype)
        else:
            basis_inv = torch.linalg.pinv(basis)

        return basis, mean, basis_inv

    def _project_to_flow_space(
        self,
        acts: torch.Tensor,
        basis: Optional[torch.Tensor],
        mean: Optional[torch.Tensor],
        train_space: str,
    ) -> torch.Tensor:
        if basis is not None and train_space != "full":
            return project_to_basis(acts, basis, mean)
        return acts

    # `_project_flow_delta` was inlined at its single call site to reduce
    # indirection and keep the inference path straightforward.

    def _restore_from_flow_space(
        self,
        coords: torch.Tensor,
        basis: Optional[torch.Tensor],
        mean: Optional[torch.Tensor],
        basis_inv: Optional[torch.Tensor],
        train_space: str,
    ) -> torch.Tensor:
        if basis is not None and train_space != "full":
            if basis_inv is not None:
                restored = coords @ basis_inv.T
                if mean is not None:
                    restored = restored + mean
                return restored
            return unproject_from_basis(coords, basis, mean)
        return coords

    def _project_onto_flow_basis(
        self,
        acts: torch.Tensor,
        basis: Optional[torch.Tensor],
        mean: Optional[torch.Tensor],
        basis_inv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if basis is None:
            return acts
        coords = project_to_basis(acts, basis, mean)
        return self._restore_from_flow_space(coords, basis, mean, basis_inv, "reduced")

    def _flow_nullspace_component(
        self,
        acts: torch.Tensor,
        basis: Optional[torch.Tensor],
        mean: Optional[torch.Tensor],
        basis_inv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if basis is None:
            return torch.zeros_like(acts)
        projected = self._project_onto_flow_basis(acts, basis, mean, basis_inv)
        return acts - projected

    def _cpu_payload_to_device(self, payload: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
        """Move any CPU-side tensors in a nested payload to `device`.

        Kept as a FlowSteerModel method only (per design decision).
        """
        out: Dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, dict):
                out[k] = self._cpu_payload_to_device(v, device)
            else:
                out[k] = v
        return out

    def _flow_target(self, layer: int, acts: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
        payload = self._cpu_payload_to_device(self.flow_payload[layer], acts.device)
        model = self._get_model(layer, acts.device).to(dtype=torch.float32)
        x = acts.to(torch.float32)

        train_space = str(payload.get("flow_train_space", "full")).strip().lower()
        if train_space not in {"full", "pca_diff", "pca_stack", "lda"}:
            raise ValueError(f"Unsupported flow_train_space '{train_space}' in payload")
        basis, mean, basis_inv = self._get_flow_basis(payload, x.device, x.dtype)

        guidance = None
        if layer in self.steering_vector:
            gv = self.steering_vector[layer]
            if isinstance(gv, torch.Tensor):
                guidance = gv.to(device=x.device, dtype=x.dtype)

        return flow_steer_activations(
            x=x,
            flow_model=model,
            source_stats=payload["source_stats"],
            target_stats=payload["target_stats"],
            basis=basis,
            basis_mean=mean,
            basis_inv=basis_inv,
            train_space=train_space,
            steps=self.flow_steps,
            denoise_mode=self.flow_denoise_mode,
            coeff=coeff,
            guidance=guidance,
            guidance_strength=self.flow_guidance_strength,
            guidance_mode=self.flow_guidance_mode,
        )

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
        acts = get_resid_acts(resid, position)
        # _flow_target returns the final activations.
        updated = self._flow_target(layer, acts, coeff=float(coeff))
        return set_resid_acts(resid, position, updated.to(acts.dtype))


class LoReFTSteerModel(BaseSteerModel):
    """Low-rank rotate-and-reconstruct steering model (supports standard ReFT and preference-trained RePS)."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        rotate_basis: Dict[int, torch.Tensor],
        learned_weight: Dict[int, torch.Tensor],
        learned_bias: Optional[Dict[int, torch.Tensor]] = None,
        add_bias: bool = True,
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
        substraction_type: str = "zero",
        **kwargs,
    ):
        super().__init__(model=model, layer=layer, steering_vector=steering_vector, hook_point=hook_point, position=position, **kwargs)
        self.rotate_basis = {int(k): v.detach().clone() for k, v in rotate_basis.items()}
        self.learned_weight = {int(k): v.detach().clone() for k, v in learned_weight.items()}
        self.learned_bias = {
            int(k): v.detach().clone()
            for k, v in (learned_bias or {}).items()
            if v is not None
        }
        self.add_bias = bool(add_bias)
        self.substraction_type = substraction_type

    def _get_layer_payload(self, layer: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis = self.rotate_basis[layer].to(device=device, dtype=dtype)
        weight = self.learned_weight[layer].to(device=device, dtype=dtype)
        bias = self.learned_bias.get(layer)
        if bias is None:
            if self.add_bias:
                raise ValueError(
                    f"Missing learned_bias for layer {layer} while add_bias=True. "
                    "Check the extractor payload and metadata serialization."
                )
            bias = torch.zeros(weight.shape[0], device=device, dtype=dtype)
        else:
            bias = bias.to(device=device, dtype=dtype)
        return basis, weight, bias

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
        acts = get_resid_acts(resid, position)
        basis, weight, bias = self._get_layer_payload(layer, acts.device, acts.dtype)

        factor = float(coeff)

        if factor < 0 and self.substraction_type == "zero":
            # Null out any projection along the low-rank subspace
            updated = acts - (acts @ basis.T) @ basis
        else:
            rotated_base = acts @ basis.T
            source_output = acts @ weight.T
            if self.add_bias:
                source_output = source_output + bias

            delta = source_output - rotated_base
            updated = acts + factor * delta @ basis

        return set_resid_acts(resid, position, updated.to(acts.dtype))


# Replicated odesteer classes to remove dependency on Code/
from typing import Literal as _Literal
import numpy as _np
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.svm import LinearSVC as _LinearSVC

class RFF(torch.nn.Module):
    def __init__(
        self, 
        n_components: Optional[int] = None, 
        sigma: float | str = 'median',
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_components = n_components
        self.sigma = sigma
        self.random_state = random_state
        
    def fit(self, X: torch.Tensor):
        d, device, dtype = X.shape[1], X.device, X.dtype
        if self.n_components is None:
            self.n_components = d
        if self.random_state is not None:
            generator = torch.Generator(device = device).manual_seed(self.random_state)
        else:
            generator = None
        self.register_buffer(
            'W', 
            torch.randn(self.n_components, d, device = device, dtype = dtype, generator = generator) / self.get_sigma(X)
        )
        self.register_buffer(
            'b', 
            torch.rand(self.n_components, device = device, dtype = dtype, generator = generator) * 2 * _np.pi
        )
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (2 / self.n_components)**0.5 * torch.cos(X @ self.W.T + self.b)
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self(X)
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)
    
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        XW = X @ self.W.T + self.b
        term = -(2 / self.n_components)**0.5 * torch.sin(XW)
        return self.W * term.unsqueeze(-1)
    
    def jvp(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        XW = torch.matmul(X, self.W.T, out=None)
        torch.add(XW, self.b, out=XW)
        torch.sin(XW, out=XW)
        XW.mul_(-(2 / self.n_components)**0.5)
        
        if v.ndim == 1:
            Wv = torch.matmul(self.W, v, out=None)
            return XW * Wv
        else:
            assert X.shape[0] == v.shape[0]
            Wv = torch.einsum('nd, bd -> bn', self.W, v)
            return XW * Wv
        
    def vjp(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        XW = torch.matmul(X, self.W.T, out=None)
        torch.add(XW, self.b, out=XW)
        torch.sin(XW, out=XW)
        XW.mul_(-(2 / self.n_components)**0.5)
        
        if v.ndim == 1:
            v_scaled = XW * v  # [batch, n_components]
            return torch.einsum('bn, nd -> bd', v_scaled, self.W)
        else:
            v_scaled = XW * v
            return torch.einsum('nd, bn -> bd', self.W, v_scaled)
        
    def laplacian(self, X: torch.Tensor) -> torch.Tensor:
        w_norm = torch.norm(self.W, dim = 1)**2
        term = -(2 / self.n_components)**0.5 * torch.cos(X @ self.W.T + self.b)
        return term * w_norm.unsqueeze(0)
    
    def get_sigma(self, X: torch.Tensor) -> float:
        if self.sigma == 'median':
            n_samples = min(1000, X.shape[0])
            X_sampled = X[torch.randperm(X.shape[0])[:n_samples]]
            cdist = torch.tril(torch.cdist(X_sampled, X_sampled))
            median = torch.median(cdist[cdist != 0].flatten())
            return median / 2**0.5
        elif self.sigma == 'scale':
            return (X.shape[1] / 2)**0.5 * X.std()
        else:
            return self.sigma


class PolyCntSketch(torch.nn.Module):
    def __init__(
        self,
        degree: int = 2,
        n_components: int = 100,
        gamma: float = 1.0,
        coef0: float = 0.0,
    ):
        super().__init__()
        assert degree >= 1, "degree must be >= 1"
        assert n_components >= 1, "n_components must be >= 1"
        self.degree = int(degree)
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)

    @staticmethod
    def _ensure_fitted(buf):
        if buf is None:
            raise RuntimeError("Call .fit(X) first to initialize sketch hashes.")

    def _ext_feature_count(self) -> int:
        nf = int(self.n_features_)
        return nf + (1 if self.coef0 != 0 else 0)

    def extra_repr(self) -> str:
        return (f"degree={self.degree}, n_components={self.n_components}, "
                f"gamma={self.gamma}, coef0={self.coef0}, "
                f"n_features_={self.n_features_})")

    def fit(self, X: torch.Tensor):        
        self.n_features_ = X.shape[0] if X.dim() == 1 else X.shape[1]
        n_features_ext = self._ext_feature_count()

        indexHash = torch.randint(
            low = 0, high = self.n_components,
            size = (self.degree, n_features_ext),
            dtype = torch.long, device = X.device,
        )
        bitHash = (torch.randint(
            low = 0, high = 2,
            size = (self.degree, n_features_ext),
            dtype = torch.int8, device = X.device,
        ) * 2 - 1)
        
        self.register_buffer("indexHash_", indexHash)
        self.register_buffer("bitHash_", bitHash)
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted(self.indexHash_)

        if X.dim() == 1:
            y = self._forward_batch(X.unsqueeze(0)).squeeze(0)
        elif X.dim() == 2:
            y = self._forward_batch(X)
        else:
            raise ValueError("X must be 1D or 2D tensor.")
        return y

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self(X)

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)

    def _forward_batch(self, X: torch.Tensor) -> torch.Tensor:
        B, F = X.shape
        dtype = X.dtype
        D, n = self.degree, self.n_components

        X_gamma = X * (self.gamma ** 0.5)

        if self.coef0 != 0:
            bias = X_gamma.new_full((B, 1), (self.coef0 ** 0.5))
            X_ext = torch.cat([X_gamma, bias], dim=1)
        else:
            X_ext = X_gamma
        Fext = X_ext.shape[1]

        idx = self.indexHash_[:, :Fext]
        sgn = self.bitHash_[:, :Fext].to(dtype = dtype)

        sketches = []
        for d in range(D):
            vals_d = X_ext * sgn[d].unsqueeze(0)
            sketch_d = X_ext.new_zeros((B, n))
            sketch_d.scatter_add_(1, idx[d].unsqueeze(0).expand(B, Fext), vals_d)
            sketches.append(sketch_d)
        sketches = torch.stack(sketches, dim=0)

        F_r = torch.fft.rfft(sketches, dim = -1)
        prod = torch.prod(F_r, dim = 0)
        y = torch.fft.irfft(prod, n = n, dim = -1)
        return y

    @torch.no_grad()
    def grad(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 1:
            return self._grad_single(X)
        else:
            if hasattr(torch, "vmap"):
                return torch.vmap(self._grad_single)(X)
            else:
                outs = [self._grad_single(x) for x in X]
                return torch.stack(outs, dim=0)

    def _grad_single(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted(self.indexHash_)
        device, dtype = x.device, x.dtype
        n, D = self.n_components, self.degree

        x_gamma = x * (self.gamma ** 0.5)
        if self.coef0 != 0:
            x_ext = torch.cat([x_gamma, x.new_tensor([(self.coef0 ** 0.5)])], dim=0)
        else:
            x_ext = x_gamma
        Fext = x_ext.shape[0]

        sketches = []
        for d in range(D):
            idxs = self.indexHash_[d, :Fext]
            bits = self.bitHash_[d, :Fext].to(dtype=dtype)
            vals = bits * x_ext
            sd = x.new_zeros(n)
            sd.index_add_(0, idxs, vals)
            sketches.append(sd)
        S = torch.stack(sketches, dim=0)

        Fr = torch.fft.rfft(S, dim=1)
        n_r = Fr.shape[1]

        P = torch.empty_like(Fr)
        P[0] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        if D > 1:
            P[1:] = torch.cumprod(Fr[:-1], dim=0)

        U = torch.empty_like(Fr)
        U[-1] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        if D > 1:
            tmp = torch.cumprod(torch.flip(Fr[1:], dims=[0]), dim=0)
            U[:-1] = torch.flip(tmp, dims=[0])

        other_prod = P * U
        q = torch.fft.irfft(other_prod, n=n, dim=1)

        Forig = self.n_features_
        J_cols = []
        sqrt_gamma = self.gamma ** 0.5
        base = torch.arange(n, device=device)

        for j in range(Forig):
            idxs_d = self.indexHash_[:D, j]
            bits_d = self.bitHash_[:D, j].to(dtype=dtype)
            col = x.new_zeros(n)
            for d in range(D):
                shift = int(idxs_d[d])
                col.add_(bits_d[d] * q[d].take((base - shift) % n))
            J_cols.append(sqrt_gamma * col)

        J = torch.stack(J_cols, dim=1)
        return J

    @torch.no_grad()
    def vjp(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self._ensure_fitted(self.indexHash_)
        if v.dim() != 1 or v.numel() != self.n_components:
            raise ValueError(
                f"v must be 1-D of length n_components={self.n_components}, "
                f"got shape {tuple(v.shape)}"
            )
        if X.dim() == 1:
            return self._vjp_single(X, v)
        elif X.dim() == 2:
            return self._vjp_batch(X, v)
        else:
            raise ValueError("X must be 1D or 2D tensor.")

    def _vjp_single(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        n, D = self.n_components, self.degree

        x_gamma = x * (self.gamma ** 0.5)
        if self.coef0 != 0:
            x_ext = torch.cat([x_gamma, x.new_tensor([(self.coef0 ** 0.5)])], dim=0)
        else:
            x_ext = x_gamma
        Fext = x_ext.shape[0]

        S_rows = []
        for d in range(D):
            idxs = self.indexHash_[d, :Fext]
            bits = self.bitHash_[d, :Fext].to(dtype=dtype)
            vals = bits * x_ext
            sd = x.new_zeros(n)
            sd.index_add_(0, idxs, vals)
            S_rows.append(sd)
        S = torch.stack(S_rows, dim=0)

        Fr = torch.fft.rfft(S, dim=1)
        n_r = Fr.shape[1]
        P = torch.empty_like(Fr)
        U = torch.empty_like(Fr)
        P[0] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        U[-1] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        if D > 1:
            P[1:] = torch.cumprod(Fr[:-1], dim=0)
            tmp = torch.cumprod(torch.flip(Fr[1:], dims=[0]), dim=0)
            U[:-1] = torch.flip(tmp, dims=[0])
        other_prod = P * U
        q = torch.fft.irfft(other_prod, n=n, dim=1).to(dtype)

        Vr = torch.fft.rfft(v.to(dtype), dim=0)
        Q = torch.fft.rfft(q, dim=1)
        C = torch.fft.irfft(torch.conj(Q) * Vr.unsqueeze(0), n=n, dim=1).real

        Forig = self.n_features_
        idx = self.indexHash_[:D, :Forig]
        bits = self.bitHash_[:D, :Forig].to(dtype=dtype)
        gathered = C.gather(1, idx)
        out = (gathered * bits).sum(dim=0)
        return (self.gamma ** 0.5) * out

    def _vjp_batch(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, F = X.shape
        device, dtype = X.device, X.dtype
        D, n = self.degree, self.n_components

        X_gamma = X * (self.gamma ** 0.5)
        if self.coef0 != 0:
            bias = X_gamma.new_full((B, 1), (self.coef0 ** 0.5))
            X_ext = torch.cat([X_gamma, bias], dim=1)
        else:
            X_ext = X_gamma
        Fext = X_ext.shape[1]

        idx = self.indexHash_[:, :Fext]
        bits = self.bitHash_[:, :Fext].to(dtype=dtype)

        sketches = []
        for d in range(D):
            vals_d = X_ext * bits[d].unsqueeze(0)
            sk = X_ext.new_zeros((B, n))
            sk.scatter_add_(1, idx[d].unsqueeze(0).expand(B, Fext), vals_d)
            sketches.append(sk)
        S = torch.stack(sketches, dim=1)

        Fr = torch.fft.rfft(S, dim=2)
        n_r = Fr.shape[2]
        P = torch.empty_like(Fr)
        U = torch.empty_like(Fr)
        P[:, 0] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        U[:, -1] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        if D > 1:
            P[:, 1:] = torch.cumprod(Fr[:, :-1], dim=1)
            tmp = torch.cumprod(torch.flip(Fr[:, 1:], dims=[1]), dim=1)
            U[:, :-1] = torch.flip(tmp, dims=[1])
        other = P * U
        q = torch.fft.irfft(other, n=n, dim=2).to(dtype)

        Vr = torch.fft.rfft(v.to(dtype), dim=0)
        Q = torch.fft.rfft(q, dim=2)
        C = torch.fft.irfft(torch.conj(Q) * Vr.view(1, 1, -1), n=n, dim=2).real

        Forig = self.n_features_
        idxF = idx[:, :Forig]
        bitsF = bits[:, :Forig]
        gathered = C.gather(2, idxF.unsqueeze(0).expand(B, -1, -1))
        out = (gathered * bitsF.unsqueeze(0)).sum(dim=1)
        return out


class NormedPolyCntSketch(PolyCntSketch):
    def __init__(
        self,
        degree: int = 2,
        n_components: int = 100,
        gamma: float = 1.0,
        coef0: float = 0.0,
        eps: float = 1e-12,
    ):
        super().__init__(degree=degree, n_components=n_components, gamma=gamma, coef0=coef0)
        self.eps = float(eps)

    def _normalize(self, X: torch.Tensor):
        r = X.norm(p = 2, dim = -1, keepdim=True) + self.eps
        return X / r, r

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self._normalize(X)
        return super().forward(x_hat)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self(X)

    @torch.no_grad()
    def grad(self, X: torch.Tensor) -> torch.Tensor:
        x_hat, r = self._normalize(X)
        J_f = super().grad(x_hat)

        if X.dim() == 1:
            proj = J_f @ x_hat
            J = (J_f - proj.unsqueeze(1) * x_hat.unsqueeze(0)) / r
            return J
        else:
            proj = torch.einsum('bnf,bf->bn', J_f, x_hat)
            J = (J_f - proj.unsqueeze(-1) * x_hat.unsqueeze(1)) / r.unsqueeze(1)
            return J

    @torch.no_grad()
    def vjp(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if v.dim() != 1 or v.numel() != self.n_components:
            raise ValueError(
                f"v must be 1-D with length n_components={self.n_components}, "
                f"got {tuple(v.shape)}"
            )

        x_hat, r = self._normalize(X)
        u = super().vjp(x_hat, v)

        if X.dim() == 1:
            dot = (x_hat * u).sum()
            return (u - x_hat * dot) / r
        else:
            dot = (x_hat * u).sum(dim=1, keepdim=True)
            return (u - x_hat * dot) / r


class KernelClassifier(torch.nn.Module):
    def __init__(self, lin_clf_type: _Literal['lr', 'svm'] = 'lr'):
        super().__init__()
        self.lin_clf_type = lin_clf_type
        self.kernel: torch.nn.Module = None
        self.fitted: bool = False
        
    def fit(self, pos_X: torch.Tensor, neg_X_or_labels: torch.Tensor) -> 'KernelClassifier':
        if neg_X_or_labels.ndim == 1:
            return self._fit_with_labels(pos_X, neg_X_or_labels)
        else:
            assert neg_X_or_labels.shape[1] == pos_X.shape[1], \
                f'neg_X_or_labels.shape[1] = {neg_X_or_labels.shape[1]} != pos_X.shape[1] = {pos_X.shape[1]}'
            return self._fit_with_two_sets(pos_X, neg_X_or_labels)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert self.fitted, 'KernelClassifier is not fitted'
        return self.predict_proba(X)
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.fitted, 'KernelClassifier is not fitted'
        self.kernel.to(X.device)
        return self.predict_proba(X) > 0.5
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        assert self.fitted, 'KernelClassifier is not fitted'
        self.kernel.to(X.device)
        Z = self.kernel.transform(X)
        return (Z @ self.coef + self.intercept).sigmoid()
    
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        assert self.fitted, 'KernelClassifier is not fitted'
        return (self.predict(X) == y).float().mean().item()
    
    def density_ratio(self, X: torch.Tensor) -> torch.Tensor:
        assert self.fitted, 'RFFClassifier is not fitted'
        self.kernel.to(X.device)
        Z = self.rff.transform(X)
        return (Z @ self.coef + self.intercept).exp() * self.dre_coeff
    
    def log_dre(self, X: torch.Tensor) -> torch.Tensor:
        assert self.fitted, 'RFFClassifier is not fitted'
        Z = self.kernel.transform(X)
        return Z @ self.coef + self.intercept + _np.log(self.dre_coeff)
    
    def grad(self, X: torch.Tensor) -> torch.Tensor:
        assert self.fitted, 'KernelClassifier is not fitted'
        self.kernel.to(X.device)
        return self.kernel.vjp(X, self.coef) * self.dre_coeff
    
    def _fit_with_two_sets(self, pos_X: torch.Tensor, neg_X: torch.Tensor) -> 'KernelClassifier':
        pi = len(pos_X) / (len(pos_X) + len(neg_X))
        self.register_buffer(
            'dre_coeff', 
            torch.as_tensor((1 - pi) / pi, dtype = pos_X.dtype, device = pos_X.device),
        )
        X = torch.cat([pos_X, neg_X], dim = 0)
        y = torch.cat([torch.ones(pos_X.shape[0]), torch.zeros(neg_X.shape[0])])
        return self._fit_with_labels(X, y)
    
    def _fit_with_labels(self, X: torch.Tensor, y: torch.Tensor) -> 'KernelClassifier':
        pi = len(X[y == 1]) / len(X)
        self.register_buffer(
            'dre_coeff', 
            torch.as_tensor((1 - pi) / pi, dtype = X.dtype, device = X.device),
        )
        Z = self.kernel.fit_transform(X)
        coef, intercept = self._fit_linear_clf(Z, y)
        self.register_buffer(
            'coef', 
            torch.as_tensor(coef, dtype = X.dtype, device = X.device),
        )
        self.register_buffer(
            'intercept', 
            torch.as_tensor(intercept, dtype = X.dtype, device = X.device),
        )
        return self
    
    def _fit_linear_clf(self, X: torch.Tensor, y: torch.Tensor):
        if self.lin_clf_type == 'lr':
            clf = _LogisticRegression(max_iter = 1000)
        elif self.lin_clf_type == 'svm':
            clf = _LinearSVC(max_iter = 1000)
        else:
            raise ValueError(f'Invalid linear classifier type: {self.lin_clf_type}')
        clf.fit(X.cpu().numpy(), y.cpu().numpy())
        coef = torch.as_tensor(clf.coef_.ravel(), dtype = X.dtype, device = X.device)
        intercept = torch.as_tensor(clf.intercept_.ravel(), dtype = X.dtype, device = X.device)
        self.fitted = True
        return coef, intercept    


class RFFClassifier(KernelClassifier):
    def __init__(
        self,
        n_components: int,
        sigma: float | _Literal['median', 'scale'] = 'median',
        lin_clf_type: _Literal['lr', 'svm'] = 'lr',
    ):
        super().__init__(lin_clf_type)
        self.kernel = RFF(n_components, sigma)


class PolyClassifier(KernelClassifier):
    def __init__(
        self,
        degree: int = 2,
        n_components: int = 100,
        gamma: float = 1.0,
        coef0: float = 0.1,
        lin_clf_type: str = 'lr',
    ):
        super().__init__(lin_clf_type)
        self.kernel = PolyCntSketch(degree, n_components, gamma, coef0)    
        

class NormedPolyClassifier(KernelClassifier):
    def __init__(
        self,
        degree: int = 2,
        n_components: int = 100,
        gamma: float = 1.0,
        coef0: float = 0.1,
        lin_clf_type: str = 'lr',
    ):
        super().__init__(lin_clf_type)
        self.kernel = NormedPolyCntSketch(degree, n_components, gamma, coef0)


class ODESteerModel(BaseSteerModel):
    """Steer model that reconstructs odesteer classifiers from extractor metadata
    and applies continuous ODE integration (or single-step Euler) to activations.
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        ode_payload: Optional[Dict[int, Dict[str, Any]]] = None,
        solver: Optional[str] = None,
        steps: Optional[int] = None,
        hook_point: Union[str, List[str]] = "post",
        position: Union[str, int] = "last",
        one_step: bool = False,
        **kwargs,
    ):
        super().__init__(model, layer, steering_vector, hook_point=hook_point, position=position, **kwargs)
        self.ode_payload = {int(k): v for k, v in (ode_payload or {}).items()}
        self.solver = solver
        self.steps = steps
        self._clfs: Dict[int, Any] = {}
        self.one_step = bool(one_step)

    def _reconstruct_clf(self, layer: int):
        if layer in self._clfs:
            return self._clfs[layer]

        payload = self.ode_payload.get(layer)
        if payload is None:
            raise ValueError(f"Missing ode_payload for layer {layer}")

        ctype = payload.get("classifier_type", "normed_poly")
        kp = payload.get("kernel_params", {})
        if ctype in {"normed_poly", "poly", "poly_norm"}:
            clf = NormedPolyClassifier(
                degree=kp.get("degree", 2),
                n_components=kp.get("n_components", 100),
                gamma=kp.get("gamma", 1.0),
                coef0=kp.get("coef0", 0.1),
                lin_clf_type=kp.get("lin_clf_type", "lr"),
            )
        else:
            clf = RFFClassifier(
                n_components=kp.get("n_components", 100),
                sigma=kp.get("sigma", "median"),
                lin_clf_type=kp.get("lin_clf_type", "lr"),
            )

        # Load state dict
        state = payload.get("state_dict", {})
        
        # Pre-register any buffers that are in the state_dict but not yet on the module/submodules
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                parts = k.split(".")
                submod = clf
                for part in parts[:-1]:
                    if not hasattr(submod, part):
                        setattr(submod, part, torch.nn.Module())
                    submod = getattr(submod, part)
                buf_name = parts[-1]
                if not hasattr(submod, buf_name):
                    submod.register_buffer(buf_name, v.clone())
        
        # Ensure dre_coeff exists
        if not hasattr(clf, "dre_coeff") and "dre_coeff" not in state:
            clf.register_buffer("dre_coeff", torch.tensor(1.0))

        try:
            clf.load_state_dict(state)
        except Exception:
            # Fallback: if buffers are plain tensors, set manually with correct submodule resolution
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    parts = k.split(".")
                    submod = clf
                    for part in parts[:-1]:
                        if not hasattr(submod, part):
                            setattr(submod, part, torch.nn.Module())
                        submod = getattr(submod, part)
                    buf_name = parts[-1]
                    submod.register_buffer(buf_name, v.to(clf.kernel.weight.device if hasattr(clf, 'kernel') and hasattr(clf.kernel, 'weight') else v.device))

        # Set other required attributes
        clf.fitted = True
        if hasattr(clf, "kernel") and hasattr(clf.kernel, "indexHash_"):
            n_features_ext = clf.kernel.indexHash_.shape[1]
            clf.kernel.n_features_ = n_features_ext - (1 if getattr(clf.kernel, "coef0", 0.0) != 0 else 0)

        self._clfs[layer] = clf.eval()
        return self._clfs[layer]

    def vector_field(self, X: torch.Tensor, layer: int) -> torch.Tensor:
        clf = self._reconstruct_clf(layer)
        clf.to(X.device)
        raw_grad = clf.grad(X)
        norm = raw_grad.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        return raw_grad / norm

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
        acts = get_resid_acts(resid, position)

        solver = self.solver or self.ode_payload.get(layer, {}).get("solver", "euler")
        steps = int(self.steps or self.ode_payload.get(layer, {}).get("steps", 10))

        if float(coeff) == 0.0:
            return resid

        if self.one_step:
            vf = self.vector_field(acts.to(torch.float32), layer)
            updated = acts + float(coeff) * vf.to(acts.dtype)
        return set_resid_acts(resid, position, updated)


class IDSSteerModel(BaseSteerModel):
    """IDS Steer Model.

    Dynamically calculates the maximum steering coefficient at each token position
    to ensure the steered activations remain within the target distribution ellipsoid.
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        layer_stats: Dict[int, Dict[str, torch.Tensor]],
        ids_f1_threshold: float = 0.70,
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
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
        self.layer_stats = {int(k): v for k, v in layer_stats.items()}
        self.ids_f1_threshold = float(ids_f1_threshold)

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

        if layer not in self.layer_stats:
            return resid

        stats = self.layer_stats[layer]

        f1_score = stats.get("f1_score", 1.0)
        if f1_score < self.ids_f1_threshold:
            return resid

        acts = get_resid_acts(resid, position)
        orig_shape = acts.shape
        orig_dtype = acts.dtype
        device = acts.device

        h = acts.reshape(-1, orig_shape[-1]).to(torch.float32)

        v = steering_vector.to(device=device, dtype=torch.float32)
        P = stats["pca_components"].to(device=device, dtype=torch.float32)
        mu = stats["pca_mean"].to(device=device, dtype=torch.float32)
        mu_tgt_pca = stats["mu_tgt_pca"].to(device=device, dtype=torch.float32)
        L_inv = stats["L_inv"].to(device=device, dtype=torch.float32)
        epsilon_sq = float(stats["epsilon_sq"])

        diff_pca = (h - mu.unsqueeze(0)) @ P
        h_tilde = diff_pca @ L_inv.T
        mu_tilde = L_inv @ mu_tgt_pca
        h_minus_mu = h_tilde - mu_tilde.unsqueeze(0)

        v_pca = v @ P
        v_tilde = v_pca @ L_inv.T

        a = torch.sum(v_tilde ** 2)
        b = 2.0 * (h_minus_mu @ v_tilde)
        c = torch.sum(h_minus_mu ** 2, dim=-1) - epsilon_sq

        disc = b ** 2 - 4.0 * a * c

        alpha = torch.where(disc >= 0, (-b + torch.sqrt(disc)) / (2.0 * a), -b / (2.0 * a))
        alpha = torch.clamp(alpha, min=0.0)

        updated = h + (float(coeff) * alpha).unsqueeze(-1) * v.unsqueeze(0)
        updated = updated.reshape(orig_shape).to(orig_dtype)

        return set_resid_acts(resid, position, updated)

        # continuous integration via torchdiffeq
        from torchdiffeq import odeint
        T = float(coeff)
        device = acts.device

        def f(t, state):
            return self.vector_field(state, layer)

        times = torch.tensor([0.0, T], device=device)
        out = odeint(func=lambda t, s: f(t, s), y0=acts.to(torch.float32), t=times, method=solver, options={"step_size": T / steps})
        updated = out[1].to(acts.dtype)
        return set_resid_acts(resid, position, updated)


class _FlowCtx:
    """A context container to hold active state variables during FLAS generation."""
    def __init__(
        self,
        flowtimes: torch.Tensor,
        concept_hidden: torch.Tensor,
        concept_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        position_ids: torch.Tensor,
        sa_caches: List[Any],
        past_len: int,
    ):
        self.flowtimes = flowtimes
        self.concept_hidden = concept_hidden
        self.concept_mask = concept_mask
        self.padding_mask = padding_mask
        self.position_ids = position_ids
        self.sa_caches = sa_caches
        self.past_len = past_len


class FLASSteerModel(BaseSteerModel):
    """FLAS generation with HF decoder-layer hooks only."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        model_name: str,
        flas_flow_fn: Optional[FlowFunction] = None,
        flas_flow_fn_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        flas_concept_enc: Optional[ConceptEncoder] = None,
        flas_concept_enc_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        flas_model_id: Optional[str] = None,
        flas_num_blocks: int = 1,
        flas_concept_encoder_layers: int = 2,
        flas_time_conditioned: bool = True,
        flas_disable_cross_attn: bool = False,
        flas_disable_self_attn: bool = False,
        flas_disable_mlp: bool = False,
        flas_concept_text: Optional[str] = None,
        flas_concept_max_len: int = 64,
        flas_n_steps: int = 2,
        flas_max_prompt_len: int = 512,
        hook_point: Union[str, List[str]] = "pre",
        **kwargs,
    ):
        super().__init__(
            model=model,
            layer=layer,
            steering_vector=steering_vector,
            hook_point=hook_point,
            **kwargs,
        )
        if len(layer) != 1:
            raise ValueError("FLASSteerModel supports one flow-hook layer per instance.")
        if flas_model_id is None:
            flas_model_id = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if flas_flow_fn is None:
            if flas_flow_fn_state_dict is None:
                raise ValueError(
                    "FLASSteerModel requires flas_flow_fn or flas_flow_fn_state_dict from FLAS metadata."
                )
            flow_fn = FlowFunction(
                AutoConfig.from_pretrained(flas_model_id),
                num_blocks=int(flas_num_blocks),
                time_conditioned=bool(flas_time_conditioned),
                layer_idx=int(layer[0]),
                disable_cross_attn=bool(flas_disable_cross_attn),
                disable_self_attn=bool(flas_disable_self_attn),
                disable_mlp=bool(flas_disable_mlp),
            )
            flow_fn.load_state_dict(flas_flow_fn_state_dict, strict=True)
            flas_flow_fn = flow_fn

        if flas_concept_enc is None:
            concept_enc = ConceptEncoder(flas_model_id, num_layers=int(flas_concept_encoder_layers))
            if flas_concept_enc_state_dict is not None:
                concept_enc.load_state_dict(flas_concept_enc_state_dict, strict=True)
            flas_concept_enc = concept_enc
        target_device = self._model_device()
        self.flow_fn = flas_flow_fn.to(target_device).eval()
        self.concept_enc = flas_concept_enc.to(target_device).eval()
        self.flas_concept_text = flas_concept_text
        self.flas_concept_max_len = int(flas_concept_max_len)
        self.flas_n_steps = int(flas_n_steps)
        self.flas_max_prompt_len = int(flas_max_prompt_len)
        self._flow_dtype = next(self.flow_fn.parameters()).dtype

        self._hook_handle = None
        self._active = False
        self._ctx = None

    def hook_fn(self, resid, position, coeff, steering_vector, hook, **kwargs):
        """Retain the shared hook interface, but FLAS generation uses the HF path below."""
        return resid

    def _model_device(self):
        if hasattr(self.model, "parameters"):
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device(getattr(getattr(self.model, "cfg", None), "device", self.device))

    def _hook_fn(self, module, input, output):
        if not self._active or self._ctx is None:
            return output
        is_tuple = isinstance(output, tuple)
        h_orig = output[0] if is_tuple else output
        bsz = h_orig.size(0)
        is_decode = h_orig.shape[1] == 1

        if is_decode and self.steer_once:
            return output

        ctx = self._ctx

        # Decide if we steer all sequence tokens or just the last one
        use_all = is_decode or (self.position == "all")

        if use_all:
            h_target = h_orig.to(self._flow_dtype)
            pos_ids = ctx.position_ids[:bsz]
            pad_mask = ctx.padding_mask[:bsz]
            past_len = ctx.past_len if is_decode else 0
        else:
            h_target = h_orig[:, -1:, :].to(self._flow_dtype)
            pos_ids = ctx.position_ids[:bsz, -1:]
            pad_mask = ctx.padding_mask[:bsz, -1:]
            past_len = 0

        dt = (ctx.flowtimes[:bsz] / self.flas_n_steps).to(self._flow_dtype)

        for k in range(self.flas_n_steps):
            t_k = dt * k
            flow_kwargs = {
                "t": t_k,
                "padding_mask": pad_mask,
                "use_cache": True,
                "past_len": past_len,
                "position_ids": pos_ids,
            }
            if is_decode:
                flow_kwargs["self_attn_caches"] = ctx.sa_caches[k]

            v, kv_caches = self.flow_fn(
                h_target,
                ctx.concept_hidden[:bsz],
                ctx.concept_mask[:bsz],
                **flow_kwargs
            )
            ctx.sa_caches[k] = kv_caches
            h_target = h_target + dt.unsqueeze(1).unsqueeze(2) * v

        h_steered = h_orig.clone()
        if is_decode or self.position != "all":
            h_steered[:, -1:, :] = h_target.to(h_orig.dtype)
        else:
            prompt_mask = pad_mask.to(device=h_orig.device, dtype=torch.bool).unsqueeze(-1)
            h_steered = torch.where(prompt_mask, h_target.to(h_orig.dtype), h_steered)
        return (h_steered,) + output[1:] if is_tuple else h_steered

    def _install_hook(self):
        if self._hook_handle is None:
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                layer_module = self.model.model.layers[self.layer[0]]
            elif hasattr(self.model, "blocks"):
                layer_module = self.model.blocks[self.layer[0]]
            else:
                raise AttributeError("Could not find the layers module in the model. Supported models: HF (self.model.model.layers) and TransformerLens (self.model.blocks)")
            self._hook_handle = layer_module.register_forward_hook(
                self._hook_fn
            )

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def encode_concept(self, text: str):
        concept_device = self._model_device()
        self.concept_enc = self.concept_enc.to(concept_device)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.flas_concept_max_len,
        ).to(concept_device)
        hidden = self.concept_enc(enc.input_ids, enc.attention_mask)
        return hidden.to(self._flow_dtype), enc.attention_mask.float()

    def _resolve_flowtime(self, coeff: Any) -> float:
        if coeff is None:
            return 1.0
        if isinstance(coeff, dict):
            return float(coeff.get(self.layer[0], next(iter(coeff.values()))))
        if isinstance(coeff, (list, tuple)):
            return float(coeff[0])
        return float(coeff)

    def _prepare_flow_context(self, concept_text: str, flowtime: float, attention_mask: torch.Tensor, past_len: int = 0):
        concept_hidden, concept_mask = self.encode_concept(concept_text)
        bsz = attention_mask.size(0)
        device = self._model_device()
        self._ctx = _FlowCtx(
            flowtimes=torch.full((bsz,), flowtime, device=device, dtype=torch.float32),
            concept_hidden=concept_hidden.expand(bsz, -1, -1).contiguous(),
            concept_mask=concept_mask.expand(bsz, -1).contiguous(),
            padding_mask=attention_mask.float(),
            position_ids=(attention_mask.cumsum(-1) - 1).clamp(min=0),
            sa_caches=[None] * self.flas_n_steps,
            past_len=past_len,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        apply_steer: bool = True,
        coeff: Optional[Dict[int, float]] = None,
        flas_concept_text: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else list(prompt)

        if not apply_steer:
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.flas_max_prompt_len,
                add_special_tokens=False,
            ).to(self._model_device())
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
            results = [
                self.tokenizer.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True)
                for i in range(out.shape[0])
            ]
            self.metadata = [{} for _ in results]
            return results if not is_single else results[0]

        concept_text = flas_concept_text or self.flas_concept_text
        if concept_text is None or not concept_text.strip():
            raise ValueError("FLAS generation requires flas_concept_text.")

        flowtime = self._resolve_flowtime(coeff)
        device = self._model_device()
        bsz = len(prompts)

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.flas_max_prompt_len,
            add_special_tokens=False,
        ).to(device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask
        prompt_len = input_ids.shape[1]

        self._prepare_flow_context(concept_text, flowtime, attention_mask, prompt_len)
        self._install_hook()
        self._active = True

        try:
            if self.steer_once:
                # With steer_once=True, we steer during the prefill phase only.
                # Since decoding tokens are not actively steered, we do not need the custom loop!
                # We can call HuggingFace's standard highly-optimized generate function directly.
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )
            else:
                from transformers.cache_utils import DynamicCache
                past_kv = DynamicCache()
                out = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=self._ctx.position_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                next_logits = out.logits[:, -1, :]

                generated = input_ids
                unfinished = torch.ones(bsz, dtype=torch.bool, device=device)

                for _ in range(max_new_tokens):
                    if do_sample and temperature > 0:
                        logits = next_logits
                        if top_k is not None and top_k > 0:
                            kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                            logits = logits.masked_fill(logits < kth, -float("inf"))
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = next_logits.argmax(dim=-1, keepdim=True)

                    next_token = next_token.masked_fill(
                        ~unfinished.unsqueeze(1), self.tokenizer.pad_token_id
                    )
                    generated = torch.cat([generated, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, unfinished.unsqueeze(1).long()], dim=1
                    )

                    eos_hit = next_token.squeeze(1) == self.tokenizer.eos_token_id
                    unfinished = unfinished & ~eos_hit
                    if not unfinished.any():
                        break

                    self._ctx.padding_mask = attention_mask.float()
                    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
                    self._ctx.position_ids = position_ids[:, -1:]

                    out = self.model(
                        next_token,
                        attention_mask=attention_mask,
                        position_ids=self._ctx.position_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = out.past_key_values
                    next_logits = out.logits[:, -1, :]
                    self._ctx.past_len += 1

            self._active = False

            results = []
            self.metadata = []
            for i in range(bsz):
                gen_ids = generated[i, prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append(text)
                self.metadata.append(
                    {
                        "prompt_idx": i,
                        "flas_flowtime": float(flowtime),
                        "flas_n_steps": self.flas_n_steps,
                        "flas_concept_text": concept_text,
                    }
                )

            if 'past_kv' in locals():
                del past_kv
            if 'out' in locals():
                del out
            if self._ctx is not None:
                self._ctx.sa_caches = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self._active = False
            self._remove_hook()
            self._ctx = None

        return results if not is_single else results[0]

    @torch.no_grad()
    def get_token_probs(
        self,
        prompt: str,
        tokens: List[str],
        coeff: Optional[Dict[int, float]] = None,
        flas_concept_text: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, float]:
        apply_steer = bool(kwargs.pop("apply_steer", True))
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(
            [formatted],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.flas_max_prompt_len,
            add_special_tokens=False,
        ).to(self._model_device())

        if not apply_steer:
            out = self.model(**enc, use_cache=False)
        else:
            concept_text = flas_concept_text or self.flas_concept_text
            if concept_text is None or not concept_text.strip():
                raise ValueError("FLAS token probabilities require flas_concept_text.")
            flowtime = self._resolve_flowtime(coeff)
            self._prepare_flow_context(concept_text, flowtime, enc.attention_mask, 0)
            try:
                self._install_hook()
                self._active = True
                out = self.model(
                    enc.input_ids,
                    attention_mask=enc.attention_mask,
                    position_ids=self._ctx.position_ids,
                    use_cache=False,
                )
            finally:
                self._active = False
                self._remove_hook()
                self._ctx = None

        probs = torch.softmax(out.logits[0, -1, :], dim=-1)
        result = {}
        total = 0.0
        for token in tokens:
            tid = self.tokenizer.encode(token, add_special_tokens=False)[-1]
            value = float(probs[tid].item())
            result[token] = value
            total += value
        for token in tokens:
            result[token] /= (total + 1e-9)
        self.metadata = [{"flas_token_prob_flowtime": self._resolve_flowtime(coeff)}]
        return result

    def get_output_metadata(self):
        return self.metadata


# =============================================================================
# FishBack Steer Model — Multi-Step Flow Integration
# =============================================================================

class FishBackSteerModel(BaseSteerModel):
    """Steering via multi-step Euler integration of trained velocity field.

    h_0 = h_current
    for t in [0, dt, 2dt, ...]:
        h += v_θ(h, t) * dt
    h' = h_0 + coeff * (h_n - h_0)
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        flow_state_dicts: Dict[int, Dict[str, Any]],
        flow_config: Dict[str, Any],
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
        **kwargs,
    ):
        super().__init__(
            model=model, layer=layer, steering_vector=steering_vector,
            hook_point=hook_point, position=position, **kwargs,
        )
        from Steering.flow_utils import FlowMLP

        d_model = next(iter(steering_vector.values())).shape[-1]
        self.n_steps = flow_config["n_steps"]
        self.flow_models: Dict[int, FlowMLP] = {}

        for lyr in layer:
            key = lyr if lyr in flow_state_dicts else str(lyr)
            flow = FlowMLP(d_model, flow_config["hidden_dim"], flow_config.get("n_layers", 2)).to(self.device)
            flow.load_state_dict(
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in flow_state_dicts[key].items()},
                strict=False,
            )
            flow.eval()
            self.flow_models[lyr] = flow

    def hook_fn(self, resid, coeff, steering_vector, position, hook, **kwargs):
        layer = int(hook.name.split(".")[1])
        if layer not in self.flow_models:
            return resid

        acts = get_resid_acts(resid, position)
        orig_shape = acts.shape
        h = acts.reshape(-1, orig_shape[-1]).to(torch.float32)

        with torch.no_grad():
            dt = 1.0 / self.n_steps
            for step in range(self.n_steps):
                t = torch.full((h.shape[0],), step * dt, device=self.device)
                h = h + self.flow_models[layer](h, t) * dt

            h = acts.reshape(-1, orig_shape[-1]).to(torch.float32) + \
                float(coeff) * (h - acts.reshape(-1, orig_shape[-1]).to(torch.float32))

        return set_resid_acts(resid, position, h.reshape(orig_shape).to(acts.dtype))



# =============================================================================
# GINN Steer Model — Encode-Steer-Decode with Path-Cost Mapping
# =============================================================================

class GINNSteerModel(BaseSteerModel):
    """GINN steering via encode-steer-decode.

    z = φ(h) → z' = z + coeff·v → h' = φ⁻¹(z')
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        inn_state_dicts: Dict[int, Dict[str, Any]],
        inn_config: Dict[str, Any],
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
        **kwargs,
    ):
        super().__init__(
            model=model, layer=layer, steering_vector=steering_vector,
            hook_point=hook_point, position=position, **kwargs,
        )
        from Steering.extractors.nonlinear import GeometricInvertibleNN

        d_model = next(iter(steering_vector.values())).shape[-1]
        n_coupling = inn_config["n_coupling"]
        hidden_dim = inn_config["hidden_dim"]

        self.inns: Dict[int, GeometricInvertibleNN] = {}
        for lyr in layer:
            key = lyr if lyr in inn_state_dicts else str(lyr)
            inn = GeometricInvertibleNN(d_model, n_coupling, hidden_dim).to(self.device)
            inn.load_state_dict(
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inn_state_dicts[key].items()},
                strict=False,
            )
            inn.eval()
            self.inns[lyr] = inn

    def hook_fn(self, resid, coeff, steering_vector, position, hook, **kwargs):
        layer = int(hook.name.split(".")[1])
        if layer not in self.inns:
            return resid

        acts = get_resid_acts(resid, position)
        orig_shape = acts.shape
        x = acts.reshape(-1, orig_shape[-1]).to(torch.float32)

        with torch.no_grad():
            z = self.inns[layer].encode(x)
            x_steered = self.inns[layer].decode(z + float(coeff) * steering_vector.to(self.device))

        return set_resid_acts(resid, position, x_steered.reshape(orig_shape).to(acts.dtype))




class BIPOSteerModel(BaseSteerModel):
    """BIPO steering model. Adds optimized steering vector directly to residual stream."""

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        **kwargs,
    ) -> torch.Tensor:
        """Add steering vector directly to activations."""
        vec = steering_vector.to(device=resid.device, dtype=resid.dtype)
        h = get_resid_acts(resid, position)
        h = h + coeff * vec
        return set_resid_acts(resid, position, h)





class CobraSteerModel(BaseSteerModel):
    """
    COBRA: Cluster-Optimized Barycentric Representation Alignment.

    Steers by projecting activations into a learned concept subspace
    (via SVD of contrast differences), applying cluster-adaptive OT
    in that subspace, then reconstructing with the language subspace
    untouched to preserve fluency.
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        cobra_centroids_A: Dict[int, torch.Tensor],
        cobra_centroids_B: Dict[int, torch.Tensor],
        cobra_coupling: Dict[int, torch.Tensor],
        cobra_k: Dict[int, int],
        cobra_P_concept: Dict[int, torch.Tensor],
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
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
        self.cobra_centroids_A = {int(k): v for k, v in cobra_centroids_A.items()}
        self.cobra_centroids_B = {int(k): v for k, v in cobra_centroids_B.items()}
        self.cobra_coupling = {int(k): v for k, v in cobra_coupling.items()}
        self.cobra_k = {int(k): v for k, v in cobra_k.items()}
        self.cobra_P_concept = {int(k): v for k, v in cobra_P_concept.items()}

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        position: Union[str, int],
        hook,
        **kwargs,
    ) -> torch.Tensor:
        device = resid.device
        dtype = resid.dtype
        layer = int(hook.name.split(".")[1])

        acts = get_resid_acts(resid, position)
        orig_shape = acts.shape
        x = acts.reshape(-1, orig_shape[-1])

        P_concept = self.cobra_P_concept[layer].to(device=device, dtype=dtype)
        centroids_A = self.cobra_centroids_A[layer].to(device=device, dtype=dtype)
        centroids_B = self.cobra_centroids_B[layer].to(device=device, dtype=dtype)
        coupling = self.cobra_coupling[layer].to(device=device, dtype=dtype)
        K = self.cobra_k[layer]

        # --- 1. Project to concept subspace ---
        z = x @ P_concept  # (n_tokens, k)

        # --- 2. Language residual (untouched) ---
        z_lang = x - z @ P_concept.T  # (n_tokens, d)

        # --- 3. Barycentric shift in concept space (float32 for cdist) ---
        z32 = z.to(torch.float32)
        centroids_A32 = centroids_A.to(torch.float32)
        centroids_B32 = centroids_B.to(torch.float32)
        coupling32 = coupling.to(torch.float32)

        if K == 1:
            v_concept = (centroids_B32[0] - centroids_A32[0]).unsqueeze(0)
        else:
            dists = torch.cdist(z32, centroids_A32, p=2) ** 2
            dists_median = torch.median(dists, dim=-1).values.clamp(min=1e-8)
            kernel = torch.exp(-dists / (2.0 * dists_median.unsqueeze(-1)))

            diffs = centroids_B32.unsqueeze(0) - centroids_A32.unsqueeze(1)
            u = torch.einsum("ij,ijk->ik", coupling32, diffs)
            s = coupling32.sum(dim=-1)
            denom = torch.matmul(kernel, s)
            num = torch.matmul(kernel, u)
            v_concept = num / (denom.unsqueeze(-1) + 1e-12)

        v_concept = v_concept.to(device=device, dtype=dtype)

        # --- 4. Steer in concept space ---
        z_steered = z + float(coeff) * v_concept

        # --- 5. Reconstruct with untouched language subspace ---
        steered = z_steered @ P_concept.T + z_lang
        steered = steered.reshape(orig_shape).to(device=device, dtype=dtype)

        return set_resid_acts(resid, position, steered)

"""Invertible Neural Network (INN) for INNSteer.

Affine coupling layers (RealNVP-style) with random permutation between blocks.
"""


class AffineCouplingLayer(nn.Module):
    """One affine coupling layer with alternating split pattern.

    Splits input in half, transforms the second half with an affine map
    conditioned on the first half.
    """

    def __init__(self, d_model: int, hidden_dim: int, parity: int):
        super().__init__()
        self.d_model = d_model
        self.split = d_model // 2
        self.parity = parity

        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.split * 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = self._split(x)
        if self.parity == 1:
            x1, x2 = x2, x1
        out = self.net(x1)
        s, t = out.chunk(2, dim=-1)
        s = torch.tanh(s)
        y2 = x2 * torch.exp(s) + t
        if self.parity == 1:
            y = torch.cat([y2, x1], dim=-1)
        else:
            y = torch.cat([x1, y2], dim=-1)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = self._split(y)
        if self.parity == 1:
            y1, y2 = y2, y1
        out = self.net(y1)
        s, t = out.chunk(2, dim=-1)
        s = torch.tanh(s)
        x2 = (y2 - t) * torch.exp(-s)
        if self.parity == 1:
            x = torch.cat([x2, y1], dim=-1)
        else:
            x = torch.cat([y1, x2], dim=-1)
        return x

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[..., :self.split], x[..., self.split:]


class InvertibleNN(nn.Module):
    """Stack of affine coupling layers with an initial activation normalization.

    Maps between original activation space and a Gaussian latent space.
    """

    def __init__(
        self,
        d_model: int,
        n_coupling_layers: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        assert d_model >= 2, "d_model must be at least 2"
        self.d_model = d_model
        self.register_buffer("shift", torch.zeros(d_model))
        self.register_buffer("scale", torch.ones(d_model))

        layers = []
        for i in range(n_coupling_layers):
            layers.append(AffineCouplingLayer(d_model, hidden_dim, parity=i % 2))
        self.layers = nn.ModuleList(layers)

    def fit_actnorm(self, x: torch.Tensor):
        """Initialize activation normalization from data."""
        with torch.no_grad():
            self.shift.copy_(x.mean(dim=0))
            self.scale.copy_(x.std(dim=0).clamp(min=1e-6))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = (x - self.shift) / self.scale
        log_det_sum = -self.scale.log().sum().expand(h.shape[0])
        for layer in self.layers:
            h, ld = layer(h)
            log_det_sum = log_det_sum + ld
        return h, log_det_sum

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in reversed(self.layers):
            h = layer.inverse(h)
        h = h * self.scale + self.shift
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.forward(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inverse(z)


class INNSteerModel(BaseSteerModel):
    """Apply INN-based nonlinear steering via encode-steer-decode."""

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        inn_state_dicts: Dict[int, Dict[str, Any]],
        inn_config: Dict[str, Any],
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
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
        n_coupling = inn_config.get("n_coupling", 4)
        hidden_dim = inn_config.get("hidden_dim", 512)
        d_model = next(iter(steering_vector.values())).shape[-1]

        self.inns: Dict[int, InvertibleNN] = {}
        for lyr in layer:
            inn = InvertibleNN(
                d_model=d_model,
                n_coupling_layers=n_coupling,
                hidden_dim=hidden_dim,
            ).to(self.device)
            raw_sd = inn_state_dicts[lyr] if lyr in inn_state_dicts else inn_state_dicts[str(lyr)]
            sd = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in raw_sd.items()}
            inn.load_state_dict(sd, strict=False)
            inn.eval()
            self.inns[lyr] = inn

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
        if layer not in self.inns:
            return resid

        inn = self.inns[layer]
        latent_vec = steering_vector.to(self.device)

        acts = get_resid_acts(resid, position)
        orig_shape = acts.shape
        x = acts.reshape(-1, orig_shape[-1]).to(torch.float32)

        with torch.no_grad():
            z = inn.encode(x)
            z_steered = z + float(coeff) * latent_vec
            x_steered = inn.decode(z_steered)

        return set_resid_acts(resid, position, x_steered.reshape(orig_shape).to(acts.dtype))


class LQRSteerModel(BaseSteerModel):
    """Closed-loop LQR activation steering.

    Feedback law (per layer per token):
        β      = v_k^T · z_k
        α      = λ · µ_k  -  β
        u_k    = α · (K_k @ v_k)
        z_out  = z_in  + u_k

    Lives in nonlinear.py because the steering magnitude α
    depends on the instantaneous activation z (input-dependent).
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        hook_point: Union[str, List[str]] = "pre",
        position: Union[str, int] = "last",
        norm: bool = False,
        lqr_mu: Optional[Dict[int, float]] = None,
        lqr_kv: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            layer=layer,
            steering_vector=steering_vector,
            hook_point=hook_point,
            position=position,
            norm=norm,
            **kwargs,
        )

        if lqr_mu is None:
            raise ValueError("LQRSteerModel requires lqr_mu in kwargs")
        if lqr_kv is None:
            raise ValueError("LQRSteerModel requires lqr_kv in kwargs")

        self.lqr_mu = {int(k): float(v) for k, v in lqr_mu.items()}
        self.lqr_kv = {
            int(k): v.detach().clone() for k, v in lqr_kv.items()
        }

        missing_mu = set(self.layer) - set(self.lqr_mu.keys())
        missing_kv = set(self.layer) - set(self.lqr_kv.keys())
        if missing_mu:
            raise ValueError(f"Missing lqr_mu for layers: {missing_mu}")
        if missing_kv:
            raise ValueError(f"Missing lqr_kv for layers: {missing_kv}")

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
        layer = int(hook.name.split(".")[1])

        v = steering_vector.to(dtype=acts.dtype, device=acts.device)
        mu = self.lqr_mu[layer]
        kv = self.lqr_kv[layer].to(dtype=acts.dtype, device=acts.device)

        beta = (acts * v).sum(dim=-1, keepdim=True)
        alpha = float(coeff) * mu - beta
        steering = alpha * kv

        updated = acts + steering

        if self.norm:
            orig_norm = acts.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            new_norm = updated.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            updated = updated * (orig_norm / new_norm)

        return set_resid_acts(resid, position, updated)