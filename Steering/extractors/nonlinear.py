"""Dense nonlinear and transport extractors (including IDS)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA, PCA
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm
from ..base import BaseExtractor
from ..config import SteeringVector
from ..flow_utils import FlowMLP, denormalize, flow_steer_activations, gaussian_stats, normalize, robust_stats, train_flow_model, project_to_basis, unproject_from_basis, ConceptEncoder, FlowBlock, FlowFunction
from ..logger import setup_logger
from ..utils import collect_dense_activations, get_hook_name, get_resid_acts, set_resid_acts
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

logger = setup_logger(__name__)


class AngularExtractor(BaseExtractor):
    """
    Angular Steering Vector Extractor.

    This lives with nonlinear extractors because inference steers by rotating
    activations in a learned 2D subspace instead of adding a single vector.
    """

    METHOD_NAME = "ANGULAR"

    def __init__(
        self,
        model,
        batch_size: int = 8,
        position: str = "last",
        layer: Optional[List[int]] = None,
        strategy: str = "max_sim",
        device: Optional[torch.device] = None,
        hook_point: List[str] = ["mid", "post"],
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.layer = range(self.model.cfg.n_layers)
        self.strategy = strategy
        self.first_direction = None
        self.second_direction = None
        self.steering_plane = None

        if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
            self.model.tokenizer.padding_side = "left"
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        if hasattr(self.model, "cfg"):
            self.model.cfg.default_prepend_bos = False

    def _get_activations(self, inputs: List[str]) -> dict:
        logger.info(f"Pre-tokenizing {len(inputs)} inputs for consistent padding")
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=list(self.layer),
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="auto",
            pretokenize_all=True,
            change_pad_token=False,
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: List[str],
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        logger.info(f"Angular: Scanning {len(self.layer)} layers with strategy '{self.strategy}'...")

        target_acts_dict = self._get_activations(target_data)
        contrast_acts_dict = self._get_activations(contrast_data)

        candidate_directions = {}
        norms = {}
        common_keys = set(target_acts_dict.keys()) & set(contrast_acts_dict.keys())

        for key in common_keys:
            target = target_acts_dict[key]
            contrast = contrast_acts_dict[key]

            target_norm = target / target.norm(dim=-1, keepdim=True)
            contrast_norm = contrast / contrast.norm(dim=-1, keepdim=True)
            target_mean = target_norm.mean(dim=0)
            contrast_mean = contrast_norm.mean(dim=0)
            target_mean_norm = target_mean / target_mean.norm()
            contrast_mean_norm = contrast_mean / contrast_mean.norm()
            diff = target_mean_norm - contrast_mean_norm
            candidate_directions[key] = diff
            norms[key] = diff.norm()

        def sort_key(k):
            return int(k.split("_")[1])

        sorted_keys = sorted(candidate_directions.keys(), key=sort_key)
        if not sorted_keys:
            raise ValueError("No common layers found between target and contrast activations")

        all_candidates = torch.stack([candidate_directions[k] for k in sorted_keys])
        pca = PCA()
        pca.fit(all_candidates.float().cpu().numpy())
        second_direction_pca = torch.from_numpy(pca.components_[0]).to(self.device).float()

        if self.strategy == "max_sim":
            candidates_normalized = {
                k: v.double() / v.double().norm() for k, v in candidate_directions.items()
            }
            candidates_stack = torch.stack([candidates_normalized[k] for k in sorted_keys])
            pairwise = candidates_stack @ candidates_stack.T
            mean_cosine = pairwise.mean(dim=-1)
            selected_key = sorted_keys[mean_cosine.argmax().item()]
        elif self.strategy == "max_norm":
            selected_key = max(norms.keys(), key=lambda k: norms[k])
        else:
            selected_key = sorted_keys[0]

        first_direction = candidate_directions[selected_key].double()
        first_direction = first_direction / first_direction.norm()
        second_direction = second_direction_pca.double()

        self.first_direction = first_direction
        self.second_direction = second_direction
        self.steering_plane = torch.stack([first_direction, second_direction])

        layer_idx = int(selected_key.split("_")[1])
        self.layer = [layer_idx]
        logger.info(f"Angular: Auto-selected layer {layer_idx} (strategy: {self.strategy})")

        self.vector = {layer_idx: self.steering_plane[0]}
        self.metadata = {
            "method": "ANGULAR",
            "selected_layer": [self.layer],
            "steering_plane": self.steering_plane,
            "first_direction": self.first_direction,
            "second_direction": self.second_direction,
            "candidate_directions": {k: v.cpu() for k, v in candidate_directions.items()},
        }
        return self.vector


def _match_pairs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[0], b.shape[0])
    if n <= 0:
        raise ValueError("Need at least one paired activation sample")
    return a[:n].to(torch.float32), b[:n].to(torch.float32)


def _sinkhorn_align(
    src: torch.Tensor,
    dst: torch.Tensor,
    reg: float = 0.05,
    max_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder *src* via Sinkhorn barycentric projection to align with *dst*.

    Computes a soft transport plan ``P`` (``reg``-entropic regularisation)
    that maps the *src* distribution onto the *dst* distribution, then
    projects each *src* point onto the barycentric combination of its
    transport partners.  The returned ``(src_aligned, dst)`` pair preserves
    the number of samples while improving the OT-wise alignment that the
    downstream flow model sees.

    Args:
        src: Source activations, shape ``(n, d)``.
        dst: Target activations, shape ``(n, d)``.
        reg: Entropic regularisation strength (``epsilon``).
        max_iter: Sinkhorn iterations.

    Returns:
        Tuple ``(src_aligned, dst)`` where ``src_aligned`` is the
        barycentrically reordered version of *src*.
    """
    n = src.shape[0]
    if n < 2:
        return src, dst

    device = src.device

    # Squared L2 cost, scale-normalised
    C = torch.cdist(src, dst, p=2).pow(2)
    C = C / (C.detach().mean() + 1e-8)

    # Uniform marginals
    a = torch.full((n,), 1.0 / n, device=device)
    b = torch.full((n,), 1.0 / n, device=device)
    K = torch.exp(-C / reg)

    u = torch.ones_like(a)
    for _ in range(max_iter):
        v = b / (K.T @ u + 1e-8)
        u = a / (K @ v + 1e-8)

    # Transport plan P[i, j] = mass moved from src[i] to dst[j]
    P = torch.diag(u) @ K @ torch.diag(v)

    # Barycentric projection: src_aligned[j] = n * sum_i P[i,j] * src[i]
    # Marginal sum over rows = b[j] = 1/n, hence the n * factor.
    src_aligned = n * (P.T @ src)

    return src_aligned, dst





class PIDExtractor(BaseExtractor):
    """Layerwise PID steering-vector extractor for LLM residual streams.

    Follows the GaussianOTPIDHook / OnlyMeanPIDHook reference from Mean-AcT:
      1. Processes layers sequentially (causal order).
      2. For each layer, activations are collected WHILE previous layers'
         steering hooks are already active (closed-loop / incremental).
      3. The error signal is the contrastive mean difference at that layer.
      4. The integral term is the MEAN of all previous layers' errors
         (not a cumulative sum), matching the reference.
      5. The derivative is the current error minus the previous layer's error.
      6. The PID-compensated direction is:
             diff_m = kp * error + ki * integral + kd * derivative
      7. This direction is registered as a steering hook on the model
         before extracting the next layer.

    Reference: ``Code/pid-steering/Mean-AcT/act/hooks/transport.py``
    ``GaussianOTPIDHook.load_state_dict()`` lines 354-402.
    """

    METHOD_NAME = "PID"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "mean",
        hook_point: List[str] | str = "post",
        pid_kp: float = 1.0,
        pid_ki: float = 0.005,
        pid_kd: float = 0.0,
        pid_normalize_error: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.pid_kp = float(pid_kp)
        self.pid_ki = float(pid_ki)
        self.pid_kd = float(pid_kd)
        self.pid_normalize_error = bool(pid_normalize_error)

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=self.layer,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("PIDExtractor requires contrast_data")

        self.layer = sorted(self.layer)
        vectors: Dict[int, torch.Tensor] = {}
        pid_diffs: Dict[int, torch.Tensor] = {}

        # --- incremental steering hook factory ---
        def make_incremental_hook(direction: torch.Tensor):
            def hook_fn(resid, hook):
                acts = get_resid_acts(resid, self.position)
                d = acts.dtype
                x = acts.to(torch.float32)
                dir_vec = direction.to(device=x.device, dtype=torch.float32)
                updated = x + dir_vec
                return set_resid_acts(resid, self.position, updated.to(d))
            return hook_fn

        hp_list = (
            [self.hook_point]
            if isinstance(self.hook_point, (str, bytes))
            else list(self.hook_point)
        )

        try:
            for layer in self.layer:
                # 1. Collect activations at this layer (previous hooks steer the model)
                target_acts = self._get_activations(target_data, layers=[layer])
                contrast_acts = self._get_activations(contrast_data, layers=[layer])

                target_mean = target_acts[layer].mean(dim=0)
                contrast_mean = contrast_acts[layer].mean(dim=0)

                if self.pid_normalize_error:
                    target_mean = F.normalize(target_mean, dim=-1)
                    contrast_mean = F.normalize(contrast_mean, dim=-1)

                diff = target_mean - contrast_mean
                pid_diffs[layer] = diff.detach().cpu()

                # 2. Compute PID-compensated direction
                prev_layers = sorted([l for l in pid_diffs if l != layer])
                if prev_layers:
                    prev_diffs = torch.stack(
                        [pid_diffs[l].to(device=diff.device, dtype=diff.dtype)
                         for l in prev_layers]
                    )
                    # integral = mean of ALL previous errors (reference formula)
                    integral = prev_diffs.mean(dim=0)
                    # derivative = current error - previous error
                    prev_e = pid_diffs[prev_layers[-1]].to(device=diff.device, dtype=diff.dtype)
                    derivative = diff - prev_e
                else:
                    integral = torch.zeros_like(diff)
                    derivative = torch.zeros_like(diff)

                diff_m = (
                    self.pid_kp * diff
                    + self.pid_ki * integral
                    + self.pid_kd * derivative
                )
                vectors[layer] = diff_m.to(self.device)

                # 3. Register hook for this layer (active for subsequent layers)
                for hp in hp_list:
                    hook_name = get_hook_name(layer, hp)
                    self.model.add_hook(
                        hook_name,
                        make_incremental_hook(vectors[layer]),
                    )
        finally:
            self.model.reset_hooks()

        self.vector = vectors
        self.metadata = {
            "method": "PID",
            "pid_kp": self.pid_kp,
            "pid_ki": self.pid_ki,
            "pid_kd": self.pid_kd,
            "pid_normalize_error": self.pid_normalize_error,
            "pid_layer_order": [int(l) for l in self.layer],
            "pid_diffs": {k: v.cpu() for k, v in pid_diffs.items()},
            "n_target": len(target_data),
            "n_contrast": len(contrast_data),
        }
        return self.vector


class CurveballExtractor(BaseExtractor):
    """Polynomial Kernel-PCA extractor for Curveball Steering."""

    METHOD_NAME = "CURVEBALL"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        hook_point: List[str] | str = "post",
        curveball_kernel: str = "rbf",
        curveball_dim: int = 8,
        curveball_degree: int = 2,
        curveball_gamma: float = 0.001,
        curveball_coef0: float = 1.0,
        curveball_inverse_alpha: float = 1e-3,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position, **kwargs)
        self.curveball_kernel = curveball_kernel    
        self.curveball_dim = int(curveball_dim)
        self.curveball_degree = int(curveball_degree)
        self.curveball_gamma = curveball_gamma
        self.curveball_coef0 = float(curveball_coef0)
        self.curveball_inverse_alpha = float(curveball_inverse_alpha)

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=self.layer,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("CurveballExtractor requires contrast_data")

        target_acts = self._get_activations(target_data)
        contrast_acts = self._get_activations(contrast_data)
        vectors: Dict[int, torch.Tensor] = {}
        models: Dict[int, Any] = {}
        directions: Dict[int, torch.Tensor] = {}

        for layer in self.layer:
            contrast, target = _match_pairs(contrast_acts[layer], target_acts[layer])
            train = torch.cat([contrast, target], dim=0).cpu().numpy()
            n_components = min(self.curveball_dim, max(1, train.shape[0] - 1), train.shape[1])
            kpca = KernelPCA(
                n_components=n_components,
                kernel=self.curveball_kernel,
                degree=self.curveball_degree,
                gamma=self.curveball_gamma,
                coef0=self.curveball_coef0,
                fit_inverse_transform=True,
                alpha=self.curveball_inverse_alpha,
            )
            z = torch.tensor(kpca.fit_transform(train), dtype=torch.float32)
            z_src = z[: contrast.shape[0]]
            z_dst = z[contrast.shape[0] :]
            z_dir = z_dst.mean(dim=0) - z_src.mean(dim=0)
            directions[layer] = z_dir
            models[layer] = kpca

            # Dense fallback/vector summary for metadata consumers.
            vectors[layer] = target.mean(dim=0).to(self.device) - contrast.mean(dim=0).to(self.device)

        self.vector = vectors
        self.metadata = {
            "method": "CURVEBALL",
            "curveball_models": models,
            "curveball_directions": directions,
            "curveball_dim": self.curveball_dim,
            "curveball_degree": self.curveball_degree,
            "curveball_gamma": self.curveball_gamma,
            "curveball_coef0": self.curveball_coef0,
            "curveball_inverse_alpha": self.curveball_inverse_alpha,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data),
        }
        return self.vector


class FlowExtractor(BaseExtractor):
    """Flow Matching extractor supporting concept transport or correction learning."""

    METHOD_NAME = "FLOW"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        hook_point: List[str] | str = "post",
        flow_hidden_dim: int = 512,
        flow_layers: int = 6,
        flow_lr: float = 1e-3,
        flow_epochs: int = 200,
        flow_batch_size: int = 64,
        flow_seed: int = 42,
        flow_subspace_dim: Optional[int] = None,
        flow_loss_mode: str = "huber",
        flow_max_weight: Optional[float] = None,
        flow_norm_mode: str = "iqr",
        flow_weighted: bool = False,
        flow_train_space: str = "full",
        flow_target_type: str = "concept",
        flow_denoise_mode: str = "none",
        flow_ot: Optional[str] = None,
        flow_lm_loss: bool = False,
        flow_lm_lambda: float = 0.1,
        flow_lm_lr: float = 5e-5,
        flow_lm_epochs: int = 3,
        flow_lm_grad_accum: int = 4,
        flow_lm_batch_size: int = 4,
        flow_max_new_tokens: int = 30,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.flow_hidden_dim = int(flow_hidden_dim)
        self.flow_layers = int(flow_layers)
        self.flow_lr = float(flow_lr)
        self.flow_epochs = int(flow_epochs)
        self.flow_batch_size = int(flow_batch_size)
        self.flow_seed = int(flow_seed)
        self.flow_subspace_dim = flow_subspace_dim
        self.flow_loss_mode = str(flow_loss_mode).strip().lower()
        self.flow_max_weight = float(flow_max_weight) if flow_max_weight is not None else None
        self.flow_norm_mode = str(flow_norm_mode).strip().lower()
        if self.flow_norm_mode not in {"iqr", "gaussian"}:
            raise ValueError(f"Unsupported flow_norm_mode '{flow_norm_mode}'. Expected 'iqr' or 'gaussian'.")
        self.flow_weighted = bool(flow_weighted)
        self.flow_train_space = str(flow_train_space).strip().lower()
        if self.flow_train_space not in {"full", "pca_diff", "pca_stack", "lda"}:
            raise ValueError(
                "Unsupported flow_train_space; expected 'full', 'pca_diff', 'pca_stack', or 'lda'"
            )
        self.flow_target_type = str(flow_target_type).strip().lower()
        if self.flow_target_type not in {"concept", "correction"}:
            raise ValueError("Unsupported flow_target_type; expected 'concept' or 'correction'")
        self.flow_denoise_mode = str(flow_denoise_mode).strip().lower()
        if self.flow_denoise_mode not in {"none", "proj", "correction"}:
            raise ValueError("Unsupported flow_denoise_mode; expected one of 'none','proj','correction'")
        self.flow_ot = flow_ot.lower() if flow_ot is not None else None
        if self.flow_ot is not None and self.flow_ot not in {"sinkhorn"}:
            raise ValueError(
                f"Unsupported flow_ot '{flow_ot}'. Expected None or 'sinkhorn'."
            )
        self.flow_lm_loss = bool(flow_lm_loss)
        self.flow_lm_lambda = float(flow_lm_lambda)
        self.flow_lm_lr = float(flow_lm_lr)
        self.flow_lm_epochs = int(flow_lm_epochs)
        self.flow_lm_grad_accum = int(flow_lm_grad_accum)
        self.flow_lm_batch_size = int(flow_lm_batch_size)
        self.flow_max_new_tokens = int(flow_max_new_tokens)

    def _build_flow_basis(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], int]:
        if self.flow_train_space == "full":
            return None, None, None, src.shape[-1]

        if self.flow_train_space in {"pca_diff", "pca_stack"}:
            basis_data = torch.cat([src, dst], dim=0) if self.flow_train_space == "pca_stack" else (dst - src)
            basis_dim = basis_data.shape[-1] if self.flow_subspace_dim is None else int(self.flow_subspace_dim)
            basis_dim = max(1, min(basis_dim, basis_data.shape[0], basis_data.shape[-1]))
            centered = basis_data - basis_data.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            basis = vh[:basis_dim].cpu().to(torch.float32)
            basis_mean = basis_data.mean(dim=0, keepdim=True).cpu().to(torch.float32)
            basis_inv = torch.linalg.pinv(basis)
            return basis, basis_mean, basis_inv, basis_dim

        if self.flow_train_space == "lda":
            basis_data = torch.cat([src, dst], dim=0)
            labels = torch.cat(
                [
                    torch.zeros(src.shape[0], dtype=torch.long, device=basis_data.device),
                    torch.ones(dst.shape[0], dtype=torch.long, device=basis_data.device),
                ],
                dim=0,
            )
            # LDA max components limited by min(n_features, n_classes - 1).
            n_classes = int(torch.unique(labels).numel())
            max_lda = max(1, min(basis_data.shape[-1], max(1, n_classes - 1)))
            # Use configured flow_subspace_dim when provided, otherwise fall back to max_lda.
            if self.flow_subspace_dim is not None:
                lda_dim = max(1, min(int(self.flow_subspace_dim), max_lda))
            else:
                lda_dim = max_lda
            lda = LinearDiscriminantAnalysis(n_components=lda_dim)
            lda.fit(basis_data.cpu().numpy(), labels.cpu().numpy())
            basis = torch.from_numpy(lda.scalings_[:, :lda_dim].T).to(torch.float32)
            basis_mean = basis_data.mean(dim=0, keepdim=True).cpu().to(torch.float32)
            basis_inv = torch.linalg.pinv(basis)
            return basis, basis_mean, basis_inv, lda_dim

        raise ValueError(f"Unsupported flow_train_space '{self.flow_train_space}'")

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=kwargs.get("layers", self.layer),
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def _flow_lm_finetune(
        self,
        flow_model: FlowMLP,
        src: torch.Tensor,
        dst: torch.Tensor,
        basis: Optional[torch.Tensor],
        basis_mean: Optional[torch.Tensor],
        basis_inv: Optional[torch.Tensor],
        src_stats: Dict[str, torch.Tensor],
        dst_stats: Dict[str, torch.Tensor],
        prompts: List[str],
        target_response: List[str],
    ) -> FlowMLP:
        from transformers import get_linear_schedule_with_warmup, set_seed

        set_seed(self.flow_seed)
        flow_model.to(self.device).train()

        optimizer = torch.optim.AdamW(flow_model.parameters(), lr=self.flow_lm_lr, weight_decay=0.0)
        n_steps = math.ceil(self.flow_lm_epochs * math.ceil(len(prompts) / self.flow_lm_batch_size) / self.flow_lm_grad_accum)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(1, n_steps))

        original_requires_grad = {}
        for name, p in self.model.named_parameters():
            original_requires_grad[name] = p.requires_grad
            p.requires_grad = False

        batch_state: Dict[str, Any] = {}
        hp = self.hook_point[0] if isinstance(self.hook_point, (list, tuple)) else self.hook_point
        hook_name = get_hook_name(self.layer[0], hp)

        def make_hook(layer_idx: int):
            def hook_fn(resid, hook):
                prompt_lens = batch_state["prompt_lens"]
                batch_size, seq_len, d_model = resid.shape
                train_space = str(getattr(self, "flow_train_space", "full")).strip().lower()

                # Modify all positions from last prompt token onwards
                pos = torch.arange(seq_len, device=resid.device)
                mask = pos[None, :] >= (prompt_lens[:, None] - 1)

                acts_flat = resid.reshape(-1, d_model)
                x_pred_flat = flow_steer_activations(
                    x=acts_flat.to(torch.float32),
                    flow_model=flow_model,
                    source_stats=src_stats,
                    target_stats=dst_stats,
                    basis=basis,
                    basis_mean=basis_mean,
                    basis_inv=basis_inv,
                    train_space=train_space,
                    steps=1,
                    denoise_mode="proj" if basis is not None else "none",
                    coeff=1.0,
                ).to(dtype=acts_flat.dtype)

                updated = resid.clone()
                updated[mask] = x_pred_flat.reshape(batch_size, seq_len, d_model)[mask]
                return updated
            return hook_fn

        fwd_hooks = [(hook_name, make_hook(self.layer[0]))]

        try:
            ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            accum_count = 0
            optimizer.zero_grad()

            for epoch in range(self.flow_lm_epochs):
                shuffle_rng = random.Random(self.flow_seed + epoch)
                indices = list(range(len(prompts)))
                shuffle_rng.shuffle(indices)

                pbar = tqdm(range(0, len(prompts), self.flow_lm_batch_size), desc=f"Flow-LM Epoch {epoch+1}/{self.flow_lm_epochs}")
                for batch_start in pbar:
                    batch_idx = indices[batch_start:batch_start + self.flow_lm_batch_size]
                    batch_texts = [prompts[i] + target_response[i] for i in batch_idx]

                    tokens, labels, attention_mask, prompt_lens = _get_completion_masked_labels(
                        self.model, batch_texts
                    )
                    batch_state["prompt_lens"] = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)

                    # Truncate CE loss to only the first flow_max_new_tokens response tokens
                    seq_len = labels.shape[1]
                    max_stop = torch.tensor(prompt_lens, device=labels.device) + self.flow_max_new_tokens
                    pos = torch.arange(seq_len, device=labels.device)
                    labels[pos[None, :] >= max_stop[:, None]] = -100

                    # CE-only fine-tuning (CFM was pre-trained above by train_flow_model)
                    with self.model.hooks(fwd_hooks):
                        logits = self.model(tokens, attention_mask=attention_mask)

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    ce_loss = ce_loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    loss = ce_loss / self.flow_lm_grad_accum
                    loss.backward()
                    pbar.set_postfix({"ce": f"{ce_loss.item():.4f}"})
                    accum_count += 1

                    if accum_count % self.flow_lm_grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

            if accum_count % self.flow_lm_grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        finally:
            for name, p in self.model.named_parameters():
                p.requires_grad = original_requires_grad[name]

        flow_model.eval()
        return flow_model

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("FlowExtractor requires contrast_data")

        target_acts = self._get_activations(target_data)
        source_acts = None
        query_acts = None
        contrast_acts = None

        if self.flow_target_type == "correction":
            query_data = kwargs.get("prompts")
            if query_data is None:
                query_data = contrast_data
            query_acts = self._get_activations(query_data)
            contrast_acts = self._get_activations(contrast_data)
        else:
            source_acts = self._get_activations(contrast_data)

        vectors: Dict[int, torch.Tensor] = {}
        flow_payload: Dict[int, Dict[str, Any]] = {}
        lm_trained: bool = False

        prompts = kwargs.get("prompts") if self.flow_lm_loss else None
        target_response = kwargs.get("target_response") if self.flow_lm_loss else None
        if self.flow_lm_loss and (prompts is None or target_response is None):
            raise ValueError(
                "flow_lm_loss=True requires 'prompts' and 'target_response' in kwargs. "
                "Ensure the dataset formatter emits these fields."
            )

        for layer in self.layer:
            if self.flow_target_type == "correction":
                n = min(query_acts[layer].shape[0], target_acts[layer].shape[0], contrast_acts[layer].shape[0])
                src = query_acts[layer][:n].to(torch.float32)
                target = target_acts[layer][:n].to(torch.float32)
                contrast = contrast_acts[layer][:n].to(torch.float32)
                dst = target - contrast
            else:
                src, dst = _match_pairs(source_acts[layer], target_acts[layer])

            if self.flow_ot == "sinkhorn":
                src, dst = _sinkhorn_align(src, dst)

            basis, basis_mean, basis_inv, basis_dim = self._build_flow_basis(src, dst)

            # choose normalization stats and optionally project for training
            if basis is not None:
                src_proj = project_to_basis(src, basis, basis_mean)
                dst_proj = project_to_basis(dst, basis, basis_mean)
                if self.flow_norm_mode == "gaussian":
                    src_stats = gaussian_stats(src_proj)
                    dst_stats = gaussian_stats(dst_proj)
                else:
                    src_stats = robust_stats(src_proj)
                    dst_stats = robust_stats(dst_proj)
                src_norm = normalize(src_proj, src_stats)
                dst_norm = normalize(dst_proj, dst_stats)
                train_dim = basis_dim
            else:
                if self.flow_norm_mode == "gaussian":
                    src_stats = gaussian_stats(src)
                    dst_stats = gaussian_stats(dst)
                else:
                    src_stats = robust_stats(src)
                    dst_stats = robust_stats(dst)
                src_norm = normalize(src, src_stats)
                dst_norm = normalize(dst, dst_stats)
                train_dim = src.shape[-1]

            model, train_info = train_flow_model(
                src_norm,
                dst_norm,
                hidden_dim=self.flow_hidden_dim,
                n_layers=self.flow_layers,
                lr=self.flow_lr,
                epochs=self.flow_epochs,
                batch_size=self.flow_batch_size,
                seed=self.flow_seed,
                device=str(self.device),
                loss_mode=self.flow_loss_mode,
                max_weight=self.flow_max_weight,
                weighted=self.flow_weighted,
            )

            if self.flow_lm_loss:
                n_lm = min(len(prompts), len(target_response), len(src), len(dst))
                model = self._flow_lm_finetune(
                    model, src[:n_lm], dst[:n_lm],
                    basis, basis_mean, basis_inv,
                    src_stats, dst_stats,
                    prompts[:n_lm], target_response[:n_lm],
                )
                lm_trained = True

            if self.flow_target_type == "correction":
                vectors[layer] = dst.mean(dim=0).to(self.device)
            else:
                vectors[layer] = dst.mean(dim=0).to(self.device) - src.mean(dim=0).to(self.device)

            flow_payload[layer] = {
                "state_dict": model.state_dict(),
                # dim is model input dim: basis dim if trained in a reduced space, else full-dim
                "dim": int(train_dim),
                "hidden_dim": self.flow_hidden_dim,
                "n_layers": self.flow_layers,
                "source_stats": src_stats,
                "target_stats": dst_stats,
                "flow_basis": basis,
                "flow_basis_mean": basis_mean,
                "flow_basis_inv": basis_inv,
                "flow_subspace_dim": self.flow_subspace_dim,
                "flow_train_space": self.flow_train_space,
                "flow_target_type": self.flow_target_type,
                "flow_denoise_mode": self.flow_denoise_mode,
                "flow_ot": self.flow_ot,
                "flow_loss_mode": self.flow_loss_mode,
                "flow_max_weight": self.flow_max_weight,
                "flow_weighted": self.flow_weighted,
                "flow_norm_mode": self.flow_norm_mode,
                "flow_lm_trained": lm_trained,
                "train_info": train_info,
            }

        self.vector = vectors
        self.metadata = {
            "method": "FLOW",
            "flow_models": flow_payload,
            "flow_ot": self.flow_ot,
            "flow_train_space": self.flow_train_space,
            "flow_target_type": self.flow_target_type,
            "flow_subspace_dim": self.flow_subspace_dim,
            "flow_lm_trained": lm_trained,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data),
        }
        return self.vector


def _get_completion_masked_labels(model, batch_texts) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    tokenizer = model.tokenizer
    seps = [
        "<start_of_turn>model\n",
        "<start_of_turn>assistant\n",
        "<|start_header_id|>assistant<|end_header_id|>\n",
        "<|im_start|>assistant\n",
        "\nAssistant:",
        "Response:",
        "<end_of_turn>\n",
        "<end_of_turn>",
        "[/INST]"
    ]
    
    batch_input_ids = []
    batch_label_ids = []
    batch_prompt_lens = []
    
    for text in batch_texts:
        bos_token = tokenizer.bos_token if hasattr(tokenizer, "bos_token") else None
        if bos_token and text.startswith(bos_token):
            text = text[len(bos_token):]
        elif text.startswith("<bos>"):
            text = text[len("<bos>"):]
            
        parts = []
        for sep in seps:
            if sep in text:
                parts = text.split(sep)
                prompt_part = parts[0] + sep
                completion_part = parts[1]
                break
        else:
            # If no separator is found, treat the entire text as the completion
            prompt_part = ""
            completion_part = text
            
        prompt_ids = tokenizer(prompt_part, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(prompt_part + completion_part, add_special_tokens=True)["input_ids"]
        
        prompt_len = len(prompt_ids)
        label_ids = list(full_ids)
        for i in range(min(prompt_len, len(label_ids))):
            label_ids[i] = -100
            
        batch_input_ids.append(torch.tensor(full_ids))
        batch_label_ids.append(torch.tensor(label_ids))
        batch_prompt_lens.append(prompt_len)
        
    max_len = max(len(x) for x in batch_input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    padded_inputs = []
    padded_labels = []
    padded_attention_masks = []
    
    for input_ids, label_ids in zip(batch_input_ids, batch_label_ids):
        pad_len = max_len - len(input_ids)
        
        padded_input = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_label = torch.cat([label_ids, torch.full((pad_len,), -100, dtype=torch.long)])
        attention_mask = torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        
        padded_inputs.append(padded_input)
        padded_labels.append(padded_label)
        padded_attention_masks.append(attention_mask)
        
    tokens = torch.stack(padded_inputs).to(model.cfg.device)
    labels = torch.stack(padded_labels).to(model.cfg.device)
    attention_mask = torch.stack(padded_attention_masks).to(model.cfg.device)
    
    return tokens, labels, attention_mask, batch_prompt_lens


def _get_sequence_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    shift_log_probs = log_probs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != -100).float()
    
    # Temporarily set -100 to 0 to avoid index errors in gather, masked out later anyway
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0
    
    gathered = torch.gather(shift_log_probs, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, device, dtype):
        super().__init__()
        # Initialize orthogonal basis on CPU in float32 to bypass CUDA BFloat16 geqrf limitations
        weight_init = torch.empty(n, m, dtype=torch.float32)
        torch.nn.init.orthogonal_(weight_init)
        self.weight = torch.nn.Parameter(weight_init.to(device=device))

    def forward(self, x):
        # Cast input to float32, perform matmul, and cast result back to input's dtype
        return torch.matmul(x.to(torch.float32), self.weight).to(x.dtype)

    def _apply(self, fn):
        # Intercept datatype cast and force all parameters (including unconstrained originals) to stay in float32
        super()._apply(fn)
        for param in self.parameters():
            param.data = param.data.to(dtype=torch.float32)
        return self


class ReFTTrainModule(torch.nn.Module):
    def __init__(self, layers: List[int], d_model: int, r: int, add_bias: bool, device, dtype):
        super().__init__()
        self.layers = layers
        self.d_model = d_model
        self.r = r
        self.add_bias = add_bias
        
        self.rotate_layer = torch.nn.ModuleDict()
        self.learned_weight = torch.nn.ParameterDict()
        self.learned_bias = torch.nn.ParameterDict()
        
        for layer in layers:
            # LowRankRotateLayer is initialized and then wrapped with PyTorch's orthogonal parametrization
            rotate_mod = LowRankRotateLayer(d_model, r, device, dtype)
            self.rotate_layer[str(layer)] = torch.nn.utils.parametrizations.orthogonal(rotate_mod)

            # Exactly replicate nn.Linear(d_model, r) default init so that our random
            # state sequence matches the GT's PreferenceLoreftIntervention:
            #   1. kaiming_uniform_(weight, a=sqrt(5))
            #   2. uniform_(bias, -bound, bound)
            W_init = torch.empty(r, d_model, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(W_init, a=math.sqrt(5))
            W = torch.nn.Parameter(W_init.to(device=device, dtype=dtype))
            self.learned_weight[str(layer)] = W

            if add_bias:
                fan_in = d_model
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                b_init = torch.empty(r, dtype=torch.float32)
                torch.nn.init.uniform_(b_init, -bound, bound)
                b = torch.nn.Parameter(b_init.to(device=device, dtype=dtype))
                self.learned_bias[str(layer)] = b




class LoReFTExtractor(BaseExtractor):
    """Low-rank rotate-and-reconstruct extractor used as the base ReFT/RePS family."""

    METHOD_NAME = "LOREFT"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        hook_point: List[str] | str = "pre",
        reft_low_rank_dimension: int = 8,
        dropout: float = 0.0,
        act_fn: str = "linear",
        add_bias: bool = True,
        device: Optional[torch.device] = None,
        lr: float = 5e-3,
        weight_decay: float = 0.0,
        epochs: int = 3,
        preference_pairs=None,
        substraction_type: str = "zero",
        steering_factors=None,
        grad_accum: int = 1,
        reft_seed: int = 42,
        reft_steer_once: bool = True,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.reft_steer_once = reft_steer_once
        self.low_rank_dimension = int(reft_low_rank_dimension)
        self.dropout = float(dropout)
        self.act_fn = "linear" if act_fn is None else str(act_fn)
        self.add_bias = bool(add_bias)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)

        if preference_pairs is None:
            self.preference_pairs = ["orig_add"]
        elif isinstance(preference_pairs, str):
            self.preference_pairs = [preference_pairs]
        else:
            self.preference_pairs = list(preference_pairs)

        self.substraction_type = substraction_type

        if steering_factors is None or len(steering_factors) == 0:
            self.steering_factors = [1.0]
        else:
            self.steering_factors = [float(x) for x in steering_factors]

        self.grad_accum = int(grad_accum)
        self.reft_seed = int(reft_seed)

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=self.layer,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        from transformers import get_linear_schedule_with_warmup, set_seed

        # Seed ALL random state before any weight initialization or shuffling
        set_seed(self.reft_seed)

        d_model = self.model.cfg.d_model
        dtype = self.model.cfg.dtype

        # 1. Build training module
        train_module = ReFTTrainModule(
            layers=self.layer,
            d_model=d_model,
            r=self.low_rank_dimension,
            add_bias=self.add_bias,
            device=self.device,
            dtype=dtype,
        )

        # 2. Setup AdamW optimizer & calculate training steps
        optimizer = torch.optim.AdamW(
            train_module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        n_samples = 0
        if "orig_add" in self.preference_pairs:
            n_samples += len(target_data)
        if "orig_sub" in self.preference_pairs:
            if contrast_data is not None:
                n_samples += len(contrast_data)
            else:
                logger.warning("orig_sub requested but contrast_data is None. Using target_data as fallback.")
                n_samples += len(target_data)

        n_steps = math.ceil(
            self.epochs * math.ceil(n_samples / self.batch_size)
            / self.grad_accum
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=max(1, n_steps)
        )

        # 3. Dynamic batch state and unified hook
        batch_state = {}
        hook_pt = (
            self.hook_point[0] if isinstance(self.hook_point, (list, tuple)) else self.hook_point
        )

        def make_reft_hook(layer_str: str):
            def reps_hook(resid, hook):
                prompt_lens_tensor = batch_state["prompt_lens"]
                sf_tensor = batch_state["sf"]
                batch_size = resid.shape[0]

                R = train_module.rotate_layer[layer_str].weight.T.to(resid.dtype)
                W = train_module.learned_weight[layer_str].to(resid.dtype)
                b = train_module.learned_bias.get(layer_str)

                if self.reft_steer_once:
                    # Steer once at the last prompt token position only
                    indices = prompt_lens_tensor - 1
                    acts = resid[torch.arange(batch_size), indices]

                    rotated_base = torch.matmul(acts, R.T)
                    source_output = torch.matmul(acts, W.T)
                    if b is not None:
                        source_output = source_output + b.to(source_output.dtype)

                    delta = source_output - rotated_base
                    updated = acts + sf_tensor * torch.matmul(delta, R)

                    if self.dropout > 0.0:
                        updated = torch.nn.functional.dropout(updated, p=self.dropout, training=True)

                    resid_new = resid.clone()
                    resid_new[torch.arange(batch_size), indices] = updated
                    return resid_new
                else:
                    # Steer last prompt token and all subsequent response tokens
                    resid_new = resid.clone()
                    for i in range(batch_size):
                        start_idx = prompt_lens_tensor[i].item() - 1
                        acts = resid[i, start_idx:] # shape [seq_len - start_idx, d_model]

                        rotated_base = torch.matmul(acts, R.T)
                        source_output = torch.matmul(acts, W.T)
                        if b is not None:
                            source_output = source_output + b.to(source_output.dtype)

                        delta = source_output - rotated_base
                        updated = acts + sf_tensor[i] * torch.matmul(delta, R)

                        if self.dropout > 0.0:
                            updated = torch.nn.functional.dropout(updated, p=self.dropout, training=True)

                        resid_new[i, start_idx:] = updated
                    return resid_new
            return reps_hook

        fwd_hooks = [
            (get_hook_name(layer, hook_pt), make_reft_hook(str(layer)))
            for layer in self.layer
        ]

        # 4. Freeze base model and run training loop
        original_requires_grad = {}
        for name, p in self.model.named_parameters():
            original_requires_grad[name] = p.requires_grad
            p.requires_grad = False

        shuffle_rng = random.Random(self.reft_seed)

        try:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            accum_count = 0
            optimizer.zero_grad()

            for epoch in range(self.epochs):
                # Construct combined training data for this epoch
                epoch_data = []
                data_len = len(target_data)
                if contrast_data is not None:
                    data_len = min(data_len, len(contrast_data))

                for i in range(data_len):
                    if "orig_add" in self.preference_pairs:
                        epoch_data.append({"text": target_data[i], "type": "add"})
                    if "orig_sub" in self.preference_pairs:
                        src_text = contrast_data[i] if contrast_data is not None else target_data[i]
                        epoch_data.append({"text": src_text, "type": "sub"})

                # Shuffle the mixed training list at the start of each epoch
                shuffle_rng.shuffle(epoch_data)

                pbar = tqdm(range(0, len(epoch_data), self.batch_size), desc=f"LoREFT Epoch {epoch + 1}/{self.epochs}")
                for idx in pbar:
                    batch_items = epoch_data[idx : idx + self.batch_size]
                    if not batch_items:
                        continue

                    batch_texts = [item["text"] for item in batch_items]

                    # Generate dynamic steering factors per element
                    sf_list = []
                    for item in batch_items:
                        if item["type"] == "add":
                            sf = float(shuffle_rng.choice(self.steering_factors))
                        else:  # type == "sub"
                            if self.substraction_type == "zero": sf = 0.0
                            else: sf = -float(shuffle_rng.choice(self.steering_factors))
                        sf_list.append(sf)

                    tokens, labels, attention_mask, prompt_lens = _get_completion_masked_labels(
                        self.model, batch_texts
                    )

                    batch_state["prompt_lens"] = torch.tensor(
                        prompt_lens, dtype=torch.long, device=self.device
                    )
                    batch_state["sf"] = torch.tensor(
                        sf_list, dtype=dtype, device=self.device
                    ).unsqueeze(1)

                    with self.model.hooks(fwd_hooks):
                        logits = self.model(tokens, attention_mask=attention_mask)

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    (loss / self.grad_accum).backward()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    accum_count += 1

                    if accum_count % self.grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(train_module.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

            # Flush leftover accumulated gradients
            if accum_count % self.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(train_module.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        finally:
            for name, p in self.model.named_parameters():
                p.requires_grad = original_requires_grad[name]

        # 5. Extract trained parameters
        vectors = {}
        rotate_basis = {}
        learned_weight = {}
        learned_bias = {}

        for layer in self.layer:
            R = train_module.rotate_layer[str(layer)].weight.T.detach().clone()
            W = train_module.learned_weight[str(layer)].detach().clone()
            b = train_module.learned_bias.get(str(layer))
            if b is not None:
                b = b.detach().clone()

            rotate_basis[layer] = R
            learned_weight[layer] = W
            learned_bias[layer] = b
            vectors[layer] = self._get_activations(target_data)[layer].mean(dim=0).to(self.device) - self._get_activations(contrast_data)[layer].mean(dim=0).to(self.device)

        self.vector = vectors
        self.metadata = {
            "method": self.METHOD_NAME,
            "reft_low_rank_dimension": self.low_rank_dimension,
            "dropout": self.dropout,
            "act_fn": self.act_fn,
            "add_bias": self.add_bias,
            "grad_accum": self.grad_accum,
            "rotate_basis": rotate_basis,
            "learned_weight": learned_weight,
            "learned_bias": learned_bias,
            "preference_pairs": self.preference_pairs,
            "substraction_type": self.substraction_type,
            "steering_factors": self.steering_factors,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data) if contrast_data else 0,
        }
        return self.vector






def _flas_xavier_init_linear(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def _flas_build_flow_model(
    model_id: str,
    layer: int,
    concept_encoder_layers: int = 2,
    num_blocks: int = 2,
    time_conditioned: bool = True,
    init_from_gemma: bool = True,
    disable_cross_attn: bool = False,
    disable_self_attn: bool = False,
    disable_mlp: bool = False,
):
    config = AutoConfig.from_pretrained(model_id)
    flow_fn = FlowFunction(
        config,
        num_blocks=num_blocks,
        time_conditioned=time_conditioned,
        layer_idx=layer,
        disable_cross_attn=disable_cross_attn,
        disable_self_attn=disable_self_attn,
        disable_mlp=disable_mlp,
    )

    if init_from_gemma:
        from transformers import AutoModelForCausalLM

        full_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        src_layer = full_model.model.layers[layer]
        for block in flow_fn.blocks:
            block.gate_proj.load_state_dict(src_layer.mlp.gate_proj.state_dict())
            block.up_proj.load_state_dict(src_layer.mlp.up_proj.state_dict())
            block.down_proj.load_state_dict(src_layer.mlp.down_proj.state_dict())
            block.pre_mlp_norm.load_state_dict(src_layer.pre_feedforward_layernorm.state_dict())
            block.post_mlp_norm.load_state_dict(src_layer.post_feedforward_layernorm.state_dict())
            block.self_attn.load_state_dict(src_layer.self_attn.state_dict())
            block.pre_sa_norm.load_state_dict(src_layer.input_layernorm.state_dict())
            block.post_sa_norm.load_state_dict(src_layer.post_attention_layernorm.state_dict())
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        for block in flow_fn.blocks:
            _flas_xavier_init_linear(block.gate_proj)
            _flas_xavier_init_linear(block.up_proj)
            _flas_xavier_init_linear(block.down_proj)
            for _, module in block.self_attn.named_modules():
                _flas_xavier_init_linear(module)

    concept_enc = ConceptEncoder(model_id, num_layers=int(concept_encoder_layers))
    return flow_fn, concept_enc


def _flas_collate_prompt_output(
    samples,
    tokenizer,
    max_len: int,
    concept_max_len: int,
):
    """Collate (input, output, concept, concept_id) rows for FLAS training."""
    input_texts, output_texts, concept_texts, concept_ids = zip(*samples)

    full_ids_list = []
    prompt_lens = []
    for inp, out in zip(input_texts, output_texts):
        messages = [{"role": "user", "content": inp}]
        prompt_enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = prompt_enc.input_ids if hasattr(prompt_enc, "input_ids") else prompt_enc
        output_ids = tokenizer(out, add_special_tokens=False).input_ids
        full_ids = (prompt_ids + output_ids)[:max_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        if prompt_len <= 0:
            raise ValueError("FLAS training prompt must have at least one token")
        if prompt_len >= len(full_ids):
            raise ValueError(
                "FLAS training sample has no output tokens after truncation; "
                "increase flas_train_max_len or provide shorter prompts."
            )
        full_ids_list.append(full_ids)
        prompt_lens.append(prompt_len)

    for concept_text in concept_texts:
        if not str(concept_text).strip():
            raise ValueError("FLAS concept text must be non-empty")

    enc = tokenizer.pad({"input_ids": full_ids_list}, return_tensors="pt", padding=True)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    labels = input_ids.clone()
    prompt_mask = torch.zeros_like(attention_mask)
    for i, prompt_len in enumerate(prompt_lens):
        labels[i, :prompt_len] = -100
        prompt_mask[i, :prompt_len] = 1
    prompt_mask = prompt_mask * attention_mask
    labels[attention_mask == 0] = -100

    concept_enc = tokenizer(
        list(concept_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=concept_max_len,
    )
    return (
        input_ids,
        attention_mask,
        labels,
        prompt_mask,
        concept_enc.input_ids,
        concept_enc.attention_mask,
        torch.tensor(list(concept_ids), dtype=torch.long),
    )


def _flas_diversity_loss(velocity, concept_ids, attention_mask=None):
    if velocity is None:
        return torch.tensor(0.0)
    if attention_mask is not None:
        mask = attention_mask.float().unsqueeze(-1)
        v_pooled = (velocity * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        v_pooled = velocity.mean(dim=1)
    if v_pooled.shape[0] < 2:
        return torch.tensor(0.0, device=v_pooled.device)
    v_norm = F.normalize(v_pooled, dim=-1)
    sim = torch.matmul(v_norm, v_norm.t())
    diff_mask = (concept_ids.unsqueeze(0) != concept_ids.unsqueeze(1)).float()
    if diff_mask.sum() == 0:
        return torch.tensor(0.0, device=v_pooled.device)
    return (sim * diff_mask).sum() / diff_mask.sum()


def _flas_build_training_samples(
    inputs,
    outputs,
    concepts,
    concept_ids,
):
    """Assemble (input, output, concept, concept_id) tuples aligned to the shortest list."""
    n = min(len(inputs), len(outputs), len(concepts), len(concept_ids))
    return [
        (str(inputs[i]), str(outputs[i]), str(concepts[i]), int(concept_ids[i]))
        for i in range(n)
    ]


def _flas_train_flow_model(
    model_id: str,
    layer: int,
    samples,
    tokenizer,
    device: str = "cuda",
    concept_encoder_layers: int = 2,
    num_blocks: int = 1,
    lr: float = 5e-5,
    enc_lr: float = 1e-5,
    div_weight: float = 0.1,
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 8,
    max_len: int = 256,
    concept_max_len: int = 64,
    T_min: float = 0.5,
    T_max: float = 2.0,
    n_steps: int = 3,
    seed: int = 42,
    init_from_gemma: bool = True,
    unfreeze_concept_enc: bool = False,
    disable_cross_attn: bool = False,
    disable_self_attn: bool = False,
    disable_mlp: bool = False,
    max_steps: int | None = None,
    resume_checkpoint_path: str | None = None,
    flas_steer_once: bool = True,
    position: str = "last",
):
    """Train FLAS flow matching using the bundled HF model and concept encoder."""
    if not samples:
        raise ValueError("FLAS training requires at least one sample")

    from functools import partial
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    train_device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
    torch.manual_seed(int(seed))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(train_device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    flow_fn, concept_enc = _flas_build_flow_model(
        model_id=model_id,
        layer=layer,
        concept_encoder_layers=concept_encoder_layers,
        num_blocks=num_blocks,
        time_conditioned=True,
        init_from_gemma=init_from_gemma,
        disable_cross_attn=disable_cross_attn,
        disable_self_attn=disable_self_attn,
        disable_mlp=disable_mlp,
    )
    flow_fn.to(train_device)
    concept_enc.to(train_device)

    if resume_checkpoint_path is not None:
        logger.info(f"Loading resume checkpoint from {resume_checkpoint_path}...")
        from ..base import SteeringVector
        try:
            saved_vector = SteeringVector.load(resume_checkpoint_path, layer=[layer], device="cpu")
            saved_metadata = saved_vector.metadata or {}
            
            flow_state = _flas_load_state_dict_payload(saved_metadata, "flas_flow_fn")
            if flow_state is not None:
                flow_fn.load_state_dict(flow_state, strict=False)
                logger.info("Successfully loaded flow_fn state dict from resume checkpoint.")
            else:
                logger.warning("No flow_fn state dict found in resume checkpoint metadata.")
            
            concept_state = _flas_load_state_dict_payload(saved_metadata, "flas_concept_enc")
            if concept_state is not None:
                concept_enc.load_state_dict(concept_state, strict=False)
                logger.info("Successfully loaded concept_enc state dict from resume checkpoint.")
        except Exception as e:
            logger.error("Failed to load resume checkpoint from %s: %s", resume_checkpoint_path, e)

    if not unfreeze_concept_enc:
        for p in concept_enc.parameters():
            p.requires_grad = False
    else:
        for p in concept_enc.parameters():
            p.requires_grad = True
        for p in concept_enc.embed_tokens.parameters():
            p.requires_grad = False

    enc_params = [p for p in concept_enc.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(flow_fn.parameters()), "lr": float(lr)},
            {"params": enc_params, "lr": float(enc_lr)},
        ],
        weight_decay=0.01,
    )

    collate = partial(
        _flas_collate_prompt_output,
        tokenizer=tokenizer,
        max_len=int(max_len),
        concept_max_len=int(concept_max_len),
    )
    loader = DataLoader(
        samples,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
    )

    trainable = [p for p in flow_fn.parameters() if p.requires_grad] + enc_params
    optimizer.zero_grad(set_to_none=True)
    step_count = 0
    batch_count = 0
    last_info = {"lm_loss": None, "div_loss": None, "loss": None}

    def forward_with_flow(input_ids, attention_mask, labels, prompt_mask, concept_input_ids, concept_attention_mask, T_total):
        concept_hidden = concept_enc(concept_input_ids, concept_attention_mask)
        velocity_capture = {}
        concept_mask_f = concept_attention_mask.float()
        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            h_orig = output[0] if is_tuple else output
            h = h_orig.float()
            bsz = h.size(0)

            if position == "all":
                steer_mask = prompt_mask.bool()
            else:
                steer_mask = torch.zeros_like(prompt_mask, dtype=torch.bool)
                last_indices = prompt_mask.sum(dim=-1).long() - 1
                steer_mask[torch.arange(bsz, device=h.device), last_indices] = True

            if not flas_steer_once:
                output_mask = attention_mask.bool() & ~prompt_mask.bool()
                steer_mask = steer_mask | output_mask

            dt = T_total / max(int(n_steps), 1)
            padding_mask = steer_mask.float()
            h_target = h
            last_v = None
            for k in range(max(int(n_steps), 1)):
                t_k = torch.full((bsz,), k * dt, device=h.device)
                v, _ = flow_fn(
                    h_target,
                    concept_hidden,
                    concept_mask_f,
                    t=t_k,
                    padding_mask=padding_mask,
                )
                h_target = h_target + dt * v
                last_v = v

            velocity_capture["v"] = last_v
            velocity_capture["mask"] = steer_mask.float()
            h_out = torch.where(steer_mask.unsqueeze(-1), h_target.to(h_orig.dtype), h_orig)
            return (h_out,) + output[1:] if is_tuple else h_out

        handle = llm.model.layers[int(layer)].register_forward_hook(hook)
        try:
            outputs = llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            handle.remove()
        return outputs.loss, velocity_capture

    epochs_val = max(int(epochs), 1)
    for epoch in range(epochs_val):
        pbar = tqdm(loader, desc=f"FLAS Epoch {epoch + 1}/{epochs_val}")
        for batch in pbar:
            input_ids, attn_mask, labels, _prompt_mask, c_ids, c_mask, concept_ids = batch
            input_ids = input_ids.to(train_device)
            attn_mask = attn_mask.to(train_device)
            labels = labels.to(train_device)
            prompt_mask = _prompt_mask.to(train_device)
            c_ids = c_ids.to(train_device)
            c_mask = c_mask.to(train_device)
            concept_ids = concept_ids.to(train_device)

            T_total = float(torch.rand(1).item() * (float(T_max) - float(T_min)) + float(T_min))
            lm_loss, velocity_capture = forward_with_flow(input_ids, attn_mask, labels, prompt_mask, c_ids, c_mask, T_total)
            div_loss = _flas_diversity_loss(velocity_capture.get("v"), concept_ids, velocity_capture.get("mask")).to(train_device)
            loss = (lm_loss + float(div_weight) * div_loss) / max(int(grad_accum), 1)
            loss.backward()
            batch_count += 1

            if batch_count % max(int(grad_accum), 1) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1

            last_info = {
                "lm_loss": float(lm_loss.detach().cpu().item()),
                "div_loss": float(div_loss.detach().cpu().item()),
                "loss": float((lm_loss + float(div_weight) * div_loss).detach().cpu().item()),
            }
            pbar.set_postfix({
                "loss": f"{last_info['loss']:.4f}",
                "lm_loss": f"{last_info['lm_loss']:.4f}"
            })
            if max_steps is not None and step_count >= int(max_steps):
                break
        if max_steps is not None and step_count >= int(max_steps):
            break

    if batch_count % max(int(grad_accum), 1) != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_count += 1

    flow_fn.eval()
    concept_enc.eval()
    for p in flow_fn.parameters():
        p.requires_grad = False
    for p in concept_enc.parameters():
        p.requires_grad = False
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_info = {
        "optimizer_steps": int(step_count),
        "batches": int(batch_count),
        "epochs": int(epochs),
        "n_samples": int(len(samples)),
        "position": position,
        "flas_steer_once": bool(flas_steer_once),
        **last_info,
    }
    return flow_fn, concept_enc, train_info


def _flas_state_dict_to_cpu(module_or_state_dict):
    if hasattr(module_or_state_dict, "state_dict"):
        return {k: v.detach().cpu() for k, v in module_or_state_dict.state_dict().items()}
    if isinstance(module_or_state_dict, dict):
        return {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in module_or_state_dict.items()
        }
    raise TypeError(f"Unsupported FLAS payload type: {type(module_or_state_dict).__name__}")


def _flas_load_state_dict_payload(payload, key: str):
    if key in payload:
        value = payload[key]
        if hasattr(value, "state_dict"):
            return value.state_dict()
        return value




class FLASExtractor(BaseExtractor):
    """Train or load a FLAS flow payload for ``FLASSteerModel``."""

    METHOD_NAME = "FLAS"

    def __init__(
        self,
        model,
        layer: List[int],
        model_name: str,
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
        position: str = "last",
        flas_checkpoint_path: Optional[str] = None,
        flas_num_blocks: Optional[int] = None,
        flas_time_conditioned: bool = True,
        flas_disable_cross_attn: bool = False,
        flas_disable_self_attn: bool = False,
        flas_disable_mlp: bool = False,
        flas_strict_load: bool = True,
        flas_concept_encoder_layers: int = 2,
        flas_train_concept_text: Union[str, List[str], None] = None,
        flas_binary_class: bool = False,
        flas_train_lr: float = 5e-5,
        flas_train_enc_lr: float = 1e-5,
        flas_train_div_weight: float = 0.1,
        flas_train_epochs: int = 1,
        flas_train_batch_size: int = 4,
        flas_train_grad_accum: int = 8,
        flas_train_max_len: int = 256,
        flas_train_concept_max_len: int = 64,
        flas_train_T_min: float = 0.5,
        flas_train_T_max: float = 2.0,
        flas_train_n_steps: int = 3,
        flas_train_seed: int = 42,
        flas_train_max_steps: Optional[int] = None,
        flas_unfreeze_concept_enc: bool = False,
        flas_no_gemma_init: bool = False,
        flas_steer_once: bool = True,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        if len(layer) != 1:
            raise ValueError("FLAS ground-truth implementation steers one layer per flow checkpoint.")

        # Normalise concept_text to List[str] so downstream code never has to branch on type.
        if isinstance(flas_train_concept_text, str):
            flas_train_concept_text = [flas_train_concept_text]
        elif flas_train_concept_text is not None and not isinstance(flas_train_concept_text, list):
            raise ValueError("flas_train_concept_text must be a str, List[str], or None.")

        if flas_binary_class:
            if flas_train_concept_text is None or len(flas_train_concept_text) != 2:
                raise ValueError(
                    "flas_binary_class=True requires flas_train_concept_text to be a "
                    "list of exactly 2 concept strings: [target_concept, contrast_concept]."
                )

        self.model_name = model_name
        self.flas_checkpoint_path = str(flas_checkpoint_path) if flas_checkpoint_path is not None else None
        self.flas_num_blocks = flas_num_blocks
        self.flas_time_conditioned = flas_time_conditioned
        self.flas_disable_cross_attn = flas_disable_cross_attn
        self.flas_disable_self_attn = flas_disable_self_attn
        self.flas_disable_mlp = flas_disable_mlp
        self.flas_strict_load = flas_strict_load
        self.flas_concept_encoder_layers = flas_concept_encoder_layers
        self.flas_train_concept_text = flas_train_concept_text  # List[str] or None
        self.flas_binary_class = flas_binary_class
        self.flas_train_lr = flas_train_lr
        self.flas_train_enc_lr = flas_train_enc_lr
        self.flas_train_div_weight = flas_train_div_weight
        self.flas_train_epochs = flas_train_epochs
        self.flas_train_batch_size = flas_train_batch_size
        self.flas_train_grad_accum = flas_train_grad_accum
        self.flas_train_max_len = flas_train_max_len
        self.flas_train_concept_max_len = flas_train_concept_max_len
        self.flas_train_T_min = flas_train_T_min
        self.flas_train_T_max = flas_train_T_max
        self.flas_train_n_steps = flas_train_n_steps
        self.flas_train_seed = flas_train_seed
        self.flas_train_max_steps = flas_train_max_steps
        self.flas_unfreeze_concept_enc = flas_unfreeze_concept_enc
        self.flas_no_gemma_init = flas_no_gemma_init
        self.flas_steer_once = flas_steer_once
        self.flow_fn = None
        self.concept_enc = None

    def _get_activations(self, inputs: List[str], **kwargs):
        raise NotImplementedError("FLAS uses a learned flow checkpoint, not activation averaging.")



    def _build_training_samples(self, target_data, contrast_data, kwargs):
        target_response = kwargs.get("target_response")
        prompts = kwargs.get("prompts")

        if target_response is None:
            raise ValueError(
                "FLASExtractor requires 'target_response' (response-only strings) in kwargs. "
                "Ensure the dataset formatter emits a 'target_response' field."
            )
        if prompts is None:
            raise ValueError(
                "FLASExtractor requires 'prompts' (question-only input strings) in kwargs. "
                "Ensure the dataset formatter emits a 'question' field so the pipeline "
                "populates the 'prompts' kwarg."
            )

        concept_texts = self.flas_train_concept_text
        if concept_texts is None:
            raise ValueError(
                "FLASExtractor requires flas_train_concept_text (a str or list of 1-2 strings) "
                "set at construction time."
            )

        n = min(len(prompts), len(target_response))
        inputs = list(prompts)[:n]
        outputs = list(target_response)[:n]

        if len(concept_texts) == 1:
            # ── Single-label mode ────────────────────────────────────────────
            # All samples share one concept text and get concept_id = 0.
            return _flas_build_training_samples(
                inputs, outputs,
                [concept_texts[0]] * n, [0] * n,
            )
        else:
            # ── Binary-class mode (len == 2) ─────────────────────────────────
            # target samples   → (prompt, target_response,  concept_texts[0], 0)
            # contrast samples → (prompt, contrast_response, concept_texts[1], 1)
            # Interleaved so each batch sees both classes.
            contrast_response = kwargs.get("contrast_response")
            if contrast_response is None:
                raise ValueError(
                    "FLAS binary-class mode (len(flas_train_concept_text) == 2) requires "
                    "'contrast_response' in kwargs. Ensure the dataset formatter emits a "
                    "'contrast_response' field."
                )
            n = min(n, len(contrast_response))
            inputs = inputs[:n]
            outputs = outputs[:n]
            target_samples = _flas_build_training_samples(
                inputs, outputs,
                [concept_texts[0]] * n, [0] * n,
            )
            contrast_samples = _flas_build_training_samples(
                inputs, list(contrast_response)[:n],
                [concept_texts[1]] * n, [1] * n,
            )
            # Interleave: target[0], contrast[0], target[1], contrast[1], ...
            return [s for pair in zip(target_samples, contrast_samples) for s in pair]

    def _train_payload(self, target_data, contrast_data, kwargs):
        samples = self._build_training_samples(target_data, contrast_data, kwargs)
        num_blocks = self.flas_num_blocks if self.flas_num_blocks is not None else 1
        
        # If flas_checkpoint_path is set, let's resolve its num_blocks and metadata so we warm-start correctly
        concept_encoder_layers_resolved = self.flas_concept_encoder_layers
        if self.flas_checkpoint_path is not None:
            try:
                from ..base import SteeringVector
                saved_vector = SteeringVector.load(self.flas_checkpoint_path, layer=[int(self.layer[0])], device="cpu")
                saved_metadata = saved_vector.metadata or {}
                if "flas_num_blocks" in saved_metadata:
                    num_blocks = int(saved_metadata["flas_num_blocks"])
                if "flas_concept_encoder_layers" in saved_metadata:
                    concept_encoder_layers_resolved = int(saved_metadata["flas_concept_encoder_layers"])
            except Exception as e:
                logger.warning("Could not read metadata from flas_checkpoint_path for pre-configuration: %s", e)

        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

        flow_fn, concept_enc, train_info = _flas_train_flow_model(
            model_id=self.model_name,
            layer=int(self.layer[0]),
            samples=samples,
            tokenizer=tokenizer,
            device=str(self.device),
            concept_encoder_layers=concept_encoder_layers_resolved,
            num_blocks=int(num_blocks),
            lr=self.flas_train_lr,
            enc_lr=self.flas_train_enc_lr,
            div_weight=self.flas_train_div_weight,
            epochs=self.flas_train_epochs,
            batch_size=self.flas_train_batch_size,
            grad_accum=self.flas_train_grad_accum,
            max_len=self.flas_train_max_len,
            concept_max_len=self.flas_train_concept_max_len,
            T_min=self.flas_train_T_min,
            T_max=self.flas_train_T_max,
            n_steps=self.flas_train_n_steps,
            seed=self.flas_train_seed,
            init_from_gemma=not self.flas_no_gemma_init,
            unfreeze_concept_enc=self.flas_unfreeze_concept_enc,
            disable_cross_attn=self.flas_disable_cross_attn,
            disable_self_attn=self.flas_disable_self_attn,
            disable_mlp=self.flas_disable_mlp,
            max_steps=self.flas_train_max_steps,
            resume_checkpoint_path=self.flas_checkpoint_path,
            flas_steer_once=self.flas_steer_once,
            position=self.position,
        )
        flow_fn.to(self.device).eval()
        concept_enc.to(self.device).eval()
        self.flow_fn = flow_fn
        self.concept_enc = concept_enc
        cfg = {"model_id": self.model_name}
        return cfg, int(num_blocks), train_info

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        cfg, num_blocks, train_info = self._train_payload(target_data, contrast_data, kwargs)
        logger.info(
            "Trained FLAS flow for layer %s with %s block(s): %s",
            self.layer[0],
            num_blocks,
            train_info,
        )

        hidden_size = int(self.flow_fn.hidden_size)
        self.vector = {
            int(self.layer[0]): torch.zeros(hidden_size, device=self.device, dtype=torch.float32)
        }
        metadata = {
            "method": "FLAS",
            "flas_flow_fn_state_dict": _flas_state_dict_to_cpu(self.flow_fn),
            "flas_model_id": cfg.get("model_id", self.model_name),
            "flas_num_blocks": num_blocks,
            "flas_time_conditioned": self.flas_time_conditioned,
            "flas_disable_cross_attn": cfg.get("disable_cross_attn", self.flas_disable_cross_attn),
            "flas_disable_self_attn": cfg.get("disable_self_attn", self.flas_disable_self_attn),
            "flas_disable_mlp": cfg.get("disable_mlp", self.flas_disable_mlp),
            "flas_concept_encoder_layers": self.flas_concept_encoder_layers,
        }
        if self.flas_unfreeze_concept_enc:
            metadata["flas_concept_enc_state_dict"] = _flas_state_dict_to_cpu(self.concept_enc)

        self.metadata = metadata
        return self.vector



# =============================================================================
# FLAS extractor moved to `Steering/extractors/flas.py` to keep nonlinear methods tidy.


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


class ODEExtractor(BaseExtractor):
    """Extractor that fits an ODE-based classifier (from Code/odesteer) and
    stores the classifier payload for steering.

    The extractor fits a kernel classifier on SAE/dense activations and
    returns a placeholder steering vector while placing the classifier
    state and hyperparameters into metadata under `ode_payload`.
    """

    METHOD_NAME = "ODE"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        hook_point: List[str] | str = "post",
        classifier_type: str = "normed_poly",
        degree: int = 2,
        n_components: int = 512,
        gamma: float = 1.0,
        coef0: float = 0.1,
        sigma: float | str = "median",
        lin_clf_type: str = "lr",
        solver: str = "euler",
        steps: int = 10,
        prebuilt_classifier: Optional[Any] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.classifier_type = classifier_type
        self.degree = int(degree)
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)
        self.sigma = sigma
        self.lin_clf_type = lin_clf_type
        self.solver = solver
        self.steps = int(steps)
        self.prebuilt_classifier = prebuilt_classifier

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=self.layer,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def extract(self, target_data: List[str], contrast_data: Optional[List[str]] = None, **kwargs) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("ODEExtractor requires contrast_data for classifier fitting")

        target_acts = self._get_activations(target_data)
        source_acts = self._get_activations(contrast_data)

        vectors: Dict[int, torch.Tensor] = {}
        ode_payload: Dict[int, Dict[str, Any]] = {}

        for layer in self.layer:
            src, dst = _match_pairs(source_acts[layer], target_acts[layer])

            # Fit classifier
            if self.prebuilt_classifier is not None:
                clf = self.prebuilt_classifier
            else:
                if self.classifier_type in {"normed_poly", "poly", "poly_norm"}:
                    clf = NormedPolyClassifier(
                        degree=self.degree,
                        n_components=self.n_components,
                        gamma=self.gamma,
                        coef0=self.coef0,
                        lin_clf_type=self.lin_clf_type,
                    )
                elif self.classifier_type in {"rff", "rff_poly"}:
                    clf = RFFClassifier(n_components=self.n_components, sigma=self.sigma, lin_clf_type=self.lin_clf_type)
                else:
                    raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")

            # Fit using two-set protocol (pos, neg)
            clf.fit(dst.to(torch.float32), src.to(torch.float32))

            # Save payload
            payload = {
                "state_dict": clf.state_dict(),
                "classifier_type": self.classifier_type,
                "kernel_params": {
                    "degree": self.degree,
                    "n_components": self.n_components,
                    "gamma": self.gamma,
                    "coef0": self.coef0,
                    "sigma": self.sigma,
                    "lin_clf_type": self.lin_clf_type,
                },
                "solver": self.solver,
                "steps": self.steps,
            }

            vectors[layer] = (dst.mean(dim=0) - src.mean(dim=0)).to(self.device)
            ode_payload[layer] = payload

        self.vector = vectors
        self.metadata = {
            "method": "ODE",
            "ode_payload": ode_payload,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data),
        }
        return self.vector






class BIPOExtractor(BaseExtractor):
    """Train or load a BIPO steering vector payload for ``BIPOSteerModel``."""

    METHOD_NAME = "BIPO"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
        bipo_lr: float = 5e-4,
        bipo_beta: float = 0.1,
        bipo_epochs: int = 5,
        bipo_vector_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point)
        self.bipo_lr = bipo_lr
        self.bipo_beta = bipo_beta
        self.bipo_epochs = bipo_epochs
        self.bipo_vector_path = bipo_vector_path

    def _get_activations(self, inputs: List[str], **kwargs):
        raise NotImplementedError("BIPO uses direct preference optimization, not activation averaging.")

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        import random
        from ..utils import get_hook_name

        if self.bipo_vector_path is not None:
            # Load pre-trained vector directly
            vec_path = Path(self.bipo_vector_path)
            if not vec_path.exists():
                raise FileNotFoundError(f"BIPO vector path not found: {vec_path}")
            vec = torch.load(vec_path, map_location="cpu").to(device=self.device, dtype=self.model.cfg.dtype)
            self.vector = {l: vec for l in self.layer}
            self.metadata = {
                "method": "BIPO",
                "bipo_vector_path": str(vec_path),
            }
        return self.vector


"""Invertible Neural Network (INN) for INNSteer.

Affine coupling layers (RealNVP-style) with random permutation between blocks.
"""

import torch.nn as nn

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

class INNExtractor(BaseExtractor):
    """Train an Invertible Neural Network for nonlinear activation steering."""

    METHOD_NAME = "INNSTEER"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        hook_point: List[str] | str = "pre",
        inn_n_coupling: int = 4,
        inn_hidden_dim: int = 512,
        inn_lr: float = 1e-3,
        inn_weight_decay: float = 1e-3,
        inn_epochs: int = 300,
        inn_batch_size: int = 64,
        inn_lambda_dir: float = 1.0,
        inn_lambda_logdet: float = 0.5,
        inn_grad_clip: float = 1.0,
        inn_warmup_epochs: int = 60,
        inn_checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.inn_n_coupling = inn_n_coupling
        self.inn_hidden_dim = inn_hidden_dim
        self.inn_lr = inn_lr
        self.inn_weight_decay = inn_weight_decay
        self.inn_epochs = inn_epochs
        self.inn_batch_size = inn_batch_size
        self.inn_lambda_dir = inn_lambda_dir
        self.inn_lambda_logdet = inn_lambda_logdet
        self.inn_grad_clip = inn_grad_clip
        self.inn_warmup_epochs = inn_warmup_epochs
        self.inn_checkpoint_dir = inn_checkpoint_dir

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=kwargs.get("layers", self.layer),
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def _train_inn(
        self, src_acts: torch.Tensor, dst_acts: torch.Tensor, d_model: int
    ) -> tuple[InvertibleNN, torch.Tensor]:
        inn = InvertibleNN(
            d_model=d_model,
            n_coupling_layers=self.inn_n_coupling,
            hidden_dim=self.inn_hidden_dim,
        ).to(self.device)

        all_acts = torch.cat([src_acts, dst_acts], dim=0)
        inn.fit_actnorm(all_acts)

        n_src = src_acts.shape[0]
        dataset = TensorDataset(
            torch.cat([src_acts, dst_acts], dim=0),
            torch.cat([torch.zeros(n_src), torch.ones(dst_acts.shape[0])], dim=0),
        )
        loader = DataLoader(dataset, batch_size=self.inn_batch_size, shuffle=True)

        optimizer = AdamW(inn.parameters(), lr=self.inn_lr, weight_decay=self.inn_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.inn_epochs)

        pbar = tqdm(range(self.inn_epochs), desc="INN training")
        for epoch in pbar:
            if epoch < self.inn_warmup_epochs:
                warmup_factor = epoch / max(1, self.inn_warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = self.inn_lr * warmup_factor

            total_loss = 0.0
            total_nll = 0.0
            total_dir = 0.0
            total_logdet = 0.0
            for step, (batch_x, batch_labels) in enumerate(loader, start=1):
                batch_x = batch_x.to(self.device)
                batch_labels = batch_labels.to(self.device)
                z, log_det = inn(batch_x)

                nll_loss = 0.5 * (z ** 2).sum(dim=-1) - log_det
                nll_loss = nll_loss.mean()

                z_src = z[batch_labels < 0.5]
                z_dst = z[batch_labels > 0.5]
                if z_src.numel() > 0 and z_dst.numel() > 0:
                    mu_src = z_src.mean(dim=0)
                    mu_dst = z_dst.mean(dim=0)
                    dir_loss = -F.cosine_similarity(mu_src.unsqueeze(0), mu_dst.unsqueeze(0))
                else:
                    dir_loss = torch.tensor(0.0, device=self.device)

                logdet_mean = log_det.mean()
                logdet_var = (log_det - logdet_mean).pow(2).mean()
                logdet_loss = logdet_mean.pow(2) + logdet_var

                loss = nll_loss + self.inn_lambda_dir * dir_loss + self.inn_lambda_logdet * logdet_loss

                optimizer.zero_grad()
                loss.backward()
                if self.inn_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(inn.parameters(), self.inn_grad_clip)
                optimizer.step()
                total_loss += loss.item()
                total_nll += nll_loss.item()
                total_dir += dir_loss.item()
                total_logdet += logdet_loss.item()

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / step:.4f}",
                    "nll": f"{total_nll / step:.4f}",
                    "dir": f"{total_dir / step:.4f}",
                    "logdet": f"{total_logdet / step:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            scheduler.step()

        with torch.no_grad():
            z_src_all = inn.encode(src_acts.to(self.device))
            z_dst_all = inn.encode(dst_acts.to(self.device))
            latent_vector = (z_dst_all.mean(dim=0) - z_src_all.mean(dim=0)).cpu()

        return inn, latent_vector

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("INNExtractor requires contrast_data")

        self.layer = sorted(self.layer)
        vectors: Dict[int, torch.Tensor] = {}
        inn_state_dicts: Dict[int, Dict[str, Any]] = {}

        for layer in self.layer:
            logger.info(f"INNSteer: Processing layer {layer}")
            target_acts = self._get_activations(target_data, layers=[layer])
            source_acts = self._get_activations(contrast_data, layers=[layer])
            src = source_acts[layer].to(torch.float32)
            dst = target_acts[layer].to(torch.float32)

            n = min(src.shape[0], dst.shape[0])
            src = src[:n]
            dst = dst[:n]
            d_model = src.shape[1]

            inn, latent_vec = self._train_inn(src, dst, d_model)

            inn_state_dicts[layer] = inn.state_dict()
            vectors[layer] = latent_vec.to(self.device)

        self.vector = vectors
        self.metadata = {
            "method": "INNSTEER",
            "inn_state_dicts": inn_state_dicts,
            "inn_config": {
                "n_coupling": self.inn_n_coupling,
                "hidden_dim": self.inn_hidden_dim,
            },
        }

        if self.inn_checkpoint_dir:
            base = Path(self.inn_checkpoint_dir)
            base.mkdir(parents=True, exist_ok=True)
            for layer, sd in inn_state_dicts.items():
                torch.save(sd, base / f"inn_layer_{layer}.pt")
            logger.info(f"Saved INN checkpoints to {base}")

        return self.vector



class CobraExtractor(BaseExtractor):
    """
    COBRA: Cluster-Optimized Barycentric Representation Alignment.

    First , learns a concept subspace via SVD of per-sample contrast differences
    (isolating concept from language). Then applies k-means clustering and
    Sinkhorn OT couplings within this low-dimensional concept subspace.

    Extraction (per layer):
        1. Compute per-sample difference Δ = h_target - h_source (from paired data)
        2. SVD of Δ matrix → P_concept (top-k left singular vectors)
        3. Project all activations to concept space z = h @ P_concept
        4. K-means clustering in concept space
        5. Sinkhorn OT between source and target cluster centroids
        6. Save: P_concept, centroids in concept space, coupling, k

    Steering:
        RBF-weighted barycentric direction in concept space +
        residual (language subspace) preservation.
    """

    METHOD_NAME = "COBRA"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        device: Optional[torch.device] = None,
        hook_point: List[str] = ["pre"],
        cobra_k: int = 10,
        cobra_lambda: float = 0.1,
        cobra_tau: float = 1e-4,
        cobra_max_iter: int = 1000,
        cobra_dim: int = 8,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.cobra_k = cobra_k
        self.cobra_lambda = cobra_lambda
        self.cobra_tau = cobra_tau
        self.cobra_max_iter = cobra_max_iter
        self.cobra_dim = int(cobra_dim)

    def _get_activations(self, inputs: List[str], reduce: str = "none") -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=self.layer,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce=reduce,
            return_key_format="layer",
            pretokenize_all=False,
            change_pad_token=False,
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("COBRA requires contrast_data (source distribution)")

        from sklearn.cluster import KMeans
        import numpy as np

        target_acts_dict = self._get_activations(target_data, reduce="none")
        contrast_acts_dict = self._get_activations(contrast_data, reduce="none")

        self.vector = {}

        centroids_A_dict = {}
        centroids_B_dict = {}
        coupling_dict = {}
        k_dict = {}
        P_concept_dict = {}

        for layer in self.layer:
            X_A = contrast_acts_dict[layer].to(torch.float32)  # (N, d)
            X_B = target_acts_dict[layer].to(torch.float32)    # (N, d)

            # Fallback vector (mean diff)
            self.vector[layer] = X_B.mean(dim=0) - X_A.mean(dim=0)

            # --- Step 1: SVD concept subspace ---
            # Δ = per-sample differences (target - source)
            # With position="last" + reduce="none", each row is one sample
            Delta = X_B - X_A  # (N, d)

            if Delta.shape[0] < 2:
                k_svd = min(self.cobra_dim, X_A.shape[1])
                P_concept = torch.eye(X_A.shape[1], k_svd, device=self.device)
            else:
                U, S, Vt = torch.linalg.svd(Delta, full_matrices=False)
                cumvar = torch.cumsum(S ** 2, dim=0) / (torch.sum(S ** 2) + 1e-12)
                k_svd = max(1, int(torch.searchsorted(cumvar, 0.9).item()) + 1)
                k_svd = min(self.cobra_dim, k_svd, Delta.shape[0] - 1)
                P_concept = Vt[:k_svd].T.contiguous()  # (d, k)

            # --- Step 2: Project to concept space ---
            z_A = X_A @ P_concept  # (N, k)
            z_B = X_B @ P_concept  # (N, k)

            N_A = z_A.shape[0]
            K_actual = min(self.cobra_k, N_A)
            K_actual = max(1, K_actual)

            if K_actual == 1:
                centroids_A = z_A.mean(dim=0, keepdim=True)
                centroids_B = z_B.mean(dim=0, keepdim=True)
                w_A = torch.ones(1, device=self.device)
                w_B = torch.ones(1, device=self.device)
                P_star = torch.ones((1, 1), device=self.device)
            else:
                # KMeans on source activations in concept space
                kmeans_A = KMeans(n_clusters=K_actual, random_state=42, n_init="auto")
                labels_A = kmeans_A.fit_predict(z_A.cpu().numpy())
                centroids_A = torch.tensor(kmeans_A.cluster_centers_, dtype=torch.float32, device=self.device)
                counts_A = np.bincount(labels_A, minlength=K_actual)
                w_A = torch.tensor(counts_A / len(labels_A), dtype=torch.float32, device=self.device)

                # KMeans on target activations in concept space
                kmeans_B = KMeans(n_clusters=K_actual, random_state=42, n_init="auto")
                labels_B = kmeans_B.fit_predict(z_B.cpu().numpy())
                centroids_B = torch.tensor(kmeans_B.cluster_centers_, dtype=torch.float32, device=self.device)
                counts_B = np.bincount(labels_B, minlength=K_actual)
                w_B = torch.tensor(counts_B / len(labels_B), dtype=torch.float32, device=self.device)

                # Step 3: Sinkhorn coupling in concept space
                C = torch.cdist(centroids_A, centroids_B, p=2) ** 2
                C = C / (C.max() + 1e-12)
                K_matrix = torch.exp(-C / self.cobra_lambda)

                u = torch.ones(K_actual, dtype=torch.float32, device=self.device)
                v = torch.ones(K_actual, dtype=torch.float32, device=self.device)
                for _ in range(self.cobra_max_iter):
                    v_prev = v.clone()
                    u = w_A / (torch.matmul(K_matrix, v) + 1e-12)
                    v = w_B / (torch.matmul(K_matrix.T, u) + 1e-12)
                    if torch.norm(v - v_prev, p=1) < self.cobra_tau:
                        break
                P_star = torch.diag(u) @ K_matrix @ torch.diag(v)

            centroids_A_dict[layer] = centroids_A.cpu()
            centroids_B_dict[layer] = centroids_B.cpu()
            coupling_dict[layer] = P_star.cpu()
            k_dict[layer] = K_actual
            P_concept_dict[layer] = P_concept.cpu()

        self.metadata = {
            "method": "COBRA",
            "cobra_centroids_A": centroids_A_dict,
            "cobra_centroids_B": centroids_B_dict,
            "cobra_coupling": coupling_dict,
            "cobra_k": k_dict,
            "cobra_P_concept": P_concept_dict,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data),
        }

        return self.vector


# =============================================================================
# LQR — Activation LQR (closed-loop, input-dependent feedback)
# Lives in nonlinear.py because feedback α = λ·µ − vᵀ·z varies per sample.
# =============================================================================


class LQRExtractor(BaseExtractor):
    """Activation-LQR steering vector extractor.

    Contrastive activation extraction with offline Jacobian linearization
    and Riccati-based optimal gain synthesis.

    Stores in self.vector:  v_k  — unit-norm contrastive direction per layer
    Stores in self.metadata:
      - lqr_mu:      feature-strength scales (||e_k|| per layer)
      - lqr_kv:      precomputed K_k @ v_k (online steering vector)
      - lqr_jacobians (optional): A_k matrices for inspection
    """

    METHOD_NAME = "LQR"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        hook_point: Union[str, List[str]] = "pre",
        device: Optional[torch.device] = None,
        lqr_Q: float = 1.0,
        lqr_R: float = 1.0,
        lqr_Qf: float = 1.0,
        lqr_jac_chunk_size: int = 64,
        lqr_store_jacobians: bool = False,
    ):
        super().__init__(
            model, layer, batch_size, device,
            hook_point=hook_point, position=position,
        )
        self.lqr_Q = float(lqr_Q)
        self.lqr_R = float(lqr_R)
        self.lqr_Qf = float(lqr_Qf)
        self.lqr_jac_chunk_size = int(lqr_jac_chunk_size)
        self.lqr_store_jacobians = bool(lqr_store_jacobians)

    # ------------------------------------------------------------------
    # Activation collection
    # ------------------------------------------------------------------

    def _get_activations(
        self, inputs: List[str], **kwargs
    ) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=self.layer,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="mean",
            return_key_format="layer",
            pretokenize_all=False,
            change_pad_token=False,
        )

    def _get_per_sample_activations(
        self, inputs: List[str]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        cache: Dict[int, torch.Tensor] = {}

        def cache_hook(resid, hook):
            layer_idx = int(hook.name.split(".")[1])
            if layer_idx in self.layer:
                cache[layer_idx] = resid.detach().clone()
            return resid

        hook_names = [
            f"blocks.{l}.hook_resid_pre" for l in self.layer
        ]
        handles = []
        for hn in hook_names:
            handles.append(self.model.add_hook(hn, cache_hook))

        try:
            tokens = self.model.to_tokens(
                inputs[:1], prepend_bos=False,
            ).to(self.device)
            self.model.run_with_hooks(tokens)
        finally:
            self.model.reset_hooks()

        result: Dict[int, Dict[str, torch.Tensor]] = {}
        for l in self.layer:
            full_seq = cache[l].squeeze(0)
            result[l] = {
                "seq": full_seq,
                "last": full_seq[-1].clone(),
            }
        return result

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def _compute_block_jacobian(
        self, layer: int, seq_context: torch.Tensor
    ) -> torch.Tensor:
        block = self.model.blocks[layer]
        seq_context = seq_context.detach().to(self.device)
        d_model = seq_context.shape[-1]

        # One forward pass through block builds autograd graph
        z0 = seq_context[-1].detach().clone().requires_grad_(True)
        ctx = seq_context.clone()
        ctx[-1] = z0
        out = block(ctx.unsqueeze(0))
        y = out[0, -1, :]  # [d_model]

        # Row-by-row Jacobian via grad loop — avoids jacrev+vmap OOM
        J = torch.zeros(d_model, d_model, device=z0.device, dtype=y.dtype)
        for i in range(d_model):
            (grad,) = torch.autograd.grad(y[i], z0, retain_graph=(i < d_model - 1))
            J[i, :] = grad

        return J

    # ------------------------------------------------------------------
    # Riccati recursion
    # ------------------------------------------------------------------

    def _solve_riccati(
        self,
        jacobians: Dict[int, torch.Tensor],
        d_model: int,
    ) -> Dict[int, torch.Tensor]:
        sorted_layers = sorted(self.layer, reverse=True)
        device = jacobians[sorted_layers[0]].device

        Q = self.lqr_Q * torch.eye(d_model, device=device, dtype=torch.float32)
        R = self.lqr_R * torch.eye(d_model, device=device, dtype=torch.float32)
        S = self.lqr_Qf * torch.eye(d_model, device=device, dtype=torch.float32)

        K: Dict[int, torch.Tensor] = {}
        for k in sorted_layers:
            A = jacobians[k].to(torch.float32)
            M = S @ torch.linalg.inv(R + S) @ S
            S_next = A.mT @ (S - M) @ A + Q
            K_k = torch.linalg.inv(R + S) @ S @ A

            K[k] = K_k.detach().cpu()
            S = S_next

        return K

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        target_means = self._get_activations(target_data)
        if contrast_data is not None:
            contrast_means = self._get_activations(contrast_data)
        else:
            contrast_means = {
                l: torch.zeros_like(target_means[l]) for l in self.layer
            }

        diff = {
            l: target_means[l] - contrast_means[l] for l in self.layer
        }
        mu = {l: diff[l].norm(p=2).clamp(min=1e-12) for l in self.layer}
        v = {l: diff[l] / mu[l] for l in self.layer}
        self.vector = v

        logger.info("LQR: contrastive directions computed")

        ctx = self._get_per_sample_activations(target_data)
        logger.info("LQR: full-sequence context cached")

        d_model = target_means[self.layer[0]].shape[-1]
        jacobians: Dict[int, torch.Tensor] = {}

        self.model.eval()
        with torch.no_grad():
            ctx_device = {
                l: {k: t.to(self.device) for k, t in ctx[l].items()}
                for l in self.layer
            }

        for l in self.layer:
            logger.info(f"LQR: computing Jacobian for layer {l} ...")
            J = self._compute_block_jacobian(l, ctx_device[l]["seq"])
            jacobians[l] = J
            logger.info(f"LQR: layer {l} Jacobian ready — shape {J.shape}")
            torch.cuda.empty_cache()

        logger.info("LQR: solving Riccati recursion ...")
        K = self._solve_riccati(jacobians, d_model)
        logger.info("LQR: Riccati solved")

        kv: Dict[int, torch.Tensor] = {}
        for l in self.layer:
            v_l = v[l].to(dtype=K[l].dtype, device=K[l].device)
            kv[l] = (K[l] @ v_l).cpu()

        self.metadata = {
            "method": "LQR",
            "lqr_mu": {int(k): float(v.item()) for k, v in mu.items()},
            "lqr_kv": kv,
            "lqr_Q": self.lqr_Q,
            "lqr_R": self.lqr_R,
            "lqr_Qf": self.lqr_Qf,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data) if contrast_data else 0,
        }

        if self.lqr_store_jacobians:
            self.metadata["lqr_jacobians"] = {
                int(k): v.cpu() for k, v in jacobians.items()
            }
            self.metadata["lqr_gains"] = {
                int(k): v.cpu() for k, v in K.items()
            }

        logger.info("LQR: extraction complete")
        return self.vector


# =============================================================================
# JSPACE — Jacobian lens / J-space steering
# Fits per-layer J_ℓ matrices (average Jacobian over prompts), then projects
# steering vectors onto the J-space (sparse nonnegative combination of k
# J-lens vectors). Lives in nonlinear.py because fitting involves backprop
# through the model (like LQR), though the steer model itself is dense.
# =============================================================================

# --- helpers ----------------------------------------------------------------

SKIP_FIRST_N_POSITIONS = 16


def _valid_position_mask(seq_len: int, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first: seq_len - 1] = True
    return mask


def _atomic_save(obj: object, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --- JSpaceLens: per-layer J_ℓ matrices -------------------------------------

class JSpaceLens:
    """Fitted Jacobian lens: per-layer ``J_ℓ`` matrices.

    Each ``J_ℓ`` is a ``[d_model, d_model]`` matrix that maps an activation
    at layer ℓ into the final-layer residual-stream basis::

        lens_ℓ(h) = unembed(J_ℓ @ h)

    Fitting follows the estimator from Gurnee et al. (2026): for each output
    dimension, inject a one-hot cotangent at every valid target position and
    backprop. The gradient at source position p is then
    ``sum_{p' >= p} dh_final[p'] / dh_l[p]`` (mean over source positions).
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        d_model: int,
        n_prompts: int,
    ) -> None:
        self.jacobians = {k: v.float() for k, v in jacobians.items()}
        self.d_model = d_model
        self.n_prompts = n_prompts
        self.source_layers = sorted(self.jacobians)

    # -- transport & readout -------------------------------------------------

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        J = self.jacobians[layer].to(residual.device, residual.dtype)
        return residual @ J.T

    def j_lens_vectors(self, layer: int, unembed_W: torch.Tensor) -> torch.Tensor:
        J = self.jacobians[layer].to(unembed_W.device, unembed_W.dtype)
        return unembed_W @ J

    def j_lens_vector(self, token_id: int, layer: int, unembed_W: torch.Tensor) -> torch.Tensor:
        return self.j_lens_vectors(layer, unembed_W)[token_id].clone()

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "J": self.jacobians, "d_model": self.d_model,
            "n_prompts": self.n_prompts, "source_layers": self.source_layers,
        }, path)

    @classmethod
    def load(cls, path: str) -> "JSpaceLens":
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data["J"], d_model=data["d_model"], n_prompts=data["n_prompts"])

    # -- fitting (one-time per model) ----------------------------------------

    @classmethod
    def fit(
        cls,
        model,
        prompts: list[str],
        *,
        source_layers: list[int] | None = None,
        target_layer: int | None = None,
        dim_batch: int = 8,
        max_seq_len: int = 128,
        skip_first: int = SKIP_FIRST_N_POSITIONS,
        device: torch.device | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every: int | None = 1,
        resume: bool = True,
    ) -> "JSpaceLens":
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        dev = device or model.cfg.device

        if target_layer is None:
            target_layer = n_layers - 1
        if source_layers is None:
            source_layers = list(range(target_layer))
        source_layers = sorted(set(source_layers))
        if not source_layers or source_layers[-1] >= target_layer:
            raise ValueError(
                f"source_layers must all be < target_layer={target_layer}; "
                f"got max={max(source_layers)}"
            )

        logger.info("JSpaceLens: fitting %d layers d=%d on %d prompts target=L%d",
                     len(source_layers), d_model, len(prompts), target_layer)

        jacobian_sum: dict[int, torch.Tensor]
        n_done: int
        next_idx: int
        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            jacobian_sum, n_done, next_idx = state["jacobian_sum"], state["n_done"], state["next_idx"]
            logger.info("  resumed: %d/%d done", next_idx, len(prompts))
        else:
            jacobian_sum = {l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers}
            n_done = 0
            next_idx = 0

        def write_ckpt():
            if checkpoint_path is not None:
                _atomic_save({
                    "jacobian_sum": jacobian_sum, "n_done": n_done,
                    "next_idx": next_idx, "source_layers": source_layers,
                    "target_layer": target_layer, "skip_first": skip_first,
                }, checkpoint_path)

        sqrt_d = math.sqrt(d_model)
        for idx, prompt in enumerate(prompts):
            if idx < next_idx:
                continue
            t0 = time.perf_counter()
            try:
                per_prompt_J, seq_len, n_valid = _jacobian_for_prompt(
                    model, prompt, source_layers, target_layer=target_layer,
                    dim_batch=dim_batch, max_seq_len=max_seq_len,
                    skip_first=skip_first, device=dev,
                )
            except ValueError as e:
                logger.warning("  skipping prompt %d: %s", idx, e)
                next_idx = idx + 1
                continue
            pn = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d
            rc = max(
                ((per_prompt_J[l] - jacobian_sum[l] / n_done).norm()
                 / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())).item()
                if n_done > 0 else 0.0 for l in source_layers
            ) if n_done > 0 else float("nan")
            for l in source_layers:
                jacobian_sum[l] += per_prompt_J[l]
            n_done += 1
            next_idx = idx + 1
            logger.info("  prompt %d/%d seq=%d n_valid=%d %.0fs ||J||/√d=%.3f Δ=%.2e",
                         idx + 1, len(prompts), seq_len, n_valid,
                         time.perf_counter() - t0, pn, rc)
            if checkpoint_every is not None and next_idx % checkpoint_every == 0:
                write_ckpt()
        write_ckpt()
        if n_done == 0:
            raise ValueError("no prompts were long enough")
        mean_J = {l: jacobian_sum[l] / n_done for l in source_layers}
        logger.info("JSpaceLens: done, %d prompts", n_done)
        return cls(jacobians=mean_J, d_model=d_model, n_prompts=n_done)


# --- per-prompt Jacobian estimator ------------------------------------------

def _jacobian_for_prompt(
    model, prompt: str, source_layers: list[int], *,
    target_layer: int, dim_batch: int = 8, max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS, device: torch.device,
) -> tuple[dict[int, torch.Tensor], int, int]:
    d_model = model.cfg.d_model
    source_layers = sorted(source_layers)
    tokens = model.to_tokens(prompt, prepend_bos=True)
    seq_len = tokens.shape[1]
    if seq_len > max_seq_len:
        tokens = tokens[:, :max_seq_len]
        seq_len = max_seq_len
    mask = _valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        raise ValueError(f"prompt too short: seq_len={seq_len}")
    J = {l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers}
    n_passes = math.ceil(d_model / dim_batch)
    bi = torch.arange(dim_batch, device=device)
    hooks = [f"blocks.{l}.hook_resid_pre" for l in [*source_layers, target_layer]]
    cache: dict[int, torch.Tensor] = {}
    def _mk(layer_idx):
        def _fn(resid, hook):
            cache[layer_idx] = resid
            return resid
        return _fn
    handles = []
    for hn in hooks:
        li = int(hn.split(".")[1])
        handles.append(model.add_hook(hn, _mk(li)))
    try:
        with torch.enable_grad():
            tokens = tokens.to(device)
            rep = tokens.expand(dim_batch, -1)
            model.run_with_hooks(rep)
            target_act = cache[target_layer]
            src_acts = [cache[l] for l in source_layers]
            vp = mask.nonzero(as_tuple=True)[0].to(device)
            cot = torch.zeros_like(target_act)
            for pi, ds in enumerate(range(0, d_model, dim_batch)):
                nd = min(dim_batch, d_model - ds)
                cot.zero_()
                cot[bi[:nd, None], vp[None, :], ds + bi[:nd, None]] = 1.0
                grads = torch.autograd.grad(
                    target_act, src_acts, cot,
                    retain_graph=(pi < n_passes - 1), only_inputs=True,
                )
                for layer, g in zip(source_layers, grads):
                    pdev = vp.to(g.device, non_blocking=True)
                    rows = g[:nd, pdev, :].float().mean(dim=1)
                    J[layer][ds: ds + nd, :] = rows.cpu()
                del grads
    finally:
        model.reset_hooks()
    return J, seq_len, n_valid


# --- sparse decomposition (gradient pursuit) ---------------------------------

def gradient_pursuit(
    h: torch.Tensor,
    j_lens_vectors: torch.Tensor,
    k: int = 16,
    max_iter: int = 100,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse nonnegative decomposition of ``h`` into ``k`` J-lens vectors.

    Returns ``(coefficients [n_vocab], reconstructed [d_model])``.
    """
    nv = j_lens_vectors.shape[0]
    dev, dty = h.device, h.dtype
    res = h.clone()
    coeffs = torch.zeros(nv, device=dev, dtype=dty)
    vns = (j_lens_vectors * j_lens_vectors).sum(dim=-1).clamp(min=1e-12)
    for _ in range(min(k, max_iter)):
        sims = torch.clamp(j_lens_vectors @ res, min=0.0)
        best = torch.argmax(sims)
        if sims[best] < eps:
            break
        c = torch.clamp(sims[best] / vns[best], min=0.0)
        coeffs[best] += c
        res = res - c * j_lens_vectors[best]
    return coeffs, h - res


JSPACE_DEFAULT_LENS_DIR = "Vector/JSPACE"
JSPACE_DEFAULT_LENS_PATH = f"{JSPACE_DEFAULT_LENS_DIR}/lens.pt"

def project_jspace(vector: torch.Tensor, j_lens_vectors: torch.Tensor, k: int = 16) -> torch.Tensor:
    _, proj = gradient_pursuit(vector, j_lens_vectors, k=k)
    return proj


# --- JSpaceExtractor --------------------------------------------------------

class JSpaceExtractor(BaseExtractor):
    """J-space steering vector extractor.

    Two modes:
    - ``"project"`` (default): CAA diff-of-means → project onto J-space.
    - ``"direct"``: use the J-lens vector of a specified token directly.

    Requires pre-computed J_ℓ matrices (``jspace_lens_path``).
    """

    METHOD_NAME = "JSPACE"

    def __init__(
        self,
        model,
        layer: list[int],
        batch_size: int = 8,
        position: str = "last",
        device: torch.device | None = None,
        hook_point: str = "pre",
        jspace_mode: str = "project",
        jspace_k: int = 16,
        jspace_lens_path: str | None = None,
        jspace_target_token: str | None = None,
        jspace_dim_batch: int = 8,
        jspace_max_seq_len: int = 128,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.jspace_mode = jspace_mode
        self.jspace_k = int(jspace_k)
        self.jspace_lens_path = jspace_lens_path
        self.jspace_target_token = jspace_target_token
        self.jspace_dim_batch = int(jspace_dim_batch)
        self.jspace_max_seq_len = int(jspace_max_seq_len)
        self._lens: JSpaceLens | None = None

    def _sample_c4_prompts(self, n: int = 1000) -> list[str]:
        logger.info("JSpace: sampling %d prompts from C4 ...", n)
        from datasets import load_dataset
        ds = load_dataset("c4", "en", split="train", streaming=True)
        prompts: list[str] = []
        for example in ds:
            if len(prompts) >= n:
                break
            text = example["text"].strip()
            if len(text) < 128:
                continue
            prompts.append(text[:512])
        return prompts

    def _get_lens(self) -> JSpaceLens:
        if self._lens is not None:
            return self._lens
        lens_path = self.jspace_lens_path or JSPACE_DEFAULT_LENS_PATH
        if os.path.exists(lens_path):
            logger.info("JSpace: loading lens from %s", lens_path)
            self._lens = JSpaceLens.load(lens_path)
        else:
            logger.info("JSpace: no lens at %s, fitting from C4 prompts ...", lens_path)
            prompts = self._sample_c4_prompts()
            self._lens = JSpaceLens.fit(
                self.model, prompts, source_layers=self.layer,
                dim_batch=self.jspace_dim_batch, max_seq_len=self.jspace_max_seq_len,
                device=self.device,
            )
            os.makedirs(Path(lens_path).parent, exist_ok=True)
            self._lens.save(lens_path)
            logger.info("JSpace: lens saved to %s", lens_path)
        return self._lens

    def _get_activations(self, inputs: list[str]) -> dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model, texts=inputs, layers=self.layer,
            hook_point=self.hook_point, batch_size=self.batch_size,
            pooling=self.position, device=self.device,
            tokenizer=self.model.tokenizer, reduce="mean",
            return_key_format="layer", pretokenize_all=False, change_pad_token=False,
        )

    def extract(
        self,
        target_data: list[str],
        contrast_data: list[str] | None = None,
        **kwargs,
    ) -> dict[int, torch.Tensor]:
        if self.jspace_mode == "direct":
            return self._extract_direct()
        return self._extract_project(target_data, contrast_data)

    def _extract_direct(self) -> dict[int, torch.Tensor]:
        if self.jspace_target_token is None:
            raise ValueError("jspace_target_token required in 'direct' mode")
        tid = self.model.tokenizer.encode(self.jspace_target_token, add_special_tokens=False)
        tid = tid[0] if isinstance(tid, list) else tid
        uw = self.model.W_U.detach()
        lens = self._get_lens()
        vec: dict[int, torch.Tensor] = {}
        for l in self.layer:
            if l not in lens.source_layers:
                continue
            v = lens.j_lens_vector(tid, l, uw)
            vec[l] = v / v.norm().clamp(min=1e-12)
        self.vector = vec
        self.metadata = {"method": "JSPACE", "jspace_mode": "direct",
                         "jspace_target_token": self.jspace_target_token,
                         "jspace_k": self.jspace_k, "n_lens_prompts": lens.n_prompts}
        return self.vector

    def _extract_project(
        self, target_data: list[str], contrast_data: list[str] | None = None,
    ) -> dict[int, torch.Tensor]:
        tgt = self._get_activations(target_data)
        ctr = self._get_activations(contrast_data) if contrast_data else {l: torch.zeros_like(tgt[l]) for l in self.layer}
        diff = {l: tgt[l] - ctr[l] for l in self.layer}
        uw = self.model.W_U.detach()
        lens = self._get_lens()
        vec: dict[int, torch.Tensor] = {}
        for l in self.layer:
            if l not in lens.source_layers:
                continue
            jlv = lens.j_lens_vectors(l, uw)
            proj = project_jspace(diff[l], jlv, k=self.jspace_k)
            vec[l] = proj / proj.norm().clamp(min=1e-12)
        self.vector = vec
        self.metadata = {"method": "JSPACE", "jspace_mode": "project",
                         "jspace_k": self.jspace_k, "n_target": len(target_data),
                         "n_contrast": len(contrast_data) if contrast_data else 0,
                         "n_lens_prompts": lens.n_prompts}
        return self.vector


class IDSExtractor(BaseExtractor):
    """In-Distribution Steering (IDS) Extractor.

    Extracts steering vectors via contrastive difference-in-means,
    then fits target Mahalanobis distance boundaries in PCA-reduced subspaces.
    Reference: Vogels et al., 2025 (arXiv:2510.13285)
    """

    METHOD_NAME = "IDS"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
        ids_var_explained: float = 0.40,
        ids_epsilon_pct: float = 0.95,
        ids_f1_threshold: float = 0.70,
        ids_ot_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.ids_var_explained = float(ids_var_explained)
        self.ids_epsilon_pct = float(ids_epsilon_pct)
        self.ids_f1_threshold = float(ids_f1_threshold)
        self.ids_ot_eps = float(ids_ot_eps)

    def _get_activations(self, inputs: List[str], layers: List[int]) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=layers,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
            pretokenize_all=False,
            change_pad_token=False,
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("IDS Extractor requires contrast_data (source distribution)")

        self.layer = sorted(self.layer)
        self.vector = {}
        layer_stats = {}

        logger.info("Collecting target activations...")
        target_acts = self._get_activations(target_data, self.layer)
        logger.info("Collecting contrast activations...")
        contrast_acts = self._get_activations(contrast_data, self.layer)

        for layer in self.layer:
            tgt = target_acts[layer].to(torch.float32)
            contr = contrast_acts[layer].to(torch.float32)

            n = min(tgt.shape[0], contr.shape[0])
            tgt = tgt[:n]
            contr = contr[:n]

            v_l = tgt.mean(dim=0) - contr.mean(dim=0)
            self.vector[layer] = v_l

            proj_tgt = tgt @ v_l
            proj_contr = contr @ v_l

            projs = torch.cat([proj_contr, proj_tgt])
            labels = torch.cat([torch.zeros_like(proj_contr), torch.ones_like(proj_tgt)])

            sorted_projs, idx = torch.sort(projs)
            sorted_labels = labels[idx]

            tp = sorted_labels.sum() - torch.cumsum(sorted_labels, dim=0)
            fp = (1 - sorted_labels).sum() - torch.cumsum(1 - sorted_labels, dim=0)
            fn = torch.cumsum(sorted_labels, dim=0)

            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
            max_f1 = float(f1_scores.max().item())

            logger.info(f"Layer {layer} Steering Vector F1-Score: {max_f1:.4f}")

            import numpy as np

            union_acts = torch.cat([contr, tgt], dim=0).cpu().numpy()
            pca_full = PCA()
            pca_full.fit(union_acts)
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)

            n_components = np.argmax(cum_var >= self.ids_var_explained) + 1
            n_components = max(1, min(n_components, union_acts.shape[0], union_acts.shape[1]))

            pca = PCA(n_components=n_components)
            pca.fit(union_acts)

            P_layer = torch.from_numpy(pca.components_.T).to(torch.float32)
            mu_layer = torch.from_numpy(pca.mean_).to(torch.float32)

            z_tgt = (tgt.cpu() - mu_layer) @ P_layer
            mu_tgt_pca = z_tgt.mean(dim=0)
            z_tgt_centered = z_tgt - mu_tgt_pca
            cov_tgt_pca = (z_tgt_centered.T @ z_tgt_centered) / max(1, n - 1)

            cov_tgt_pca = cov_tgt_pca + self.ids_ot_eps * torch.eye(n_components)

            L = torch.linalg.cholesky(cov_tgt_pca)
            L_inv = torch.linalg.solve_triangular(L, torch.eye(n_components), upper=False)

            normalized_diff = z_tgt_centered @ L_inv.T
            dists_sq = torch.sum(normalized_diff ** 2, dim=-1)
            epsilon_sq = float(np.percentile(dists_sq.numpy(), self.ids_epsilon_pct * 100))

            layer_stats[layer] = {
                "pca_components": P_layer,
                "pca_mean": mu_layer,
                "mu_tgt_pca": mu_tgt_pca,
                "L_inv": L_inv,
                "epsilon_sq": epsilon_sq,
                "f1_score": max_f1,
            }

        self.metadata = {
            "method": self.METHOD_NAME,
            "ids_var_explained": self.ids_var_explained,
            "ids_epsilon_pct": self.ids_epsilon_pct,
            "ids_f1_threshold": self.ids_f1_threshold,
            "ids_ot_eps": self.ids_ot_eps,
            "layer_stats": layer_stats,
            "n_samples": min(target_acts[self.layer[0]].shape[0], contrast_acts[self.layer[0]].shape[0]),
        }

        return self.vector


# =============================================================================
# FishBack — Flow Matching + OT Loss + Task-Contrastive Fisher Subspace
# =============================================================================
#
# Old FishBack used next-token Fisher (0% on safety tasks). This replaces it
# with flow-based transport in a task-contrastive behavioral subspace.
#
# Components:
#   1. Task-contrastive Fisher subspace: directions that differentiate
#      harmful from safe activations (not variance, not next-token)
#   2. Flow matching: MLP velocity field for nonlinear safe→harmful transport
#   3. OT loss: Sinkhorn distance in Fisher subspace (distribution geometry)
#   4. Jacobian regularization: penalize velocity norm variance (smooth transport)



def compute_fisher_subspace(
    acts_harmful: torch.Tensor,
    acts_safe: torch.Tensor,
    k: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Task-contrastive Fisher subspace (Variation B from plan.md).

    G = Cov(∇_{h} L) where L = ||h_harmful - h_safe||²
    → top-k eigenvectors = directions that differentiate harmful from safe.

    Returns:
        R: [k, d_model] — projection matrix
        eigenvalues: [k] — eigenvalues (importance weights)
    """
    diff = acts_harmful - acts_safe
    G = (diff.T @ diff) / diff.shape[0]  # [d, d] outer product covariance

    eigenvalues, eigenvectors = torch.linalg.eigh(G)
    # eigh returns ascending order; take the k largest
    R = eigenvectors.T[-k:]  # [k, d]
    return R, eigenvalues[-k:]


def sinkhorn_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 20,
) -> torch.Tensor:
    """Differentiable Sinkhorn approximation to Wasserstein-2 distance.

    Uses log-domain iterations for numerical stability (high-dim activations
    produce large cost matrices that overflow standard Sinkhorn).

    Iterates are DETACHED: gradients flow only through the final cost,
    preventing deep computation graphs through the iteration loop.
    """
    C = torch.cdist(x, y, p=2)  # [n, m]
    log_K = -C / epsilon

    # Log-domain Sinkhorn with detached iterates
    log_u = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    log_v = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)

    for _ in range(n_iters):
        log_u = -torch.logsumexp(log_K + log_v.detach().unsqueeze(0), dim=1)
        log_v = -torch.logsumexp(log_K.T + log_u.detach().unsqueeze(0), dim=1)

    # Transport plan and cost (gradients flow through C here)
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = torch.exp(log_P)
    return (P * C).sum()


class FishBackExtractor(BaseExtractor):
    """Flow matching + OT + Fisher-subspace extractor.

    Trains a velocity-field MLP to transport safe activations → harmful,
    with Sinkhorn OT loss in a task-contrastive Fisher subspace.

    The stored vector is the CAA mean-difference in Fisher subspace
    (for compatibility); the real steering uses the trained flow model
    at inference time via FishBackSteerModel.
    """

    METHOD_NAME = "FISHBACK"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        hook_point: List[str] = ["pre"],
        position: Union[str, int] = "last",
        fb_hidden_dim: int = 128,
        fb_n_layers: int = 2,
        fb_lr: float = 1e-3,
        fb_weight_decay: float = 1e-4,
        fb_epochs: int = 100,
        fb_grad_clip: float = 1.0,
        fb_n_steps: int = 10,
        fb_max_grad_len: int = 32,
        **kwargs,
    ):
        super().__init__(
            model, layer, batch_size, device,
            hook_point=hook_point, position=position,
        )
        self.fb_hidden_dim = fb_hidden_dim
        self.fb_n_layers = fb_n_layers
        self.fb_lr = fb_lr
        self.fb_weight_decay = fb_weight_decay
        self.fb_epochs = fb_epochs
        self.fb_grad_clip = fb_grad_clip
        self.fb_n_steps = fb_n_steps
        self.fb_max_grad_len = fb_max_grad_len

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=kwargs.get("layers", self.layer),
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def _train_flow(
        self,
        src: torch.Tensor,       # safe activations  [n, d] (CPU)
        dst: torch.Tensor,       # harmful activations [n, d] (CPU)
        target_prompts: List[str],  # original harmful prompts
        target_layer: int,       # layer to inject at
        **kwargs,
    ) -> FlowMLP:
        """Train velocity field with VGG-Flow style gradient guidance (plan.md).

        Hooks model at the user question prompt boundary and aligns target response tokens.
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR
        import gc

        d_model = src.shape[-1]
        n_samples = src.shape[0]

        GRAD_BATCH = 2  # Micro-batch size 2 to keep peak VRAM <6.2 GB and eliminate OOM risk
        REFRESH_FREQ = 1  # Refresh VGG-Flow target gradients every epoch (no staleness)

        raw_prompts = kwargs.get("prompts")
        raw_responses = kwargs.get("target_response")

        # Aggressively free cached memory
        gc.collect()
        torch.cuda.empty_cache()

        flow = FlowMLP(d_model, self.fb_hidden_dim, self.fb_n_layers).to(self.device)
        opt = AdamW(flow.parameters(), lr=self.fb_lr, weight_decay=self.fb_weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.fb_epochs, eta_min=self.fb_lr * 0.01)

        hook_name = get_hook_name(target_layer, self.hook_point[0])

        # Buffer to cache target tuples (h_t, t, v_target) between refresh epochs
        cached_targets: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        pbar = tqdm(range(self.fb_epochs), desc=f"FishBack L{target_layer}: VGG-Flow training")
        for epoch in pbar:
            is_refresh_epoch = (epoch % REFRESH_FREQ == 0)
            epoch_loss = 0.0
            epoch_raw_gn = 0.0
            epoch_targ_norm = 0.0
            n_batches = 0

            if is_refresh_epoch:
                perm = torch.randperm(n_samples)
                cached_targets.clear()

                for i in range(0, n_samples, GRAD_BATCH):
                    idx = perm[i : i + GRAD_BATCH]
                    bs = idx.shape[0]

                    # 1. Transfer batch safe & harmful activations to GPU
                    h_s = src[idx].to(self.device)
                    h_d = dst[idx].to(self.device)

                    # 2. Sample t ~ U(0,1) and interpolate h_t
                    t = torch.rand(bs, device=self.device)
                    h_t = (1 - t.unsqueeze(-1)) * h_s + t.unsqueeze(-1) * h_d

                    # 3. Predict velocity from flow MLP
                    v_pred = flow(h_t, t)

                    # 4. Predicted final state with stop-gradient
                    h_predicted = (h_t + (1 - t.unsqueeze(-1)) * v_pred.detach()).detach().requires_grad_(True)

                    # 5. Tokenize user question prompt + target response from kwargs
                    batch_full_ids = []
                    prompt_end_indices = []

                    for b_idx, idx_val in enumerate(idx.tolist()):
                        p_text = raw_prompts[idx_val] if raw_prompts is not None else target_prompts[idx_val]
                        r_text = raw_responses[idx_val] if raw_responses is not None else ""
                        formatted_p = self.model.tokenizer.apply_chat_template(
                            [{"role": "user", "content": p_text}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        p_ids = self.model.tokenizer.encode(formatted_p)
                        r_ids = self.model.tokenizer.encode(r_text, add_special_tokens=False)[: self.fb_max_grad_len]
                        p_idx = len(p_ids) - 1
                        full_ids = p_ids + r_ids

                        batch_full_ids.append(torch.tensor(full_ids, dtype=torch.long))
                        prompt_end_indices.append(p_idx)

                    pad_id = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id is not None else 0
                    sub_ids = torch.nn.utils.rnn.pad_sequence(
                        batch_full_ids, batch_first=True, padding_value=pad_id
                    ).to(self.device)
                    sub_mask = (sub_ids != pad_id).long().to(self.device)

                    # 6. Inject h_predicted at prompt end token index (exact location matching inference steer)
                    def _inject(activation, hook, hp=h_predicted):
                        torch.set_grad_enabled(True)
                        act_copy = activation.clone()
                        for b_idx in range(act_copy.shape[0]):
                            act_copy[b_idx, prompt_end_indices[b_idx], :] = hp[b_idx].to(activation.dtype)
                        return act_copy

                    torch.set_grad_enabled(True)
                    logits = self.model.run_with_hooks(
                        sub_ids, attention_mask=sub_mask,
                        fwd_hooks=[(hook_name, _inject)],
                        return_type="logits",
                    )

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = sub_ids[:, 1:].contiguous()

                    # Mask out all question prompt tokens so CE loss evaluates ONLY on target response tokens
                    response_mask = sub_mask[:, 1:].clone()
                    for b_idx in range(bs):
                        response_mask[b_idx, : prompt_end_indices[b_idx]] = 0

                    ce_flat = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1), reduction="none",
                    )
                    ce_per_sample = (ce_flat.view_as(response_mask) * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
                    ce_loss = ce_per_sample.sum()

                    # 7. Compute VGG-Flow target gradient: v_target = ∇_{h_predicted} CE
                    ce_loss.backward()
                    v_target = h_predicted.grad.clone().detach()
                    h_predicted.grad = None
                    self.model.zero_grad()

                    raw_gn = v_target.norm(dim=-1).mean().item()

                    # Scale-preserved guidance: normalize direction & match physical activation displacement scale
                    target_disp_norm = (h_d - h_s).norm(dim=-1, keepdim=True)
                    v_grad_norm = v_target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    v_target = (v_target / v_grad_norm) * target_disp_norm
                    targ_norm = v_target.norm(dim=-1).mean().item()

                    # Cache target tuple for intermediate epochs
                    cached_targets[n_batches] = (h_t.detach(), t.detach(), v_target.detach())

                    # 8. Train flow matching MLP
                    loss = F.mse_loss(v_pred, v_target)
                    opt.zero_grad()
                    loss.backward()
                    if self.fb_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(flow.parameters(), self.fb_grad_clip)
                    opt.step()

                    epoch_loss += loss.item()
                    epoch_raw_gn += raw_gn
                    epoch_targ_norm += targ_norm
                    n_batches += 1

                    del h_s, h_d, logits, ce_loss, sub_ids, sub_mask, shift_logits, shift_labels, response_mask
                    torch.cuda.empty_cache()

            else:
                # Fast intermediate epochs using cached target gradients (no LLM forward/backward!)
                for batch_idx in range(len(cached_targets)):
                    h_t, t, v_target = cached_targets[batch_idx]
                    v_pred = flow(h_t, t)
                    loss = F.mse_loss(v_pred, v_target)

                    opt.zero_grad()
                    loss.backward()
                    if self.fb_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(flow.parameters(), self.fb_grad_clip)
                    opt.step()

                    epoch_loss += loss.item()
                    epoch_raw_gn += v_target.norm(dim=-1).mean().item()
                    epoch_targ_norm += v_target.norm(dim=-1).mean().item()
                    n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_raw_gn = epoch_raw_gn / max(n_batches, 1)
            avg_targ_norm = epoch_targ_norm / max(n_batches, 1)
            lr = opt.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{avg_loss:.4e}", raw_gn=f"{avg_raw_gn:.4e}", targ_norm=f"{avg_targ_norm:.4e}", lr=f"{lr:.1e}")
            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(f"FishBack epoch {epoch+1}/{self.fb_epochs}: loss={avg_loss:.4e} raw_gn={avg_raw_gn:.4e} targ_norm={avg_targ_norm:.4e} lr={lr:.1e}")
            sched.step()

        cached_targets.clear()
        gc.collect()
        torch.cuda.empty_cache()

        return flow

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("FishBackExtractor requires contrast_data")

        import gc

        self.layer = sorted(self.layer)
        vectors: Dict[int, torch.Tensor] = {}
        flow_state_dicts: Dict[int, Dict[str, Any]] = {}

        for lyr in self.layer:
            logger.info(f"FishBack: Processing layer {lyr}")
            target_acts = self._get_activations(target_data, layers=[lyr])
            source_acts = self._get_activations(contrast_data, layers=[lyr])
            src = source_acts[lyr].to(torch.float32)
            dst = target_acts[lyr].to(torch.float32)

            n = min(src.shape[0], dst.shape[0])
            src, dst = src[:n].cpu(), dst[:n].cpu()
            prompts = target_data[:n]

            # Explicitly free activation dicts before training flow model
            del target_acts, source_acts
            gc.collect()
            torch.cuda.empty_cache()

            # Train flow model with VGG-Flow gradient guidance (plan.md)
            flow = self._train_flow(src, dst, prompts, lyr, **kwargs)

            # CAA vector in full space (for pipeline compatibility / d_model inference)
            with torch.no_grad():
                vectors[lyr] = (dst.mean(0) - src.mean(0)).cpu()

            flow_state_dicts[lyr] = flow.state_dict()
            del flow, src, dst
            gc.collect()
            torch.cuda.empty_cache()

        self.vector = vectors
        self.metadata = {
            "method": "FISHBACK",
            "flow_state_dicts": flow_state_dicts,
            "flow_config": {
                "hidden_dim": self.fb_hidden_dim,
                "n_layers": self.fb_n_layers,
                "n_steps": self.fb_n_steps,
            },
        }
        return self.vector


# =============================================================================
# Geometric INN Extractor — Path-Cost Regularized Invertible Mapping
# =============================================================================
#
# Key differences from INNExtractor (old INNSteer):
#   1. Contrastive loss (max-margin) instead of cosine similarity
#   2. Path cost in MAPPING: Jacobian smoothness so linear paths in z-space
#      correspond to smooth, meaningful paths in h-space
#   3. Fisher regularization: local curvature from model's own gradients
#   4. Smaller architecture (4-6 layers, 512 dim) to avoid overfitting
#   5. Separate INN per concept (each call trains one concept)
#   6. Steering: CAA in latent space (no joint optimization)
#
# Loss composition:
#   L = L_nll + λ_sep·L_separation + λ_path·L_path + λ_fisher·L_fisher
#
# L_path = mean ||J(z)·d||² for random directions d
#   → Penalizes Jacobian variation, so straight lines in z map to smooth paths in h
#
# L_fisher = mean ||J(z) - J_avg||²_F
#   → Regularizes Jacobian to be locally constant (uniform geometry)

import torch.nn.functional as F_nn


class GeometricInvertibleNN(nn.Module):
    """Invertible NN with path-cost regularization for activation steering.
    
    Architecture: RealNVP-style affine coupling layers (same as InvertibleNN)
    but with additional forward methods for computing path cost.
    """

    def __init__(
        self,
        d_model: int,
        n_coupling_layers: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        assert d_model >= 2 and d_model % 2 == 0, "d_model must be even and >= 2"
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




class GINNExtractor(BaseExtractor):
    """Geometric INN extractor with path-cost regularization.
    
    Trains an invertible mapping φ such that:
    - Contrastive clusters are well-separated in latent space
    - Linear paths in latent space correspond to smooth paths in activation space
    - Jacobian geometry is locally uniform (Fisher regularization)
    
    Then stores the trained INN and computes CAA vector in latent space.
    
    Key improvement over INNExtractor:
    - Path cost in mapping (not steering) — computed once during training
    - Contrastive separation loss (not cosine similarity)
    - Fisher regularization connecting to model geometry
    """

    METHOD_NAME = "GINN"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        hook_point: List[str] | str = "pre",
        # INN architecture
        ginn_n_coupling: int = 4,
        ginn_hidden_dim: int = 512,
        # Training
        ginn_lr: float = 1e-3,
        ginn_weight_decay: float = 1e-4,
        ginn_epochs: int = 100,
        ginn_batch_size: int = 64,
        ginn_grad_clip: float = 1.0,
        # Loss weights
        ginn_lambda_sep: float = 2.0,      # Contrastive separation
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.ginn_n_coupling = ginn_n_coupling
        self.ginn_hidden_dim = ginn_hidden_dim
        self.ginn_lr = ginn_lr
        self.ginn_weight_decay = ginn_weight_decay
        self.ginn_epochs = ginn_epochs
        self.ginn_batch_size = ginn_batch_size
        self.ginn_grad_clip = ginn_grad_clip
        self.ginn_lambda_sep = ginn_lambda_sep
    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=kwargs.get("layers", self.layer),
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def _train_geometric_inn(
        self, src_acts: torch.Tensor, dst_acts: torch.Tensor, d_model: int
    ) -> Tuple[GeometricInvertibleNN, torch.Tensor]:
        """Train Geometric INN with contrastive + NLL losses."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        inn = GeometricInvertibleNN(
            d_model=d_model,
            n_coupling_layers=self.ginn_n_coupling,
            hidden_dim=self.ginn_hidden_dim,
        ).to(self.device)

        # Move training data to GPU and initialize actnorm
        all_acts = torch.cat([src_acts, dst_acts], dim=0).to(self.device)
        labels = torch.cat([
            torch.zeros(src_acts.shape[0], device=self.device),
            torch.ones(dst_acts.shape[0], device=self.device)
        ], dim=0)
        inn.fit_actnorm(all_acts)

        optimizer = AdamW(inn.parameters(), lr=self.ginn_lr, weight_decay=self.ginn_weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.ginn_epochs, eta_min=self.ginn_lr * 0.01)

        pbar = tqdm(range(self.ginn_epochs), desc="GINN training")
        for epoch in pbar:
            # Direct GPU-side shuffling and batching (much faster than DataLoader)
            perm = torch.randperm(all_acts.shape[0], device=self.device)
            epoch_loss = 0.0
            n_steps = 0
            
            for i in range(0, all_acts.shape[0], self.ginn_batch_size):
                idx = perm[i : i + self.ginn_batch_size]
                batch_x = all_acts[idx]
                batch_labels = labels[idx]

                z, log_det = inn(batch_x)
                nll_loss = 0.5 * (z ** 2).sum(dim=-1).mean() - log_det.mean()

                z_src = z[batch_labels < 0.5]
                z_dst = z[batch_labels > 0.5]
                if z_src.numel() > 0 and z_dst.numel() > 0:
                    # Stabilized separation distance (added epsilon to prevent NaN gradient at 0)
                    sep_loss = -torch.sqrt(torch.sum((z_dst.mean(dim=0) - z_src.mean(dim=0)) ** 2) + 1e-12)
                else:
                    sep_loss = torch.tensor(0.0, device=self.device)

                loss = nll_loss + self.ginn_lambda_sep * sep_loss

                optimizer.zero_grad()
                loss.backward()
                if self.ginn_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(inn.parameters(), self.ginn_grad_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_steps += 1
                del nll_loss, sep_loss, loss, z, log_det

            avg_loss = epoch_loss / max(n_steps, 1)
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{avg_loss:.2f}", "lr": f"{lr:.1e}"})

            if (epoch + 1) % 50 == 0 or epoch == 0:
                logger.info(f"GINN epoch {epoch+1}/{self.ginn_epochs}: loss={avg_loss:.2f} lr={lr:.1e}")

            scheduler.step()

        # Compute CAA vector in latent space
        with torch.no_grad():
            z_src_all = inn.encode(src_acts.to(self.device))
            z_dst_all = inn.encode(dst_acts.to(self.device))
            latent_vector = (z_dst_all.mean(dim=0) - z_src_all.mean(dim=0)).cpu()

        return inn, latent_vector

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Extract Geometric INN steering vector.
        
        Args:
            target_data: Prompts for target concept (e.g., Evil)
            contrast_data: Prompts for contrast concept (e.g., Safe)
            kwargs: Additional fields (unused for this method)
        """
        if contrast_data is None:
            raise ValueError("GeometricINNExtractor requires contrast_data")

        self.layer = sorted(self.layer)
        vectors: Dict[int, torch.Tensor] = {}
        inn_state_dicts: Dict[int, Dict[str, Any]] = {}

        for layer in self.layer:
            logger.info(f"GINN: Processing layer {layer}")
            target_acts = self._get_activations(target_data, layers=[layer])
            source_acts = self._get_activations(contrast_data, layers=[layer])
            src = source_acts[layer].to(torch.float32)
            dst = target_acts[layer].to(torch.float32)

            n = min(src.shape[0], dst.shape[0])
            src = src[:n]
            dst = dst[:n]
            d_model = src.shape[1]

            inn, latent_vec = self._train_geometric_inn(src, dst, d_model)

            inn_state_dicts[layer] = inn.state_dict()
            vectors[layer] = latent_vec.to(self.device)

        self.vector = vectors
        self.metadata = {
            "method": "GINN",
            "inn_state_dicts": inn_state_dicts,
            "inn_config": {
                "n_coupling": self.ginn_n_coupling,
                "hidden_dim": self.ginn_hidden_dim,
            },
        }

        return self.vector
