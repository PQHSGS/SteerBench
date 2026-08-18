"""
SAE-based steering model implementations.

Contains steer models that apply steering in Sparse Autoencoder latent space.
"""

from typing import List, Optional, Dict, Any, Union
from functools import partial

import torch
import torch.nn.functional as F
from ..base import BaseSteerModel, BaseSAESteerModel
from ..logger import setup_logger
from ..utils import get_resid_acts, set_resid_acts
logger = setup_logger(__name__)



class SASSteerModel(BaseSAESteerModel):
    """
    Wrapper for SAE-based sparse steering.

    Applies steering in SAE latent space, then decodes back.
    Used with: SAS
    """
    def hook_fn(
        self,
        resid: torch.Tensor,
        sae,
        sparse_latent,
        position: Union[int, str] = "last",
        coeff: float = 1.0,
        top_idx=None,
        hook=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply steering in SAE latent space.

        Implements: a' = decode(σ(encode(a) + λv)) + (a - decode(encode(a)))
        where σ matches the SAE's activation function (Paper Algorithm 2).
        """
        original_resid = get_resid_acts(resid, position)   

        # Encode to sparse space
        sparse = sae.encode(original_resid)
        # Sparse-space editing uses extractor-provided sparse_latent metadata.
        vec = sparse_latent
        if getattr(vec, "is_sparse", False):
            vec = vec.to_dense()
        vec = torch.as_tensor(vec, device=sparse.device, dtype=sparse.dtype)

        if top_idx is None:
            raise ValueError(
                "SAS steering requires `top_idx` in SteeringVector metadata. "
                "Re-run extraction or load a vector file that includes top_idx."
            )

        flat_vec = vec.reshape(-1)
        ranked = torch.as_tensor(top_idx, device=flat_vec.device, dtype=torch.long).reshape(-1)
        selected_idx = self.select_top_k_indices(ranked).to(device=flat_vec.device)
        selected_idx = selected_idx[(selected_idx >= 0) & (selected_idx < flat_vec.numel())]
        if selected_idx.numel() == 0:
            raise ValueError(
                f"No valid SAS feature indices selected from top_idx for latent size {flat_vec.numel()}"
            )

        filtered_vec = torch.zeros_like(flat_vec)
        filtered_vec[selected_idx] = flat_vec[selected_idx]
        vec = filtered_vec.reshape_as(vec)

        steered_sparse = sparse + coeff * vec

        steered_sparse = F.relu(steered_sparse)

        # Decode
        steered_decoded = sae.decode(steered_sparse)

        # Preserve non-SAE component (Δ = a - decode(encode(a)))
        original_decoded = sae.decode(sparse)
        residual = original_resid - original_decoded

        # Final output
        resid = set_resid_acts(resid, position, steered_decoded + residual)

        if self._prompt_metadata is None:
            self._reset_prompt_metadata()

        pre_vals = sparse[..., selected_idx]
        post_vals = steered_sparse[..., selected_idx]
        if pre_vals.dim() == 3:
            pre_vals = pre_vals[0, -1, :]
            post_vals = post_vals[0, -1, :]
        elif pre_vals.dim() == 2:
            pre_vals = pre_vals[0, :]
            post_vals = post_vals[0, :]
        pre_vals = pre_vals.detach().float().cpu().tolist()
        post_vals = post_vals.detach().float().cpu().tolist()

        self._prompt_metadata.setdefault("steering_idx", selected_idx.detach().cpu().tolist())
        self._prompt_metadata.setdefault("pre_steer_top_idx_acts", pre_vals)
        self._prompt_metadata.setdefault("post_steer_top_idx_acts", post_vals)

        return resid


class SRESteerModel(BaseSAESteerModel):
    """
    Wrapper for SRE-style sparse steering.

    Enhances I+ features and suppresses I- features.
    Used with: SRE  
    """

    def __init__(
        self,
        model,
        layer: List[int],
        sae: Dict[int, Any],
        steering_vector: Dict[int, torch.Tensor],
        I_plus: Optional[torch.Tensor] = None,
        I_minus: Optional[torch.Tensor] = None,
        target_sparse: Optional[torch.Tensor] = None,
        contrast_sparse: Optional[torch.Tensor] = None,
        sparse_latent: Optional[Union[Dict[int, torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Args:
            model: Language model
            layer: Target layer(s) as List[int]
            sae: Dict[int, SAE] keyed by layer
            steering_vector: Dict[int, Tensor] keyed by layer
            I_plus: Indices of features to enhance
            I_minus: Indices of features to suppress
            target_sparse: Target sparse activations
            contrast_sparse: Contrast sparse activations (optional, for reference)
        """
        super().__init__(model, layer, sae, steering_vector, **kwargs)
        self.I_plus = I_plus if I_plus is not None else torch.tensor([], dtype=torch.long)
        self.I_minus = I_minus if I_minus is not None else torch.tensor([], dtype=torch.long)
        # Prefer explicit target_sparse metadata; otherwise fall back to sparse_latent.
        if target_sparse is not None:
            self.target_sparse = target_sparse
        elif isinstance(sparse_latent, dict) and self.layer[0] in sparse_latent:
            self.target_sparse = sparse_latent[self.layer[0]]
        elif sparse_latent is not None:
            self.target_sparse = sparse_latent
        else:
            raise ValueError(
                "SRESteerModel requires `target_sparse` (or `sparse_latent`) metadata."
            )

    def hook_fn(
        self,
        resid: torch.Tensor,
        position: Union[int, str],
        coeff: float,
        steering_vector: torch.Tensor,
        sae,
        top_idx=None,
        hook=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply SRE steering: enhance I+, suppress I-.

        z'[j] = z[j] + k * mean(z_target[j]) if j in I+
        z'[j] = 0                            if j in I-
        """
        original_resid = get_resid_acts(resid, position)

        # Encode
        sparse = sae.encode(original_resid)
        steered_sparse = sparse.clone()

        target_sparse = self.target_sparse
        if getattr(target_sparse, "is_sparse", False):
            target_sparse = target_sparse.to_dense()
        target_sparse = torch.as_tensor(target_sparse, device=sparse.device, dtype=sparse.dtype)
        idx_plus = torch.as_tensor(self.I_plus, device=sparse.device, dtype=torch.long)
        idx_minus = torch.as_tensor(self.I_minus, device=sparse.device, dtype=torch.long)

        if top_idx is None:
            raise ValueError(
                "SRE steering requires `top_idx` in SteeringVector metadata. "
                "Re-run extraction or load a vector file that includes top_idx."
            )

        ranked = torch.as_tensor(top_idx, device=sparse.device, dtype=torch.long).reshape(-1)
        selected = self.select_top_k_indices(ranked).to(device=sparse.device)
        selected = selected[(selected >= 0) & (selected < sparse.shape[-1])]
        if selected.numel() == 0:
            raise ValueError(
                f"No valid SRE feature indices selected from top_idx for latent size {sparse.shape[-1]}"
            )

        idx_plus = idx_plus[torch.isin(idx_plus, selected)]
        idx_minus = idx_minus[torch.isin(idx_minus, selected)]

        # Enhance I+ features
        if idx_plus.numel() > 0:
            steered_sparse[:, idx_plus] += (
                coeff * target_sparse[idx_plus]
            )

        # Suppress I- features
        if idx_minus.numel() > 0:
            steered_sparse[:, idx_minus] = 0.0

        # Decode
        resid = set_resid_acts(resid, position, sae.decode(steered_sparse))

        if self._prompt_metadata is None:
            self._reset_prompt_metadata()

        pre_vals = sparse[..., selected]
        post_vals = steered_sparse[..., selected]
        if pre_vals.dim() == 3:
            pre_vals = pre_vals[0, -1, :]
            post_vals = post_vals[0, -1, :]
        elif pre_vals.dim() == 2:
            pre_vals = pre_vals[0, :]
            post_vals = post_vals[0, :]
        pre_vals = pre_vals.detach().float().cpu().tolist()
        post_vals = post_vals.detach().float().cpu().tolist()

        self._prompt_metadata.setdefault("steering_idx", selected.detach().cpu().tolist())
        self._prompt_metadata.setdefault("pre_steer_top_idx_acts", pre_vals)
        self._prompt_metadata.setdefault("post_steer_top_idx_acts", post_vals)

        return resid

class SSVSteerModel(BaseSAESteerModel):
    """
    Wrapper for SSV steering with language modeling loss consideration.

    Applies steering in SAE latent space with optional LM loss weighting.
    Used with: SSV
    """

    def hook_fn(
        self,
        resid: torch.Tensor,
        position: Union[int, str],
        coeff: float,
        steering_vector: torch.Tensor,
        sae,
        top_idx=None,
        sparse_latent=None,
        hook=None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply SSV steering."""
        original_resid = get_resid_acts(resid, position)

        # Encode
        sparse = sae.encode(original_resid)

        # Dense steering_vector is for residual-space methods; SSV steers in latent space.
        if sparse_latent is None:
            raise ValueError(
                "SSV steering requires `sparse_latent` in SteeringVector metadata. "
                "Re-run extraction or load a vector file that includes sparse_latent."
            )
        vec = sparse_latent
        if getattr(vec, "is_sparse", False):
            vec = vec.to_dense()
        vec = torch.as_tensor(vec, device=sparse.device, dtype=sparse.dtype)
        if top_idx is None:
            raise ValueError(
                "SSV steering requires `top_idx` in SteeringVector metadata. "
                "Re-run extraction or load a vector file that includes top_idx."
            )

        flat_vec = vec.reshape(-1)
        ranked = torch.as_tensor(top_idx, device=flat_vec.device, dtype=torch.long).reshape(-1)
        selected_idx = self.select_top_k_indices(ranked).to(device=flat_vec.device)
        selected_idx = selected_idx[(selected_idx >= 0) & (selected_idx < flat_vec.numel())]
        if selected_idx.numel() == 0:
            raise ValueError(
                f"No valid SSV feature indices selected from top_idx for latent size {flat_vec.numel()}"
            )

        filtered_vec = torch.zeros_like(flat_vec)
        filtered_vec[selected_idx] = flat_vec[selected_idx]
        vec = filtered_vec.reshape_as(vec)
        steering = coeff * vec
        
        steered_sparse = sparse + steering

        # Decode WITHOUT residual preservation (matching reference saessv-demo logic exactly)
        steered_decoded = sae.decode(steered_sparse)

        resid = set_resid_acts(resid, position, steered_decoded)

        if self._prompt_metadata is None:
            self._reset_prompt_metadata()

        pre_vals = sparse[..., selected_idx]
        post_vals = steered_sparse[..., selected_idx]
        if pre_vals.dim() == 3:
            pre_vals = pre_vals[0, -1, :]
            post_vals = post_vals[0, -1, :]
        elif pre_vals.dim() == 2:
            pre_vals = pre_vals[0, :]
            post_vals = post_vals[0, :]
        pre_vals = pre_vals.detach().float().cpu().tolist()
        post_vals = post_vals.detach().float().cpu().tolist()

        self._prompt_metadata.setdefault("steering_idx", selected_idx.detach().cpu().tolist())
        self._prompt_metadata.setdefault("pre_steer_top_idx_acts", pre_vals)
        self._prompt_metadata.setdefault("post_steer_top_idx_acts", post_vals)

        return resid


class SPARESteerModel(BaseSAESteerModel):
    """
    Wrapper for SPARE steering mechanism.

    Implements: h' = h + α(-gϕ(z-) + gϕ(z+))
    where z- removes undesired features and z+ adds desired features.

    Used with: SPARE
    """

    def __init__(
        self,
        model,
        layer: List[int],
        sae: Dict[int, Any],
        steering_vector: Dict[int, torch.Tensor],
        target_behavior: str = "contextual",  # "contextual" or "parametric"
        position: Union[int, str] = "last",
        **kwargs,
    ):
        # Default to all-token steering for GT parity; callers can still set
        # position="last" (or an index) for token-local steering.
        super().__init__(model, layer, sae, steering_vector, position=position, **kwargs)
        self.target_behavior = target_behavior

    @staticmethod
    def _to_dense_latents(sae, encoded_output: Any, reference: torch.Tensor) -> torch.Tensor:
        """Normalize SAE encoder outputs to dense [..., d_sae] latents.

        Supports both:
        - Dense tensor outputs (sae_lens-style encode/pre_acts)
        - Top-k style outputs with (top_acts, top_indices)
        """
        if isinstance(encoded_output, torch.Tensor):
            return encoded_output.to(device=reference.device)

        if hasattr(encoded_output, "top_acts") and hasattr(encoded_output, "top_indices"):
            top_acts = encoded_output.top_acts
            top_indices = encoded_output.top_indices
        elif isinstance(encoded_output, (tuple, list)) and len(encoded_output) == 2:
            top_acts, top_indices = encoded_output
        else:
            raise TypeError(
                "Unsupported SAE output from pre_acts/encode. "
                f"Expected Tensor or (acts, indices), got {type(encoded_output).__name__}."
            )

        d_sae = getattr(getattr(sae, "cfg", None), "d_sae", None)
        if d_sae is None:
            d_sae = int(sae.W_dec.shape[0])

        top_acts = top_acts.to(device=reference.device)
        top_indices = top_indices.to(device=reference.device, dtype=torch.long)

        dense_shape = (*top_acts.shape[:-1], int(d_sae))
        dense = torch.zeros(dense_shape, device=top_acts.device, dtype=top_acts.dtype)
        dense.scatter_(-1, top_indices, top_acts)
        return dense

    def _compute_pre_acts(self, sae, h: torch.Tensor) -> torch.Tensor:
        """Compute pre-activation latents via SAE-native path for GT parity.

        Reference: Code/SPARE/spare/spare_for_generation.py patch_func_signal()
        which calls sae.pre_acts(activations) before remove/add clamping.
        """
        encoded_output = None
        if hasattr(sae, "pre_acts"):
            try:
                encoded_output = sae.pre_acts(h)
            except Exception as e:
                # Some wrappers expose pre_acts but only support SAE-device inputs.
                # Fall back to wrapper-safe encode path.
                logger.debug(f"SPARE: pre_acts failed; falling back to encode(). Error: {e}")

        if encoded_output is None:
            encoded_output = sae.encode(h)

        return self._to_dense_latents(sae, encoded_output, h)

    def hook_fn(
        self,
        resid: torch.Tensor,
        position: Union[int, str],
        coeff: float,  # alpha in paper
        steering_vector: torch.Tensor,
        sae,
        zC,
        zM,
        indices_pos,
        indices_neg,
        top_idx=None,
        top_idx_neg=None,
        hook=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply SPARE steering.

        h' = h + α(-gϕ(z-) + gϕ(z+))
        Per-layer weights are bound via setup_hooks() partial; the optional
        kwargs let the same hook work for any layer when called directly.
        """
        h = get_resid_acts(resid, position)
        z_pre = self._compute_pre_acts(sae, h)

        z_rem = torch.zeros_like(z_pre)
        z_add = torch.zeros_like(z_pre)

        if getattr(zC, "is_sparse", False):
            zC = zC.to_dense()
        if getattr(zM, "is_sparse", False):
            zM = zM.to_dense()

        context_ref = torch.as_tensor(zC, device=z_pre.device, dtype=z_pre.dtype)
        parametric_ref = torch.as_tensor(zM, device=z_pre.device, dtype=z_pre.dtype)
        idx_pos = torch.as_tensor(
            indices_pos,
            device=z_pre.device,
            dtype=torch.long,
        )
        idx_neg = torch.as_tensor(
            indices_neg,
            device=z_pre.device,
            dtype=torch.long,
        )

        if top_idx is None:
            raise ValueError(
                "SPARE steering requires `top_idx` in SteeringVector metadata. "
                "Re-run extraction or load a vector file that includes top_idx."
            )

        if idx_pos.numel() > 0:
            ranked_pos = torch.as_tensor(
                top_idx,
                device=z_pre.device,
                dtype=torch.long,
            ).reshape(-1)
            selected_pos = self.select_top_k_indices(ranked_pos).to(device=z_pre.device)
            selected_pos = selected_pos[(selected_pos >= 0) & (selected_pos < z_pre.shape[-1])]
            idx_pos = idx_pos[torch.isin(idx_pos, selected_pos)]

        if idx_neg.numel() > 0:
            neg_rank_source = top_idx_neg if top_idx_neg is not None else indices_neg
            ranked_neg = torch.as_tensor(
                neg_rank_source,
                device=z_pre.device,
                dtype=torch.long,
            ).reshape(-1)
            selected_neg = self.select_top_k_indices(ranked_neg).to(device=z_pre.device)
            selected_neg = selected_neg[(selected_neg >= 0) & (selected_neg < z_pre.shape[-1])]
            idx_neg = idx_neg[torch.isin(idx_neg, selected_neg)]

        context_scaled = context_ref * coeff
        parametric_scaled = parametric_ref * coeff

        if self.target_behavior == "contextual":
            if idx_pos.numel() > 0:
                z_add[..., idx_pos] = torch.clamp(
                    context_scaled[idx_pos] - z_pre[..., idx_pos],
                    min=0
                ).to(z_add.dtype)
            if idx_neg.numel() > 0:
                z_rem[..., idx_neg] = torch.min(
                    z_pre[..., idx_neg],
                    parametric_scaled[idx_neg]
                ).to(z_rem.dtype)

        elif self.target_behavior == "parametric":
            if idx_neg.numel() > 0:
                z_add[..., idx_neg] = torch.clamp(
                    parametric_scaled[idx_neg] - z_pre[..., idx_neg],
                    min=0
                ).to(z_add.dtype)
            if idx_pos.numel() > 0:
                z_rem[..., idx_pos] = torch.min(
                    z_pre[..., idx_pos],
                    context_scaled[idx_pos]
                ).to(z_rem.dtype)

        steering_rem = sae.decode(z_rem)
        steering_add = sae.decode(z_add)

        z_post = z_pre - z_rem + z_add
        tracked_idx = torch.cat([idx_pos, idx_neg]) if (idx_pos.numel() + idx_neg.numel()) > 0 else torch.empty(0, device=z_pre.device, dtype=torch.long)

        h_steered = h - steering_rem.to(h.dtype) + steering_add.to(h.dtype)

        if self._prompt_metadata is None:
            self._reset_prompt_metadata()

        if tracked_idx.numel() > 0:
            pre_vals = z_pre[..., tracked_idx]
            post_vals = z_post[..., tracked_idx]
            if pre_vals.dim() == 3:
                pre_vals = pre_vals[0, -1, :]
                post_vals = post_vals[0, -1, :]
            elif pre_vals.dim() == 2:
                pre_vals = pre_vals[0, :]
                post_vals = post_vals[0, :]
            pre_vals = pre_vals.detach().float().cpu().tolist()
            post_vals = post_vals.detach().float().cpu().tolist()

            self._prompt_metadata.setdefault("steering_idx", tracked_idx.detach().cpu().tolist())
            self._prompt_metadata.setdefault("pre_steer_top_idx_acts", pre_vals)
            self._prompt_metadata.setdefault("post_steer_top_idx_acts", post_vals)

        return set_resid_acts(resid, position, h_steered)


class SRPSSteerModel(BaseSAESteerModel):
    """
    Wrapper that preserves the norm of the residual after steering.

    Used with: SRPS (SAE Representation Projection Steering) where norm preservation is important.
    """

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        position: Union[int,str],
        steering_vector: torch.Tensor,
        sae,
        top_idx=None,
        sparse_latent=None,
        hook=None,
        **kwargs,
    ) -> torch.Tensor:
        """Add steering vector while preserving residual norm."""
        current_resid = get_resid_acts(resid, position)
        vec = steering_vector.to(dtype=resid.dtype, device=current_resid.device)

        if top_idx is None:
            raise ValueError(
                "SRPS steering requires `top_idx` in SteeringVector metadata. "
                "Re-run extraction or load a vector file that includes top_idx."
            )

        latent_size = sae.W_dec.shape[0]
        ranked = torch.as_tensor(top_idx, device=vec.device, dtype=torch.long).reshape(-1)
        selected_idx = self.select_top_k_indices(ranked).to(device=vec.device)
        selected_idx = selected_idx[(selected_idx >= 0) & (selected_idx < latent_size)]
        if selected_idx.numel() == 0:
            raise ValueError(
                f"No valid SRPS feature indices selected from top_idx for latent size {latent_size}"
            )

        pre_sparse = sae.encode(current_resid)

        new_resid = current_resid + coeff * vec
        if self.norm:
            # Preserve norm
            current_norm = current_resid.norm(dim=-1, keepdim=True)
            new_norm = new_resid.norm(dim=-1, keepdim=True)
            updated_resid = new_resid * (current_norm / (new_norm + 1e-10))
            resid = set_resid_acts(resid, position, updated_resid)
        else:
            updated_resid = new_resid
            resid = set_resid_acts(resid, position, updated_resid)

        post_sparse = sae.encode(updated_resid)
        if self._prompt_metadata is None:
            self._reset_prompt_metadata()

        pre_vals = pre_sparse[..., selected_idx]
        post_vals = post_sparse[..., selected_idx]
        if pre_vals.dim() == 3:
            pre_vals = pre_vals[0, -1, :]
            post_vals = post_vals[0, -1, :]
        elif pre_vals.dim() == 2:
            pre_vals = pre_vals[0, :]
            post_vals = post_vals[0, :]
        pre_vals = pre_vals.detach().float().cpu().tolist()
        post_vals = post_vals.detach().float().cpu().tolist()

        self._prompt_metadata.setdefault("steering_idx", selected_idx.detach().cpu().tolist())
        self._prompt_metadata.setdefault("pre_steer_top_idx_acts", pre_vals)
        self._prompt_metadata.setdefault("post_steer_top_idx_acts", post_vals)
        return resid


class SAERSVSteerModel(BaseSteerModel):
    """
    SAE-RSV Steering Model.

    Applies refined (denoised + augmented) steering vector.
    SAE is used only during extraction, not at steering time.

    Paper: "Enhancing LLM Steering through SAE-based Vector Refinement"
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        **kwargs,
    ):
        super().__init__(model, layer, steering_vector, **kwargs)

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        position: Union[int,str],
        steering_vector: torch.Tensor,
        hook,
        **kwargs,
    ) -> torch.Tensor:
        """Apply refined steering vector."""
        vec = steering_vector.to(dtype=resid.dtype)
        resid = set_resid_acts(resid, position, get_resid_acts(resid, position) + coeff * vec)
        return resid


class SAECoTSteerModel(BaseSAESteerModel):
    """
    SAE-BASE steering model matching GT Code/SAE-free/sae_gsm.py behavior.

    Core operation:
    1) encode residual into SAE latents
    2) overwrite selected latent feature(s) to coeff * max_act
    3) decode and optionally add reconstruction error
    4) optionally preserve residual L2 norm
    """

    def __init__(
        self,
        model,
        layer: List[int],
        sae: Dict[int, Any],
        steering_vector: Dict[int, torch.Tensor],
        saecot_overwrite: bool = True,
        use_reconstruction_error: bool = True,
        saecot_norm_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(model, layer, sae, steering_vector, **kwargs)
        self.overwrite = bool(saecot_overwrite)
        self.use_reconstruction_error = bool(use_reconstruction_error)
        self.norm_eps = float(saecot_norm_eps)

    def hook_fn(
        self,
        resid: torch.Tensor,
        position: Union[int, str],
        coeff: float,
        steering_vector: torch.Tensor,
        sae,
        top_idx=None,
        sparse_latent=None,
        hook=None,
        **kwargs,
    ) -> torch.Tensor:
        original = get_resid_acts(resid, position)
        x = original.float()

        sparse = sae.encode(x)
        if sparse_latent is None:
            raise ValueError(
                "SAECoT steering requires `sparse_latent` in SteeringVector metadata. "
                "Re-run extraction or load a vector file that includes sparse_latent."
            )
        vector = sparse_latent
        if getattr(vector, "is_sparse", False):
            vector = vector.to_dense()
        vector = torch.as_tensor(vector, device=sparse.device, dtype=sparse.dtype)

        if top_idx is None:
            raise ValueError(
                "SAECoT steering requires `top_idx` in SteeringVector metadata. "
                "Re-run extraction or load a vector file that includes top_idx."
            )

        ranked = torch.as_tensor(top_idx, device=vector.device, dtype=torch.long).reshape(-1)
        selected_idx = self.select_top_k_indices(ranked).to(device=vector.device)
        selected_idx = selected_idx[(selected_idx >= 0) & (selected_idx < vector.numel())]
        if selected_idx.numel() == 0:
            raise ValueError(
                f"No valid SAECoT feature indices selected from top_idx for latent size {vector.numel()}"
            )

        flat_vector = vector.reshape(-1)
        filtered_vector = torch.zeros_like(flat_vector)
        filtered_vector[selected_idx] = flat_vector[selected_idx]
        vector = filtered_vector.reshape_as(vector)

        target = coeff * vector

        if self.overwrite:
            steered_sparse = sparse.clone()
            mask = (vector != 0)
            while mask.dim() < steered_sparse.dim():
                mask = mask.unsqueeze(0)
            while target.dim() < steered_sparse.dim():
                target = target.unsqueeze(0)
            steered_sparse = torch.where(mask, target, steered_sparse)
        else:
            steered_sparse = sparse + target

        decoded = sae.decode(steered_sparse)

        if self.use_reconstruction_error:
            recon = sae.decode(sparse)
            decoded = decoded + (x - recon)

        if self.norm:
            before = x.norm(p=2, dim=-1, keepdim=True).clamp(min=self.norm_eps)
            after = decoded.norm(p=2, dim=-1, keepdim=True).clamp(min=self.norm_eps)
            decoded = decoded * (before / after)

        updated = decoded.to(dtype=original.dtype)

        if self._prompt_metadata is None:
            self._reset_prompt_metadata()
        pre_vals = sparse[..., selected_idx]
        post_vals = steered_sparse[..., selected_idx]
        if pre_vals.dim() == 3:
            pre_vals = pre_vals[0, -1, :]
            post_vals = post_vals[0, -1, :]
        elif pre_vals.dim() == 2:
            pre_vals = pre_vals[0, :]
            post_vals = post_vals[0, :]
        pre_vals = pre_vals.detach().float().cpu().tolist()
        post_vals = post_vals.detach().float().cpu().tolist()

        self._prompt_metadata.setdefault("steering_idx", selected_idx.detach().cpu().tolist())
        self._prompt_metadata.setdefault("pre_steer_top_idx_acts", pre_vals)
        self._prompt_metadata.setdefault("post_steer_top_idx_acts", post_vals)
        self._prompt_metadata.setdefault("selected_feature_count", int((vector != 0).sum().item()))

        return set_resid_acts(resid, position, updated)


class FeatSteerModel(BaseSAESteerModel):
    """
    Feature-testing steering model.

    This steer model edits selected SAE latent features at inference time.

    Modes:
    - Scale mode: multiply selected feature activations by ``coeff``
    - Overwrite mode: set selected feature activations to ``max_act``
    """

    def __init__(
        self,
        model,
        layer: List[int],
        sae: Dict[int, Any],
        feature_list: Dict[int, List[int]],
        max_act: Optional[float] = None,
        overwrite_with_max_act: bool = False,
        use_reconstruction_error: bool = True,
        **kwargs,
    ):
        if feature_list is None:
            raise ValueError("FeatSteerModel requires `feature_list`.")

        # Ignore any extracted steering vector injected by generic pipeline paths.
        kwargs.pop("steering_vector", None)
        feature_masks = self._build_feature_masks(layer=layer, sae=sae, feature_list=feature_list)

        super().__init__(
            model=model,
            layer=layer,
            sae=sae,
            steering_vector=feature_masks,
            **kwargs,
        )
        self.overwrite = bool(overwrite_with_max_act)
        self.max_act = None if max_act is None else float(max_act)
        self.use_reconstruction_error = bool(use_reconstruction_error)

        if self.overwrite and self.max_act is None:
            raise ValueError("FeatSteerModel requires `max_act` when overwrite_with_max_act=True")

    @staticmethod
    def _latent_dim(sae_module: Any) -> int:
        """Infer SAE latent dimensionality from cfg or decoder weights."""
        cfg = getattr(sae_module, "cfg", None)
        d_sae = getattr(cfg, "d_sae", None)
        if d_sae is not None:
            return int(d_sae)

        if hasattr(sae_module, "W_dec"):
            return int(sae_module.W_dec.shape[0])

        raise ValueError("Could not infer SAE latent dimension for FeatSteerModel")

    @classmethod
    def _build_feature_masks(
        cls,
        layer: List[int],
        sae: Dict[int, Any],
        feature_list: Dict[int, List[int]],
    ) -> Dict[int, torch.Tensor]:
        """Build dense binary masks (layer -> [d_sae]) from feature indices."""
        masks: Dict[int, torch.Tensor] = {}
        for l in layer:
            d_sae = cls._latent_dim(sae[l])
            feat_idx = torch.as_tensor(feature_list[l], dtype=torch.long).reshape(-1)
            mask = torch.zeros(d_sae, dtype=torch.float32)
            mask[feat_idx] = 1.0
            masks[l] = mask

        return masks

    def hook_fn(
        self,
        resid: torch.Tensor,
        position: Union[int, str],
        coeff: float,
        steering_vector: torch.Tensor,
        sae,
        hook=None,
        **kwargs,
    ) -> torch.Tensor:
        """Scale or overwrite selected SAE feature activations."""
        original_resid = get_resid_acts(resid, position)
        sparse = sae.encode(original_resid)

        feature_mask = steering_vector.to(device=sparse.device)
        selected_idx = torch.nonzero(feature_mask != 0, as_tuple=False).reshape(-1)

        steered_sparse = sparse.clone()
        pre_steer_feat_act = sparse[..., selected_idx]
        overwrite_value = None
        if self.overwrite:
            overwrite_value = self.max_act

            steered_sparse[..., selected_idx] = torch.as_tensor(
                overwrite_value,
                device=steered_sparse.device,
                dtype=steered_sparse.dtype,
            )
            mode = "overwrite"
        else:
            steered_sparse[..., selected_idx] += coeff
            mode = "scale"

        post_steer_feat_act = steered_sparse[..., selected_idx]

        steered_decoded = sae.decode(steered_sparse)
        if self.use_reconstruction_error:
            original_decoded = sae.decode(sparse)
            steered_decoded = steered_decoded + (original_resid - original_decoded)

        updated = steered_decoded.to(dtype=original_resid.dtype)
        resid = set_resid_acts(resid, position, updated)

        if self._prompt_metadata is None:
            self._reset_prompt_metadata()

        pre_feat_stats = pre_steer_feat_act.detach().float()
        post_feat_stats = post_steer_feat_act.detach().float()
        pre_mean = float(pre_feat_stats.mean().item()) if pre_feat_stats.numel() > 0 else -1
        post_mean = float(post_feat_stats.mean().item()) if post_feat_stats.numel() > 0 else -1
        pre_active = int((pre_feat_stats > 0).sum().item())
        post_active = int((post_feat_stats > 0).sum().item())

        self._prompt_metadata.setdefault("pre_steer_feat_act", pre_mean)
        self._prompt_metadata.setdefault("post_steer_feat_act", post_mean)
        self._prompt_metadata.setdefault("pre_steer_feat_active_count", pre_active)
        self._prompt_metadata.setdefault("post_steer_feat_active_count", post_active)

        return resid


class SAETSSteerModel(BaseSAESteerModel):
    """
    SAE-TS (SAE-Targeted Steering) Model.

    Applies targeted steering vectors designed to affect specific SAE features
    while minimizing unintended side effects. SAE is used only during extraction.

    Paper: "Improving Steering Vectors by Targeting Sparse Autoencoder Features"
    """

    def __init__(
        self,
        model,
        layer: List[int],
        sae: Dict[int, Any],
        steering_vector: Dict[int, torch.Tensor],
        auto_scale: bool = True,
        target_loss: float = 4.0,
        **kwargs,
    ):
        super().__init__(model, layer, sae, steering_vector, **kwargs)
        self.auto_scale = auto_scale
        self.target_loss = target_loss
        self._optimal_scale = None

        if self.auto_scale and self.steering_vector:
            logger.info(f"SAE-TS: Auto-scaling enabled. Finding scale for target loss {self.target_loss}...")
            self._optimal_scale = self._find_optimal_scale()
            logger.info(f"SAE-TS: Optimal scale found: {self._optimal_scale:.4f}")
        else: 
            self._optimal_scale = 1

    def _find_optimal_scale(
        self,
        sample_prompt: str = "The quick brown fox",
        target_loss: float = 4.0,
        max_scale: float = 300.0,
        scale_step: float = 10.0,
    ) -> float:
        """
        Find optimal scaling factor using linear interpolation to absolute target loss.

        Reference: Code/SAE-TS/src/sae_ts/ft_effects/utils.py get_scale()
        Iterates through scales, finds where loss exceeds target, then linearly
        interpolates between the last two points.
        """
        primary_layer = self.layer[0]
        vec = self.steering_vector[primary_layer]
        hook_names = self.get_hook_name()

        with torch.no_grad():
            tokens = self.model.to_tokens(sample_prompt)
            scales = [float(s) for s in range(0, int(max_scale) + 1, int(scale_step))]
            losses = []

            for scale in scales:
                def hook_fn(resid, hook, v=vec, s=scale):
                    v = v.to(dtype=resid.dtype)
                    resid[:, :, :] = resid[:, :, :] + s * v
                    return resid

                steered_logits = self.model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_names[0], hook_fn)],
                )
                loss = F.cross_entropy(
                    steered_logits[:, :-1].reshape(-1, steered_logits.shape[-1]),
                    tokens[:, 1:].reshape(-1),
                ).item()
                losses.append(loss)
                if loss > target_loss:
                    break

        if len(losses) < 2:
            return 1.0

        # Linear interpolation between last two points (per GT)
        used_scales = scales[:len(losses)]
        x1, x2 = used_scales[-2], used_scales[-1]
        y1, y2 = losses[-2], losses[-1]
        if abs(y2 - y1) < 1e-8:
            return float(x2)
        m = (y2 - y1) / (x2 - x1)
        b_intercept = y1 - m * x1
        return (target_loss - b_intercept) / m

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        position: Union[int,str],   
        steering_vector: torch.Tensor,
        sae,
        hook,
        **kwargs,
    ) -> torch.Tensor:
        """Apply targeted steering vector with optional auto-scaling.
        
        Reference: Code/SAE-TS/src/sae_ts/steering/patch.py patch_resid
        Steers ALL sequence positions (not just last token).
        """
        if self.auto_scale and self._optimal_scale is not None:
            effective_coeff = coeff * self._optimal_scale
        else:
            effective_coeff = coeff
        vec = steering_vector.to(dtype=resid.dtype)
        resid = set_resid_acts(resid, position, get_resid_acts(resid, position) + effective_coeff * vec)
        return resid


class SAEIOSteerModel(BaseSAESteerModel):
    def __init__(
        self,
        model,
        layer: List[int],
        sae: Dict[int, Any],
        steering_vector: Dict[int, torch.Tensor],
        top_k: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            model,
            layer,
            sae,
            steering_vector,
            top_k=top_k,
            **kwargs,
        )
    """
    SAE Input/Output (IO) Steering Model.
    
    Applies steering to SAE features selected by Output Score.
    Uses adaptive scaling based on current max activation (per SAE-IO paper/code).
    
    Sparse feature masks are read from `sparse_latent` metadata.
    """

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        position: Union[int,str],   
        steering_vector: torch.Tensor,
        sae,
        top_idx,
        sparse_latent=None,
        hook=None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply adaptive steering.

        This method expects `top_idx` to be passed via the SteeringVector metadata (per-layer).
        """
        if top_idx is None:
            raise ValueError(
                "SAEIO steering requires `top_idx` in SteeringVector metadata. "
                "Ensure the steering vector file includes `top_idx` or re-run extraction."
            )
        if sparse_latent is None:
            raise ValueError(
                "SAEIO steering requires `sparse_latent` in SteeringVector metadata. "
                "Ensure the steering vector file includes sparse_latent or re-run extraction."
            )

        original_resid = get_resid_acts(resid, position)

        # 1. Encode
        sparse = sae.encode(original_resid)  # [B, n_latents]

        # 2. Compute adaptive scale (max activation in current batch/features)
        adaptive_scale = sparse.max()

        # 3. Create steering delta
        # sparse_latent is a latent mask [n_latents]. We optionally filter the
        # ranked feature list with top_k and steer only the selected features.
        vec = sparse_latent
        if getattr(vec, "is_sparse", False):
            vec = vec.to_dense()
        vec = torch.as_tensor(vec, dtype=sparse.dtype, device=sparse.device)

        if vec.dim() == 1:
            vec = vec.unsqueeze(0)  # [1, D]
            
        D = vec.size(1)

        top_idx = self.select_top_k_indices(top_idx).to(device=vec.device)
        top_idx = top_idx[(top_idx >= 0) & (top_idx < D)]
        if top_idx.numel() == 0:
            raise ValueError(
                f"No valid SAEIO feature indices selected from top_idx for latent size {D}"
            )

        mask = torch.zeros(D, dtype=torch.bool, device=vec.device)
        mask[top_idx] = True

        # expand to batch if needed
        mask = mask.unsqueeze(0)  # [1, D]

        delta = vec * mask * (adaptive_scale * coeff)

        # 4. Apply
        steered_sparse = sparse + delta

        # 5. Decode with residual preservation
        steered_decoded = sae.decode(steered_sparse)
        original_decoded = sae.decode(sparse)

        recon_error = original_resid - original_decoded
        resid = set_resid_acts(resid, position, steered_decoded + recon_error)
        if self._prompt_metadata is None:
            self._reset_prompt_metadata()

        pre_vals = sparse[..., top_idx]
        post_vals = steered_sparse[..., top_idx]
        if pre_vals.dim() == 3:
            pre_vals = pre_vals[0, -1, :]
            post_vals = post_vals[0, -1, :]
        elif pre_vals.dim() == 2:
            pre_vals = pre_vals[0, :]
            post_vals = post_vals[0, :]
        pre_vals = pre_vals.detach().float().cpu().tolist()
        post_vals = post_vals.detach().float().cpu().tolist()

        self._prompt_metadata.setdefault("adaptive_scale", adaptive_scale.item())
        self._prompt_metadata.setdefault("steering_idx", top_idx.cpu().tolist())
        self._prompt_metadata.setdefault("pre_steer_top_idx_acts", pre_vals)
        self._prompt_metadata.setdefault("post_steer_top_idx_acts", post_vals)
        return resid


class CorrSteerModel(BaseSAESteerModel):
    """
    CorrSteer Model - Correlation-based Steering.

    Applies steering by adding correlated feature directions to the residual stream.
    Reference: Code/CorrSteer/corrsteer/steer.py SteeringHook

    Paper: "CorrSteer: Steering Improves Task Performance and Safety"
    
    Key formula (Eq. 7):
        x' = x + v_steer  (or x - v_steer if subtract=True)
    
    Where v_steer = c_i * W_dec[:, i] is pre-computed by CorrSteerExtractor
    and passed as steering_vector (already decoded to d_model space).
    
    Inherits from BaseSteerModel because SAE is not used at inference time.
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        sae: Optional[Dict[int, Any]] = None,
        hook_point: str = ["pre"],
        # CorrSteer-specific parameters from reference: Code/CorrSteer/train.py CorrConfig
        corrsteer_lastk: int = 1,  
        top_k: Optional[List[int]] = None,  # Optional rank positions to keep from extractor feature ranking
        **kwargs,
    ):
        """
        Args:
            model: Language model
            layer: Target layer(s) as List[int]
            steering_vector: Dict[int, Tensor] CorrSteer steering vector (already decoded to d_model space)
            sae: Dict[int, SAE] (required when steer.top_k is used)
            hook_point: Hook point ("pre" or "post")
            corrsteer_lastk: Number of last tokens to apply steering (default: 1)
        """
        if sae is None:
            raise ValueError("CorrSteerModel requires SAE modules in `sae`.")
        super().__init__(
            model,
            layer,
            sae,
            steering_vector,
            top_k=top_k,
            hook_point=hook_point,
            **kwargs,
        )
        self.lastk = corrsteer_lastk

    def _ranked_feature_indices(self, top_idx: Any, device: torch.device) -> torch.Tensor:
        """Normalize CorrSteer top_idx metadata to a ranked feature-index tensor.

        Supports:
        - list[int] / tensor[int]
        - list[dict] where each dict contains feature_index
        """
        if isinstance(top_idx, torch.Tensor):
            return top_idx.to(device=device, dtype=torch.long).reshape(-1)

        if isinstance(top_idx, list) and top_idx and isinstance(top_idx[0], dict):
            indices = [int(item["feature_index"]) for item in top_idx if "feature_index" in item]
            return torch.tensor(indices, device=device, dtype=torch.long)

        return torch.as_tensor(top_idx, device=device, dtype=torch.long).reshape(-1)

    def hook_fn(
        self,
        resid: torch.Tensor,
        coeff: float,
        steering_vector: torch.Tensor,
        top_idx,
        sparse_latent,
        sae=None,
        hook=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply CorrSteer steering vector (Eq. 7 in paper).
        
        Note: ``position`` is accepted for API consistency but unused;
        CorrSteer steers the last ``self.lastk`` tokens directly.
        
        Reference: Code/CorrSteer/corrsteer/steer.py SteeringHook.__call__
        
        ```python
        steering = -self.steering if self.substract else self.steering
        output[:, -self.lastk :, :] = output[:, -self.lastk :, :] + steering
        ```
        """
        vec = steering_vector.to(dtype=resid.dtype, device=resid.device)

        if sparse_latent is None:
            raise ValueError(
                "CorrSteer steering requires `sparse_latent` in metadata."
            )
        if top_idx is None:
            raise ValueError(
                "CorrSteer steering requires `top_idx` in metadata."
            )

        ranked = self._ranked_feature_indices(top_idx, device=resid.device)
        selected_feat_idx = self.select_top_k_indices(ranked).to(device=resid.device)

        latent = sparse_latent
        if getattr(latent, "is_sparse", False):
            latent = latent.to_dense()
        latent = torch.as_tensor(latent, device=resid.device).reshape(-1)

        selected_feat_idx = selected_feat_idx[
            (selected_feat_idx >= 0) & (selected_feat_idx < latent.numel())
        ]
        if selected_feat_idx.numel() == 0:
            raise ValueError(
                f"No valid CorrSteer feature indices selected from top_idx for latent size {latent.numel()}"
            )

        masked_latent = torch.zeros_like(latent)
        masked_latent[selected_feat_idx] = latent[selected_feat_idx]

        decoded = masked_latent.to(dtype=sae.W_dec.dtype) @ sae.W_dec.to(device=resid.device)
        vec = decoded.to(dtype=resid.dtype)

        pre_sparse = sae.encode(resid[:, -self.lastk:, :])

        # Apply to last K tokens (default: 1 = last token only)
        # Reference: output[:, -self.lastk :, :] = output[:, -self.lastk :, :] + steering
        resid[:, -self.lastk:, :] = resid[:, -self.lastk:, :] + coeff*vec

        post_sparse = sae.encode(resid[:, -self.lastk:, :])
        if self._prompt_metadata is None:
            self._reset_prompt_metadata()

        pre_vals = pre_sparse[..., selected_feat_idx]
        post_vals = post_sparse[..., selected_feat_idx]
        if pre_vals.dim() == 3:
            pre_vals = pre_vals[0, -1, :]
            post_vals = post_vals[0, -1, :]
        elif pre_vals.dim() == 2:
            pre_vals = pre_vals[0, :]
            post_vals = post_vals[0, :]
        pre_vals = pre_vals.detach().float().cpu().tolist()
        post_vals = post_vals.detach().float().cpu().tolist()

        self._prompt_metadata.setdefault("steering_idx", selected_feat_idx.detach().cpu().tolist())
        self._prompt_metadata.setdefault("pre_steer_top_idx_acts", pre_vals)
        self._prompt_metadata.setdefault("post_steer_top_idx_acts", post_vals)
        
        return resid