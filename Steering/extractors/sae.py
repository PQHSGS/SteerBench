"""
SAE-based extractors for steering vector computation.

These extractors work with Sparse Autoencoder latents for more interpretable steering.
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import numpy as np
from functools import partial
from tqdm import tqdm
import os
import transformer_lens.utils as tl_utils
from ..evaluators import EVALUATOR_MAP
from ..base import BaseExtractor
from ..utils import (
    build_token_mask,
    build_top_feature_tracker_from_stats,
    collect_dense_activations,
    collect_sae_activations,
    get_hook_name,
)
from ..logger import setup_logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from huggingface_hub import hf_hub_download

logger = setup_logger(__name__)


def _contrast_or_zeros(
    contrast_by_layer: Optional[Dict[int, torch.Tensor]],
    layer: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return per-layer contrast tensor or a zero tensor matching reference shape."""
    if contrast_by_layer is None:
        return torch.zeros_like(reference)
    return contrast_by_layer[layer]


def _safe_topk_indices(values: torch.Tensor, k: int, by_abs: bool = True) -> torch.Tensor:
    """Top-k index selection that safely handles empty tensors and non-positive k."""
    if values.numel() == 0 or k <= 0:
        return torch.empty(0, dtype=torch.long, device=values.device)

    k = min(int(k), int(values.numel()))
    scores = values.abs() if by_abs else values
    sorted_indices = torch.argsort(scores, descending=True)
    return sorted_indices[:k]



def _build_method_metadata(
    method: str,
    top_idx: Optional[Dict[int, Any]] = None,
    sparse_latent: Optional[Dict[int, torch.Tensor]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a consistent metadata payload shared across extractor methods."""
    metadata: Dict[str, Any] = {"method": method}
    if top_idx is not None:
        metadata["top_idx"] = top_idx
    if sparse_latent is not None:
        metadata["sparse_latent"] = sparse_latent
    metadata.update(extra)
    return metadata


class SAEIOExtractor(BaseExtractor):
    """
    SAE Input/Output (IO) Extractor.

    Filters SAE features based on their "Output Score" - how much they
    actually affect the model's output distribution.

    Paper: "SAEs Are Good for Steering"
    """

    METHOD_NAME = "SAE_IO"

    def __init__(
        self,
        model,
        sae: Dict[int, Any],
        layer: List[int],
        top_k: int,
        act_threshold: float,
        amp_factor: float,
        neutral_prompt: str = "From my experience,",
        **kwargs,
    ):
        super().__init__(model, layer, **kwargs)
        self.sae = sae
        self.top_k = top_k
        self.act_threshold = act_threshold
        self.neutral_prompt = neutral_prompt
        self.amp_factor = amp_factor

        # Per-layer results (all dicts keyed by layer)
        self.feature_scores: Dict[int, Dict[int, float]] = {}
        self.selected_features: Dict[int, List[Tuple[int, float]]] = {}
        self.sparse_latent: Dict[int, torch.Tensor] = {}
        self.logit_lens_topk: Dict[int, Any] = {}

    def _get_activations(
        self, inputs: List[str], layers: List[int]
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, Any]]]:
        """
        Get mean-of-max activations and streaming stats in one pass.

        This avoids retaining full raw [N, D] activation matrices just for tracker metadata.
        """
        collected = collect_sae_activations(
            self.model,
            self.sae,
            inputs,
            layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling="max",
            tokenizer=self.model.tokenizer,
            active_threshold=self.act_threshold,
        )
        layer_acts = collected.activations
        layer_stats = collected.stats

        mean_acts: Dict[int, torch.Tensor] = {}
        for layer in layers:
            mean_acts[layer] = layer_acts[layer].to(self.device).mean(dim=0)

        return mean_acts, layer_stats

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract steering vector using Output Score filtering for each layer.
        """
        from ..utils import cache_logit_lens, get_output_score

        precomputed_logit_lens = {}
        for layer in self.layer:
            sae = self.sae[layer]
            logger.info(f"[Layer {layer}] Precomputing logit lens...")
            precomputed_logit_lens[layer] = cache_logit_lens(self.model, sae, k=20)
            torch.cuda.empty_cache()

        # 2. Identify candidate features (active in target data)
        logger.info(f"Finding active features in {len(target_data)} prompts...")
        mean_acts, layer_stats = self._get_activations(target_data, self.layer)

        self.vector = {}
        feature_tracker: Dict[int, List[Dict[str, Any]]] = {}
        for layer in self.layer:
            layer_mean = mean_acts[layer]
            self.logit_lens_topk[layer] = precomputed_logit_lens[layer]

            # Filter by threshold
            active_indices = (layer_mean > self.act_threshold).nonzero(as_tuple=True)[0].tolist()
            logger.info(
                f"[Layer {layer}] Found {len(active_indices)} active features > {self.act_threshold}"
            )

            if not active_indices:
                logger.info(f"[Layer {layer}] Warning: No active features found.")
                self.vector[layer] = torch.zeros_like(layer_mean)
                self.feature_scores[layer] = {}
                self.selected_features[layer] = []
                continue

            # 3. Compute Output Scores for candidates
            topk_tokens = precomputed_logit_lens[layer]
            torch.cuda.empty_cache()

            logger.info(f"[Layer {layer}] Computing Output Scores...")
            scores: Dict[int, float] = {}
            for idx in tqdm(active_indices, desc=f"Scoring features (layer {layer})"):
                feature_top_tokens = topk_tokens.indices[idx].tolist()
                score = get_output_score(
                    self.model,
                    self.sae[layer],
                    layer,
                    idx,
                    feature_top_tokens,
                    self.neutral_prompt,
                    amp_factor=self.amp_factor,
                )
                scores[idx] = score

            # FIX A: store per layer, not overwritten flat attrs
            self.feature_scores[layer] = scores

            # 4. Select top-k by Output Score
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            layer_selected = sorted_features[: self.top_k]
            self.selected_features[layer] = layer_selected

            feature_tracker[layer] = build_top_feature_tracker_from_stats(
                [idx for idx, _ in layer_selected],
                target_stats=layer_stats[layer],
                top_n=self.top_k,
            )

            # 5. Construct sparse vector with 1.0 at selected indices
            vec = torch.zeros_like(layer_mean)
            for idx, _ in layer_selected:
                vec[idx] = 1.0
            self.sparse_latent[layer] = vec.to_sparse().coalesce().cpu()
            self.vector[layer] = self.sae[layer].decode(
                vec.to(device=self.sae[layer].W_dec.device, dtype=self.sae[layer].W_dec.dtype)
            ).to(self.device)

        self.metadata = _build_method_metadata(
            "SAE_IO",
            # FIX C: expose per-layer selected indices, not a single flat list
            top_idx={
                layer: [idx for idx, _ in features]
                for layer, features in self.selected_features.items()
            },
            sparse_latent=self.sparse_latent,
            feature_tracker=feature_tracker,
            # "feature_scores": self.feature_scores,
        )

        return self.vector

class SASExtractor(BaseExtractor):
    """
    SAE-based Activation Steering (SAS) Extractor.

    Uses SAE latent activations with shared feature removal
    for more precise steering.

    Paper: "Activation Steering with SAEs"
    """

    METHOD_NAME = "SAS"

    def __init__(
        self,
        model,
        sae,
        layer: List[int],
        top_k: int = 10,
        act_threshold: float = 0,
        act_frac: float = 0.7,  # Paper default τ=0.7
        **kwargs,
    ):
        super().__init__(model, layer, **kwargs)
        self.sae = sae
        self.top_k = top_k
        self.act_threshold = act_threshold
        self.act_frac = act_frac

        self.sparse_latent: Dict[int, torch.Tensor] = {}
        self.top_idx: Dict[int, List[int]] = {}

    def _get_activations(
        self,
        inputs: List[str],
        layers: List[int],
        **kwargs,
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Dict[str, Any]]]:
        """
        Collect SAE latent activations for all layers in a single forward pass.

        Returns:
            Dict mapping layer -> (mean_act, frac_act)
        """

        collected = collect_sae_activations(
            self.model,
            self.sae,
            inputs,
            layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            tokenizer=self.model.tokenizer,
            active_threshold=self.act_threshold,
        )
        all_layer_acts = collected.activations
        all_layer_stats = collected.stats

        result = {}
        for layer in layers:
            layer_acts = all_layer_acts[layer].to(self.device)
            total_samples = max(int(layer_acts.shape[0]), 1)

            active_mask = layer_acts > float(self.act_threshold)
            count_act = active_mask.sum(dim=0).to(dtype=layer_acts.dtype)
            sum_act = (layer_acts * active_mask.to(dtype=layer_acts.dtype)).sum(dim=0)

            mean_act = torch.zeros_like(sum_act)
            nonzero = count_act > 0
            mean_act[nonzero] = sum_act[nonzero] / count_act[nonzero]

            frac_act = count_act / float(total_samples)

            keep = frac_act > self.act_frac
            mean_act = mean_act * keep
            frac_act = frac_act * keep

            result[layer] = (mean_act, frac_act)
        return result, all_layer_stats

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Extract SAE-based steering vector with shared feature removal for each layer."""
        target_acts, target_stats = self._get_activations(target_data, self.layer)

        if contrast_data is not None:
            contrast_acts, contrast_stats = self._get_activations(
                contrast_data,
                self.layer
            )
            contrast_mean_by_layer = {
                layer: values[0] for layer, values in contrast_acts.items()
            }
        else:
            contrast_acts = None
            contrast_stats = None
            contrast_mean_by_layer = None

        self.vector = {}
        feature_tracker: Dict[int, List[Dict[str, Any]]] = {}

        for layer in self.layer:
            sae = self.sae[layer]
            target_mean, _ = target_acts[layer]
            contrast_mean = _contrast_or_zeros(
                contrast_mean_by_layer,
                layer,
                target_mean,
            )

            # Remove shared features (key SAS step)
            shared_idx = (target_mean != 0) & (contrast_mean != 0)
            target_mean[shared_idx] = 0.0
            contrast_mean[shared_idx] = 0.0

            sparse_latent = target_mean - contrast_mean
            k = min(self.top_k, int((sparse_latent != 0).sum().item()) or self.top_k)
            top_idx = _safe_topk_indices(sparse_latent, k=k, by_abs=True)

            self.sparse_latent[layer] = sparse_latent.to_sparse().coalesce().cpu()
            self.vector[layer] = sae.decode(
                sparse_latent.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)
            ).to(self.device)
            self.top_idx[layer] = top_idx.cpu().tolist()

            feature_tracker[layer] = build_top_feature_tracker_from_stats(
                self.top_idx[layer],
                target_stats=target_stats[layer],
                contrast_stats=contrast_stats[layer] if contrast_stats is not None else None,
                top_n=self.top_k,
            )

        self.metadata = _build_method_metadata(
            "SAS",
            top_idx=self.top_idx,
            sparse_latent=self.sparse_latent,
            feature_tracker=feature_tracker,
        )

        return self.vector


class SPAREExtractor(BaseExtractor):
    """
    SPARE: Steering via Pre-trained SAE Representations.

    Uses mutual information between SAE activations and knowledge
    selection behavior to identify steering features.

    Paper: "SPARE: Sparse Pre-trained SAE Representations for Knowledge Conflicts"
    """

    METHOD_NAME = "SPARE"

    def __init__(
        self,
        model,
        sae,
        layer: List[int],
        top_k_proportion: float = 0.07,
        loss_weight: bool = True,
        top_k: int = 15,
        n_neighbors: int = 3,
        **kwargs,
    ):
        super().__init__(model, layer, **kwargs)
        self.sae = sae
        self.top_k_proportion = top_k_proportion
        self.loss_weight = bool(loss_weight)
        self.top_k = top_k
        self.n_neighbors = int(n_neighbors)

        # FIX A: all per-layer results stored as dicts, not overwritten flat attrs
        self.mutual_info: Dict[int, torch.Tensor] = {}
        self.expectation: Dict[int, torch.Tensor] = {}
        self.indices_pos: Dict[int, torch.Tensor] = {}
        self.indices_neg: Dict[int, torch.Tensor] = {}
        self.top_idx: Dict[int, List[int]] = {}
        self.top_idx_neg: Dict[int, List[int]] = {}
        self.z_contextual: Dict[int, torch.Tensor] = {}
        self.z_parametric: Dict[int, torch.Tensor] = {}
        self.sparse_latent: Dict[int, torch.Tensor] = {}

    def _get_activations(
        self, inputs: List[str], layers: List[int], **kwargs
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, Any]]]:
        """Collect SAE activations and streaming stats in one pass for all layers."""
        collected = collect_sae_activations(
            self.model,
            self.sae,
            inputs,
            layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            tokenizer=self.model.tokenizer,
            active_threshold=0.0,
        )
        return collected.activations, collected.stats

    def _calculate_mutual_information(
        self,
        activations_C: torch.Tensor,
        activations_M: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate mutual information I(Zi; Y) using sklearn mutual_info_classif.
        """
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import MinMaxScaler

        acts = torch.cat([activations_C, activations_M], dim=0)
        labels = np.array([1] * activations_C.shape[0] + [0] * activations_M.shape[0])

        mean_A = activations_C.mean(0)
        mean_B = activations_M.mean(0)
        expectation = mean_A - mean_B

        X = acts.float().cpu().numpy()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info(f"SPARE: Computing MI for {X_scaled.shape[1]} features...")
        mi_scores = mutual_info_classif(
            X_scaled, labels, discrete_features=False, n_neighbors=self.n_neighbors, random_state=42
        )

        return torch.from_numpy(mi_scores).float(), expectation

    @staticmethod
    def _normalize_sample_weights(
        weights: Optional[Union[List[float], np.ndarray, torch.Tensor]],
        n_samples: int,
        name: str,
        device: Union[str, torch.device],
    ) -> Optional[torch.Tensor]:
        """Convert sample weights to normalized non-negative tensor (sum=1)."""
        if weights is None:
            return None

        if isinstance(weights, torch.Tensor):
            w = weights.detach().flatten().to(device=device, dtype=torch.float32)
        else:
            w = torch.tensor(weights, device=device, dtype=torch.float32).flatten()

        if w.numel() != n_samples:
            raise ValueError(
                f"{name} length mismatch: expected {n_samples}, got {w.numel()}"
            )

        w = torch.clamp(w, min=0.0)
        total = w.sum()
        if total <= 0:
            raise ValueError(f"{name} must contain at least one positive value")

        return w / total

    @staticmethod
    def _weighted_mean(
        activations: torch.Tensor,
        normalized_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted mean when weights are provided, otherwise plain mean."""
        if normalized_weights is None:
            return activations.mean(dim=0)

        w = normalized_weights.to(device=activations.device, dtype=activations.dtype)
        return (activations * w.unsqueeze(1)).sum(dim=0)

    @torch.no_grad()
    def _compute_prompt_loss(self, prompt: str) -> float:
        """Compute mean next-token NLL for a single prompt."""
        if not prompt:
            return 0.0

        tokens = self.model.to_tokens(prompt, prepend_bos=True).to(self.device)
        if tokens.shape[1] <= 1:
            return 0.0

        loss = self.model(tokens, return_type="loss")
        return float(loss.item())

    def _derive_loss_ratio_weights(
        self,
        target_data: List[str],
        contrast_data: List[str],
    ) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, Any]]:
        """Derive weights from prompt losses following the GT ratio form."""
        meta: Dict[str, Any] = {
            "source": "derived:prompt_nll",
            "n_pairs": 0,
            "target_loss_mean": None,
            "contrast_loss_mean": None,
        }

        if not target_data or not contrast_data:
            meta["source"] = "missing_prompts"
            return None, None, meta

        if len(target_data) != len(contrast_data):
            raise ValueError(
                "SPARE loss weighting requires paired prompts: "
                f"len(target_data)={len(target_data)} != len(contrast_data)={len(contrast_data)}"
            )

        target_losses = [
            self._compute_prompt_loss(prompt)
            for prompt in tqdm(target_data, desc="SPARE loss(target)", disable=len(target_data) < 32)
        ]
        contrast_losses = [
            self._compute_prompt_loss(prompt)
            for prompt in tqdm(contrast_data, desc="SPARE loss(contrast)", disable=len(contrast_data) < 32)
        ]

        target_weights: List[float] = []
        contrast_weights: List[float] = []
        for target_loss, contrast_loss in zip(target_losses, contrast_losses):
            denom = target_loss + contrast_loss
            if not np.isfinite(denom) or denom <= 0:
                target_weights.append(0.5)
                contrast_weights.append(0.5)
            else:
                target_weights.append(float(target_loss / denom))
                contrast_weights.append(float(contrast_loss / denom))

        meta["n_pairs"] = len(target_weights)
        if target_losses:
            meta["target_loss_mean"] = float(np.mean(target_losses))
        if contrast_losses:
            meta["contrast_loss_mean"] = float(np.mean(contrast_losses))

        return target_weights, contrast_weights, meta

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Extract SPARE steering features using mutual information for each layer."""
        if contrast_data is None:
            raise ValueError("SPARE requires contrast_data (parametric knowledge)")

        weight_loss_meta: Dict[str, Any] = {
            "enabled": bool(self.loss_weight),
            "applied": False,
            "source": "disabled",
        }

        target_weights_raw = None
        contrast_weights_raw = None
        if self.loss_weight:
            target_weights_raw, contrast_weights_raw, derived_meta = self._derive_loss_ratio_weights(
                target_data,
                contrast_data,
            )
            weight_loss_meta.update(derived_meta)

            if target_weights_raw is not None and contrast_weights_raw is not None:
                weight_loss_meta["applied"] = True
                weight_loss_meta.update(
                    {
                        "target": [float(v) for v in target_weights_raw],
                        "contrast": [float(v) for v in contrast_weights_raw],
                    }
                )
            else:
                logger.info(
                    "SPARE: loss_weight=True but prompt-loss derivation was unavailable. "
                    "Falling back to unweighted mean activations."
                )

        all_acts_C, all_stats_C = self._get_activations(target_data, self.layer)
        all_acts_M, all_stats_M = self._get_activations(contrast_data, self.layer)

        self.vector = {}
        n_selected_per_layer: Dict[int, int] = {}
        n_selected_pos_per_layer: Dict[int, int] = {}
        n_selected_neg_per_layer: Dict[int, int] = {}
        feature_tracker: Dict[int, List[Dict[str, Any]]] = {}

        for layer in self.layer:
            sae = self.sae[layer]
            n_latents = sae.W_dec.shape[0]
            acts_C = all_acts_C[layer]
            acts_M = all_acts_M[layer]

            # FIX A: store per layer instead of overwriting flat attrs
            mi, exp = self._calculate_mutual_information(acts_C, acts_M)
            self.mutual_info[layer] = mi
            self.expectation[layer] = exp

            sorted_mi_values, sorted_indices = torch.sort(mi, descending=True)

            # Match reference SPARE: use one cumulative MI budget over all features,
            # then split selected features by expectation sign.
            target_mi = float(sorted_mi_values.sum().item()) * self.top_k_proportion
            cumulative_mi = 0.0
            select_num_activations = 0
            for idx in sorted_indices:
                cumulative_mi += float(mi[idx].item())
                select_num_activations += 1
                if cumulative_mi > target_mi:
                    break

            selected_pos_list: List[int] = []
            selected_neg_list: List[int] = []
            for idx in sorted_indices:
                exp_val = exp[idx]
                if exp_val > 0:
                    selected_pos_list.append(int(idx.item()))
                elif exp_val < 0:
                    selected_neg_list.append(int(idx.item()))

                if (len(selected_pos_list) + len(selected_neg_list)) >= select_num_activations:
                    break

            selected_pos = torch.tensor(
                selected_pos_list,
                dtype=torch.long,
                device=mi.device,
            )
            selected_neg = torch.tensor(
                selected_neg_list,
                dtype=torch.long,
                device=mi.device,
            )

            self.indices_pos[layer] = selected_pos
            self.indices_neg[layer] = selected_neg

            # top_idx is positive-only ranking by request; negatives use top_idx_neg.
            self.top_idx[layer] = selected_pos.detach().cpu().tolist()
            self.top_idx_neg[layer] = selected_neg.detach().cpu().tolist()

            final_selected_indices = (
                torch.cat([selected_pos, selected_neg])
                if (selected_pos.numel() + selected_neg.numel()) > 0
                else torch.empty(0, dtype=torch.long, device=mi.device)
            )

            n_selected_pos_per_layer[layer] = int(selected_pos.numel())
            n_selected_neg_per_layer[layer] = int(selected_neg.numel())
            n_selected_per_layer[layer] = int(final_selected_indices.numel())

            norm_target_weights = self._normalize_sample_weights(
                target_weights_raw,
                n_samples=acts_C.shape[0],
                name="target_weights",
                device=acts_C.device,
            )
            norm_contrast_weights = self._normalize_sample_weights(
                contrast_weights_raw,
                n_samples=acts_M.shape[0],
                name="contrast_weights",
                device=acts_M.device,
            )

            mean_acts_C = self._weighted_mean(acts_C, norm_target_weights)
            mean_acts_M = self._weighted_mean(acts_M, norm_contrast_weights)

            z_ctx = torch.zeros(n_latents, device=self.device, dtype=mean_acts_C.dtype)
            z_par = torch.zeros(n_latents, device=self.device, dtype=mean_acts_M.dtype)

            if len(self.indices_pos[layer]) > 0:
                z_ctx[self.indices_pos[layer]] = mean_acts_C[self.indices_pos[layer]].to(
                    self.device
                )
            if len(self.indices_neg[layer]) > 0:
                z_par[self.indices_neg[layer]] = mean_acts_M[self.indices_neg[layer]].to(
                    self.device
                )

            self.z_contextual[layer] = z_ctx.to_sparse().coalesce().cpu()
            self.z_parametric[layer] = z_par.to_sparse().coalesce().cpu()

            sparse_vector = torch.zeros(n_latents, device=self.device)
            combined_indices = (
                torch.cat([self.indices_pos[layer], self.indices_neg[layer]])
                if len(self.indices_pos[layer]) > 0 or len(self.indices_neg[layer]) > 0
                else torch.tensor([], dtype=torch.long)
            )
            if len(combined_indices) > 0:
                sparse_vector[combined_indices] = exp[combined_indices].to(self.device).float()

            self.sparse_latent[layer] = sparse_vector.to_sparse().coalesce().cpu()
            decode_latent = sparse_vector.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)
            self.vector[layer] = (decode_latent @ sae.W_dec).to(self.device)

            feature_tracker[layer] = build_top_feature_tracker_from_stats(
                self.top_idx[layer],
                target_stats=all_stats_C[layer],
                contrast_stats=all_stats_M[layer],
                top_n=self.top_k,
            )

        self.metadata = _build_method_metadata(
            "SPARE",
            top_idx=self.top_idx,
            top_idx_neg=self.top_idx_neg,
            sparse_latent=self.sparse_latent,
            zC=self.z_contextual,
            zM=self.z_parametric,
            indices_pos=self.indices_pos,
            indices_neg=self.indices_neg,
            n_selected=n_selected_per_layer,
            n_selected_pos=n_selected_pos_per_layer,
            n_selected_neg=n_selected_neg_per_layer,
            n_context={layer: len(v) for layer, v in self.indices_pos.items()},
            n_param={layer: len(v) for layer, v in self.indices_neg.items()},
            uses_weighted_zC=target_weights_raw is not None,
            uses_weighted_zM=contrast_weights_raw is not None,
            weight_loss=weight_loss_meta,
            top_mi_values={layer: self.mutual_info[layer][:10].tolist() for layer in self.layer},
            feature_tracker=feature_tracker,
        )

        return self.vector


class SREExtractor(BaseExtractor):
    """
    Sparse Representation Engineering (SRE) Extractor.

    Identifies positive (I+) and negative (I-) feature indices
    for sparse steering.

    Paper: "Sparse Representation Engineering"
    """

    METHOD_NAME = "SRE"

    def __init__(
        self,
        model,
        sae,
        layer: List[int],
        act_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(model, layer, **kwargs)
        self.sae = sae
        self.act_threshold = act_threshold

        # FIX A: per-layer dicts instead of flat attrs overwritten in loop
        self.I_plus: Dict[int, torch.Tensor] = {}
        self.I_minus: Dict[int, torch.Tensor] = {}
        self.target_sparse: Dict[int, torch.Tensor] = {}
        self.contrast_sparse: Dict[int, torch.Tensor] = {}
        self.sparse_latent: Dict[int, torch.Tensor] = {}

    def _get_activations(
        self, inputs: List[str], layers: List[int], **kwargs
    ) -> Dict[int, torch.Tensor]:
        """Get mean sparse activation at last token, all layers in a single forward pass."""
        all_layer_acts = collect_sae_activations(
            self.model,
            self.sae,
            inputs,
            layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            tokenizer=self.model.tokenizer,
        ).activations
        return {
            layer: acts.to(self.device).mean(dim=0) for layer, acts in all_layer_acts.items()
        }

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Extract sparse feature indices (I+ and I-) for each layer."""
        target_acts = self._get_activations(target_data, self.layer)

        if contrast_data is not None:
            contrast_acts = self._get_activations(contrast_data, self.layer)
        else:
            contrast_acts = None

        self.vector = {}
        for layer in self.layer:
            sae = self.sae[layer]
            t_sparse = target_acts[layer]
            c_sparse = _contrast_or_zeros(contrast_acts, layer, t_sparse)

            # FIX A: store per layer
            self.target_sparse[layer] = t_sparse
            self.contrast_sparse[layer] = c_sparse

            self.I_plus[layer] = (
                (t_sparse > self.act_threshold) & (c_sparse <= self.act_threshold)
            ).nonzero(as_tuple=True)[0]

            self.I_minus[layer] = (
                (c_sparse > self.act_threshold) & (t_sparse <= self.act_threshold)
            ).nonzero(as_tuple=True)[0]

            sparse_latent = t_sparse - c_sparse
            self.sparse_latent[layer] = sparse_latent.to_sparse().coalesce().cpu()
            self.vector[layer] = sae.decode(
                sparse_latent.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)
            ).to(self.device)

        self.metadata = _build_method_metadata(
            "SRE",
            top_idx={
                layer: torch.cat([self.I_plus[layer], self.I_minus[layer]]).detach().cpu().tolist()
                if (self.I_plus[layer].numel() + self.I_minus[layer].numel()) > 0
                else []
                for layer in self.layer
            },
            sparse_latent=self.sparse_latent,
            n_I_plus={layer: len(v) for layer, v in self.I_plus.items()},
            n_I_minus={layer: len(v) for layer, v in self.I_minus.items()},
        )

        return self.vector

    def get_steer_params(self, layer: Optional[int] = None):
        """
        Get SRE-specific params for SRESteerModel.

        Args:
            layer: specific layer to retrieve; defaults to first layer if None.
        """
        # FIX B: was silently reading last-layer flat attrs; now explicitly selects layer
        if layer is None:
            layer = self.layer[0]
        if layer not in self.I_plus:
            raise ValueError(
                f"Layer {layer} not found. Call extract() first. "
                f"Available layers: {list(self.I_plus.keys())}"
            )
        sv = self.get_steering_vector()
        sv.metadata["I_plus"] = self.I_plus[layer]
        sv.metadata["I_minus"] = self.I_minus[layer]
        sv.metadata["target_sparse"] = self.target_sparse[layer]
        sv.metadata["contrast_sparse"] = self.contrast_sparse[layer]
        return sv.to_steer_params()


class SRPSExtractor(BaseExtractor):
    """
    Semantic-aware SAE Steering (SRPS) Extractor.

    Uses semantic token masking to focus on meaningful tokens
    when computing SAE latent averages.

    Paper: "Semantic-Robust Prompt Steering"

    Hyperparameters:
    - beta: Controls balance between activation magnitude and active dimensions
            Higher beta = more weight on activation magnitude
            Paper default: 1.0
    """
    
    METHOD_NAME = "SRPS"

    def __init__(
        self,
        model,
        sae,
        layer: List[int],
        position: str = 'last',  # 'last', 'mean', or 'mask' for token pooling
        act_threshold: float = 0.1,
        top_k: int = 15,
        beta: float = 1.0,  # Now stored at init time from config
        **kwargs,
    ):
        super().__init__(model, layer, position=position, **kwargs)
        self.sae = sae  # Dict[int, SAE]
        self.act_threshold = act_threshold
        self.top_k = top_k
        self.beta = beta

        self.sparse_latent = {}
        self.top_idx = {}

    def _get_activations(self, inputs: List[str], layers: List[int], **kwargs) -> Dict[int, tuple]:
        """Collect SAE activations with semantic token masking for all layers in a single forward pass.

        Paper Eq.1-2:
          µ_i  = (1/N) Σ mean-over-semantic-tokens(a_ij)  per-sample mean, then avg
          δ_i  = f+_i - f-_i  where f_i = fraction of N samples where feature > θ
        """
        collected = collect_sae_activations(
            model=self.model,
            saes=self.sae,
            texts=inputs,
            layers=layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer if self.position == "mask" else None,
            active_threshold=self.act_threshold,
        )
        layer_acts = collected.activations
        layer_stats = collected.stats

        result = {}
        for layer in layers:
            sae = self.sae[layer]
            stats = layer_stats[layer]
            acts = layer_acts[layer].to(self.device, dtype=sae.W_dec.dtype)

            # Eq.1: µ_i = average of per-sample means
            mean_act = acts.mean(dim=0)  # [d_sae]

            # Eq.2: δ_i = frequency difference (fraction of samples with feature > θ)
            freq_active = (acts > float(self.act_threshold)).to(dtype=acts.dtype).mean(dim=0)

            value_variation = ((acts - mean_act) ** 2).mean(dim=0).clamp(min=0.0).sqrt()
            result[layer] = (mean_act, freq_active, value_variation, stats)
        return result

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Extract steering vector with semantic token focus for each layer.

        Uses self.beta from config to balance activation vs active-dimension scoring.
        """
        target_acts = self._get_activations(target_data, self.layer)

        if contrast_data is not None:
            contrast_acts = self._get_activations(contrast_data, self.layer)
            contrast_mean_by_layer = {
                layer: values[0] for layer, values in contrast_acts.items()
            }
            contrast_active_by_layer = {
                layer: values[1] for layer, values in contrast_acts.items()
            }
            contrast_stats_by_layer = {
                layer: values[3] for layer, values in contrast_acts.items()
            }
        else:
            contrast_acts = None
            contrast_mean_by_layer = None
            contrast_active_by_layer = None
            contrast_stats_by_layer = None

        self.vector = {}
        feature_tracker: Dict[int, List[Dict[str, Any]]] = {}
        for layer in self.layer:
            sae = self.sae[layer]
            target_mean, target_active, _, target_stats = target_acts[layer]
            contrast_mean = _contrast_or_zeros(
                contrast_mean_by_layer,
                layer,
                target_mean,
            )
            contrast_active = _contrast_or_zeros(
                contrast_active_by_layer,
                layer,
                target_active,
            )
            contrast_stats = (
                contrast_stats_by_layer[layer]
                if contrast_stats_by_layer is not None
                else None
            )

            # SRPS scoring
            self.acts = target_mean - contrast_mean
            self.actives = target_active - contrast_active
            self.score = self.acts + self.beta * self.actives

            # Top-k selection
            k = min(self.top_k, self.score.numel())
            top_idx = _safe_topk_indices(self.score, k=k, by_abs=False)

            # Sparse latent mask
            sparse_latent = torch.zeros(
                sae.W_dec.shape[0],
                device=self.device,
                dtype=sae.W_dec.dtype,
            )
            sparse_latent[top_idx] = target_mean[top_idx]

            # Decode to dense: s = Σ α_i * W_dec[i,:] (paper Eq.4, no bias)
            self.vector[layer] = sparse_latent @ sae.W_dec.to(self.device)
            self.sparse_latent[layer] = sparse_latent.to_sparse().coalesce().cpu()
            self.top_idx[layer] = top_idx.cpu().tolist()

            feature_tracker[layer] = build_top_feature_tracker_from_stats(
                self.top_idx[layer],
                target_stats=target_stats,
                contrast_stats=contrast_stats,
                top_n=self.top_k,
            )

        self.metadata = _build_method_metadata(
            "SRPS",
            top_idx=self.top_idx,
            sparse_latent=self.sparse_latent,
            feature_tracker=feature_tracker,
        )

        return self.vector


class SSVExtractor(BaseExtractor):
    """
    SAE Supervised Steering Vector (SSV) Extractor.

    Two-stage extraction per the SAE-SSV paper:
    1. Dimension Selection via Probing (ANOVA + linear classifier)
    2. Supervised Steering Vector Optimization (gradient descent with combined loss)

    Paper: "SAE Supervised Steering Vectors"

    Hyperparameters (from paper):
    - ssv_lambda_dist: Distance loss weight (default: 1.0)
    - ssv_lambda_lm:   Language model loss weight (default: 0.5)
    - ssv_lambda_l1:   L1 sparsity regularization weight (default: 0.01)
    - ssv_opt_lr:      Optimization learning rate (default: 0.05)
    - ssv_opt_steps:   Number of optimization steps (default: 100)
    """

    METHOD_NAME = "SSV"

    def __init__(
        self,
        model,
        sae,
        layer: List[int],
        hook_point: List[str] = ["pre"],
        position: Union[int, str] = "last",
        batch_size: int = 8,
        ssv_feature_dim: int = 128,
        ssv_lambda_dist: float = 1.0,
        ssv_lambda_lm: float = 0.5,
        # FIX D: renamed from ssv_lambda_reg to ssv_lambda_l1 to avoid
        # semantic collision with SAETSSExtractor.lambda_reg (correction bias scale)
        ssv_lambda_l1: float = 0.01,
        ssv_opt_lr: float = 0.01,
        ssv_opt_steps: int = 100,
        ssv_feature_refinement_k: int = 30,
        **kwargs,
    ):
        # FIX B: forward **kwargs so extra params are not silently dropped
        super().__init__(
            model,
            layer,
            batch_size=batch_size,
            hook_point=hook_point,
            position=position,
            **kwargs,
        )
        self.sae = sae
        self.feature_dim = ssv_feature_dim

        self.lambda_dist = ssv_lambda_dist
        self.lambda_lm = ssv_lambda_lm
        self.lambda_l1 = ssv_lambda_l1      # FIX D: was lambda_reg
        self.opt_lr = ssv_opt_lr
        self.opt_steps = ssv_opt_steps
        self.refinement_k = ssv_feature_refinement_k

        # FIX A: per-layer dicts instead of flat attrs overwritten in loop
        self.selected_indices: Dict[int, np.ndarray] = {}
        self.mu_plus: Dict[int, torch.Tensor] = {}
        self.mu_minus: Dict[int, torch.Tensor] = {}
        self.sparse_latent: Dict[int, torch.Tensor] = {}

    def _get_activations(
        self, inputs: List[str], layers: List[int], **kwargs
    ) -> Dict[int, List[torch.Tensor]]:
        """Precompute SAE latents for all inputs across all layers in a single forward pass."""
        all_layer_acts = collect_sae_activations(
            self.model,
            self.sae,
            inputs,
            layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            tokenizer=self.model.tokenizer,
        ).activations
        return {
            layer: [acts[i].float() for i in range(acts.shape[0])]
            for layer, acts in all_layer_acts.items()
        }

    def _select_features(
        self,
        latents: List[torch.Tensor],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select top features using ANOVA F-statistic."""
        latents_np = np.stack([l.numpy() for l in latents])
        labels_np = np.array(labels)

        from sklearn.feature_selection import f_classif

        f_scores, _ = f_classif(latents_np, labels_np)
        f_scores[np.isnan(f_scores)] = 0

        selected_indices = np.argsort(f_scores)[-self.feature_dim :]
        return latents_np[:, selected_indices], selected_indices

    def _train_classifier(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_epochs: int = 20,
        lr: float = 1e-4,
    ) -> "torch.nn.Linear":
        """Train linear classifier on selected features."""
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import torch.nn as nn

        n_samples = len(labels)
        test_size = 0.5 if n_samples < 10 else 0.2

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        classifier = nn.Linear(features.shape[1], 2).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=5e-2)
        criterion = nn.CrossEntropyLoss()

        classifier.train()
        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                criterion(classifier(batch_x), batch_y).backward()
                optimizer.step()

        classifier.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            preds = classifier(X_test_t).argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_test, preds)
        logger.info(f"Classifier accuracy: {acc:.2%}")
        return classifier

    def _optimize_steering_vector(
        self,
        latents: List[torch.Tensor],
        labels: List[int],
        selected_indices: np.ndarray,
        all_texts: List[str],
        target_texts: List[str],
        layer: int,
    ) -> torch.Tensor:
        """
        Stage 2: Supervised Steering Vector Optimization with LM Loss (Mini-Batch SGD).

        Distance & sparsity gradients computed analytically (numpy).
        LM gradients computed via numerical finite differences (no autograd through model).
        All model forward passes run under torch.no_grad() to avoid OOM.
        """
        import copy
        import gc

        sae_gpu = self.sae[layer]
        d_sae = sae_gpu.cfg.d_sae

        sae_cpu = copy.deepcopy(sae_gpu).cpu()
        for param in sae_cpu.parameters():
            param.data = param.data.to(torch.float32)
        sae_cpu.eval()

        latents_np = np.array([l.cpu().numpy().astype(np.float32) for l in latents])
        labels_np = np.array(labels)

        mask_np = np.zeros(d_sae, dtype=bool)
        mask_np[selected_indices] = True

        pos_latents = latents_np[labels_np == 1]
        neg_latents = latents_np[labels_np == 0]

        if len(pos_latents) == 0 or len(neg_latents) == 0:
            logger.warning("Missing positive or negative samples for SSV optimization.")
            return torch.zeros(d_sae, device=self.device)

        target_centroid = np.mean(pos_latents, axis=0)
        contrast_centroid = np.mean(neg_latents, axis=0)

        # FIX A: store per layer
        self.mu_plus[layer] = torch.tensor(
            target_centroid[selected_indices], device=self.device
        )
        self.mu_minus[layer] = torch.tensor(
            contrast_centroid[selected_indices], device=self.device
        )

        ssv = np.zeros(d_sae, dtype=np.float32)
        # Initialize according to paper: v_init = mu_plus - mu_minus (target - contrast)
        initial_direction = target_centroid - contrast_centroid
        direction_norm = np.linalg.norm(initial_direction[mask_np])
        if direction_norm > 0:
            ssv[mask_np] = initial_direction[mask_np] / direction_norm

        logger.info(f"Initial direction norm: {direction_norm:.4f}")

        hook_point = get_hook_name(layer, self.hook_point[0])

        # Prepare text pools: contrast_texts are inputs to be steered; target_texts are targets
        target_texts = [all_texts[i] for i in range(len(labels)) if labels[i] == 1]
        contrast_texts = [all_texts[i] for i in range(len(labels)) if labels[i] == 0]

        logger.info(
            f"SSV Stage 2: Optimizing with λ_dist={self.lambda_dist}, "
            f"λ_lm={self.lambda_lm}, λ_l1={self.lambda_l1}, batch_size={self.batch_size}"
        )

        for step in range(self.opt_steps):
            batch_size = min(self.batch_size, len(target_texts), len(contrast_texts))
            truth_idx = np.random.choice(len(target_texts), batch_size, replace=True)
            false_idx = np.random.choice(len(contrast_texts), batch_size, replace=True)

            truth_batch = [target_texts[i] for i in truth_idx]
            false_batch = [contrast_texts[i] for i in false_idx]

            distance_loss = 0.0
            lm_loss = 0.0
            distance_grad = np.zeros_like(ssv)
            lm_grad = np.zeros_like(ssv)
            processed_samples = 0

            for i in range(batch_size):
                try:
                    # Per paper: use negative example as the input to be steered,
                    # and positive example as the LM target sequence.
                    false_tokens = self.model.to_tokens(false_batch[i])
                    truth_tokens = self.model.to_tokens(truth_batch[i])

                    activation = None

                    def get_activation(act, hook):
                        nonlocal activation
                        activation = act[0, -1, :].detach().clone()
                        return act

                    with torch.no_grad():
                        # Extract activation for the negative (to be steered)
                        self.model.run_with_hooks(
                            false_tokens, fwd_hooks=[(hook_point, get_activation)]
                        )
                        if activation is None:
                            continue

                        activation_cpu = activation.cpu().float()
                        neg_latent = (
                            sae_cpu.encode(activation_cpu.unsqueeze(0)).squeeze(0).cpu().numpy()
                        )

                        # Apply steering vector to contrast (negative) latent as in paper
                        steered_latent = neg_latent + ssv

                        # Distance loss per paper: ||z' - target_centroid||^2 - ||z' - contrast_centroid||^2
                        dist = (
                            np.sum((steered_latent - target_centroid) ** 2)
                            - np.sum((steered_latent - contrast_centroid) ** 2)
                        )
                        distance_loss += dist / batch_size

                        # Gradient of distance term: 2*(z' - mu_plus) - 2*(z' - mu_minus)
                        distance_grad += (
                            2 * (steered_latent - target_centroid)
                            - 2 * (steered_latent - contrast_centroid)
                        ) / batch_size

                        try:
                            steered_latent_tensor = torch.tensor(
                                steered_latent, dtype=torch.float32
                            )
                            steered_act = sae_cpu.decode(
                                steered_latent_tensor.unsqueeze(0)
                            ).squeeze(0)
                            steered_act = steered_act.to(activation.device, activation.dtype)

                            def modify_activation(act, hook):
                                act[0, -1, :] = steered_act
                                return act

                            # Run with modified activation for the negative input
                            modified_output = self.model.run_with_hooks(
                                false_tokens,
                                fwd_hooks=[(hook_point, modify_activation)],
                            )

                            # LM loss: cross-entropy of the positive target sequence given steered negative input
                            batch_lm_loss = 0.0
                            token_count = 0
                            for t in range(1, min(truth_tokens.size(1), 20)):
                                if t < modified_output.size(1):
                                    token_logits = modified_output[0, t - 1, :]
                                    token_log_probs = torch.log_softmax(token_logits, dim=0)
                                    target_token_id = truth_tokens[0, t].item()
                                    if target_token_id < token_log_probs.size(0):
                                        batch_lm_loss += -token_log_probs[target_token_id].item()
                                        token_count += 1

                            if token_count > 0:
                                batch_lm_loss /= token_count
                                lm_loss += batch_lm_loss / batch_size

                                # Numerical LM gradient via finite differences over selected dims
                                epsilon = 1e-4
                                for dim in selected_indices:
                                    perturbed_ssv = ssv.copy()
                                    perturbed_ssv[dim] += epsilon

                                    perturbed_latent = neg_latent + perturbed_ssv
                                    perturbed_act = sae_cpu.decode(
                                        torch.tensor(perturbed_latent, dtype=torch.float32).unsqueeze(0)
                                    ).squeeze(0)
                                    perturbed_act = perturbed_act.to(
                                        activation.device, activation.dtype
                                    )

                                    def perturbed_hook(act, hook):
                                        act[0, -1, :] = perturbed_act
                                        return act

                                    perturbed_output = self.model.run_with_hooks(
                                        false_tokens,
                                        fwd_hooks=[(hook_point, perturbed_hook)],
                                    )

                                    perturbed_lm_loss = 0.0
                                    p_token_count = 0
                                    for t in range(1, min(truth_tokens.size(1), 20)):
                                        if t < perturbed_output.size(1):
                                            p_logits = perturbed_output[0, t - 1, :]
                                            p_log_probs = torch.log_softmax(p_logits, dim=0)
                                            target_token_id = truth_tokens[0, t].item()
                                            if target_token_id < p_log_probs.size(0):
                                                perturbed_lm_loss += -p_log_probs[target_token_id].item()
                                                p_token_count += 1

                                    if p_token_count > 0:
                                        perturbed_lm_loss /= p_token_count
                                        lm_grad[dim] += (
                                            perturbed_lm_loss - batch_lm_loss
                                        ) / epsilon / batch_size

                                processed_samples += 1

                        except Exception as e:
                            logger.debug(f"LM loss computation error: {e}")

                except Exception as e:
                    logger.debug(f"Batch sample processing error: {e}")

            # L1 regularization over active subspace
            reg_loss = self.lambda_l1 * np.sum(np.abs(ssv[mask_np]))
            reg_grad = np.zeros_like(ssv)
            reg_grad[mask_np] = self.lambda_l1 * np.sign(ssv[mask_np])

            if processed_samples > 0:
                total_loss = self.lambda_dist * distance_loss + self.lambda_lm * lm_loss + reg_loss
                ssv -= self.opt_lr * (
                    self.lambda_dist * distance_grad
                    + self.lambda_lm * lm_grad
                    + reg_grad
                )
            else:
                total_loss = self.lambda_dist * distance_loss + reg_loss
                ssv -= self.opt_lr * (self.lambda_dist * distance_grad + reg_grad)

            # Enforce zero outside selected mask
            ssv[~mask_np] = 0

            if (step + 1) % 5 == 0 or step == 0:
                logger.info(
                    f"  Step {step+1}/{self.opt_steps}: "
                    f"L_dist={distance_loss:.4f}, L_lm={lm_loss:.4f}, "
                    f"L_l1={reg_loss:.4f}, total={total_loss:.4f}"
                )

        logger.info(
            f"SSV Stage 2 optimization complete: final loss={total_loss:.4f}, "
            f"SSV norm={np.linalg.norm(ssv):.4f}"
        )

        del sae_cpu
        gc.collect()

        return torch.tensor(ssv, device=self.device, dtype=torch.float32)

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract SSV using two-stage process per SAE-SSV paper, for each layer.

        Stage 1: Dimension selection via probing (ANOVA + classifier)
        Stage 2: Supervised steering vector optimization
        """
        if contrast_data is None:
            raise ValueError("SSV requires contrast_data")

        all_texts = target_data + contrast_data
        if labels is None:
            labels = [1] * len(target_data) + [0] * len(contrast_data)

        logger.info("SSV Stage 1: Computing SAE latents...")
        all_latents = self._get_activations(all_texts, self.layer)

        self.vector = {}
        for layer in self.layer:
            latents = all_latents[layer]

            logger.info(f"[Layer {layer}] SSV Stage 1: Feature selection via ANOVA F-statistic...")
            selected_features, layer_selected_indices = self._select_features(latents, labels)

            logger.info(
                f"[Layer {layer}] SSV Stage 1b: Refining features "
                f"(selecting top {self.refinement_k})..."
            )

            mean = np.mean(selected_features, axis=0)
            std = np.std(selected_features, axis=0)
            std[std == 0] = 1.0
            normalized_features = (selected_features - mean) / std

            classifier = self._train_classifier(
                features=normalized_features,
                labels=labels,
                num_epochs=13,
            )

            w = classifier.weight.detach()
            diff_vector = w[1] - w[0]
            importance = diff_vector.abs()
            top_k = min(self.refinement_k, len(importance))
            top_k_indices = _safe_topk_indices(
                importance,
                k=top_k,
                by_abs=False,
            ).cpu().numpy()

            # FIX A: store per layer
            self.selected_indices[layer] = layer_selected_indices[top_k_indices]
            logger.info(f"[Layer {layer}] Refinement complete. Selected top {top_k} features.")

            logger.info(f"[Layer {layer}] SSV Stage 2: Optimizing steering vector with LM loss...")
            sparse_latent = self._optimize_steering_vector(
                latents=latents,
                labels=labels,
                selected_indices=self.selected_indices[layer],
                all_texts=all_texts,
                target_texts=target_data,
                layer=layer,
            )
            self.sparse_latent[layer] = sparse_latent.to_sparse().coalesce().cpu()
            self.vector[layer] = self.sae[layer].decode(
                sparse_latent.to(
                    device=self.sae[layer].W_dec.device,
                    dtype=self.sae[layer].W_dec.dtype,
                )
            ).to(self.device)

        self.metadata = _build_method_metadata(
            "SSV",
            top_idx={
                layer: idx.tolist() if isinstance(idx, np.ndarray) else idx
                for layer, idx in self.selected_indices.items()
            },
            sparse_latent=self.sparse_latent,
            feature_dim=self.feature_dim,
            optimization_steps=self.opt_steps,
            lambda_dist=self.lambda_dist,
            lambda_lm=self.lambda_lm,
            lambda_l1=self.lambda_l1,  # FIX D: renamed key in metadata too
        )

        return self.vector


class SAERSVExtractor(BaseExtractor):
    """
    SAE Residual Steering Vector (SAE-RSV) Extractor.

    Combines SAE feature selection with residual stream steering.

    Paper: "SAE-based Residual Steering Vectors"
    """

    METHOD_NAME = "SAE-RSV"

    def __init__(
        self,
        model,
        sae,
        layer: List[int],
        batch_size: int = 8,
        top_k: int = 50,
        **kwargs,
    ):
        # Style: pass batch_size as keyword arg, consistent with all other extractors.
        # Original positional form super().__init__(model, layer, batch_size, ...) was
        # also correct (batch_size is the 3rd param of BaseExtractor), but keyword is
        # more explicit and prevents silent mismatches if BaseExtractor's signature shifts.
        super().__init__(model, layer, batch_size=batch_size, **kwargs)
        self.sae = sae
        self.top_k = top_k
        self.top_idx: Dict[int, List[int]] = {}
        self.sparse_latent: Dict[int, torch.Tensor] = {}

    def _get_activations(
        self, inputs: List[str], layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Get SAE latents for inputs across all layers in a single forward pass. Returns [n_samples, d_sae]."""
        all_layer_acts = collect_sae_activations(
            self.model,
            self.sae,
            inputs,
            layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            tokenizer=self.model.tokenizer,
        ).activations
        return all_layer_acts

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Extract SAE-RSV steering vector for each layer."""
        target_latents = self._get_activations(target_data, self.layer)

        if contrast_data is not None:
            contrast_latents = self._get_activations(contrast_data, self.layer)
        else:
            contrast_latents = None

        self.vector = {}

        for layer in self.layer:
            sae = self.sae[layer]
            t_latents = target_latents[layer]
            c_latents = _contrast_or_zeros(contrast_latents, layer, t_latents)

            target_mean = t_latents.mean(dim=0)
            contrast_mean = c_latents.mean(dim=0)
            latent_diff = target_mean - contrast_mean

            k = min(self.top_k, latent_diff.numel())
            top_indices = _safe_topk_indices(latent_diff, k=k, by_abs=True)

            sparse_latent = torch.zeros_like(latent_diff)
            sparse_latent[top_indices] = latent_diff[top_indices]

            # Base steering vector in model space
            v_steer = sparse_latent @ sae.W_dec.to(self.device)

            # Optional refinement: compute noise/useful vectors when indices are provided.
            # Users can pass `noise_idx`, `useful_idx`, `alpha1`, `alpha2`, `alpha3` via kwargs.
            noise_idx = kwargs.get("noise_idx", None)
            useful_idx = kwargs.get("useful_idx", None)
            alpha1 = float(kwargs.get("alpha1", 1.0))
            alpha2 = float(kwargs.get("alpha2", 0.5))
            alpha3 = float(kwargs.get("alpha3", 0.5))

            device = self.device
            v_noise = torch.zeros_like(v_steer)
            v_useful = torch.zeros_like(v_steer)

            if noise_idx is not None and len(noise_idx) > 0:
                idx_t = torch.as_tensor(noise_idx, dtype=torch.long, device=latent_diff.device)
                # mean activations across target latents as weights
                mean_acts = t_latents.mean(dim=0).detach().float()
                alphas = mean_acts[idx_t].to(device)
                if alphas.sum() > 0:
                    alphas = alphas / alphas.sum()
                else:
                    alphas = torch.ones_like(alphas, device=device) / alphas.numel()
                dec = sae.W_dec.to(device)
                # weighted sum of decoder rows for noise vector
                v_noise = (alphas.unsqueeze(-1) * dec[idx_t]).sum(dim=0)

            if useful_idx is not None and len(useful_idx) > 0:
                idx_u = torch.as_tensor(useful_idx, dtype=torch.long, device=latent_diff.device)
                dec = sae.W_dec.to(device)
                # simple average of decoder rows for useful vector
                v_useful = dec[idx_u].mean(dim=0)

            # Combine according to SAE-RSV formula (paper): v' = α1·v_steer − α2·v_noise + α3·v_useful
            v_prime = alpha1 * v_steer - alpha2 * v_noise + alpha3 * v_useful

            self.vector[layer] = v_prime
            self.sparse_latent[layer] = sparse_latent.to_sparse().coalesce().cpu()
            self.top_idx[layer] = top_indices.detach().cpu().tolist()

            # Add refinement metadata for downstream steering code and inspection
            self._sae_rsv_meta = {
                "alpha1": alpha1,
                "alpha2": alpha2,
                "alpha3": alpha3,
                "noise_idx": list(noise_idx) if noise_idx is not None else [],
                "useful_idx": list(useful_idx) if useful_idx is not None else [],
            }

        # Include SAE-RSV specific refinement metadata if present
        meta_kwargs = {
            "top_idx": self.top_idx,
            "sparse_latent": self.sparse_latent,
        }
        if hasattr(self, "_sae_rsv_meta"):
            meta_kwargs["sae_rsv_meta"] = self._sae_rsv_meta

        self.metadata = _build_method_metadata("SAE-RSV", **meta_kwargs)

        return self.vector


class LinearAdapter(torch.nn.Module):
    """
    Linear Effect Approximator for SAE-TS.

    Maps steering vectors to their effects on SAE feature activations.
    y_hat = x @ W + b

    Reference: SAE-TS/src/sae_ts/ft_effects/utils.py
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_model, d_sae))
        )
        self.b = torch.nn.Parameter(torch.zeros(d_sae))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W + self.b


class SAETSSExtractor(BaseExtractor):
    """
    SAE Token-Specific Steering (SAE-TS) Extractor.

    Implements the paper's "Targeted Steering" algorithm:
    1. Train LinearAdapter (Effect Approximator) on (steering_vector, effect) pairs
    2. Identify target feature j from input data
    3. Compute steering vector: s = M_j/||M_j|| - λ * M_b/||M_b||

    Reference: SAE-TS/src/sae_ts/ft_effects/train.py
    """

    METHOD_NAME = "SAE-TS"

    SAETS_TRAIN_BATCH_SIZE = 64
    SAETS_EFFECTS_HF_REPO = "schalnev/sae-ts-effects"
    SAETS_EFFECTS_HF_FILE = "effects_2b.pt"

    def __init__(
        self,
        model,
        sae,
        layer: List[int],
        batch_size: int = 8,
        saets_lr: float = 2e-4,
        saets_epochs: int = 15,
        # FIX D: renamed saets_lambda -> saets_bias_scale to make semantics explicit
        # (this is a correction-bias scale, not an L1 penalty like SSVExtractor.lambda_l1)
        saets_bias_scale: float = 1.0,
        saets_adapter_path: Optional[str] = None,
        target_features: Optional[Dict[int, List[Tuple[int, float]]]] = None,
        saets_seed: Optional[int] = 42,
        saets_effects_n_samples: int = 512,
        saets_effects_loss_batch_size: int = 64,
        saets_effects_feature_batch_size: int = 32,
        saets_effects_baseline_batches: int = 10,
        saets_effects_steer_batches: int = 1,
        hook_point: List[str] = None,
        position: Union[int, str] = "last",
        **kwargs,
    ):
        if hook_point is None:
            hook_point = ["post"]
        super().__init__(model, layer, batch_size, hook_point=hook_point, **kwargs)
        self.sae = sae
        self.lr = saets_lr
        self.epochs = saets_epochs
        # FIX D: attribute name is now bias_scale (was lambda_reg, clashing with SSVExtractor)
        self.bias_scale = saets_bias_scale
        self.adapter_path = saets_adapter_path
        self.target_features = target_features
        self.seed = saets_seed
        self.effects_n_samples = saets_effects_n_samples
        self.effects_loss_batch_size = saets_effects_loss_batch_size
        self.effects_feature_batch_size = saets_effects_feature_batch_size
        self.effects_baseline_batches = saets_effects_baseline_batches
        self.effects_steer_batches = saets_effects_steer_batches
        self.adapters: Dict[int, LinearAdapter] = {}
        self.sparse_latent: Dict[int, torch.Tensor] = {}

    def _seed_rng(self) -> None:
        if self.seed is None:
            return
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def _normalize_rows(features: torch.Tensor) -> torch.Tensor:
        return features / (features.norm(dim=-1, keepdim=True) + 1e-8)

    def _build_target_vectors(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]],
    ) -> Tuple[Dict[int, int], Dict[int, List[Tuple[int, float]]], Dict[int, torch.Tensor]]:
        feature_indices: Dict[int, int] = {}
        feature_weights: Dict[int, List[Tuple[int, float]]] = {}
        target_vectors: Dict[int, torch.Tensor] = {}

        if self.target_features is None:
            # FIX B: _identify_target_features replaces the misleadingly-named _get_activations
            inferred = self._identify_target_features(target_data, contrast_data)
            target_feature_lists = {layer: [(int(inferred[layer]), 1.0)] for layer in self.layer}
        else:
            logger.info(f"SAE-TS: Using provided target feature lists: {self.target_features}")
            target_feature_lists = self.target_features

        for layer in self.layer:
            features = target_feature_lists.get(layer, [])
            if not features:
                raise ValueError(f"SAE-TS target_features missing for layer {layer}")

            d_sae = self.sae[layer].W_enc.shape[1]
            target_vector = torch.zeros(d_sae, device=self.device, dtype=torch.float32)
            normalized_features = [(int(index), float(scale)) for index, scale in features]
            for index, scale in normalized_features:
                target_vector[index] = scale

            feature_indices[layer] = max(normalized_features, key=lambda item: abs(item[1]))[0]
            feature_weights[layer] = normalized_features
            target_vectors[layer] = target_vector

        return feature_indices, feature_weights, target_vectors

    def _get_adapter(self, layer: int, target_data: List[str]) -> LinearAdapter:
        if layer not in self.adapters:
            if self.adapter_path is not None and os.path.exists(self.adapter_path):
                self.adapters[layer] = self._load_adapter(layer)
            else:
                self.adapters[layer] = self._train_adapter(target_data, layer)
        return self.adapters[layer]

    def _load_effects_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load effects data from author-provided HF dataset."""
        hf_path = hf_hub_download(
            repo_id=self.SAETS_EFFECTS_HF_REPO,
            filename=self.SAETS_EFFECTS_HF_FILE,
            repo_type="dataset",
            force_download=False,
        )
        logger.info(
            f"SAE-TS: Loading effects data from "
            f"{self.SAETS_EFFECTS_HF_REPO}/{self.SAETS_EFFECTS_HF_FILE}"
        )
        data = torch.load(hf_path, map_location="cpu")
        features = self._normalize_rows(data["features"].float())
        effects = data["effects"].float()
        return features.to(self.device), effects.to(self.device)

    def _split_effects_data(
        self,
        features: torch.Tensor,
        effects: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Split effects data into train/validation sets."""
        n_val = min(100, max(features.shape[0] - 1, 0))
        if n_val == 0:
            return features, effects, None, None
        return features[:-n_val], effects[:-n_val], features[-n_val:], effects[-n_val:]

    def _train_adapter(self, target_data: List[str], layer: int) -> LinearAdapter:
        """Train LinearAdapter on effects data."""
        sae = self.sae[layer]
        d_model = sae.W_enc.shape[0]
        d_sae = sae.W_enc.shape[1]

        features, effects = self._load_effects_data()
        features, effects, val_features, val_effects = self._split_effects_data(features, effects)

        adapter = LinearAdapter(d_model, d_sae).to(self.device)
        optimizer = torch.optim.Adam(adapter.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        n_samples = features.shape[0]
        batch_size = self.SAETS_TRAIN_BATCH_SIZE
        n_val = 0 if val_features is None else val_features.shape[0]

        logger.info(
            f"SAE-TS: Training adapter on {n_samples} samples "
            f"({n_val} held out for validation) for {self.epochs} epochs..."
        )

        self._seed_rng()

        dataset = torch.utils.data.TensorDataset(features, effects)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if val_features is not None and val_effects is not None:
            val_dataset = torch.utils.data.TensorDataset(val_features, val_effects)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            val_dataloader = None

        for epoch in range(self.epochs):
            adapter.train()
            total_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(adapter(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 5 == 0:
                train_msg = f"  Epoch {epoch+1}/{self.epochs}, Train Loss: {total_loss/n_batches:.6f}"
                if n_val > 0:
                    adapter.eval()
                    with torch.no_grad():
                        val_total = 0.0
                        val_batches = 0
                        for val_x, val_y in val_dataloader:
                            val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                            val_total += torch.nn.functional.mse_loss(
                                adapter(val_x), val_y
                            ).item()
                            val_batches += 1
                        train_msg += f", Val Loss: {val_total / max(val_batches, 1):.6f}"
                logger.info(train_msg)

        if self.adapter_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.adapter_path)), exist_ok=True)
            logger.info(f"SAE-TS: Saving trained adapter to {self.adapter_path}")
            torch.save(adapter.state_dict(), self.adapter_path)

        return adapter

    def _load_adapter(self, layer: int) -> LinearAdapter:
        """Load pre-trained LinearAdapter."""
        sae = self.sae[layer]
        d_model = sae.W_enc.shape[0]
        d_sae = sae.W_enc.shape[1]

        logger.info(f"SAE-TS: Loading adapter from {self.adapter_path}")
        adapter = LinearAdapter(d_model, d_sae).to(self.device)
        adapter.load_state_dict(torch.load(self.adapter_path, map_location=self.device))
        return adapter

    def _calculate_correction_bias(self, adapter: LinearAdapter) -> torch.Tensor:
        """
        Calculate correction bias direction M_b.

        M_b = W @ b (normalized to unit norm)
        Represents the bias contribution projected back to steering space.
        """
        b = adapter.W @ adapter.b
        return b / (b.norm() + 1e-8)

    def _identify_target_features(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]],
    ) -> Dict[int, int]:
        """
        Identify target feature index for all layers in a single pass.

        FIX B: renamed from _get_activations — this method does feature selection,
        not raw activation collection. The old name clashed with the standard
        _get_activations(inputs, layers) -> Dict[int, Tensor] contract.

        Returns:
            Dict mapping layer -> feature index (int)
        """
        target_acts_dict = collect_sae_activations(
            self.model,
            self.sae,
            target_data,
            self.layer,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            tokenizer=self.model.tokenizer,
        ).activations

        contrast_acts_dict = None
        if contrast_data and len(contrast_data) > 0:
            contrast_acts_dict = collect_sae_activations(
                self.model,
                self.sae,
                contrast_data,
                self.layer,
                hook_point=self.hook_point[0],
                batch_size=self.batch_size,
                pooling=self.position,
                tokenizer=self.model.tokenizer,
            ).activations

        result: Dict[int, int] = {}
        for layer in self.layer:
            target_mean = target_acts_dict[layer].to(self.device).mean(dim=0)
            if contrast_acts_dict is not None:
                diff = target_mean - contrast_acts_dict[layer].to(self.device).mean(dim=0)
            else:
                diff = target_mean
            top_idx = diff.argmax().item()
            logger.info(
                f"SAE-TS: [Layer {layer}] Selected target feature {top_idx} "
                f"(value: {diff[top_idx].item():.4f})"
            )
            result[layer] = top_idx
        return result

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract SAE-TS steering vector for each layer.

        Algorithm (per paper):
        1. Train adapter if not already trained
        2. Identify target feature j from data
        3. Compute s = M_j/||M_j|| - bias_scale * M_b/||M_b||
        4. Normalize s
        """
        self.vector = {}
        self.sparse_latent = {}

        feature_indices, feature_weights, target_vectors = self._build_target_vectors(
            target_data, contrast_data
        )

        for layer in self.layer:
            sae = self.sae[layer]
            adapter = self._get_adapter(layer, target_data)

            M_j = adapter.W @ target_vectors[layer].to(adapter.W.device)
            M_j_norm = M_j / (M_j.norm() + 1e-8)

            M_b_norm = self._calculate_correction_bias(adapter)

            # FIX D: use self.bias_scale (was self.lambda_reg)
            s = M_j_norm - self.bias_scale * M_b_norm

            self.vector[layer] = s / (s.norm() + 1e-8)
            self.sparse_latent[layer] = sae.encode(self.vector[layer].to(sae.W_dec.device)).to_sparse().coalesce().cpu()

        self.metadata = _build_method_metadata(
            "SAE-TS",
            top_idx={
                layer: [int(feature_indices[layer])]
                for layer in self.layer
                if layer in feature_indices
            },
            sparse_latent=self.sparse_latent,
            position=self.position,
            feature_idx=feature_indices,
            feature_weights=feature_weights,
            # FIX D: renamed key to match attribute
            bias_scale=self.bias_scale,
            has_adapter=bool(self.adapters),
            adapter_path=self.adapter_path,
        )

        logger.info("SAE-TS: Steering vector computed")
        return self.vector


class StreamingCorrelationAccumulator:
    """
    Optimized streaming accumulator (device/dtype-aware).
    Key points:
      - Tensors allocated once on chosen device/dtype
      - Vectorized updates
      - MI histogram uses torch.bucketize + index_add_
    """

    def __init__(
        self,
        dict_size: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        real: bool = False,
        mi_edges: Optional[torch.Tensor] = None,
        fisher_enabled: bool = False,
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.dict_size = dict_size
        self.real = real
        self.mi_enabled = mi_edges is not None
        self.fisher_enabled = fisher_enabled

        self.sum_x = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.sum_xx = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.sum_xy = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.sum_y = 0.0
        self.sum_yy = 0.0
        self.n = 0

        self.sum_x_pos = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.count_pos = 0
        self.sum_x_pos_active = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.count_pos_active = torch.zeros(dict_size, dtype=torch.int64, device=self.device)

        self.sum_x_success = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.sum_x_failure = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.sum_xx_success = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.sum_xx_failure = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.count_success = 0
        self.count_failure = 0

        self.sum_x_success_active = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.sum_x_failure_active = torch.zeros(dict_size, dtype=self.dtype, device=self.device)
        self.count_success_active = torch.zeros(
            dict_size, dtype=torch.int64, device=self.device
        )
        self.count_failure_active = torch.zeros(
            dict_size, dtype=torch.int64, device=self.device
        )

        if self.mi_enabled:
            self.mi_edges = mi_edges.to(device=self.device, dtype=self.dtype)
            self.num_bins = len(self.mi_edges)
            self.mi_counts_success = torch.zeros(
                dict_size, self.num_bins, dtype=torch.int64, device=self.device
            )
            self.mi_counts_failure = torch.zeros(
                dict_size, self.num_bins, dtype=torch.int64, device=self.device
            )
        else:
            self.mi_edges = None
            self.num_bins = 0

    @torch.no_grad()
    def update_corr(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
        if batch_x.numel() == 0:
            return
        batch_x = batch_x.to(device=self.device, dtype=self.dtype, copy=False)
        batch_y = batch_y.to(device=self.device, dtype=self.dtype, copy=False)

        B = batch_x.shape[0]
        self.n += int(B)
        self.sum_x += batch_x.sum(dim=0)
        self.sum_xx += (batch_x * batch_x).sum(dim=0)
        y_col = batch_y.view(-1, 1)
        self.sum_xy += (batch_x * y_col).sum(dim=0)
        self.sum_y += float(batch_y.sum().item())
        self.sum_yy += float((batch_y * batch_y).sum().item())

    @torch.no_grad()
    def update_coeff(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
        if batch_x.numel() == 0:
            return
        batch_x = batch_x.to(device=self.device, dtype=self.dtype, copy=False)
        batch_y = batch_y.to(device=self.device, dtype=self.dtype, copy=False)

        pos_mask_bool = (batch_y > 0.0).view(-1)
        if pos_mask_bool.any():
            n_pos = int(pos_mask_bool.sum().item())
            self.count_pos += n_pos
            pos_mask = pos_mask_bool.to(dtype=self.dtype).unsqueeze(1)
            self.sum_x_pos += (batch_x * pos_mask).sum(dim=0)
            pos_active_mask = ((batch_x > 0.0) & pos_mask_bool.view(-1, 1)).to(
                dtype=self.dtype
            )
            self.sum_x_pos_active += (batch_x * pos_active_mask).sum(dim=0)
            self.count_pos_active += (
                (batch_x > 0.0) & pos_mask_bool.view(-1, 1)
            ).sum(dim=0).to(dtype=torch.int64)

        success_mask_bool = (batch_y > 0.0).view(-1)
        failure_mask_bool = (batch_y == 0.0).view(-1)

        if success_mask_bool.any():
            n_success = int(success_mask_bool.sum().item())
            self.count_success += n_success
            succ_mask = success_mask_bool.to(dtype=self.dtype).unsqueeze(1)
            self.sum_x_success += (batch_x * succ_mask).sum(dim=0)
            self.sum_xx_success += ((batch_x * batch_x) * succ_mask).sum(dim=0)
            active_mask = (batch_x > 0.0) & success_mask_bool.view(-1, 1)
            if active_mask.any():
                self.sum_x_success_active += (
                    batch_x * active_mask.to(dtype=self.dtype)
                ).sum(dim=0)
                self.count_success_active += active_mask.sum(dim=0).to(dtype=torch.int64)

        if failure_mask_bool.any():
            n_failure = int(failure_mask_bool.sum().item())
            self.count_failure += n_failure
            fail_mask = failure_mask_bool.to(dtype=self.dtype).unsqueeze(1)
            self.sum_x_failure += (batch_x * fail_mask).sum(dim=0)
            self.sum_xx_failure += ((batch_x * batch_x) * fail_mask).sum(dim=0)
            active_mask = (batch_x > 0.0) & failure_mask_bool.view(-1, 1)
            if active_mask.any():
                self.sum_x_failure_active += (
                    batch_x * active_mask.to(dtype=self.dtype)
                ).sum(dim=0)
                self.count_failure_active += active_mask.sum(dim=0).to(dtype=torch.int64)

    @torch.no_grad()
    def update_mi(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
        if not self.mi_enabled or batch_x.numel() == 0:
            return
        batch_x = batch_x.to(device=self.device, dtype=self.dtype, copy=False)
        batch_y = batch_y.to(device=self.device, dtype=self.dtype, copy=False)
        B, D = batch_x.shape

        bins = torch.bucketize(batch_x, self.mi_edges)
        bins = torch.clamp(bins, min=0, max=self.num_bins - 1).to(
            dtype=torch.int64, device=self.device
        )

        feat_idx = (
            torch.arange(D, device=self.device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(B, D)
            .reshape(-1)
        )
        bins_flat = bins.reshape(-1)
        linear_idx = feat_idx * self.num_bins + bins_flat

        succ_flag = (batch_y > 0.0).view(-1).to(dtype=torch.int64, device=self.device)
        succ_rep = succ_flag.unsqueeze(1).expand(-1, D).reshape(-1)

        ones = torch.ones(linear_idx.shape[0], dtype=torch.int64, device=self.device)
        flat_success = self.mi_counts_success.view(-1)
        flat_failure = self.mi_counts_failure.view(-1)

        if succ_rep.any():
            idx_succ = linear_idx[succ_rep.bool()]
            flat_success.index_add_(0, idx_succ, ones[succ_rep.bool()])

        if (~succ_rep.bool()).any():
            idx_fail = linear_idx[~succ_rep.bool()]
            flat_failure.index_add_(0, idx_fail, ones[~succ_rep.bool()])

    @torch.no_grad()
    def correlations(self) -> torch.Tensor:
        if self.n == 0:
            return torch.zeros(self.dict_size, dtype=self.dtype, device=self.device)
        n = float(self.n)
        mean_y = self.sum_y / n
        var_y = self.sum_yy / n - mean_y * mean_y
        mean_x = self.sum_x / n
        var_x = self.sum_xx / n - mean_x * mean_x
        cov_xy = self.sum_xy / n - mean_x * mean_y
        denom = torch.sqrt(var_x.clamp(min=1e-12) * max(var_y, 1e-12))
        r = torch.where(denom > 0.0, cov_xy / denom, torch.zeros_like(cov_xy))
        return r.to(dtype=self.dtype)

    @torch.no_grad()
    def fisher_scores(self) -> torch.Tensor:
        eps = 1e-12
        c1 = max(self.count_success, 1)
        c0 = max(self.count_failure, 1)
        return (self.sum_xx_success / c1 / torch.clamp(self.sum_xx_failure / c0, min=eps)).to(
            dtype=self.dtype
        )

    @torch.no_grad()
    def mi_scores(self) -> torch.Tensor:
        if not self.mi_enabled:
            r = self.correlations().clamp(min=-0.999999, max=0.999999)
            return (-0.5 * torch.log1p(-(r * r))).to(dtype=self.dtype)

        eps = 1e-12
        counts1 = self.mi_counts_success.to(dtype=self.dtype)
        counts0 = self.mi_counts_failure.to(dtype=self.dtype)
        n1 = counts1.sum(dim=1)
        n0 = counts0.sum(dim=1)
        N = n1 + n0
        valid = N > 0

        p1 = torch.zeros_like(N)
        p0 = torch.zeros_like(N)
        p1[valid] = n1[valid] / N[valid]
        p0[valid] = n0[valid] / N[valid]

        pz = torch.zeros_like(counts1)
        pz[valid] = (counts1[valid] + counts0[valid]) / N[valid].unsqueeze(1)
        pz1 = torch.zeros_like(counts1)
        pz0 = torch.zeros_like(counts0)
        pz1[valid] = counts1[valid] / N[valid].unsqueeze(1)
        pz0[valid] = counts0[valid] / N[valid].unsqueeze(1)

        def safe_term(pzy, pz_, py):
            num = pzy.clamp(min=eps)
            den = pz_.clamp(min=eps) * py.clamp(min=eps).unsqueeze(1)
            return num * torch.log(num / den)

        I = (safe_term(pz1, pz, p1) + safe_term(pz0, pz, p0)).sum(dim=1)
        I[~valid] = 0.0
        return I.to(dtype=self.dtype)

    @torch.no_grad()
    def contrastive_diff(self) -> torch.Tensor:
        c1 = max(self.count_success, 1)
        c0 = max(self.count_failure, 1)
        mu1 = self.sum_x_success / c1
        mu0 = self.sum_x_failure / c0
        return (mu1 - mu0).abs().to(dtype=self.dtype)

    @torch.no_grad()
    def mean_coefficient(self, feature_idx: int) -> float:
        if self.real:
            cnt = int(self.count_pos_active[feature_idx].item())
            if cnt == 0:
                return 0.0
            return float((self.sum_x_pos_active[feature_idx] / cnt).item())
        if self.count_pos == 0:
            return 0.0
        return float((self.sum_x_pos[feature_idx] / self.count_pos).item())

    @torch.no_grad()
    def mean_coefficient_success(self, feature_idx: int) -> float:
        if self.real:
            cnt = int(self.count_success_active[feature_idx].item())
            if cnt == 0:
                return 0.0
            return float((self.sum_x_success_active[feature_idx] / cnt).item())
        if self.count_success == 0:
            return 0.0
        return float((self.sum_x_success[feature_idx] / self.count_success).item())

    @torch.no_grad()
    def mean_coefficient_failure(self, feature_idx: int) -> float:
        if self.real:
            cnt = int(self.count_failure_active[feature_idx].item())
            if cnt == 0:
                return 0.0
            return float((self.sum_x_failure_active[feature_idx] / cnt).item())
        if self.count_failure == 0:
            return 0.0
        return float((self.sum_x_failure[feature_idx] / self.count_failure).item())

    @torch.no_grad()
    def get_ranking_scores(self, selection: str = "correlation") -> torch.Tensor:
        if selection == "mi":
            return self.mi_scores()
        elif selection == "fisher":
            return self.fisher_scores()
        elif selection == "caa":
            return self.contrastive_diff()
        else:
            return self.correlations().abs()

    @torch.no_grad()
    def top_features_signed(
        self,
        k: int = 1,
        pos_only: bool = True,
        neg_only: bool = False,
        selection: str = "correlation",
        caacoeff: bool = False,
        scale: float = 1.0,
        real: bool = False,
    ) -> List[dict]:
        r = self.correlations()
        scores = self.get_ranking_scores(selection)

        if pos_only:
            mask_pos = r > 0
            mask_neg = torch.zeros_like(mask_pos)
        elif neg_only:
            mask_pos = torch.zeros_like(r, dtype=torch.bool)
            mask_neg = r < 0
        else:
            mask_pos = r > 0
            mask_neg = r < 0

        neg_inf = torch.tensor(-float("inf"), device=r.device, dtype=scores.dtype)

        def _build_topk(mask: torch.Tensor, k_: int, negate_coeff: bool = False) -> List[dict]:
            if not mask.any():
                return []
            masked_scores = torch.where(mask, scores, neg_inf)
            available = int(mask.sum().item())
            take = min(k_, available)
            top_idx = _safe_topk_indices(masked_scores, k=take, by_abs=False)
            results = []
            for idx_t in top_idx:
                i = int(idx_t.item())
                if caacoeff:
                    coeff = self.mean_coefficient_success(i) - self.mean_coefficient_failure(i)
                else:
                    coeff = self.mean_coefficient(i)
                final_coeff = float(coeff) * scale
                if negate_coeff:
                    final_coeff = -abs(final_coeff)
                results.append(
                    {
                        "feature_index": i,
                        "correlation": float(r[i].item()),
                        "coefficient": final_coeff,
                    }
                )
            return results

        pos_features = _build_topk(mask_pos, k)
        neg_features = _build_topk(mask_neg, k, negate_coeff=True)

        if pos_only:
            return pos_features
        elif neg_only:
            return neg_features
        else:
            return pos_features if pos_features else neg_features


class UnsteeredModelWrapper:
    """Adapts HookedTransformer to steer_model interface for evaluator.batch()."""

    def __init__(self, model, max_new_tokens: int = 1):
        self.model = model
        self._max_new_tokens = max_new_tokens
        self.metadata = []

    def generate(self, prompt, coeff=None, max_new_tokens=None, **kwargs):
        max_new_tokens = max_new_tokens or self._max_new_tokens
        prompts = [prompt] if isinstance(prompt, str) else prompt

        with torch.no_grad():
            self.model.tokenizer.padding_side = "left"
            orig_pad = self.model.tokenizer.pad_token_id
            self.model.tokenizer.pad_token_id = 0
            input_tokens = self.model.to_tokens(prompts)

            try:
                if max_new_tokens <= 1:
                    gen_texts = self._greedy_decode(input_tokens)
                else:
                    gen_texts = self._multi_token_generate(input_tokens, max_new_tokens)
            finally:
                self.model.tokenizer.pad_token_id = orig_pad

        self.metadata = [{} for _ in prompts]
        return gen_texts

    def _greedy_decode(self, input_tokens: torch.Tensor) -> List[str]:
        act_name = tl_utils.get_act_name("resid_post", self.model.cfg.n_layers - 1)
        _, cache = self.model.run_with_cache(
            input_tokens, names_filter=[act_name], return_type=None
        )
        resid_post = cache[act_name]
        last_resid = resid_post[:, -1:, :]
        ln_out = self.model.ln_final(last_resid) if hasattr(self.model, "ln_final") else last_resid
        logits = self.model.unembed(ln_out)
        predicted_ids = logits[:, -1, :].argmax(dim=-1)
        del cache, resid_post, last_resid, ln_out, logits
        return [self.model.tokenizer.decode(pid.item()).strip() for pid in predicted_ids]

    def _multi_token_generate(
        self, input_tokens: torch.Tensor, max_new_tokens: int
    ) -> List[str]:
        generated = self.model.generate(
            input_tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False
        )
        prompt_len = input_tokens.shape[1]
        return [
            t.strip()
            for t in self.model.tokenizer.batch_decode(
                generated[:, prompt_len:], skip_special_tokens=True
            )
        ]

    def get_token_probs(self, prompt, tokens, coeff=None, **kwargs):
        with torch.no_grad():
            input_ids = self.model.to_tokens(prompt)
            logits = self.model(input_ids, return_type="logits")[0, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            result = {}
            total = 0.0
            for t in tokens:
                tid = self.model.tokenizer.encode(t)[-1]
                result[t] = probs[tid].item()
                total += result[t]
            for t in tokens:
                result[t] /= total + 1e-9
            return result

    def get_output_metadata(self):
        return self.metadata


class CorrSteerExtractor(BaseExtractor):
    """
    CorrSteer: Correlation-based steering vector extraction.

    Attaches hooks on all target layers and runs generation once per batch.
    Generation and scoring delegated to evaluator.batch() via UnsteeredModelWrapper.
    """

    METHOD_NAME = "CORRSTEER"

    def __init__(
        self,
        model,
        sae: Dict[int, Any],
        layer: List[int],
        batch_size: int = 8,
        top_k: int = 10,
        corrsteer_pool: str = "max",
        corrsteer_steer_pool: str = "max",
        corrsteer_pos_only: bool = True,
        corrsteer_neg_only: bool = False,
        corrsteer_real: bool = False,
        corrsteer_decode: bool = False,
        corrsteer_max_new_tokens: int = 1,
        corrsteer_raw: bool = False,
        corrsteer_reverse: bool = False,
        corrsteer_selection: str = "correlation",
        corrsteer_caacoeff: bool = False,
        corrsteer_layer_mode: str = "foreach",
        corrsteer_prompt_suffix: Optional[str] = None,
        corrsteer_reward_evaluator: Optional[str] = None,
        device: Optional[torch.device] = None,
        hook_point: List[str] = ["pre"],
        position: Union[str, int] = "last",
        **kwargs,
    ):
        super().__init__(
            model, layer, batch_size, device, hook_point=hook_point, position=position
        )
        self.sae = sae
        self.top_k = top_k
        self.max_new_tokens = corrsteer_max_new_tokens
        self.pool = corrsteer_pool
        self.steer_pool = corrsteer_steer_pool
        self.pos_only = corrsteer_pos_only
        self.neg_only = corrsteer_neg_only
        self.real = corrsteer_real
        self.use_decode = corrsteer_decode
        self.raw = corrsteer_raw
        self.reverse = corrsteer_reverse
        self.selection = corrsteer_selection
        self.caacoeff = corrsteer_caacoeff
        self.layer_mode = corrsteer_layer_mode
        self.reward_evaluator_name = corrsteer_reward_evaluator
        self.prompt_suffix = corrsteer_prompt_suffix
        if self.reward_evaluator_name:
            if self.reward_evaluator_name not in EVALUATOR_MAP:
                raise ValueError(f"Unknown evaluator: {self.reward_evaluator_name}")
            self.evaluator = EVALUATOR_MAP[self.reward_evaluator_name](device=self.device)
            logger.info(f"Initialized CorrSteer evaluator: {self.reward_evaluator_name}")
        else:
            self.evaluator = EVALUATOR_MAP["multiple_choice"](device=self.device)

        if self.raw:
            self.d_sae = model.cfg.d_model
        else:
            _first_sae = next(iter(sae.values()))
            self.d_sae = _first_sae.cfg.d_sae

        if self.pos_only and self.neg_only:
            raise ValueError("Cannot use both pos_only and neg_only simultaneously")
        valid_selections = {"correlation", "mi", "fisher", "caa"}
        if self.selection not in valid_selections:
            raise ValueError(
                f"Unknown selection method: {self.selection}. Use: {valid_selections}"
            )

        self.accumulators: Dict[int, StreamingCorrelationAccumulator] = {}
        self.selected_features: Dict[int, List[dict]] = {}
        self.top_idx: Dict[int, List[int]] = {}
        self.sparse_latent: Dict[int, torch.Tensor] = {}
        self.vector: Dict[int, torch.Tensor] = {}

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[int]] = None,
        ground_truth: Optional[List[int]] = None,
        prompts: Optional[List[str]] = None,
        choices: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        self.accumulators = {}
        self.selected_features = {}
        self.top_idx = {}
        self.sparse_latent = {}
        self.vector = {}
        for layer in self.layer:
            sae = self.sae[layer]
            device = getattr(sae, "device", self.device) or self.device
            self.accumulators[layer] = self._create_accumulator(device=device)

        hooks = self.get_hook_name()
        layer_hook_map = {layer: hook for layer, hook in zip(self.layer, hooks)}

        gen_prompts = prompts if prompts is not None else target_data
        if self.prompt_suffix is not None:
            gen_prompts = [prompt + self.prompt_suffix for prompt in gen_prompts]
        self._run_collection(gen_prompts, ground_truth, layer_hook_map, self.sae, choices=choices)

        for layer in self.layer:
            self.accumulator = self.accumulators[layer]
            self._build_vector(layer, self.sae[layer])

        if self.layer_mode == "global" and len(self.layer) > 1:
            layers_list = []
            corrs = []

            for layer in self.layer:
                feats = self.selected_features[layer]
                if not feats:
                    continue

                layer_corr = torch.tensor(
                    [abs(f["correlation"]) for f in feats],
                    device=self.vector[layer].device,
                )
                corrs.append(layer_corr)
                layers_list.append(
                    torch.full((len(layer_corr),), layer, device=layer_corr.device)
                )

            if corrs:
                corrs_cat = torch.cat(corrs)
                layers_cat = torch.cat(layers_list)

                k = min(self.top_k, len(corrs_cat))
                topk_idx = _safe_topk_indices(corrs_cat, k=k, by_abs=False)
                selected_layers = torch.unique(layers_cat[topk_idx])

                logger.info(
                    f"CorrSteer [Global mode]: Selected layers {selected_layers.tolist()} "
                    f"with top-{k} |correlation| values"
                )

                for layer in self.layer:
                    if layer not in selected_layers:
                        self.vector[layer] = torch.zeros_like(self.vector[layer])

        self._set_metadata()
        return self.vector

    def _create_accumulator(
        self, device: Optional[torch.device] = None
    ) -> StreamingCorrelationAccumulator:
        mi_edges = None
        if self.selection == "mi":
            edges = []
            for d in range(-6, 1):
                for k in range(1, 4):
                    edges.append(10 ** (d + k / 4))
            edges.append(1.0)
            mi_edges = torch.tensor(edges, dtype=torch.float32)

        device = device or self.device
        return StreamingCorrelationAccumulator(
            self.d_sae,
            device=device,
            dtype=torch.float32,
            real=self.real,
            mi_edges=mi_edges,
            fisher_enabled=(self.selection == "fisher"),
        )

    def _run_collection(
        self,
        target_data: List[str],
        ground_truth: List[int],
        layer_hook_map: Dict[int, str],
        sae_map: Dict[int, Any],
        choices: Optional[List[str]] = None,
    ) -> None:
        for batch_start in tqdm(range(0, len(target_data), self.batch_size)):
            batch_end = min(batch_start + self.batch_size, len(target_data))
            questions = target_data[batch_start:batch_end]
            gt_batch = ground_truth[batch_start:batch_end]
            gen_acts_map, rewards = self._capture_and_score(
                questions, gt_batch, layer_hook_map, sae_map, choices=choices
            )

            if gen_acts_map is None:
                continue

            if self.reverse:
                rewards = [1.0 - r for r in rewards]

            B = next(iter(gen_acts_map.values())).shape[0] if gen_acts_map else 0
            if self.max_new_tokens <= 1:
                eos_positions = torch.ones(
                    B,
                    dtype=torch.int64,
                    device=next(iter(gen_acts_map.values())).device,
                )
            else:
                T = next(iter(gen_acts_map.values())).shape[1]
                eos_positions = torch.full(
                    (B,),
                    T,
                    dtype=torch.int64,
                    device=next(iter(gen_acts_map.values())).device,
                )

            rewards_tensor_cache = {}
            for layer in self.layer:
                gen_acts = gen_acts_map.get(layer, None)
                if gen_acts is None:
                    continue
                pooled_corr = self._get_activations(gen_acts, eos_positions, self.pool)
                pooled_coeff = self._get_activations(gen_acts, eos_positions, self.steer_pool)

                acc = self.accumulators[layer]
                if acc not in rewards_tensor_cache:
                    rewards_tensor_cache[acc] = torch.tensor(
                        rewards, dtype=acc.dtype, device=acc.device
                    )

                rewards_tensor = rewards_tensor_cache[acc]
                pooled_corr = pooled_corr.to(device=acc.device, dtype=acc.dtype, copy=False)
                pooled_coeff = pooled_coeff.to(device=acc.device, dtype=acc.dtype, copy=False)

                acc.update_corr(pooled_corr, rewards_tensor)
                acc.update_coeff(pooled_coeff, rewards_tensor)
                if self.selection == "mi":
                    acc.update_mi(pooled_corr, rewards_tensor)

            del gen_acts_map, rewards, rewards_tensor_cache

    def _capture_and_score(
        self,
        questions: List[str],
        gt_batch: List,
        layer_hook_map: Dict[int, str],
        sae_map: Dict[int, Any],
        choices: Optional[List[str]] = None,
    ) -> Tuple[Optional[Dict[int, torch.Tensor]], List[float]]:
        """
        Attach hooks for all layers, run evaluator.batch() and scoring in one pass.
        """
        activation_buffers: Dict[int, List[torch.Tensor]] = {
            layer: [] for layer in layer_hook_map
        }

        def make_hook(layer: int, sae):
            def hook(residual, hook):
                if self.position == "last":
                    tokens_to_capture = residual[:, -1:, :]
                elif self.position == "mean":
                    tokens_to_capture = residual.mean(dim=1, keepdim=True)
                else:
                    tokens_to_capture = residual[:, self.position : self.position + 1, :]
                if self.raw:
                    activations = tokens_to_capture.detach()
                else:
                    tokens_for_sae = tokens_to_capture.to(device=sae.device, dtype=sae.dtype)
                    B, S, H = tokens_for_sae.shape
                    encoded = sae.encode(tokens_for_sae.view(-1, H))
                    activations = encoded.view(B, S, -1).detach()
                activation_buffers[layer].append(activations)
                return residual

            return hook

        for layer, hook_name in layer_hook_map.items():
            sae = sae_map[layer]
            self.model.add_hook(hook_name, make_hook(layer, sae))

        try:
            samples = [
                {"question": q, "answer": str(gt), "choices": choices}
                for q, gt in zip(questions, gt_batch)
            ]
            wrapper = UnsteeredModelWrapper(self.model, max_new_tokens=self.max_new_tokens)
            evaluator = self.evaluator
            results = evaluator.batch(wrapper, samples, coeff=None, max_new_tokens=self.max_new_tokens)
            rewards = [float(r[0]) for r in results]
        finally:
            self.model.reset_hooks()

        expected_B = len(questions)
        stacked_map: Dict[int, Optional[torch.Tensor]] = {}
        for layer, buf in activation_buffers.items():
            if not buf:
                stacked_map[layer] = None
            elif len(buf) == expected_B and buf[0].shape[0] == 1 and expected_B > 1:
                stacked_map[layer] = torch.cat(buf, dim=0)
            else:
                stacked_map[layer] = torch.cat(buf, dim=1)

        if not any(v is not None for v in stacked_map.values()):
            return None, rewards

        return stacked_map, rewards

    def _get_activations(
        self,
        activations: torch.Tensor,
        eos_positions: Optional[torch.Tensor] = None,
        pool_type: str = "max",
    ) -> torch.Tensor:
        """Pool [B, T, D] activations to [B, D]."""
        B, T, D = activations.shape
        device = activations.device
        if eos_positions is None:
            return (
                activations.max(dim=1).values
                if pool_type == "max"
                else activations.mean(dim=1)
            )

        eos = eos_positions.to(device=device)
        idx = torch.arange(T, device=device).unsqueeze(0)
        valid = idx < eos.unsqueeze(1)

        if pool_type == "max":
            neg_inf = torch.finfo(activations.dtype).min
            masked = torch.where(valid.unsqueeze(2), activations, neg_inf)
            pooled = masked.max(dim=1).values
            no_valid = valid.sum(dim=1) == 0
            if no_valid.any():
                pooled[no_valid] = torch.zeros(D, dtype=pooled.dtype, device=device)
        else:
            masked = activations * valid.unsqueeze(2).to(dtype=activations.dtype)
            denom = valid.sum(dim=1).clamp(min=1).unsqueeze(1).to(dtype=activations.dtype)
            pooled = masked.sum(dim=1) / denom

        return pooled

    def _build_vector(self, layer: int, sae) -> None:
        self.accumulator = self.accumulators[layer]
        selected_features = self.accumulator.top_features_signed(
            k=self.top_k,
            pos_only=self.pos_only,
            neg_only=self.neg_only,
            selection=self.selection,
            caacoeff=self.caacoeff,
            scale=1.0,
            real=self.real,
        )

        if not selected_features:
            logger.warning(f"[Layer {layer}] CorrSteer: No features found with current settings")
            selected_features = self.accumulator.top_features_signed(
                k=self.top_k,
                pos_only=False,
                neg_only=False,
                selection="correlation",
                scale=1.0,
            )

        dtype = sae.W_dec.dtype if not self.raw else torch.bfloat16
        sparse_latent = torch.zeros(self.d_sae, device=self.device, dtype=dtype)
        for feat in selected_features:
            sparse_latent[feat["feature_index"]] = feat["coefficient"]
        self.sparse_latent[layer] = sparse_latent.to_sparse().coalesce().cpu()
        self.selected_features[layer] = selected_features
        self.top_idx[layer] = [
            int(feat["feature_index"])
            for feat in selected_features
            if "feature_index" in feat
        ]

        if self.raw:
            self.vector[layer] = sparse_latent
        else:
            latent_for_decode = sparse_latent.to(device=sae.device, dtype=sae.dtype)
            if self.use_decode:
                self.vector[layer] = sae.decode(latent_for_decode).to(self.device)
            else:
                self.vector[layer] = (latent_for_decode @ sae.W_dec).to(self.device)

        logger.info(
            f"[Layer {layer}] CorrSteer: {len(selected_features)} features, "
            f"selection={self.selection}, vector norm={self.vector[layer].norm():.4f}"
        )

    @staticmethod
    def _build_feature_tracker(
        accumulator: "StreamingCorrelationAccumulator",
        selected_features: List[Dict[str, Any]],
        top_n: int = 15,
    ) -> List[Dict[str, Any]]:
        """Build CorrSteer top-feature tracker from streaming accumulator stats."""
        if not selected_features:
            return []

        total_count = max(int(accumulator.n), 1)
        rows: List[Dict[str, Any]] = []

        for rank, feature in enumerate(selected_features[:top_n], start=1):
            feature_index = int(feature.get("feature_index", -1))
            if feature_index < 0 or feature_index >= accumulator.dict_size:
                continue

            mean_value = float((accumulator.sum_x[feature_index] / total_count).item())
            mean_square = float((accumulator.sum_xx[feature_index] / total_count).item())
            variation = max(mean_square - mean_value * mean_value, 0.0) ** 0.5

            active_count = int(
                (
                    accumulator.count_success_active[feature_index]
                    + accumulator.count_failure_active[feature_index]
                ).item()
            )
            activation_frequency = float(active_count / total_count)

            rows.append(
                {
                    "rank": rank,
                    "feature_index": feature_index,
                    "activation_frequency": activation_frequency,
                    "feature_mean": mean_value,
                    "feature_value_variation": float(variation),
                }
            )

        return rows

    def _set_metadata(self) -> None:
        feature_tracker = {
            layer: self._build_feature_tracker(
                self.accumulators[layer],
                self.selected_features.get(layer, []),
                top_n=self.top_k,
            )
            for layer in self.layer
        }

        self.metadata = _build_method_metadata(
            "CORRSTEER",
            top_idx=self.top_idx,
            sparse_latent=self.sparse_latent,
            selected_features=self.selected_features,
            layer_mode=self.layer_mode,
            pool=self.pool,
            steer_pool=self.steer_pool,
            pos_only=self.pos_only,
            neg_only=self.neg_only,
            decode=self.use_decode,
            raw=self.raw,
            reverse=self.reverse,
            selection=self.selection,
            caacoeff=self.caacoeff,
            max_new_tokens=self.max_new_tokens,
            feature_tracker=feature_tracker,
        )


class SAEFreeExtractor(BaseExtractor):
    """
    SAE-Free Steering via Eigendecomposition of Activation Differences.

    Reference: SAE-free/eigenvec.ipynb

    Algorithm:
    1. For each (target, contrast) pair: compute mean activation difference
    2. Stack all differences into matrix A
    3. Compute covariance: AA^T
    4. Eigenvalue decomposition: eigenvalues, eigenvectors = eigh(AA^T)
    5. Select top eigenvector (largest eigenvalue = last column)
    """

    METHOD_NAME = "SAE-FREE"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        saefree_component_idx: int = -1,
        saefree_center_diffs: bool = False,
        saefree_cov_normalize: bool = False,
        saefree_align_sign: bool = True,
        position: Union[int, str] = "mask",
        device: Optional[torch.device] = None,
        hook_point: List[str] = ["pre"],
    ):
        super().__init__(
            model,
            layer,
            batch_size,
            device,
            hook_point=hook_point,
            position=position,
        )
        self.component_idx = saefree_component_idx
        self.center_diffs = saefree_center_diffs
        self.cov_normalize = saefree_cov_normalize
        self.align_sign = saefree_align_sign

        self.eigenvectors: Dict[int, np.ndarray] = {}
        self.eigenvalues: Dict[int, np.ndarray] = {}

    def _get_activations(self, inputs: List[str]) -> Dict[int, torch.Tensor]:
        """
        Get mean-pooled layer activations for multiple texts.

        Returns:
            Dict mapping layer to [n_samples, d_model] tensor of mean activations.
        """
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
            pretokenize_all=False,
            change_pad_token=False,
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: List[str],
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Extract steering vector via eigendecomposition for each layer."""
        n_pairs = min(len(target_data), len(contrast_data))
        target_data = target_data[:n_pairs]
        contrast_data = contrast_data[:n_pairs]

        logger.info(f"SAE-Free: Extracting activations for {n_pairs} target samples...")
        target_acts_dict = self._get_activations(target_data)

        logger.info(f"SAE-Free: Extracting activations for {n_pairs} contrast samples...")
        contrast_acts_dict = self._get_activations(contrast_data)

        self.vector = {}
        for layer in self.layer:
            target_acts = target_acts_dict[layer]
            contrast_acts = contrast_acts_dict[layer]

            diffs = (target_acts - contrast_acts).float()
            if self.center_diffs:
                diffs = diffs - diffs.mean(dim=0, keepdim=True)

            A = diffs.T.cpu().numpy()
            AAT = A @ A.T
            if self.cov_normalize and diffs.shape[0] > 0:
                AAT = AAT / float(diffs.shape[0])

            logger.info(f"[Layer {layer}] SAE-Free: Computing eigendecomposition...")
            evals, evecs = np.linalg.eigh(AAT)
            self.eigenvalues[layer] = evals
            self.eigenvectors[layer] = evecs

            comp_idx = self.component_idx
            if comp_idx < 0:
                comp_idx = len(evals) + comp_idx
            if comp_idx < 0 or comp_idx >= len(evals):
                raise IndexError(
                    f"Invalid saefree_component_idx={self.component_idx} for d_model={len(evals)}"
                )

            selected_eigenvec = evecs[:, comp_idx]
            vec = torch.from_numpy(selected_eigenvec).float().to(self.device)

            # Eigenvectors are sign-ambiguous. Align to mean diff direction so
            # runs are deterministic and compatible with steering expectations.
            if self.align_sign:
                mean_diff = diffs.mean(dim=0).to(self.device)
                if torch.dot(vec, mean_diff) < 0:
                    vec = -vec

            self.vector[layer] = vec / (vec.norm() + 1e-8)

            logger.info(
                f"[Layer {layer}] SAE-Free: eigenvalue={evals[comp_idx]:.4f}, "
                f"norm={self.vector[layer].norm():.4f}"
            )

        self.metadata = _build_method_metadata(
            "SAE-FREE",
            top_idx={layer: [] for layer in self.layer},
            sparse_latent={layer: None for layer in self.layer},
            component_idx=self.component_idx,
            center_diffs=self.center_diffs,
            cov_normalize=self.cov_normalize,
            align_sign=self.align_sign,
        )

        return self.vector


class SAECoTExtractor(BaseExtractor):
    """
    SAE-BASE extractor matching GT SAE-free/sae_gsm.py controls.

    Selects top-k SAE features from data, stores sparse latent values in metadata,
    and exports dense decoded steering vectors for runtime injection.
    """

    METHOD_NAME = "SAE-COT"

    def __init__(
        self,
        model,
        sae: Dict[int, Any],
        layer: List[int],
        top_k: int = 15,
        saecot_value_mode: str = "target_mean",
        saecot_max_act: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, layer, **kwargs)
        self.sae = sae
        self.top_k = top_k
        self.value_mode = str(saecot_value_mode)
        self.max_act = float(saecot_max_act)

        """Create sparse latent steering vectors using top-k SAE feature selection."""
        self.vector = {}
        self.sparse_latent = {}
        self.top_idx = {}
        self.score_stats = {}

    def _get_activations(self, inputs: List[str], layers: List[int], **kwargs) -> Dict[int, torch.Tensor]:
        """Collect pooled SAE latent activations for optional auto-selection."""
        all_layer_acts = collect_sae_activations(
            self.model,
            self.sae,
            inputs,
            layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            tokenizer=self.model.tokenizer,
        ).activations
        return all_layer_acts

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:


        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        if self.value_mode not in {"target_mean", "selected"}:
            raise ValueError(
                f"Unsupported saecot_value_mode={self.value_mode!r}. "
                "Use target_mean, diff_mean, or selected."
            )

        target_acts = self._get_activations(target_data, self.layer)
        contrast_acts = (
            self._get_activations(contrast_data, self.layer)
            if contrast_data is not None
            else None
        )

        for layer in self.layer:
            sae = self.sae[layer]
            d_sae = int(getattr(getattr(sae, "cfg", None), "d_sae", sae.W_dec.shape[0]))

            target = target_acts[layer].to(self.device)
            target_mean = target.mean(dim=0).float()

            if contrast_acts is not None:
                contrast = contrast_acts[layer].to(self.device)
                n = min(target.shape[0], contrast.shape[0])
                diff_mean = (target[:n].mean(dim=0) - contrast[:n].mean(dim=0)).float()
            else:
                diff_mean = target_mean.clone()

            top_idx = _safe_topk_indices(
                diff_mean,
                k=min(self.top_k, diff_mean.numel()),
                by_abs=True,
            )

            vec = torch.zeros(d_sae, dtype=torch.float32, device=self.device)
            if self.value_mode == "selected":
                vec[top_idx] = self.max_act
            else:
                vec[top_idx] = target_mean[top_idx]

            self.sparse_latent[layer] = vec.to_sparse().coalesce().cpu()
            self.vector[layer] = sae.decode(
                vec.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)
            ).to(self.device)
            self.top_idx[layer] = top_idx.detach().cpu().tolist()
            self.score_stats[layer] = {
                "score_mean": float(diff_mean.mean().item()),
                "score_abs_mean": float(diff_mean.abs().mean().item()),
                "score_abs_max": float(diff_mean.abs().max().item()),
            }

        self.metadata = _build_method_metadata(
            "SAE-COT",
            top_idx=self.top_idx,
            sparse_latent=self.sparse_latent,
            value_mode=self.value_mode,
            max_act=self.max_act,
            score_stats=self.score_stats,
            n_target=len(target_data) if target_data is not None else 0,
            n_contrast=len(contrast_data) if contrast_data is not None else 0,
        )

        return self.vector


class FGAAExtractor(BaseExtractor):
    """
    Feature Guided Activation Additions (FGAA) extractor.

    Paper defaults implemented:
    - Density threshold theta = 0.01
    - BOS-feature removal enabled
    - Top-k feature selection (n1 positive, n2 negative)
    - L1-normalized target latent before steering-vector construction

    This implementation supports two output modes:
    1) Effect approximator mode (if `fgaa_effect_matrix_path` is provided)
    2) SAE decoder fallback (no external approximator required)
    """

    METHOD_NAME = "FGAA"

    def __init__(
        self,
        model,
        sae: Dict[int, Any],
        layer: List[int],
        batch_size: int = 8,
        position: str = "mean",
        top_k: int = 15,
        fgaa_density_threshold: float = 0.01,
        fgaa_n1: int = 8,
        fgaa_n2: int = 0,
        fgaa_remove_bos: bool = True,
        fgaa_bos_feature_ids: Optional[List[int]] = None,
        fgaa_effect_matrix_path: Optional[str] = None,
        fgaa_l1_normalize_target: bool = True,
        **kwargs,
    ):
        # Use paper default token aggregation when caller keeps the global default.
        if position == "last":
            position = "mean"
        super().__init__(model, layer, batch_size=batch_size, position=position, **kwargs)
        self.sae = sae
        self.top_k = top_k
        self.density_threshold = float(fgaa_density_threshold)
        self.n1 = int(fgaa_n1)
        self.n2 = int(fgaa_n2)
        self.remove_bos = bool(fgaa_remove_bos)
        self.bos_feature_ids = list(fgaa_bos_feature_ids or [3220, 11752, 12160, 11498])
        self.effect_matrix_path = fgaa_effect_matrix_path
        self.l1_normalize_target = bool(fgaa_l1_normalize_target)

        if self.n1 < 0 or self.n2 < 0:
            raise ValueError(f"FGAA requires non-negative n1/n2, got n1={self.n1}, n2={self.n2}")

        self.sparse_latent: Dict[int, torch.Tensor] = {}
        self.top_idx: Dict[int, List[int]] = {}
        self.top_idx_neg: Dict[int, List[int]] = {}

    def _get_activations(
        self,
        inputs: List[str],
        layers: List[int],
        **kwargs,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, Any]]]:
        """Collect pooled SAE activations and streaming stats in one pass."""
        collected = collect_sae_activations(
            self.model,
            self.sae,
            inputs,
            layers,
            hook_point=self.hook_point[0],
            batch_size=self.batch_size,
            pooling=self.position,
            tokenizer=self.model.tokenizer,
            active_threshold=0.0,
        )
        return collected.activations, collected.stats

    def _load_effect_approximator(
        self,
        layer: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load optional linear effect approximator weights from a torch checkpoint."""
        if not self.effect_matrix_path:
            return None, None

        path = self.effect_matrix_path.format(layer=layer)
        if not os.path.exists(path):
            logger.warning(f"FGAA: effect approximator path not found: {path}. Falling back to SAE decode.")
            return None, None

        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            logger.warning("FGAA: effect approximator checkpoint is not a dict. Falling back to SAE decode.")
            return None, None

        if "layers" in payload and isinstance(payload["layers"], dict):
            layer_payload = payload["layers"].get(layer)
            if isinstance(layer_payload, dict):
                return layer_payload.get("W"), layer_payload.get("b")

        return payload.get("W"), payload.get("b")

    @staticmethod
    def _normalize_safe(vec: torch.Tensor) -> torch.Tensor:
        norm = vec.norm(p=2).clamp(min=1e-8)
        return vec / norm

    def _build_dense_vector(
        self,
        layer: int,
        vtarget: torch.Tensor,
    ) -> torch.Tensor:
        """Build dense steering vector from target latent via approximator or SAE decode."""
        W, b = self._load_effect_approximator(layer)
        if W is not None:
            try:
                W = W.to(device=self.device, dtype=torch.float32)
                v = vtarget.to(device=self.device, dtype=torch.float32)
                if W.ndim == 2 and W.shape[1] == v.shape[0]:
                    part_target = W @ v

                    if b is None:
                        part_bias = torch.zeros_like(part_target)
                    else:
                        b = b.to(device=self.device, dtype=torch.float32)
                        if b.ndim == 1 and b.shape[0] == v.shape[0]:
                            part_bias = W @ b
                        elif b.ndim == 1 and b.shape[0] == part_target.shape[0]:
                            part_bias = b
                        else:
                            part_bias = torch.zeros_like(part_target)

                    return self._normalize_safe(part_target) - self._normalize_safe(part_bias)
            except Exception as exc:
                logger.warning(f"FGAA: effect-approximator solve failed ({exc}). Falling back to SAE decode.")

        sae = self.sae[layer]
        latent = vtarget.to(device=sae.W_dec.device, dtype=sae.W_dec.dtype)
        return sae.decode(latent).to(self.device)

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("FGAA requires contrast_data (desired vs undesired examples).")
        if not target_data or not contrast_data:
            raise ValueError("FGAA requires non-empty target_data and contrast_data.")

        target_acts, target_stats = self._get_activations(target_data, self.layer)
        contrast_acts, contrast_stats = self._get_activations(contrast_data, self.layer)

        self.vector = {}
        feature_tracker: Dict[int, List[Dict[str, Any]]] = {}
        filter_stats: Dict[int, Dict[str, Any]] = {}

        for layer in self.layer:
            target_mean = target_acts[layer].to(self.device).mean(dim=0).float()
            contrast_mean = contrast_acts[layer].to(self.device).mean(dim=0).float()
            vdiff = target_mean - contrast_mean

            # Step 1: density filtering (paper Eq. 2)
            t_num = max(int(target_stats[layer]["num_samples"]), 1)
            c_num = max(int(contrast_stats[layer]["num_samples"]), 1)
            total_num = float(t_num + c_num)
            density = (
                target_stats[layer]["active_count"].to(self.device).float()
                + contrast_stats[layer]["active_count"].to(self.device).float()
            ) / total_num

            vfiltered = vdiff.clone()
            density_mask = density > self.density_threshold
            vfiltered[density_mask] = 0.0

            # Step 2: BOS feature removal (paper Eq. 3)
            if self.remove_bos and self.bos_feature_ids:
                bos_idx = torch.as_tensor(self.bos_feature_ids, dtype=torch.long, device=self.device)
                bos_idx = bos_idx[(bos_idx >= 0) & (bos_idx < vfiltered.numel())]
                if bos_idx.numel() > 0:
                    vfiltered[bos_idx] = 0.0

            # Step 3: top-k selection (paper Eq. 4)
            pos_candidates = torch.nonzero(vfiltered > 0, as_tuple=True)[0]
            neg_candidates = torch.nonzero(vfiltered < 0, as_tuple=True)[0]

            pos_idx = torch.empty(0, dtype=torch.long, device=self.device)
            neg_idx = torch.empty(0, dtype=torch.long, device=self.device)

            if pos_candidates.numel() > 0 and self.n1 > 0:
                pos_scores = vfiltered[pos_candidates]
                keep = min(self.n1, int(pos_scores.numel()))
                ranked = torch.argsort(pos_scores, descending=True)[:keep]
                pos_idx = pos_candidates[ranked]

            if neg_candidates.numel() > 0 and self.n2 > 0:
                neg_scores = -vfiltered[neg_candidates]
                keep = min(self.n2, int(neg_scores.numel()))
                ranked = torch.argsort(neg_scores, descending=True)[:keep]
                neg_idx = neg_candidates[ranked]

            vtarget = torch.zeros_like(vfiltered)
            if pos_idx.numel() > 0:
                vtarget[pos_idx] = vfiltered[pos_idx]
            if neg_idx.numel() > 0:
                vtarget[neg_idx] = vfiltered[neg_idx]

            # Keep extractor robust when filtering drops all features.
            if torch.count_nonzero(vtarget) == 0:
                fallback_idx = _safe_topk_indices(vdiff, k=1, by_abs=True)
                if fallback_idx.numel() > 0:
                    vtarget[fallback_idx] = vdiff[fallback_idx]

            if self.l1_normalize_target:
                l1 = vtarget.abs().sum().clamp(min=1e-8)
                vtarget = vtarget / l1

            dense_vector = self._build_dense_vector(layer, vtarget)
            self.vector[layer] = dense_vector
            self.sparse_latent[layer] = vtarget.to_sparse().coalesce().cpu()
            self.top_idx[layer] = pos_idx.detach().cpu().tolist()
            self.top_idx_neg[layer] = neg_idx.detach().cpu().tolist()

            feature_tracker[layer] = build_top_feature_tracker_from_stats(
                self.top_idx[layer],
                target_stats=target_stats[layer],
                contrast_stats=contrast_stats[layer],
                top_n=self.top_k,
            )
            filter_stats[layer] = {
                "density_filtered_count": int(density_mask.sum().item()),
                "selected_positive": len(self.top_idx[layer]),
                "selected_negative": len(self.top_idx_neg[layer]),
                "vector_norm": float(dense_vector.norm().item()),
            }

        self.metadata = _build_method_metadata(
            "FGAA",
            top_idx=self.top_idx,
            sparse_latent=self.sparse_latent,
            top_idx_neg=self.top_idx_neg,
            feature_tracker=feature_tracker,
            density_threshold=self.density_threshold,
            n1=self.n1,
            n2=self.n2,
            remove_bos=self.remove_bos,
            bos_feature_ids=self.bos_feature_ids,
            l1_normalize_target=self.l1_normalize_target,
            filter_stats=filter_stats,
        )

        return self.vector