"""
Dense extractors for steering vector computation.

These extractors work directly on dense model activations (residual stream).
"""

from typing import List, Optional, Dict, Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from ..base import BaseExtractor
from ..data import DataLoader, TRAIN_DATASET_REGISTRY
from ..logger import setup_logger
from ..utils import collect_dense_activations

from tqdm import tqdm

logger = setup_logger(__name__)

class CAAExtractor(BaseExtractor):
    """
    Contrastive Activation Addition (CAA) Extractor.

    Computes steering vector as the difference between mean activations
    of target and contrast prompts in the residual stream.

    Paper: "Steering Language Models with Activation Engineering"
    """
    
    METHOD_NAME = "CAA"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",  # "last" or "mean"
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point)
        self.position = position

    def _get_activations(self, inputs: List[str]) -> Dict[int, torch.Tensor]:
        """Collect mean residual activations from inputs for multiple layers."""
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

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Extract dense steering vectors via mean difference for all layers."""
        target_means = self._get_activations(target_data)

        if contrast_data is not None:
            contrast_means = self._get_activations(contrast_data)
        else:
            contrast_means = {layer: torch.zeros_like(target_means[layer]) for layer in self.layer}

        self.vector = {layer: target_means[layer] - contrast_means[layer] for layer in self.layer}

        self.metadata = {
            "method": "CAA",
            "n_target": len(target_data),
            "n_contrast": len(contrast_data) if contrast_data else 0,
        }

        return self.vector


class ManifoldPaperExtractor(BaseExtractor):
    """
    Manifold Steering extractor (paper-specific variant).

    Implements a PCA-based manifold learner but preserves the learned
    basis and mean in metadata so downstream steer models may use the
    explicit manifold representation (U_k, mean).
    """

    METHOD_NAME = "MANIFOLD_PAPER"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        manifold_dim: int = 10,
        whiten: bool = False,
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.manifold_dim = int(manifold_dim)
        self.whiten = bool(whiten)

    def _get_activations(self, inputs: List[str]) -> Dict[int, torch.Tensor]:
        change_pad_token = self.change_pad_token
        if isinstance(change_pad_token, tuple):
            change_pad_token = change_pad_token[0]

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
            change_pad_token=bool(change_pad_token),
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: List[str],
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("MANIFOLD_PAPER requires contrast_data")
        if self.manifold_dim <= 0:
            raise ValueError(f"manifold_dim must be positive, got {self.manifold_dim}.")

        target_acts_dict = self._get_activations(target_data)
        contrast_acts_dict = self._get_activations(contrast_data)

        vectors = {}
        basis_store: Dict[int, torch.Tensor] = {}
        mean_store: Dict[int, torch.Tensor] = {}
        explained_var: Dict[int, float] = {}

        for layer in self.layer:
            tgt = target_acts_dict[layer].to(torch.float32)
            ctr = contrast_acts_dict[layer].to(torch.float32)

            # raw direction (difference of means)
            mean_t = tgt.mean(dim=0)
            mean_c = ctr.mean(dim=0)
            raw = mean_t - mean_c

            # Build reasoning dataset and center it
            reasoning = torch.cat([tgt, ctr], dim=0)
            global_mean = reasoning.mean(dim=0, keepdim=True)
            centered = (reasoning - global_mean).cpu().numpy()

            n_components = min(self.manifold_dim, centered.shape[0], centered.shape[1])
            if n_components < 1:
                raise ValueError(
                    f"Cannot run PCA for MANIFOLD_PAPER at layer {layer}: n_samples={centered.shape[0]}, d_model={centered.shape[1]}"
                )

            pca = PCA(n_components=n_components, whiten=self.whiten)
            pca.fit(centered)

            basis = torch.from_numpy(pca.components_.T).to(self.device, dtype=torch.float32)
            # Project raw (centered by global mean) onto manifold basis
            raw_centered = (raw.to(self.device) - global_mean.squeeze(0).to(self.device))
            manifold_dir = basis @ (basis.T @ raw_centered)
            manifold_dir = manifold_dir / (manifold_dir.norm() + 1e-8)

            vectors[layer] = manifold_dir
            basis_store[layer] = basis.cpu()
            mean_store[layer] = global_mean.squeeze(0).cpu()
            explained_var[layer] = float(pca.explained_variance_ratio_.sum())

        self.vector = vectors
        self.metadata = {
            "method": "MANIFOLD_PAPER",
            "manifold_dim": self.manifold_dim,
            "manifold_basis": basis_store,
            "manifold_mean": mean_store,
            "explained_variance_ratio": explained_var,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data),
        }

        return self.vector


class CASTExtractor(BaseExtractor):
    """
    Conditional Activation Steering (CAST) Extractor.
    
    Extracts BOTH:
    1. Steering vector (at steering_layer from behavior_data)
    2. Conditional vector (at conditional_layer from conditional_data)
    
    Paper: "CAST: Conditional Activation Steering"
    
    The conditional vector is used to trigger steering only when the input
    matches a certain condition (e.g., harmful content detection).
    """
    
    METHOD_NAME = "CAST"
    
    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        use_pca: bool = True,
        conditional_layer: Optional[int] = None,
        conditional_dataset: Optional[str] = None,  # Will load if provided
        save_conditional_vector: Optional[str] = None,
        apply_conditional_chat_template: bool = True,
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
        change_pad_token: bool = False,
    ):
        """
        Args:
            model: Language model
            layer: Steering layer (later layer, e.g., 14-18)
            batch_size: Batch size for processing
            position: Token position ("last" per paper)
            use_pca: Whether to use PCA for extraction (True per paper)
            conditional_layer: Layer for conditional vector (earlier layer, e.g., 5-7)
            conditional_dataset: Dataset name for conditional data extraction
        """
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position = position, change_pad_token = change_pad_token)
        self.use_pca = use_pca
        self.conditional_layer = conditional_layer
        self.conditional_dataset = conditional_dataset
        self.apply_conditional_chat_template = apply_conditional_chat_template
        self.save_conditional_vector = save_conditional_vector
        
        # Will be set during extraction
        self.conditional_vector = None
    
    def _get_activations(
        self,
        inputs: List[str],
        return_all: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """Collect residual activations from inputs for multiple layers.
        
        Follows the same multi-layer hook pattern as CAAExtractor.
        
        Args:
            inputs: List of input strings
            return_all: If True, return all per-sample activations [n_inputs, d_model] per layer.
                        If False, return mean activation [d_model] per layer.
            
        Returns:
            Dict[int, Tensor] keyed by layer
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
            reduce="none" if return_all else "mean",
            return_key_format="layer",
            pretokenize_all=False,
            change_pad_token=bool(self.change_pad_token),
        )
    
    def _extract_pca_vector(self, target_acts: torch.Tensor, contrast_acts: torch.Tensor) -> torch.Tensor:
        """Extract PCA-based steering vector using pairwise centering (like CAST library).
        
        IMPORTANT: CAST library uses INTERLEAVED data [pos1, neg1, pos2, neg2, ...]
        and accesses positives via h[::2] and negatives via h[1::2].
        
        Args:
            target_acts: All target activations (shape: n_examples, d_model)
            contrast_acts: All contrast activations (shape: n_examples, d_model)
            
        Returns:
            PCA component vector
        """
        # Match CAST library layout: interleave [pos1, neg1, pos2, neg2, ...]
        n_pairs = min(len(target_acts), len(contrast_acts))
        target_acts = target_acts[:n_pairs]
        contrast_acts = contrast_acts[:n_pairs]
        
        # Interleave like CAST library: [pos1, neg1, pos2, neg2, ...]
        h = torch.zeros((n_pairs * 2, target_acts.shape[1]), dtype=target_acts.dtype, device=target_acts.device)
        h[::2] = target_acts    # positions 0, 2, 4, ... = positives
        h[1::2] = contrast_acts  # positions 1, 3, 5, ... = negatives
        
        # CAST pca_pairwise method: pairwise centering
        # center = (h[::2] + h[1::2]) / 2
        centers = (h[::2] + h[1::2]) / 2
        
        # Create training data with pairwise centering
        train = h.clone()
        train[::2] -= centers   # Subtract center from positives
        train[1::2] -= centers  # Subtract center from negatives
        
        # PCA on interleaved centered activations (exactly like CAST library)
        pca_model = PCA(n_components=1, whiten=False).fit(train.cpu().numpy())
        pca_vector = pca_model.components_.astype(np.float32).squeeze(axis=0)
        
        # CRITICAL: Sign flip to match CAST library behavior
        # Project interleaved activations onto PCA vector
        h_np = h.cpu().numpy()
        projected = h_np @ pca_vector
        positive_smaller = np.mean([projected[i] < projected[i+1] for i in range(0, n_pairs * 2, 2)])
        positive_larger = np.mean([projected[i] > projected[i+1] for i in range(0, n_pairs * 2, 2)])
        if positive_smaller > positive_larger:
            pca_vector *= -1
        
        return torch.tensor(pca_vector, device=self.device, dtype=torch.float32)
    
    def _extract_vector(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]],
        position: Optional[str] = None,
    ) -> Dict[int, torch.Tensor]:
        """Extract vectors from target and contrast data at given layers.
        
        Follows the same dict-based multi-layer pattern as CAAExtractor.
        
        Args:
            target_data: Target prompts
            contrast_data: Contrast prompts  
            layers: Layers to extract from
            position: Override token position for this call (e.g. "mean" for conditional vector)
            
        Returns:
            Dict[int, Tensor] keyed by layer
        """
        original_position = self.position
        if position is not None:
            self.position = position

        if contrast_data is not None and self.use_pca:
            # CAST PCA path: interleave [pos1, neg1, pos2, neg2, ...]
            n_pairs = min(len(target_data), len(contrast_data))
            interleaved = []
            for i in range(n_pairs):
                interleaved.append(target_data[i])
                interleaved.append(contrast_data[i])

            # One forward pass for all layers
            all_acts = self._get_activations(interleaved, return_all=True)

            vectors = {}
            for layer in self.layer:
                target_acts = all_acts[layer][::2]
                contrast_acts = all_acts[layer][1::2]
                vectors[layer] = self._extract_pca_vector(target_acts, contrast_acts)
        else:
            # Mean-difference path (same as CAA)
            target_means = self._get_activations(target_data)
            if contrast_data is not None:
                contrast_means = self._get_activations(contrast_data)
                vectors = {layer: target_means[layer] - contrast_means[layer] for layer in self.layer}
            else:
                vectors = target_means

        self.position = original_position
        return vectors
    
    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        conditional_target: Optional[List[str]] = None,
        conditional_contrast: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract steering vectors and optionally conditional vector.
        
        Args:
            target_data: Target prompts for steering vectors
            contrast_data: Contrast prompts for steering vectors
            conditional_target: Target prompts for conditional vector (optional)
            conditional_contrast: Contrast prompts for conditional vector (optional)
            
        Returns:
            Dict of {layer: steering_vector} (conditional vector stored in self.conditional_vector)
        """

        # --- Extract steering vectors at all layers ---
        self.vector = self._extract_vector(
            target_data, contrast_data
        )
        
        # --- Extract conditional vector at earlier layer ---
        if conditional_target is not None and conditional_contrast is not None:
            self.conditional_vector = self._extract_vector(
                conditional_target, conditional_contrast, [self.conditional_layer], "mean"
            )[self.conditional_layer]
        elif self.conditional_dataset is not None:
            logger.info(f"CAST: Loading conditional dataset '{self.conditional_dataset}' for vector extraction...")
            cfg = TRAIN_DATASET_REGISTRY[self.conditional_dataset]
            data = DataLoader().load(self.conditional_dataset, apply_chat_template=self.apply_conditional_chat_template, tokenizer = self.model.tokenizer)
            cond_target = [d[cfg.target_key] for d in data]
            cond_contrast = [d[cfg.contrast_key] for d in data]
            self.conditional_vector = self._extract_vector(
                cond_target, cond_contrast, [self.conditional_layer], "mean"
            )[self.conditional_layer]
            if self.save_conditional_vector is not None:
                torch.save(self.conditional_vector.cpu(), self.save_conditional_vector)
                logger.info(f"Saved conditional vector to {self.save_conditional_vector}")
        else:
            self.conditional_vector = None
        
        self.metadata = {
            "method": "CAST",
            "conditional_layer": self.conditional_layer,
            "conditional_vector": self.conditional_vector,
            "has_conditional_vector": self.conditional_vector is not None,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data) if contrast_data else 0,
        }
        
        return self.vector

class ManifoldExtractor(BaseExtractor):
    """
    Manifold-based Steering Vector Extractor.

    Learns a low-dimensional manifold and extracts steering vectors
    that navigate within this manifold.

    Paper: "Manifold Steering for Language Models"
    """
    
    METHOD_NAME = "MANIFOLD"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        manifold_dim: int = 10,
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.manifold_dim = manifold_dim

    def _get_activations(self, inputs: List[str]) -> Dict[int, torch.Tensor]:
        """Collect dense activations for each configured layer."""
        change_pad_token = self.change_pad_token
        if isinstance(change_pad_token, tuple):
            change_pad_token = change_pad_token[0]

        activations = collect_dense_activations(
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
            change_pad_token=bool(change_pad_token),
        )
        for layer, acts in activations.items():
            if acts.ndim != 2:
                raise ValueError(
                    f"MANIFOLD expects 2D activations [n_samples, d_model], got {acts.shape} at layer {layer}. "
                    "Use position='last' or position='mean' for extraction."
                )
        return activations

    def extract(
        self,
        target_data: List[str],
        contrast_data: List[str],
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract manifold-projected steering vectors.

        Matches paper equations:
        - Raw direction via difference-in-means (Eq. 2)
        - Manifold projection with top-k PCA basis (Eq. 9)
        """
        if contrast_data is None:
            raise ValueError("MANIFOLD requires contrast_data for difference-in-means extraction.")
        if self.manifold_dim <= 0:
            raise ValueError(f"manifold_dim must be positive, got {self.manifold_dim}.")

        target_acts_dict = self._get_activations(target_data)
        contrast_acts_dict = self._get_activations(contrast_data)

        vectors = {}
        raw_directions = {}
        explained_var = {}

        for layer in self.layer:
            target_acts = target_acts_dict[layer].to(torch.float32)
            contrast_acts = contrast_acts_dict[layer].to(torch.float32)

            # Eq. (2): difference in means between redundant and concise sets.
            raw_direction = target_acts.mean(dim=0) - contrast_acts.mean(dim=0)
            raw_direction = raw_direction / (raw_direction.norm() + 1e-8)
            raw_directions[layer] = raw_direction.to(self.device)

            # Build reasoning activation set D_reasoning = D_redundant U D_concise.
            reasoning_acts = torch.cat([target_acts, contrast_acts], dim=0)
            d_model = reasoning_acts.shape[1]
            n_samples = reasoning_acts.shape[0]
            n_components = min(self.manifold_dim, d_model, n_samples)
            if n_components < 1:
                raise ValueError(
                    f"Cannot run PCA for MANIFOLD at layer {layer}: "
                    f"n_samples={n_samples}, d_model={d_model}."
                )

            pca = PCA(n_components=n_components, whiten=False)
            pca.fit(reasoning_acts.cpu().numpy())

            # Eq. (9): r_manifold = U_k U_k^T r_raw, then normalize.
            basis = torch.from_numpy(pca.components_.T).to(self.device, dtype=torch.float32)
            raw_device = raw_direction.to(self.device)
            manifold_direction = basis @ (basis.T @ raw_device)
            manifold_direction = manifold_direction / (manifold_direction.norm() + 1e-8)
            vectors[layer] = manifold_direction
            explained_var[layer] = float(pca.explained_variance_ratio_.sum())

        self.vector = vectors
        self.raw_direction = raw_directions

        self.metadata = {
            "method": "MANIFOLD",
            "manifold_dim": self.manifold_dim,
            "explained_variance_ratio": explained_var,
        }

        return self.vector


class COLDExtractor(BaseExtractor):
    """
    COLD-Steer extractor (training-free learning-dynamics approximation).

    Paper defaults used here:
    - variant: finite-difference (COLD-FD)
    - epsilon: 1e-6
    - eta: 1.0

    Notes:
    - The original paper defines an input-conditional intervention.
      This implementation adapts COLD to the repository's static-vector API by
      extracting a per-layer dense vector from training data.
    - `cold_variant="kernel"` falls back to the unit-kernel approximation
      (equivalent to contrastive mean-difference under the COLD discussion).
    """

    METHOD_NAME = "COLD"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        cold_variant: str = "fd",
        cold_eta: float = 1.0,
        cold_epsilon: float = 1e-6,
        cold_pair_margin: float = 0.0,
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.cold_variant = str(cold_variant).lower()
        self.cold_eta = float(cold_eta)
        self.cold_epsilon = float(cold_epsilon)
        self.cold_pair_margin = float(cold_pair_margin)

        if self.cold_variant not in {"fd", "kernel"}:
            raise ValueError("cold_variant must be one of {'fd', 'kernel'}")
        if self.cold_epsilon <= 0:
            raise ValueError(f"cold_epsilon must be > 0, got {self.cold_epsilon}")

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        """Collect per-layer mean residual activations."""
        change_pad_token = self.change_pad_token
        if isinstance(change_pad_token, tuple):
            change_pad_token = change_pad_token[0]

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
            change_pad_token=bool(change_pad_token),
        )

    def _prompt_nll(self, prompt: str) -> torch.Tensor:
        """Compute next-token negative log-likelihood for a single prompt."""
        tokens = self.model.to_tokens(prompt, prepend_bos=True).to(self.device)
        return self.model(tokens, return_type="loss")

    def _build_loss(
        self,
        target_prompt: str,
        contrast_prompt: Optional[str],
    ) -> torch.Tensor:
        """Build COLD training loss for one sample (pairwise or positive-only)."""
        target_loss = self._prompt_nll(target_prompt)
        if contrast_prompt is None:
            return target_loss

        contrast_loss = self._prompt_nll(contrast_prompt)
        pair_loss = target_loss - contrast_loss
        if self.cold_pair_margin > 0:
            pair_loss = torch.relu(pair_loss + self.cold_pair_margin)
        return pair_loss

    def _accumulate_param_grads(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]],
    ) -> Dict[str, torch.Tensor]:
        """Accumulate gradients of COLD loss w.r.t. model parameters over train examples."""
        self.model.zero_grad(set_to_none=True)
        self.model.train(False)

        n = len(target_data)
        for idx, target_prompt in enumerate(tqdm(target_data, desc="COLD: accumulating grads")):
            contrast_prompt = None
            if contrast_data is not None and idx < len(contrast_data):
                contrast_prompt = contrast_data[idx]

            loss = self._build_loss(target_prompt, contrast_prompt)
            loss.backward()

        grads: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()

        self.model.zero_grad(set_to_none=True)

        if not grads:
            raise RuntimeError("COLD-FD failed to collect parameter gradients.")

        return grads

    def _apply_param_delta(
        self,
        grads: Dict[str, torch.Tensor],
        scale: float,
    ) -> None:
        """Apply in-place parameter perturbation theta <- theta + scale * grad."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                grad = grads.get(name)
                if grad is None:
                    continue
                param.add_(scale * grad.to(device=param.device, dtype=param.dtype))

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if not target_data:
            raise ValueError("COLD requires non-empty target_data.")

        base_target = self._get_activations(target_data)
        if contrast_data is not None:
            base_contrast = self._get_activations(contrast_data)
        else:
            base_contrast = {layer: torch.zeros_like(base_target[layer]) for layer in self.layer}

        # Unit-kernel COLD approximation (matches paper discussion and is static).
        if self.cold_variant == "kernel":
            self.vector = {
                layer: self.cold_eta * (base_target[layer] - base_contrast[layer])
                for layer in self.layer
            }
            self.metadata = {
                "method": "COLD",
                "variant": "kernel",
                "eta": self.cold_eta,
                "n_target": len(target_data),
                "n_contrast": len(contrast_data) if contrast_data is not None else 0,
            }
            return self.vector

        # Finite-difference COLD-FD approximation.
        grads = self._accumulate_param_grads(target_data, contrast_data)
        self._apply_param_delta(grads, self.cold_epsilon)
        try:
            pert_target = self._get_activations(target_data)
            if contrast_data is not None:
                pert_contrast = self._get_activations(contrast_data)
            else:
                pert_contrast = {layer: torch.zeros_like(pert_target[layer]) for layer in self.layer}
        finally:
            self._apply_param_delta(grads, -self.cold_epsilon)

        n = max(len(target_data), 1)
        fd_scale = -self.cold_eta / (self.cold_epsilon * float(n))
        self.vector = {}
        for layer in self.layer:
            delta_target = pert_target[layer] - base_target[layer]
            delta_contrast = pert_contrast[layer] - base_contrast[layer]
            self.vector[layer] = fd_scale * (delta_target - delta_contrast)

        self.metadata = {
            "method": "COLD",
            "variant": "fd",
            "eta": self.cold_eta,
            "epsilon": self.cold_epsilon,
            "pair_margin": self.cold_pair_margin,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data) if contrast_data is not None else 0,
        }

        return self.vector


class SphericalExtractor(BaseExtractor):
    """
    Spherical Steering extractor.

    Builds contrastive prototypes and exports the truthful prototype (`mu_T`) as
    steering vector per layer. The hallucinated prototype is antipodal (`mu_H=-mu_T`)
    and stored in metadata for steer-time gating.
    """

    METHOD_NAME = "SPHERICAL"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        """Collect non-reduced per-sample residual activations for configured layers."""
        change_pad_token = self.change_pad_token
        if isinstance(change_pad_token, tuple):
            change_pad_token = change_pad_token[0]

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
            change_pad_token=bool(change_pad_token),
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if not target_data or not contrast_data:
            raise ValueError("SPHERICAL requires both target_data and contrast_data.")

        target_acts = self._get_activations(target_data)
        contrast_acts = self._get_activations(contrast_data)

        self.vector = {}
        spherical_mu_h: Dict[int, torch.Tensor] = {}
        prototype_stats: Dict[int, Dict[str, Any]] = {}

        for layer in self.layer:
            mean_target = target_acts[layer].mean(dim=0).float()
            mean_contrast = contrast_acts[layer].mean(dim=0).float()
            diff = mean_target - mean_contrast

            mu_t = diff / diff.norm(p=2).clamp(min=1e-8)
            mu_h = -mu_t

            self.vector[layer] = mu_t
            spherical_mu_h[layer] = mu_h
            prototype_stats[layer] = {
                "target_norm": float(mean_target.norm().item()),
                "contrast_norm": float(mean_contrast.norm().item()),
                "diff_norm": float(diff.norm().item()),
            }

        self.metadata = {
            "method": "SPHERICAL",
            "spherical_mu_h": spherical_mu_h,
            "prototype_stats": prototype_stats,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data),
        }

        return self.vector


def riemannian_block_update(
    h_last: "np.ndarray",
    T: int = 50,
    calpha_k: float = 1.0,
    seed: int = 0,
) -> "np.ndarray":
    """Riemannian block-coordinate descent on product of spheres.

    Ported from SPREAD (AAAI 2026, arxiv:2511.08305).

    Finds perturbation vectors V ∈ R^{N×D} on per-sample unit spheres that
    maximize log det(I + (H+V)(H+V)^T), where H = h_last.

    Args:
        h_last: (N, D) matrix of per-sample activations.
        T: Number of block-coordinate Riemannian gradient steps.
        calpha_k: Scale constant for per-sample sphere radius.
        seed: Random seed for reproducibility.

    Returns:
        V: (N, D) perturbation vectors (each row lies on a sphere).
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    N, D = h_last.shape

    alpha_k = [calpha_k * (np.linalg.norm(h_last[k]) / D) for k in range(N)]
    H_norm_fro = np.linalg.norm(h_last, ord="fro")
    H_bar = np.mean(h_last, axis=0)

    V = np.empty((N, D), dtype=np.float64)
    for k in range(N):
        epsilon = rng.uniform(0, 1, size=D)
        vk0 = h_last[k] - H_bar + epsilon
        norm_k = np.linalg.norm(vk0)
        if norm_k < 1e-12:
            vk0 = rng.randn(D)
            norm_k = np.linalg.norm(vk0)
        V[k] = vk0 / norm_k * np.sqrt(alpha_k[k])

    for _ in range(T):
        for k in range(N):
            HV = h_last + V
            M = np.eye(N) + HV @ HV.T
            M_inv = np.linalg.inv(M)
            L = 2 + 4 * (H_norm_fro + alpha_k[k]) ** 2 + (
                2 / np.sqrt(alpha_k[k])
            ) * (H_norm_fro + alpha_k[k])
            eta_k = 1.0 / L
            g_k = -2 * HV.T @ M_inv[:, k]
            v_k_prev = V[k]
            proj_grad = g_k - (1 / alpha_k[k]) * np.dot(v_k_prev, g_k) * v_k_prev
            d_k = eta_k * proj_grad
            d_k_norm = np.linalg.norm(d_k)
            if d_k_norm > 1e-12:
                V[k] = (
                    np.cos(d_k_norm / np.sqrt(alpha_k[k])) * v_k_prev
                    - np.sin(d_k_norm / np.sqrt(alpha_k[k])) * d_k / d_k_norm * np.sqrt(alpha_k[k])
                )

    return V


class RiemannianExtractor(BaseExtractor):
    """
    Riemannian Activation Steering extractor (SPREAD, AAAI 2026).

    Collects per-sample activations from target and contrast prompts,
    runs Riemannian block-coordinate descent on the product of unit spheres
    to find diversity-maximizing perturbation vectors, then extracts the
    contrastive direction from the Riemannian-optimized perturbation space.

    Paper: "SPREAD: SPherical Intervention for REAsoning Diversity" (arxiv:2511.08305)
    """

    METHOD_NAME = "RIEMANNIAN"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
        riemannian_steps: int = 50,
        riemannian_calpha: float = 1.0,
        riemannian_seed: int = 0,
        riemannian_norm_output: bool = True,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point)
        self.position = position
        self.riemannian_steps = riemannian_steps
        self.riemannian_calpha = riemannian_calpha
        self.riemannian_seed = riemannian_seed
        self.riemannian_norm_output = riemannian_norm_output

    def _get_activations(self, inputs: List[str]) -> Dict[int, "torch.Tensor"]:
        """Collect per-sample (not averaged) activations for multiple layers."""
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
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, "torch.Tensor"]:
        """Extract Riemannian-optimized contrastive steering vectors.

        For each layer:
        1. Collect per-sample activations H_target, H_contrast
        2. Stack into H ∈ R^{(N_t + N_c) × D}
        3. Run riemannian_block_update(H) → V ∈ R^{(N_t + N_c) × D}
        4. v = mean(V[:N_t]) - mean(V[N_t:])
        5. Optionally L2-normalize v
        6. Compute Mahalanobis metric M = (Σ_w + λI)^{-1} from activations
        """
        target_acts = self._get_activations(target_data)
        if contrast_data is not None and len(contrast_data) > 0:
            contrast_acts = self._get_activations(contrast_data)
        else:
            contrast_acts = None

        self.vector = {}
        rie_stats = {}
        mahal_metrics = {}

        for layer_idx in self.layer:
            h_target = target_acts[layer_idx].float().cpu().numpy()
            N_t = h_target.shape[0]

            if contrast_acts is not None and layer_idx in contrast_acts:
                h_contrast = contrast_acts[layer_idx].float().cpu().numpy()
            else:
                h_contrast = None

            if h_contrast is not None and h_contrast.shape[0] > 0:
                N_c = h_contrast.shape[0]
                H_all = np.concatenate([h_target, h_contrast], axis=0)
                V_all = riemannian_block_update(
                    H_all, T=self.riemannian_steps, calpha_k=self.riemannian_calpha, seed=self.riemannian_seed
                )
                v_target = V_all[:N_t].mean(axis=0)
                v_contrast = V_all[N_t:].mean(axis=0)
                v = v_target - v_contrast

                mu_t = h_target.mean(axis=0)
                mu_c = h_contrast.mean(axis=0)
                cov_t = np.cov(h_target, rowvar=False) if N_t > 1 else np.zeros((h_target.shape[1], h_target.shape[1]))
                cov_c = np.cov(h_contrast, rowvar=False) if N_c > 1 else np.zeros((h_contrast.shape[1], h_contrast.shape[1]))
                sigma_w = (N_t * cov_t + N_c * cov_c) / (N_t + N_c)
            else:
                H_all = h_target
                V_all = riemannian_block_update(
                    H_all, T=self.riemannian_steps, calpha_k=self.riemannian_calpha, seed=self.riemannian_seed
                )
                v = V_all.mean(axis=0)
                sigma_w = np.cov(h_target, rowvar=False) if N_t > 1 else np.zeros((h_target.shape[1], h_target.shape[1]))

            D = sigma_w.shape[0]
            lam = 0.01 * np.trace(sigma_w) / D
            sigma_reg = sigma_w + lam * np.eye(D)
            try:
                M = np.linalg.inv(sigma_reg)
            except np.linalg.LinAlgError:
                M = np.linalg.pinv(sigma_reg)

            M_norm = np.linalg.norm(M)
            if M_norm > 1e-12:
                M = M / M_norm

            mahal_metrics[layer_idx] = torch.tensor(M, dtype=torch.float32)

            v_tensor = torch.tensor(v, dtype=torch.float32)
            if self.riemannian_norm_output:
                norm = v_tensor.norm(p=2)
                if norm > 1e-12:
                    v_tensor = v_tensor / norm

            self.vector[layer_idx] = v_tensor.to(
                device=self.device or "cuda"
            )

            rie_stats[layer_idx] = {
                "v_norm": float(v_tensor.norm().item()) if not self.riemannian_norm_output else 1.0,
                "n_target": N_t,
                "n_contrast": h_contrast.shape[0] if h_contrast is not None else 0,
                "mahal_norm": float(M_norm),
                "sigma_w_cond": float(np.linalg.cond(sigma_reg)),
            }

        self.metadata = {
            "method": "RIEMANNIAN",
            "riemannian_steps": self.riemannian_steps,
            "riemannian_calpha": self.riemannian_calpha,
            "riemannian_norm_output": self.riemannian_norm_output,
            "riemannian_mahal": mahal_metrics,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data) if contrast_data else 0,
            "rie_stats": rie_stats,
        }

        return self.vector
