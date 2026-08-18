"""
HF-based CAST extraction for exact match with CAST library.
Identical to CASTExtractor but uses HuggingFace model instead of TransformerLens hooks.
"""
import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
from typing import List, Optional, Dict, Any, Union
from tqdm import tqdm

from ..base import BaseExtractor
from ..data import DataLoader, TRAIN_DATASET_REGISTRY
from ..logger import setup_logger
logger = setup_logger(__name__)

class HFCASTExtractor(BaseExtractor):
    """
    Conditional Activation Steering (CAST) Extractor using HuggingFace model.
    
    Identical to CASTExtractor but uses HF model's output_hidden_states
    instead of TransformerLens hooks. Use when exact match with CAST library is needed.
    
    Takes inputs, returns vectors. Data handling done externally.
    """
    
    METHOD_NAME = "CAST_HF"
    
    def __init__(
        self,
        model,
        layer: List[int],
        model_name: str,
        batch_size: int = 8,
        position: str = "mean",  # CAST library uses accumulate_last_x_tokens="all" = mean
        use_pca: bool = True,
        conditional_layer: Optional[int] = None,
        conditional_dataset: Optional[str] = None,
        save_conditional_vector: Optional[str] = None,
        apply_chat_conditional_template: bool = True,
        device: Optional[torch.device] = None,
        hook_point: str = "pre",  # ignored, kept for interface compatibility
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point,position = position)
        self.use_pca = use_pca
        self.conditional_layer = conditional_layer
        self.conditional_dataset = conditional_dataset
        self.save_conditional_vector = save_conditional_vector
        self.apply_chat_conditional_template = apply_chat_conditional_template
        
        # Setup tokenizer with CAST library settings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = 0  # CRITICAL: Match CAST library
        
        # Will be set during extraction
        self.conditional_vector = None
    
    def _get_activations(
        self,
        inputs: List[str],
        layers: List[int],
        position: Optional[str] = None,
        return_all: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """Collect residual activations at specified layers using HF model.
        
        Args:
            inputs: List of input strings
            layers: Layers to extract from
            position: Token position ("last" or "mean")
            return_all: If True, return all individual activations, else return mean
            
        Returns:
            Dict mapping layer -> activation tensor
        """
        pos = position if position is not None else self.position
        
        result = {}
        for target_layer in layers:
            if return_all:
                all_acts = []
            else:
                sum_resid = None
                count = 0
            
            with torch.no_grad():
                for i in tqdm(range(0, len(inputs), self.batch_size), desc=f"HF L{target_layer}"):
                    batch = inputs[i:i + self.batch_size]
                    tokens = self.tokenizer(batch, padding=True, return_tensors="pt").to(self.model.device)
                    out = self.model(**tokens, output_hidden_states=True)
                    
                    # HF uses layer+1 indexing (hidden_states[0] is embeddings)
                    hidden = out.hidden_states[target_layer + 1]
                    
                    if pos == "last":
                        acts = hidden[:, -1, :]
                    elif pos == "mean":  # "mean"
                        acts = hidden.mean(dim=1)
                    else: 
                        acts = hidden[:,-2,:]    

                    if return_all:
                        all_acts.extend(acts.cpu().float().numpy())
                    else:
                        if sum_resid is None:
                            sum_resid = torch.zeros(acts.shape[-1], device=self.device, dtype=torch.float32)
                        sum_resid += acts.sum(dim=0).float().to(self.device)
                        count += acts.shape[0]
                    
                    del out
            
            if return_all:
                result[target_layer] = torch.tensor(np.array(all_acts), device=self.device, dtype=torch.float32)
            else:
                result[target_layer] = sum_resid / max(count, 1)
        
        return result
    
    def _extract_pca_vector(self, target_acts: torch.Tensor, contrast_acts: torch.Tensor) -> torch.Tensor:
        """Extract PCA-based steering vector using pairwise centering (like CAST library).
        
        IMPORTANT: CAST library uses INTERLEAVED data [pos1, neg1, pos2, neg2, ...]
        and accesses positives via h[::2] and negatives via h[1::2].
        """
        n_pairs = min(len(target_acts), len(contrast_acts))
        target_acts = target_acts[:n_pairs]
        contrast_acts = contrast_acts[:n_pairs]
        
        # Interleave like CAST library
        h = torch.zeros((n_pairs * 2, target_acts.shape[1]), dtype=target_acts.dtype, device=target_acts.device)
        h[::2] = target_acts
        h[1::2] = contrast_acts
        
        # Pairwise centering
        centers = (h[::2] + h[1::2]) / 2
        train = h.clone()
        train[::2] -= centers
        train[1::2] -= centers
        
        # PCA
        h_np = h.cpu().numpy()
        train_np = train.cpu().numpy()
        pca_model = PCA(n_components=1, whiten=False).fit(train_np)
        pca_vector = pca_model.components_.astype(np.float32).squeeze(axis=0)
        
        # Sign flip based on projection (like CAST library)
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
        layer: int,
        position: str,
    ) -> torch.Tensor:
        """Extract vector from target and contrast data at given layer and position."""
        if contrast_data is not None and self.use_pca:
            # Interleave for consistent batching
            n_pairs = min(len(target_data), len(contrast_data))
            interleaved = []
            for i in range(n_pairs):
                interleaved.append(target_data[i])
                interleaved.append(contrast_data[i])
            
            all_acts_dict = self._get_activations(interleaved, layers=[layer], position=position, return_all=True)
            all_acts = all_acts_dict[layer]
            target_acts = all_acts[::2]
            contrast_acts = all_acts[1::2]
            
            return self._extract_pca_vector(target_acts, contrast_acts)
        else:
            acts_dict = self._get_activations(target_data, layers=[layer], position=position)
            target_mean = acts_dict[layer]
            
            if contrast_data is not None:
                contrast_dict = self._get_activations(contrast_data, layers=[layer], position=position)
                contrast_mean = contrast_dict[layer]
                return target_mean - contrast_mean
            else:
                return target_mean
    
    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        conditional_target: Optional[List[str]] = None,
        conditional_contrast: Optional[List[str]] = None,
        conditional_layer: Optional[int] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract steering vector and optionally conditional vector.
        
        Args:
            target_data: Target prompts for steering vector
            contrast_data: Contrast prompts for steering vector
            conditional_target: Target prompts for conditional vector (optional)
            conditional_contrast: Contrast prompts for conditional vector (optional)
            
        Returns:
            Dict[int, Tensor] of steering vectors keyed by layer
        """
        # Update conditional_layer if provided
        self.conditional_layer = self.conditional_layer or conditional_layer or (self.layer[0] // 2)
        
        # --- Extract steering vectors at all layers ---
        self.vector = {}
        for layer in self.layer:
            self.vector[layer] = self._extract_vector(
                target_data, contrast_data, layer, self.position
            )
        
        # --- Extract conditional vector at earlier layer ---
        if conditional_target is not None and conditional_contrast is not None:
            self.conditional_vector = self._extract_vector(
                conditional_target, conditional_contrast, self.conditional_layer, "mean"
            )
        elif self.conditional_dataset is not None:
            logger.info(f"CAST_HF: Loading conditional dataset '{self.conditional_dataset}' for vector extraction...")
            cfg = TRAIN_DATASET_REGISTRY[self.conditional_dataset]
            data = DataLoader().load(self.conditional_dataset, apply_chat_template=self.apply_chat_conditional_template, tokenizer=self.tokenizer)
            cond_target = [d[cfg.target_key] for d in data]
            cond_contrast = [d[cfg.contrast_key] for d in data]
            self.conditional_vector = self._extract_vector(
                cond_target, cond_contrast, self.conditional_layer, "mean"
            )
        else:
            self.conditional_vector = None
        
        if self.save_conditional_vector and self.conditional_vector is not None:
            torch.save(self.conditional_vector.cpu(), self.save_conditional_vector)
            logger.info(f"Saved conditional vector to {self.save_conditional_vector}")
        
        self.metadata = {
            "method": "CAST_HF",
            "layer": self.layer,
            "conditional_layer": self.conditional_layer,
            "conditional_vector": self.conditional_vector,
            "has_conditional_vector": self.conditional_vector is not None,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data) if contrast_data else 0,
        }
        
        return self.vector
