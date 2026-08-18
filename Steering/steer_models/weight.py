"""
Weight Steering Model.
"""

from typing import List, Optional, Dict, Any, Union
import re
import torch
from ..base import BaseSteerModel
from ..logger import setup_logger

logger = setup_logger(__name__)

def get_layer_number(key: str) -> Optional[int]:
    """Parse layer index from parameter name."""
    match = re.search(r"blocks\.(\d+)\.", key)
    if match:
        return int(match.group(1))
    return None

class WeightSteerModel(BaseSteerModel):
    """
    Weight Steering Model.
    
    Temporarily applies the contrastive weight deltas (w_b) to the base model's
    target parameters during text generation and logit evaluation, and safely
    restores the baseline parameter weights immediately afterward.
    
    Paper: "Steering Language Models with Weight Arithmetic"
    """

    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        weight_deltas: Optional[Dict[str, torch.Tensor]] = None,
        positive_lora_state: Optional[Dict[str, torch.Tensor]] = None,
        negative_lora_state: Optional[Dict[str, torch.Tensor]] = None,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        **kwargs,
    ):
        # We store the parameter-level weight deltas mapping
        self.weight_deltas = {}
        if weight_deltas is not None:
            for name, delta in weight_deltas.items():
                # Cast/move weight delta to parameter device
                self.weight_deltas[name] = delta.to(device=model.cfg.device, dtype=model.cfg.dtype)
        elif positive_lora_state is not None:
            reconstructed = self._reconstruct_weight_deltas(
                model=model,
                positive_lora_state=positive_lora_state,
                negative_lora_state=negative_lora_state or {},
                lora_r=lora_r or 32,
                lora_alpha=lora_alpha or 64,
            )
            for name, delta in reconstructed.items():
                self.weight_deltas[name] = delta.to(device=model.cfg.device, dtype=model.cfg.dtype)
                
        # Initialize BaseSteerModel with a dummy zero vector map
        dummy_vector = {int(l): torch.zeros(1, device=model.cfg.device) for l in layer}
        super().__init__(model, layer, dummy_vector, **kwargs)

    def _reconstruct_weight_deltas(
        self,
        model,
        positive_lora_state: Dict[str, torch.Tensor],
        negative_lora_state: Dict[str, torch.Tensor],
        lora_r: int,
        lora_alpha: int,
    ) -> Dict[str, torch.Tensor]:
        import re
        
        pattern = re.compile(
            r"base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_A\.default\.weight"
        )
        
        combinations = set()
        for state_dict in [positive_lora_state, negative_lora_state]:
            if state_dict:
                for k in state_dict.keys():
                    match = pattern.match(k)
                    if match:
                        combinations.add((int(match.group(1)), match.group(2), match.group(3)))
                    
        weight_deltas = {}
        scaling = lora_alpha / lora_r
        
        for layer_idx, sub_module, module_name in combinations:
            key_A = f"base_model.model.model.layers.{layer_idx}.{sub_module}.{module_name}.lora_A.default.weight"
            key_B = f"base_model.model.model.layers.{layer_idx}.{sub_module}.{module_name}.lora_B.default.weight"
            
            # Positive lora delta
            if positive_lora_state and key_A in positive_lora_state and key_B in positive_lora_state:
                pos_A = positive_lora_state[key_A].to(device=model.cfg.device)
                pos_B = positive_lora_state[key_B].to(device=model.cfg.device)
                delta_pos = pos_B.to(pos_A.dtype) @ pos_A
            else:
                delta_pos = None
                
            # Negative lora delta
            if negative_lora_state and key_A in negative_lora_state and key_B in negative_lora_state:
                neg_A = negative_lora_state[key_A].to(device=model.cfg.device)
                neg_B = negative_lora_state[key_B].to(device=model.cfg.device)
                delta_neg = neg_B.to(neg_A.dtype) @ neg_A
            else:
                delta_neg = None
                
            if delta_pos is not None and delta_neg is not None:
                delta_HF = delta_pos - delta_neg
            elif delta_pos is not None:
                delta_HF = delta_pos
            elif delta_neg is not None:
                delta_HF = -delta_neg
            else:
                continue
                
            delta_HF = scaling * delta_HF
            
            # Retrieve model config properties with fallbacks
            n_heads = model.cfg.n_heads
            d_head = model.cfg.d_head
            d_model = model.cfg.d_model
            n_key_value_heads = getattr(model.cfg, "n_key_value_heads", n_heads)
            
            # Map delta_HF (shape [out_features, in_features]) to HookedTransformer named parameters
            if sub_module == "self_attn":
                if module_name == "q_proj":
                    ht_name = f"blocks.{layer_idx}.attn.W_Q"
                    mapped_delta = delta_HF.view(n_heads, d_head, d_model).transpose(1, 2)
                elif module_name == "k_proj":
                    ht_name = f"blocks.{layer_idx}.attn.W_K"
                    mapped_delta = delta_HF.view(n_key_value_heads, d_head, d_model).transpose(1, 2)
                elif module_name == "v_proj":
                    ht_name = f"blocks.{layer_idx}.attn.W_V"
                    mapped_delta = delta_HF.view(n_key_value_heads, d_head, d_model).transpose(1, 2)
                elif module_name == "o_proj":
                    ht_name = f"blocks.{layer_idx}.attn.W_O"
                    mapped_delta = delta_HF.view(d_model, n_heads, d_head).permute(1, 2, 0)
                else:
                    logger.warning(f"Unknown attention module: {module_name}")
                    continue
            elif sub_module == "mlp":
                if module_name == "gate_proj":
                    ht_name = f"blocks.{layer_idx}.mlp.W_gate"
                    mapped_delta = delta_HF.t()
                elif module_name == "up_proj":
                    ht_name = f"blocks.{layer_idx}.mlp.W_in"
                    mapped_delta = delta_HF.t()
                elif module_name == "down_proj":
                    ht_name = f"blocks.{layer_idx}.mlp.W_out"
                    mapped_delta = delta_HF.t()
                else:
                    logger.warning(f"Unknown MLP module: {module_name}")
                    continue
            else:
                continue
                
            weight_deltas[ht_name] = mapped_delta
            
        return weight_deltas

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        # Override to avoid registering any dummy activation hooks
        return []

    def hook_fn(
        self,
        resid: torch.Tensor,
        position: Union[str, int],
        coeff: float,
        steering_vector: torch.Tensor,
        hook,
        **kwargs
    ) -> torch.Tensor:
        # Dummy implementation to satisfy Abstract Base Class constraint.
        # Weight steering does not use activation hooks, so this is unused.
        return resid

    def _apply_weight_modifications(self, coeff: Dict[int, float]):
        """Apply weight delta: param = param + coeff * delta."""
        if not hasattr(self, "_original_weights_cache"):
            self._original_weights_cache = {}
            
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.weight_deltas:
                    # Cache baseline parameter data if not already cached
                    if name not in self._original_weights_cache:
                        self._original_weights_cache[name] = param.data.clone()
                        
                    layer = get_layer_number(name)
                    # Use coefficient for that layer, default to 1.0 if not found
                    c = coeff.get(layer, 1.0) if (layer is not None and isinstance(coeff, dict)) else 1.0
                    param.add_(c * self.weight_deltas[name])

    def _restore_weights(self, coeff: Dict[int, float]):
        """Restore original weights exactly from cached tensors to prevent numerical drift."""
        if hasattr(self, "_original_weights_cache"):
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self._original_weights_cache:
                        param.copy_(self._original_weights_cache[name])
            # Clear cache to save memory
            delattr(self, "_original_weights_cache")

    def generate(
        self,
        prompt: Union[str, List[str]],
        coeff: Optional[Dict[int, float]] = None,
        max_new_tokens: int = 150,
        apply_steer: bool = True,
        **kwargs,
    ) -> List[str]:
        # Parse coefficients
        if coeff is None:
            coeff = {layer: 1.0 for layer in self.layer}
            
        if apply_steer:
            self._apply_weight_modifications(coeff)
            
        try:
            return super().generate(
                prompt=prompt,
                coeff=coeff,
                max_new_tokens=max_new_tokens,
                apply_steer=apply_steer,
                **kwargs
            )
        finally:
            if apply_steer:
                self._restore_weights(coeff)

    def get_token_probs(
        self,
        prompt: str,
        tokens: List[str],
        coeff: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        apply_steer = bool(kwargs.get("apply_steer", True))
        if coeff is None:
            coeff = {layer: 1.0 for layer in self.layer}
            
        if apply_steer:
            self._apply_weight_modifications(coeff)
            
        try:
            return super().get_token_probs(
                prompt=prompt,
                tokens=tokens,
                coeff=coeff,
                **kwargs
            )
        finally:
            if apply_steer:
                self._restore_weights(coeff)
