"""
Model Configuration.

Contains ModelConfig and MODEL_SAE_REGISTRY.
"""

from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, Optional
import torch


@dataclass
class ModelConfig:
    """Model and SAE configuration."""
    name: str = "google/gemma-2-2b"
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_compile: bool = False  # Enable torch.compile for faster inference
    sae_release: Optional[str] = None
    sae_id: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    do_sample: bool = False
    sae_cpu: bool = False  # Load SAE on CPU (True) or same CUDA device as model (False)
    model_kwargs: Optional[Dict[str, Any]] = None
    
    def get_dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.dtype, torch.bfloat16)
    
    def get_sae_release(self) -> str:
        if self.sae_release:
            return MODEL_SAE_REGISTRY.get((self.name + "-" + self.sae_release).split("/")[-1].lower(), {}).get("release", "")
        return MODEL_SAE_REGISTRY.get(self.name.split("/")[-1].lower(), {}).get("release", "")
    
    def get_sae_id(self, layer) -> str:
        if self.sae_id:
            registry = MODEL_SAE_REGISTRY.get((self.name + "-" + self.sae_id).split("/")[-1].lower(), {})
        else: registry = MODEL_SAE_REGISTRY.get(self.name.split("/")[-1].lower(), {})
        id = registry.get("id", "")
        if registry.get('shift_layer', False):
            return id.format(layer - 1)
        return id.format(layer)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dict, ignoring invalid keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_args(cls, args: Dict[str, Any]) -> "ModelConfig":
        """Create from args (alias for from_dict for ModelConfig)."""
        return cls.from_dict(args)


MODEL_SAE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gemma-2-2b": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "id": "layer_{}/width_16k/canonical",
        "n_layers": 26,
        "d_model": 2304,
        "available_widths": ["16k", "65k"],
    },
    "gemma-2-2b-it": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "id": "layer_{}/width_16k/canonical",
        "n_layers": 26,
        "d_model": 2304,
        "available_widths": ["16k", "65k"],
    },
    "gemma-2-2b-65": {
        "release": "gemma-scope-2b-pt-res-canonical",
        "id": "layer_{}/width_65k/canonical",
        "n_layers": 26
    },
    "gemma-2-9b": {
        "release": "gemma-scope-9b-pt-res-canonical",
        "id": "layer_{}/width_65k/canonical",
        "n_layers": 42
    },
    "gemma-2-9b-131": {
        "release": "gemma-scope-9b-pt-res-canonical",
        "id": "layer_{}/width_131k/canonical",
        "n_layers": 46,
    },

    "llama-2-7b-hf": {
        "release": "yuzhaouoe/Llama2-7b-SAE",
        "id": "layers.{}", # Based on spare/utils.py logic
        "n_layers": 32
    },
    "llama-2-7b-chat-hf": {
        "release": "yuzhaouoe/Llama2-7b-SAE",
        "id": "layers.{}",
        "n_layers": 32
    },
    "llama-3.1-8b": {
        "release": "llama_scope_lxr_8x",
        "id": "l{}r_8x",
        "n_layers": 32
    },
    "llama-3.1-8b-instruct": {
        "release": "llama_scope_lxr_8x",
        "id": "l{}r_8x",
        "n_layers": 32
    },
    "llama-3.2-1b-instruct": {
        "release": "chanind/sae-llama-3.2-1b-topk-res",
        "id": "blocks.{}.hook_resid_post/l0-10",
        "n_layers": 32,
        "shift_layer": True
    }
}
