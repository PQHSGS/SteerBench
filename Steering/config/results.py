"""
Result Configuration/Data Structures.

Defines the structure for evaluation results and steering vectors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import torch
from typing import Union


@dataclass
class SteeringVector:
    """
    Encapsulates a steering vector with method-specific metadata.
    
    This is the output of extraction and input to steering.
    Only `vector` is mandatory - all method-specific data goes in `metadata`.
    
    The layer/method/hook_point are already in ExtractorConfig/SteerConfig,
    so this class only carries the extracted data.
    
    Vector can be:
    - Single tensor: torch.Tensor (single layer)
    - Multiple tensors: Dict[int, torch.Tensor] (layer_idx -> vector)
    
    Example metadata by method:
        - CAST: {"conditional_vector": tensor, "conditional_threshold": float}
        - ANGULAR: {"steering_plane": tensor, "feature_direction": tensor}
        - SPARE: {"zC": tensor, "zM": tensor, "top_k_indices": tensor}
        - SRE: {"I_plus": tensor, "I_minus": tensor}
        - SSV: {"selected_indices": list, "important_dims": tensor}
    """
    vector: Union[torch.Tensor, Dict[int, torch.Tensor]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    VECTOR_FILENAME = "vector.pt"
    METADATA_FILENAME = "metadata.pt"
    
    def to_steer_params(self) -> Dict[str, Any]:
        """
        Convert to params dict for steer model constructor.
        
        Returns steering_vector + all metadata keys flattened.
        For multi-layer vectors, returns {"steering_vector": dict} instead of {"steering_vector": tensor}
        """
        params = {"steering_vector": self.vector}
        params.update(self.metadata)
        return params
    
    def save(self, path: str):
        """Save vector in a single folder layout: <name>/vector.pt + <name>/metadata.pt."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        vector_path = save_dir / self.VECTOR_FILENAME
        metadata_path = save_dir / self.METADATA_FILENAME

        torch.save(self.vector, vector_path)
        torch.save(self.metadata or {}, metadata_path)


            
    @classmethod
    def load(cls, path: str, layer: List[int], device: str = "cuda") -> "SteeringVector":
        """Load vector from the single folder layout: <name>/vector.pt + <name>/metadata.pt."""
        vector_dir = Path(path)
        vector_path = vector_dir / cls.VECTOR_FILENAME
        metadata_path = vector_dir / cls.METADATA_FILENAME

        if not vector_path.exists():
            raise FileNotFoundError(f"Vector file not found: {vector_path}")

        vec_data = torch.load(vector_path, map_location="cpu")
        if not isinstance(vec_data, dict):
            raise TypeError(
                f"Expected vector payload as Dict[int, Tensor], got {type(vec_data).__name__}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        try:
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        except TypeError:
            metadata = torch.load(metadata_path, map_location="cpu")

        if not isinstance(metadata, dict):
            raise TypeError("metadata.pt must contain a dictionary")

        normalized_vec = {int(k): v for k, v in vec_data.items()}
        
        # For methods like Angular that select their own layer, overwrite requested_layers from metadata
        selected_layer = metadata.get("selected_layer")
        if selected_layer is not None:
            if isinstance(selected_layer, list) and len(selected_layer) > 0 and isinstance(selected_layer[0], list):
                requested_layers = selected_layer[0]
            elif isinstance(selected_layer, list):
                requested_layers = selected_layer
            else:
                requested_layers = [selected_layer]
            requested_layers = [int(l) for l in requested_layers]
        else:
            requested_layers = [int(l) for l in layer]

        missing = [l for l in requested_layers if l not in normalized_vec]
        if missing:
            raise KeyError(
                f"Missing steering vectors for requested layers {missing}. "
                f"Available: {sorted(normalized_vec.keys())}"
            )

        filtered_vec = {l: normalized_vec[l].to(device) for l in requested_layers}
        return cls(vector=filtered_vec, metadata=metadata)


@dataclass
class SampleResult:
    """Detailed result for a single sample (for debugging)."""
    prompt: str
    response: str
    ground_truth: Any
    is_correct: int
    confidence: float
    sample_data: Optional[Dict] = None  # Original sample data
    metadata: Optional[Dict] = None  # Additional metadata
    
    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "ground_truth": str(self.ground_truth),
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "sample_data": self.sample_data,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    method: str
    train_dataset: Optional[str]
    test_dataset: str
    layer: Union[int, List[int]]
    coeff: Union[float, Dict]
    accuracy: float
    baseline_accuracy: float = 0.0
    total: int = 0
    correct: int = 0
    samples: Optional[List[SampleResult]] = None  # Detailed sample results
    baseline_samples: Optional[List[SampleResult]] = None  # Baseline sample results
    perplexity: Optional[float] = None              # Mean perplexity (steered)
    baseline_perplexity: Optional[float] = None     # Mean perplexity (baseline)
    extraction_flops: Optional[int] = None
    inference_flops: Optional[int] = None
    repetition_rate: Optional[float] = None
    baseline_repetition_rate: Optional[float] = None
    compression_ratio: Optional[float] = None
    baseline_compression_ratio: Optional[float] = None
    lsp_score: Optional[float] = None           # Localized Suffix-Penalized Perplexity (PPL_lsp)
    baseline_lsp_score: Optional[float] = None  # Baseline PPL_lsp

    @property
    def delta(self) -> float:
        return self.accuracy - self.baseline_accuracy
    
    def to_dict(self, include_samples: bool = True) -> Dict:
        d = {
            "method": self.method,
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "layer": self.layer,
            "coeff": self.coeff,
            "accuracy": self.accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "total": self.total,
            "correct": self.correct,
            "delta": self.delta,
            "perplexity": self.perplexity,
            "baseline_perplexity": self.baseline_perplexity,
            "repetition_rate": self.repetition_rate,
            "baseline_repetition_rate": self.baseline_repetition_rate,
            "compression_ratio": self.compression_ratio,
            "baseline_compression_ratio": self.baseline_compression_ratio,
            "lsp_score": self.lsp_score,
            "baseline_lsp_score": self.baseline_lsp_score,
            "extraction_flops": self.extraction_flops,
            "inference_flops": self.inference_flops,
        }
        if include_samples and self.samples:
            d["samples"] = [s.to_dict() for s in self.samples]
        if include_samples and self.baseline_samples:
            d["baseline_samples"] = [s.to_dict() for s in self.baseline_samples]
        return d

