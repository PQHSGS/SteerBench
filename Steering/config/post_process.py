"""Configuration for GLP post-processing."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Union


@dataclass
class PostProcessConfig:
    """Config for optional GLP denoising on residual stream activations."""

    enabled: bool = False
    source: Optional[str] = 'PQPQPQHUST/glp-gemma'  # Local path or Hugging Face repo id
    checkpoint: str = "final"

    layer: Optional[Union[int, List[int]]] = None
    hook_point: Union[str, List[str]] = "pre"
    position: Union[str, int] = "last"

    noise_rate: float = 0.5
    num_timesteps: int = 20

    apply_on_baseline: bool = False

    # Classifier guidance
    use_classifier: bool = False
    scale: float = 1.0
    negative: bool = False
    classifier_checkpoint: str = "final"
    classifier_source: Optional[str] = None
    classifier_guidance_start_step: int = 0
    classifier_guidance_end_step: Optional[int] = None
    classifier_grad_clip: Optional[float] = 5.0
    classifier_normalize_grad: bool = True

    def to_dict(self, include_none: bool = False) -> Dict[str, Any]:
        """Return dict representation. When `include_none` is False, drop keys with None values."""
        data = asdict(self)
        if include_none:
            return data
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostProcessConfig":
        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            warnings.warn(
                f"PostProcessConfig: ignoring unknown keys {sorted(unknown)}. "
                "Check for typos or stale config fields.",
                stacklevel=2,
            )
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def resolve(self) -> "PostProcessConfig":
        if self.num_timesteps <= 0:
            raise ValueError("post_process.num_timesteps must be > 0")
        if not (0.0 <= float(self.noise_rate) <= 1.0):
            raise ValueError("post_process.noise_rate must be in [0, 1]")
        if float(self.scale) < 0.0:
            raise ValueError("post_process.scale must be >= 0")
        if int(self.classifier_guidance_start_step) < 0:
            raise ValueError("post_process.classifier_guidance_start_step must be >= 0")
        if self.classifier_guidance_end_step is not None:
            if int(self.classifier_guidance_end_step) < int(self.classifier_guidance_start_step):
                raise ValueError(
                    "post_process.classifier_guidance_end_step must be >= "
                    "post_process.classifier_guidance_start_step"
                )
        if self.classifier_grad_clip is not None and float(self.classifier_grad_clip) <= 0.0:
            raise ValueError("post_process.classifier_grad_clip must be > 0 when provided")
        if self.enabled and not self.source:
            raise ValueError("post_process.source is required when post_process.enabled is true")
        return self
