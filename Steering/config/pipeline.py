"""
Experiment Configuration.

Contains PipelineConfig and quick builders.
"""

import warnings
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json

from .models import ModelConfig
from .methods import ExtractorConfig, SteerConfig
from .post_process import PostProcessConfig
from ..logger import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(raw: Union[Any, List[int], None]) -> List[int]:
    """int | List[int] | None → List[int]."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _normalize_top_k(raw: Union[int, List[int], range, None]) -> Optional[List[int]]:
    """Normalize steer.top_k to list form for SAE steer models.

    Semantics:
    - None: pass-through (no top-k post-selection)
    - int k: first k ranks -> [0, ..., k-1]
    - list/range: explicit rank positions
    """
    if raw is None:
        return None

    if isinstance(raw, int):
        if raw <= 0:
            return None
        values = list(range(raw))
    elif isinstance(raw, (list, range)):
        values = list(raw)
    else:
        raise TypeError(
            f"steer.top_k must be None, int, list[int], or range; got {type(raw).__name__}"
        )

    normalized: List[int] = []
    for item in values:
        idx = int(item)
        if idx < 0:
            raise ValueError(f"steer.top_k indices must be >= 0, got {idx}")
        normalized.append(idx)

    return normalized if normalized else None


def _broadcast(
    value: Union[Any, Dict],
    layers: List[int],
    name: str,
) -> Dict[int, Any]:
    """
    Expand a scalar / list / dict value to a Dict[int, T] keyed by *layers*.

    Rules
    -----
    - scalar → replicate for every layer
    - list   → zip with *layers* (lengths must match)
    - dict   → validate keys ⊆ layers (keys are stringified in JSON, so
      accept both str and int keys and convert to int)
    """
    if isinstance(value, dict):
        # JSON dicts always have string keys – normalise to int
        converted = {int(k): v for k, v in value.items()}
        return converted

    if isinstance(value, list):
        if len(value) != len(layers):
            raise ValueError(
                f"'{name}' list length ({len(value)}) must match "
                f"layer count ({len(layers)}): layers={layers}"
            )
        return dict(zip(layers, value))

    # scalar → broadcast
    return {layer: value for layer in layers}


def _infer_layers_from_dicts(*dicts: Optional[Dict]) -> Optional[List[int]]:
    """If any of the provided dicts is a non-None dict, return its int keys sorted."""
    for d in dicts:
        if isinstance(d, dict) and d:
            return sorted(int(k) for k in d.keys())
    return None


def _normalize_feat_feature_list(
    raw: Optional[Union[int, List[int], Dict[int, List[int]]]],
    layers: List[int],
) -> Optional[Dict[int, List[int]]]:
    """Normalize FEAT feature_list to Dict[int, List[int]] keyed by steer layers."""
    if raw is None:
        return None

    def _to_feature_list(value, field_name: str) -> List[int]:
        if isinstance(value, int):
            features = [value]
        elif isinstance(value, (list, tuple, range)):
            features = [int(v) for v in value]
        else:
            raise TypeError(
                f"{field_name} entries must be int or list[int], got {type(value).__name__}"
            )

        if not features:
            raise ValueError(f"{field_name} entries must be non-empty")

        for idx in features:
            if idx < 0:
                raise ValueError(f"{field_name} values must be >= 0, got {idx}")

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(features))

    if isinstance(raw, dict):
        converted = {int(k): v for k, v in raw.items()}
        keys = set(converted.keys())
        expected = set(layers)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                "steer.feature_list dict keys must match steer.layer. "
                f"Missing: {missing}, Extra: {extra}"
            )

        return {
            layer: _to_feature_list(converted[layer], f"steer.feature_list[{layer}]")
            for layer in layers
        }

    shared_features = _to_feature_list(raw, "steer.feature_list")
    return {layer: shared_features for layer in layers}


@dataclass
class PipelineConfig:
    """Complete experiment specification."""
    name: str = "experiment"
    description: str = ""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    extractor: ExtractorConfig = field(default_factory=lambda: ExtractorConfig(method="CAA", layer=14))
    steer: SteerConfig = field(default_factory=lambda: SteerConfig(method="CAA", layer=14))
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)
    
    # Evaluation
    test_dataset: str = "csqa"
    n_test: Optional[int] = None  # Number of test samples (optional, can be set in test dataset config)

    # -------------------------------------------------------------------------
    # DATA (moved from PipelineConfig)
    # -------------------------------------------------------------------------
    train_dataset: Optional[str] = None               # Dataset name from TRAIN_DATASET_REGISTRY
    n_train: Optional[int] = None                     # Number of training samples
    
    # Pipeline-level
    output: str = "./results"
    save_samples: bool = False
    include_baseline: bool = False
    compute_perplexity: bool = True   # Compute perplexity alongside evaluation
    compute_lsp: bool = True          # Compute Localized Suffix-Penalized Perplexity (PPL_lsp)
    seed: int = 42
    
    # Vector I/O
    save_vector: Optional[str] = None
    load_vector: Optional[str] = None

    def to_dict(self, include_none: bool = False, method_scoped: bool = False) -> Dict[str, Any]:
        if method_scoped:
            return {
                "name": self.name,
                "description": self.description,
                "model": self.model.to_dict(),
                "extractor": self.extractor.to_dict(include_none=include_none, method_scoped=True),
                "steer": self.steer.to_dict(include_none=include_none, method_scoped=True),
                "post_process": self.post_process.to_dict(include_none=include_none),
                "train_dataset": self.train_dataset,
                "n_train": self.n_train,
                "test_dataset": self.test_dataset,
                "n_test": self.n_test,
                "output": self.output,
                "include_baseline": self.include_baseline,
                "compute_perplexity": self.compute_perplexity,
                "compute_lsp": self.compute_lsp,
                "save_samples": self.save_samples,
                "seed": self.seed,
                "save_vector": self.save_vector,
                "load_vector": self.load_vector,
            }

        return {
            "name": self.name,
            "description": self.description,
            "model": self.model.to_dict(),
            "extractor": self.extractor.to_dict(),
            "steer": self.steer.to_dict(),
            "post_process": self.post_process.to_dict(),
                "train_dataset": self.train_dataset,
                "n_train": self.n_train,
            "test_dataset": self.test_dataset,
            "n_test": self.n_test,
            "output": self.output,
            "include_baseline": self.include_baseline,
            "compute_perplexity": self.compute_perplexity,
            "compute_lsp": self.compute_lsp,
            "save_samples": self.save_samples,
            "seed": self.seed,
            "save_vector": self.save_vector,
            "load_vector": self.load_vector,
        }

    def save(
        self,
        path: Union[str, Path],
        method_scoped: bool = True,
        include_none: bool = False,
    ):
        with open(path, "w") as f:
            payload = self.to_dict(include_none=include_none, method_scoped=method_scoped)
            json.dump(payload, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dict, auto-handling nested configs and legacy field migration."""

        # 1. Model
        model_data = data.get("model", {})
        if isinstance(model_data, dict):
            model_config = ModelConfig.from_dict(model_data)
        else:
            model_config = model_data

        # 2. Extractor
        ext_data = dict(data.get("extractor", {}))
        extractor_config = ExtractorConfig.from_dict(ext_data)

        # 3. Steer
        steer_data = dict(data.get("steer", {}))
        steer_config = SteerConfig.from_dict(steer_data)

        # 4. Post-process
        post_process_data = dict(data.get("post_process", {}))
        post_process_config = PostProcessConfig.from_dict(post_process_data)

        """Create from dict, warning on unknown keys."""
        skip_keys = {"model", "extractor", "steer", "post_process"}
        valid_keys = {f.name for f in fields(cls)} - skip_keys
        unknown = set(data.keys()) - valid_keys - skip_keys
        if unknown:
            warnings.warn(
                f"PipelineConfig: ignoring unknown keys {sorted(unknown)}. "
                f"Check for typos or stale config fields.",
                stacklevel=2,
            )
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(
            model=model_config,
            extractor=extractor_config,
            steer=steer_config,
            post_process=post_process_config,
            **filtered,
        )
        # return cls(
        #     name=data.get("name", "experiment"),
        #     description=data.get("description", ""),
        #     model=model_config,
        #     extractor=extractor_config,
        #     steer=steer_config,
        #     train_dataset=data.get("train_dataset"),
        #     n_train=data.get("n_train"),
        #     test_dataset=data.get("test_dataset"),
        #     n_test=data.get("n_test"),
        #     output=data.get("output", "./results"),
        #     save_samples=data.get("save_samples", False),
        #     include_baseline=data.get("include_baseline", True),
        #     seed=data.get("seed", 42),
        #     save_vector=data.get("save_vector"),
        #     load_vector=data.get("load_vector"),
        # )

    def resolve(self) -> "PipelineConfig":
        """
        Resolve this config by normalizing layer → List[int] and coeff → Dict[int, float].

        Layer inference priority:
        1. If coeff is already a dict → layers = sorted(dict.keys())
        2. Else use the explicit ``layer`` field in ExtractorConfig / SteerConfig
        """

        ext = self.extractor
        steer = self.steer

        # --- Extractor layers --------------------------------------------------
        ext_layers = _normalize(ext.layer)
        if not ext_layers:
            raise ValueError("Cannot determine extractor layers: provide 'layer' in ExtractorConfig")
        # --- Extractor hook points --------------------------------------------------
        ext_hook_points = _normalize(ext.hook_point)
        if not ext_hook_points:
            raise ValueError("Cannot determine extractor hook points: provide 'hook_point' in ExtractorConfig")
        # --- Steer layers ------------------------------------------------------
        steer_layers_from_dict = _infer_layers_from_dicts(
            steer.coeff if isinstance(steer.coeff, dict) else None,
        )
        steer_layers_explicit = _normalize(steer.layer)
        steer_layers = steer_layers_from_dict or steer_layers_explicit
        if not steer_layers:
            raise ValueError("Cannot determine steer layers: provide 'layer' or a dict-valued 'coeff'")

        # --- Steer hook points --------------------------------------------------
        steer_hook_points = _normalize(steer.hook_point)
        if not steer_hook_points:
            raise ValueError("Cannot determine steer hook points: provide 'hook_point' in SteerConfig")

        # --- Broadcast coeff to Dict[int, float] -------------------------------
        coeff_map = _broadcast(steer.coeff, steer_layers, "steer.coeff")

        # --- Modify configs in-place -------------------------------------------
        ext.layer = ext_layers
        ext.hook_point = ext_hook_points
        steer.layer = steer_layers
        steer.hook_point = steer_hook_points
        steer.coeff = coeff_map
        steer.top_k = _normalize_top_k(steer.top_k)

        # --- Post-process ------------------------------------------------------
        post_process = self.post_process.resolve()
        post_layers = _normalize(post_process.layer)
        if post_layers:
            post_process.layer = post_layers
        else:
            post_process.layer = list(steer.layer)

        post_hook_points = _normalize(post_process.hook_point)
        if not post_hook_points:
            raise ValueError("Cannot determine post-process hook points: provide 'hook_point' in PostProcessConfig")
        post_process.hook_point = post_hook_points

        if steer.method.upper() == "FEAT":
            steer.feature_list = _normalize_feat_feature_list(steer.feature_list, steer_layers)
            if steer.feature_list is None:
                raise ValueError("FEAT requires steer.feature_list")

            steer.overwrite_with_max_act = bool(steer.overwrite_with_max_act)

            if steer.max_act is not None:
                steer.max_act = float(steer.max_act)

            if steer.overwrite_with_max_act and steer.max_act is None:
                raise ValueError(
                    "FEAT requires steer.max_act when overwrite_with_max_act is enabled"
                )

        # --- Validate ------------------------------------------------------------
        self._validate()

        return self

    def _validate(self) -> "PipelineConfig":
        """
        Run cross-field validation. Raises ValueError on problems.
        Returns self for chaining.
        """
        ext = self.extractor
        steer = self.steer
        post_process = self.post_process

        # 1. Layers must be non-empty
        if not ext.layer:
            raise ValueError("Extractor layers list is empty")
        if not steer.layer:
            raise ValueError("Steer layers list is empty")

        # 2. train_dataset must be set when load_vector is not provided
        if not self.load_vector and not self.train_dataset:
            if ext.method.upper() not in ("SAS",):
                raise ValueError(
                    "train_dataset required in extractor config when load_vector not provided"
                )

        # 3. coeff keys must match steer layers
        if set(steer.coeff.keys()) != set(steer.layer):
            raise ValueError(
                f"coeff keys {sorted(steer.coeff.keys())} "
                f"do not match steer layers {steer.layer}"
            )

        # 4. CAST-specific: conditional_layer < min(extractor layers)
        cond_layer = ext.conditional_layer
        if ext.method.upper() == "CAST" and cond_layer is not None:
            if cond_layer >= min(ext.layer):
                logger.warning(
                    f"CAST: conditional_layer ({cond_layer}) >= min extractor layer "
                    f"({min(ext.layer)}). This may cause unexpected results."
                )

        # 5. Post-process sanity checks
        if post_process.enabled:
            if not post_process.source:
                raise ValueError("post_process.source is required when post_process.enabled is true")
            if not post_process.layer:
                raise ValueError("post_process.layer is empty")

        return self

    def normalize_vector(self, vec) -> "Dict[int, Any]":
        """
        Normalize a steering vector to Dict[int, Tensor] keyed by steer layers.

        Accepts:
        - Dict[int, Tensor] → returned as-is (keys normalised to int)
        - Single Tensor → broadcast to all steer layers
        """
        if isinstance(vec, dict):
            return {int(k): v for k, v in vec.items()}
        return {l: vec for l in self.steer.layer}

    def make_zero_coeff(self) -> "Dict[int, float]":
        """Create zero-valued coeff dict for baseline evaluation."""
        return {l: 0.0 for l in self.steer.layer}
