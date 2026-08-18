"""Configuration objects for adapter finetuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import json
import re


def _normalize_sequence(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(value) for value in values)


@dataclass
class FinetuneMethodSpec:
    """Method-specific adapter recipe."""

    name: str = "lora"
    backend: str = "peft"
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    train_field: str = "target_response"
    prompt_field: str = "question"
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinetuneMethodSpec":
        payload = dict(data)
        if "target_modules" in payload and payload["target_modules"] is not None:
            payload["target_modules"] = _normalize_sequence(payload["target_modules"])
        elif "target_modules" in payload:
            payload.pop("target_modules")
        return cls(**payload)


@dataclass
class FinetuneConfig:
    """Top-level finetune run configuration."""

    name: str = "finetune"
    description: str = ""

    model_name: str = "google/gemma-2-2b-it"
    train_dataset: str = "refusal_cast_responses"
    n_train: Optional[int] = None
    test_dataset: Optional[str] = "refusal_open"
    n_test: Optional[int] = None
    apply_chat_template: bool = True
    inverse: bool = False

    method_name: str = "lora"
    backend: Optional[str] = None

    output: str = "./Results/finetune"

    save_vector: Optional[str] = None
    load_vector: Optional[str] = None

    compute_perplexity: bool = True
    eval_max_new_tokens: int = 128

    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    num_epochs: int = 1
    train_batch_size: int = 2
    eval_batch_size: int = 2
    grad_accum: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_length: int = 2048
    grad_clip: float = 1.0

    method: FinetuneMethodSpec = field(default_factory=FinetuneMethodSpec)

    def to_dict(self, include_none: bool = False) -> Dict[str, Any]:
        payload = asdict(self)
        payload["method"] = self.method.to_dict()
        if not include_none:
            payload = {key: value for key, value in payload.items() if value is not None}
        return payload

    def save(self, path: str | Path, include_none: bool = False) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(include_none=include_none), handle, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinetuneConfig":
        payload = dict(data)
        if "train_dataset" not in payload and "dataset_name" in payload:
            payload["train_dataset"] = payload.pop("dataset_name")
        if "output" not in payload and "output_root" in payload:
            payload["output"] = payload.pop("output_root")
        
        method_name = payload.get("method_name", "lora")
        from .registry import get_method_spec
        try:
            default_spec = get_method_spec(method_name)
            default_dict = default_spec.to_dict()
        except KeyError:
            default_dict = {}

        method_data = payload.get("method", {})
        if isinstance(method_data, FinetuneMethodSpec):
            method = method_data
        else:
            merged_method_data = dict(default_dict)
            if method_data:
                merged_method_data.update(method_data)
            method = FinetuneMethodSpec.from_dict(merged_method_data)
        payload["method"] = method
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path) -> "FinetuneConfig":
        with open(path) as handle:
            return cls.from_dict(json.load(handle))


def slugify(text: str) -> str:
    """Convert model/dataset names into filesystem-safe directory names."""

    value = text.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._-") or "run"