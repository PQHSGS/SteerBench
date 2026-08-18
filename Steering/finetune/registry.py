"""Registry for finetune method recipes."""

from __future__ import annotations

from typing import Dict

from .config import FinetuneMethodSpec


METHOD_REGISTRY: Dict[str, FinetuneMethodSpec] = {}


def register_method(spec: FinetuneMethodSpec) -> FinetuneMethodSpec:
    METHOD_REGISTRY[spec.name.lower()] = spec
    return spec


def get_method_spec(name: str) -> FinetuneMethodSpec:
    key = name.lower()
    if key not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown finetune method '{name}'. Available: {available}")
    return METHOD_REGISTRY[key]


def list_methods() -> list[str]:
    return sorted(METHOD_REGISTRY.keys())


register_method(
    FinetuneMethodSpec(
        name="lora",
        backend="peft",
    )
)

register_method(
    FinetuneMethodSpec(
        name="simple-lora",
        backend="peft",
    )
)