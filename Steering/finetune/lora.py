"""LoRA backend adapter for finetuning."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import FinetuneConfig, FinetuneMethodSpec

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None


def _parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


class PEFTLoraBackend:
    """Portable Hugging Face + PEFT LoRA backend."""

    def __init__(self, config: FinetuneConfig, method: FinetuneMethodSpec) -> None:
        self.config = config
        self.method = method

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_base_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=_parse_dtype(self.config.dtype),
        )
        model.config.use_cache = False
        return model

    def prepare_model(self, model):
        if get_peft_model is None or LoraConfig is None:
            raise RuntimeError(
                "PEFT is not installed. Install it or use a different backend."
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM if TaskType is not None else self.method.task_type,
            inference_mode=False,
            r=self.method.r,
            lora_alpha=self.method.lora_alpha,
            lora_dropout=self.method.lora_dropout,
            bias=self.method.bias,
            target_modules=list(self.method.target_modules),
            modules_to_save=self.method.backend_kwargs.get("modules_to_save"),
            use_rslora=self.method.backend_kwargs.get("use_rslora", False),
        )
        return get_peft_model(model, lora_config)

    def save_adapter(self, model, output_dir):
        model.save_pretrained(output_dir)

    def load_adapter(self, model, adapter_dir, is_trainable: bool = False):
        if PeftModel is None:
            raise RuntimeError("PEFT is not installed; cannot load an adapter.")
        return PeftModel.from_pretrained(model, adapter_dir, is_trainable=is_trainable)


class UnslothBackend:
    """Reserved adapter backend for future optional accelerated support."""

    def __init__(self, *_args, **_kwargs) -> None:
        try:
            import unsloth  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Unsloth is not installed. Use the PEFT backend or install Unsloth explicitly."
            ) from exc


def get_backend(name: Optional[str], config: FinetuneConfig, method: FinetuneMethodSpec):
    backend_name = (name or method.backend or "peft").lower()
    if backend_name in {"peft", "lora"}:
        return PEFTLoraBackend(config, method)
    if backend_name == "unsloth":
        return UnslothBackend(config, method)
    raise ValueError(f"Unknown finetune backend '{backend_name}'")
