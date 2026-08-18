# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import os
import typing as t
from functools import partial
from pathlib import Path

import torch
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
    FluxPipeline,
    LCMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from lineas.utils.utils import get_device


class BPUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        *unet_args,
        **unet_kwargs,
    ) -> UNet2DConditionOutput | tuple:
        # Force gradients, overrides torch.no_grad() wrapper in StableDiffusionPipeline.__call__()
        with torch.set_grad_enabled(True):
            return super().forward(*unet_args, **unet_kwargs)


EXTRA_KWARGS = {
    "google/gemma-2-2b": {"attn_implementation": "eager"},
    "google/gemma-2-9b": {"attn_implementation": "eager"},
    "apple/OpenELM-270M-Instruct": {"trust_remote_code": True},
}


def get_model(
    model_path: str | Path,
    cache_dir: str | Path | None,
    dtype: t.Any = None,
    device: t.Any = None,
    model_task: str = None,
    **kwargs,
):
    """Loads a pre-trained model based on the specified task.

    This function acts as a dispatcher, selecting the appropriate
    model loading function from `TASK_MAPPING` based on the provided `model_task`.

    Args:
        model_path (Union[str, Path]): The path to the pre-trained model weights.
        cache_dir (Optional[Union[str, Path]], optional): Directory to cache downloaded models. Defaults to None.
        dtype (Any, optional): Data type for the model weights. Defaults to None (inferred from the model).
        device (Any, optional): Device to load the model on (e.g., 'cpu', 'cuda'). Defaults to None (uses default device).
        model_task (str): The task the model is intended for (e.g., "text-classification", "image-generation").
        **kwargs: Additional keyword arguments passed to the specific model loading function.

    Returns:
        Any: The loaded pre-trained model.

    Raises:
        AssertionError: If `model_task` is not provided.
    """
    assert model_task is not None
    return TASK_MAPPING[model_task](
        model_path=model_path,
        model_task=model_task,
        cache_dir=cache_dir,
        dtype=dtype,
        device=device,
        **kwargs,
    )


def get_text_to_image_model(
    model_path: str | Path,
    cache_dir: str | Path | None,
    inference_steps: int = 4,
    dtype: t.Any = None,
    device: str = None,
    unet_with_grads: bool = False,
    **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a text-to-image model based on the specified path.

    Supports models from HuggingFace Hub and specific implementations like SDXL
    and FLUX.

    Args:
        model_path (Union[str, Path]): The path to the pre-trained model weights.
        cache_dir (Optional[Union[str, Path]], optional): Directory to cache downloaded models. Defaults to None.
        inference_steps (int, optional): Number of diffusion steps for text-to-image generation. Defaults to 4.
        dtype (Any, optional): Data type for the model weights. Defaults to None (inferred from the model).
        device (str, optional): Device to load the model on (e.g., 'cpu', 'cuda'). Defaults to None (uses default device).

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The loaded text-to-image model and its tokenizer (if applicable).

    """
    if model_path == "ByteDance/SDXL-Lightning":
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        if inference_steps == 1:
            ckpt = "sdxl_lightning_1step_unet_x0.safetensors"  # Use the correct ckpt for your step setting!
            prediction_type = "sample"
        else:
            ckpt = f"sdxl_lightning_{inference_steps}step_unet.safetensors"  # Use the correct ckpt for your step setting!
            prediction_type = "epsilon"
        cache_dir = os.environ.get("HF_HUB_CACHE", cache_dir)
        # Load model.
        if unet_with_grads:
            unet = BPUNet2DConditionModel.from_config(
                base, subfolder="unet", cache_dir=cache_dir
            ).to(device, torch.float16)
            unet.requires_grad_(True)
        else:
            unet = UNet2DConditionModel.from_config(
                base, subfolder="unet", cache_dir=cache_dir
            ).to(device, torch.float16)

        unet.load_state_dict(
            load_file(
                hf_hub_download(model_path, ckpt, cache_dir=cache_dir), device=device
            )
        )

        if unet_with_grads:
            unet.requires_grad_(True)

        pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir,
        ).to(device)
        pipe.vae.requires_grad_(False)

        # Ensure sampler uses "trailing" timesteps.
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            cache_dir=cache_dir,
            prediction_type=prediction_type,
        )

        # # Ensure using the same inference steps as the loaded model and CFG set to 0.
        # pipe("A girl smiling", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")

        return pipe, None
    elif model_path == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=cache_dir,
        ).to(device)

        return pipe, None
    elif model_path == "black-forest-labs/FLUX.1-schnell":
        bfl_repo = "black-forest-labs/FLUX.1-schnell"
        # text_encoder_2 = T5EncoderModel.from_pretrained(
        #     bfl_repo,
        #     subfolder="text_encoder_2",
        #     torch_dtype=torch.bfloat16,
        #     cache_dir=cache_dir,
        # )

        if unet_with_grads:

            class BPFluxTransformer2DModel(FluxTransformer2DModel):
                def forward(
                    self,
                    *unet_args,
                    **unet_kwargs,
                ) -> FluxTransformer2DModel | tuple:
                    # Force gradients, overrides torch.no_grad() wrapper in StableDiffusionPipeline.__call__()
                    with torch.enable_grad():
                        return super().forward(*unet_args, **unet_kwargs)

            transformer = BPFluxTransformer2DModel.from_config(
                bfl_repo,
                subfolder="transformer",
                cache_dir=cache_dir,
            ).to(device, torch.bfloat16)
        else:
            transformer = FluxTransformer2DModel.from_config(
                bfl_repo,
                subfolder="transformer",
                cache_dir=cache_dir,
            ).to(device, torch.bfloat16)

        if unet_with_grads:
            transformer.requires_grad_(True)

        pipe = FluxPipeline.from_pretrained(
            bfl_repo,
            torch_dtype=torch.bfloat16,
            # text_encoder_2=text_encoder_2,
            transformer=transformer,
            cache_dir=cache_dir,
            force_download=False,
        ).to(device)

        return pipe, None
    elif model_path == "tianweiy/DMD2":
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"
        # Load model.
        cache_dir = os.environ.get("HF_HUB_CACHE", cache_dir)
        if unet_with_grads:
            unet = BPUNet2DConditionModel.from_config(
                base_model_id, subfolder="unet", cache_dir=cache_dir
            ).to(device, dtype)
        else:
            unet = UNet2DConditionModel.from_config(
                base_model_id, subfolder="unet", cache_dir=cache_dir
            ).to(device, dtype)
        unet.load_state_dict(
            torch.load(
                hf_hub_download(repo_name, ckpt_name, cache_dir=cache_dir),
                map_location=device,
            )
        )
        # unet.requires_grad_(True)
        pipe = DiffusionPipeline.from_pretrained(
            base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(device, dtype)
        # if unet_with_grads:
        #     pipe.__call__ = pipe.__call__.__wrapped__ # removes no_grad
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        return pipe, None
    else:
        return (
            DiffusionPipeline.from_pretrained(model_path, cache_dir=cache_dir).to(
                device, dtype
            ),
            None,
        )


def get_huggingface_model(
    model_path: str | Path,
    model_task: str = "text_generation",
    dtype: t.Any = None,
    device: str = None,
    seq_len: int | None = None,
    rand_weights: bool = False,
    model_class: str | None = AutoModel,
    padding_side: str = "left",
    use_lora: bool = False,
    lora_state_path: Path = None,
    lora_params: DictConfig = None,
    **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a Huggingface model and tokenizer.

    This function downloads and loads a pretrained model from the Huggingface Model Hub.
    It supports loading various types of models (e.g., text generation, sequence classification)
    and allows for customization of device, data type, sequence length, and padding side.

    Args:
        model_path (str or Path): The path to the model on the Huggingface Model Hub.
        cache_dir (str or Path, optional): The directory where cached models are stored.
                                          Defaults to using the default Huggingface cache if not specified.
        dtype (Any, optional): The desired data type for the model tensors. Defaults to the default dtype of the current device.
        device (str, optional): The device to load the model on ("cuda" or "cpu"). Defaults to "cuda:0" if available, otherwise "cpu".
        seq_len (int, optional): The maximum sequence length for the tokenizer. Defaults to None.
        rand_weights (bool, optional): If True, initializes a new model with random weights instead of loading pretrained weights.
                                     Defaults to False.
        model_class (str, optional): The class of the Huggingface model to load. Defaults to AutoModel.
        padding_side (str, optional): Specifies which side to pad tokens on ("left" or "right"). Defaults to "left".
        **kwargs: Additional keyword arguments passed to the model loading function.

    Returns:
        tuple: A tuple containing the loaded Huggingface model and tokenizer.

    """

    # If HF_HUB_CACHE is set, use it. Otherwise, use the default Huggingface one.
    cache_dir = os.environ.get("HF_HUB_CACHE", None)
    # Also fetching your huggingface token for specific model downloads.
    hf_token = os.environ.get("HF_TOKEN", None)

    # Defaults for device and dtype
    if device is None:
        device = get_device()

    if dtype is None:
        dtype = torch.get_default_dtype()
    else:
        dtype = dtype

    if cache_dir is not None:
        cache_dir = Path(cache_dir).absolute()
        full_model_path = cache_dir / model_path
        if full_model_path.exists():
            model_path = full_model_path

    if rand_weights:
        config = AutoConfig.from_pretrained(
            model_path, cache_dir=cache_dir, device_map=device
        )
        model = model_class.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        extra_kwargs = EXTRA_KWARGS.get(model_path, {})
        tokenizer_path = (
            "meta-llama/Llama-2-7b-hf" if "OpenELM" in model_path else model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            device_map=device,
            dtype=dtype,
            force_download=False,
            token=hf_token,
            **extra_kwargs,
        )
        model = model_class.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            device_map=device,
            force_download=False,
            torch_dtype=dtype,
            token=hf_token,
            **extra_kwargs,
        )

    if model_task == "text_generation":
        if seq_len:
            tokenizer.model_max_length = seq_len
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            # If bos exists, it only makes sense to use it for left padding
            if tokenizer.bos_token is not None and padding_side == "left":
                tokenizer.pad_token = tokenizer.bos_token
            # If eos exists, it only makes sense to use it for right padding
            elif tokenizer.eos_token is not None and padding_side == "right":
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                model.resize_token_embeddings(len(tokenizer))
                tokenizer.pad_token = "<pad>"
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if use_lora:
        lora_config = LoraConfig(**lora_params)
        model = get_peft_model(model, lora_config)
        if lora_state_path is not None:
            # TODO: add support for loading multiple adapters
            model.load_adapter(lora_state_path, "default")
            model = model.merge_and_unload()

    return model, tokenizer


TASK_MAPPING = {
    "text_generation": partial(get_huggingface_model, model_class=AutoModelForCausalLM),
    "text-classification": partial(
        get_huggingface_model, model_class=AutoModelForSequenceClassification
    ),
    "sequence-classification": partial(
        get_huggingface_model, model_class=AutoModelForSequenceClassification
    ),
    "question-answering": partial(
        get_huggingface_model, model_class=AutoModelForQuestionAnswering
    ),
    "text_to_image_generation": get_text_to_image_model,
}
