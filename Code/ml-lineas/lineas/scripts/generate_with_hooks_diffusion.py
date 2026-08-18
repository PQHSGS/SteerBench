# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Loads a model and a dataset and extracts intermediate responses
import json
import logging
import os
import typing as t
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import set_seed

from lineas.datasets import get_dataloader, get_dataset
from lineas.estimators.base_estimator import BaseEstimator
from lineas.models import get_model
from lineas.models.model_with_hooks import get_model_with_hooks
from lineas.utils import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Already run in parallel inside DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "False"


def create_gif(image_paths: list[Path], output_path: Path, duration=500, loop=1):
    """
    Create a GIF from a list of images.

    Args:
        image_paths (list): List of file paths to images.
        output_path (str): Path to save the output GIF.
        duration (int): Duration for each frame in milliseconds.
        loop (int): Number of times the GIF should loop (0 for infinite).
    """
    # Open the images and ensure they're in RGB mode
    images = [Image.open(image).convert("RGB") for image in image_paths]

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
    )


def set_hook_values(hooks, strength: float = 0.0, forward: bool = True):
    """Sets the values for a specific hook.

    This method allows you to modify the behavior of a registered hook by
    setting its parameters.

    Args:
        hook_name: The name of the hook to set values for.
        values: A list of values to be assigned to the hook's parameters.

    Returns:
        None

    Raises:
        ValueError: If the hook name is not found or the number of values does
                    not match the expected number of parameters for the hook.
    """
    for hook in hooks:
        hook.strength = strength
        hook.hook_forward = forward


def generate_with_hooks_diffusion(
    batch,
    batch_idx,
    model_hooks,
    hooks,
    images_per_prompt,
    strength,
    guidance_scale,
    output_path,
    cfg,
) -> dict[str, dict[str, t.Any]]:
    """Generates images with hooks.

    Args:
        batch (Dict): A dictionary containing the batch data.
        batch_idx (int): The index of the current batch.
        model_hooks (ModelHooks): An instance of ModelHooks used for applying
            hooks to the model.
        hooks (list): A list of hooks to apply.
        images_per_prompt (int): Number of images to generate per prompt.
        strength (float): Strength of image generation.
        guidance_scale (float): Guidance scale for diffusion model.
        output_path (Path): Path to the directory where generated images will be saved.
    """
    generator = torch.Generator(device=cfg.device)
    generator.manual_seed(cfg.get("seed", 42))

    module = model_hooks.module
    model_hooks.remove_hooks()
    if strength != 0:
        # Register hooks
        set_hook_values(hooks, strength=abs(strength), forward=(strength > 0))
        model_hooks.register_hooks(hooks)
    prompt = (
        batch["original_prompt"]
        if cfg.task_params.dataset == "coco-captions-styles"
        else batch["prompt"]
    )
    if cfg.prompt_override is not None:
        if isinstance(cfg.prompt_override, str):
            prompt = cfg.prompt_override
        else:
            prompt = list(cfg.prompt_override)
    cfg.model_params.generation_kwargs.guidance_scale = guidance_scale
    decoded = module(
        prompt,
        num_images_per_prompt=images_per_prompt,
        generator=generator,
        **cfg.model_params.generation_kwargs,
    ).images

    if decoded[0].size[0] != cfg.generation_resolution:
        decoded = [
            im.resize(
                (cfg.generation_resolution, cfg.generation_resolution), Image.BICUBIC
            )
            for im in decoded
        ]

    generation_info = {}
    for j, image in enumerate(decoded):
        prompt_id = batch["id"][j]
        prompt_dir = output_path / f"{strength:.03f}_{guidance_scale:.03f}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        image.save(prompt_dir / f"{int(prompt_id)}.png")
        generated = {
            "id": int(prompt_id),
            "original_prompt": batch["original_prompt"][j],
            "conditional_prompt": batch["prompt"][j],
            "strength": strength,
            "guidance_scale": guidance_scale,
            "dataset": cfg.task_params.dataset,
            "intervention": str(cfg.intervention_params.state_path),
            "src_subsets": list(
                cfg.task_params.src_subsets
            ),  # Intervention's src subsets
            "dst_subsets": list(
                cfg.task_params.dst_subsets
            ),  # Intervention's dst subsets
            "prompt_subset": list(
                cfg.task_params.prompt_subset
            ),  # Whether to use the prompt before or after appending a concept
            "image_path": str(prompt_dir / f"{int(prompt_id)}.png"),
        }
        generation_info[prompt_id] = generated
        with open(prompt_dir / f"{int(prompt_id)}.json", "w") as f:
            json.dump(generated, f)

    return generation_info


def generate(cfg: DictConfig) -> None:
    """Generates images from text prompts using a diffusion model.

    This function loads a pre-trained diffusion model, prepares the dataset,
    and generates images based on the provided text prompts. It also applies
    optional interventions (hooks) to modify the model's behavior during
    generation.

    Args:
        cfg: A DictConfig object containing configuration parameters for image generation.

    Returns:
        None
    """
    logger.info("Generating images...")
    module_names = "".join(cfg.model_params.module_names).replace("*", "")
    if len(module_names) > 16:
        module_names = (
            module_names[:8] + ".." + module_names[-8:]
        )  # summarize to fit in path
    if cfg.results_dir is not None:
        output_path = Path(Path(__file__).stem)
        output_path = (
            Path(cfg.results_dir)
            / output_path
            / cfg.task_params.dataset
            / f"{'-'.join(cfg.task_params.src_subsets)}:{'-'.join(cfg.task_params.dst_subsets)}"
            / Path(cfg.intervention_params.state_path).name
            / module_names
        )
    else:
        output_path = None
    module, tokenizer = get_model(
        cache_dir=cfg.cache_dir,
        device=cfg.device,
        model_task="text_to_image_generation",
        **cfg.model_params,
    )

    # Datasets
    # When trying to remove concepts, we prompt with no_concept
    if cfg.task_params.dataset == "coco-captions-concepts":
        subsets = [f"no_{subset}" for subset in cfg.task_params.prompt_subset]
    elif cfg.task_params.dataset == "coco-captions-styles":
        subsets = list(cfg.task_params.dst_subsets)
    else:
        subsets = list(cfg.task_params.prompt_subset)

    dataset, collate_fn = get_dataset(
        name=cfg.task_params.dataset,
        datasets_folder=Path(cfg.data_dir),
        split="val",
        subsets=subsets,
        tokenizer=None,
        **cfg.task_params.get("dataset_params", {}),
    )
    # Sampling and dataloader
    loader = get_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=True,
        balanced=False,
        seed=cfg.seed,
    )
    logger.info(module)
    model_hooks = get_model_with_hooks(module, model_task="text_to_image_generation")

    # Load hooks
    hooks = model_hooks.load_hooks_from_folder(
        folder=Path(cfg.intervention_params.state_path),
        module_names=cfg.model_params.module_names,
        hook_type=cfg.intervention_params.name,
        **cfg.intervention_params.hook_params,
    )

    # Generate without hooks
    set_seed(42)
    images_per_prompt = 1
    max_batches = (
        len(loader) if cfg.max_batches is None else min(len(loader), cfg.max_batches)
    )
    iloader = iter(loader)
    for batch_idx in tqdm(range(max_batches), desc="Generating images"):
        batch = next(iloader)
        generation_info_lists = defaultdict(list)
        for strength in np.linspace(
            cfg.min_strength, cfg.max_strength, cfg.strength_steps
        ):
            logger.info(f"Generating with strength {strength}")
            generate_info = generate_with_hooks_diffusion(
                batch=batch,
                batch_idx=batch_idx,
                model_hooks=model_hooks,
                hooks=hooks,
                images_per_prompt=images_per_prompt,
                strength=float(strength),
                guidance_scale=cfg.guidance_scale,
                output_path=output_path,
                cfg=cfg,
            )
            for prompt_id, info in generate_info.items():
                generation_info_lists[f"{prompt_id}"].append(info)
        logger.info(f"Generations saved in {output_path}")
        if cfg.create_gif:
            for prompt_id, info_list in generation_info_lists.items():
                image_paths = [info["image_path"] for info in info_list]
                create_gif(
                    image_paths=image_paths,
                    output_path=output_path / f"{prompt_id}_animation.gif",
                )

        for guidance_scale in cfg.diffusion_guidance_scale:
            logger.info(f"Generating with guidance scale {guidance_scale}")
            generate_with_hooks_diffusion(
                batch=batch,
                batch_idx=batch_idx,
                model_hooks=model_hooks,
                hooks=hooks,
                images_per_prompt=images_per_prompt,
                strength=0,
                guidance_scale=guidance_scale,
                output_path=output_path,
                cfg=cfg,
            )
        # WARNING: not safe in concurrent environments
        # if metadata_path.exists():
        #     with metadata_path.open("r") as fp:
        #         metadata = json.load(fp) + metadata
        # with metadata_path.open("w") as fp:
        #     json.dump(metadata, fp)
    if cfg.wandb.mode != "disabled":
        utils.log_image_folder_wandb(output_path, limit=100)


@hydra.main(
    config_path="../configs", config_name="text_to_image_generation", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    intervention_state_path = BaseEstimator.get_output_path(cfg.interventions)
    cfg.text_to_image_generation.intervention_params.state_path = (
        intervention_state_path
    )
    cfg = utils.copy_hydra_config(cfg)
    try:
        cfg.model.unet_with_grads = False
    except Exception as e:
        logger.warning(str(e))
    generate(cfg.text_to_image_generation)


if __name__ == "__main__":
    main()
