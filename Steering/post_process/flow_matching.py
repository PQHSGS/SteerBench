"""Flow-matching utilities for GLP denoising."""

from __future__ import annotations

from typing import Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


def canonicalize_solver(solver: Optional[str]) -> str:
    if solver is None:
        return "euler"
    solver = solver.lower()
    if solver not in ("euler", "heun"):
        raise ValueError(f"Unsupported solver {solver}")
    return solver

def build_inference_scheduler(solver: str, num_train_timesteps: int = 1000):
    if solver == "euler":
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
    elif solver == "heun":
        from diffusers import FlowMatchHeunDiscreteScheduler
        return FlowMatchHeunDiscreteScheduler(num_train_timesteps=num_train_timesteps)
    raise ValueError(f"Unsupported solver {solver}")

def fm_scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler()


def fm_prepare(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_input: torch.Tensor,
    noise: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """Prepare noisy latents and velocity targets for flow matching."""
    if model_input.ndim != 3:
        raise ValueError(f"Expected model_input with shape [batch, seq, dim], got {model_input.shape}")

    if u is None:
        u = torch.rand(size=(model_input.shape[0],), generator=generator, device=model_input.device)
    else:
        u = u.to(model_input.device)

    indices = (u * len(scheduler.timesteps)).long().clamp(min=0, max=len(scheduler.timesteps) - 1)
    timesteps_tensor = scheduler.timesteps
    sigmas_tensor = scheduler.sigmas

    timestep_index_device = timesteps_tensor.device if torch.is_tensor(timesteps_tensor) else torch.device("cpu")
    sigma_index_device = sigmas_tensor.device if torch.is_tensor(sigmas_tensor) else torch.device("cpu")

    timestep_indices = indices.to(timestep_index_device)
    sigma_indices = indices.to(sigma_index_device)

    timesteps = timesteps_tensor[timestep_indices].to(model_input.device)
    sigmas = sigmas_tensor[sigma_indices].to(model_input.device)

    timesteps_expanded = timesteps[:, None, None]
    sigmas_expanded = sigmas[:, None, None]

    noisy_model_input = (1.0 - sigmas_expanded) * model_input.to(sigmas_expanded.dtype) + sigmas_expanded * noise
    noisy_model_input = noisy_model_input.to(model_input.dtype)
    target = noise - model_input

    return noisy_model_input, target, timesteps_expanded, {
        "sigmas": sigmas_expanded,
        "u": u,
    }


def _apply_start_timestep_mask(
    latents: torch.Tensor,
    start_latents: torch.Tensor,
    start_timestep: Optional[torch.Tensor | float | int],
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Apply SDEdit-style start timestep logic.

    Returns:
        (latents, should_skip_step)
    """
    if start_timestep is None:
        return latents, False

    if torch.is_tensor(start_timestep):
        start_t = start_timestep[:, 0, 0] if start_timestep.ndim == 3 else start_timestep
        timestep_mask = start_t <= timestep
        latents[timestep_mask] = start_latents[timestep_mask]
        return latents, False

    if float(timestep) > float(start_timestep):
        return latents, True

    return latents, False


def _classifier_step_active(
    *,
    classifier,
    classifier_scale: float,
    step_idx: int,
    classifier_guidance_start_step: int,
    classifier_guidance_end_step: Optional[int],
) -> bool:
    if classifier is None:
        return False
    if float(classifier_scale) <= 0.0:
        return False
    if step_idx < int(classifier_guidance_start_step):
        return False
    if classifier_guidance_end_step is not None and step_idx > int(classifier_guidance_end_step):
        return False
    return True


def _classifier_guidance_grad(
    *,
    classifier,
    scheduler,
    step_idx: int,
    latents: torch.Tensor,
    classifier_negative: bool,
    classifier_grad_clip: Optional[float],
    classifier_normalize_grad: bool,
) -> torch.Tensor:
    # HookedTransformer.generate runs under inference mode, which suppresses autograd.
    # Classifier guidance needs d/dz log p(y|z_t, t), so explicitly disable inference mode
    # and re-enable grads for this local block.
    with torch.inference_mode(False):
        with torch.enable_grad():
            # `latents` may be an inference tensor coming from model.generate.
            # Clone here to materialize a regular tensor that can track gradients.
            latents_for_grad = latents.detach().clone().requires_grad_(True)
            sigma_val = scheduler.sigmas[step_idx].to(device=latents_for_grad.device, dtype=latents_for_grad.dtype)
            t_batch = torch.full(
                (latents_for_grad.shape[0],),
                float(sigma_val.item()),
                device=latents_for_grad.device,
                dtype=latents_for_grad.dtype,
            ).clamp(min=0.0, max=1.0)

            log_prob = classifier.log_prob(latents_for_grad, t_batch, negative=classifier_negative)
            guidance_obj = log_prob.mean()
            cond_grad = torch.autograd.grad(
                guidance_obj,
                latents_for_grad,
                retain_graph=False,
                create_graph=False,
            )[0].detach()

    if classifier_normalize_grad:
        flat = cond_grad.reshape(cond_grad.shape[0], -1)
        denom = flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cond_grad = cond_grad / denom.view(cond_grad.shape[0], *([1] * (cond_grad.ndim - 1)))

    if classifier_grad_clip is not None:
        cond_grad = cond_grad.clamp(min=-float(classifier_grad_clip), max=float(classifier_grad_clip))

    return cond_grad


def sample_on_manifold(
    model,
    latents: torch.Tensor,
    num_timesteps: int = 20,
    start_timestep: Optional[torch.Tensor | float | int] = None,
    show_progress: bool = False,
    classifier=None,
    classifier_scale: float = 1.0,
    classifier_negative: bool = False,
    classifier_guidance_start_step: int = 0,
    classifier_guidance_end_step: Optional[int] = None,
    classifier_grad_clip: Optional[float] = 5.0,
    classifier_normalize_grad: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Denoise latents with SDEdit-style start timestep control."""
    del show_progress  # Keep API parity without extra dependency.

    start_latents = latents.clone()
    model.scheduler.set_timesteps(num_timesteps)

    for step_idx, timestep in enumerate(model.scheduler.timesteps):
        latents, should_skip_step = _apply_start_timestep_mask(
            latents,
            start_latents,
            start_timestep,
            timestep,
        )
        if should_skip_step:
            continue

        step_timesteps = timestep[None, ...]
        cond_grad = None
        if _classifier_step_active(
            classifier=classifier,
            classifier_scale=classifier_scale,
            step_idx=step_idx,
            classifier_guidance_start_step=classifier_guidance_start_step,
            classifier_guidance_end_step=classifier_guidance_end_step,
        ):
            cond_grad = _classifier_guidance_grad(
                classifier=classifier,
                scheduler=model.scheduler,
                step_idx=step_idx,
                latents=latents,
                classifier_negative=classifier_negative,
                classifier_grad_clip=classifier_grad_clip,
                classifier_normalize_grad=classifier_normalize_grad,
            )

        with torch.no_grad():
            noise_pred = model.denoiser(
                latents=latents,
                timesteps=step_timesteps.repeat(latents.shape[0], 1, 1),
                **kwargs,
            )

        latents = model.scheduler.step(noise_pred, step_timesteps, latents, return_dict=False)[0]

        # Classifier guidance follows z_{t-1} = z_t + Δt * u_theta + s * ∇_z log p_phi(y|z_t,t)
        if cond_grad is not None:
            latents = latents + float(classifier_scale) * cond_grad.to(device=latents.device, dtype=latents.dtype)

    return latents
