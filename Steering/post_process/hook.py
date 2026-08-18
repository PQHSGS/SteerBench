"""Hook-based GLP post-processing for residual stream denoising."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, List, Optional

import torch

from ..utils import get_hook_name, get_resid_acts, set_resid_acts
from . import flow_matching
from .classifier import ConceptClassifier, load_classifier
from .glp import GLP, load_glp

if TYPE_CHECKING:
    from ..config.post_process import PostProcessConfig


def _normalize_to_list(raw) -> List:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


class GLPPostProcessor:
    """Applies GLP denoising on selected residual activations via hooks."""

    def __init__(
        self,
        *,
        enabled: bool,
        source: Optional[str],
        checkpoint: str,
        noise_rate: float,
        num_timesteps: int,
        position: str | int,
        layer: Optional[List[int]] = None,
        hook_point: Optional[List[str]] = None,
        device: str = "cuda",
        apply_on_baseline: bool = False,
        use_classifier: bool = False,
        scale: float = 1.0,
        negative: bool = False,
        classifier_checkpoint: str = "final",
        classifier_source: Optional[str] = None,
        classifier_guidance_start_step: int = 0,
        classifier_guidance_end_step: Optional[int] = None,
        classifier_grad_clip: Optional[float] = 5.0,
        classifier_normalize_grad: bool = True,
    ):
        self.enabled = bool(enabled)
        self.source = source
        self.checkpoint = checkpoint
        self.noise_rate = float(noise_rate)
        self.num_timesteps = int(num_timesteps)
        self.position = position
        self.layer = _normalize_to_list(layer)
        self.hook_point = _normalize_to_list(hook_point) or ["pre"]
        self.device = device
        self.apply_on_baseline = bool(apply_on_baseline)
        self.use_classifier = bool(use_classifier)
        self.scale = float(scale)
        self.negative = bool(negative)
        self.classifier_checkpoint = classifier_checkpoint
        self.classifier_source = classifier_source
        self.classifier_guidance_start_step = int(classifier_guidance_start_step)
        self.classifier_guidance_end_step = (
            int(classifier_guidance_end_step)
            if classifier_guidance_end_step is not None
            else None
        )
        self.classifier_grad_clip = (
            float(classifier_grad_clip)
            if classifier_grad_clip is not None
            else None
        )
        self.classifier_normalize_grad = bool(classifier_normalize_grad)

        self.model: Optional[GLP] = None
        self.classifier: Optional[ConceptClassifier] = None

    @classmethod
    def from_config(cls, cfg: "PostProcessConfig", device: str) -> "GLPPostProcessor":
        """Construct a post-processor directly from resolved PostProcessConfig."""
        return cls(
            enabled=cfg.enabled,
            source=cfg.source,
            checkpoint=cfg.checkpoint,
            noise_rate=cfg.noise_rate,
            num_timesteps=cfg.num_timesteps,
            position=cfg.position,
            layer=cfg.layer,
            hook_point=cfg.hook_point,
            device=device,
            apply_on_baseline=cfg.apply_on_baseline,
            use_classifier=cfg.use_classifier,
            scale=cfg.scale,
            negative=cfg.negative,
            classifier_checkpoint=cfg.classifier_checkpoint,
            classifier_source=cfg.classifier_source,
            classifier_guidance_start_step=cfg.classifier_guidance_start_step,
            classifier_guidance_end_step=cfg.classifier_guidance_end_step,
            classifier_grad_clip=cfg.classifier_grad_clip,
            classifier_normalize_grad=cfg.classifier_normalize_grad,
        )

    def _classifier_enabled(self) -> bool:
        return self.use_classifier

    @staticmethod
    def _as_latents(acts: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Convert activations to [batch, seq, dim], returning (latents, use_seq)."""
        if acts.ndim == 2:
            return acts[:, None, :], False
        if acts.ndim == 3:
            return acts, True
        raise ValueError(f"Unsupported activation shape for GLP post-process: {acts.shape}")

    def load(self) -> None:
        if not self.enabled:
            return
        if self.model is not None:
            return
        if not self.source:
            raise ValueError("Post-process is enabled but no GLP source is configured")
        self.model = load_glp(self.source, device=str(self.device), checkpoint=self.checkpoint)
        if self._classifier_enabled():
            classifier_source = self.classifier_source or self.source
            if not classifier_source:
                raise ValueError(
                    "Classifier guidance is enabled but no classifier source is configured"
                )
            self.classifier = load_classifier(
                classifier_source,
                device=str(self.device),
                checkpoint=self.classifier_checkpoint,
            )

    def _denoise(self, acts: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("GLP model is not loaded")
        glp = self.model

        latents, use_seq = self._as_latents(acts)

        latents = latents.to(device=glp.denoiser.device, dtype=torch.float32)
        latents = glp.normalizer.normalize(latents, layer_idx=layer_idx)

        noise = torch.randn_like(latents)
        u = torch.full((latents.shape[0],), self.noise_rate, device=latents.device)
        glp.scheduler.set_timesteps(glp.scheduler.config.num_train_timesteps)

        noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
            glp.scheduler,
            latents,
            noise,
            u=u,
        )
        denoised = flow_matching.sample_on_manifold(
            glp,
            noisy_latents,
            start_timestep=timesteps,
            num_timesteps=self.num_timesteps,
            show_progress=False,
            classifier=self.classifier if self._classifier_enabled() else None,
            classifier_scale=self.scale,
            classifier_negative=self.negative,
            classifier_guidance_start_step=self.classifier_guidance_start_step,
            classifier_guidance_end_step=self.classifier_guidance_end_step,
            classifier_grad_clip=self.classifier_grad_clip,
            classifier_normalize_grad=self.classifier_normalize_grad,
            layer_idx=layer_idx,
        )

        denoised = glp.normalizer.denormalize(denoised, layer_idx=layer_idx)
        denoised = denoised.to(device=acts.device, dtype=acts.dtype)

        if use_seq:
            return denoised
        return denoised[:, 0, :]

    def _hook_fn(self, resid: torch.Tensor, hook, layer_idx: int) -> torch.Tensor:
        del hook
        acts = get_resid_acts(resid, self.position)
        denoised = self._denoise(acts, layer_idx=layer_idx)
        return set_resid_acts(resid, self.position, denoised)

    def setup_hooks(
        self,
        model,
    ) -> List:
        """Install post-process hooks and return handles."""
        if not self.enabled:
            return []

        if not self.layer:
            raise ValueError("post_process.layer must be configured when post_process.enabled is true")

        self.load()

        return [
            model.add_hook(
                get_hook_name(lay, hp),
                partial(self._hook_fn, layer_idx=lay),
            )
            for lay in self.layer
            for hp in self.hook_point
        ]
