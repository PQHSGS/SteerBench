"""
Abstract Base Classes for Steering Vector Extraction and Model Steering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Tuple, TYPE_CHECKING
from functools import partial

import torch
import torch.nn.functional as F
from .utils import get_hook_name, get_resid_acts, set_resid_acts
from .logger import setup_logger

if TYPE_CHECKING:
    from .config import SteeringVector

logger = setup_logger(__name__)
class BaseExtractor(ABC):
    """
    Abstract base class for steering vector extractors.
    
    All extractors follow a common pattern:
    1. Collect activations from target and contrast data
    2. Compute a steering direction (dense or sparse)
    
    Attributes:
        model: The language model (HookedTransformer or HookedSAETransformer)
        layer: Target layer(s) for extraction
        batch_size: Batch size for processing
        device: Computation device
        hook_point: Hook location ("pre" or "post")
    """
    
    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        hook_point: List[str] = ["pre"],
        position: Union[str, int] = "last",
        change_pad_token: bool = False,
        **kwargs,
    ):
        self.model = model
        self.layer = layer
        self.batch_size = batch_size
        self.device = device or getattr(model, 'cfg', None) and getattr(model.cfg, 'device', 'cuda') or 'cuda'
        self.hook_point = hook_point
        self.position = position
        self.change_pad_token = change_pad_token
        
        # Store extracted vectors
        self.vector = None
        self.metadata: Dict[str, Any] = {}
    
    # Method name for SteeringVector
    METHOD_NAME: str = "BASE"
    
    @abstractmethod
    def _get_activations(self, inputs: List[str], **kwargs) -> torch.Tensor:
        """
        Collect activations from input prompts.
        
        Args:
            inputs: List of text prompts
            
        Returns:
            Aggregated activation tensor
        """
        pass
    
    @abstractmethod
    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Extract steering vector from target and contrast data.
        
        Args:
            target_data: Prompts representing desired behavior
            contrast_data: Prompts representing opposite behavior (optional)
            
        Returns:
            Steering vector (dense or sparse depending on method)
        """
        pass
    
    def get_hook_name(self, layer: Optional[List[int]] = None, position: Optional[List[str]] = None) -> List[str]:
        """
        Get the hook name for the target layer.
        
        Args:
            layer: Override layer. If None, uses self.layer. Must be List[int].
            position: Override position ("pre", "post", "mid"). If None, uses self.hook_point.
        """
        pos = position or self.hook_point
        if isinstance(pos, str):
            pos = [pos]
        lay = layer if layer is not None else self.layer
        
        hook_names = [get_hook_name(l, p) for l in lay for p in pos]
        return hook_names


    def get_steering_vector(self) -> "SteeringVector":
        """
        Get the extracted steering vector as a SteeringVector object.
        
        Override in subclasses to include method-specific data in metadata.
        
        Returns:
            SteeringVector containing vector and method-specific metadata
        """
        from .config import SteeringVector
        return SteeringVector(
            vector=self.vector,
            metadata=self.metadata.copy() if self.metadata else {},
        )

    def get_steer_params(self) -> Dict[str, Any]:
        """
        Get parameters to pass to the corresponding steer model.
        
        Override in subclasses to provide method-specific params.
        Default returns the steering vector.
        
        Returns:
            Dictionary of parameters for steer model constructor.
        """
        return self.get_steering_vector().to_steer_params()

    def save(self, path: Union[str, Any]):
        """
        Save the extracted vector to disk using SteeringVector wrapper.
        
        Args:
            path: Path to save the vector to.
        """
        if self.vector is None:
            raise ValueError("No vector to save. Run extract() first.")
            
        self.get_steering_vector().save(path)



class BaseSteerModel(ABC):
    """
    Abstract base class for steered model generation.
    
    All wrappers apply steering vectors during inference via hooks.
    
    Attributes:
        model: The language model
        layer: Layer at which to apply steering
        steering_vector: The extracted steering vector
        hook_point: Hook point ("pre" or "post")
        norm: Whether to L2-normalize the steering vector before applying
    """
    
    def __init__(
        self,
        model,
        layer: List[int],
        steering_vector: Dict[int, torch.Tensor],
        hook_point: List[str] = ["pre"],
        position: Union[int,str] = "last",
        norm: bool = False,
        post_processor: Optional[Any] = None,
        steer_once: bool = False,
        **kwargs,
    ):
        self.steer_once = steer_once
        self.model = model
        self.layer = layer
        self.device = getattr(model, 'cfg', None) and getattr(model.cfg, 'device', 'cuda') or 'cuda'
        self.hook_point = hook_point
        self.position = position
        self.norm = norm
        self.post_processor = post_processor
        self.kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, dict):
                try:
                    self.kwargs[k] = {int(key): val for key, val in v.items()}
                except (ValueError, TypeError):
                    self.kwargs[k] = v
            else:
                self.kwargs[k] = v
        logger.debug(f"Kargs passed to BaseSteerModel: {kwargs}")
        
        # Prep all vectors uniformly (detach, clone, squeeze)
        self.steering_vector = {
            int(k): self._prep_vec(v) for k, v in steering_vector.items()
        }
        
        # Validate layer-vector alignment
        missing = set(self.layer) - set(self.steering_vector.keys())
        if missing:
            raise ValueError(f"Missing steering vectors for layers: {missing}")
        
        # Initialize hook handles for cleanup
        self._hook_handles = []
        self.metadata = list()
        self._prompt_metadata: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def _prep_vec(vec: torch.Tensor) -> torch.Tensor:
        """Detach, clone, squeeze [1, d] → [d]."""
        v = vec.detach().clone()
        if v.dim() == 2 and v.shape[0] == 1:
            v = v.squeeze(0)
        return v
    
    def _resolve_layer_kwargs(self, layer: int) -> Dict[str, Any]:
        """Resolve per-layer kwargs for hook installation.

        Steering configs can pass parameters as either:
          * Scalars (applied to all layers)
          * Dicts keyed by layer index (per-layer values)

        Example:
            top_idx = {17: [...], 18: [...]}

        This method returns the resolved dict for a given layer.
        """
        resolved: Dict[str, Any] = {}
        for k, v in self.kwargs.items():
            if isinstance(v, dict):
                if layer in v:
                    resolved[k] = v[layer]
                elif str(layer) in v:
                    resolved[k] = v[str(layer)]
                elif str(int(layer)) in v:
                    resolved[k] = v[str(int(layer))]
                else:
                    resolved[k] = v
            else:
                resolved[k] = v
        logger.debug(f"Resolved kwargs for layer {layer}: {resolved.keys()}")
        return resolved

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        """Setup steering hooks for the model.

        Args:
            coeff: Per-layer steering coefficients Dict[int, float]

        Returns:
            List of hook handles for cleanup
        """
        hook_handles = []
        hook_names = self.get_hook_name()

        for i, layer in enumerate(self.layer):
            hook_name = hook_names[i]
            vec = self.steering_vector[layer]
            layer_coeff = coeff.get(layer, coeff.get(min(coeff.keys(), key=lambda k: abs(k - layer)), next(iter(coeff.values()))))
            pos = self.position
            layer_kwargs = self._resolve_layer_kwargs(layer)

            handle = self.model.add_hook(
                hook_name,
                partial(
                    self._apply_steering_hook,
                    hook_fn=self.hook_fn,
                    position=pos,
                    coeff=layer_coeff,
                    steering_vector=vec,
                    **layer_kwargs,
                ),
            )
            hook_handles.append(handle)

        hook_handles.extend(self._setup_post_process_hooks())

        return hook_handles

    def _setup_post_process_hooks(self) -> List:
        if self.post_processor is None:
            return []
        return self.post_processor.setup_hooks(self.model)
    
    @abstractmethod
    def hook_fn(
        self,
        resid: torch.Tensor,
        position: Union[str,int],
        coeff: float,
        steering_vector: torch.Tensor,
        hook,
        **kwargs
    ) -> torch.Tensor:
        """
        Hook function to apply steering to residual stream.
        
        Args:
            resid: Residual stream tensor [batch, seq, d_model]
            coeff: Steering coefficient
            steering_vector: The steering vector for this layer
            hook: Hook object
            **kwargs: Additional keyword arguments
            
        Returns:
            Modified residual stream
        """
        pass
    
    def get_hook_name(self, layer: Optional[List[int]] = None, position: Optional[List[str]] = None) -> List[str]:
        """
        Get the hook name for the target layer.
        
        Args:
            layer: Override layer. If None, uses self.layer. Must be List[int].
            position: Override position ("pre", "post", "mid"). If None, uses self.hook_point.
        """
        pos = position or self.hook_point
        if isinstance(pos, str):
            pos = [pos]
        lay = layer if layer is not None else self.layer
        
        hook_names = [get_hook_name(l, p) for l in lay for p in pos]
        return hook_names

    def _reset_prompt_metadata(self) -> None:
        """Reset metadata container for one prompt generation."""
        self._prompt_metadata = {"_captured": False}

    def _finalize_prompt_metadata(self) -> Dict[str, Any]:
        """Return prompt metadata and clear internal state."""
        if self._prompt_metadata is None:
            return {}
        result = {k: v for k, v in self._prompt_metadata.items() if not k.startswith("_")}
        self._prompt_metadata = None
        return result

    def _apply_steering_hook(
        self,
        resid: torch.Tensor,
        hook,
        hook_fn,
        coeff,
        position: Optional[Union[str, int]] = None,
        **hook_kwargs,
    ) -> torch.Tensor:
        """Wrapper that runs steering hook and records shared norm metadata."""
        if self.steer_once and resid.shape[1] == 1:
            return resid
        if self._prompt_metadata is None:
            self._reset_prompt_metadata()
        if self._prompt_metadata.get("_captured", False):
            return hook_fn(resid=resid, hook=hook, coeff=coeff, position=position, **hook_kwargs)

        pre_acts = get_resid_acts(resid, position).detach().clone()
        updated_resid = hook_fn(resid=resid, hook=hook, coeff=coeff, position=position, **hook_kwargs)
        post_acts = get_resid_acts(updated_resid, position).detach()

        # Shared steering metadata (first token only). Edit this block to swap metrics.
        self._prompt_metadata["resid_norm_pre_steer"] = pre_acts.norm(dim=-1).mean().item()
        self._prompt_metadata["resid_norm_post_steer"] = post_acts.norm(dim=-1).mean().item()
        # Save full pre-steer activation for tail-distribution analysis
        # Shape: [batch, d_model]; generate() processes 1 prompt at a time
        # self._prompt_metadata["pre_steer_acts"] = pre_acts[0].float().cpu().tolist()
        
        # Track projections and orthogonal norms for non-OT steering
        layer = int(hook.name.split(".")[1])
        if self.steering_vector is not None and layer in self.steering_vector:
            vec = self.steering_vector[layer].to(device=pre_acts.device, dtype=pre_acts.dtype)
            v_unit = vec / (vec.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            
            proj_unsteered = torch.sum(pre_acts * v_unit, dim=-1, keepdim=True)
            proj_steered = torch.sum(post_acts * v_unit, dim=-1, keepdim=True)
            self._prompt_metadata["proj_unsteered"] = proj_unsteered.mean().item()
            self._prompt_metadata["proj_steered"] = proj_steered.mean().item()
            
            # Capture Orthogonal Subspace Norms (Standard steered)
            unsteered_orth = pre_acts - proj_unsteered * v_unit
            steered_orth = post_acts - proj_steered * v_unit
            self._prompt_metadata["norm_orth_pre_steer"] = unsteered_orth.norm(dim=-1).mean().item()
            self._prompt_metadata["norm_orth_post_steer"] = steered_orth.norm(dim=-1).mean().item()
            
        self._prompt_metadata["_captured"] = True

        return updated_resid



    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        apply_steer: bool = True,
        coeff: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text with optional steering applied.
        
        Args:
            prompt: Input text prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            apply_steer: Whether to apply steering
            coeff: Per-layer steering coefficients Dict[int, float]. Defaults to 1.0 for all layers.
            
        Returns:
            Generated text or list of generated texts
        """
        # Convert single prompt to list for unified processing
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt
        if coeff is None:
            coeff = {layer: 0.0 for layer in self.layer}
        
        if prompts:
            self.metadata = []

        try:
            # Generate for each prompt individually
            outputs = []
            for p in prompts:
                self._reset_prompt_metadata()
                install_post_on_baseline = (
                    self.post_processor is not None and
                    getattr(self.post_processor, "apply_on_baseline", False)
                )
                if apply_steer or install_post_on_baseline:
                    self.model.reset_hooks()
                    self._hook_handles = self.setup_hooks(coeff)

                output = self.model.generate(
                    p,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=do_sample,
                    verbose=False,
                    **kwargs,
                )
                outputs.append(output)
                self.metadata.append(self._finalize_prompt_metadata())

            # Return only generated parts
            results = [output[len(p):] for p, output in zip(prompts, outputs)]

            # Return single result if input was single, otherwise return list
            return results
        finally:
            self.model.reset_hooks()
    
    def get_token_probs(
        self,
        prompt: str,
        tokens: List[str],
        coeff: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Get normalized probabilities for specific tokens at next position.
        Used by logit-based evaluators (CAA, MMLU).
        
        Args:
            prompt: Input text
            tokens: List of token strings to get probs for (e.g., ["A", "B"])
            coeff: Per-layer steering coefficients Dict[int, float]. Defaults to 1.0 for all layers.
            
        Returns:
            Dict mapping token -> normalized probability
        """
        try:    
            self.metadata = []
            self._reset_prompt_metadata()
            apply_steer = bool(kwargs.pop("apply_steer", True))
            if coeff is None:
                coeff = {layer: 1.0 for layer in self.layer}
            install_post_on_baseline = (
                self.post_processor is not None and
                getattr(self.post_processor, "apply_on_baseline", False)
            )
            if (apply_steer and any(v != 0 for v in coeff.values())) or install_post_on_baseline:
                self.model.reset_hooks()
                self._hook_handles = self.setup_hooks(coeff)
            
            with torch.no_grad():
                input_ids = self.model.to_tokens(prompt)
                logits = self.model.run_with_hooks(input_ids, return_type="logits")[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                
                result = {}
                total = 0
                for t in tokens:
                    tid = self.model.tokenizer.encode(t)[-1]
                    result[t] = probs[tid].item()
                    total += result[t]
                
                # Normalize relative to candidates
                for t in tokens:
                    result[t] /= (total + 1e-9)

                self.metadata.append(self._finalize_prompt_metadata())
                
                return result
        finally:
            self.model.reset_hooks()
    
    def get_output_metadata(self):
        return self.metadata    



class BaseSAESteerModel(BaseSteerModel):
    """
    Base class for SAE-based steering models.
    
    Handles the common setup_hooks logic for SAE steering models that need
    to pass both steering_vector and sae to the hook function.
    """
    
    def __init__(
        self,
        model,
        layer: List[int],
        sae: Dict[int, Any],
        steering_vector: Dict[int, torch.Tensor],
        top_k: Optional[List[int]] = None,
        hook_point: List[str] = ["pre"],
        position: Union[str,int] = 'last',
        **kwargs,
    ):
        super().__init__(model, layer, steering_vector, hook_point=hook_point, position = position, **kwargs)
        self.sae = sae
        self.top_k = top_k

    def select_top_k_indices(self, top_idx: Union[torch.Tensor, List[int]]) -> torch.Tensor:
        """Select feature indices from ranked candidates using normalized top_k.

        If self.top_k is None, all candidate indices are returned unchanged.
        """
        if not isinstance(top_idx, torch.Tensor):
            top_idx = torch.tensor(top_idx, dtype=torch.long)
        else:
            top_idx = top_idx.to(dtype=torch.long)

        if self.top_k is None:
            return top_idx

        selector = torch.tensor(self.top_k, dtype=torch.long, device=top_idx.device)
        if selector.numel() == 0:
            return top_idx

        valid_selector = selector[(selector >= 0) & (selector < top_idx.numel())]
        if valid_selector.numel() == 0:
            raise ValueError(
                f"Configured top_k positions {self.top_k} are out of bounds for "
                f"{top_idx.numel()} extracted features"
            )

        return top_idx[valid_selector]

    @abstractmethod
    def hook_fn(
        self,
        resid: torch.Tensor,
        position: Union[str,int],
        coeff: float,
        steering_vector: torch.Tensor,
        sae: Optional[Any],
        hook,
    ) -> torch.Tensor:
        pass

    def setup_hooks(self, coeff: Dict[int, float]) -> List:
        """Setup steering hooks with SAE selection for each layer.

        NOTE: Uses residual stream hooks (not SAE activation hooks) because
        _apply_steering_hook implementations manually compute SAE encode/decode.
        """
        hook_handles = []
        hook_names = self.get_hook_name()

        for i, layer in enumerate(self.layer):
            hook_name = hook_names[i]
            vec = self.steering_vector[layer]
            layer_coeff = coeff[layer]
            sae_for_layer = self.sae[layer]
            pos = self.position
            layer_kwargs = self._resolve_layer_kwargs(layer)

            handle = self.model.add_hook(
                hook_name,
                partial(
                    self._apply_steering_hook,
                    hook_fn=self.hook_fn,
                    coeff=layer_coeff,
                    position=pos,
                    steering_vector=vec,
                    sae=sae_for_layer,
                    **layer_kwargs,
                ),
            )
            hook_handles.append(handle)

        hook_handles.extend(self._setup_post_process_hooks())

        return hook_handles


# =============================================================================
# Base Evaluator
# =============================================================================

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.

    Subclasses must implement :meth:`check`.
    Override :meth:`_get_responses` only when text generation is not
    the right way to obtain model outputs (e.g. logit-based evaluation).
    """

    is_model_based = False

    def unload(self):
        """Unload any loaded models to free VRAM."""
        unloaded = []
        for attr in ["_model", "_tokenizer"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)
                unloaded.append(attr)
        if unloaded:
            logger.info(f"Unloaded evaluator GPU resources: {unloaded}")
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Pipeline interface
    # -----------------------------------------------------------------

    def batch(
        self,
        steer_model,
        samples: List[dict],
        coeff: Dict[int, float] = None,
        **kwargs,
    ) -> List[Tuple[int, float, str]]:
        """
        Generate responses and score them.  Called by ``pipeline.evaluate()``.

        Args:
            steer_model: Steering model (has ``.generate()`` / ``.get_token_probs()``)
            samples: Single sample dict or list thereof
            coeff:   Per-layer steering coefficients Dict[int, float]
            **kwargs: Forwarded to ``_get_responses`` (e.g. ``max_new_tokens``)

        Returns:
            ``(is_correct, confidence, response_str)`` or list of such tuples.
        """
        if not samples:
            return []

        responses = self._get_responses(steer_model, samples, coeff, **kwargs)
        metadata = steer_model.get_output_metadata()
        results = []
        for id, sample, response in zip(range(len(samples)), samples, responses):
            data = metadata[id] if id < len(metadata) else None 
            is_correct, confidence = self.check(
                response,
                ground_truth=sample.get("answer"),
                prompt=sample.get("question", ""),
            )
            results.append((is_correct, confidence, self._format_response(response), data)) if data \
                else results.append((is_correct, confidence, self._format_response(response), dict()))

        return results

    # -----------------------------------------------------------------
    # Response generation (override only when needed)
    # -----------------------------------------------------------------

    def _get_responses(
        self,
        steer_model,
        samples: List[dict],
        coeff: Dict[int, float],
        **kwargs,
    ) -> List[Union[str, Dict[str, float]]]:
        """Default: text generation.  Override for logit-based evaluators."""
        prompts = [s["question"] for s in samples]
        return steer_model.generate(prompts, coeff=coeff, **kwargs)

    # -----------------------------------------------------------------
    # Core scoring (the ONE abstract method)
    # -----------------------------------------------------------------

    @abstractmethod
    def check(
        self,
        response,
        ground_truth=None,
        **context,
    ) -> Tuple[int, float]:
        """
        Score a single model output.

        Args:
            response:     Model output (``str`` for text, ``dict`` for logits).
            ground_truth: Expected answer, if applicable.
            **context:    Extra info pulled from the sample dict
                          (e.g. ``prompt=`` for RefusalMatcher).

        Returns:
            ``(is_correct, confidence)`` — both in [0, 1].
        """
        ...

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _format_response(response) -> str:
        """Convert a response (str or logit dict) to a display string."""
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and response:
            return str(max(response, key=response.get))
        return str(response)

    def get_output_metadata(self):
        """Get metadata from the steer model after generation."""
        return self.metadata if hasattr(self, 'metadata') else None
