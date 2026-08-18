"""
Unified Pipeline for Steering Vector Experiments.

Core class: SteeringPipeline
For execution, use the CLI: python -m Steering.cli
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import torch
from huggingface_hub import login
from tqdm import tqdm


from .data import DataLoader, EvalDataLoader, TRAIN_DATASET_REGISTRY, TEST_DATASET_REGISTRY
from .utils import (
    make_params, 
    load_fallback_sae,
    SAEWrapper,
)
from .config import (
    PipelineConfig,
    SAE_METHODS,
    SampleResult, EvalResult,
    SteeringVector,
)
from .logger import setup_logger
from .exceptions import ModelError

logger = setup_logger(__name__)

from transformer_lens import HookedTransformer
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluators import EVALUATOR_MAP
from .extractors import EXTRACTOR_MAP
from .steer_models import STEER_MAP


class SteeringPipeline:
    """
    Unified pipeline for steering experiments (Extraction, Generation, Evaluation).
    
    All operations are config-driven. Pass PipelineConfig to constructor,
    then call run() or evaluate().
    
    Usage:
        # From config file
        config = PipelineConfig.load("config.json")
        pipeline = SteeringPipeline(config)
        result = pipeline.evaluate()
        
        # From config object
        config = PipelineConfig(
            model=ModelConfig(name="meta-llama/Llama-2-7b-chat-hf"),
            extractor=ExtractorConfig(method="CAA", layer=14),
            steer=SteerConfig(method="CAA", layer=14, coeff=1.5),
            train_dataset="sycophancy",
            test_dataset="csqa",
        )
        pipeline = SteeringPipeline(config)
        result = pipeline.evaluate()
    """
    
    # =========================================================================
    # Static Helpers
    # =========================================================================
    
    @staticmethod
    def list_methods() -> list:
        return list(EXTRACTOR_MAP.keys())

    @staticmethod
    def list_evaluators() -> list:
        return list(EVALUATOR_MAP.keys())
    
    @staticmethod
    def list_train_datasets() -> list:
        return list(TRAIN_DATASET_REGISTRY.keys())
    
    @staticmethod
    def list_test_datasets() -> list:
        return list(TEST_DATASET_REGISTRY.keys())
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline from config.
        
        Args:
            config: PipelineConfig object containing all experiment settings
        """
        if not isinstance(config, PipelineConfig):
            raise TypeError(
                f"Expected PipelineConfig, got {type(config).__name__}. "
                "Use PipelineConfig.load('path.json') or PipelineConfig(...) to create config."
            )
        
        self.config = config.resolve()

        # Convenience aliases (references into self.config)
        self.model_config = self.config.model
        self.extract_config = self.config.extractor
        self.steer_config = self.config.steer
        self.post_process_config = self.config.post_process
        self.model_name = self.config.model.name
        self.device = self.config.model.device
        self.dtype = self.config.model.dtype
        
        # Runtime state
        self.model = None
        self._current_model_type = None  # "tl", "hf", or "sae"
        self._sae_cache: Dict[int, SAE] = {}
        self.extractor = None
        self.steer_model = None
        self.post_processor = None
        self._evaluator_cache = {}
        self.extraction_flops = 0

    @classmethod
    def from_config(cls, config: Union[PipelineConfig, str, Path]) -> "SteeringPipeline":
        """
        Create pipeline from config file or object.
        
        Args:
            config: PipelineConfig object, or path to JSON config file
            
        Returns:
            SteeringPipeline instance
        """
        if isinstance(config, (str, Path)):
            config = PipelineConfig.load(config)
        return cls(config)
    
    # =========================================================================
    # Model Loading
    # =========================================================================
    
    def authenticate(self, token: Optional[str] = None):
        """Authenticate with HuggingFace."""
        token = token or os.environ.get("HF_TOKEN")
        if token:
            logger.info("Authenticating with HuggingFace...")
            try:
                login(token=token)
                logger.info("Authentication successful")
            except Exception as e:
                logger.exception("Authentication failed during HuggingFace login")
        else:
            logger.warning("No HF_TOKEN found, attempting to use cached credentials...")
    
    def load_model(self, model_type: str = "tl"):
        """
        Load model (cached - skips if already loaded with same type).
        
        Args:
            model_type: "tl" (TransformerLens), "hf" (HuggingFace), or "sae" (SAE-enabled TL)
        """
        # 1. Check if we need to reload
        if self.model is not None:
            if self._current_model_type == model_type:
                logger.debug(f"Model already loaded ({model_type})")
                return self.model
            
            # Unload previous model
            logger.info(f"Switching model type: {self._current_model_type} -> {model_type}")
            del self.model
            self.model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # 2. Load new model
        logger.info(f"Loading model: {self.model_name} ({model_type})")
        try:
            if model_type == "hf":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=self.dtype, device_map=str(self.device)
                )
            else:
                extra_kwargs = self.config.model.model_kwargs or {}
                self.model = HookedTransformer.from_pretrained(
                    self.model_name, dtype=self.dtype, device=self.device, **extra_kwargs
                )
                
                # Optional compilation
                if self.model_config.use_compile and hasattr(torch, 'compile'):
                    try:
                        logger.info("Compiling model with torch.compile...")
                        self.model = torch.compile(self.model)
                    except Exception as e:
                        logger.warning(f"Model compilation failed: {e}. Using uncompiled model.")
            
            self._current_model_type = model_type
            return self.model
            
        except Exception as e:
            raise ModelError(f"Failed to load model {self.model_name}: {e}") from e

    def unload_model(self):
        """Unload generator model from GPU and free VRAM.

        Clears ALL references to the model — not just ``self.model``.
        The extractor, steer model, post-processor, and SAE cache all
        hold their own references; if any survive, ``gc.collect()``
        cannot free the CUDA tensors and VRAM stays allocated.
        """
        if self.model is not None:
            logger.info("Unloading generator model to free VRAM...")
            self.steer_model = None
            self.extractor = None          # extractor.model holds the same ref
            self.post_processor = None     # GLP denoiser on GPU
            self.model = None
            self._current_model_type = None
            self._sae_cache.clear()        # SAE objects (GPU if sae_cpu=False)
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def load_sae(self, layers: List[int]) -> Dict[int, SAE]:
        """
        Load SAE for layer(s) (cached). Always returns Dict[int, SAE].
        """
        for l in layers:
            if l not in self._sae_cache:
                self._sae_cache[l] = self._load_single_sae(l)
        return self._sae_cache
    
    def _load_single_sae(self, layer: int) -> SAE:
        """Load single SAE, respecting model_config.sae_cpu for device placement."""
        release = self.model_config.get_sae_release()
        sae_id = self.model_config.get_sae_id(layer)
        
        if not release:
            raise ValueError(f"SAE release not found for model: {self.model_name}")

        # Determine target device: CPU or same as model
        sae_cpu = getattr(self.model_config, 'sae_cpu', True)
        load_device = "cpu" if sae_cpu else self.device

        import gc
        try:
            logger.info(f"Loading SAE from {release}, id={sae_id} (device={load_device})")
            gc.collect()
            torch.cuda.empty_cache()
            sae, _, _ = SAE.from_pretrained(release, sae_id, device=str(load_device))
        except Exception as e:
            logger.warning(f"Standard loading failed: {e}. Using fallback...")
            try:
                sae = load_fallback_sae(
                    release=release, 
                    sae_id=sae_id, 
                    layer=layer, 
                    model_config=self.model_config,
                    device=str(load_device)
                )
            except Exception as fallback_e:
                raise RuntimeError(f"SAE loading failed. Standard: {e}, Fallback: {fallback_e}") from fallback_e

        # Ensure SAE dtype matches model dtype to prevent mismatches
        model_dtype = self.model_config.get_dtype()
        sae = sae.to(dtype=model_dtype)

        return SAEWrapper(sae)
    


    def _get_model_type_for_method(self, method: str) -> str:
        """Determine the required model type based on extraction/steering method."""
        method = method.upper()
        if method == "CAST_HF" or method == "FLAS":
            return "hf"
        elif method in SAE_METHODS:
            return "sae"
        else:
            return "tl"

    def _instantiate_component(
        self,
        registry: dict,
        method: str,
        layers: List[int],
        config_params: Dict[str, Any],
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate extractor or steer model from resolved config."""
        method_upper = method.upper()
        cls = registry.get(method_upper)
        if not cls:
            raise ValueError(f"Unknown method: {method_upper}. Available: {list(registry.keys())}")
        
        # Prepare base params
        params = {
            "model": self.model,
            "model_name": self.model_name,
            "device": self.device,
            "layer": layers,
            **config_params,
            **(extra_params or {}),
        }
        
        # Inject SAEs if needed
        if method_upper in SAE_METHODS:
            params["sae"] = self.load_sae(layers)
        
        # Filter params to match signature and instantiate
        return cls(**make_params(cls.__init__, **params))
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_train_data(
        self,
        dataset_name: str,
        n_samples: int,
        apply_chat_template: bool = False,
    ) -> tuple:
        """
        Load training data (target, contrast prompts).
        
        When contrast_key is None (e.g. CorrSteer), returns raw data dicts
        as target_data with contrast_data=None.
        
        Args:
            dataset_name: Name of dataset to load
            n_samples: Number of samples to load
            apply_chat_template: If True, applies chat template format
            
        Returns:
            Tuple of (target_data, contrast_data) where contrast_data is
            None when the dataset has no contrastive split.
        """
        return DataLoader().load(
            dataset_name,
            n_samples,
            apply_chat_template=apply_chat_template,
            tokenizer=self._get_tokenizer(),
        )
    
    def load_test_data(
        self,
        dataset_name: str,
        n_samples: Optional[int] = None,
        apply_chat_template: bool = False,
    ) -> list:
        """
        Load test data.
        
        Args:
            dataset_name: Name of dataset to load
            n_samples: Number of samples to load (None = all)
            apply_chat_template: If True, applies chat template format
            
        Returns:
            List of test data samples
        """
        return EvalDataLoader().load(
            dataset_name,
            n_samples,
            apply_chat_template=apply_chat_template,
            tokenizer=self._get_tokenizer(),
        )

    def _get_tokenizer(self):
        """Helper to get tokenizer if chat template is requested."""
        if hasattr(self.model, 'tokenizer'):  # TL model
                return self.model.tokenizer
        return AutoTokenizer.from_pretrained(self.model_name)

    @staticmethod
    def _get_config_params(config, skip_keys: set) -> Dict[str, Any]:
        """
        Extract pass-through params from a config dataclass, skipping structural fields.
        
        Args:
            config: ExtractorConfig or SteerConfig instance
            skip_keys: Set of field names to exclude (e.g., {"method", "layer", "train_dataset", "n_train"})
            
        Returns:
            Dict of remaining params for constructor injection
        """
        return {k: v for k, v in config.to_dict().items() if k not in skip_keys}

    def _attach_post_processor(self, extra_params: Dict[str, Any]) -> Dict[str, Any]:
        """Attach GLP post-processor to steer model params when enabled."""
        if not self.post_process_config.enabled:
            return extra_params

        from .post_process.hook import GLPPostProcessor

        self.post_processor = GLPPostProcessor.from_config(
            self.post_process_config,
            device=str(self.device),
        )
        self.post_processor.load()
        extra_params["post_processor"] = self.post_processor
        return extra_params

    
    # =========================================================================
    # Extraction & Steering (Config-driven)
    # =========================================================================
    
    def extract(
        self,
        target_data,
        contrast_data=None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Extract steering vector using extractor config.
        
        Args:
            target_data: Target prompts (List[str]) or raw data dicts (List[Dict])
                         for methods like CorrSteer that need full data.
            contrast_data: Contrast prompts (List[str]) or None for methods
                           that don't use contrastive data.
            **kwargs: Additional args forwarded to extractor.
            
        Returns:
            Extracted steering vector (single tensor or dict of layer->tensor)
        """
        self.extractor = self._instantiate_component(
            EXTRACTOR_MAP,
            self.extract_config.method,
            self.extract_config.layer,
            self._get_config_params(self.extract_config, {"method", "layer"}),
        )
        
        # Extract with FLOP tracking
        from .utils import FlopTracker
        with FlopTracker() as tracker:
            vector_tensor = self.extractor.extract(target_data=target_data, contrast_data=contrast_data, **kwargs)
        self.extraction_flops = tracker.total_flops
        logger.info(f"Extraction FLOPs: {self.extraction_flops:,}")

        if self.config.save_vector:
            logger.info(f"Saving extracted vector to: {self.config.save_vector}")
            try:
                save_path = Path(self.config.save_vector)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                self.extractor.save(str(save_path))
            except Exception as e:
                logger.error(f"Failed to save vector to {self.config.save_vector}: {e}")
        return vector_tensor
    
    def steering(self):
        """
        Setup steer model for generation using steer config.
        Returns:
            Configured steer model
        """
        # Get extractor-specific params if available
        if self.extractor:
            self.steer_config.layer = list(self.extractor.layer)
            extra_params = self.extractor.get_steer_params()
        elif self.config.load_vector:
            sv = SteeringVector.load(
                self.config.load_vector, device=str(self.device), layer = self.steer_config.layer
            )
            self.steer_config.layer = list(sv.vector.keys())
            extra_params = sv.to_steer_params()
        else:
            extra_params = {}

        extra_params = self._attach_post_processor(extra_params)
        
        self.steer_model = self._instantiate_component(
            STEER_MAP,
            self.steer_config.method,
            self.steer_config.layer,
            self._get_config_params(self.steer_config, {"method", "layer", "coeff"}),
            extra_params
        )
        
        logger.info(f"Initialized {self.steer_config.method.upper()} steer model at layer {self.steer_config.layer}")
        return self.steer_model
    
    def generate(
        self,
        prompt: str,
        coeff: Optional[Dict[int, float]] = None,
        max_new_tokens: int = 150,
        apply_steer: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate with steering.
        
        Args:
            prompt: Input text prompt
            coeff: Per-layer steering coefficients Dict[int, float] (default: from resolved config)
            max_new_tokens: Maximum tokens to generate
            apply_steer: Whether to apply steering (False for baseline)
            **kwargs: Additional args passed to model.generate
            
        Returns:
            Generated text
        """
        # Use resolved config defaults if not specified
        coeff = coeff if coeff is not None else self.steer_config.coeff

        return self.steer_model.generate(
            prompt,
            coeff=coeff,
            max_new_tokens=max_new_tokens,
            apply_steer=apply_steer,
            **kwargs,
        )
    
    # =========================================================================
    # Evaluator
    # =========================================================================
    
    def get_evaluator(self, evaluator_type: str):
        """Get or create evaluator (cached by type)."""
        if evaluator_type not in self._evaluator_cache:
            factory = EVALUATOR_MAP.get(evaluator_type)
            if factory is None:
                raise ValueError(
                    f"Unknown evaluator: {evaluator_type}. "
                    f"Available: {list(EVALUATOR_MAP.keys())}"
                )
            self._evaluator_cache[evaluator_type] = factory(device=self.device)
        return self._evaluator_cache[evaluator_type]
    
    # =========================================================================
    # Pipeline Setup
    # =========================================================================
    
    def setup(self):
        """
        Setup pipeline: authenticate, load model, extract/load vector, init steer model.
        
        Uses self.config for all settings. Call this before generate() or run individual steps.
        """
        # Authenticate
        self.authenticate()
        
        # Skip if steering already configured
        if self._is_steering_ready():
            logger.debug("Steering model already configured, skipping setup.")
            return
        
        # Setup vector: load or extract
        if self.config.load_vector:
            self._setup_vector_from_file()
        else:
            self._setup_vector_from_extraction()

    def _is_steering_ready(self) -> bool:
        """Check if steering model is already properly configured."""
        return (
            self.steer_model is not None and
            self.steer_model.steering_vector is not None
        )

    def _setup_vector_from_file(self):
        """Load steering vector from file and setup steer model."""
        logger.info(f"Skipping extraction - loading vector from: {self.config.load_vector}")
        steer_model_type = self._get_model_type_for_method(self.steer_config.method)
        self.load_model(steer_model_type)
        self.steering()

    def _setup_vector_from_extraction(self):
        """Extract steering vector from data and setup steer model."""
        train_dataset = self.config.train_dataset
        if not train_dataset:
            raise ValueError("train_dataset required in extractor config when load_vector not provided")
        
        # Load model for extraction
        extraction_model_type = self._get_model_type_for_method(self.extract_config.method)
        self.load_model(extraction_model_type)
        
        # Unified data loading — contrast_key=None yields (raw_dicts, None)
        train_data = self.load_train_data(
            train_dataset,
            self.config.n_train,
            apply_chat_template=self.extract_config.apply_chat_template,
        )
        target = [d['correct_prompt'] for d in train_data]
        contrast = [d['false_prompt'] for d in train_data] if 'false_prompt' in train_data[0] else None
        target_response = [d["target_response"] for d in train_data] if train_data and "target_response" in train_data[0] else None
        contrast_response = [d["contrast_response"] for d in train_data] if train_data and "contrast_response" in train_data[0] else None

        if self.extract_config.inverse and contrast is not None:
            target, contrast = contrast, target
            if target_response is not None and contrast_response is not None:
                target_response, contrast_response = contrast_response, target_response
            logger.info("Extractor inverse=True: swapped target and contrast training prompts and responses.")
        ground_truth = [d['answer'] for d in train_data] if 'answer' in train_data[0] else [1]*len(target)
        prompts = [d['question'] for d in train_data] if 'question' in train_data[0] else None
        choices = train_data[0].get('choices') if train_data else None

        logger.info(f"Example input: \nTarget input: {target[0]}\nContrast input: {contrast[0] if contrast else 'None'}")
        extract_kwargs = {
            "ground_truth": ground_truth,
            "prompts": prompts,
            "choices": choices,
            # Response-only fields emitted by unified formatters (e.g. deception, cais_mask,
            # ifeval, cast_combined).  Consumed by FLAS as (input, output) pairs and by REPS
            # for completion-masked training.  Methods that don't need them ignore via **kwargs.
            "target_response": target_response,
            "contrast_response": contrast_response,
        }

        self.extract(target_data=target, contrast_data=contrast, **extract_kwargs)
        
        # Load model for steering if needed
        steer_model_type = self._get_model_type_for_method(self.steer_config.method)
        if steer_model_type != extraction_model_type:
            self.load_model(steer_model_type)
        self.steering()
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    
    def run(self) -> Dict[str, Any]:
        """
        Run generation pipeline (no evaluation).
        
        Returns:
            Dict containing generation results
        """
        
        # Setup pipeline
        self.setup()
        test_dataset = self.config.test_dataset
        n_test = self.config.n_test
        
        if not test_dataset:
            raise ValueError("test_dataset is required in resolved config")
        
        
        # Load test data and generate
        test_data = self.load_test_data(
            dataset_name=test_dataset,
            n_samples=n_test,
            apply_chat_template=self.steer_config.apply_chat_template,
        )
        
        results = []
        for sample in tqdm(test_data, desc="Generating"):
            prompt = str(sample.get("question", sample))
            response = self.generate(prompt=prompt, max_new_tokens=self.model_config.max_new_tokens, \
                temperature=self.model_config.temperature, top_k=self.model_config.top_k, do_sample=self.model_config.do_sample)
            results.append({
                "prompt": prompt,
                "response": response,
                "sample": sample,
            })
        
        return {"results": results}

    def evaluate(self, verbose: bool = True) -> EvalResult:
        """
        Evaluate steering method.
        
        Args:
            verbose: Whether to show progress bars and logs
            
        Returns:
            EvalResult with accuracy metrics and sample details
        """
        test_dataset = self.config.test_dataset
        n_test = self.config.n_test
        include_baseline = self.config.include_baseline
        
        if not test_dataset:
            raise ValueError("test_dataset is required in resolved config")
        
        # Setup pipeline
        self.setup()
        
        # Load test data (HF sources handled by EvalDataLoader)
        test_data = self.load_test_data(
            dataset_name=test_dataset,
            n_samples=n_test,
            apply_chat_template=self.steer_config.apply_chat_template,
        )
        test_data_cfg = TEST_DATASET_REGISTRY[test_dataset]
        evaluator = self.get_evaluator(test_data_cfg.evaluator)

        # Evaluation helper
        def run_eval_wrapper(coeff, apply_steer=True):
             return self._run_eval_loop(
                 evaluator, test_data, coeff, 
                 self.steer_config.batch_size, verbose, 
                 apply_steer=apply_steer, 
                 max_new_tokens=self.model_config.max_new_tokens
             )
        
        # Check if the evaluator is model-based
        is_model_based = getattr(evaluator, "is_model_based", False)
        
        if is_model_based:
            logger.info("Using VRAM-optimized evaluation flow for model-based evaluator.")
            
            # 1. Generate steered responses
            steered_responses = []
            steered_metadata = []
            self.last_inference_flops = 0
            
            data_list = list(test_data)
            batch_size = self.steer_config.batch_size
            
            for i in tqdm(range(0, len(data_list), batch_size), desc="Generating (steered)", disable=not verbose):
                batch = data_list[i:i + batch_size]
                
                # Profile FLOPs only on the first batch
                if i == 0:
                    from .utils import FlopTracker
                    with FlopTracker() as tracker:
                        batch_responses = evaluator._get_responses(
                            self.steer_model,
                            samples=batch,
                            coeff=self.steer_config.coeff,
                            max_new_tokens=self.model_config.max_new_tokens
                        )
                    first_batch_flops = tracker.total_flops
                    self.last_inference_flops = int(first_batch_flops * (len(data_list) / len(batch)))
                else:
                    batch_responses = evaluator._get_responses(
                        self.steer_model,
                        samples=batch,
                        coeff=self.steer_config.coeff,
                        max_new_tokens=self.model_config.max_new_tokens
                    )
                
                batch_metadata = self.steer_model.get_output_metadata()
                steered_responses.extend(batch_responses)
                steered_metadata.extend(batch_metadata)
            
            # Compute steered metrics using self.model while it is loaded
            temp_steered_samples = []
            for sample, response, meta in zip(data_list, steered_responses, steered_metadata):
                temp_steered_samples.append(
                    self._create_sample_result(sample, evaluator._format_response(response), 0, 0.0, meta)
                )
            
            compute_lsp = self.config.compute_lsp
            steered_metrics = self._compute_response_score(
                temp_steered_samples,
                compute_ppl=self.config.compute_perplexity,
                compute_lsp=compute_lsp,
            )
            mean_ppl = steered_metrics["perplexity"]
            mean_rep = steered_metrics["repetition_rate"]
            mean_comp = steered_metrics["compression_ratio"]
            
            # Compute OOD (LSP) score
            lsp_score = None
            if compute_lsp:
                per_sample_nlls = [s.metadata.get("token_nlls") for s in temp_steered_samples if s.metadata]
                per_sample_tokens = [s.metadata.get("token_ids") for s in temp_steered_samples if s.metadata]
                res = self._compute_lsp_scores(per_sample_nlls, per_sample_tokens)
                lsp_score = res.get("lsp_score")
                
                # Clean up metadata token lists
                for s in temp_steered_samples:
                    if s.metadata:
                        s.metadata.pop("token_nlls", None)
                        s.metadata.pop("token_ids", None)
            
            # Log steered metrics
            logger.info(f"Steered metrics - 3-gram Repetition: {mean_rep:.2%}, Compression Ratio: {mean_comp:.3f}")
            if mean_ppl is not None:
                logger.info(f"Mean perplexity (steered): {mean_ppl:.2f}")
            if lsp_score is not None:
                logger.info(f"OOD score (steered): {lsp_score:.2f}")
            
            # 2. Generate baseline responses (if requested)
            baseline_responses = []
            baseline_metadata = []
            temp_baseline_samples = []
            baseline_mean_ppl, baseline_mean_rep, baseline_mean_comp = None, None, None
            baseline_lsp_score = None
            
            if include_baseline:
                for i in tqdm(range(0, len(data_list), batch_size), desc="Generating (baseline)", disable=not verbose):
                    batch = data_list[i:i + batch_size]
                    batch_responses = evaluator._get_responses(
                        self.steer_model,
                        samples=batch,
                        coeff=self.config.make_zero_coeff(),
                        apply_steer=False,
                        max_new_tokens=self.model_config.max_new_tokens
                    )
                    batch_metadata = self.steer_model.get_output_metadata()
                    baseline_responses.extend(batch_responses)
                    baseline_metadata.extend(batch_metadata)
                
                for sample, response, meta in zip(data_list, baseline_responses, baseline_metadata):
                    temp_baseline_samples.append(
                        self._create_sample_result(sample, evaluator._format_response(response), 0, 0.0, meta)
                    )
                
                base_metrics = self._compute_response_score(
                    temp_baseline_samples,
                    compute_ppl=self.config.compute_perplexity,
                    compute_lsp=compute_lsp,
                )
                baseline_mean_ppl = base_metrics["perplexity"]
                baseline_mean_rep = base_metrics["repetition_rate"]
                baseline_mean_comp = base_metrics["compression_ratio"]
                
                if compute_lsp:
                    base_sample_nlls = [s.metadata.get("token_nlls") for s in temp_baseline_samples if s.metadata]
                    base_sample_tokens = [s.metadata.get("token_ids") for s in temp_baseline_samples if s.metadata]
                    base_res = self._compute_lsp_scores(base_sample_nlls, base_sample_tokens)
                    baseline_lsp_score = base_res.get("lsp_score")
                    
                    # Clean up
                    for s in temp_baseline_samples:
                        if s.metadata:
                            s.metadata.pop("token_nlls", None)
                            s.metadata.pop("token_ids", None)
                
                # Log baseline metrics
                logger.info(f"Baseline metrics - 3-gram Repetition: {baseline_mean_rep:.2%}, Compression Ratio: {baseline_mean_comp:.3f}")
                if baseline_mean_ppl is not None:
                    logger.info(f"Mean perplexity (baseline): {baseline_mean_ppl:.2f}")
                if baseline_lsp_score is not None:
                    logger.info(f"OOD score (baseline): {baseline_lsp_score:.2f}")
            
            # 3. Unload the generator model to free VRAM before evaluator model is loaded
            self.unload_model()
            
            # 4. Evaluate/score the generated responses using evaluator
            # Steered evaluation
            correct = 0
            total = len(data_list)
            samples = []
            
            for sample, response, temp_sample in tqdm(zip(data_list, steered_responses, temp_steered_samples), desc="Scoring (steered)", total=total, disable=not verbose):
                is_correct, confidence = evaluator.check(
                    response,
                    ground_truth=sample.get("answer"),
                    prompt=sample.get("question", ""),
                )
                correct += is_correct
                final_sample = temp_sample
                final_sample.is_correct = is_correct
                final_sample.confidence = confidence
                samples.append(final_sample)
                
            accuracy = correct / total if total > 0 else 0.0
            logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
            inference_flops = self.last_inference_flops
            logger.info(f"Inference FLOPs: {inference_flops:,}")
            
            # Baseline evaluation
            baseline_accuracy = 0.0
            baseline_samples = []
            if include_baseline:
                base_correct = 0
                for sample, response, temp_sample in tqdm(zip(data_list, baseline_responses, temp_baseline_samples), desc="Scoring (baseline)", total=total, disable=not verbose):
                    is_correct, confidence = evaluator.check(
                        response,
                        ground_truth=sample.get("answer"),
                        prompt=sample.get("question", ""),
                    )
                    base_correct += is_correct
                    final_sample = temp_sample
                    final_sample.is_correct = is_correct
                    final_sample.confidence = confidence
                    baseline_samples.append(final_sample)
                baseline_accuracy = base_correct / total if total > 0 else 0.0
            
            # 5. Unload evaluator model to leave VRAM clean
            evaluator.unload()
            
        else:
            # Run evaluation with resolved coeff (Dict[int, float])
            correct, total, samples = run_eval_wrapper(self.steer_config.coeff)
            inference_flops = getattr(self, "last_inference_flops", 0)
            accuracy = correct / total if total > 0 else 0.0
            logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
            logger.info(f"Inference FLOPs: {inference_flops:,}")
            
            # Run baseline if requested (zero coeff dict)
            baseline_accuracy, baseline_samples = 0.0, None
            if include_baseline:
                base_correct, _, baseline_samples = run_eval_wrapper(self.config.make_zero_coeff(), apply_steer=False)
                baseline_accuracy = base_correct / total if total > 0 else 0.0

            # Compute generation response metrics in one pass
            compute_lsp = self.config.compute_lsp
            metrics = self._compute_response_score(
                samples, compute_ppl=self.config.compute_perplexity, compute_lsp=compute_lsp,
            )
            mean_ppl = metrics["perplexity"]
            mean_rep = metrics["repetition_rate"]
            mean_comp = metrics["compression_ratio"]

            logger.info(f"Steered metrics - 3-gram Repetition: {mean_rep:.2%}, Compression Ratio: {mean_comp:.3f}")
            if mean_ppl is not None:
                logger.info(f"Mean perplexity (steered): {mean_ppl:.2f}")

            # OOD score: PPL / (1 - rep_rate + ε). High if degenerate (low PPL + high rep)
            # OR incoherent (high PPL). Single continuous metric replacing separate PPL/rep checks.
            lsp_score = None
            if compute_lsp:
                per_sample_nlls = [s.metadata.get("token_nlls") for s in samples if s.metadata]
                per_sample_tokens = [s.metadata.get("token_ids") for s in samples if s.metadata]
                res = self._compute_lsp_scores(per_sample_nlls, per_sample_tokens)
                lsp_score = res.get("lsp_score")
                if lsp_score is not None:
                    logger.info(f"OOD score (steered): {lsp_score:.2f}")
                
                # Clean up steered samples to avoid polluting JSON results
                for s in samples:
                    if s.metadata:
                        s.metadata.pop("token_nlls", None)
                        s.metadata.pop("token_ids", None)

            baseline_mean_ppl, baseline_mean_rep, baseline_mean_comp = None, None, None
            baseline_lsp_score = None
            if include_baseline and baseline_samples:
                base_metrics = self._compute_response_score(
                    baseline_samples, compute_ppl=self.config.compute_perplexity, compute_lsp=compute_lsp,
                )
                baseline_mean_ppl = base_metrics["perplexity"]
                baseline_mean_rep = base_metrics["repetition_rate"]
                baseline_mean_comp = base_metrics["compression_ratio"]
                logger.info(f"Baseline metrics - 3-gram Repetition: {baseline_mean_rep:.2%}, Compression Ratio: {baseline_mean_comp:.3f}")
                if baseline_mean_ppl is not None:
                    logger.info(f"Mean perplexity (baseline): {baseline_mean_ppl:.2f}")

                if compute_lsp:
                    base_sample_nlls = [s.metadata.get("token_nlls") for s in baseline_samples if s.metadata]
                    base_sample_tokens = [s.metadata.get("token_ids") for s in baseline_samples if s.metadata]
                    base_res = self._compute_lsp_scores(base_sample_nlls, base_sample_tokens)
                    baseline_lsp_score = base_res.get("lsp_score")
                    if baseline_lsp_score is not None:
                        logger.info(f"OOD score (baseline): {baseline_lsp_score:.2f}")

                    # Clean up baseline samples to avoid polluting JSON results
                    for s in baseline_samples:
                        if s.metadata:
                            s.metadata.pop("token_nlls", None)
                            s.metadata.pop("token_ids", None)

        if verbose:
            ppl_str = f", ppl: {mean_ppl:.2f}" if mean_ppl is not None else ""
            metrics_str = f", rep: {mean_rep:.1%}, comp: {mean_comp:.2f}"
            lsp_str = f", ood: {lsp_score:.2f}" if lsp_score is not None else ""
            logger.info(
                f"{self.extract_config.method} L{self.extract_config.layer} c={self.steer_config.coeff}: {accuracy:.2%} "
                f"(baseline: {baseline_accuracy:.2%}, delta: {accuracy - baseline_accuracy:+.2%}{ppl_str}{metrics_str}{lsp_str})"
            )

        return EvalResult(
            method=self.extract_config.method,
            train_dataset=self.config.train_dataset,
            test_dataset=test_dataset,
            layer=self.extract_config.layer,
            coeff=self.steer_config.coeff,
            accuracy=accuracy,
            baseline_accuracy=baseline_accuracy,
            total=total,
            correct=correct,
            samples=samples,
            baseline_samples=baseline_samples,
            perplexity=mean_ppl,
            baseline_perplexity=baseline_mean_ppl,
            repetition_rate=mean_rep,
            baseline_repetition_rate=baseline_mean_rep,
            compression_ratio=mean_comp,
            baseline_compression_ratio=baseline_mean_comp,
            lsp_score=lsp_score,
            baseline_lsp_score=baseline_lsp_score,
            extraction_flops=self.extraction_flops,
            inference_flops=inference_flops,
        )

    def _run_eval_loop(self, evaluator, data, coeff, batch_size, verbose, apply_steer, max_new_tokens):
        """Internal evaluation loop using unified batch processing."""
        correct, total = 0, 0
        samples = []
        
        # Process data in batches using the configured batch_size
        data_list = list(data)
        self.last_inference_flops = 0
        
        for i in tqdm(range(0, len(data_list), batch_size), desc="Evaluating", disable=not verbose):
            batch = data_list[i:i + batch_size]
            
            # Profile FLOPs only on the first batch to avoid huge Python overhead of FlopCounterMode
            if i == 0 and apply_steer:
                from .utils import FlopTracker
                with FlopTracker() as tracker:
                    batch_results = evaluator.batch(
                        steer_model=self.steer_model,
                        samples=batch,
                        coeff=coeff,
                        apply_steer=apply_steer,    
                        max_new_tokens=max_new_tokens
                    )
                first_batch_flops = tracker.total_flops
                self.last_inference_flops = int(first_batch_flops * (len(data_list) / len(batch)))
            else:
                batch_results = evaluator.batch(
                    steer_model=self.steer_model,
                    samples=batch,
                    coeff=coeff,
                    apply_steer=apply_steer,    
                    max_new_tokens=max_new_tokens
                )
            
            for sample, (is_correct, confidence, response, data) in zip(batch, batch_results):
                correct += is_correct
                total += 1
                samples.append(self._create_sample_result(sample, response, is_correct, confidence, data))
        
        return correct, total, samples

    def _create_sample_result(self, sample, response, is_correct, confidence, data):
        """Create a SampleResult from evaluation data."""
        return SampleResult(
            prompt=sample.get("question", ""),
            response=response,
            ground_truth=sample.get("answer", ""),
            is_correct=is_correct,
            confidence=confidence,
            sample_data=sample,
            metadata=data,
        )

    # =========================================================================
    # Perplexity (optional post-processing)
    # =========================================================================

    def _compute_perplexity(
        self,
        model,
        prompt: str,
        response: str,
        max_length: int = 512,
        tokenizer=None,
        return_nlls: bool = False,
    ) -> dict:
        """Compute conditional perplexity PPL(response | prompt).

        Returns dict with 'ppl' (float) and 'nlls' (List[float] or None).
        nlls contains per-token NLL for response tokens when return_nlls=True.
        """
        nlls = None
        try:
            # TransformerLens path
            if hasattr(model, "cfg") and hasattr(model, "to_tokens"):
                device = model.cfg.device

                prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
                full_tokens = model.to_tokens(prompt + response, prepend_bos=True)

                if full_tokens.shape[1] > max_length:
                    full_tokens = full_tokens[:, :max_length]

                prompt_len = prompt_tokens.shape[1]
                full_tokens = full_tokens.to(device)

                with torch.no_grad():
                    loss_per_token = model(
                        full_tokens,
                        return_type="loss",
                        loss_per_token=True,
                    )
                    response_loss = loss_per_token[:, prompt_len - 1 :]
                    ppl = torch.exp(response_loss.mean()).item()
                    nlls = None
                    tokens = None
                    if return_nlls:
                        nlls = response_loss[0].tolist()
                        tokens = full_tokens[0, prompt_len :].tolist()
                    return {"ppl": ppl, "nlls": nlls, "tokens": tokens}

            # HuggingFace path
            if tokenizer is None:
                tokenizer = self._get_tokenizer()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_device = next(model.parameters()).device
            prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
            full_ids = tokenizer(prompt + response, add_special_tokens=True).input_ids
            full_ids = full_ids[:max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))
            if prompt_len >= len(full_ids):
                return {"ppl": float("inf"), "nlls": None, "tokens": None}

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=model_device)
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                ppl = float(torch.exp(outputs.loss).item())
                nlls = None
                tokens = None
                if return_nlls:
                    logits = outputs.logits  # [1, seq_len, vocab]
                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]
                    ce = torch.nn.functional.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                        reduction='none',
                    )
                    response_ce = ce[prompt_len - 1:]
                    nlls = response_ce.tolist()
                    tokens = full_ids[prompt_len:]
                return {"ppl": ppl, "nlls": nlls, "tokens": tokens}

        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")
            return {"ppl": float("inf"), "nlls": None, "tokens": None}

    def _compute_response_score(self, samples: List[SampleResult], compute_ppl: bool = True, compute_lsp: bool = False) -> Dict[str, float]:
        """Compute perplexity and repetition rate over generated responses.

        Args:
            samples: List of SampleResult objects containing prompts and generated responses.
            compute_ppl: Whether to run perplexity calculation (requires model forward passes).
            compute_lsp: Whether to return token-level NLLs and tokens for OOD calculation.
        """
        import zlib
        from .config.results import SampleResult
        
        ppls = []
        rep_rates = []
        comp_ratios = []
        
        # 1. Perplexity and per-token NLLs/IDs
        do_ppl = compute_ppl or compute_lsp  # lsp needs forward pass too
        tokenizer = None
        if do_ppl and not (hasattr(self.model, "cfg") and hasattr(self.model, "to_tokens")):
            tokenizer = self._get_tokenizer()
            
        for sample in tqdm(samples, desc="Computing response metrics"):
            text = sample.response
            if sample.metadata is None:
                sample.metadata = {}
                
            ppl = float("inf")
            if do_ppl:
                result = self._compute_perplexity(
                    self.model,
                    sample.prompt,
                    sample.response,
                    tokenizer=tokenizer,
                    return_nlls=compute_lsp,
                )
                ppl = result["ppl"]
                if compute_ppl:
                    ppls.append(ppl)
                if compute_lsp and result["nlls"] is not None:
                    sample.metadata["token_nlls"] = result["nlls"]
                if compute_lsp and result["tokens"] is not None:
                    sample.metadata["token_ids"] = result["tokens"]
            sample.metadata["perplexity"] = ppl
            
            # 2. 3-gram repetition rate
            words = text.strip().split()
            n = 3
            if len(words) < n:
                rep_rate = 0.0
            else:
                ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
                unique_ngrams = set(ngrams)
                rep_rate = 1.0 - (len(unique_ngrams) / len(ngrams))
            sample.metadata["repetition_rate"] = rep_rate
            sample.metadata["repetition"] = rep_rate
            rep_rates.append(rep_rate)
            
            # 3. zlib compression ratio
            if not text:
                comp_ratio = 1.0
            else:
                bytes_text = text.encode("utf-8")
                compressed = zlib.compress(bytes_text)
                comp_ratio = len(compressed) / len(bytes_text)
            sample.metadata["compression_ratio"] = comp_ratio
            sample.metadata["compression"] = comp_ratio
            comp_ratios.append(comp_ratio)
            
        # Calculate means
        mean_ppl = None
        if compute_ppl:
            finite_ppls = [p for p in ppls if p != float("inf")]
            mean_ppl = sum(finite_ppls) / len(finite_ppls) if finite_ppls else float("inf")
            
        mean_rep = sum(rep_rates) / len(rep_rates) if rep_rates else 0.0
        mean_comp = sum(comp_ratios) / len(comp_ratios) if comp_ratios else 1.0
        
        return {
            "perplexity": mean_ppl,
            "repetition_rate": mean_rep,
            "compression_ratio": mean_comp,
        }

    @staticmethod
    def _compute_lsp_scores(
        per_sample_nlls: List[List[float]],
        per_sample_tokens: List[List[int]],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute Localized Suffix-Penalized Perplexity (PPL_lsp) with window_size = 30.

        Fix vs old formula:
        - Old: exp(mean_nll_per_sample) averaged arithmetically across samples
          => one high-penalty sample explodes exp() and dominates the mean.
        - New: collect ALL penalized token NLLs across ALL samples, average globally,
          exponentiate ONCE -- same as standard PPL, with a per-token
          log1p(lcs-2) repetition penalty. Clean output ~2-6, OOD ~20-100.
        """
        import math

        window_size = 30
        all_losses = []
        n_valid = 0

        for nlls, tokens in zip(per_sample_nlls, per_sample_tokens):
            if not nlls or not tokens or len(nlls) != len(tokens) or len(tokens) <= 1:
                continue
            n_valid += 1
            for t in range(len(tokens)):
                raw_loss = nlls[t]
                start_idx = max(0, t - window_size)
                history = tokens[start_idx:t]

                lcs = 0
                for idx in range(len(history) - 1):
                    length = 0
                    while idx - length >= 0 and history[idx - length] == history[-1 - length]:
                        length += 1
                    if length > lcs:
                        lcs = length

                # Soft log penalty x2: amplifies gap between clean and degenerate
                penalty = 2.0 * math.log1p(max(0, lcs - 2))
                all_losses.append(raw_loss + penalty)

        if not all_losses:
            return {"lsp_score": None, "n_valid": 0}

        mean_loss = sum(all_losses) / len(all_losses)
        try:
            lsp_score = math.exp(mean_loss)
        except OverflowError:
            lsp_score = float("inf")

        return {"lsp_score": lsp_score, "n_valid": n_valid}
