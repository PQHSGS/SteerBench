"""
Data Loaders.

Provides DataLoader and EvalDataLoader classes for loading datasets.
Uses the unified TRAIN_DATASET_REGISTRY, COMPOSITE_DATASET_REGISTRY, and TEST_DATASET_REGISTRY.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from .readers import read_file
from .formatters import FORMATTERS, COMPOSITE_FORMATTERS
from .data_registry import (
    COMPOSITE_DATASET_REGISTRY,
    TRAIN_DATASET_REGISTRY,
    TEST_DATASET_REGISTRY,
)
from ..config import (
    TrainDatasetConfig,
    CompositeDatasetConfig,
    TestDatasetConfig,
)
from ..exceptions import DataLoadError
from ..logger import setup_logger
from ..utils import build_chat_input, clean_text

import numpy as np
import random

logger = setup_logger(__name__)


# Concept-level filtering was a temporary helper for smoke tests and has
# been removed. Data formatters may still include a `concept_id` field per
# sample, but loaders no longer perform automatic dataset-level filtering.


class DataLoader:
    """
    Loader for training datasets.
    
    Usage:
        loader = DataLoader()
        
        # Load simple dataset
        data = loader.load("sycophancy")
        
        # Load composite dataset (CAST, SRPS)
        data = loader.load("cast_refusal")  # Auto-detects composite
        
        # Get config for target/contrast keys
        cfg = loader.get_config("sycophancy")
        targets = [d[cfg.target_key] for d in data]
        contrasts = [d[cfg.contrast_key] for d in data]
    """
    
    def __init__(self, root: Path = None) -> None:
        """Initialize loader. Root is ignored (paths come from registry)."""
        pass  # Root is determined by registry

    def load(
        self,
        dataset_name: str,
        n_samples: Optional[int] = None,
        format: bool = True,
        apply_chat_template: bool = True,
        tokenizer = None,
        augmentation: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Load a training dataset by friendly name.
        
        Args:
            dataset_name: Name from TRAIN_DATASET_REGISTRY or COMPOSITE_DATASET_REGISTRY
            n_samples: Limit number of samples (None = all)
            format: Apply schema formatter
            apply_chat_template: If True, applies [INST]q[/INST]{answer} format
            tokenizer: Required if apply_chat_template is True
            augmentation: For composite datasets, cycle questions N times to reach larger sample counts (e.g., 700 * 6 = 4200)
            
        Returns:
            List of formatted data samples
        """
        # Check composite registry first (CAST, SRPS)
        if dataset_name in COMPOSITE_DATASET_REGISTRY:
            data = self._load_composite(dataset_name, n_samples, augmentation=augmentation)
            cfg = COMPOSITE_DATASET_REGISTRY[dataset_name]
        else:
            # Check regular registry
            if dataset_name not in TRAIN_DATASET_REGISTRY:
                all_available = list(TRAIN_DATASET_REGISTRY.keys()) + list(COMPOSITE_DATASET_REGISTRY.keys())
                raise DataLoadError(f"Unknown dataset: {dataset_name}. Available: {all_available}")
            
            cfg = TRAIN_DATASET_REGISTRY[dataset_name]
            
            # Read from local file
            data = read_file(cfg.path)
            
            # Format if requested
            if format and cfg.schema in FORMATTERS:
                data = FORMATTERS[cfg.schema](data)

        # Limit samples
        if n_samples is not None and n_samples < len(data):
            data = random.sample(data, n_samples)
        # Apply chat template if requested
        for d in data:
            q = d.get("question", "")
            if apply_chat_template and tokenizer is not None:
                if not q:
                    d[cfg.target_key] = build_chat_input(
                        tokenizer, 
                        d[cfg.target_key], 
                        add_generation_prompt=True)
                    if cfg.contrast_key is not None:
                        d[cfg.contrast_key] = build_chat_input(
                            tokenizer, 
                            d[cfg.contrast_key], 
                            add_generation_prompt=True
                        ) if d.get(cfg.contrast_key) else ""
                else:
                    # Apply chat template to question only.
                    # add_gen=True for schemas where target/contrast are response-only completions
                    # (no trailing space needed between template and response text).
                    add_gen = (cfg.schema in {
                        "concept500_reps", "cais_mask", "cais_mask_reps", "cais_mask_flas",
                        "ifeval", "ifeval_reps", "ifeval_flas",
                        "toxic", "sycophancy_personas",
                        "cast_combined", "evil", "deception",
                    })
                    base = build_chat_input(tokenizer, q, add_generation_prompt=add_gen)
                    sep = "" if add_gen else " "

                    # Update target/contrast prompts: prepend templated question
                    target_val = d.get(cfg.target_key, "")
                    d[cfg.target_key] = f"{base}{sep}{target_val}"

                    if cfg.contrast_key is not None:
                        contrast_val = d.get(cfg.contrast_key, "")
                        d[cfg.contrast_key] = f"{base}{sep}{contrast_val}"

                    # NOTE: target_response / contrast_response are intentionally left
                    # as raw response-only strings (no template prepended). FLAS reads
                    # them raw because it applies the chat template internally; REPS
                    # reads target_data / contrast_data (correct_prompt / false_prompt)
                    # which are already templated above.

            else:
                # No chat template - concatenate question + answer as-is
                if q:
                    target_val = d.get(cfg.target_key, "")
                    d[cfg.target_key] = f"{q} {target_val}"
                    if cfg.contrast_key is not None:
                        contrast_val = d.get(cfg.contrast_key, "")
                        d[cfg.contrast_key] = f"{q} {contrast_val}"

                    # target_response / contrast_response stay as raw response strings.

        return data
    
    def _load_composite(
        self,
        dataset_name: str,
        n_samples: Optional[int] = None,
        augmentation: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Load a composite dataset that combines multiple files.
        
        Used for CAST (response + questions) and SRPS (roleplay + questions).
        
        Args:
            dataset_name: Composite dataset name
            n_samples: Limit number of samples
            augmentation: For CAST, cycle questions N times to reach larger counts
        """
        cfg = COMPOSITE_DATASET_REGISTRY[dataset_name]
        
        # Read both files
        response_data = read_file(cfg.response_path)
        question_data = read_file(cfg.question_path)
        
        # Apply composite formatter, passing augmentation if supported
        if cfg.schema in COMPOSITE_FORMATTERS:
            formatter = COMPOSITE_FORMATTERS[cfg.schema]
            # Try to pass augmentation; fall back if formatter doesn't support it
            try:
                data = formatter(response_data, question_data, augmentation=augmentation)
            except TypeError:
                # Formatter doesn't support augmentation
                data = formatter(response_data, question_data)
        else:
            raise DataLoadError(f"Unknown composite schema: {cfg.schema}")
        
        # Limit samples
        if n_samples is not None and n_samples < len(data):
            idx = np.random.choice(len(data), n_samples, replace=False)
            data = [data[i] for i in idx]
        
        logger.info(f"Loaded {len(data)} composite samples from {dataset_name} (augmentation={augmentation})")
        return data
    
    def get_config(self, dataset_name: str) -> Union[TrainDatasetConfig, CompositeDatasetConfig]:
        """Get dataset config (supports both regular and composite)."""
        if dataset_name in COMPOSITE_DATASET_REGISTRY:
            return COMPOSITE_DATASET_REGISTRY[dataset_name]
        if dataset_name in TRAIN_DATASET_REGISTRY:
            return TRAIN_DATASET_REGISTRY[dataset_name]
        raise DataLoadError(f"Unknown dataset: {dataset_name}")

    def list_datasets(self) -> List[str]:
        """List all available training datasets (regular + composite)."""
        return list(TRAIN_DATASET_REGISTRY.keys()) + list(COMPOSITE_DATASET_REGISTRY.keys())




class EvalDataLoader:
    """
    Loader for test/evaluation datasets.
    
    Usage:
        loader = EvalDataLoader()
        
        # Load by friendly name
        data = loader.load("csqa")
        
        # Get config for prompt template and evaluator
        cfg = loader.get_config("csqa")
    """
    
    def __init__(self, root: Path = None) -> None:
        """Initialize loader. Root is ignored (paths come from registry)."""
        pass  # Root is determined by registry

    def load(
        self,
        dataset_name: str,
        n_samples: Optional[int] = None,
        format: bool = True,
        apply_chat_template: bool = False,
        tokenizer = None,
    ) -> List[Dict[str, Any]]:
        """
        Load a test dataset by friendly name.
        
        Args:
            dataset_name: Name from TEST_DATASET_REGISTRY (e.g., "csqa")
            n_samples: Limit number of samples (None = all)
            format: Apply schema formatter
            apply_chat_template: If True, applies [INST]q[/INST]{cfg.prefix} format
            tokenizer: Required if apply_chat_template is True
            
        Returns:
            List of formatted data samples
        """
        if dataset_name not in TEST_DATASET_REGISTRY:
            available = list(TEST_DATASET_REGISTRY.keys())
            raise DataLoadError(f"Unknown dataset: {dataset_name}. Available: {available}")
        
        cfg = TEST_DATASET_REGISTRY[dataset_name]
        
        # Read from local file
        data = read_file(cfg.path)
        
        # Format if requested
        if format and cfg.schema in FORMATTERS:
            data = FORMATTERS[cfg.schema](data)

        # Limit samples
        if n_samples is not None:
            if 0 <= n_samples < len(data):
                data = random.sample(data, n_samples)
            elif n_samples < 0:
                data = data[n_samples:]
        
        # Apply chat template if requested
        if apply_chat_template and tokenizer is not None:
            data = self._apply_test_template(data, tokenizer, cfg.prefix)
        
        logger.info(f"Loaded {len(data)} samples from {dataset_name}")
        return data
    
    def _apply_test_template(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        prefix: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Apply chat template for testing: [INST] question [/INST] {prefix}
        """
        for d in data:
            q = d.get("question", "")
            if not q:
                continue
            
            # Apply chat template + prefix
            base = build_chat_input(tokenizer, q, add_generation_prompt=True)
            d["question"] = f"{base}{prefix}"
        
        return data
    
    def get_config(self, dataset_name: str) -> TestDatasetConfig:
        """Get dataset config."""
        if dataset_name not in TEST_DATASET_REGISTRY:
            raise DataLoadError(f"Unknown dataset: {dataset_name}")
        return TEST_DATASET_REGISTRY[dataset_name]
    
    def list_datasets(self) -> List[str]:
        """List all available test datasets."""
        return list(TEST_DATASET_REGISTRY.keys())

