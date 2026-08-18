"""
Dataset Registries.

UNIFIED registry - all dataset paths, schemas, and configurations in ONE place.
No more TRAIN_REGISTRY / TRAIN_DATASET_REGISTRY separation.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "TrainDataset"
TEST_ROOT = PROJECT_ROOT / "TestDataset"


# =============================================================================
# Config Dataclasses
# =============================================================================

@dataclass(frozen=True)
class TrainDatasetConfig:
    """
    Unified training dataset config.
    
    Contains both file info AND semantic keys (target/contrast).
    When contrast_key is None, data is passed as raw dicts (no target/contrast split).
    """
    file: str = ""                      # Relative path from TRAIN_ROOT
    schema: str = ""                    # Format schema for data loading
    target_key: str = "correct_prompt"
    contrast_key: Optional[str] = "false_prompt"  # None = no contrastive split (raw dicts)
    
    @property
    def path(self) -> Path:
        """Full absolute path to dataset file."""
        return TRAIN_ROOT / self.file


@dataclass(frozen=True)
class CompositeDatasetConfig:
    """
    Composite dataset config for CAST/SRPS that combine multiple data sources.
    
    Used for:
    - CAST: combines behaviour_responses (agree/disagree prefixes) with questions (alpaca)
    - SRPS: combines roleplay prompts with reasoning questions
    """
    response_file: str          # File with response prefixes (CAST) or roleplay prompts (SRPS)
    question_file: str          # File with questions to combine
    schema: str                 # Format schema (cast_combined, srps_roleplay)
    target_key: str = "correct_prompt"
    contrast_key: str = "false_prompt"
    
    @property
    def response_path(self) -> Path:
        """Full path to response/roleplay file."""
        return TRAIN_ROOT / self.response_file
    
    @property
    def question_path(self) -> Path:
        """Full path to question file."""
        return TRAIN_ROOT / self.question_file


@dataclass(frozen=True)
class TestDatasetConfig:
    """
    Unified test dataset config.
    
    Contains both file info AND evaluation config.
    """
    file: str = ""                      # Relative path from TEST_ROOT
    schema: str = ""                    # Format schema for data loading
    evaluator: str = "logit"            # Evaluator type
    ground_truth_key: Optional[str] = None
    prefix: str = ""                    # Appended after chat template for logit matching
    
    @property
    def path(self) -> Path:
        """Full absolute path to dataset file."""
        return TEST_ROOT / self.file

