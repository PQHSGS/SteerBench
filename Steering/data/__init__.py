"""
Data Module.

Provides data loading utilities:
- readers.py: File readers (JSON, JSONL, CSV)
- formatters.py: Schema formatters
- loader.py: DataLoader, EvalDataLoader classes

To add a new data format:
- Add reader to readers.py
- Add formatter to formatters.py
- Add to TRAIN_REGISTRY or TEST_REGISTRY in config/datasets.py
"""

from .loader import DataLoader, EvalDataLoader
from .readers import read_file, read_json, read_jsonl, read_csv, READERS
from .formatters import FORMATTERS
from .data_registry import TRAIN_DATASET_REGISTRY, TEST_DATASET_REGISTRY

__all__ = [
    "DataLoader",
    "EvalDataLoader",
    "read_file",
    "read_json",
    "read_jsonl",
    "read_csv",
    "READERS",
    "FORMATTERS",
]
