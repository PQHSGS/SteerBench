"""
File Readers.

Provides functions to read various data file formats.
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union


def read_json(path: Union[str, Path]) -> Union[List, Dict]:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSONL (JSON Lines) file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a CSV file as list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_parquet(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a Parquet file as list of dicts."""
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")

def read_tsv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a TSV (Tab-Separated Values) file as list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

# Extension → reader function
READERS = {
    ".json": read_json,
    ".jsonl": read_jsonl,
    ".csv": read_csv,
    ".parquet": read_parquet,
    ".tsv": read_tsv
}


def read_file(path: Union[str, Path]) -> Union[List, Dict]:
    """Read a data file, auto-detecting format from extension."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in READERS:
        raise ValueError(f"Unsupported format: {ext}. Supported: {list(READERS.keys())}")
    return READERS[ext](path)
