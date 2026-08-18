import pandas as pd
import pytest
from pathlib import Path
from Steering.data.readers import read_file, read_parquet, READERS

def test_parquet_reader(tmp_path):
    """Test reading a parquet file."""
    # Create a temporary parquet file
    data = [
        {"col1": 1, "col2": "a"},
        {"col1": 2, "col2": "b"},
        {"col1": 3, "col2": "c"},
    ]
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path)
    
    # Read it back using read_file (auto-detection)
    loaded_data = read_file(parquet_path)
    
    # Assertions
    assert len(loaded_data) == 3
    assert loaded_data == data

def test_parquet_reader_direct(tmp_path):
    """Test reading a parquet file directly with read_parquet."""
    # Create a temporary parquet file
    data = [
        {"val": 10},
        {"val": 20}
    ]
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_direct.parquet"
    df.to_parquet(parquet_path)
    
    # Read it back using read_parquet
    loaded_data = read_parquet(parquet_path)
    
    # Assertions
    assert len(loaded_data) == 2
    assert loaded_data == data

def test_reader_registration():
    """Ensure .parquet is registered."""
    assert ".parquet" in READERS
    assert READERS[".parquet"] == read_parquet
