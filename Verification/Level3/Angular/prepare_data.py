
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Use absolute path to ensure correct file location regardless of CWD
BENCHMARK_ROOT = "/home/aiotlab/mnt/hoplt/Benchmark"

def prepare_data():
    base_dir = os.path.join(BENCHMARK_ROOT, "TrainDataset/behaviour/refusal")
    advbench_path = os.path.join(base_dir, "AdvBench/harmful_behaviors.csv")
    alpaca_path = os.path.join(base_dir, "CAST/alpaca.json")
    
    # Check if files exist
    if not os.path.exists(advbench_path):
        print(f"Error: {advbench_path} not found.")
        return
    if not os.path.exists(alpaca_path):
        print(f"Error: {alpaca_path} not found.")
        return

    output_dir = os.path.join(base_dir, "AdvBench")
    output_alpaca_dir = os.path.join(base_dir, "CAST")
    
    # 1. Split AdvBench (Harmful)
    print(f"Loading {advbench_path}...")
    df = pd.read_csv(advbench_path)
    # Match reference: instructions = dataset["goal"].tolist()
    # train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    # The CSV has column "goal".
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_out = os.path.join(output_dir, "train.csv")
    test_out = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    
    print(f"AdvBench Train: {len(train_df)} -> {train_out}")
    print(f"AdvBench Test:  {len(test_df)} -> {test_out}")
    
    # 2. Split Alpaca (Harmless)
    print(f"Loading {alpaca_path}...")
    with open(alpaca_path, 'r') as f:
        alpaca_data = json.load(f)
    
    # Alpaca data is list of dicts {"question": ...}
    # Reference uses instructions list and splits it.
    
    train_list, test_list = train_test_split(alpaca_data, test_size=0.2, random_state=42)
    
    # Take first 512 for angular contrast (matching reference limit)
    train_set = train_list[:512]
    # No need to save test set unless we test on harmless data (reference doesn't seem to test on harmless for ASR)
    
    print(f"Alpaca Train (subset): {len(train_set)} (from {len(train_list)})")
    
    train_out_al = os.path.join(output_alpaca_dir, "alpaca_train.json")
    
    with open(train_out_al, 'w') as f:
        json.dump(train_set, f, indent=2)
        
    print(f"Saved Alpaca Train to {train_out_al}")

if __name__ == "__main__":
    prepare_data()
