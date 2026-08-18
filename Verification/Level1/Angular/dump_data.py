import sys
import os
import json
import itertools

# Add reference code to path
sys.path.append("/home/aiotlab/mnt/hoplt/Benchmark/Code/angular-steering/pytorch_pure")

from utils import get_harmful_instructions, get_harmless_instructions

def main():
    print("Loading reference data...")
    # Get raw data from reference utils (includes train_test_split logic)
    harmful_train, _ = get_harmful_instructions()
    harmless_train, _ = get_harmless_instructions()

    print(f"Harmful Train: {len(harmful_train)}")
    print(f"Harmless Train: {len(harmless_train)}")

    # Limit to N_SAMPLES for validation (e.g., 20 or all)
    # Reference extractor uses --n-samples argument.
    # We should dump enough data. Let's dump the first 100 for validation.
    N_SAMPLES = 100
    harmful_train = harmful_train[:N_SAMPLES]
    harmless_train = harmless_train[:N_SAMPLES]
    
    # Pair them up (harmless cycled)
    paired_data = []
    harmless_cycler = itertools.cycle(harmless_train)
    
    for harmful, harmless in zip(harmful_train, harmless_cycler):
        paired_data.append({
            "harmful": harmful,
            "harmless": harmless
        })
        
    output_path = "/home/aiotlab/mnt/hoplt/Benchmark/Verification/Level1/Angular/data.json"
    with open(output_path, "w") as f:
        json.dump(paired_data, f, indent=2)
        
    print(f"Saved {len(paired_data)} paired samples to {output_path}")

if __name__ == "__main__":
    main()
