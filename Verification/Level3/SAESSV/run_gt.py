"""
Level 3 GT Runner: SAE-SSV
Runs the actual SAE-SSV pipeline from Code/SAE-SSV/saessv-demo.py.

As per the README, the GT implementation is notebook-based (saessv-demo.ipynb),
converted to saessv-demo.py. The pipeline is:
  1. Load model + SAE
  2. Precompute latents for target/contrast data
  3. Feature selection (ANOVA-based) via LinearConceptExtractor
  4. Train SSV direction (train_falseness_ssv_with_hooks_and_lm)
  5. Test steering (test_falseness_ssv_with_hooks)

We run the full saessv-demo.py script end-to-end as a subprocess.
Since the script has hardcoded globals, we pre-set the data and call it directly.

Output: Results/l3_saessv_gt/
"""

import subprocess
import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
GT_DIR = ROOT / "Code" / "SAE-SSV"
OUTPUT_DIR = ROOT / "Results" / "l3_saessv_gt"


def main():
    print("=" * 60)
    print("Level 3 GT: SAE-SSV (using Code/SAE-SSV/saessv-demo.py)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # saessv-demo.py is a notebook-converted script that runs the full pipeline
    # when executed: load model → precompute latents → feature selection →
    # train SSV → test steering. It contains global code that runs end-to-end.
    #
    # The README says: "The implementation can be referred to saessv-demo.ipynb"
    # We run the .py version directly.
    cmd = [
        sys.executable, "saessv-demo.py",
    ]

    print(f"\n[1] Running GT script: {' '.join(cmd)}")
    print(f"    Working directory: {GT_DIR}")

    # Set environment to control output
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(
        cmd,
        cwd=str(GT_DIR),
        capture_output=False,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"\n[ERROR] saessv-demo.py exited with code {result.returncode}")
        sys.exit(1)

    print(f"\n[2] GT pipeline completed successfully")
    
    print("\n[3] Extracting and saving SAESSV dense and sparse vectors...")
    
    gt_output_dir = GT_DIR / "Vector" / "SSV" / "gemma2layer13"
    if not gt_output_dir.exists():
        print(f"\n[ERROR] GT output dir {gt_output_dir} not found")
        sys.exit(1)
        
    import shutil
    import torch
    import numpy as np
    
    results_file = gt_output_dir / "falseness_ssv_results.pt"
    dims_file = gt_output_dir / "political_important_dimensions.npy"
    
    if results_file.exists():
        shutil.copy2(results_file, OUTPUT_DIR / "falseness_ssv_results.pt")
    if dims_file.exists():
        shutil.copy2(dims_file, OUTPUT_DIR / "political_important_dimensions.npy")
        
    results = torch.load(results_file, map_location="cpu", weights_only=False)
    ssv_np = results["ssv"]
    
    sparse_vector = torch.from_numpy(ssv_np).float()
    
    sys.path.insert(0, str(ROOT))
    from Verification.shared_utils import load_model_and_sae
    
    _, sae, _ = load_model_and_sae(layer=13, device="cpu")
    
    dense_vector = sparse_vector @ sae.W_dec.cpu().float()
    
    torch.save(sparse_vector, OUTPUT_DIR / "saessv_sparse_vector.pt")
    torch.save(dense_vector, OUTPUT_DIR / "saessv_dense_vector.pt")
    
    important_dims = np.load(dims_file)
    indices_dict = {
        "layer": 13,
        "important_dims": important_dims.tolist()
    }
    with open(OUTPUT_DIR / "saessv_indices.json", "w") as f:
        json.dump(indices_dict, f)
        
    print(f"    Saved vectors to: {OUTPUT_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
