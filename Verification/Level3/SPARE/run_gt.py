"""
Level 3 GT Runner: SPARE
Runs the actual SPARE pipeline from Code/SPARE/ via CLI subprocess calls.

Pipeline steps (all via subprocess, matching CorrSteer pattern):
  1. Run setup_gemma2b.sh to generate cache data (grouped_prompts,
     grouped_activations, mutual_information) if not already cached.
  2. Run scripts/run_spare.py to evaluate SPARE steering.
  3. Run extract_vectors.py (in Code/SPARE/ context) to extract and save
     sparse/dense vectors + diagnostics from the cached data.

EXACT GT parameters (from demo.py get_gemma_spare + scripts/run_spare.py):
  model_path:  google/gemma-2-2b-it
  data_name:   nqswap
  layer_ids:   13 14 15 16
  edit_degree:         2.0
  select_topk_proportion: 0.07
  seed:        42
  hiddens_name: grouped_activations
  mutual_information_save_name: mutual_information
  k_shot:      3
  max_new_tokens: 12
  do_sample:   False
  run_use_parameter: True
  run_use_context:   True

Output: Results/l3_spare_gt_{n_samples}_{seed}/

Usage:
    cd <project_root>
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    python Verification/Level3/SPARE/run_gt.py [--n_samples 1000] [--seed 42]
"""

import subprocess
import sys
import os
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
GT_DIR = ROOT / "Code" / "SPARE"
EXTRACT_SCRIPT = Path(__file__).parent / "extract_vectors.py"

# ============================================================================
# Defaults — matching GT run_spare.py params
# ============================================================================
MODEL_PATH = "google/gemma-2-2b-it"
DATA_NAME = "nqswap"
LAYER_IDS = [13, 14, 15, 16]
EDIT_DEGREE = 2.0
SELECT_TOPK_PROPORTION = 0.07
SEED = 42
HIDDENS_NAME = "grouped_activations"
MI_SAVE_NAME = "mutual_information"


def _make_env():
    """Build subprocess env with PYTHONPATH pointing to Code/SPARE."""
    env = os.environ.copy()
    env["PROJ_DIR"] = str(GT_DIR)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{GT_DIR}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(GT_DIR)
    return env


def _check_cache_exists(model_name: str, layer_ids: list[int]) -> bool:
    """Check if SPARE cache data required for vector extraction exists."""
    cache_dir = GT_DIR / "cache_data" / model_name
    grouped_dir = cache_dir / HIDDENS_NAME
    mi_dir = cache_dir / MI_SAVE_NAME
    if not grouped_dir.exists() or not mi_dir.exists():
        return False
    for layer in layer_ids:
        if not (grouped_dir / f"layer{layer}-use-parameter.pt").exists():
            return False
        if not (grouped_dir / f"layer{layer}-use-context.pt").exists():
            return False
        if not (mi_dir / f"layer-{layer} mi_expectation.pt").exists():
            return False
    return True


def run_setup(env: dict):
    """Step 1: Run setup_gemma2b.sh to generate cache data."""
    setup_script = GT_DIR / "scripts" / "setup_gemma2b.sh"
    print(f"\n[1] Running setup to generate cache data...")
    print(f"    Script: {setup_script}")
    print(f"    Working directory: {GT_DIR}")

    result = subprocess.run(
        ["bash", str(setup_script)],
        cwd=str(GT_DIR),
        capture_output=False,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] setup_gemma2b.sh exited with code {result.returncode}")
        sys.exit(1)
    print(f"    Setup complete.")


def run_evaluation(env: dict, n_samples: int, seed: int, layer_ids: list[int]):
    """Step 2: Run scripts/run_spare.py for evaluation."""
    cmd = [
        sys.executable, "scripts/run_spare.py",
        f"--model_path={MODEL_PATH}",
        f"--data_name={DATA_NAME}",
        "--layer_ids", *[str(l) for l in layer_ids],
        f"--edit_degree={EDIT_DEGREE}",
        f"--select_topk_proportion={SELECT_TOPK_PROPORTION}",
        f"--seed={seed}",
        f"--hiddens_name={HIDDENS_NAME}",
        f"--mutual_information_save_name={MI_SAVE_NAME}",
        "--run_use_parameter",
        "--run_use_context",
    ]

    print(f"\n[2] Running GT evaluation CLI:")
    print(f"    {' '.join(cmd)}")
    print(f"    Working directory: {GT_DIR}")

    result = subprocess.run(
        cmd,
        cwd=str(GT_DIR),
        capture_output=False,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] run_spare.py exited with code {result.returncode}")
        sys.exit(1)
    print(f"    Evaluation complete.")


def run_vector_extraction(env: dict, output_dir: Path, device: str, layer_ids: list[int]):
    """Step 3: Run extract_vectors.py to extract vectors from cached data."""
    cmd = [
        sys.executable, str(EXTRACT_SCRIPT),
        f"--model_name={os.path.basename(MODEL_PATH)}",
        "--layer_ids", *[str(l) for l in layer_ids],
        f"--edit_degree={EDIT_DEGREE}",
        f"--select_topk_proportion={SELECT_TOPK_PROPORTION}",
        f"--hiddens_name={HIDDENS_NAME}",
        f"--mi_save_name={MI_SAVE_NAME}",
        f"--output_dir={output_dir}",
        f"--gt_code_dir={GT_DIR}",
        f"--n_samples={output_dir.name.split('_')[-2]}",
        f"--device={device}",
    ]

    print(f"\n[3] Running vector extraction CLI:")
    print(f"    {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(GT_DIR),
        capture_output=False,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] extract_vectors.py exited with code {result.returncode}")
        sys.exit(1)
    print(f"    Vector extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="SPARE GT Runner")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of NQSwap samples (default: 1000)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip_setup", action="store_true",
                        help="Skip setup if cache data already exists")
    parser.add_argument("--extract_only", action="store_true",
                        help="Skip setup+eval, just re-extract vectors")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=LAYER_IDS,
        help="Layer(s) to extract/evaluate (default: 13 14 15 16)",
    )
    args = parser.parse_args()

    output_dir = ROOT / "Results" / f"l3_spare_gt_{args.n_samples}_{args.seed}"
    model_name = os.path.basename(MODEL_PATH)

    print("=" * 60)
    print("Level 3 GT: SPARE (Code/SPARE/ pipeline via CLI)")
    print("=" * 60)
    print(f"  n_samples:  {args.n_samples}")
    print(f"  seed:       {args.seed}")
    print(f"  layers:     {args.layers}")
    print(f"  output:     {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    env = _make_env()

    cache_ready = _check_cache_exists(model_name, args.layers)

    if not args.extract_only:
        # Step 1: Setup (generate cache data if needed)
        if args.skip_setup and cache_ready:
            print(f"\n[1] Cache data found for {model_name}, skipping setup.")
        else:
            run_setup(env)

        # Step 2: Run evaluation
        run_evaluation(env, args.n_samples, args.seed, args.layers)

        # Copy evaluation outputs
        spare_output_dir = GT_DIR / "spare_outputs"
        if spare_output_dir.exists():
            print(f"\n  Copying evaluation outputs...")
            for f in spare_output_dir.iterdir():
                if f.is_file():
                    print(f"    {f.name}")
                    shutil.copy2(f, output_dir / f.name)

    elif not cache_ready:
        print(f"\n[1] Cache data missing for {model_name}; using direct extraction fallback.")

    # Step 3: Extract vectors
    run_vector_extraction(env, output_dir, args.device, args.layers)

    # Summary
    print(f"\n[4] Outputs in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")

    print("\nDone!")


if __name__ == "__main__":
    main()
