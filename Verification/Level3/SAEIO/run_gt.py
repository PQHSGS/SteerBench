"""
Level 3 GT Runner: SAEIO (saes-are-good-for-steering)

Runs the SAEIO GT pipeline via CLI subprocess calls to
Code/saes-are-good-for-steering/, matching the CorrSteer pattern.

Pipeline steps (all via subprocess):
  1. Run extract_vectors.py (in Code/saes-are-good-for-steering/ context)
     which performs the full pipeline:
     - Load HuggingFace model (google/gemma-2-2b) + SAE (layer 20, 16k)
     - Load sycophancy data → compute mean-of-max SAE activations
     - Filter active features (> threshold)
     - Compute logit lens via GT cache_logit_lens()
     - Compute output scores via GT get_output_score()
     - Select top-K features, build sparse + dense vectors
     - Optionally generate steered text via GT AmlifySAEHook

EXACT GT parameters (from src/output_score.py + src/steer.py):
  model:           google/gemma-2-2b
  SAE:             gemma-scope-2b-pt-res-canonical, layer_20/width_16k/canonical
  amp_factor_score: 10.0   (output_score.py default)
  amp_factor_steer: 1.2    (steer.py default)
  neutral_prompt:  "From my experience,"
  logit_lens_k:    20
  threshold:       0.0
  top_k:           5

Output: Results/l3_saeio_gt_{n_train}_{seed}/

Usage:
    cd <project_root>
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    python Verification/Level3/SAEIO/run_gt.py [--n_train 1000] [--device cuda:0]
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
GT_DIR = ROOT / "Code" / "saes-are-good-for-steering"
EXTRACT_SCRIPT = Path(__file__).parent / "extract_vectors.py"

# ============================================================================
# Defaults — matching GT src/output_score.py + src/steer.py params
# ============================================================================
LAYER = 20
AMP_FACTOR_SCORE = 10.0
AMP_FACTOR_STEER = 1.2
ACT_THRESHOLD = 0.0
LOGIT_LENS_K = 20
TOP_K = 5
NEUTRAL_PROMPT = "From my experience,"


def _make_env():
    """Build subprocess env with PYTHONPATH pointing to GT src/."""
    env = os.environ.copy()
    gt_src = str(GT_DIR / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{gt_src}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = gt_src
    return env


def run_extraction(env: dict, output_dir: Path, args):
    """Run extract_vectors.py to compute features, scores, and vectors."""
    cmd = [
        sys.executable, str(EXTRACT_SCRIPT),
        f"--n_train={args.n_train}",
        f"--top_k={args.top_k}",
        f"--layer={args.layer}",
        f"--threshold={args.threshold}",
        f"--amp_factor_score={AMP_FACTOR_SCORE}",
        f"--amp_factor_steer={AMP_FACTOR_STEER}",
        f"--logit_lens_k={LOGIT_LENS_K}",
        f"--neutral_prompt={NEUTRAL_PROMPT}",
        f"--device={args.device}",
        f"--output_dir={output_dir}",
        f"--project_root={ROOT}",
    ]

    if args.extract_only:
        cmd.append("--skip_generation")
    else:
        cmd.append(f"--n_gen_prefixes={args.n_gen_prefixes}")

    print(f"\n[1] Running extraction CLI:")
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
        print(f"\n[ERROR] extract_vectors.py exited with code {result.returncode}")
        sys.exit(1)
    print(f"    Extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="SAEIO GT Runner (CLI subprocess)")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--threshold", type=float, default=ACT_THRESHOLD)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--extract_only", action="store_true",
                        help="Only run extraction (skip generation)")
    parser.add_argument("--n_gen_prefixes", type=int, default=50,
                        help="Number of prefixes for generation (default: all 50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = ROOT / "Results" / f"l3_saeio_gt_{args.n_train}_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Level 3 GT: SAEIO (Code/saes-are-good-for-steering/ via CLI)")
    print("=" * 60)
    print(f"  n_train:  {args.n_train}")
    print(f"  top_k:    {args.top_k}")
    print(f"  layer:    {args.layer}")
    print(f"  device:   {args.device}")
    print(f"  output:   {output_dir}")

    env = _make_env()
    run_extraction(env, output_dir, args)

    # Summary
    print(f"\n[2] Outputs in {output_dir}:")
    for f in sorted(output_dir.iterdir()):
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
