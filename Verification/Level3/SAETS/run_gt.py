"""
Level 3 GT Runner: SAE-TS
Runs the actual SAE-TS vector extraction from Code/SAE-TS/ via CLI
subprocess calls (matching the CorrSteer pattern).

Pipeline steps (all via subprocess):
  1. Run extract_vectors.py (in Code/SAE-TS/ context) to load steering
     configs and compute SAE, Optimised, Pinverse, and Rotation vectors
     using the GT analysis.py functions.
  2. Save all vectors + metadata JSON to output directory.

The SAE-TS pipeline uses pre-trained LinearAdapters (auto-downloaded from
HuggingFace) plus per-concept config files in steer_cfgs/gemma2/.

Output: Results/l3_saets_gt_{concept}_{seed}/

Usage:
    cd <project_root>
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    python Verification/Level3/SAETS/run_gt.py [--concept anger] [--seed 42]
    python Verification/Level3/SAETS/run_gt.py --all_concepts
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
GT_DIR = ROOT / "Code" / "SAE-TS"
EXTRACT_SCRIPT = Path(__file__).parent / "extract_vectors.py"

ALL_CONCEPTS = [
    "anger", "christian_evangelist", "conspiracy",
    "french", "london", "love",
    "praise", "want_to_die", "wedding",
]


def _make_env():
    """Build subprocess env with PYTHONPATH pointing to Code/SAE-TS/src."""
    env = os.environ.copy()
    src_dir = str(GT_DIR / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_dir}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_dir
    return env


def run_vector_extraction(env: dict, concept: str, output_dir: Path):
    """Run extract_vectors.py via subprocess to compute GT vectors."""
    cfg_path = str(GT_DIR / "steer_cfgs" / "gemma2" / concept)

    cmd = [
        sys.executable, str(EXTRACT_SCRIPT),
        f"--concept={concept}",
        f"--cfg_path={cfg_path}",
        f"--output_dir={output_dir}",
        f"--gt_src_dir={GT_DIR / 'src'}",
    ]

    print(f"\n  Running vector extraction CLI:")
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


def main():
    parser = argparse.ArgumentParser(description="SAE-TS GT Runner")
    parser.add_argument("--concept", type=str, default="anger",
                        choices=ALL_CONCEPTS,
                        help="Steering concept (default: anger)")
    parser.add_argument("--all_concepts", action="store_true",
                        help="Run all concepts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    concepts = ALL_CONCEPTS if args.all_concepts else [args.concept]

    print("=" * 60)
    print("Level 3 GT: SAE-TS (Code/SAE-TS/ pipeline via CLI)")
    print("=" * 60)
    print(f"  Concepts: {concepts}")
    print(f"  seed:     {args.seed}")

    env = _make_env()

    for concept in concepts:
        output_dir = ROOT / "Results" / f"l3_saets_gt_{concept}_{args.seed}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Processing concept: {concept}")
        print(f"  Output: {output_dir}")

        run_vector_extraction(env, concept, output_dir)

        # Summary
        print(f"\n  Outputs in {output_dir}:")
        for f in sorted(output_dir.iterdir()):
            print(f"    {f.name} ({f.stat().st_size:,} bytes)")

    print("\nDone!")


if __name__ == "__main__":
    main()
