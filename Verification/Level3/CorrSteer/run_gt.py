"""
Level 3 GT Runner: CorrSteer
Runs the actual CorrSteer CLI pipeline from Code/CorrSteer/ and saves
comprehensive diagnostics for comparison with our Steering/ pipeline.

EXACT GT parameters (from CorrConfig defaults + model_config):
  model:   google/gemma-2-2b-it (gemma2b in config.py)
  task:    mmlu (cais/mmlu from HuggingFace)
  layer:   13 (single layer for fast L3)
  scale:   1.0
  pool:    max
  topk:    20
  pos:     True (positive-only features)
  batch:   8 (from model_config.gemma2b.batch_size)
  max_new_tokens: 1 (MMLU select type)
  dtype:   bfloat16
  seed:    42

Saved outputs (one folder per run):
  Results/corrsteer_gt_{n_samples}_{seed}/
    ├── gemma2b_mmlu_13_corr.json           # Raw GT output (from train.py)
    ├── gemma2b_mmlu_13_corr_accuracy.json   # Evaluation accuracy
    ├── gt_diagnostics.json                  # ★ Full diagnostics for comparison
    ├── corrsteer_selected_sparse.pt         # Sparse vector (selected feature only)
    ├── corrsteer_selected_dense.pt          # Dense vector (selected feature only)
    ├── corrsteer_topk_sparse.pt             # Sparse vector (all top-K features)
    └── corrsteer_topk_dense.pt              # Dense vector (all top-K features)

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    python Verification/Level3/CorrSteer/run_gt.py [--n_samples 100] [--seed 42]

    # To re-extract diagnostics from an existing run (skip train.py):
    python Verification/Level3/CorrSteer/run_gt.py --n_samples 100 --extract_only
"""

import subprocess
import sys
import os
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
GT_DIR = ROOT / "Code" / "CorrSteer"

# ============================================================================
# Defaults — matching GT CorrConfig + run_gt params
# ============================================================================
LAYER = 13
TOP_K = 20
SCALE = 1.0
POOL = "max"
POS = True
BATCH_SIZE = 8
VAL_LIMIT = 30
TEST_LIMIT = 30


def extract_diagnostics(output_dir: Path, layer: int = LAYER, topk: int = TOP_K):
    """
    Extract comprehensive diagnostics from the GT train.py output JSON.

    Saves:
      - gt_diagnostics.json : all top-K feature indices, coefficients,
        correlations, frequencies, stats; selected feature; config.
      - corrsteer_selected_{sparse,dense}.pt : vectors for selected feature
      - corrsteer_topk_{sparse,dense}.pt    : vectors for all top-K features
    """
    import torch

    results_file = output_dir / f"gemma2b_mmlu_{layer}_corr.json"
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    layer_str = str(layer)
    layer_results = data["results"][layer_str]
    selected = layer_results["selected"]
    top_positive = layer_results.get("top_positive", [])
    top_negative = layer_results.get("top_negative", [])

    n_samples = data.get("samples", "unknown")
    pool = data.get("pool", POOL)

    # ---- Print top-K summary ----
    print(f"\n  n_samples: {n_samples}, pool: {pool}, topk: {len(top_positive)}")
    print(f"  Selected: idx={selected['feature_index']}, "
          f"coeff={selected['coefficient']:.6f}, corr={selected['correlation']:.6f}")
    print(f"\n  Top-{len(top_positive)} positive features:")
    for i, feat in enumerate(top_positive):
        print(f"    {i:>2}: idx={feat['feature_index']:>5}, "
              f"coeff={feat['coefficient']:>10.6f}, "
              f"corr={feat['correlation']:>10.6f}, "
              f"freq={feat['frequency']:>6.1f}%")

    # ---- Load SAE for vector construction (SAE only, no model needed) ----
    from sae_lens import SAE
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{layer}/width_16k/canonical",
    )
    n_latents = sae.cfg.d_sae

    # ---- Build vectors for selected feature ----
    sel_sparse = torch.zeros(n_latents, dtype=torch.float32)
    sel_sparse[selected["feature_index"]] = selected["coefficient"]
    sel_dense = (sel_sparse @ sae.W_dec.float().cpu())

    # ---- Build vectors for ALL top-K features ----
    topk_sparse = torch.zeros(n_latents, dtype=torch.float32)
    for feat in top_positive:
        topk_sparse[feat["feature_index"]] = feat["coefficient"]
    topk_dense = (topk_sparse @ sae.W_dec.float().cpu())

    # ---- Save vectors ----
    torch.save(sel_sparse, output_dir / "corrsteer_selected_sparse.pt")
    torch.save(sel_dense, output_dir / "corrsteer_selected_dense.pt")
    torch.save(topk_sparse, output_dir / "corrsteer_topk_sparse.pt")
    torch.save(topk_dense, output_dir / "corrsteer_topk_dense.pt")

    # ---- Save diagnostics JSON ----
    # Read accuracy if available
    acc_file = output_dir / f"gemma2b_mmlu_{layer}_corr_accuracy.json"
    accuracy = None
    if acc_file.exists():
        with open(acc_file) as f:
            acc_data = json.load(f)
        accuracy = acc_data.get("accuracy")

    diagnostics = {
        "layer": layer,
        "n_samples": n_samples,
        "pool": pool,
        "topk": len(top_positive),
        "scale": SCALE,
        "pos_only": POS,
        "seed": 42,
        "selected": selected,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "accuracy": accuracy,
        "feature_indices": [f["feature_index"] for f in top_positive],
        "feature_coefficients": [f["coefficient"] for f in top_positive],
        "feature_correlations": [f["correlation"] for f in top_positive],
        "feature_frequencies": [f["frequency"] for f in top_positive],
        # Vector norms for quick sanity
        "selected_dense_norm": float(sel_dense.norm()),
        "topk_dense_norm": float(topk_dense.norm()),
        "topk_nonzero_count": int((topk_sparse != 0).sum()),
    }

    diag_path = output_dir / "gt_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n  Saved diagnostics → {diag_path}")
    print(f"  Saved selected vectors (sparse + dense)")
    print(f"  Saved top-K vectors (sparse + dense)")
    print(f"  Accuracy: {accuracy}")
    print(f"  Selected dense norm: {diagnostics['selected_dense_norm']:.4f}")
    print(f"  Top-K dense norm:    {diagnostics['topk_dense_norm']:.4f}")

    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="CorrSteer GT Runner")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of MMLU training samples (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--topk", type=int, default=TOP_K)
    parser.add_argument("--val_limit", type=int, default=VAL_LIMIT)
    parser.add_argument("--test_limit", type=int, default=TEST_LIMIT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--extract_only", action="store_true",
                        help="Skip train.py, just re-extract diagnostics from existing results")
    args = parser.parse_args()

    output_base = ROOT / "Results" / "corrsteer_gt"
    # The GT train.py appends _{seed} to the output_dir
    actual_output_dir = ROOT / "Results" / f"corrsteer_gt_{args.n_samples}_{args.seed}"

    print("=" * 60)
    print("Level 3 GT: CorrSteer (Code/CorrSteer/ pipeline)")
    print("=" * 60)
    print(f"  n_samples: {args.n_samples}")
    print(f"  seed:      {args.seed}")
    print(f"  layer:     {args.layer}")
    print(f"  topk:      {args.topk}")
    print(f"  output:    {actual_output_dir}")

    if not args.extract_only:
        # ---- Step 1: Run GT train.py ----
        os.makedirs(output_base, exist_ok=True)

        # GT train.py appends _{seed} to output_dir internally
        # So we pass the base name and let it create the actual dir
        # BUT: the output_dir naming depends on what train.py does.
        # From the existing runs: output_dir="Results/corrsteer_gt" → "corrsteer_gt_42"
        # We want: output_dir → "corrsteer_gt_{n_samples}" → "corrsteer_gt_{n_samples}_{seed}"
        gt_output_arg = str(ROOT / "Results" / f"corrsteer_gt_{args.n_samples}")

        cmd = [
            sys.executable, "train.py", "train",
            f"--model=gemma2b",
            f"--task=mmlu",
            f"--layer={args.layer}",
            "--eval",
            f"--limit={args.val_limit}",
            f"--num_samples={args.n_samples}",
            f"--test_limit={args.test_limit}",
            f"--topk={args.topk}",
            f"--scale={SCALE}",
            f"--pool={POOL}",
            f"--pos={POS}",
            f"--batch_size={args.batch_size}",
            f"--seed={args.seed}",
            f"--output_dir={gt_output_arg}",
        ]

        print(f"\n[1] Running GT CLI:")
        print(f"    {' '.join(cmd)}")
        print(f"    cwd: {GT_DIR}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        result = subprocess.run(
            cmd, cwd=str(GT_DIR), capture_output=False, text=True, env=env,
        )

        if result.returncode != 0:
            print(f"\n[ERROR] GT pipeline exited with code {result.returncode}")
            sys.exit(1)

    # ---- Step 2: Check outputs ----
    if not actual_output_dir.exists():
        print(f"\n[ERROR] Output dir not found: {actual_output_dir}")
        print(f"  Check if train.py uses a different naming scheme.")
        # Try fallback naming (train.py may use base_{seed})
        fallback = ROOT / "Results" / f"corrsteer_gt_{args.seed}"
        if fallback.exists():
            print(f"  Found fallback: {fallback}")
            actual_output_dir = fallback
        else:
            sys.exit(1)

    print(f"\n[2] Outputs in {actual_output_dir}:")
    for f in sorted(actual_output_dir.iterdir()):
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")

    # ---- Step 3: Extract diagnostics ----
    print(f"\n[3] Extracting diagnostics...")
    extract_diagnostics(actual_output_dir, layer=args.layer, topk=args.topk)

    print("\nDone!")


if __name__ == "__main__":
    main()
