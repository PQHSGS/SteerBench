"""
SPARE Level 1 validation (Extractor parity).

Rules enforced:
- GT path is executed only via CLI (no direct GT algorithm imports here).
- Ours uses SPAREExtractor.extract().
- Compare vector/latents/indices from identical method hyperparameters.

Usage:
  unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
  PYTHONPATH=. python Verification/Level1/SPARE/validate_spare_l1.py --n_samples 1000 --layer 14
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Steering.data import DataLoader
from Steering.extractors.sae import SPAREExtractor
from Steering.utils import SAEWrapper
from sae_lens import SAE
from transformer_lens import HookedTransformer


MODEL_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
TOP_K_PROPORTION = 0.07
EDIT_DEGREE = 2.0


def _safe_load_tensor(path: Path, map_location: str = "cpu") -> torch.Tensor:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _jaccard(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def run_gt_cli(n_samples: int, seed: int, device: str, layers: List[int]) -> Path:
    cmd = [
        sys.executable,
        "Verification/Level3/SPARE/run_gt.py",
        f"--n_samples={n_samples}",
        f"--seed={seed}",
        f"--device={device}",
        "--layers",
        *[str(layer) for layer in layers],
        "--extract_only",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GT CLI failed with exit code {result.returncode}")
    return ROOT / "Results" / f"l3_spare_gt_{n_samples}_{seed}"


def run_ours_extract(
    layers: List[int], n_samples: int, device: str
) -> Tuple[SPAREExtractor, Dict[int, torch.Tensor]]:
    """Load all SAEs up-front then run a single multi-layer extraction pass."""
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
    sae_dict: Dict[int, Any] = {}
    for layer in layers:
        sae_raw, _, _ = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"layer_{layer}/width_16k/canonical",
        )
        sae_dict[layer] = SAEWrapper(sae_raw.to(device=device, dtype=torch.bfloat16))

    train_data = DataLoader().load("nqswap_train", n_samples, apply_chat_template=False)
    target_data = [d["correct_prompt"] for d in train_data]
    contrast_data = [d["false_prompt"] for d in train_data]

    extractor = SPAREExtractor(
        model=model,
        sae=sae_dict,
        layer=layers,
        top_k_proportion=TOP_K_PROPORTION,
        batch_size=1,
        device=device,
        hook_point=["post"],
    )
    vectors = extractor.extract(target_data=target_data, contrast_data=contrast_data)
    return extractor, vectors


def _compare_single_layer(
    gt_dir: Path,
    extractor: SPAREExtractor,
    vectors: Dict[int, torch.Tensor],
    layer: int,
) -> Dict[str, float]:
    """Compare extractor output vs GT artifacts for one layer."""
    diag_path = gt_dir / f"gt_diagnostics_L{layer}.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing GT diagnostics: {diag_path}")

    with open(diag_path, "r", encoding="utf-8") as f:
        diag = json.load(f)

    gt_pos = set(int(x) for x in diag.get("indices_pos", []))
    gt_neg = set(int(x) for x in diag.get("indices_neg", []))
    our_pos = set(int(x) for x in extractor.indices_pos[layer].detach().cpu().tolist())
    our_neg = set(int(x) for x in extractor.indices_neg[layer].detach().cpu().tolist())

    gt_sparse = _safe_load_tensor(gt_dir / f"spare_sparse_vector_L{layer}.pt")
    gt_dense  = _safe_load_tensor(gt_dir / f"spare_dense_vector_L{layer}.pt")
    gt_zc     = _safe_load_tensor(gt_dir / f"spare_zC_L{layer}.pt")
    gt_zm     = _safe_load_tensor(gt_dir / f"spare_zM_L{layer}.pt")

    our_sparse = extractor.sparse_latent[layer].detach().cpu()
    our_dense  = vectors[layer].detach().cpu()
    our_zc     = extractor.z_contextual[layer].detach().cpu()
    our_zm     = extractor.z_parametric[layer].detach().cpu()

    return {
        "jaccard_pos": _jaccard(gt_pos, our_pos),
        "jaccard_neg": _jaccard(gt_neg, our_neg),
        "jaccard_total": _jaccard(gt_pos | gt_neg, our_pos | our_neg),
        "cos_sparse": _cos(gt_sparse, our_sparse),
        "cos_dense": _cos(gt_dense, our_dense),
        "cos_zc": _cos(gt_zc, our_zc),
        "cos_zm": _cos(gt_zm, our_zm),
    }


def compare_l1(
    gt_dir: Path,
    extractor: SPAREExtractor,
    vectors: Dict[int, torch.Tensor],
    layers: List[int],
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Compare all requested layers; return (aggregate, per_layer) metrics."""
    per_layer: Dict[int, Dict[str, float]] = {}
    for layer in layers:
        per_layer[layer] = _compare_single_layer(gt_dir, extractor, vectors, layer)

    keys = list(next(iter(per_layer.values())).keys())
    aggregate = {k: float(np.mean([per_layer[l][k] for l in layers])) for k in keys}
    return aggregate, per_layer


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARE Level1 parity validation")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[14],
        help="Layer(s) to validate (e.g. --layers 13 14 15 16)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 64)
    print("SPARE Level 1 (Extractor parity) on Gemma-2-2B")
    print("=" * 64)
    print(f"n_samples={args.n_samples}, seed={args.seed}, layers={args.layers}, threshold={args.threshold}")
    print(f"GT hyperparams: top_k_proportion={TOP_K_PROPORTION}, edit_degree={EDIT_DEGREE}")

    gt_dir = run_gt_cli(args.n_samples, args.seed, args.device, args.layers)
    extractor, vectors = run_ours_extract(args.layers, args.n_samples, args.device)

    # Save our validated vector for Level 2 fast loading.
    out_vec = ROOT / "Results" / f"l1_spare_ours_vector_{args.n_samples}_{args.seed}.pt"
    extractor.save(str(out_vec))

    metrics, per_layer = compare_l1(gt_dir, extractor, vectors, args.layers)

    print("\nPer-layer metrics:")
    for layer, lm in per_layer.items():
        print(f"  Layer {layer}:")
        for k, v in lm.items():
            print(f"    {k}: {v:.6f}")

    print("\nAggregate metrics (mean across layers):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    checks = {
        "total index jaccard": metrics["jaccard_total"] >= args.threshold,
        "dense vector cosine": metrics["cos_dense"] >= args.threshold,
        "sparse latent cosine": metrics["cos_sparse"] >= args.threshold,
    }

    all_passed = True
    print("\nChecks:")
    for name, ok in checks.items():
        if not ok:
            all_passed = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    summary = {
        "level": 1,
        "method": "SPARE",
        "model": MODEL_NAME,
        "layers": args.layers,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "threshold": args.threshold,
        "gt_dir": str(gt_dir),
        "ours_vector_path": str(out_vec),
        "metrics": metrics,
        "per_layer_metrics": {str(l): v for l, v in per_layer.items()},
        "passed": all_passed,
    }
    out_summary = ROOT / "Results" / f"spare_l1_summary_{args.n_samples}_{args.seed}.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_summary}")
    print(f"Saved: {out_vec}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
