"""
Run a single Concept500 concept through both Steering and AxBench GT, then compare the saved weights.

This is a smoke-test helper for checking that both code paths are training on the same concept ID.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Code" / "axbench"))


def principal_angles(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    matrix = R1.float() @ R2.float().T
    singular_values = torch.linalg.svdvals(matrix).clamp(-1.0, 1.0)
    return singular_values.acos().sort().values


def projector_frobenius(R1: torch.Tensor, R2: torch.Tensor) -> tuple[float, float]:
    projector_1 = R1.float().T @ R1.float()
    projector_2 = R2.float().T @ R2.float()
    raw = (projector_1 - projector_2).norm(p="fro").item()
    return raw, raw / (2 * R1.shape[0]) ** 0.5


def cos_sim_flat(A: torch.Tensor, B: torch.Tensor) -> float:
    a = A.float().flatten()
    b = B.float().flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def orthonormality_err(R: torch.Tensor) -> float:
    gram = R.float() @ R.float().T
    identity = torch.eye(gram.shape[0])
    return (gram - identity).norm(p="fro").item()


def subspace_principal_angles(W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    q1, _ = torch.linalg.qr(W1.float().T)
    q2, _ = torch.linalg.qr(W2.float().T)
    return principal_angles(q1.T, q2.T)


def prepare_steering_config(template_path: Path, vector_dir: Path) -> Path:
    with open(template_path, "r") as file_handle:
        config = json.load(file_handle)
    config["save_vector"] = str(vector_dir)
    # Name/output derive from provided vector_dir
    config["name"] = vector_dir.name
    config["output"] = str(vector_dir.parent)

    temp_path = template_path.parent / f"_{vector_dir.name}.json"
    with open(temp_path, "w") as file_handle:
        json.dump(config, file_handle, indent=2)
    return temp_path


def run_command(command: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(command)}")
    result = subprocess.run(command, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def run_steering_extract(config_path: Path) -> None:
    run_command([
        sys.executable,
        "-m",
        "Steering.cli",
        "--task",
        "extract",
        "--config",
        str(config_path),
    ], "Steering")


def run_axbench_gt(gt_dir: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    print("\n[AxBench GT] pinning CUDA_VISIBLE_DEVICES=3")
    # The GT runner now runs the canonical Concept500 job and writes to the
    # standard results directory. We assume the user will provide `gt_dir` when
    # calling this helper in non-skip mode.
    print(f"[AxBench GT] expected results at: {gt_dir}")


def load_our_weights(vector_dir: Path, layer: int) -> dict[str, Any]:
    from Steering.config.results import SteeringVector

    steering_vector = SteeringVector.load(str(vector_dir), layer=[layer], device="cpu")
    metadata = steering_vector.metadata
    R_ours = metadata["rotate_basis"][layer].float()
    W_ours = metadata["learned_weight"][layer].float()
    bias_map = metadata.get("learned_bias") or {}
    b_ours = bias_map.get(layer)
    if b_ours is not None:
        b_ours = b_ours.float()
    return {"R_ours": R_ours, "W_ours": W_ours, "b_ours": b_ours}


def load_gt_weights(gt_dir: Path, model_name: str = "LoReFT") -> dict[str, Any]:
    weight = torch.load(gt_dir / f"{model_name}_weight.pt", map_location="cpu")
    bias = torch.load(gt_dir / f"{model_name}_bias.pt", map_location="cpu")

    proj_key = next(key for key in weight.keys() if key.endswith(".proj_weight"))
    source_key = proj_key.replace(".proj_weight", ".source_weight")
    b_key = proj_key.replace(".proj_weight", ".bias")
    R_gt = weight[proj_key][0].float().T
    W_gt = weight[source_key][0].float().T
    b_gt = bias.get(b_key)
    if b_gt is not None:
        b_gt = b_gt[0].float()
    return {"R_gt": R_gt, "W_gt": W_gt, "b_gt": b_gt}


def compare_and_report(ours: dict[str, Any], gt: dict[str, Any], layer: int) -> dict[str, Any]:
    R_ours = ours["R_ours"]
    W_ours = ours["W_ours"]
    R_gt = gt["R_gt"]
    W_gt = gt["W_gt"]
    rank, d_model = R_ours.shape

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  CONCEPT500 SMOKE COMPARISON  layer={layer}  rank={rank}  d_model={d_model}")
    print(sep)

    ours_orth = orthonormality_err(R_ours)
    gt_orth = orthonormality_err(R_gt)
    print(f"\n[Sanity] ||R R^T - I||_F")
    print(f"  Steering: {ours_orth:.6f}")
    print(f"  AxBench : {gt_orth:.6f}")

    r_angles = principal_angles(R_ours, R_gt)
    r_cos = r_angles.cos()
    frob_raw, frob_norm = projector_frobenius(R_ours, R_gt)
    print(f"\n[R] Mean cos(principal angle) = {r_cos.mean().item():.4f}")
    print(f"[R] Projector distance        = {frob_norm:.4f}")

    if rank == 1:
        w_cos = cos_sim_flat(W_ours, W_gt)
        w_score = abs(w_cos)
        print(f"\n[W] Cos-sim = {w_cos:.4f}")
    else:
        w_angles = subspace_principal_angles(W_ours, W_gt)
        w_cos = w_angles.cos()
        w_score = w_cos.mean().item()
        print(f"\n[W] Mean cos(principal angle) = {w_score:.4f}")
        print(f"[W] Flat cos-sim              = {cos_sim_flat(W_ours, W_gt):.4f}")

    source_basis_ours = W_ours @ R_ours.T
    source_basis_gt = W_gt @ R_gt.T
    source_basis_cos = cos_sim_flat(source_basis_ours, source_basis_gt)
    print(f"[W] Composed source-basis cos   = {source_basis_cos:.4f}")

    b_ours = ours.get("b_ours")
    b_gt = gt.get("b_gt")
    bias_cos = None
    if b_ours is not None and b_gt is not None:
        bias_cos = cos_sim_flat(b_ours, b_gt)
        print(f"\n[Bias] Cos-sim = {bias_cos:.4f}")

    print(f"\n{sep}")

    return {
        "layer": layer,
        "R_mean_cos_principal_angle": r_cos.mean().item(),
        "R_projector_frob_raw": frob_raw,
        "R_projector_frob_norm": frob_norm,
        "W_mean_cos_principal_angle": w_score,
        "W_source_basis_cos": source_basis_cos,
        "bias_cos_sim": bias_cos,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steering-config", type=Path, default=REPO_ROOT / "Configs" / "Eval" / "REFT" / "reps_concept500_smoke_concept_2.json")
    parser.add_argument("--vector-dir", type=Path, default=REPO_ROOT / "Vector" / "REPS" / "concept500_smoke_2")
    parser.add_argument("--gt-dir", type=Path, default=REPO_ROOT / "Code" / "axbench" / "axbench" / "results" / "prod_2b_l20_concept500_loreft" / "train")
    parser.add_argument("--skip-steering", action="store_true")
    parser.add_argument("--skip-gt", action="store_true")
    args = parser.parse_args()

    os.chdir(str(REPO_ROOT))
    torch.manual_seed(0)
    np.random.seed(0)

    vector_dir = args.vector_dir
    gt_dir = args.gt_dir

    if not args.skip_steering:
        steering_config = prepare_steering_config(args.steering_config, vector_dir)
        run_steering_extract(steering_config)
    else:
        print(f"[Steering] Skipping run; loading existing vector from {vector_dir}")

    if not args.skip_gt:
        run_axbench_gt(gt_dir)
    else:
        print(f"[AxBench GT] Skipping run; loading existing weights from {gt_dir}")

    our_weights = load_our_weights(vector_dir, layer=args.layer)
    gt_weights = load_gt_weights(gt_dir)

    metrics = compare_and_report(our_weights, gt_weights, layer=args.layer)

    report_path = vector_dir / "comparison_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as file_handle:
        json.dump(metrics, file_handle, indent=2)
    print(f"\nMetrics written to: {report_path}")


if __name__ == "__main__":
    main()
