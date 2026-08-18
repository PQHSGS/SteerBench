"""Run a strict FLAS Concept500 smoke comparison (GT vs Steering).

Goals:
1) Use Concept500 data only (same source parquet for both runs).
2) Optionally restrict to one concept_id and a tiny sample budget.
3) Run GT FLAS training code (Code/FLAS) and our FLAS extraction code.
4) Keep overlapping hyperparameters identical and record them.
5) Compare saved artifacts (state_dict key/shape parity + summary stats).

Usage:
    conda run -n sae_circuit python Verification/Level1/FLAS/compare_concept500_smoke.py \
        --device cuda:4 --concept-id 0 --max-samples 64 --gt-total-steps 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
FLAS_CODE_ROOT = REPO_ROOT / "Code" / "FLAS"
AXBENCH_C500_DIR = REPO_ROOT / "Code" / "axbench" / "axbench" / "concept500" / "prod_2b_l20_v1" / "generate"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_filter_concept500(
    data_dir: Path,
    concept_id: int | None,
    max_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    parquet_path = data_dir / "train_data.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Concept500 parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    pos_df = df[df["category"] == "positive"].reset_index(drop=True)
    if pos_df.empty:
        raise RuntimeError("No positive rows found in Concept500 train_data.parquet")

    all_cids = sorted(int(x) for x in pos_df["concept_id"].unique().tolist())
    if concept_id is None:
        concept_id = all_cids[0]

    if concept_id not in all_cids:
        raise ValueError(f"concept_id={concept_id} not found. Available examples: {all_cids[:10]}")

    one_df = pos_df[pos_df["concept_id"] == concept_id].reset_index(drop=True)
    if one_df.empty:
        raise RuntimeError(f"No rows for concept_id={concept_id}")

    n = min(int(max_samples), len(one_df))
    if n <= 0:
        raise ValueError("max_samples must be > 0")
    one_df = one_df.sample(n=n, random_state=seed).reset_index(drop=True)

    required_cols = {"input", "output", "output_concept", "concept_id", "category"}
    missing = required_cols - set(one_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in Concept500 data: {missing}")

    return one_df, int(concept_id)


def write_smoke_dataset(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_data.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, label: str = "run") -> None:
    print(f"\n[{label}] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def run_gt_flas_train(
    smoke_data_dir: Path,
    gt_out_root: Path,
    run_name: str,
    hyper: dict[str, Any],
) -> Path:
    gt_out_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "flas.train",
        "--data-dir",
        str(smoke_data_dir),
        "--model-id",
        str(hyper["model_id"]),
        "--layer",
        str(hyper["layer"]),
        "--output-dir",
        str(gt_out_root),
        "--run-name",
        run_name,
        "--num-blocks",
        str(hyper["num_blocks"]),
        "--batch-size",
        str(hyper["batch_size"]),
        "--grad-accum",
        str(hyper["grad_accum"]),
        "--lr",
        str(hyper["lr"]),
        "--enc-lr",
        str(hyper["enc_lr"]),
        "--div-weight",
        str(hyper["div_weight"]),
        "--total-steps",
        str(hyper["gt_total_steps"]),
        "--warmup-steps",
        "0",
        "--val-every",
        str(hyper["gt_val_every"]),
        "--patience",
        "2",
        "--n-val-samples",
        str(hyper["gt_n_val_samples"]),
        "--max-len",
        str(hyper["max_len"]),
        "--concept-max-len",
        str(hyper["concept_max_len"]),
        "--T-min",
        str(hyper["T_min"]),
        "--T-max",
        str(hyper["T_max"]),
        "--n-steps",
        str(hyper["n_steps"]),
        "--num-workers",
        "0",
    ]

    if hyper.get("no_gemma_init", False):
        cmd.append("--no-gemma-mlp-init")
    if hyper.get("disable_cross_attn", False):
        cmd.append("--disable-cross-attn")
    if hyper.get("disable_self_attn", False):
        cmd.append("--disable-self-attn")
    if hyper.get("disable_mlp", False):
        cmd.append("--disable-mlp")
    if hyper.get("unfreeze_concept_enc", False):
        cmd.append("--unfreeze-concept-enc")

    env = os.environ.copy()
    py_parts = [str(FLAS_CODE_ROOT / "src"), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([p for p in py_parts if p])

    run_cmd(cmd, cwd=FLAS_CODE_ROOT, env=env, label="GT-FLAS")

    run_dir = gt_out_root / run_name
    final_ckpt = run_dir / "final.pt"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"GT final checkpoint not found: {final_ckpt}")
    return final_ckpt


def run_ours_flas_extract(
    concept_df: pd.DataFrame,
    ours_vector_dir: Path,
    hyper: dict[str, Any],
) -> dict[str, torch.Tensor]:
    sys.path.insert(0, str(REPO_ROOT))
    from Steering.extractors.nonlinear import FLASExtractor

    class DummyModel:
        def __init__(self, device_str: str):
            self.cfg = SimpleNamespace(device=device_str)
            self.tokenizer = None

    target_data = concept_df["output"].astype(str).tolist()
    contrast_data = concept_df["input"].astype(str).tolist()
    concepts = concept_df["output_concept"].astype(str).tolist()
    concept_ids = concept_df["concept_id"].astype(int).tolist()

    dummy = DummyModel(str(hyper["device"]))
    extractor = FLASExtractor(
        model=dummy,
        layer=[int(hyper["layer"])],
        model_name=str(hyper["model_id"]),
        batch_size=int(hyper["batch_size"]),
        device=torch.device(str(hyper["device"])),
        hook_point="pre",
        flas_checkpoint_path=None,
        flas_num_blocks=int(hyper["num_blocks"]),
        flas_time_conditioned=True,
        flas_disable_cross_attn=bool(hyper["disable_cross_attn"]),
        flas_disable_self_attn=bool(hyper["disable_self_attn"]),
        flas_disable_mlp=bool(hyper["disable_mlp"]),
        flas_strict_load=True,
        flas_concept_encoder_layers=int(hyper["concept_encoder_layers"]),
        flas_train_concept_text=None,
        flas_train_lr=float(hyper["lr"]),
        flas_train_enc_lr=float(hyper["enc_lr"]),
        flas_train_div_weight=float(hyper["div_weight"]),
        flas_train_epochs=int(hyper["ours_epochs"]),
        flas_train_batch_size=int(hyper["batch_size"]),
        flas_train_grad_accum=int(hyper["grad_accum"]),
        flas_train_max_len=int(hyper["max_len"]),
        flas_train_concept_max_len=int(hyper["concept_max_len"]),
        flas_train_T_min=float(hyper["T_min"]),
        flas_train_T_max=float(hyper["T_max"]),
        flas_train_n_steps=int(hyper["n_steps"]),
        flas_train_seed=int(hyper["seed"]),
        flas_train_max_steps=int(hyper["ours_max_steps"]),
        flas_unfreeze_concept_enc=bool(hyper["unfreeze_concept_enc"]),
        flas_no_gemma_init=bool(hyper["no_gemma_init"]),
    )

    extractor.extract(
        target_data=target_data,
        contrast_data=contrast_data,
        flas_train_concepts=concepts,
        flas_train_concept_ids=concept_ids,
    )

    ours = extractor.metadata.get("flas_flow_fn") or extractor.metadata.get("flas_flow_fn_state_dict")
    if not isinstance(ours, dict):
        raise KeyError("Our extraction metadata missing flas_flow_fn state_dict")
    return ours


def tensor_summary(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    n_tensors = len(state_dict)
    n_params = sum(int(t.numel()) for t in state_dict.values())
    l2 = float(torch.sqrt(sum((t.float() ** 2).sum() for t in state_dict.values())).item())
    return {
        "n_tensors": n_tensors,
        "n_params": n_params,
        "l2_norm": l2,
    }


def load_gt_state_dict(final_ckpt: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(final_ckpt, map_location="cpu", weights_only=True)
    flow_fn = payload.get("flow_fn")
    if not isinstance(flow_fn, dict):
        raise KeyError(f"GT checkpoint missing flow_fn state_dict: {final_ckpt}")
    return flow_fn


def load_ours_state_dict(vector_dir: Path) -> dict[str, torch.Tensor]:
    artifact_path = vector_dir / "ours_flas_artifact.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing our artifact: {artifact_path}")

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    meta = artifact.get("metadata", {})
    ours = meta.get("flas_flow_fn") or meta.get("flas_flow_fn_state_dict")
    if not isinstance(ours, dict):
        raise KeyError("Our vector metadata missing flas_flow_fn state_dict")
    return ours


def compare_state_dicts(gt_sd: dict[str, torch.Tensor], ours_sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    gt_keys = set(gt_sd.keys())
    ours_keys = set(ours_sd.keys())
    shared = sorted(gt_keys & ours_keys)
    only_gt = sorted(gt_keys - ours_keys)
    only_ours = sorted(ours_keys - gt_keys)

    shape_mismatches = []
    for key in shared:
        if tuple(gt_sd[key].shape) != tuple(ours_sd[key].shape):
            shape_mismatches.append(key)

    same_shape_keys = [k for k in shared if k not in set(shape_mismatches)]
    mean_abs_diffs = []
    cosine_sims = []
    per_key_cosine = {}
    per_block_stats: dict[str, dict[str, list[float]]] = {}

    def block_group_name(key: str) -> str:
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
            return f"blocks.{parts[1]}"
        return "global"

    def cosine_similarity_1d(a: torch.Tensor, b: torch.Tensor) -> float:
        a64 = a.to(dtype=torch.float64)
        b64 = b.to(dtype=torch.float64)
        denom = float(torch.linalg.vector_norm(a64) * torch.linalg.vector_norm(b64))
        if denom == 0.0:
            return float("nan")
        cos = float(torch.dot(a64, b64).item() / denom)
        return float(max(-1.0, min(1.0, cos)))

    for key in same_shape_keys:
        gt_tensor = gt_sd[key].float().reshape(-1)
        ours_tensor = ours_sd[key].float().reshape(-1)
        diff = (gt_tensor - ours_tensor).abs().mean().item()
        mean_abs_diffs.append(diff)

        cos = cosine_similarity_1d(gt_tensor, ours_tensor)
        cosine_sims.append(cos)
        per_key_cosine[key] = cos

        group = block_group_name(key)
        bucket = per_block_stats.setdefault(group, {"cosine": [], "mean_abs_diff": []})
        bucket["cosine"].append(cos)
        bucket["mean_abs_diff"].append(diff)

    per_block_cosine = {}
    for group, stats in per_block_stats.items():
        valid_cosines = [x for x in stats["cosine"] if np.isfinite(x)]
        per_block_cosine[group] = {
            "n_keys": len(stats["cosine"]),
            "cosine_avg": float(np.mean(valid_cosines)) if valid_cosines else None,
            "cosine_min": float(np.min(valid_cosines)) if valid_cosines else None,
            "mean_abs_diff_avg": float(np.mean(stats["mean_abs_diff"])),
        }

    return {
        "gt_only_keys": only_gt,
        "ours_only_keys": only_ours,
        "shape_mismatches": shape_mismatches,
        "shared_key_count": len(shared),
        "same_shape_key_count": len(same_shape_keys),
        "mean_abs_diff_avg": float(np.mean(mean_abs_diffs)) if mean_abs_diffs else None,
        "mean_abs_diff_max": float(np.max(mean_abs_diffs)) if mean_abs_diffs else None,
        "cosine_similarity_avg": float(np.mean([x for x in cosine_sims if np.isfinite(x)])) if cosine_sims else None,
        "cosine_similarity_min": float(np.min([x for x in cosine_sims if np.isfinite(x)])) if cosine_sims else None,
        "per_key_cosine_similarity": per_key_cosine,
        "per_block_cosine_similarity": per_block_cosine,
    }


def build_hyperparams(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_id": "google/gemma-2-2b-it",
        "layer": int(args.layer),
        "device": str(args.device),
        "seed": int(args.seed),
        "num_blocks": 1,
        "concept_encoder_layers": 2,
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "lr": float(args.lr),
        "enc_lr": float(args.enc_lr),
        "div_weight": float(args.div_weight),
        "max_len": int(args.max_len),
        "concept_max_len": int(args.concept_max_len),
        "T_min": float(args.T_min),
        "T_max": float(args.T_max),
        "n_steps": int(args.n_steps),
        "disable_cross_attn": bool(args.disable_cross_attn),
        "disable_self_attn": bool(args.disable_self_attn),
        "disable_mlp": bool(args.disable_mlp),
        "no_gemma_init": bool(args.no_gemma_init),
        "unfreeze_concept_enc": bool(args.unfreeze_concept_enc),
        # GT-specific
        "gt_total_steps": int(args.gt_total_steps),
        "gt_val_every": int(args.gt_val_every),
        "gt_n_val_samples": int(args.gt_n_val_samples),
        # Ours-specific (mapped from same budget intent)
        "ours_epochs": int(args.ours_epochs),
        "ours_max_steps": int(args.ours_max_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict FLAS Concept500 smoke comparison (GT vs Steering)")
    parser.add_argument("--data-dir", type=Path, default=AXBENCH_C500_DIR)
    parser.add_argument("--device", type=str, default="cuda:4")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--concept-id", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--enc-lr", type=float, default=1e-5)
    parser.add_argument("--div-weight", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--concept-max-len", type=int, default=48)
    parser.add_argument("--T-min", type=float, default=0.5)
    parser.add_argument("--T-max", type=float, default=2.0)
    parser.add_argument("--n-steps", type=int, default=3)

    parser.add_argument("--gt-total-steps", type=int, default=8)
    parser.add_argument("--gt-val-every", type=int, default=4)
    parser.add_argument("--gt-n-val-samples", type=int, default=8)

    parser.add_argument("--ours-epochs", type=int, default=20)
    parser.add_argument("--ours-max-steps", type=int, default=8)

    parser.add_argument("--disable-cross-attn", action="store_true")
    parser.add_argument("--disable-self-attn", action="store_true")
    parser.add_argument("--disable-mlp", action="store_true")
    parser.add_argument("--no-gemma-init", action="store_true")
    parser.add_argument("--unfreeze-concept-enc", action="store_true")

    parser.add_argument("--skip-gt", action="store_true")
    parser.add_argument("--skip-ours", action="store_true")

    args = parser.parse_args()

    set_seed(int(args.seed))

    artifacts_root = REPO_ROOT / "Verification" / "Level1" / "FLAS" / "artifacts"
    smoke_data_dir = artifacts_root / "concept500_smoke_data"
    gt_out_root = artifacts_root / "gt_checkpoints"
    ours_vector_dir = artifacts_root / "ours_vector"

    concept_df, selected_cid = load_and_filter_concept500(
        data_dir=args.data_dir,
        concept_id=args.concept_id,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    write_smoke_dataset(concept_df, smoke_data_dir)

    hyper = build_hyperparams(args)

    run_name = f"flas_smoke_cid_{selected_cid}"
    gt_ckpt = gt_out_root / run_name / "final.pt"
    if not args.skip_gt:
        gt_ckpt = run_gt_flas_train(
            smoke_data_dir=smoke_data_dir,
            gt_out_root=gt_out_root,
            run_name=run_name,
            hyper=hyper,
        )
    elif not gt_ckpt.exists():
        raise FileNotFoundError(f"--skip-gt set but missing checkpoint: {gt_ckpt}")

    if not args.skip_ours:
        ours_sd = run_ours_flas_extract(
            concept_df=concept_df,
            ours_vector_dir=ours_vector_dir,
            hyper=hyper,
        )
    else:
        if not (ours_vector_dir / "ours_flas_artifact.pt").exists():
            raise FileNotFoundError(
                "--skip-ours requires an existing saved artifact at "
                f"{ours_vector_dir / 'ours_flas_artifact.pt'}"
            )
        ours_sd = load_ours_state_dict(ours_vector_dir)

    gt_sd = load_gt_state_dict(gt_ckpt)

    report = {
        "selected_concept_id": int(selected_cid),
        "n_samples": int(len(concept_df)),
        "smoke_data_dir": str(smoke_data_dir),
        "gt_checkpoint": str(gt_ckpt),
        "ours_vector_dir": str(ours_vector_dir),
        "hyperparameters": hyper,
        "gt_summary": tensor_summary(gt_sd),
        "ours_summary": tensor_summary(ours_sd),
        "state_dict_comparison": compare_state_dicts(gt_sd, ours_sd),
    }

    report_path = artifacts_root / "comparison_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("FLAS Concept500 smoke comparison finished")
    print("=" * 72)
    print(f"concept_id: {selected_cid}")
    print(f"n_samples : {len(concept_df)}")
    print(f"GT ckpt   : {gt_ckpt}")
    print(f"Our vector: {ours_vector_dir}")
    print(f"Report    : {report_path}")


if __name__ == "__main__":
    main()
