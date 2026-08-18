"""
Level 3 SPARE Comparison: GT vs Framework
Compares:
  1. SAE layer-wise feature indices and values
  2. Steering vector cosine similarity (sparse + dense) if saved
  3. Eval metrics (accuracy / EM)
"""
import json
import sys
import os
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_GT = ROOT / "Results" / "l3_spare_gt"
RESULTS_FW = ROOT / "Results" / "l3_spare"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-10))

def sep(title=""):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def compare_vectors():
    # ─── 1. Selected Features & Vectors per Layer ────────────────────────────
    layers = [13, 14, 15, 16]
    
    for layer in layers:
        sep(f"ANALYSIS FOR LAYER {layer}")
        
        gt_diag_file = RESULTS_GT / f"gt_diagnostics_L{layer}.json"
        if not gt_diag_file.exists():
            print(f"  [WARN] GT diagnostics for layer {layer} not found.")
            continue
            
        gt_diag = load_json(gt_diag_file)
        print(f"  GT Positive Indices : {len(gt_diag.get('indices_pos', []))} features")
        print(f"  GT Negative Indices : {len(gt_diag.get('indices_neg', []))} features")
        
        gt_sparse_file = RESULTS_GT / f"spare_sparse_vector_L{layer}.pt"
        gt_dense_file = RESULTS_GT / f"spare_dense_vector_L{layer}.pt"
        
        gt_sv = torch.load(gt_sparse_file, map_location="cpu") if gt_sparse_file.exists() else None
        gt_dv = torch.load(gt_dense_file, map_location="cpu") if gt_dense_file.exists() else None
        
        if gt_sv is not None:
            nnz = int((gt_sv != 0).sum())
            print(f"  GT sparse vector shape={gt_sv.shape}, nnz={nnz}, norm={float(gt_sv.norm()):.6f}")
            print(f"  GT dense vector shape={gt_dv.shape}, norm={float(gt_dv.norm()):.6f}")
        
        # Check FW Vectors
        fw_sparse_file = RESULTS_FW / f"spare_l3_validation_L{layer}_sparse.pt"  # Assumed naming if dumped
        fw_dense_file = RESULTS_FW / f"spare_l3_validation_L{layer}.pt"
        
        if fw_sparse_file.exists() and gt_sv is not None:
            fw_sv = torch.load(fw_sparse_file, map_location="cpu")
            sc = cosine(gt_sv, fw_sv)
            print(f"\n  FW sparse vector found: norm={float(fw_sv.norm()):.6f}")
            print(f"  Cosine (sparse):  {sc:.6f}  {'✅' if sc > 0.99 else '❌'}")
        else:
            print(f"\n  [INFO] FW sparse vector for layer {layer} not found.")
            
        if fw_dense_file.exists() and gt_dv is not None:
            fw_dv = torch.load(fw_dense_file, map_location="cpu")
            dc = cosine(gt_dv, fw_dv)
            print(f"  FW dense vector found: norm={float(fw_dv.norm()):.6f}")
            print(f"  Cosine (dense):   {dc:.6f}  {'✅' if dc > 0.99 else '❌'}")
            
def compare_evals():
    sep("3. ACCURACY / EXACT MATCH  (Baseline vs Steered)")
    
    fw_eval_files = sorted(RESULTS_FW.glob("eval_spare_l3_validation_*.json"))
    if not fw_eval_files:
        print("  [INFO] FW eval JSON not found.")
        return
        
    fw_eval_path = fw_eval_files[-1]
    fw_eval = load_json(fw_eval_path)
    fw_result = fw_eval.get("result", {})
    fw_acc  = fw_result.get("accuracy", 0.0)
    fw_base = fw_result.get("baseline_accuracy", 0.0)
    fw_delta = fw_result.get("delta", 0.0)
    fw_total = fw_result.get("total", 0)
    
    print(f"  FW steered accuracy : {fw_acc * 100:.2f}% ({fw_result.get('correct', 0)}/{fw_total})")
    print(f"  FW baseline accuracy: {fw_base * 100:.2f}%")
    print(f"  FW delta            : {fw_delta * 100:+.2f}%")
    
    # GT demo prints evaluation to stdout usually. If GT eval json was saved:
    gt_eval_files = sorted(RESULTS_GT.glob("evaluation_*.json"))
    if gt_eval_files:
        gt_eval = load_json(gt_eval_files[-1])
        # Add logic to parse SPARE GT if they output explicit scores as json
        print(f"\n  GT Eval File Found: {gt_eval_files[-1].name}")
    else:
        print("\n  [INFO] No GT evaluation json found. Verify GT outputs manually via stdout logs.")

def main():
    if not RESULTS_GT.exists():
        print(f"[ERROR] GT results directory not found: {RESULTS_GT}")
        print("Please run Level3/SPARE/run_gt.py first.")
        # sys.exit(1)
        
    compare_vectors()
    
    if RESULTS_FW.exists():
        compare_evals()
    else:
        print(f"\n[INFO] FW results directory not found: {RESULTS_FW}")
        print("Please run the framework evaluation (Steering.cli) to compare.")

if __name__ == "__main__":
    main()
