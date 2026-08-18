"""
Level 3 CorrSteer Comparison: GT vs Framework
Compares:
  1. SAE feature index / coefficient / correlation
  2. Steering vector cosine similarity (sparse + dense)
  3. Accuracy (steered vs baseline) and delta
  4. Per-sample output comparison (for overlapping prompts)
"""
import json
import sys
import os
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_GT = ROOT / "Results" / "l3_corrsteer_gt_42"
RESULTS_FW = ROOT / "Results" / "l3_corrsteer"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-10))

def sep(title=""):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# ─── 1. Selected Feature ─────────────────────────────────────────────────
sep("1. SELECTED FEATURE  (Index / Coefficient / Correlation)")

gt_corr_data = load_json(RESULTS_GT / "gemma2b_mmlu_13_corr.json")
gt_sel = gt_corr_data["results"]["13"]["selected"]
print(f"  GT Feature Index : {gt_sel['feature_index']}")
print(f"  GT Coefficient   : {gt_sel['coefficient']:.6f}")
print(f"  GT Correlation   : {gt_sel['correlation']:.6f}")
print(f"  GT Frequency     : {gt_sel['frequency']:.2f}%")

gt_idx_data = load_json(RESULTS_GT / "corrsteer_indices.json")
print(f"\n  GT indices.json  → layer={gt_idx_data['layer']}  "
      f"index={gt_idx_data['index']}  coeff={gt_idx_data['coefficient']:.6f}")

# Framework: check if FW saved its indices
fw_idx_file = RESULTS_FW / "corrsteer_indices.json"
if fw_idx_file.exists():
    fw_idx_data = load_json(fw_idx_file)
    print(f"\n  FW indices.json  → layer={fw_idx_data['layer']}  "
          f"index={fw_idx_data['index']}  coeff={fw_idx_data['coefficient']:.6f}")
    idx_match  = "✅ MATCH" if gt_idx_data["index"] == fw_idx_data["index"] else "❌ MISMATCH"
    coef_match = "✅ MATCH" if abs(gt_idx_data["coefficient"] - fw_idx_data["coefficient"]) < 1e-4 else "⚠️ DIFFERS"
    print(f"  Index comparison : {idx_match}")
    print(f"  Coeff comparison : {coef_match}")
else:
    print(f"\n  [INFO] FW corrsteer_indices.json not found → FW evaluation run without index extraction")
    print(f"  → FW uses index from extraction. Check eval log for printed feature index.")

# ─── 2. Steering Vector Similarity ───────────────────────────────────────
sep("2. STEERING VECTOR SIMILARITY  (Sparse + Dense cosine)")

gt_sparse_path = RESULTS_GT / "corrsteer_sparse_vector.pt"
gt_dense_path  = RESULTS_GT / "corrsteer_dense_vector.pt"

if gt_sparse_path.exists():
    gt_sv = torch.load(gt_sparse_path, map_location="cpu")
    gt_dv = torch.load(gt_dense_path,  map_location="cpu")
    nnz = int((gt_sv != 0).sum())
    print(f"  GT sparse: shape={gt_sv.shape}, nnz={nnz}, norm={float(gt_sv.norm()):.6f}")
    print(f"  GT dense:  shape={gt_dv.shape}, norm={float(gt_dv.norm()):.6f}")
else:
    print("  [WARN] GT sparse/dense vectors not found"); gt_sv = gt_dv = None

fw_sparse_path = RESULTS_FW / "corrsteer_sparse_vector.pt"
fw_dense_path  = RESULTS_FW / "corrsteer_dense_vector.pt"

if fw_sparse_path.exists():
    fw_sv = torch.load(fw_sparse_path, map_location="cpu")
    fw_dv = torch.load(fw_dense_path,  map_location="cpu")
    nnz = int((fw_sv != 0).sum())
    print(f"\n  FW sparse: shape={fw_sv.shape}, nnz={nnz}, norm={float(fw_sv.norm()):.6f}")
    print(f"  FW dense:  shape={fw_dv.shape}, norm={float(fw_dv.norm()):.6f}")
    if gt_sv is not None:
        sc = cosine(gt_sv, fw_sv); dc = cosine(gt_dv, fw_dv)
        def flag(v): return "✅" if v > 0.99 else ("⚠️" if v > 0.9 else "❌")
        print(f"\n  Cosine (sparse): {sc:.6f}  {flag(sc)}")
        print(f"  Cosine (dense):  {dc:.6f}  {flag(dc)}")
else:
    print(f"\n  [INFO] FW steering vectors not found in {RESULTS_FW}")
    print("  → The framework evaluation used on-the-fly steering, not saved vectors.")
    print("  → Add save_vector to config or extract manually for vector comparison.")
    # Reconstruct FW vector from extracted feature index if indices file exists
    if fw_idx_file.exists() and gt_sv is not None:
        fw_idx_data = load_json(fw_idx_file)
        fw_sv_recon = torch.zeros_like(gt_sv)
        fw_sv_recon[fw_idx_data["index"]] = fw_idx_data["coefficient"]
        fw_dv_recon = fw_sv_recon @ torch.load(gt_dense_path, map_location="cpu").div(
            gt_sv[gt_idx_data["index"]]) if gt_idx_data["index"] != 0 else None
        sc = cosine(gt_sv, fw_sv_recon)
        print(f"\n  Reconstructed FW sparse cosine (from indices.json): {sc:.6f}  {'✅' if sc>0.99 else '❌'}")

# ─── 3. Accuracy Comparison ───────────────────────────────────────────────
sep("3. ACCURACY  (Baseline vs Steered, Delta)")

gt_acc_data  = load_json(RESULTS_GT / "gemma2b_mmlu_13_corr_accuracy.json")
gt_steered_acc = gt_acc_data.get("accuracy", None)
# GT baseline comes from the train.py print: "Baseline accuracy: 40.00%"
# It's stored in the corr.json validation step
gt_corr_val_layer = gt_corr_data["results"]["13"].get("baseline_accuracy", None)
print(f"  GT steered accuracy : {gt_steered_acc * 100:.2f}% ({int(gt_steered_acc * 30)}/30)")
if gt_corr_val_layer is not None:
    print(f"  GT baseline accuracy: {gt_corr_val_layer * 100:.2f}%")

fw_eval_files = sorted(RESULTS_FW.glob("eval_corrsteer_l3_validation_*.json"))
if fw_eval_files:
    fw_eval_path = fw_eval_files[-1]
    fw_eval = load_json(fw_eval_path)
    fw_result = fw_eval["result"]
    fw_acc  = fw_result["accuracy"]
    fw_base = fw_result["baseline_accuracy"]
    fw_delta = fw_result["delta"]
    fw_total = fw_result["total"]
    print(f"\n  FW steered accuracy : {fw_acc * 100:.2f}% ({fw_result['correct']}/{fw_total})")
    print(f"  FW baseline accuracy: {fw_base * 100:.2f}%")
    print(f"  FW delta            : {fw_delta * 100:+.2f}%")

    # GT delta
    gt_baseline_acc = 0.4  # from train.py output: "Baseline accuracy: 40.00%"
    gt_delta = gt_steered_acc - gt_baseline_acc
    print(f"\n  GT steered accuracy : {gt_steered_acc * 100:.2f}%")
    print(f"  GT baseline accuracy: {gt_baseline_acc * 100:.2f}% (from train.py log)")
    print(f"  GT delta            : {gt_delta * 100:+.2f}%")

    acc_match   = "✅ MATCH" if abs(gt_steered_acc - fw_acc)  < 0.02 else "⚠️ DIFFERS"
    base_note   = "Note: GT and FW use DIFFERENT test sets (no common prompts found)"
    delta_match = "✅ MATCH" if abs(gt_delta - fw_delta) < 0.02 else "⚠️ DIFFERS"
    print(f"\n  Steered acc match   : {acc_match}")
    print(f"  Delta match         : {delta_match}")
    print(f"  [{base_note}]")

# ─── 4. Per-sample output comparison ─────────────────────────────────────
sep("4. PER-SAMPLE OUTPUT COMPARISON  (Common prompts)")

gt_samples = load_json(RESULTS_GT / "gemma2b_mmlu_13_12281_1_l30.json")

fw_eval_files = sorted(RESULTS_FW.glob("eval_corrsteer_l3_validation_*.json"))
if fw_eval_files:
    fw_samples = load_json(fw_eval_files[-1])["result"]["samples"]

    gt_by_prompt = {s["prompt"].strip(): s for s in gt_samples}
    fw_by_prompt = {s["prompt"].strip(): s for s in fw_samples}
    common = set(gt_by_prompt.keys()) & set(fw_by_prompt.keys())
    print(f"  GT samples: {len(gt_samples)}  |  FW samples: {len(fw_samples)}")
    print(f"  Common prompts: {len(common)}")
    if len(common) > 0:
        agree = sum(1 for p in common if gt_by_prompt[p]["predicted"] == fw_by_prompt[p]["response"])
        print(f"  Output agreement on common: {agree}/{len(common)}")
        for p in list(common)[:3]:
            print(f"\n  Prompt (truncated): {p[:80]}...")
            print(f"    GT  → pred={gt_by_prompt[p]['predicted']}  gt={gt_by_prompt[p]['ground_truth']}")
            print(f"    FW  → pred={fw_by_prompt[p]['response']}   gt={fw_by_prompt[p]['ground_truth']}")
    else:
        print("  [INFO] No common prompts → GT and FW use different data splits.")
        print("  Showing GT first 5 vs FW first 5 side-by-side:")
        print(f"  {'GT pred':10} {'GT gt':8} | {'FW pred':10} {'FW gt':8}")
        for g, f in zip(gt_samples[:10], fw_samples[:10]):
            print(f"  {g['predicted']:10} {g['ground_truth']:8} | {f['response']:10} {f['ground_truth']:8}")

# ─── 5. Top-K feature list comparison ────────────────────────────────────
sep("5. TOP-K FEATURE LIST (GT)")
top_pos = gt_corr_data["results"]["13"]["top_positive"]
print(f"  Rank | Feature Index | Coefficient | Correlation | Frequency%")
print(f"  {'-'*66}")
for i, feat in enumerate(top_pos[:10], 1):
    sel_mark = " ← SELECTED" if feat["feature_index"] == gt_sel["feature_index"] else ""
    print(f"  {i:4d} | {feat['feature_index']:13d} | {feat['coefficient']:11.4f} | "
          f"{feat['correlation']:11.6f} | {feat['frequency']:9.2f}%{sel_mark}")

sep("SUMMARY")
print(f"  GT  → SAE feature={gt_sel['feature_index']}, coeff={gt_sel['coefficient']:.4f}, corr={gt_sel['correlation']:.4f}")
if fw_idx_file.exists():
    fw_idx_data = load_json(fw_idx_file)
    idx_ok = gt_idx_data['index'] == fw_idx_data['index']
    print(f"  FW  → SAE feature={fw_idx_data['index']}, coeff={fw_idx_data['coefficient']:.4f}")
    print(f"  Feature index: {'✅ MATCH' if idx_ok else '❌ MISMATCH'}")
if fw_eval_files:
    fw_result = load_json(fw_eval_files[-1])["result"]
    gt_acc_val = gt_steered_acc
    fw_acc_val = fw_result["accuracy"]
    print(f"  GT steered accuracy : {gt_acc_val*100:.2f}%  (baseline 40.00%, delta={gt_delta*100:+.2f}%)")
    print(f"  FW steered accuracy : {fw_acc_val*100:.2f}%  (baseline {fw_result['baseline_accuracy']*100:.2f}%, delta={fw_result['delta']*100:+.2f}%)")
    print(f"  Accuracy match      : {'✅ MATCH' if abs(gt_acc_val - fw_acc_val) < 0.02 else '⚠️ DIFFERS (different test sets)'}")
print()
