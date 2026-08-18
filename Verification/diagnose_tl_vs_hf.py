"""
Diagnostic: Compare TransformerLens vs HuggingFace hidden states & predictions
for the same CorrSteer MMLU samples. Memory-optimized: loads one model at a time.

Pinpoints WHERE divergence occurs:
  1. Tokenization (token IDs)
  2. Hidden states at layer 13 (pre-hook point)
  3. SAE encodings of those hidden states
  4. Logits / predictions / rewards

Usage:
  PYTHONPATH=. python Verification/diagnose_tl_vs_hf.py --device cuda:1 --n 10
"""

import argparse
import json
import sys
import gc
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run_hf(prompts, gt_answers, sae, device, n):
    """Run HF model ONE sample at a time, capture layer-13 residual, SAE on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        cache_dir="/mnt/disk1/aiotlab/.cache/huggingface/hub",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.bfloat16,
        cache_dir="/mnt/disk1/aiotlab/.cache/huggingface/hub",
    ).to(device).eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_ids = []
    all_residuals = []
    all_sae_encoded = []
    all_logits = []
    preds = []
    rewards = []

    for i in range(n):
        inp = tokenizer(prompts[i], return_tensors="pt", truncation=True, max_length=2048)
        ids = inp["input_ids"].to(device)
        mask = inp["attention_mask"].to(device)

        layer13_buf = [None]
        def hook(module, args):
            layer13_buf[0] = args[0].detach().clone().cpu()

        handle = model.model.layers[13].register_forward_pre_hook(hook)
        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=mask, num_logits_to_keep=1)
        handle.remove()

        logits = outputs.logits[:, -1, :].cpu().float()  # (1, V)
        pred_id = logits.argmax(dim=-1)
        pred = tokenizer.decode(pred_id[0].item()).strip()
        reward = 1.0 if pred == gt_answers[i] else 0.0

        residual_last = layer13_buf[0][:, -1, :].float()  # (1, d_model) already cpu
        r_sae = residual_last.to(device=sae.device, dtype=sae.dtype)
        sae_encoded = sae.encode(r_sae).detach().float()  # SAE on CPU

        all_ids.append(ids.cpu())
        all_residuals.append(residual_last)
        all_sae_encoded.append(sae_encoded)
        all_logits.append(logits)
        preds.append(pred)
        rewards.append(reward)

        del outputs, layer13_buf
        torch.cuda.empty_cache()

    result = {
        "ids_list": all_ids,  # list of tensors (different lengths)
        "residual_last": torch.cat(all_residuals, dim=0),
        "sae_encoded": torch.cat(all_sae_encoded, dim=0),
        "logits_last": torch.cat(all_logits, dim=0),
        "preds": preds,
        "rewards": rewards,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_tl(prompts, gt_answers, sae, device, n, no_processing=False):
    """Run TransformerLens model ONE sample at a time, SAE on CPU."""
    from transformer_lens import HookedTransformer

    label = "TL-NP" if no_processing else "TL"
    if no_processing:
        model = HookedTransformer.from_pretrained_no_processing(
            "google/gemma-2-2b-it", device=device, dtype=torch.bfloat16
        )
    else:
        model = HookedTransformer.from_pretrained(
            "google/gemma-2-2b-it", device=device, dtype=torch.bfloat16
        )

    model.tokenizer.padding_side = "left"

    all_ids = []
    all_residuals = []
    all_sae_encoded = []
    all_logits = []
    preds = []
    rewards = []

    for i in range(n):
        tokens = model.to_tokens([prompts[i]])  # single sample

        layer13_buf = [None]
        def hook(module, args):
            if not isinstance(args, tuple) or len(args) == 0:
                return
            layer13_buf[0] = args[0].detach().clone().cpu()

        handle = model.blocks[13].register_forward_pre_hook(hook)
        with torch.no_grad():
            logits_full = model.forward(tokens, return_type="logits")
        handle.remove()

        logits = logits_full[:, -1, :].cpu().float()  # (1, V)
        pred_id = logits.argmax(dim=-1)
        pred = model.tokenizer.decode(pred_id[0].item()).strip()
        reward = 1.0 if pred == gt_answers[i] else 0.0

        residual_last = layer13_buf[0][:, -1, :].float()  # (1, d_model) already cpu
        r_sae = residual_last.to(device=sae.device, dtype=sae.dtype)
        sae_encoded = sae.encode(r_sae).detach().float()

        all_ids.append(tokens.cpu())
        all_residuals.append(residual_last)
        all_sae_encoded.append(sae_encoded)
        all_logits.append(logits)
        preds.append(pred)
        rewards.append(reward)

        del logits_full, layer13_buf
        torch.cuda.empty_cache()

    result = {
        "ids_list": all_ids,
        "residual_last": torch.cat(all_residuals, dim=0),
        "sae_encoded": torch.cat(all_sae_encoded, dim=0),
        "logits_last": torch.cat(all_logits, dim=0),
        "preds": preds,
        "rewards": rewards,
        "label": label,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = args.device

    # ---- Load data ----
    sys.path.insert(0, str(ROOT))
    from Steering.data.formatters import mmlu_corrsteer
    mmlu_path = ROOT / "TrainDataset" / "mmlu" / "mmlu_hf_shuffled.json"
    with open(mmlu_path) as f:
        raw_data = json.load(f)[:args.n]
    data = mmlu_corrsteer(raw_data)
    prompts = [s["question"] for s in data]
    gt_answers = [s["answer"] for s in data]

    print("=" * 70)
    print(f"TL vs HF Diagnostic: {args.n} samples on {device}")
    print("=" * 70)

    # ---- Load SAE on CPU to save GPU memory ----
    from sae_lens import SAE
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_13/width_16k/canonical",
        device="cpu",
    )
    sae = sae.float()  # ensure float32 on CPU for stability
    print(f"SAE loaded on CPU: d_sae={sae.cfg.d_sae}")

    # ---- Run HF ----
    print("\n--- Running HuggingFace model ---")
    hf = run_hf(prompts, gt_answers, sae, device, args.n)
    print(f"  HF token lengths: {[t.shape[1] for t in hf['ids_list']]}")
    print(f"  HF preds:   {hf['preds']}")
    print(f"  HF rewards: {hf['rewards']}")
    print(f"  HF accuracy: {sum(hf['rewards'])/len(hf['rewards']):.1%}")

    # ---- Run TL (standard from_pretrained) ----
    print("\n--- Running TransformerLens (standard) ---")
    tl = run_tl(prompts, gt_answers, sae, device, args.n, no_processing=False)
    print(f"  TL token lengths: {[t.shape[1] for t in tl['ids_list']]}")
    print(f"  TL preds:   {tl['preds']}")
    print(f"  TL rewards: {tl['rewards']}")
    print(f"  TL accuracy: {sum(tl['rewards'])/len(tl['rewards']):.1%}")

    # ---- Run TL no_processing ----
    print("\n--- Running TransformerLens (no_processing) ---")
    np_ = run_tl(prompts, gt_answers, sae, device, args.n, no_processing=True)
    print(f"  NP token lengths: {[t.shape[1] for t in np_['ids_list']]}")
    print(f"  NP preds:   {np_['preds']}")
    print(f"  NP rewards: {np_['rewards']}")
    print(f"  NP accuracy: {sum(np_['rewards'])/len(np_['rewards']):.1%}")

    # ========================================================================
    # COMPARISONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # --- A. Tokenization ---
    print("\n--- A. Tokenization ---")
    for i in range(min(3, args.n)):  # show first 3 samples
        hf_ids = hf['ids_list'][i].squeeze()
        tl_ids = tl['ids_list'][i].squeeze()
        np_ids = np_['ids_list'][i].squeeze()
        print(f"  Sample {i}: HF len={len(hf_ids)}, TL len={len(tl_ids)}, NP len={len(np_ids)}")
        # Check if TL has extra BOS token
        if len(tl_ids) == len(hf_ids) + 1:
            match = (hf_ids == tl_ids[1:]).all()
            print(f"    TL has extra leading token (id={tl_ids[0].item()}), after removal: {'MATCH' if match else 'MISMATCH'}")
        elif len(tl_ids) == len(hf_ids):
            match = (hf_ids == tl_ids).all()
            print(f"    Same length, tokens {'MATCH' if match else 'MISMATCH'}")
        else:
            print(f"    Different lengths: HF={len(hf_ids)} vs TL={len(tl_ids)}")
    print("  (Tokenization diffs within BOS handling are expected and harmless for last-token hidden states)")


    # --- B. Hidden states at layer 13 ---
    print("\n--- B. Hidden States at Layer 13 (last position) ---")
    hf_h = hf['residual_last']
    tl_h = tl['residual_last']
    np_h = np_['residual_last']

    cos_hf_tl = torch.nn.functional.cosine_similarity(hf_h, tl_h, dim=-1)
    cos_hf_np = torch.nn.functional.cosine_similarity(hf_h, np_h, dim=-1)
    cos_tl_np = torch.nn.functional.cosine_similarity(tl_h, np_h, dim=-1)

    diff_hf_tl = (hf_h - tl_h).abs()
    diff_hf_np = (hf_h - np_h).abs()

    print(f"  HF vs TL:   cos={cos_hf_tl.mean():.8f} (min={cos_hf_tl.min():.8f})"
          f"  abs_diff: mean={diff_hf_tl.mean():.6f} max={diff_hf_tl.max():.6f}")
    print(f"  HF vs NP:   cos={cos_hf_np.mean():.8f} (min={cos_hf_np.min():.8f})"
          f"  abs_diff: mean={diff_hf_np.mean():.6f} max={diff_hf_np.max():.6f}")
    print(f"  TL vs NP:   cos={cos_tl_np.mean():.8f}")

    # --- C. SAE encodings ---
    print("\n--- C. SAE Encodings (last position) ---")
    hf_s = hf['sae_encoded']
    tl_s = tl['sae_encoded']
    np_s = np_['sae_encoded']

    hf_active = (hf_s > 0).sum(dim=-1).float()
    tl_active = (tl_s > 0).sum(dim=-1).float()
    np_active = (np_s > 0).sum(dim=-1).float()
    print(f"  Avg active features: HF={hf_active.mean():.1f}, TL={tl_active.mean():.1f}, NP={np_active.mean():.1f}")

    cos_sae_hf_tl = torch.nn.functional.cosine_similarity(hf_s, tl_s, dim=-1)
    cos_sae_hf_np = torch.nn.functional.cosine_similarity(hf_s, np_s, dim=-1)
    print(f"  SAE cosine HF vs TL:  {cos_sae_hf_tl.mean():.6f} (min={cos_sae_hf_tl.min():.6f})")
    print(f"  SAE cosine HF vs NP:  {cos_sae_hf_np.mean():.6f} (min={cos_sae_hf_np.min():.6f})")

    for label, other_s in [("TL", tl_s), ("NP", np_s)]:
        overlaps = []
        for i in range(args.n):
            hf_feat = set((hf_s[i] > 0).nonzero(as_tuple=True)[0].tolist())
            ot_feat = set((other_s[i] > 0).nonzero(as_tuple=True)[0].tolist())
            if len(hf_feat | ot_feat) == 0:
                overlaps.append(1.0)
            else:
                overlaps.append(len(hf_feat & ot_feat) / len(hf_feat | ot_feat))
        print(f"  Active feature Jaccard HF vs {label}: mean={np.mean(overlaps):.4f} min={np.min(overlaps):.4f}")

    # --- D. Logits & Predictions ---
    print("\n--- D. Logits & Predictions ---")
    cos_log_hf_tl = torch.nn.functional.cosine_similarity(hf['logits_last'], tl['logits_last'], dim=-1)
    cos_log_hf_np = torch.nn.functional.cosine_similarity(hf['logits_last'], np_['logits_last'], dim=-1)
    print(f"  Logit cos HF vs TL:  {cos_log_hf_tl.mean():.8f}")
    print(f"  Logit cos HF vs NP:  {cos_log_hf_np.mean():.8f}")

    print(f"\n  Per-sample predictions:")
    pred_match_tl = 0
    pred_match_np = 0
    reward_match_tl = 0
    reward_match_np = 0
    for i in range(args.n):
        hf_p = hf['preds'][i]
        tl_p = tl['preds'][i]
        np_p = np_['preds'][i]
        gt = gt_answers[i]
        m_tl = "=" if hf_p == tl_p else "X"
        m_np = "=" if hf_p == np_p else "X"
        r_hf = hf['rewards'][i]
        r_tl = tl['rewards'][i]
        r_np = np_['rewards'][i]
        if hf_p == tl_p: pred_match_tl += 1
        if hf_p == np_p: pred_match_np += 1
        if r_hf == r_tl: reward_match_tl += 1
        if r_hf == r_np: reward_match_np += 1
        print(f"    [{i:2d}] GT={gt} HF={hf_p:>8s} TL={tl_p:>8s}[{m_tl}] NP={np_p:>8s}[{m_np}]  "
              f"r: HF={r_hf:.0f} TL={r_tl:.0f} NP={r_np:.0f}")

    print(f"\n  Prediction match: HF vs TL: {pred_match_tl}/{args.n}, HF vs NP: {pred_match_np}/{args.n}")
    print(f"  Reward match:     HF vs TL: {reward_match_tl}/{args.n}, HF vs NP: {reward_match_np}/{args.n}")

    # --- SUMMARY ---
    print("\n" + "=" * 70)
    print("SUMMARY: Which TL variant is closer to HF?")
    print("=" * 70)
    print(f"  Hidden cos:  TL={cos_hf_tl.mean():.8f}  NP={cos_hf_np.mean():.8f}  {'NP wins' if cos_hf_np.mean() > cos_hf_tl.mean() else 'TL wins'}")
    print(f"  SAE cos:     TL={cos_sae_hf_tl.mean():.6f}  NP={cos_sae_hf_np.mean():.6f}  {'NP wins' if cos_sae_hf_np.mean() > cos_sae_hf_tl.mean() else 'TL wins'}")
    print(f"  Logit cos:   TL={cos_log_hf_tl.mean():.8f}  NP={cos_log_hf_np.mean():.8f}  {'NP wins' if cos_log_hf_np.mean() > cos_log_hf_tl.mean() else 'TL wins'}")
    print(f"  Pred match:  TL={pred_match_tl}/{args.n}  NP={pred_match_np}/{args.n}")
    print(f"  Reward match:TL={reward_match_tl}/{args.n}  NP={reward_match_np}/{args.n}")

    # Save
    results = {
        "hidden_cos_tl": float(cos_hf_tl.mean()),
        "hidden_cos_np": float(cos_hf_np.mean()),
        "sae_cos_tl": float(cos_sae_hf_tl.mean()),
        "sae_cos_np": float(cos_sae_hf_np.mean()),
        "logit_cos_tl": float(cos_log_hf_tl.mean()),
        "logit_cos_np": float(cos_log_hf_np.mean()),
        "pred_match_tl": pred_match_tl,
        "pred_match_np": pred_match_np,
        "reward_match_tl": reward_match_tl,
        "reward_match_np": reward_match_np,
        "n": args.n,
    }
    out_path = ROOT / "Verification_Results" / "tl_vs_hf_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
