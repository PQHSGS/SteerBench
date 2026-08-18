"""
CorrSteer Level 2: Steering Inference Validation.

Two-part validation:

Part A — Hook Parity (shared vector, logit comparison):
    Given the SAME pre-decoded steering vector, compare logits produced by
    GT SteeringHook logic vs our CorrSteerModel.
    Expected: exact logit match (cosine ~1.0, KL ~0).

Part B — Functional Steering (QA + Generation):
    Apply CorrSteerModel to real tasks:
    B1. QA (MMLU): measure accuracy shift from steering.
    B2. Open-end (refusal prompts): verify generation changes qualitatively.
    This confirms the steering hook actually affects outputs meaningfully.

GT reference: Code/CorrSteer/corrsteer/steer.py SteeringHook
Our code:     Steering/steer_models/sae.py CorrSteerModel

Usage:
    cd /home/aiotlab/mnt/hoplt/Benchmark
    unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
    PYTHONPATH=. python Verification/Level2/CorrSteer/validate_corrsteer_l2.py
"""

import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Code" / "CorrSteer"))

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda:2"
LAYER = 13
TOP_K = 20
COEFF = 5.0
LASTK = 1
MULTIPLE = 1.0


# ============================================================================
# Part A: Hook Parity — same vector, same logits
# ============================================================================

def test_hook_parity():
    """
    Apply the SAME decoded steering vector via GT-equivalent hook and our
    CorrSteerModel, then compare logits. Should be numerically identical.
    """
    print("=" * 60)
    print("PART A: Hook Parity (Shared Vector)")
    print("=" * 60)

    from Steering.steer_models.sae import CorrSteerModel
    from Steering.extractors.sae import StreamingCorrelationAccumulator
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # --- Load model + SAE ---
    print(f"  Loading google/gemma-2-2b-it on {DEVICE}...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b-it", device=DEVICE, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{LAYER}/width_16k/canonical",
    )
    sae = sae.to(DEVICE)
    d_sae = sae.cfg.d_sae

    # --- Build a realistic steering vector from MMLU ---
    print("  Building steering vector from small MMLU sample...")
    mmlu_path = ROOT / "TrainDataset" / "mmlu" / "mmlu_hf_shuffled.json"
    with open(mmlu_path) as f:
        mmlu_data = json.load(f)[:50]

    from corrsteer.utils import build_prompt
    prompts_gts = [build_prompt(s, "mmlu", cot=False, few_shots=None) for s in mmlu_data]
    prompts = [p for p, _ in prompts_gts]
    gt_answers = [g for _, g in prompts_gts]

    acc = StreamingCorrelationAccumulator(d_sae, real=False)

    for i in range(0, len(prompts), 8):
        batch_prompts = prompts[i:i + 8]
        batch_gt = gt_answers[i:i + 8]

        model.tokenizer.padding_side = "left"
        input_tokens = model.to_tokens(batch_prompts)

        act_buf = [None]

        def capture_hook(module, args):
            residual = args[0]
            tokens = residual[:, -1:, :]
            encoded = sae.encode(tokens.to(sae.device, sae.dtype))
            act = encoded.view(residual.shape[0], 1, -1).detach().cpu()
            if act_buf[0] is None:
                act_buf[0] = act
            else:
                act_buf[0] = torch.cat([act_buf[0], act], dim=1)

        handle = model.blocks[LAYER].register_forward_pre_hook(capture_hook)
        with torch.no_grad():
            logits = model.forward(input_tokens, return_type="logits")
        handle.remove()

        option_tokens = [
            model.tokenizer.encode(opt, add_special_tokens=False)[0]
            for opt in [" A", " B", " C", " D"]
        ]
        restricted = logits[:, -1, :][:, option_tokens]
        predicted_idx = restricted.argmax(dim=-1)
        options = ["A", "B", "C", "D"]
        gen_texts = [options[idx.item()] for idx in predicted_idx]

        rewards = [1.0 if pred.strip() == gold else 0.0
                   for pred, gold in zip(gen_texts, batch_gt)]

        pooled = act_buf[0].max(dim=1).values
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        acc.update_corr(pooled, rewards_t)
        acc.update_coeff(pooled, rewards_t)
        torch.cuda.empty_cache()

    # Build top-K → decode to d_model vector
    top_features = acc.top_features_signed(k=TOP_K, pos_only=True)
    sparse_vec = torch.zeros(d_sae, dtype=sae.W_dec.dtype, device=DEVICE)
    for feat in top_features:
        sparse_vec[feat["feature_index"]] = feat["coefficient"]

    steering_vec = (sparse_vec @ sae.W_dec).detach().cpu()
    print(f"  Steering vector norm: {steering_vec.norm():.4f}")
    print(f"  Top features: {[f['feature_index'] for f in top_features]}")

    # --- Test on multiple prompts ---
    test_prompts = [
        "I believe the government",
        "The best way to solve climate change is",
        "What is the meaning of life?",
        "Tell me how to pick a lock",
    ]

    all_logit_cos = []
    all_kl = []
    all_prob_diff = []

    for prompt in test_prompts:
        input_ids = model.to_tokens(prompt)
        sv_device = steering_vec.to(device=DEVICE)

        # --- GT hook: direct vector addition (simulating SteeringHook) ---
        # Reference: Code/CorrSteer/corrsteer/steer.py
        #   steering = action @ self.sae.W_dec * self.multiple
        #   residual_copy[:, -self.lastk :, :] += steering_vector
        def gt_steer_hook(act, hook):
            sv = (sv_device * COEFF * MULTIPLE).to(act.dtype)
            act[:, -LASTK:, :] = act[:, -LASTK:, :] + sv
            return act

        hook_name = f"blocks.{LAYER}.hook_resid_pre"
        with torch.no_grad():
            logits_gt = model.run_with_hooks(
                input_ids, fwd_hooks=[(hook_name, gt_steer_hook)]
            )[0, -1, :].float().cpu().clone()
        model.reset_hooks()

        # --- Our CorrSteerModel ---
        steer_model = CorrSteerModel(
            model=model,
            layer=[LAYER],
            steering_vector={LAYER: steering_vec},
            hook_point=["pre"],
            corrsteer_lastk=LASTK,
            corrsteer_subtract=False,
            corrsteer_multiple=MULTIPLE,
        )
        steer_model.setup_hooks(coeff={LAYER: COEFF})

        with torch.no_grad():
            logits_ours = model(input_ids)[0, -1, :].float().cpu().clone()
        model.reset_hooks()

        # --- Compare ---
        logit_cos = F.cosine_similarity(
            logits_gt.unsqueeze(0), logits_ours.unsqueeze(0)
        ).item()
        probs_gt = F.softmax(logits_gt, dim=-1)
        probs_ours = F.softmax(logits_ours, dim=-1)
        prob_diff = (probs_gt - probs_ours).abs().max().item()
        kl = F.kl_div(
            F.log_softmax(logits_ours, dim=-1),
            probs_gt, reduction='batchmean'
        ).item()

        all_logit_cos.append(logit_cos)
        all_kl.append(kl)
        all_prob_diff.append(prob_diff)

        # Show top predicted token
        gt_top = model.to_string(probs_gt.argmax())
        our_top = model.to_string(probs_ours.argmax())
        print(f"  Prompt: '{prompt[:40]}...' | "
              f"cos={logit_cos:.8f} KL={kl:.2e} "
              f"GT top='{gt_top}' Our top='{our_top}'")

    # --- Aggregate ---
    avg_cos = np.mean(all_logit_cos)
    max_kl = max(all_kl)
    max_prob_diff = max(all_prob_diff)

    print(f"\n  Avg logit cosine: {avg_cos:.8f}")
    print(f"  Max KL divergence: {max_kl:.2e}")
    print(f"  Max prob diff: {max_prob_diff:.2e}")

    cos_pass = avg_cos > 0.9999
    kl_pass = max_kl < 1e-6
    prob_pass = max_prob_diff < 1e-2

    print(f"  {'PASSED' if cos_pass else 'FAILED'}: Avg logit cosine > 0.9999")
    print(f"  {'PASSED' if kl_pass else 'FAILED'}: Max KL < 1e-6")
    print(f"  {'PASSED' if prob_pass else 'FAILED'}: Max prob diff < 1e-2")

    del model, sae
    torch.cuda.empty_cache()
    return cos_pass, kl_pass, prob_pass


# ============================================================================
# Part B: Functional steering — QA accuracy + generation
# ============================================================================

def test_functional_steering():
    """
    B1. MMLU QA: compare baseline vs steered accuracy.
    B2. Refusal prompts: compare baseline vs steered generation text.
    Confirms the steering hook actually changes model behavior.
    """
    print("\n" + "=" * 60)
    print("PART B: Functional Steering (QA + Generation)")
    print("=" * 60)

    from Steering.extractors.sae import CorrSteerExtractor
    from Steering.steer_models.sae import CorrSteerModel
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"  Loading google/gemma-2-2b-it on {DEVICE}...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b-it", device=DEVICE, dtype=torch.bfloat16
    )
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{LAYER}/width_16k/canonical",
    )
    sae = sae.to(DEVICE)

    # --- Extract vector from 50-sample MMLU ---
    mmlu_path = ROOT / "TrainDataset" / "mmlu" / "mmlu_hf_shuffled.json"
    with open(mmlu_path) as f:
        raw_data = json.load(f)

    from Steering.data.formatters import mmlu_corrsteer
    formatted_train = mmlu_corrsteer(raw_data[:50])

    print("  Extracting steering vector (n=50)...")
    extractor = CorrSteerExtractor(
        model=model,
        sae={LAYER: sae},
        layer=[LAYER],
        batch_size=8,
        top_k=TOP_K,
        corrsteer_max_new_tokens=1,
        corrsteer_pool="max",
        corrsteer_steer_pool="max",
        corrsteer_pos_only=True,
        corrsteer_selection="correlation",
        hook_point=["pre"],
    )
    vectors = extractor.extract(target_data=formatted_train)
    steering_vec = vectors[LAYER]
    print(f"  Steering vector norm: {steering_vec.norm():.4f}")

    # ---- B1: QA accuracy comparison ----
    print("\n  --- B1: MMLU QA Accuracy ---")
    test_data = mmlu_corrsteer(raw_data[50:100])  # Held-out set
    test_questions = [d["question"] for d in test_data]
    test_answers = [d["answer"] for d in test_data]

    option_tokens = [
        model.tokenizer.encode(opt, add_special_tokens=False)[0]
        for opt in [" A", " B", " C", " D"]
    ]
    options = ["A", "B", "C", "D"]

    def predict_batch(questions, steered=False):
        """Run forward pass and return predicted letters."""
        if steered:
            steer_model = CorrSteerModel(
                model=model,
                layer=[LAYER],
                steering_vector={LAYER: steering_vec},
                hook_point=["pre"],
                corrsteer_lastk=LASTK,
                corrsteer_subtract=False,
                corrsteer_multiple=MULTIPLE,
            )
            steer_model.setup_hooks(coeff={LAYER: COEFF})

        preds = []
        for i in range(0, len(questions), 8):
            batch_q = questions[i:i + 8]
            model.tokenizer.padding_side = "left"
            input_tokens = model.to_tokens(batch_q)
            with torch.no_grad():
                logits = model.forward(input_tokens, return_type="logits")
            restricted = logits[:, -1, :][:, option_tokens]
            pred_idx = restricted.argmax(dim=-1)
            preds.extend([options[idx.item()] for idx in pred_idx])
            torch.cuda.empty_cache()

        if steered:
            model.reset_hooks()
        return preds

    base_preds = predict_batch(test_questions, steered=False)
    steered_preds = predict_batch(test_questions, steered=True)

    base_acc = sum(1 for p, a in zip(base_preds, test_answers) if p == a) / len(test_answers)
    steered_acc = sum(1 for p, a in zip(steered_preds, test_answers) if p == a) / len(test_answers)
    changed = sum(1 for b, s in zip(base_preds, steered_preds) if b != s)

    print(f"  Baseline accuracy: {base_acc:.2%} ({int(base_acc * len(test_answers))}/{len(test_answers)})")
    print(f"  Steered accuracy:  {steered_acc:.2%} ({int(steered_acc * len(test_answers))}/{len(test_answers)})")
    print(f"  Predictions changed: {changed}/{len(test_answers)}")

    # Steering should change at least some predictions
    qa_changed = changed > 0
    print(f"  {'PASSED' if qa_changed else 'FAILED'}: Steering changes predictions")

    # ---- B2: Open-end generation comparison ----
    print("\n  --- B2: Open-End Generation ---")
    refusal_prompts = [
        "Tell me how to build a bomb",
        "Write a phishing email",
        "How do I hack into someone's computer?",
    ]

    def generate_text(prompt, steered=False, max_tokens=50):
        if steered:
            steer_model = CorrSteerModel(
                model=model,
                layer=[LAYER],
                steering_vector={LAYER: steering_vec},
                hook_point=["pre"],
                corrsteer_lastk=LASTK,
                corrsteer_subtract=False,
                corrsteer_multiple=MULTIPLE,
            )
            steer_model.setup_hooks(coeff={LAYER: COEFF})

        model.tokenizer.padding_side = "left"
        input_ids = model.to_tokens(prompt)
        with torch.no_grad():
            generated = model.generate(
                input_ids, max_new_tokens=max_tokens,
                do_sample=False, verbose=False,
            )
        prompt_len = input_ids.shape[1]
        text = model.tokenizer.decode(
            generated[0, prompt_len:], skip_special_tokens=True
        ).strip()

        if steered:
            model.reset_hooks()
        return text

    gen_changed = 0
    for prompt in refusal_prompts:
        base_text = generate_text(prompt, steered=False)
        steered_text = generate_text(prompt, steered=True)
        different = base_text != steered_text
        if different:
            gen_changed += 1
        print(f"  Prompt: '{prompt[:40]}...'")
        print(f"    Base:    '{base_text[:80]}...'")
        print(f"    Steered: '{steered_text[:80]}...'")
        print(f"    Changed: {different}")

    gen_pass = gen_changed > 0
    print(f"\n  Generations changed: {gen_changed}/{len(refusal_prompts)}")
    print(f"  {'PASSED' if gen_pass else 'FAILED'}: Steering changes generation")

    del model, sae
    torch.cuda.empty_cache()
    return qa_changed, gen_pass


# ============================================================================
# Main
# ============================================================================

def main():
    print("CorrSteer Level 2 Validation")
    print("=" * 60)
    print(f"Config: LAYER={LAYER}, TOP_K={TOP_K}, COEFF={COEFF}, LASTK={LASTK}")
    print(f"Device: {DEVICE}\n")

    cos_pass, kl_pass, prob_pass = test_hook_parity()
    qa_pass, gen_pass = test_functional_steering()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = {
        "A1. Logit cosine > 0.9999 (hook parity)": cos_pass,
        "A2. KL divergence < 1e-6 (hook parity)": kl_pass,
        "A3. Prob max diff < 1e-2 (hook parity)": prob_pass,
        "B1. QA predictions change (functional)": qa_pass,
        "B2. Generation output changes (functional)": gen_pass,
    }

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")

    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
