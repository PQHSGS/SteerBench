"""
Experiment F: Per-token KL on steered outputs + accuracy.

Qi et al. (2024) showed fine-tuning KL concentrates on first ~5 output tokens.
Does steering KL do the same? If yes -> steering operates through the same
shallow channel as RLHF, and the cancellation filter blocks that window.

Design:
  - Uses SteeringPipeline for data loading + vector extraction (same infra as evals).
  - Step-by-step autoregressive generation with BOTH base and steered models.
  - At each position t, both models condition on the SAME prefix (base's tokens).
  - KL(steered_logits || base_logits) at each position -> clean causal quantity.
  - Free-generation steered -> accuracy via evaluator.

Key question: does per-token KL concentration (ratio early/late) predict accuracy?
  - High ratio + low acc   = steering blocked at early tokens (shallow channel)
  - Low ratio + high acc   = steering operates deeper
  - Correct/incorrect diff = early logit change predicts success

Tasks: Toxic, Evil, Deception, Refusal via CAA/CHARS/ACT/PID/CURVE/REPS/WEIGHT
Coeffs: 1.0, 3.0, 5.0

Usage:
    conda activate sae_circuit
    export CUDA_VISIBLE_DEVICES=2
    python Experiments/ExpF/expF_per_token_kl.py --method CAA
    python Experiments/ExpF/expF_per_token_kl.py --method WEIGHT
    python Experiments/ExpF/expF_per_token_kl.py --method REPS

Returns:
    JSON with per-task per-coeff: kl_curve, kl_correct, kl_incorrect, accuracy
"""

import argparse, json, torch, numpy as np, sys, time
from pathlib import Path
from torch.nn import functional as F
from collections import defaultdict

sys.path.insert(0, ".")
from Steering.pipeline import SteeringPipeline
from Steering.config.pipeline import PipelineConfig
from Steering.evaluators import EVALUATOR_MAP
from Steering.steer_models.weight import WeightSteerModel

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cuda"
N_PROMPTS = 50
MAX_NEW_TOKENS = 20
COEFFS = [1.0, 3.0, 5.0]
OUTPUT = "Experiments/ExpF/expF_results.json"

MULTI_CONFIGS = {
    "CAA": {
        "toxic":     "Configs/Eval/CAA/Gemma/gemma_toxic.json",
        "evil":      "Configs/Eval/CAA/Gemma/gemma_evil.json",
        "deception": "Configs/Eval/CAA/Gemma/gemma_deception.json",
        "refusal":   "Configs/Eval/CAA/Gemma/gemma_refusal_response.json",
    },
    "CHARS": {
        "toxic":     "Configs/Eval/CHARS/Gemma/gemma_toxic.json",
        "evil":      "Configs/Eval/CHARS/Gemma/gemma_evil.json",
        "deception": "Configs/Eval/CHARS/Gemma/gemma_deception.json",
        "refusal":   "Configs/Eval/CHARS/Gemma/gemma_refusal_response.json",
    },
    "ACT": {
        "toxic":     "Configs/Eval/LinearAcT/Gemma/gemma_toxic.json",
        "evil":      "Configs/Eval/LinearAcT/Gemma/gemma_evil.json",
        "deception": "Configs/Eval/LinearAcT/Gemma/gemma_deception.json",
        "refusal":   "Configs/Eval/LinearAcT/Gemma/gemma_refusal_response.json",
    },
    "PID": {
        "toxic":     "Configs/Eval/PID/Gemma/gemma_toxic.json",
        "evil":      "Configs/Eval/PID/Gemma/gemma_evil.json",
        "deception": "Configs/Eval/PID/Gemma/gemma_deception.json",
        "refusal":   "Configs/Eval/PID/Gemma/gemma_refusal_response.json",
    },
    "CURVE": {
        "toxic":     "Configs/Eval/CURVE/Gemma/gemma_toxic.json",
        "evil":      "Configs/Eval/CURVE/Gemma/gemma_evil.json",
        "deception": "Configs/Eval/CURVE/Gemma/gemma_deception.json",
        "refusal":   "Configs/Eval/CURVE/Gemma/gemma_refusal_response.json",
    },
    "REPS": {
        "toxic":     "Configs/Eval/REFT/gemma_toxic.json",
        "evil":      "Configs/Eval/REFT/gemma_evil.json",
        "deception": "Configs/Eval/REFT/gemma_deception.json",
        "refusal":   "Configs/Eval/REFT/reps_refusal_response.json",
    },
    "WEIGHT": {
        "toxic":     "Configs/Eval/WEIGHTSTEER/gemma_toxic.json",
        "evil":      "Configs/Eval/WEIGHTSTEER/gemma_evil.json",
        "deception": "Configs/Eval/WEIGHTSTEER/gemma_deception.json",
        "refusal":   "Configs/Eval/WEIGHTSTEER/gemma_refusal_response.json",
    },
}

torch.set_grad_enabled(False)


def run_task(name, cfg_path, coeffs, n_prompts, max_new_tokens):
    """Per-token KL + accuracy for one task via pipeline."""
    print(f"\n{'='*60}\n  Task: {name}\n{'='*60}")

    config = PipelineConfig.load(cfg_path)
    config.n_test = n_prompts
    config.model.max_new_tokens = max_new_tokens
    config.model.do_sample = False
    config.model.temperature = 0.0

    pipeline = SteeringPipeline(config)
    pipeline.setup()
    model = pipeline.model
    steer = pipeline.steer_model
    is_weight = isinstance(steer, WeightSteerModel)
    layers = steer.layer

    test_data = pipeline.load_test_data(
        dataset_name=config.test_dataset,
        n_samples=n_prompts,
        apply_chat_template=pipeline.steer_config.apply_chat_template,
    )
    prompts = [str(s.get("question", s)) for s in test_data]

    # Diagnostics
    v0 = steer.steering_vector
    if isinstance(v0, dict):
        v0 = v0[layers[0]]
    method_full = type(steer).__name__.replace("SteerModel", "")
    print(f"  [{method_full}] Layers: {layers} | Vector norm: {v0.norm():.4f} | Prompts: {len(prompts)}")

    evaluator = EVALUATOR_MAP[name](device=DEVICE)

    results = {}
    for coeff in coeffs:
        print(f"\n  --- coeff={coeff} ---")
        all_kl = []
        all_acc = []

        for idx, prompt in enumerate(prompts):
            prompt_tokens = model.to_tokens(prompt, truncate=True)
            generated = prompt_tokens.clone()
            per_pos_kl = []
            coeff_dict = {l: coeff for l in layers}

            for step in range(max_new_tokens):
                # BASE forward (no hooks)
                base_logits = model.run_with_hooks(generated, return_type="logits")
                base_lp = F.log_softmax(base_logits[0, -1, :], dim=-1)

                # STEERED forward via steer_model's native mechanism
                if is_weight:
                    steer._apply_weight_modifications(coeff_dict)
                    try:
                        steered_logits = model.run_with_hooks(generated, return_type="logits")
                    finally:
                        steer._restore_weights(coeff_dict)
                else:
                    steer.setup_hooks(coeff_dict)
                    try:
                        steered_logits = model.run_with_hooks(generated, return_type="logits")
                    finally:
                        model.reset_hooks()
                steered_lp = F.log_softmax(steered_logits[0, -1, :], dim=-1)

                kl = F.kl_div(steered_lp, base_lp, log_target=True, reduction='sum').item()
                per_pos_kl.append(kl)

                # Next token from BASE (same conditioning path for both)
                next_tok = base_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, next_tok], dim=1)

            # Free generation for accuracy via steer_model's native generate
            steered_text = steer.generate(
                prompt, coeff=coeff_dict,
                max_new_tokens=max_new_tokens,
                temperature=0.0, do_sample=False,
            )
            if isinstance(steered_text, list):
                steered_text = steered_text[0]

            acc, _ = evaluator.check(steered_text)
            all_acc.append(int(acc))
            all_kl.append(per_pos_kl)

            if (idx + 1) % 10 == 0:
                print(f"    [{time.strftime('%H:%M:%S')}] {idx+1}/{len(prompts)}", flush=True)

        # Aggregate
        max_len = max(len(kl) for kl in all_kl)
        kl_curve = [float(np.mean([kl[p] for kl in all_kl if p < len(kl)])) for p in range(max_len)]

        kl_c, kl_i = [], []
        for i, a in enumerate(all_acc):
            if a:
                kl_c.append(all_kl[i])
            else:
                kl_i.append(all_kl[i])

        def avg_kl(lst):
            if not lst:
                return []
            ml = max(len(k) for k in lst)
            return [float(np.mean([k[p] for k in lst if p < len(k)])) for p in range(ml)]

        kl_correct = avg_kl(kl_c)
        kl_incorrect = avg_kl(kl_i)
        mean_acc = float(np.mean(all_acc))

        print(f"    Acc: {mean_acc:.1%} ({sum(all_acc)}/{len(all_acc)})")
        print(f"    KL[0:10]: {[f'{k:.4f}' for k in kl_curve[:10]]}")
        if len(kl_curve) >= 6:
            e = float(np.mean(kl_curve[:6]))
            l = float(np.mean(kl_curve[5:]))
            print(f"    KL 0-5: {e:.6f} | 5+: {l:.6f} | Ratio: {e/max(l,1e-12):.2f}x")

        results[str(int(coeff))] = {
            "kl_curve": kl_curve,
            "kl_correct": kl_correct,
            "kl_incorrect": kl_incorrect,
            "accuracy": mean_acc,
            "n_correct": sum(all_acc),
            "n_total": len(all_acc),
            "kl_0_5": float(np.mean(kl_curve[:6])) if len(kl_curve) >= 6 else None,
            "kl_5plus": float(np.mean(kl_curve[5:])) if len(kl_curve) > 5 else None,
            "kl_ratio_early_late": float(np.mean(kl_curve[:6]) / max(np.mean(kl_curve[5:]), 1e-12)) if len(kl_curve) > 5 else None,
        }

    return results


def print_summary(all_results, method_name):
    print(f"\n{'='*80}")
    print(f"  SUMMARY: Per-Token KL | {method_name} | gemma-2-2b-it")
    print(f"{'='*80}")
    print(f"{'Task':<12} {'c':>4} {'Acc':>7} {'KL[0-5]':>10} {'KL[5+]':>10} {'Ratio':>8}  {'C/E KL[0-5]':>12} {'I/E KL[0-5]':>12}")
    print(f"{'-'*80}")
    for task, results in all_results.items():
        for c_str, d in results.items():
            e = d.get('kl_0_5')
            l = d.get('kl_5plus')
            r = d.get('kl_ratio_early_late')
            r_str = f"{r:.2f}x" if r else "N/A"
            e_str = f"{e:.6f}" if e else "N/A"
            l_str = f"{l:.6f}" if l else "N/A"
            ck = f"{np.mean(d['kl_correct'][:6]):.6f}" if d.get('kl_correct') and len(d['kl_correct']) >= 6 else "N/A"
            ik = f"{np.mean(d['kl_incorrect'][:6]):.6f}" if d.get('kl_incorrect') and len(d['kl_incorrect']) >= 6 else "N/A"
            print(f"{task:<12} {c_str:>4} {d['accuracy']:>6.1%} {e_str:>10} {l_str:>10} {r_str:>8}  {ck:>12} {ik:>12}")

    print(f"\n  Find correct K-L earlier?")
    print(f"  - If Toxic KL ratio >> Deception KL ratio -> early-token blocking explains ceiling")
    print(f"  - If Toxic KL ratio ~ Deception KL ratio -> different mechanism (output filter)")
    print(f"  - If Correct KL > Incorrect KL at early positions -> early logit change = success")


def parse_args():
    parser = argparse.ArgumentParser(description="ExpF: Per-token KL on steered outputs")
    parser.add_argument("--method", type=str, default="CAA",
                        choices=list(MULTI_CONFIGS.keys()),
                        help="Steering method to evaluate")
    parser.add_argument("--n_prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--coeffs", type=float, nargs="+", default=COEFFS,
                        help="Coefficients to sweep (e.g. 1.0 3.0 5.0)")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="CUDA device (use CUDA_VISIBLE_DEVICES instead)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

    method = args.method.upper()
    configs = MULTI_CONFIGS[method]

    print(f"Method: {method} | Device: {DEVICE} | Prompts: {args.n_prompts} | Tokens: {args.max_new_tokens} | Coeffs: {args.coeffs}")

    all_results = {}
    for task_name, cfg_path in configs.items():
        all_results[task_name] = run_task(
            task_name, cfg_path, args.coeffs, args.n_prompts, args.max_new_tokens
        )

    print_summary(all_results, method)

    all_results["metadata"] = {
        "method": method,
        "model": MODEL_NAME, "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens, "coeffs": args.coeffs,
        "configs": configs,
        "description": "Per-token KL(steered || base) via step-by-step generation with fixed prefix",
    }
    out_path = OUTPUT.replace(".json", f"_{method.lower()}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path} ({time.time()-t0:.1f}s)")
