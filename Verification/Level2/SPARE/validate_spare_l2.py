"""
SPARE Level 2 validation (Steering parity).

Rules enforced:
- Uses validated extracted vector (load_vector) from Level 1.
- Ours is built through SteeringPipeline and generation uses pipeline.generate().
- GT side uses a simple direct hook function (no demo helper imports).
- Compares semantic generation plus optional logit distributions.

Usage:
  unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit
  PYTHONPATH=. python Verification/Level2/SPARE/validate_spare_l2.py --n_samples 1000 --layer 14
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Steering import PipelineConfig, ModelConfig, ExtractorConfig, SteerConfig, SteeringPipeline
from Steering.data import EvalDataLoader


MODEL_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
COEFF = 2.0
MAX_NEW_TOKENS = 12


def _safe_load_tensor(path: Path, map_location: str = "cpu") -> torch.Tensor:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _build_pipeline(vector_path: Path, layers: List[int], device: str) -> SteeringPipeline:
    coeff_dict = {l: COEFF for l in layers}
    config = PipelineConfig(
        model=ModelConfig(
            name=MODEL_NAME,
            device=device,
            dtype="bfloat16",
        ),
        extractor=ExtractorConfig(
            method="SPARE",
            layer=layers,
            hook_point=["post"],
            top_k_proportion=0.07,
        ),
        steer=SteerConfig(
            method="SPARE",
            layer=layers,
            coeff=coeff_dict,
            hook_point=["post"],
            target_behavior="contextual",
        ),
        test_dataset="nqswap_test",
        n_test=20,
        load_vector=str(vector_path),
    )

    pipeline = SteeringPipeline(config)
    pipeline.load_model("sae")
    pipeline.steering()
    return pipeline


def _gt_spare_hook_factory(
    sae,
    zc: torch.Tensor,
    zm: torch.Tensor,
    idx_pos: torch.Tensor,
    idx_neg: torch.Tensor,
    coeff: float,
):
    idx_pos = idx_pos.long()
    idx_neg = idx_neg.long()
    zc = zc.to(dtype=sae.W_enc.dtype, device=sae.W_enc.device)
    zm = zm.to(dtype=sae.W_enc.dtype, device=sae.W_enc.device)

    def _hook(activations: torch.Tensor, hook) -> torch.Tensor:
        h = activations
        sae_in = h.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        if hasattr(sae, "b_dec"):
            sae_in = sae_in - sae.b_dec

        z_pre = torch.relu(sae_in @ sae.W_enc + sae.b_enc)
        z_add = torch.zeros_like(z_pre)
        z_rem = torch.zeros_like(z_pre)

        if idx_pos.numel() > 0:
            z_add[:, :, idx_pos] = torch.clamp(
                zc[idx_pos] * coeff - z_pre[:, :, idx_pos],
                min=0,
            )
        if idx_neg.numel() > 0:
            z_rem[:, :, idx_neg] = torch.minimum(
                z_pre[:, :, idx_neg],
                zm[idx_neg] * coeff,
            )

        add = sae.decode(z_add)
        rem = sae.decode(z_rem)
        return h - rem.to(h.dtype) + add.to(h.dtype)

    return _hook


def _run_ours_logits(pipeline: SteeringPipeline, layers: List[int], prompt: str) -> torch.Tensor:
    model = pipeline.model
    steer_model = pipeline.steer_model
    toks = model.to_tokens(prompt)
    steer_model.setup_hooks(coeff={l: COEFF for l in layers})
    with torch.no_grad():
        logits = model(toks)[0, -1, :].float().cpu()
    model.reset_hooks()
    return logits


_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}
_NUM_DIGITS = {v: k for k, v in _NUM_WORDS.items()}


def _normalize_numbers(text: str) -> str:
    words = text.lower().split()
    out = []
    for w in words:
        if w in _NUM_WORDS:
            out.append(_NUM_WORDS[w])
        else:
            out.append(w)
    return " ".join(out)


def _contains_answer(text: str, answer) -> bool:
    candidates = answer if isinstance(answer, list) else [answer]
    text_norm = _normalize_numbers(text.lower().strip())
    return any(_normalize_numbers(str(candidate).lower()) in text_norm for candidate in candidates)


def _semantic_overlap(a: str, b: str, answer) -> float:
    # strip special tokens before comparing
    for tok in ("<end_of_turn>", "<eos>", "<bos>", "<pad>"):
        a = a.replace(tok, "")
        b = b.replace(tok, "")
    a_norm = _normalize_numbers(a.lower().strip())
    b_norm = _normalize_numbers(b.lower().strip())
    if a_norm == b_norm:
        return 1.0

    a_match = _contains_answer(a, answer)
    b_match = _contains_answer(b, answer)
    if a_match == b_match:
        return 1.0

    ta = set(a_norm.split())
    tb = set(b_norm.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARE Level2 parity validation")
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
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 64)
    print("SPARE Level 2 (Steering parity) on Gemma-2-2B")
    print("=" * 64)

    # Ensure GT artifacts exist; re-run extract-only if missing.
    gt_dir = ROOT / "Results" / f"l3_spare_gt_{args.n_samples}_{args.seed}"
    if not gt_dir.exists():
        cmd = [
            sys.executable,
            "Verification/Level3/SPARE/run_gt.py",
            f"--n_samples={args.n_samples}",
            f"--seed={args.seed}",
            f"--device={args.device}",
            "--layers",
            *[str(layer) for layer in args.layers],
            "--extract_only",
        ]
        ret = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if ret.returncode != 0:
            raise RuntimeError(f"GT CLI failed with exit code {ret.returncode}")

    vector_path = ROOT / "Results" / f"l1_spare_ours_vector_{args.n_samples}_{args.seed}.pt"
    if not vector_path.exists():
        raise FileNotFoundError(
            f"Missing Level1 validated vector: {vector_path}. Run Level1 first."
        )

    pipeline = _build_pipeline(vector_path, args.layers, args.device)
    model = pipeline.model

    # Load GT artifacts and build per-layer hooks.
    # We run eval against a single representative layer (last in list) for
    # the GT hook comparison — GT hooks are per-layer; we verify that the
    # pipeline adds up identically across ALL specified layers.
    all_saes = pipeline.load_sae(args.layers)

    gt_hooks_per_layer: Dict[int, Tuple] = {}  # layer -> (hook_name, hook_fn)
    for layer in args.layers:
        diag_path = gt_dir / f"gt_diagnostics_L{layer}.json"
        if not diag_path.exists():
            raise FileNotFoundError(f"Missing GT diagnostics: {diag_path}")
        diag = json.loads(diag_path.read_text(encoding="utf-8"))

        zc = _safe_load_tensor(gt_dir / f"spare_zC_L{layer}.pt", map_location=args.device)
        zm = _safe_load_tensor(gt_dir / f"spare_zM_L{layer}.pt", map_location=args.device)
        idx_pos = torch.tensor(diag.get("indices_pos", []), device=args.device)
        idx_neg = torch.tensor(diag.get("indices_neg", []), device=args.device)

        hook_name = f"blocks.{layer}.hook_resid_post"
        hook_fn = _gt_spare_hook_factory(
            sae=all_saes[layer],
            zc=zc,
            zm=zm,
            idx_pos=idx_pos,
            idx_neg=idx_neg,
            coeff=COEFF,
        )
        gt_hooks_per_layer[layer] = (hook_name, hook_fn)

    eval_data = EvalDataLoader().load("nqswap_test", args.n_eval, apply_chat_template=False)

    logit_cos_scores: List[float] = []
    semantic_scores: List[float] = []

    print(f"Comparing {len(eval_data)} prompts across layers {args.layers} ...")
    for i, sample in enumerate(eval_data):
        prompt = sample["correct_prompt"]
        answer = sample.get("answer", sample.get("ground_truth", ""))

        # GT: run with ALL per-layer hooks simultaneously
        fwd_hooks = list(gt_hooks_per_layer[l] for l in args.layers)
        toks = model.to_tokens(prompt)
        with torch.no_grad():
            gt_logits = model.run_with_hooks(
                toks, fwd_hooks=fwd_hooks
            )[0, -1, :].float().cpu()
        model.reset_hooks()

        our_logits = _run_ours_logits(pipeline, args.layers, prompt)

        logit_cos = float(F.cosine_similarity(gt_logits.unsqueeze(0), our_logits.unsqueeze(0)).item())
        logit_cos_scores.append(logit_cos)

        # GT generate with all hooks
        with model.hooks(fwd_hooks=fwd_hooks):
            out = model.generate(toks, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, verbose=False)
        gt_text = model.tokenizer.decode(out[0, toks.shape[1]:], skip_special_tokens=True)
        model.reset_hooks()

        our_text = pipeline.generate(
            prompt,
            coeff={l: COEFF for l in args.layers},
            max_new_tokens=MAX_NEW_TOKENS,
            apply_steer=True,
            do_sample=False,
        )
        if isinstance(our_text, list):
            our_text = our_text[0] if our_text else ""
        sem = _semantic_overlap(gt_text, our_text, answer)
        semantic_scores.append(sem)
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"GT generation: {gt_text}")
        print(f"Ours generation: {our_text}")
        print(f"  [{i:02d}] logit_cos={logit_cos:.6f}, semantic_overlap={sem:.6f}")

    avg_logit = float(np.mean(logit_cos_scores)) if logit_cos_scores else 0.0
    avg_sem = float(np.mean(semantic_scores)) if semantic_scores else 0.0

    checks = {
        "avg_logit_cosine": avg_logit >= args.threshold,
        "avg_semantic_overlap": avg_sem >= args.threshold,
    }

    print("\nMetrics:")
    print(f"  avg_logit_cosine: {avg_logit:.6f}")
    print(f"  avg_semantic_overlap: {avg_sem:.6f}")

    all_passed = True
    print("\nChecks:")
    for name, ok in checks.items():
        if not ok:
            all_passed = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    summary = {
        "level": 2,
        "method": "SPARE",
        "model": MODEL_NAME,
        "layers": args.layers,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "threshold": args.threshold,
        "metrics": {
            "avg_logit_cosine": avg_logit,
            "avg_semantic_overlap": avg_sem,
        },
        "passed": all_passed,
        "vector_path": str(vector_path),
        "gt_dir": str(gt_dir),
    }
    out_summary = ROOT / "Results" / f"spare_l2_summary_{args.n_samples}_{args.seed}.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_summary}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
