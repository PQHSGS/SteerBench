"""
expD_kl.py — Unified KL divergence using SteeringPipeline.

Loads each method's config JSON, calls pipeline.setup() to handle
model/vector/steer model setup, captures residual streams during
generation, computes logit-lens KL at first/3rd/10th generated tokens.

Usage:
    conda activate sae_circuit
    CUDA_VISIBLE_DEVICES=2 python Experiments/ExpD/expD_kl.py
"""

import json, csv, torch, numpy as np, time, sys
from collections import defaultdict
from torch.nn import functional as F

sys.path.insert(0, ".")
from Steering.pipeline import SteeringPipeline
from Steering.config.pipeline import PipelineConfig
from Steering.steer_models import WeightSteerModel

COEFF = 1.0
N_PROMPTS = 30
MAX_NEW_TOKENS = 10
TRACK_LAYERS = list(range(8, 26))
POSITIONS = [0, 2, 9]
OUTPUT = "Experiments/ExpD/expD_kl_results.json"
TOXIC_DATA = "TrainDataset/behaviour/toxic/jigsaw/train.csv"

CONFIGS = [
    "Configs/Eval/CAA/Gemma/gemma_evil.json",
    "Configs/Eval/LinearAcT/Gemma/gemma_evil.json",
    "Configs/Eval/CHARS/Gemma/gemma_evil.json",
    "Configs/Eval/PID/Gemma/gemma_evil.json",
    "Configs/Eval/REFT/gemma_evil.json",
    "Configs/Eval/WEIGHTSTEER/gemma_evil.json",
]

torch.set_grad_enabled(False)


class CaptureCache:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers

    def install(self):
        self.residuals = {l: [] for l in self.layers}
        for l in self.layers:
            self.model.add_hook(
                f"blocks.{l}.hook_resid_post",
                lambda acts, hook, _l=l: (self.residuals[_l].append(acts[0, -1, :].detach().cpu()) or acts),
            )

    def snapshot(self):
        return {l: list(self.residuals[l]) for l in self.layers}


def load_prompts(path, n=30):
    texts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if not int(row.get("toxic", 0)) and len(texts) < n:
                texts.append(row["comment_text"])
    return texts


def logit_lens_kl(r_steer, r_base, model):
    dev = model.cfg.device
    ls = F.log_softmax(
        (model.ln_final(r_steer.to(dev)) @ model.W_U.detach()).float(), dim=-1
    )
    lb = F.log_softmax(
        (model.ln_final(r_base.to(dev)) @ model.W_U.detach()).float(), dim=-1
    )
    return F.kl_div(ls, lb, log_target=True, reduction='sum').item()


t0 = time.time()
prompts = load_prompts(TOXIC_DATA, N_PROMPTS)
print(f"[{time.time():.0f}] {len(prompts)} prompts loaded", flush=True)

results = {}

for cfg_path in CONFIGS:
    name = cfg_path.split("/")[-1].replace(".json", "")
    print(f"\n[{time.time():.0f}] === {name} ===", flush=True)

    config = PipelineConfig.load(cfg_path)
    config.steer.coeff = COEFF
    config.model.max_new_tokens = MAX_NEW_TOKENS
    config.model.do_sample = False
    config.n_test = N_PROMPTS

    pipeline = SteeringPipeline(config)
    pipeline.setup()
    model = pipeline.model
    steer = pipeline.steer_model
    coeff_dict = {l: COEFF for l in steer.layer}
    is_weight = isinstance(steer, WeightSteerModel)
    cache = CaptureCache(model, TRACK_LAYERS)

    all_kl = {pos: defaultdict(list) for pos in POSITIONS}

    for idx, prompt in enumerate(prompts):
        cache.install()
        model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, do_sample=False, verbose=False)
        model.reset_hooks()
        base = cache.snapshot()

        cache.install()
        if is_weight:
            steer._apply_weight_modifications(coeff_dict)
        else:
            steer.setup_hooks(coeff_dict)
        try:
            model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, do_sample=False, verbose=False)
        finally:
            model.reset_hooks()
            if is_weight:
                steer._restore_weights(coeff_dict)
        steered = cache.snapshot()

        for pos in POSITIONS:
            step = pos + 1
            if step >= len(base[TRACK_LAYERS[0]]):
                continue
            for l in TRACK_LAYERS:
                all_kl[pos][l].append(logit_lens_kl(
                    steered[l][step].unsqueeze(0), base[l][step].unsqueeze(0), model
                ))

        if (idx + 1) % 10 == 0:
            print(f"  [{time.time():.0f}] {idx+1}/{len(prompts)}", flush=True)

    results[name] = {}
    for pos in POSITIONS:
        results[name][f"pos_{pos}"] = {
            str(l): float(np.mean(all_kl[pos][l])) for l in TRACK_LAYERS
        }

    print(f"\n{'Layer':>6}", *[f"{'pos_'+str(p):>14}" for p in POSITIONS])
    for l in TRACK_LAYERS:
        ls = str(l)
        print(f"L{l:>4}", *[f"{results[name][f'pos_{p}'].get(ls,0):>14.6f}" for p in POSITIONS])
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT} ({time.time()-t0:.0f}s)", flush=True)
