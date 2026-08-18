"""REPS Ablation 4: Steering Factors — Single vs Multi-Coeff Training"""
import os
from pathlib import Path



import sys, json, torch, math, random, warnings, gc
import numpy as np
warnings.filterwarnings('ignore')
# os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))
# os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_DIR / "hf_cache" / "hub"))
# os.environ.setdefault("TRANSFORMERS_CACHE", str(DATA_DIR / "hf_cache" / "transformers"))
# os.environ.setdefault("HF_DATASETS_CACHE", str(DATA_DIR / "hf_cache" / "datasets"))
# os.environ.setdefault("TORCH_HOME", str(DATA_DIR / "torch_cache"))
# os.environ.setdefault("TMPDIR", str(DATA_DIR / "tmp"))
REPO_ROOT = Path("/mnt/disk1/pquan/SAESteeringBench")
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN", "").strip()

from Steering.evaluators import BehaviorMatcher
from transformer_lens import HookedTransformer
from Steering.data import DataLoader, EvalDataLoader
from Steering.extractors.nonlinear import LoReFTExtractor, ReFTTrainModule
from Steering.steer_models.nonlinear import LoReFTSteerModel
from Steering.utils import get_hook_name, get_resid_acts, set_resid_acts, collect_dense_activations
from Steering.extractors.nonlinear import _get_completion_masked_labels
from transformers import get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

MODEL_NAME = "google/gemma-2-2b-it"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
evaluator = BehaviorMatcher(device=DEVICE, mode="evil")
LAYER = [14]
N_TRAIN = 500
N_TEST = 100
SEED = 42
COEFFS = [2]
BASE_CFG = dict(
    layer=LAYER, batch_size=2, position="last", apply_chat_template=True,
    hook_point=["pre"], dropout=0.2, act_fn="linear", add_bias=True,
    preference_pairs=["orig_add", "orig_sub"], substraction_type="zero",
    steering_factors=[1.0, 2.0, 3.0, 5.0], reft_seed=SEED, lr=0.001,
    weight_decay=0.0, epochs=20, reft_steer_once=True,
    low_rank_dimension=8, grad_accum=16,
)

# ── Load Model & Data ──────────────────────────────────
torch.cuda.empty_cache(); gc.collect()
model = HookedTransformer.from_pretrained(MODEL_NAME, dtype=DTYPE, device=DEVICE)
model.eval()

loader = DataLoader()
train_data = loader.load("evil", n_samples=N_TRAIN, apply_chat_template=True, tokenizer=model.tokenizer)
cfg = loader.get_config("evil")
target_texts = [d[cfg.target_key] for d in train_data]
contrast_texts = [d[cfg.contrast_key] for d in train_data]

eval_loader = EvalDataLoader()
test_data = eval_loader.load("evil", n_samples=N_TEST, format=True,
                              apply_chat_template=True, tokenizer=model.tokenizer)

# ── Evaluate ────────────────────────────────────────────
def compute_lsp_score(model, prompts, responses):
    import math
    per_sample_nlls = []
    per_sample_tokens = []
    
    for prompt, response in zip(prompts, responses):
        if response.startswith(prompt):
            clean_resp = response[len(prompt):]
        else:
            clean_resp = response
            
        prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
        full_tokens = model.to_tokens(prompt + clean_resp, prepend_bos=True)
        
        if full_tokens.shape[1] > 512:
            full_tokens = full_tokens[:, :512]
        
        prompt_len = prompt_tokens.shape[1]
        full_tokens = full_tokens.to(model.cfg.device)
        
        with torch.no_grad():
            loss_per_token = model(
                full_tokens,
                return_type="loss",
                loss_per_token=True,
            )
            response_loss = loss_per_token[:, prompt_len - 1 :]
            nlls = response_loss[0].tolist()
            tokens = full_tokens[0, prompt_len :].tolist()
            
            per_sample_nlls.append(nlls)
            per_sample_tokens.append(tokens)
            
    window_size = 30
    all_losses = []
    
    for nlls, tokens in zip(per_sample_nlls, per_sample_tokens):
        if not nlls or not tokens or len(nlls) != len(tokens) or len(tokens) <= 1:
            continue
        for t in range(len(tokens)):
            raw_loss = nlls[t]
            start_idx = max(0, t - window_size)
            history = tokens[start_idx:t]
            
            lcs = 0
            for idx in range(len(history) - 1):
                length = 0
                while idx - length >= 0 and history[idx - length] == history[-1 - length]:
                    length += 1
                if length > lcs:
                    lcs = length
            
            penalty = 2.0 * math.log1p(max(0, lcs - 2))
            all_losses.append(raw_loss + penalty)
            
    if not all_losses:
        return 0.0
        
    mean_loss = sum(all_losses) / len(all_losses)
    try:
        lsp_score = math.exp(mean_loss)
    except OverflowError:
        lsp_score = float("inf")
        
    return lsp_score



# ── Extraction/Training Phase ──────────────────────────
all_metas = {}
for label, factors in [("single", [1.0]), ("multi", [1.0, 2.0, 3.0, 5.0])]:
    print(f"\n{'='*60}\n{label} factors\n{'='*60}")
    cfg = dict(BASE_CFG); cfg["steering_factors"] = factors
    ex = LoReFTExtractor(model=model, **cfg)
    ex.extract(target_texts, contrast_texts)
    all_metas[label] = ex.metadata

# ── Generation Phase ───────────────────────────────────
all_generated_responses = {}
all_lsp_scores = {}

for label in ["single", "multi"]:
    print(f"\n{'='*60}\nGenerating for {label} factors\n{'='*60}")
    meta = all_metas[label]
    steer = LoReFTSteerModel(model=model, layer=LAYER,
        steering_vector={14: torch.zeros(model.cfg.d_model, device=DEVICE)},
        rotate_basis=meta["rotate_basis"], learned_weight=meta["learned_weight"],
        learned_bias=meta["learned_bias"], add_bias=True,
        hook_point=["pre"], position="last", substraction_type="zero",
        steer_once=False)
    for c in COEFFS:
        model.reset_hooks()
        steer.setup_hooks({14: c})
        responses = []
        prompts = []
        for ex in tqdm(test_data, desc=f"Generating {label} c={c}"):
            steer._reset_prompt_metadata()
            out = model.generate(ex["question"], max_new_tokens=128, do_sample=False)
            responses.append(out)
            prompts.append(ex["question"])
        all_generated_responses[(label, c)] = responses
        model.reset_hooks()
        all_lsp_scores[(label, c)] = compute_lsp_score(model, prompts, responses)

# Unload generator model
del model
import gc
gc.collect()
torch.cuda.empty_cache()

# ── Evaluation Phase ───────────────────────────────────
all_acc = {}
NAME = "steering_factors"
results_path = REPO_ROOT / "Results" / "reps_ablation" / f"{NAME}.json"
results_path.parent.mkdir(parents=True, exist_ok=True)

for label in ["single", "multi"]:
    print(f"\n{'='*60}\nEvaluating {label} factors\n{'='*60}")
    acc = {}
    for c in COEFFS:
        responses = all_generated_responses[(label, c)]
        evil_count = 0
        for ex, response in zip(test_data, responses):
            prompt = ex["question"]
            clean_resp = response[len(prompt):] if response.startswith(prompt) else response
            is_correct, _ = evaluator.check(response=clean_resp, prompt=prompt)
            if is_correct == 1:
                evil_count += 1
        acc[c] = evil_count / len(test_data)
        lsp_score = all_lsp_scores[(label, c)]
        print(f"  c={c}: accuracy: {acc[c]:.2%}, ood: {lsp_score:.2f}")
    all_acc[label] = acc
    
    # Save intermediate results
    with open(results_path, "w") as f:
        json.dump({"method": f"REPS_{NAME}", "results": all_acc}, f, indent=2, default=str)
    print(f"Saved intermediate {label} results to {results_path}")

# Unload evaluator
evaluator.unload()

for c in COEFFS:
    if "single" in all_acc and "multi" in all_acc:
        print(f"c={c}: single={all_acc['single'][c]:.2%}, multi={all_acc['multi'][c]:.2%}")
