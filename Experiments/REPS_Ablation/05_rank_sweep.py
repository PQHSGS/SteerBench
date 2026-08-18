"""REPS Ablation 5: Rank Sweep — r=1,2,4,8,16,32"""
import os
from pathlib import Path

REPO_ROOT = Path("/mnt/disk1/pquan/SAESteeringBench")
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(str(REPO_ROOT))

# os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))
# os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DATA_DIR / "hf_cache" / "hub"))
# os.environ.setdefault("TRANSFORMERS_CACHE", str(DATA_DIR / "hf_cache" / "transformers"))
# os.environ.setdefault("HF_DATASETS_CACHE", str(DATA_DIR / "hf_cache" / "datasets"))
# os.environ.setdefault("TORCH_HOME", str(DATA_DIR / "torch_cache"))
# os.environ.setdefault("TMPDIR", str(DATA_DIR / "tmp"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, json, torch, math, random, warnings, gc
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN", "").strip()
if not hf_token:
    token_path = DATA_DIR / "hf_cache" / "token"
    if token_path.exists():
        hf_token = token_path.read_text().strip()
if hf_token:
    login(token=hf_token)

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
    layer=LAYER, batch_size=1, position="last", apply_chat_template=True,
    hook_point=["pre"], dropout=0.2, act_fn="linear", add_bias=True,
    preference_pairs=["orig_add", "orig_sub"], substraction_type="zero",
    steering_factors=[1.0, 2.0, 3.0, 5.0], reft_seed=SEED, lr=0.001,
    weight_decay=0.0, epochs=20, reft_steer_once=True,
    low_rank_dimension=8, grad_accum=64,
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
def evaluate_reps(meta, coeffs):
    steer = LoReFTSteerModel(model=model, layer=LAYER,
        steering_vector={14: torch.zeros(model.cfg.d_model, device=DEVICE)},
        rotate_basis=meta["rotate_basis"], learned_weight=meta["learned_weight"],
        learned_bias=meta["learned_bias"], add_bias=True,
        hook_point=["pre"], position="last", substraction_type="zero",
        steer_once=False)
    acc = {}
    for c in coeffs:
        model.reset_hooks()
        steer.setup_hooks({14: c})
        evil_count = 0
        for ex in tqdm(test_data, desc=f"c={c}"):
            steer._reset_prompt_metadata()
            response = model.generate(ex["question"], max_new_tokens=128, do_sample=False)
            prompt = ex["question"]
            clean_resp = response[len(prompt):] if response.startswith(prompt) else response
            is_correct, _ = evaluator.check(response=clean_resp, prompt=prompt)
            if is_correct == 1:
                evil_count += 1
        acc[c] = evil_count / len(test_data)
        print(f"  c={c}: {acc[c]:.2%}")
    model.reset_hooks()
    return acc

# ── Rank sweep ──────────────────────────────────────────
all_acc = {}
NAME = "rank_sweep"
results_path = REPO_ROOT / "Results" / "reps_ablation" / f"{NAME}.json"
results_path.parent.mkdir(parents=True, exist_ok=True)

for r in [1, 2, 4, 8, 16, 32]:
    print(f"\n{'='*60}\nRank r={r}\n{'='*60}")
    cfg = dict(BASE_CFG); cfg["reft_low_rank_dimension"] = r
    ex = LoReFTExtractor(model=model, **cfg)
    ex.extract(target_texts, contrast_texts)
    all_acc[str(r)] = evaluate_reps(ex.metadata, COEFFS)
    
    # Save intermediate results
    with open(results_path, "w") as f:
        json.dump({"method": f"REPS_{NAME}", "results": all_acc}, f, indent=2, default=str)
    print(f"Saved intermediate rank r={r} results to {results_path}")

print("\n" + "="*60 + "\nRANK SWEEP\n" + "="*60)
for r in [1, 2, 4, 8, 16, 32]:
    if str(r) in all_acc:
        print(f"r={r}: " + ", ".join(f"c={c}={all_acc[str(r)][c]:.2%}" for c in COEFFS))
