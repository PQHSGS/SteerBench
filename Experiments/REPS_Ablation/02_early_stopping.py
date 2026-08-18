"""REPS Ablation 2: Early Stopping

Single REPS training run to 20 epochs. Evaluate at epochs 1,3,5,10,20.
"""
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

def get_analytical_baseline(model, target_texts, contrast_texts, r):
    print("Computing Analytical Least-Squares REPS baseline...")
    X_acts = collect_dense_activations(model, contrast_texts, layers=LAYER, hook_point="pre",
        batch_size=BASE_CFG["batch_size"], pooling="last", device=DEVICE, tokenizer=model.tokenizer, reduce="none")[14]
    Y_acts = collect_dense_activations(model, target_texts, layers=LAYER, hook_point="pre",
        batch_size=BASE_CFG["batch_size"], pooling="last", device=DEVICE, tokenizer=model.tokenizer, reduce="none")[14]
    
    X = X_acts.float()
    Y = Y_acts.float()
    
    D = Y - X
    D_c = D - D.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(D_c, full_matrices=False)
    R = Vt[:r].T.contiguous()
    
    X_p = X @ R
    Y_p = Y @ R
    
    ones = torch.ones(X_p.shape[0], 1, device=X_p.device, dtype=X_p.dtype)
    A = torch.cat([X_p, ones], dim=1)
    
    K = torch.linalg.lstsq(A, Y_p).solution
    W_low = K[:r, :].T
    b_low = K[r, :]
    
    learned_weight = W_low @ R.T
    learned_bias = b_low
    
    return {
        "rotate_basis": {14: R.T.to(DTYPE)},
        "learned_weight": {14: learned_weight.to(DTYPE)},
        "learned_bias": {14: learned_bias.to(DTYPE)},
    }

# ── Train with checkpoints ────────────────────────────
r = BASE_CFG["low_rank_dimension"]
d = model.cfg.d_model
mod = ReFTTrainModule(layers=LAYER, d_model=d, r=r, add_bias=True, device=DEVICE, dtype=model.cfg.dtype)
set_seed(SEED)
optimizer = torch.optim.AdamW(mod.parameters(), lr=BASE_CFG["lr"], weight_decay=0.0)
n_steps = math.ceil(20 * math.ceil(N_TRAIN * 2 / 2) / 16)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(1, n_steps))
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
for p in model.parameters(): p.requires_grad = False
accum = 0; optimizer.zero_grad()
shuffle_rng = random.Random(SEED)
checkpoints = {}

for epoch in range(1, 21):
    epoch_data = []
    for i in range(N_TRAIN):
        epoch_data.append({"text": target_texts[i], "type": "add"})
        epoch_data.append({"text": contrast_texts[i], "type": "sub"})
    shuffle_rng.shuffle(epoch_data)
    pbar = tqdm(range(0, len(epoch_data), 2), desc=f"Epoch {epoch}/20")
    for idx in pbar:
        batch = epoch_data[idx:idx+2]
        texts = [b["text"] for b in batch]
        sf_list = [shuffle_rng.choice([1.0,2.0,3.0,5.0]) if b["type"]=="add" else 0.0 for b in batch]
        tokens, labels, attn_mask, prompt_lens = _get_completion_masked_labels(model, texts)
        tokens = tokens.to(DEVICE); labels = labels.to(DEVICE); attn_mask = attn_mask.to(DEVICE)
        plens = torch.tensor(prompt_lens, device=DEVICE)
        sf_t = torch.tensor(sf_list, device=DEVICE, dtype=DTYPE).unsqueeze(1)
        def make_hook(sf, pl):
            def fn(resid, hook):
                R = mod.rotate_layer["14"].weight.T.to(resid.dtype)
                W = mod.learned_weight["14"].to(resid.dtype); b = mod.learned_bias.get("14")
                idx = pl - 1; acts = resid[torch.arange(resid.shape[0]), idx]
                rb = acts @ R.T; so = acts @ W.T
                if b is not None: so = so + b.to(so.dtype)
                d = so - rb; upd = acts + sf * (d @ R)
                rn = resid.clone(); rn[torch.arange(resid.shape[0]), idx] = upd; return rn
            return fn
        with model.hooks([(get_hook_name(14, "pre"), make_hook(sf_t, plens))]):
            logits = model(tokens, attention_mask=attn_mask)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_val = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        (loss_val / 64).backward()
        pbar.set_postfix({"loss": f"{loss_val.item():.4f}"})
        accum += 1
        if accum % 64 == 0:
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
    if accum % 64 != 0:
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
    if epoch in [1, 3, 5, 10, 20]:
        checkpoints[epoch] = {
            "rotate_basis": {14: mod.rotate_layer["14"].weight.T.detach().clone().cpu()},
            "learned_weight": {14: mod.learned_weight["14"].detach().clone().cpu()},
            "learned_bias": {14: mod.learned_bias["14"].detach().clone().cpu()},
        }
        print(f"  >> Saved epoch {epoch}")

# ── Evaluate each checkpoint ─────────────────────────────
all_acc = {}
NAME = "early_stopping"
results_path = REPO_ROOT / "Results" / "reps_ablation" / f"{NAME}.json"
results_path.parent.mkdir(parents=True, exist_ok=True)

for ep, meta in checkpoints.items():
    print(f"\n{'='*60}\nEvaluating Epoch {ep}\n{'='*60}")
    meta = {k: {14: v[14].to(DEVICE, dtype=DTYPE)} for k, v in meta.items()}
    all_acc[ep] = evaluate_reps(meta, COEFFS)
    
    # Save intermediate results
    with open(results_path, "w") as f:
        json.dump({"method": f"REPS_{NAME}", "results": all_acc}, f, indent=2, default=str)
    print(f"Saved intermediate epoch {ep} results to {results_path}")

# Evaluate analytical baseline
analytical_meta = get_analytical_baseline(model, target_texts, contrast_texts, r)
print("\n" + "="*60 + "\nEvaluating Analytical REPS baseline\n" + "="*60)
all_acc["analytical"] = evaluate_reps(analytical_meta, COEFFS)

with open(results_path, "w") as f:
    json.dump({"method": f"REPS_{NAME}", "results": all_acc}, f, indent=2, default=str)
print(f"Saved final results to {results_path}")
