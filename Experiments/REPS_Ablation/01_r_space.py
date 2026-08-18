"""REPS Ablation 1: R Space — Learned vs PCA vs Random

Freeze R to three conditions. Train W,b from scratch each time.
Evaluate on Evil at c=2.
"""
import os
from pathlib import Path

REPO_ROOT = Path("/mnt/disk1/pquan/SAESteeringBench")
os.chdir(str(REPO_ROOT))

env_path = REPO_ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import sys, json, torch, math, random, warnings, gc, argparse
import numpy as np
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--eval-only", action="store_true", help="Skip training, load saved metas and eval only")
args, _ = parser.parse_known_args()

sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN", "").strip()
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
DEVICE = "cuda:0"
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

evaluator = BehaviorMatcher(device=DEVICE, mode="evil")

# ── Training loop (replicates LoReFTExtractor.extract) ──
def train_reps(module, target_texts, contrast_texts, epochs, batch_size=4, grad_accum=8, lr=0.001, seed=42):
    set_seed(seed)
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.0)
    n_steps = math.ceil(epochs * math.ceil(len(target_texts) * 2 / batch_size) / grad_accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max(1, n_steps))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for p in model.parameters(): p.requires_grad = False
    accum = 0; optimizer.zero_grad()
    shuffle_rng = random.Random(seed)

    for epoch in range(1, epochs + 1):
        epoch_data = []
        for i in range(len(target_texts)):
            epoch_data.append({"text": target_texts[i], "type": "add"})
            epoch_data.append({"text": contrast_texts[i], "type": "sub"})
        shuffle_rng.shuffle(epoch_data)
        pbar = tqdm(range(0, len(epoch_data), batch_size), desc=f"Epoch {epoch}/{epochs}")
        for idx in pbar:
            batch = epoch_data[idx:idx+batch_size]
            texts = [b["text"] for b in batch]
            sf_list = [shuffle_rng.choice([1.0,2.0,3.0,5.0]) if b["type"]=="add" else 0.0 for b in batch]
            tokens, labels, attn_mask, prompt_lens = _get_completion_masked_labels(model, texts)
            tokens = tokens.to(DEVICE); labels = labels.to(DEVICE); attn_mask = attn_mask.to(DEVICE)
            plens = torch.tensor(prompt_lens, device=DEVICE)
            sf_t = torch.tensor(sf_list, device=DEVICE, dtype=DTYPE).unsqueeze(1)

            def make_hook(sf, pl):
                def fn(resid, hook):
                    R = module.rotate_layer["14"].weight.T.to(resid.dtype)
                    W = module.learned_weight["14"].to(resid.dtype)
                    b = module.learned_bias.get("14")
                    idx = pl - 1
                    acts = resid[torch.arange(resid.shape[0]), idx]
                    rb = acts @ R.T; so = acts @ W.T
                    if b is not None: so = so + b.to(so.dtype)
                    d = so - rb; upd = acts + sf * (d @ R)
                    rn = resid.clone(); rn[torch.arange(resid.shape[0]), idx] = upd
                    return rn
                return fn
            with model.hooks([(get_hook_name(14, "pre"), make_hook(sf_t, plens))]):
                logits = model(tokens, attention_mask=attn_mask)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            (loss / grad_accum).backward()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            accum += 1
            if accum % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
        if accum % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
    return module

# ── Three R conditions ────────────────────────────────
r = BASE_CFG["low_rank_dimension"]  # 8
d = model.cfg.d_model
META_DIR = REPO_ROOT / "Experiments" / "REPS_Ablation"
META_DIR.mkdir(parents=True, exist_ok=True)

if not args.eval_only:
    # 1. Learned R (standard)
    print("="*60, "\n[1/3] Learned R (standard REPS)\n", "="*60)
    mod_learned = ReFTTrainModule(layers=LAYER, d_model=d, r=r, add_bias=True, device=DEVICE, dtype=model.cfg.dtype)
    train_reps(mod_learned, target_texts, contrast_texts, epochs=BASE_CFG["epochs"],
               batch_size=BASE_CFG["batch_size"], grad_accum=BASE_CFG["grad_accum"])
    meta_learned = {
        "rotate_basis": {14: mod_learned.rotate_layer["14"].weight.T.detach().clone()},
        "learned_weight": {14: mod_learned.learned_weight["14"].detach().clone()},
        "learned_bias": {14: mod_learned.learned_bias["14"].detach().clone()},
    }
    torch.save(meta_learned, META_DIR / "meta_learned.pt")
    print(f"Saved meta_learned to {META_DIR / 'meta_learned.pt'}")

    # 2. PCA R
    print("="*60, "\n[2/3] PCA R (frozen)\n", "="*60)
    acts = collect_dense_activations(model, target_texts, layers=LAYER, hook_point="pre",
        batch_size=BASE_CFG["batch_size"], pooling="last", device=DEVICE, tokenizer=model.tokenizer, reduce="none")[14]
    U, S, Vt = torch.linalg.svd(acts - acts.mean(dim=0, keepdim=True), full_matrices=False)
    R_pca = Vt[:r].T.contiguous()  # (d, r)
    mod_pca = ReFTTrainModule(layers=LAYER, d_model=d, r=r, add_bias=True, device=DEVICE, dtype=model.cfg.dtype)
    torch.nn.utils.parametrize.remove_parametrizations(mod_pca.rotate_layer["14"], "weight", leave_parametrized=True)
    with torch.no_grad():
        mod_pca.rotate_layer["14"].weight.copy_(R_pca)
        mod_pca.rotate_layer["14"].weight.requires_grad = False
    train_reps(mod_pca, target_texts, contrast_texts, epochs=BASE_CFG["epochs"],
               batch_size=BASE_CFG["batch_size"], grad_accum=BASE_CFG["grad_accum"])
    meta_pca = {
        "rotate_basis": {14: R_pca.T.contiguous()},
        "learned_weight": {14: mod_pca.learned_weight["14"].detach().clone()},
        "learned_bias": {14: mod_pca.learned_bias["14"].detach().clone()},
    }
    torch.save(meta_pca, META_DIR / "meta_pca.pt")
    print(f"Saved meta_pca to {META_DIR / 'meta_pca.pt'}")

    # 3. Random R
    print("="*60, "\n[3/3] Random R (frozen)\n", "="*60)
    M = torch.randn(d, r, device=DEVICE, dtype=torch.float32)
    R_rnd, _ = torch.linalg.qr(M)
    R_rnd = R_rnd.to(dtype=model.cfg.dtype)
    mod_rnd = ReFTTrainModule(layers=LAYER, d_model=d, r=r, add_bias=True, device=DEVICE, dtype=model.cfg.dtype)
    torch.nn.utils.parametrize.remove_parametrizations(mod_rnd.rotate_layer["14"], "weight", leave_parametrized=True)
    with torch.no_grad():
        mod_rnd.rotate_layer["14"].weight.copy_(R_rnd)
        mod_rnd.rotate_layer["14"].weight.requires_grad = False
    train_reps(mod_rnd, target_texts, contrast_texts, epochs=BASE_CFG["epochs"],
               batch_size=BASE_CFG["batch_size"], grad_accum=BASE_CFG["grad_accum"])
    meta_rnd = {
        "rotate_basis": {14: R_rnd.T.contiguous()},
        "learned_weight": {14: mod_rnd.learned_weight["14"].detach().clone()},
        "learned_bias": {14: mod_rnd.learned_bias["14"].detach().clone()},
    }
    torch.save(meta_rnd, META_DIR / "meta_rnd.pt")
    print(f"Saved meta_rnd to {META_DIR / 'meta_rnd.pt'}")
else:
    print("Loading saved metas from disk...")
    meta_learned = torch.load(META_DIR / "meta_learned.pt", weights_only=False)
    meta_pca = torch.load(META_DIR / "meta_pca.pt", weights_only=False)
    meta_rnd = torch.load(META_DIR / "meta_rnd.pt", weights_only=False)
    # Free GPU memory from model load if not needed for training
    print(f"Loaded: meta_learned, meta_pca, meta_rnd")

# ── Evaluate ────────────────────────────────────────────
@torch.inference_mode()
def generate_one(steer_model, prompt, max_new_tokens=128):
    """Manual generate that avoids transformers position_ids-on-CPU bug."""
    toks = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    prompt_len = toks.shape[1]
    generated = toks
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        if (next_tok == model.tokenizer.eos_token_id).all():
            break
    return model.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)

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

all_acc = {}
NAME = "r_space"
results_path = REPO_ROOT / "Results" / "reps_ablation" / f"{NAME}.json"
results_path.parent.mkdir(parents=True, exist_ok=True)

for label, meta in [("learned", meta_learned), ("pca", meta_pca), ("random", meta_rnd)]:
    print(f"\n{'='*60}\nEvaluating {label} R\n{'='*60}")
    all_acc[label] = evaluate_reps(meta, COEFFS)
    
    # Save intermediate results
    with open(results_path, "w") as f:
        json.dump({"method": f"REPS_{NAME}", "results": all_acc}, f, indent=2, default=str)
    print(f"Saved intermediate {label} results to {results_path}")
