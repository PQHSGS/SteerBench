"""REPS Ablation 6: W and b Ablation

Fixes the R space to the learned R. Evaluates:
- Standard REPS (Learned W and b)
- Direct Ablation of W (W set to 0, use learned b)
- Retrained b-only (W frozen to 0 during training)
- Subspace CAA (empirical mean difference projected on R)
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

DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
evaluator = BehaviorMatcher(device=DEVICE, mode="evil")
LAYER = [14]
N_TRAIN = 500
N_TEST = 100
SEED = 42
COEFFS = [2, 3]
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

# ── Training Loop for REPS Modules ─────────────────────
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

# ── Setup and Save Metadata/Parameters ─────────────────
r = BASE_CFG["low_rank_dimension"]  # 8
d = model.cfg.d_model
META_DIR = REPO_ROOT / "Experiments" / "REPS_Ablation"
META_DIR.mkdir(parents=True, exist_ok=True)

# 1. Get/Load Standard REPS parameters
learned_meta_path = META_DIR / "meta_learned.pt"
if learned_meta_path.exists():
    print(f"Loading standard REPS from {learned_meta_path}")
    meta_learned = torch.load(learned_meta_path, weights_only=False)
else:
    print("="*60, "\nTraining Standard REPS (meta_learned)\n", "="*60)
    mod_learned = ReFTTrainModule(layers=LAYER, d_model=d, r=r, add_bias=True, device=DEVICE, dtype=model.cfg.dtype)
    train_reps(mod_learned, target_texts, contrast_texts, epochs=BASE_CFG["epochs"],
               batch_size=BASE_CFG["batch_size"], grad_accum=BASE_CFG["grad_accum"])
    meta_learned = {
        "rotate_basis": {14: mod_learned.rotate_layer["14"].weight.T.detach().clone()},
        "learned_weight": {14: mod_learned.learned_weight["14"].detach().clone()},
        "learned_bias": {14: mod_learned.learned_bias["14"].detach().clone()},
    }
    torch.save(meta_learned, learned_meta_path)
    print(f"Saved meta_learned to {learned_meta_path}")

R14_learned = meta_learned["rotate_basis"][14].to(DEVICE, dtype=DTYPE)
W14_learned = meta_learned["learned_weight"][14].to(DEVICE, dtype=DTYPE)
b14_learned = meta_learned["learned_bias"][14].to(DEVICE, dtype=DTYPE)

# 2. Train b-only model (W locked to 0)
b_only_meta_path = META_DIR / "meta_b_only.pt"
if b_only_meta_path.exists():
    print(f"Loading b-only REPS from {b_only_meta_path}")
    meta_b_only = torch.load(b_only_meta_path, weights_only=False)
else:
    print("="*60, "\nTraining b-only REPS (W frozen to 0)\n", "="*60)
    mod_b_only = ReFTTrainModule(layers=LAYER, d_model=d, r=r, add_bias=True, device=DEVICE, dtype=model.cfg.dtype)
    # Lock R to learned basis
    torch.nn.utils.parametrize.remove_parametrizations(mod_b_only.rotate_layer["14"], "weight", leave_parametrized=True)
    with torch.no_grad():
        mod_b_only.rotate_layer["14"].weight.copy_(R14_learned.T)
        mod_b_only.rotate_layer["14"].weight.requires_grad = False
        # Lock W to 0
        mod_b_only.learned_weight["14"].zero_()
        mod_b_only.learned_weight["14"].requires_grad = False
    
    train_reps(mod_b_only, target_texts, contrast_texts, epochs=BASE_CFG["epochs"],
               batch_size=BASE_CFG["batch_size"], grad_accum=BASE_CFG["grad_accum"])
    meta_b_only = {
        "rotate_basis": {14: R14_learned.cpu().clone()},
        "learned_weight": {14: mod_b_only.learned_weight["14"].detach().clone()},
        "learned_bias": {14: mod_b_only.learned_bias["14"].detach().clone()},
    }
    torch.save(meta_b_only, b_only_meta_path)
    print(f"Saved meta_b_only to {b_only_meta_path}")

W14_b_only = meta_b_only["learned_weight"][14].to(DEVICE, dtype=DTYPE)
b14_b_only = meta_b_only["learned_bias"][14].to(DEVICE, dtype=DTYPE)

# 3. Compute Subspace CAA and Standard CAA
print("Collecting dense activations for CAA calculations...")
acts_tgt = collect_dense_activations(model, target_texts, layers=LAYER, hook_point="pre",
    batch_size=2, pooling="last", device=DEVICE, tokenizer=model.tokenizer, reduce="none")[14]
acts_contrast = collect_dense_activations(model, contrast_texts, layers=LAYER, hook_point="pre",
    batch_size=2, pooling="last", device=DEVICE, tokenizer=model.tokenizer, reduce="none")[14]

v_caa = acts_tgt.mean(dim=0) - acts_contrast.mean(dim=0)
v_caa = v_caa.to(DEVICE, dtype=DTYPE)

# Subspace CAA: Project prompt mean difference onto R subspace
z_tgt = acts_tgt.to(DEVICE, dtype=DTYPE) @ R14_learned.T
z_contrast = acts_contrast.to(DEVICE, dtype=DTYPE) @ R14_learned.T
v_subspace = z_tgt.mean(dim=0) - z_contrast.mean(dim=0)
v_subspace = v_subspace.to(DEVICE, dtype=DTYPE)

# Save computed CAA/Subspace CAA vectors
torch.save({"v_caa": v_caa.cpu(), "v_subspace": v_subspace.cpu()}, META_DIR / "meta_caa.pt")
print(f"Saved CAA vectors to {META_DIR / 'meta_caa.pt'}")


# ── Evaluation Infrastructure ──────────────────────────
results_path = REPO_ROOT / "Results" / "reps_ablation" / "w_b_ablation.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
all_results = {}

def save_current_results():
    with open(results_path, "w") as f:
        json.dump({"method": "REPS_w_b_ablation", "results": all_results}, f, indent=2, default=str)
    print(f"Updated results saved to {results_path}")

def evaluate_hook(hook_name, hook_fn_maker, coeffs, condition_name):
    print("="*60, f"\nEvaluating Condition: {condition_name}\n", "="*60)
    acc = {}
    for c in coeffs:
        model.reset_hooks()
        model_hook = (hook_name, hook_fn_maker(c))
        with model.hooks([model_hook]):
            evil_count = 0
            for ex in tqdm(test_data, desc=f"{condition_name} c={c}"):
                response = model.generate(ex["question"], max_new_tokens=128, do_sample=False)
                prompt = ex["question"]
                clean_resp = response[len(prompt):] if response.startswith(prompt) else response
                is_correct, _ = evaluator.check(response=clean_resp, prompt=prompt)
                if is_correct == 1:
                    evil_count += 1
            acc[c] = evil_count / len(test_data)
            print(f"  c={c}: {acc[c]:.2%}")
    model.reset_hooks()
    all_results[condition_name] = acc
    save_current_results()

# ── Evaluate Conditions ───────────────────────────────

# Condition 1: Standard REPS (Learned W and b)
def make_reps_hook(c):
    steer = LoReFTSteerModel(
        model=model, layer=LAYER,
        steering_vector={14: torch.zeros(d, device=DEVICE)},
        rotate_basis={14: R14_learned},
        learned_weight={14: W14_learned},
        learned_bias={14: b14_learned},
        add_bias=True, hook_point=["pre"], position="last",
        substraction_type="zero", steer_once=False
    )
    def hook_fn(resid, hook):
        return steer.hook_fn(resid, c, steer.steering_vector[14], steer.position, hook)
    return hook_fn

evaluate_hook(get_hook_name(14, "pre"), make_reps_hook, COEFFS, "standard_reps")

# Condition 2: Direct Ablation of W (W set to 0, use learned b)
def make_direct_ablation_hook(c):
    steer = LoReFTSteerModel(
        model=model, layer=LAYER,
        steering_vector={14: torch.zeros(d, device=DEVICE)},
        rotate_basis={14: R14_learned},
        learned_weight={14: torch.zeros_like(W14_learned)},
        learned_bias={14: b14_learned},
        add_bias=True, hook_point=["pre"], position="last",
        substraction_type="zero", steer_once=False
    )
    def hook_fn(resid, hook):
        return steer.hook_fn(resid, c, steer.steering_vector[14], steer.position, hook)
    return hook_fn

evaluate_hook(get_hook_name(14, "pre"), make_direct_ablation_hook, COEFFS, "direct_ablation_w")

# Condition 3: Retrained b-only (W frozen to 0 during training)
def make_retrained_b_only_hook(c):
    steer = LoReFTSteerModel(
        model=model, layer=LAYER,
        steering_vector={14: torch.zeros(d, device=DEVICE)},
        rotate_basis={14: R14_learned},
        learned_weight={14: W14_b_only},
        learned_bias={14: b14_b_only},
        add_bias=True, hook_point=["pre"], position="last",
        substraction_type="zero", steer_once=False
    )
    def hook_fn(resid, hook):
        return steer.hook_fn(resid, c, steer.steering_vector[14], steer.position, hook)
    return hook_fn

evaluate_hook(get_hook_name(14, "pre"), make_retrained_b_only_hook, COEFFS, "retrained_b_only")

# Condition 4: Subspace CAA (empirical mean difference projected on R)
def make_subspace_caa_hook(c):
    steer = LoReFTSteerModel(
        model=model, layer=LAYER,
        steering_vector={14: torch.zeros(d, device=DEVICE)},
        rotate_basis={14: R14_learned},
        learned_weight={14: R14_learned},
        learned_bias={14: v_subspace},
        add_bias=True, hook_point=["pre"], position="last",
        substraction_type="zero", steer_once=False
    )
    def hook_fn(resid, hook):
        return steer.hook_fn(resid, c, steer.steering_vector[14], steer.position, hook)
    return hook_fn

evaluate_hook(get_hook_name(14, "pre"), make_subspace_caa_hook, COEFFS, "subspace_caa")

print("\n" + "="*60 + "\nFINAL W & B ABLATION RESULTS\n" + "="*60)
for cond, acc in all_results.items():
    print(f"{cond}: " + ", ".join(f"c={c}: {acc[c]:.2%}" for c in COEFFS))
