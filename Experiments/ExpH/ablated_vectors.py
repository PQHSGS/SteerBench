"""Create r̂-ablated vectors: v' = v - (v·r̂)r̂, renormalized. Saves to ablated/ subdirs."""
import sys; sys.path.insert(0, ".")
import torch, json
from pathlib import Path
from transformer_lens import HookedTransformer

device = "cuda:0"
dtype = torch.bfloat16
model_name = "google/gemma-2-2b-it"
LAYERS = [14, 18, 22, 25]

TASKS = {
    "evil":       "Vector/CAA/Gemma/evil",
    "toxic":      "Vector/CAA/Gemma/toxic",
    "deception":  "Vector/CAA/Gemma/deception",
}

print("Loading model for r̂ extraction...")
model = HookedTransformer.from_pretrained(model_name, dtype=dtype, device=device)
model.eval()

# Load refusal response text
with open("TrainDataset/behaviour/refusal/CAST/behaviour_refusal.json") as f:
    resp_data = json.load(f)
non_compliant = resp_data["non_compliant_responses"]
compliant = resp_data["compliant_responses"]

BATCH_SIZE = 8
def get_mean_act(texts, L):
    """Get mean activation at last token for a single layer."""
    total = None
    count = 0
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        tokens = model.to_tokens(batch)
        cache = {}
        model.reset_hooks()
        def cap(act, **kw):
            if "acts" not in cache:
                cache["acts"] = []
            cache["acts"].append(act[:, -1, :].clone().detach())
            return act
        model.add_hook(f"blocks.{L}.hook_resid_pre", cap, "fwd")
        with torch.no_grad(): _ = model(tokens)
        B = tokens.size(0)
        act_sum = cache["acts"][0].sum(dim=0)
        total = act_sum if total is None else total + act_sum
        count += B
    return total / count

# Extract r̂ at each layer
r_vecs = {}
for L in LAYERS:
    h_refuse = get_mean_act(non_compliant, L)
    h_comply = get_mean_act(compliant, L)
    r = (h_refuse - h_comply).to(dtype=dtype)
    r_vecs[L] = r
    print(f"r̂ layer {L}: norm={r.norm():.4f}")

# For each task and layer: create ablated vector
for task_name, vec_dir in TASKS.items():
    vec_file = Path(vec_dir) / "vector.pt"
    vec_dict = torch.load(vec_file, map_location=device)
    
    ablated_dir = Path(vec_dir.replace("Gemma", "Gemma_ablated"))
    ablated_dir.mkdir(parents=True, exist_ok=True)
    ablated_dict = {}
    
    for L in LAYERS:
        if L not in vec_dict:
            continue
        v = vec_dict[L].to(dtype=dtype)
        r = r_vecs[L]
        
        # Project out r̂ component
        v_r_dot = torch.dot(v, r) / (r.norm() ** 2 + 1e-8)
        v_ablate = v - v_r_dot * r
        
        # Renormalize to original norm
        orig_norm = v.norm()
        v_ablate = v_ablate / (v_ablate.norm() + 1e-8) * orig_norm
        
        # Compute cos after ablation
        cos_after = torch.dot(v_ablate, r) / (v_ablate.norm() * r.norm() + 1e-8)
        
        ablated_dict[L] = v_ablate.to(dtype=dtype)
        print(f"{task_name} L{L}: orig cos(v,r̂)={torch.dot(v,r)/(v.norm()*r.norm()):+.4f}  →  after ablation cos(v',r̂)={cos_after:+.4e}")
    
    # Copy metadata
    meta_file = Path(vec_dir) / "metadata.pt"
    if meta_file.exists():
        meta = torch.load(meta_file, map_location="cpu")
        torch.save(meta, ablated_dir / "metadata.pt")
    else:
        torch.save({}, ablated_dir / "metadata.pt")
    
    # Save ablated vector
    torch.save(ablated_dict, ablated_dir / "vector.pt")
    print(f"  Saved ablated vectors to {ablated_dir}/")

del model; torch.cuda.empty_cache()
print("\nDone.")
