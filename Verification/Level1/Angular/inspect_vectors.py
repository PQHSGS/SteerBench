
import torch
import sys

ref_path = "Verification/Level1/Angular/reference_output/Qwen1.5-0.5B-Chat/candidates.pt"
my_path = "Verification/Level1/Angular/my_output/vector.pt"

print(f"Loading ref: {ref_path}")
ref = torch.load(ref_path)
print(f"Loading mine: {my_path}")
mine = torch.load(my_path)
if "metadata" in mine:
    mine = mine["metadata"]["candidate_directions"]

# Normalize keys
def norm_key(k):
    p = k.split("_")
    if len(p)>=2 and p[0]=="layer": return f"layer_{p[1]}"
    return k

ref = {norm_key(k): v for k,v in ref.items()}
mine = {norm_key(k): v for k,v in mine.items()}

# layers = sorted([k for k in ref.keys() if k in mine], key=lambda x: int(x.split("_")[1]))
layers = [f"layer_{i}" for i in range(24)]

for l in layers:
    if l not in ref or l not in mine:
        print(f"Layer {l} not found in both")
        continue
    r = ref[l].float().cpu()
    m = mine[l].float().cpu()
    
    print(f"\n{l}:")
    print(f"  Ref shape: {r.shape}, Norm: {r.norm():.6f}, Mean: {r.mean():.6f}, Std: {r.std():.6f}")
    print(f"  My  shape: {m.shape}, Norm: {m.norm():.6f}, Mean: {m.mean():.6f}, Std: {m.std():.6f}")
    
    cosine = torch.nn.functional.cosine_similarity(r.flatten(), m.flatten(), dim=0).item()
    diff = torch.norm(r-m).item()
    print(f"  Cosine: {cosine:.6f}, Diff: {diff:.6f}")
    
    # Check for zeros
    if r.norm() == 0: print("  REF IS ZERO")
    if m.norm() == 0: print("  MY IS ZERO")
