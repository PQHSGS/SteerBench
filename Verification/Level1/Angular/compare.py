"""Compare Angular extraction vectors: position-aware comparison.

Compares reference (extract_directions.py) vs our (AngularExtractor) candidate directions.
Reference keys: layer_N_mid, layer_N_post
Our keys: layer_N (extracted at a single hook_point, default 'mid')

This script:
1. Shows what positions exist in each file
2. Compares mid-vs-mid, post-vs-post, and cross-position
3. Highlights where mismatches occur
"""
import torch
import torch.nn.functional as F
import argparse

def load_ref_vectors(path):
    data = torch.load(path, weights_only=False)
    return data

def load_my_vectors(path):
    data = torch.load(path, weights_only=False)
    if "metadata" in data and "candidate_directions" in data["metadata"]:
        return data["metadata"]["candidate_directions"]
    if "vector" in data:
        print("WARNING: 'candidate_directions' not found in metadata.")
        return data["vector"]
    return data

def cosine_sim(v1, v2):
    v1, v2 = v1.flatten().float(), v2.flatten().float()
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

def parse_key(k):
    """Parse key like 'layer_10_mid' -> (10, 'mid') or 'layer_10' -> (10, None)."""
    parts = k.split("_")
    layer_idx = int(parts[1])
    position = parts[2] if len(parts) > 2 else None
    return layer_idx, position

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Path to reference candidates.pt")
    parser.add_argument("--mine", required=True, help="Path to my vector.pt")
    args = parser.parse_args()

    ref_vecs = load_ref_vectors(args.ref)
    my_vecs = load_my_vectors(args.mine)

    # Analyze reference keys
    ref_parsed = {k: parse_key(k) for k in ref_vecs.keys()}
    my_parsed = {k: parse_key(k) for k in my_vecs.keys()}
    
    ref_positions = set(pos for _, pos in ref_parsed.values() if pos)
    my_positions = set(pos for _, pos in my_parsed.values() if pos)
    
    ref_layers = sorted(set(layer for layer, _ in ref_parsed.values()))
    my_layers = sorted(set(layer for layer, _ in my_parsed.values()))
    
    print(f"Reference: {len(ref_vecs)} keys, positions: {ref_positions or {'none (bare layer_N)'}}")
    print(f"Mine:      {len(my_vecs)} keys, positions: {my_positions or {'none (bare layer_N)'}}")
    print(f"Reference layers: {ref_layers}")
    print(f"My layers:        {my_layers}")
    print(f"Reference example keys: {list(ref_vecs.keys())[:5]}")
    print(f"My example keys:        {list(my_vecs.keys())[:5]}")

    # Build lookup: {(layer, position): tensor}
    ref_lookup = {}
    for k, v in ref_vecs.items():
        layer, pos = parse_key(k)
        ref_lookup[(layer, pos)] = v

    my_lookup = {}
    for k, v in my_vecs.items():
        layer, pos = parse_key(k)
        my_lookup[(layer, pos)] = v
    
    common_layers = sorted(set(ref_layers) & set(my_layers))
    
    if not common_layers:
        print("ERROR: No common layers!")
        return

    # Determine comparison pairs
    # If my keys have no position (bare layer_N), we compare against each ref position
    my_has_position = bool(my_positions)
    
    if my_has_position:
        # Both have positions: compare matching positions
        all_positions = sorted(ref_positions & my_positions)
        print(f"\nBoth have position suffixes. Comparing positions: {all_positions}")
    else:
        # My keys are bare (layer_N) — compare against each ref position separately
        all_positions = sorted(ref_positions) if ref_positions else [None]
        print(f"\nMy keys are bare (no position suffix).")
        print(f"Will compare my vectors against each reference position: {all_positions}")

    for pos in all_positions:
        print(f"\n{'='*60}")
        print(f"COMPARISON: mine vs reference position='{pos}'")
        print(f"{'='*60}")
        print(f"{'Layer':>5} | {'Cosine Sim':>10} | {'L2 Diff':>8} | {'|sim|':>6}")
        print("-" * 45)
        
        sims = []
        for layer in common_layers:
            # Get my vector
            my_key = (layer, pos if my_has_position else None)
            my_vec = my_lookup.get(my_key)
            if my_vec is None:
                print(f"{layer:5d} | {'MISSING':>10} | {'':>8} |")
                continue
            
            # Get ref vector
            ref_vec = ref_lookup.get((layer, pos))
            if ref_vec is None:
                print(f"{layer:5d} | {'REF MISS':>10} | {'':>8} |")
                continue
            
            if my_vec.shape != ref_vec.shape:
                print(f"{layer:5d} | SHAPE MISMATCH: {my_vec.shape} vs {ref_vec.shape}")
                continue
            
            sim = cosine_sim(my_vec, ref_vec)
            diff = torch.norm(my_vec.cpu().float() - ref_vec.cpu().float()).item()
            sims.append(abs(sim))
            marker = " ✗" if abs(sim) < 0.9 else " ✓"
            print(f"{layer:5d} | {sim:10.4f} | {diff:8.4f} | {abs(sim):6.4f}{marker}")
        
        if sims:
            avg = sum(sims) / len(sims)
            print("-" * 45)
            print(f"Average |cosine sim|: {avg:.4f}")
            low = sum(1 for s in sims if s < 0.9)
            print(f"Layers below 0.9: {low}/{len(sims)}")

    # Cross-position comparison (if ref has both mid and post)
    if len(ref_positions) >= 2 and not my_has_position:
        print(f"\n{'='*60}")
        print(f"CROSS-POSITION: ref 'mid' vs ref 'post' (to show position impact)")
        print(f"{'='*60}")
        print(f"{'Layer':>5} | {'mid-post sim':>12}")
        print("-" * 25)
        for layer in common_layers:
            mid_vec = ref_lookup.get((layer, "mid"))
            post_vec = ref_lookup.get((layer, "post"))
            if mid_vec is not None and post_vec is not None:
                sim = cosine_sim(mid_vec, post_vec)
                print(f"{layer:5d} | {sim:12.4f}")

if __name__ == "__main__":
    main()
