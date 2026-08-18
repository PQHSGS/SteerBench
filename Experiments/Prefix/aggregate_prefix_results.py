"""Aggregate prefix-split experiment results."""
import json, glob

results = []

for fpath in sorted(glob.glob("Results/chars/eval_gemma_toxic_prefix_*.json") +
                    glob.glob("Results/chars/eval_gemma_evil_prefix_*.json") +
                    glob.glob("Results/linearact/eval_gemma_toxic_prefix_*.json") +
                    glob.glob("Results/linearact/eval_gemma_evil_prefix_*.json")):
    with open(fpath) as f:
        data = json.load(f)
    
    result = data.get("result", {})
    config = data.get("config", {})
    
    name = config.get("name", fpath)
    method = config.get("extractor", {}).get("method", "?")
    train_ds = config.get("train_dataset", "?")
    
    # Parse split from name
    parts = name.split("_")
    task = parts[1]  # toxic or evil
    split = parts[2]  # prefix_pos or prefix_neg
    
    acc = result.get("accuracy", "?")
    ppl = result.get("perplexity", "?")
    rep = result.get("repetition_rate", "?")
    comp = result.get("compression_ratio", "?")
    delta = result.get("delta", "?")
    n = result.get("n", "?")
    
    results.append({
        "method": method,
        "task": task,
        "split": split,
        "accuracy": acc,
        "perplexity": ppl,
        "repetition_rate": rep,
        "compression_ratio": comp,
        "delta": delta,
        "n": n,
        "file": fpath,
    })

# Print table
print(f"{'Method':8s} {'Task':8s} {'Split':12s} {'Acc':8s} {'PPL':8s} {'Rep':8s} {'Delta':8s}")
print("-" * 60)
for r in results:
    acc = f"{r['accuracy']:.2%}" if isinstance(r['accuracy'], (int, float)) else str(r['accuracy'])
    ppl = f"{r['perplexity']:.2f}" if r['perplexity'] else "?"
    rep = f"{r['repetition_rate']:.2%}" if r['repetition_rate'] else "?"
    delta = f"{r['delta']:+.2%}" if isinstance(r['delta'], (int, float)) else str(r['delta'])
    print(f"{r['method']:8s} {r['task']:8s} {r['split']:12s} {acc:8s} {ppl:8s} {rep:8s} {delta:8s}")
