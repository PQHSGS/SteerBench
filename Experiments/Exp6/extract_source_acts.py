"""
Extract source activations once per task, save to disk.
Usage: python extract_source_acts.py <task_name>
  task_name: toxic | evil
"""
import sys, os, json, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "3")

from Steering.data.loader import DataLoader
from transformer_lens import HookedTransformer

MODEL_NAME = "google/gemma-2-2b-it"
LAYER = 14
N_SAMPLES = 500
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_CONFIGS = {
    "toxic": {"dataset": "toxic_jigsaw", "source_key": "false_prompt"},
    "evil": {"dataset": "evil", "source_key": "false_prompt"},
}

def main():
    task_name = sys.argv[1]
    assert task_name in TASK_CONFIGS, f"Unknown task: {task_name}"
    cfg = TASK_CONFIGS[task_name]
    
    out_path = Path(f"Vector/CHARS/Gemma/{task_name}_source_acts.pt")
    if out_path.exists():
        print(f"Source activations already exist: {out_path}")
        return
    
    print(f"Loading model...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, device=DEVICE, dtype=torch.bfloat16,
        default_padding_side="left",
    )
    model.to(DEVICE)
    
    print(f"Loading {cfg['dataset']}...")
    loader = DataLoader()
    data = loader.load(cfg["dataset"], n_samples=N_SAMPLES, format=True, apply_chat_template=False)
    source_texts = [d[cfg["source_key"]] for d in data]
    print(f"  {len(source_texts)} source texts")
    
    # Extract activations
    all_acts = []
    for i in range(0, len(source_texts), BATCH_SIZE):
        batch = source_texts[i:i+BATCH_SIZE]
        batch = [model.tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
        ) for t in batch]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch, names_filter=lambda n: n == f"blocks.{LAYER}.hook_resid_pre",
                prepend_bos=True,
            )
        acts = cache[f"blocks.{LAYER}.hook_resid_pre"][:, -1, :]
        all_acts.append(acts.detach().cpu())
    
    source_acts = torch.cat(all_acts, dim=0).float()
    print(f"  Activations shape: {source_acts.shape}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source_acts, str(out_path))
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
