#!/bin/bash
set -euo pipefail

source /data/caotue/anaconda3/etc/profile.d/conda.sh
conda activate sae_circuit
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=5
export HF_HOME="/data/caotue/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data/caotue/hf_cache/hub"
export TRANSFORMERS_CACHE="/data/caotue/hf_cache/transformers"
export HF_DATASETS_CACHE="/data/caotue/hf_cache/datasets"
export TORCH_HOME="/data/caotue/torch_cache"
export TMPDIR="/data/caotue/tmp"

COEFFS=(1 2 3 5 7 10)
CONFIG_DIR="Configs/Eval/COBRA/Gemma"

# Only these 5 tasks
CONFIGS=(
  # "gemma_refusal_response.json"
  # "gemma_deception.json"
  "gemma_evil.json"
  # "gemma_toxic.json"
  "gemma_ifeval.json"
)

run_coeff_sweep() {
  local config_path="$1"
  local idx=0

  for coeff in "${COEFFS[@]}"; do
    local tmp_config
    tmp_config="$(mktemp /tmp/cobra_cfg_XXXXXX.json)"

    python - "$config_path" "$coeff" "$idx" > "$tmp_config" <<'PY'
import json, pathlib, sys
base_path = pathlib.Path(sys.argv[1])
coeff = float(sys.argv[2])
idx = int(sys.argv[3])
with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg.setdefault("steer", {})["coeff"] = coeff
cfg["include_baseline"] = False
if "model" in cfg:
    cfg["model"]["device"] = "cuda"
if idx > 0 and "save_vector" in cfg:
    cfg["load_vector"] = cfg.pop("save_vector")
print(json.dumps(cfg, indent=2))
PY

    echo ""
    echo "--- Running $(basename "$config_path") | coeff=$coeff ---"
    python -m Steering.cli --task eval --config "$tmp_config"

    rm -f "$tmp_config"
    idx=$((idx + 1))
    sleep 5
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_coeff_sweep "$CONFIG_DIR/$cfg"
done

echo ""
echo "=== COBRA benchmark complete ==="
