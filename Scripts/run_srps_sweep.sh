#!/usr/bin/env bash
# Hyperparameter sweep for SRPS on refusal_response (act_threshold and beta)

set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
ACT_THRESHOLDS=(10 20 50 100)
BETAS=(1.0 2.0 5.0 10.0)
CONFIG_FILE="Configs/Eval/SRPS/Gemma/gemma_refusal_response.json"
RESULTS_DIR="Results/SRPS"
mkdir -p "$RESULTS_DIR"

for act_thresh in "${ACT_THRESHOLDS[@]}"; do
  for beta in "${BETAS[@]}"; do
    thresh_tag=$(echo "$act_thresh" | tr '.' 'p')
    beta_tag=$(echo "$beta" | tr '.' 'p')
    tmp_config="$RESULTS_DIR/gemma_refusal_response_thresh_${thresh_tag}_beta_${beta_tag}.json"

    # Use inline python to dynamically adjust the json configuration
    python - "$CONFIG_FILE" "$act_thresh" "$beta" "$tmp_config" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
act_thresh = float(sys.argv[2])
beta = float(sys.argv[3])
output_path = Path(sys.argv[4])

with config_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

# Update config params for extraction-only hyperparam sweep
cfg["extractor"]["act_threshold"] = act_thresh
cfg["extractor"]["beta"] = beta
cfg["name"] = f"gemma_refusal_response_thresh_{str(act_thresh).replace('.', 'p')}_beta_{str(beta).replace('.', 'p')}"
cfg["save_vector"] = f"Vector/SRPS/Gemma/refusal_response_thresh_{str(act_thresh).replace('.', 'p')}_beta_{str(beta).replace('.', 'p')}"
cfg["n_test"] = 5
cfg["include_baseline"] = False

if "model" in cfg:
    cfg["model"]["device"] = "cuda"

# Ensure load_vector is None to force extraction
if "load_vector" in cfg:
    cfg.pop("load_vector")

with output_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

    echo ""
    echo "--- Running SRPS extraction | act_threshold=$act_thresh, beta=$beta ---"
    /data/caotue/anaconda3/envs/sae_circuit/bin/python -m Steering.cli --task extract --config "$tmp_config"
  done
done
