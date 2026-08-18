# Skill: Generate Method Configs
Description: Create standardized JSON configuration files for a new steering/extraction method across the benchmark datasets.

## Instructions
When this skill is triggered, generate the config JSON templates for the specified method (e.g. `MYMETHOD`) across the 9 core benchmark datasets. Follow the shared templates and specific override rules described below.

---

## 📋 Shared Setup (Must Follow)
Every configuration JSON must include these baseline settings:
```json
{
  "name": "<model_short>_<dataset_name>",
  "description": "<Method> steering on <dataset_name>",
  "model": {
    "name": "google/gemma-2-2b-it",
    "device": "cuda",
    "dtype": "bfloat16",
    "max_new_tokens": 100
  },
  "extractor": {
    "method": "<METHOD>",
    "layer": 14,
    "batch_size": 4,
    "hook_point": "pre",
    "apply_chat_template": true
  },
  "steer": {
    "method": "<METHOD>",
    "layer": 14,
    "apply_chat_template": true,
    "hook_point": "pre"
  },
  "output": "./Results/<method_lowercase>",
  "seed": 42
}
```

---

## 🔍 Specific Dataset Configurations

### 1. `refusal_ab` (Extraction Mode)
* **Purpose:** Runs extraction to compute the refusal steering vector.
* **Extractor Parameters:**
  * `"position": -2` (hooks the second-to-last token position).
* **Steer Parameters:**
  * `"coeff": 2.0`
* **Dataset Settings:**
  * `"train_dataset": "refusal_caa"`
  * `"n_train": 1000`
  * `"test_dataset": "refusal_ab"`
  * `"save_vector": "Vector/<METHOD>/Gemma/ai_risk_refusal"`

### 2. `hallu_ab` (Extraction Mode)
* **Purpose:** Runs extraction to compute the hallucination steering vector.
* **Extractor Parameters:**
  * `"position": -2` (hooks the second-to-last token position).
* **Steer Parameters:**
  * `"coeff": 1.0`
* **Dataset Settings:**
  * `"train_dataset": "hallucination"`
  * `"n_train": 1000`
  * `"test_dataset": "hallucination_ab"`
  * `"n_test": 500`
  * `"save_vector": "Vector/<METHOD>/Gemma/ai_risk_hallucination"`

### 3. `refusal_open` (Evaluation Mode)
* **Purpose:** Evaluates steering performance on open refusal prompts using the refusal vector.
* **Steer Parameters:**
  * `"coeff": 1.0`
* **Dataset Settings:**
  * `"load_vector": "Vector/<METHOD>/Gemma/ai_risk_refusal"` (loads refusal vector from `refusal_ab`)
  * `"test_dataset": "refusal_open"`
  * `"include_baseline": false`
  * `"compute_perplexity": true`

### 4. `refusal_response` (Extraction Mode)
* **Purpose:** Trains specifically on target/contrast responses to compute a response-only refusal vector.
* **Extractor Parameters:**
  * Standard configuration (no custom position override).
* **Steer Parameters:**
  * `"coeff": 2.0`
  * `"steer_once": false` (apply steering throughout generation)
* **Dataset Settings:**
  * `"train_dataset": "refusal_cast_responses"`
  * `"test_dataset": "refusal_open"`
  * `"save_vector": "Vector/<METHOD>/Gemma/refusal_response"`
  * `"include_baseline": false`
  * `"compute_perplexity": true`

### 5. `truthfulqa` (Evaluation Mode)
* **Purpose:** Evaluates truthfulness using the hallucination vector.
* **Steer Parameters:**
  * `"coeff": 1.0`
* **Dataset Settings:**
  * `"load_vector": "Vector/<METHOD>/Gemma/ai_risk_hallucination"` (loads hallucination vector from `hallu_ab`)
  * `"test_dataset": "truthfulqa"`
  * `"include_baseline": false`

### 6. `sorrybench` (Evaluation Mode)
* **Purpose:** Evaluates safety and refusal formatting on sorrybench using the standard refusal vector.
* **Steer Parameters:**
  * `"coeff": 1.0`
  * `"batch_size": 4`
* **Dataset Settings:**
  * `"load_vector": "Vector/<METHOD>/Gemma/ai_risk_refusal"` (loads refusal vector from `refusal_ab`)
  * `"test_dataset": "sorrybench"`
  * `"n_test": null` (evaluate full dataset)
  * `"include_baseline": false`
  * `"compute_perplexity": true`

### 7. `sorrybench_refusal_response` (Evaluation Mode)
* **Purpose:** Evaluates sorrybench using the response-only refusal vector.
* **Steer Parameters:**
  * `"coeff": 1.0`
* **Dataset Settings:**
  * `"load_vector": "Vector/<METHOD>/Gemma/refusal_response"` (loads response refusal vector from `refusal_response`)
  * `"test_dataset": "sorrybench"`
  * `"n_test": null`
  * `"include_baseline": false`
  * `"compute_perplexity": true`

### 8. `deception` (Extraction Mode)
* **Purpose:** Runs extraction to steer/prevent deceptive behavior.
* **Extractor Parameters:**
  * `"inverse": true` (swaps target and contrast data definitions).
* **Steer Parameters:**
  * `"coeff": 2.0`
  * `"steer_once": false`
* **Dataset Settings:**
  * `"train_dataset": "cais_mask"`
  * `"test_dataset": "liarbench"`
  * `"save_vector": "Vector/<METHOD>/Gemma/cais"`
  * `"n_test": 100`
  * `"include_baseline": false`
  * `"compute_perplexity": true`

### 9. `ifeval` (Extraction Mode)
* **Purpose:** Steers model to improve instruction-following capabilities.
* **Steer Parameters:**
  * `"coeff": 1.0`
  * `"steer_once": false`
* **Dataset Settings:**
  * `"train_dataset": "ifeval"`
  * `"test_dataset": "ifeval"`
  * `"save_vector": "Vector/<METHOD>/Gemma/ifeval"`
  * `"n_test": 100`
  * `"include_baseline": false`
  * `"compute_perplexity": true`

---

## 🐚 Bash Script Generator (Runner Template)
Create a bash script under `Scripts/run_<method_lowercase>_benchmark.sh` to run the evaluation sweep over your method's configuration files.

### Template:
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=4 # Must be between 2 and 5 only
set -euo pipefail

# Mandatory Conda environment setup
export PATH="/data/caotue/anaconda3/envs/sae_circuit/bin:$PATH"

COEFFS=(5 7 10) # Sweep target coefficients
CONFIG_DIR="Configs/Eval/<METHOD>/Gemma"

# Order configs such that extraction configs run BEFORE loading configs
CONFIGS=(
  "gemma_refusal_ab.json"                    # Extracts refusal vector
  "gemma_refusal_open.json"                  # Loads refusal vector
  "gemma_sorrybench.json"                    # Loads refusal vector
  "gemma_refusal_response.json"              # Extracts response refusal vector
  "gemma_sorrybench_refusal_response.json"   # Loads response refusal vector
  "gemma_hallu_ab.json"                      # Extracts hallucination vector
  "gemma_truthfulqa.json"                    # Loads hallucination vector
  "gemma_deception.json"                     # Extracts cais vector
  "gemma_ifeval.json"                        # Extracts ifeval vector
)

run_coeff_sweep() {
  local config_path="$1"
  local idx=0

  for coeff in "${COEFFS[@]}"; do
    local tmp_config
    tmp_config="$(mktemp /tmp/<method_lowercase>_cfg_XXXXXX.json)"

    # Inline Python helper to adjust coefficient and prevent redundant extractions
    python - "$config_path" "$coeff" "$idx" > "$tmp_config" <<'PY'
import json
import pathlib
import sys

base_path = pathlib.Path(sys.argv[1])
coeff = float(sys.argv[2])
idx = int(sys.argv[3])

with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg.setdefault("steer", {})["coeff"] = coeff
cfg["include_baseline"] = False

# Swap save_vector to load_vector on subsequent sweep iterations to skip extraction
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
```

