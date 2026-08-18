# SAESteeringBench — Agent Guide

## CRITICAL: Read Report First

**Always read `Steering/report.md` first** before any work. It is the single source of truth for:
- Experiment history (§9) — what was tested, what was falsified, what was confirmed
- Method formulations (§2) — algorithm details for all 25+ methods
- Activation geometry diagnostics (§3) — PCA basis, cancellation filter, safety ceiling
- Cross-method benchmark results (§4) — accuracy scores, OOD rates
- Systematic weakness analysis (§5) — why specific methods fail on specific tasks
- Conclusions & future plans (§10) — ceiling candidates, remaining open questions

Without reading report.md, your work will duplicate falsified hypotheses, miss known limitations, and be out of sync with the project's current understanding.

## New Experiment Policy

- **Do NOT create experiment subfolders for simple sweeps/ablations.** Just edit the config JSON directly and run the CLI (`python -m Steering.cli --task eval --config ...`). Only create `Experiments/<ExpName>/` for genuinely complex, multi-step experiments with novel code.
- **All** new experiment scripts go in `Experiments/<ExpName>/`.
- One subdirectory per experiment, self-contained.
- Include results/logs alongside the script.
- Update report.md §9 with setup, key finding, and file reference.
- **Delete temporary scripts** after one-off tasks (docx edits, data checks, etc.) unless likely to be reused.
- Honor `.github/instructions/conda activate.instructions.md` for env activation and cache exports.

## Known Issues & Gotchas

- **Process Safety**: NEVER kill background processes or PIDs on the host system under any circumstances. Always allow existing processes to finish naturally.
- **FishBack & Gradient Guidance Prompt Boundary Alignment**:
  - Always extract `prompts` (user question) and `target_response` directly from `kwargs` in `extract(self, target_data, contrast_data, **kwargs)`.
  - Always format the user question using `apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True)`. The exact token index of the prompt boundary is **`prompt_len - 1`** (the `<start_of_turn>model\n` token).
  - In forward hooks (`_inject`), hook $h_{\text{predicted}}$ at `prompt_end_indices[b_idx]` matching the inference steering hook location during `model.generate()`.
  - Always apply `response_mask` zeroing out all question prompt tokens (`response_mask[b, :prompt_end_indices[b]] = 0`) so cross-entropy loss evaluates **exclusively on target response tokens**.
  - Keep `GRAD_BATCH = 2` (or 1) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to maintain peak VRAM footprint <6.2 GB and avoid memory fragmentation.
- **transformers 4.48 position_ids-on-CPU bug**: `model.generate()` crashes with `position_ids` on CPU for gemma2. **Always pass `**enc`** (both `input_ids` AND `attention_mask`), not just `input_ids`. If `**enc` still fails, use manual autoregressive generation — see `generate_one()` in `Experiments/REPS_Ablation/01_r_space.py`.
- **GPU utilization**: After launching, check `nvidia-smi` within 30s. If GPU util is 0% after model load, the model is on CPU.

## Quick Start

```bash
conda activate sae_circuit
unset CUDA_VISIBLE_DEVICES
nvidia-smi  # check GPU before launching runs; batch methods by ~20GB VRAM
```

## Commands

```bash
# CLI: extraction + generation + evaluation
python -m Steering.cli --task eval   --config Configs/Inference/caa_example.json
python -m Steering.cli --task run    --config Configs/Inference/cast_refusal.json
python -m Steering.cli --task extract --config Configs/Eval/sre_sycophancy.json

# Tests
python -m pytest Steering/tests/ -v
python -m pytest Steering/tests/test_config.py -v

# Verification (ground-truth extraction/inference match)
python Verification/Level1/SPARE/validate_spare_l1.py
python Verification/Level2/SAETS/validate_saets_l2.py
```

## Architecture

The `Steering/` package is the core library. Everything is **Extract → Steer → Evaluate**.

- **Extractors**: `Steering/extractors/` — `EXTRACTOR_MAP` registry at bottom of each file.
- **Steer models**: `Steering/steer_models/` — `STEER_MAP` registry at bottom of each file.
- **Evaluators**: `Steering/evaluators/` — `EVALUATOR_MAP` registry.
- **Config**: `Steering/config/methods.py` — all defaults live in `ExtractorConfig`/`SteerConfig` dataclass fields (single source of truth). `PipelineConfig` lives in `Steering/config/pipeline.py`.
- **Data**: `Steering/data/data_registry.py` — `TRAIN_DATASET_REGISTRY`, `TEST_DATASET_REGISTRY`.

Default model: `google/gemma-2-2b`. Supported: gemma-2-{2b,9b}{,-it}, llama-2-7b-hf, llama-3.1-8b.

**Type contracts** (all multi-layer params use dict types):
- `layer`: `List[int]` — `[14]` not `14`
- `steering_vector`: `Dict[int, Tensor]` — `{14: tensor}`
- `coeff`: `Dict[int, float]` — `{14: 2.0}`
- `sae`: `Dict[int, SAE]` — `{14: sae_obj}`

## Model Selection & Pipeline Enforcements (CRITICAL) ⚠️

- **Must use `google/gemma-2-2b-it` (instruction-tuned model) for safety alignment experiments.** The base model (`google/gemma-2-2b`) has no RLHF safety alignment, meaning the cancellation filter circuits do not exist. Any diagnostic experiment related to safety resistance circuits (ExpA, ExpB, ExpC, ExpD, ExpE) must run on `google/gemma-2-2b-it`.
- **Must utilize `SteeringPipeline` as much as possible.** Do not write manual hooks to modify activations (e.g., `activations[0, -1, :] += coeff * v`) when evaluating multiple methods. Manual additions collapse complex algorithms (like CHaRS barycentric transport, FLAS ODEs, and AcT mappings) into simple linear CAA additions, causing false/hallucinated results. Always instantiate steering wrappers via `SteeringPipeline` and call `setup_hooks` so the library manages hook functions correctly.
- **NEVER write manual steering code.** This means: do NOT write `residual += coeff * v`, do NOT manually `torch.load("vector.pt")`, do NOT manually build `model.add_hook("blocks.X.hook_resid_pre", ...)`. Every method in this codebase has a steer model class (`DenseSteerModel`, `TransportSteerModel`, `LoReFTSteerModel`, `WeightSteerModel`, etc.) that handles hook placement, hook logic, weight modification, and cleanup. Writing manual code for any of this means you are doing it WRONG — you will miss method-specific behavior (transport methods use PCA centroids, not `steering_vector`; parametric methods use learned interventions, not linear addition; weight methods modify LoRA weights, not activations), produce false results, and waste GPU hours on broken experiments. The correct pattern is ALWAYS: `PipelineConfig.load(path)` → `SteeringPipeline(config)` → `pipeline.setup()` → `steer.setup_hooks(coeff_dict)` → `model.run_with_hooks(...)`. If you find yourself writing a hook function, STOP and use the pipeline instead.
- **NEVER assume CAA direction is the reference or best direction.** CAA's mean-difference vector `μ+ - μ-` is just one proxy metric (implicitly M = Σ⁻¹). Other methods (FishBack, Transport, Parametric) compute fundamentally different directions using different covectors (concept probability gradient, Fisher-optimal paths, learned interventions). When implementing or evaluating a new method, its steering direction should be derived from that method's own theory — not from CAA as a starting point. Prewhitening a CAA vector with G⁻¹ (as in `v_fisher = (G+αI)⁻¹ v_CAA`) is WRONG if the method's theory specifies a different covector (e.g., `q = ∇_h P^W(1)` for FishBack). The correct pattern is: identify the method's covector q and metric M from the paper, then implement δh = M⁻¹ q.
- **NEVER hardcode a single method.** Every experiment script MUST support all methods via `--method` flag (default: CAA). Use the existing eval configs in `Configs/Eval/<METHOD>/Gemma/` and load them via `PipelineConfig.load(config_path)`. The `KL_decay/config.json` file has the `method_config_map` mapping method names to config directories — reuse it. If you find yourself writing `"method": "CAA"` hardcoded in a PipelineConfig dict, you are doing it WRONG — you are creating a script that only works for one method and produces unverifiable results for all others.
- **When extending pipeline behavior (e.g., adding hooks), WRAP, don't REPLACE.** If you need to add hooks after the method's steering hooks (e.g., for r̂ ablation), save a reference to the original `steer.setup_hooks` and call it first, then add your hooks. NEVER monkey-patch `setup_hooks` to replace the method's own hooks — this collapses all methods into CAA. Pattern: `_orig = steer.setup_hooks; def wrapped(coeff): handles = _orig(coeff); handles += my_hooks; return handles; steer.setup_hooks = wrapped`

## Storage

- Folder-per-method: `vector.pt` (tensor or layer→tensor dict) + `metadata.pt` (extraction params).
- **FLAS and REPS produce multi-GB checkpoints.** Save in `data/` instead of `Vector/`:
  - FLAS: `data/FLAS/gemma_gms8k`
  - REPS: `data/REPS/gemma_gms8k`

## Method Categories

- **Dense** (CAA, COLD, CAST, MANIFOLD, SPHERICAL, SAE-FREE, LQR, JSPACE, IDS, FISHBACK) — modify residual stream directly; work on any model.
- **Transport** (ACT, ANGULAR, CURVEBALL, FLOW, PID, ODE, BIPO, CHARS, LinNEAS, COBRA, INNSTEER) — learned/analytic maps.
- **Parametric** (REFT, LOREFT, REPS, FLAS) — train lightweight modules during extraction.
- **SAE** (SAEIO, SAS, SPARE, SRE, SRPS, SSV, SAE-RSV, SAE-TS, SAE-COT, CORRSTEER, FEAT, FGAA) — require Sparse Autoencoder; **Gemma models only**.
- **Weight** (WEIGHTSTEER) — LoRA contrastive weight steering.

## Adding Things

| What | Files to edit |
|------|--------------|
| **Method** | Extractor class + steer model class + register in `Steering/extractors/__init__.py` and `steer_models/__init__.py`. Optionally add to `SAE_METHODS` in `config/methods.py`. |
| **Train dataset** | One entry in `TRAIN_DATASET_REGISTRY` in `Steering/config/datasets.py` |
| **Test dataset** | One entry in `TEST_DATASET_REGISTRY` in `Steering/config/datasets.py` |
| **Evaluator** | Class in `Steering/evaluators/` + register in `EVALUATOR_MAP` |
| **New config field** | **(3 places)** ① Add default to `ExtractorConfig`/`SteerConfig` in `config/methods.py`. ② Register in `EXTRACTOR_METHOD_FIELDS`/`STEER_METHOD_FIELDS` under your method's set (same file, top dict). ③ Read from kwargs in your extractor/steer_model constructor. |
| **Model-SAE mapping** | Add to `MODEL_SAE_REGISTRY` in `config/models.py` |

## Verification Levels

- **L1** (Extraction match): cosine similarity ≥ 0.99 vs `Code/` reference.
- **L2** (Inference match): logit cosine ≥ 0.95, KL < 1e-6.
- **L3** (Accuracy match): matches paper-reported results.

Run verification with `unset CUDA_VISIBLE_DEVICES; conda activate sae_circuit` first.

## Reference Implementations — ALWAYS COPY THESE FIRST

Before writing any new experiment script, read and copy the pattern from these files:

| Task | Reference | What it shows |
|:-----|:----------|:-------------|
| Multi-method eval loop | `Experiments/KL_decay/01_kl_decay.py` | SteeringPipeline setup, `method_config_map`, random control, cleanup |
| Hook extension (wrapping) | `Experiments/ExpH/run_all_layer_ablation.py` | `wrap_setup_hooks()` pattern — call original first, then add custom hooks |
| DLA per-method | `Experiments/DLA/01_head_ablation.py` | SteeringPipeline per-method, validation printing, incremental save |
| Config loading | `Experiments/KL_decay/config.json` | `method_config_map` structure for all 9 methods |

**Rule**: If you are about to write `PipelineConfig.from_dict({...})` with hardcoded method names, STOP. Use `PipelineConfig.load(config_path)` from an existing eval config instead.

## Experiment Notes

- **OT Post-Processing**: Add `"ot_steer": true` to the `steer` config for 1D affine transport correction after any method's hook. Stats computed automatically during `setup()`.
- **OOD Early Termination**: If a lower coeff for a (method, task) pair shows OOD (LSP > per-task threshold), higher coeffs will be at least as OOD. No need to run them.
- **Old results without `repetition_rate`** (pre-0615): If ppl < 5, likely clean. If ppl ≥ 5, inspect samples manually and judge OOD.

## Experiment Summary Docx Rules

The single source of truth for experiment writeup is `Experiments/Experiment Summary.docx`.

### Content Rules

- **Paper references**: Keep brief (1-2 bullet points). Include only key result and how it motivated our experiment. Avoid big paragraph chunks. Professor may not have read them — include enough context for understanding (year, venue, one-sentence finding).
- **No dates**: Remove "(Jun 24)", "(Jun 26-27)", "Running (Jun 27)" etc. from headings and body. Only results and conclusions matter.
- **No code references**: Remove file paths (`Experiments/TailAnalysis/`, `pipeline.py`), config flags (`compute_lsp`, `act_mode=`), script names (`run_linearact_ablate.sh`), and dataset references (`Configs: deception, evil, toxic`).
- **No experiment labels**: Remove internal labels like "§9.13 ExpD".
- **No conversational framing**: Avoid "What we wanted to know", "What we found", "What was proven wrong". Use "Question:", "Finding:", "Falsified hypotheses:".
- **Format**: Each section: Motivation → Setup → Observation(s) → Conclusion. No bullet-point data dumps — use `doc.add_table()` with `Table Grid` style.
- **Tone**: Formal, brief. No "we"/"I"/"you", no overconfidence — 90% of experiments are failures.
- **Motivation line**: Every experiment section MUST start with a **Motivation** line right after the heading. It must state what actionable insight the experiment gives the final aim (understanding/breaking the safety ceiling). Format: "Motivation: [what this experiment tells us about the safety ceiling and what strategic decision it enables]." Do NOT include predictions/expectations — focus on what decision the result enables regardless of outcome.

### Section Structure

Every section follows: `Motivation → Setup → Observation(s) → Conclusion`
- Observations must include full data from report.md, both good and bad
- Conclusion states whether hypothesis was confirmed, falsified, or remains unconfirmed

### Zero Formatting

Replace "0.0000" with "0" in all table cells to avoid visual clutter from superfluous precision.

### Build Procedure

- Always edit in Python using `python-docx`. Never write directly.
- Save via temp file then copy (to handle WINWORD lock):
  ```python
  import tempfile, shutil
  tmp = tempfile.mktemp(suffix='.docx')
  doc.save(tmp)
  # Kill WINWORD first if needed
  shutil.copy2(tmp, PATH)
  ```
- Always verify the result after editing.

### CRITICAL: Backup Before Git Operations

**NEVER run `git checkout`, `git restore`, or `git revert` on any path containing `Experiment Summary.docx` without first copying it to a timestamped backup.** The docx is never committed to git — it is the single working copy. If deleted, check `Experiments/` for temp files and `C:\Users\Admin\AppData\Local\Temp\opencode\` before attempting git restore.

## Pareto Plot Requirements

See `.opencode/skills/pareto-plot/SKILL.md` for full build procedure.
Key rules:
- Only coeffs `1 2 3 5 7 10`
- Date filter ≥ 20260615
- OOD = LSP > per-task threshold (Deception 25, Evil 10, Toxic 20)
- LSP-only OOD: no separate ppl/rep thresholds
- Linear axis
- Deception y = deceptiveness (100 - honesty %), evil/toxic y = accuracy %
- Config-based position + train_dataset matching

## Refusal Config & Data Naming (CRITICAL)

**When running refusal experiments, always use:**
- **Config**: `Configs/Eval/*/gemma_refusal_response.json` (NOT `refusal_ab` or `refusal_open`)
- **Training data**: `refusal_cast_responses` in the config JSON's `train_dataset` field (NOT `refusal_ab` or `refusal_open`)

These are the only refusal config/dataset pairs that produce correct evaluations. The `_ab` and `_open` variants use different prompt templates and cannot be compared with `refusal_response` results.

## Research Methodology — Thinking Strategy

### How to Evaluate Experiments

Before running any experiment, ask these questions IN ORDER:

1. **Does this experiment advance the causal story?**
   - If yes → run it
   - If no → skip it (even if it's "interesting")

2. **Is this experiment correlational or interventional?**
   - Interventional > correlational (always)
   - Correlation is cheap but weak; intervention is expensive but strong

3. **What would I learn if this succeeds? What would I learn if it fails?**
   - If both answers are "not much" → redesign the experiment
   - If only one answer is informative → the experiment is weak

4. **Does this experiment require identifying the mechanism first?**
   - If yes → mechanism identification must come before intervention
   - If no → proceed directly

### Choosing Between Approaches

When presented with two options (e.g., Option A: correlate, Option B: intervene):

1. **Does Option A help Option B?**
   - If yes → do Option A first (it's a prerequisite)
   - If no → they're independent; choose the one that advances the causal story

2. **Which option utilizes existing paper works?**
   - The option that leverages proven methodologies is stronger
   - Don't reinvent the wheel when papers show HOW to do it

3. **Which option has higher payoff?**
   - Intervention > correlation (always)
   - Mechanism identification > metric refinement (always)

### The Causal Chain

Every hypothesis must be tested through this chain:

```
1. Does X exist? (boolean)
    ↓ yes
2. Where is X? (location)
    ↓ found
3. What causes X? (mechanism)
    ↓ identified
4. If we ablate the cause, does the effect disappear? (intervention)
    ↓ yes
5. X is proven to cause Y
```

**Never skip to step 4 before step 3 is complete.** You cannot ablate something you haven't identified.

### Weak vs Strong Chains

**Weak chain** (correlational):
- "KL decays for Evil, accuracy is low → KL decay causes low accuracy"
- Problem: FLAS has high KL but 0% accuracy → correlation breaks

**Strong chain** (interventional):
- "Identify suppressing heads → ablate them → accuracy improves"
- This PROVES causation, not just correlation

### The KL Trap

KL divergence is seductive because it's easy to measure. But:
- KL **magnitude** does NOT predict accuracy (FLAS paradox)
- KL **pattern** (decay vs growth) IS diagnostic (Evil decays, Deception grows)
- The contribution is NOT proving KL correlates with accuracy
- The contribution is identifying the MECHANISM that causes KL decay and showing that ablating it causally improves accuracy

### What Papers Do vs What We Should Do

| What papers do | What we should do |
|:---------------|:------------------|
| Identify mechanism (DLA, angle) | ✓ Same |
| Measure correlation (KL, cosine) | ✗ Skip (doesn't prove causation) |
| Ablate mechanism | ✓ Same |
| Measure if behavior changes | ✓ Same |

**Papers that prove causation**: Wang 2026, Perfect Detection, Three Classes
**Papers that show correlation only**: Hahami (observational, no intervention)

---

## Geometry Caveats ⚠️

- **Per-dim kurtosis** in canonical basis is misleading: tasks have heavy tails in off-manifold dims. Use PCA-based in/off-manifold energy decomposition if doing manifold analysis.
- **Cosine similarity** assumes spherical space. Prefer subspace overlap (principal angles) or projection ratios when comparing directions.
- **Wasserstein-2** is basis-dependent. Use Sliced W2 for basis-agnostic estimate.
- **"Tail-heavy failures"** (§9): original per-dim kurtosis finding was a universal structural property, not task-specific. Real CHARS causal factors: centroid norm spread and K-selection.
- **Deception directionality**: accuracy = deceptiveness (DeceptionEvaluator flipped post-20260714). Higher accuracy = MORE deceptive = BETTER. Same direction as Evil/Toxic.

Work in the representation space most informative for your question. Full 2304D is valid. PCA is a tool when needed, not a default.
