# Skill: Implement Steering Method
Description: Guide and write code to implement a new steering or extraction method in the SAESteeringBench framework.

## 🤖 Coordination with Method Implementer Agent
- When this skill is triggered, you must operate under the persona and capabilities of the **Method Implementer** agent defined in your agent profile.
- If the implementation requires design discussions, hyperparameter optimization, or architecture planning, consult with the user under the **Method Implementer** persona before writing the final code.

## ⚠️ Important Environment & System Guidelines
- **Conda Environment:** You MUST activate and use the `sae_circuit` Conda environment when running, testing, or executing any benchmarks.
- **CUDA Device Restrictions:** You MUST restrict CUDA execution to devices **2 to 5 only** (e.g., set `CUDA_VISIBLE_DEVICES=2` or select from indices `2,3,4,5`). Do NOT use device `0` or `1`.
- **Reading Papers:** If there are research papers in the `Paper/` directory describing the method to implement, you MUST use the `pypdf` library to read and extract information from the PDF files.

## Instructions
Execute the following steps to implement the new method requested by the user:

1. **Understand Base Classes (`Steering/base.py`)**:
   - Extractor: Must subclass `BaseExtractor`. Define `METHOD_NAME`. Implement `extract(self, target_data, contrast_data, **kwargs)` and `_get_activations()`.
   - Steer Model: Must subclass `BaseSteerModel` or `BaseSAESteerModel`. Implement `hook_fn(self, resid, position, coeff, steering_vector, [sae], hook, **kwargs)`.

2. **Configure Parameters (`Steering/config/methods.py`)**:
   - Classify the method into `DENSE_METHODS`, `NONLINEAR`, `SAE_METHODS`, etc.
   - Register parameter fields in `EXTRACTOR_METHOD_FIELDS` and `STEER_METHOD_FIELDS`.
   - Add default values in `ExtractorConfig` and `SteerConfig` dataclasses.

3. **Handle Non-Standard Workarounds (FLAS Pattern)**:
   - Extractor: Return a zero dummy vector of shape `[d_model]` and store state dicts in `self.metadata`.
   - Steer Model: Override `generate()` and `get_token_probs()` to run custom loops. Hook decoder layers directly on model blocks for `"hf"` backend mode.

4. **Observe Storage Constraints**:
   - Large checkpoint files must write outputs to a single unified path per method to avoid filling up the disk space (never use versioned names like `grid_1`, `grid_2`).

5. **Register Registries**:
   - Add class mappings to `EXTRACTOR_MAP` in `Steering/extractors/__init__.py` and `STEER_MAP` in `Steering/steer_models/__init__.py`.
   - Specify backend mode (`hf`, `sae`, or `tl`) in `_get_model_type_for_method()` inside `Steering/pipeline.py`.

6. **Verify Implementation**:
   - Run verification via CLI using the `sae_circuit` environment and CUDA devices 2-5:
     `CUDA_VISIBLE_DEVICES=2 python -m Steering.cli --config Configs/your_test_config.json`
