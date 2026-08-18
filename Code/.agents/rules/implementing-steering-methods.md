---
trigger: glob
applyTo: 'Steering/**'
description: Architectural rules and guidelines for implementing new steering and extraction methods in SAESteeringBench.
---

# Guidelines for Implementing Steering Methods

Whenever you are asked to implement, modify, or extend a steering or extraction method in the `Steering/` directory, you MUST follow these guidelines and architectural patterns.

## 1. Core Architecture Constraints

All steering methods must be split into two parts conforming to the base classes in `Steering/base.py`:
1. **Extractor:** Must inherit from `BaseExtractor` and define `METHOD_NAME`. Must implement `extract()` and `_get_activations()` (or raise `NotImplementedError` if optimized via custom training).
2. **Steer Model:** Must inherit from `BaseSteerModel` (for dense/nonlinear methods) or `BaseSAESteerModel` (for SAE-based methods). Must implement `hook_fn()` to modify the residual stream.

## 2. Configuration & Parameter Flow

All parameters must be declared in `Steering/config/methods.py`:
- Add the method name to the correct category set (`DENSE_METHODS`, `NONLINEAR`, `SAE_METHODS`, etc.).
- Register method-specific fields in the `EXTRACTOR_METHOD_FIELDS` and `STEER_METHOD_FIELDS` dictionaries.
- Define the fields with default values in the `ExtractorConfig` and `SteerConfig` dataclasses.

## 3. Handling Non-Standard Methods (The FLAS Pattern)

If a method does not fit the simple addition of a 1D steering vector (e.g. flow matching or low-rank adapters):
- **Dummy Carrier Vector:** Return a zero-tensor dummy of shape `[d_model]` in `extract()`.
- **Metadata Payloads:** Save state dicts or learned weights in `self.metadata`.
- **Custom Generative Loop:** Override the steer model's `generate()` or `get_token_probs()` methods to implement custom token generation.
- **Direct Layer Hooking:** In `"hf"` backend mode, hook model decoder layers directly using PyTorch hooks rather than using standard TransformerLens hooks.

## 4. Storage Constraints (Disk Full Protection)

- Heavy weight files/checkpoints (such as FLAS or ReFT weights) must always write outputs to a **single unified path per method** (e.g. `Vector/FLAS/gemma_gms8k`).
- Do NOT save sequential files (e.g. `grid_1`, `grid_2`). Every new run must overwrite the previous file at that path.

## 5. Integration Workflow Checklist

1. **Configure:** Define parameters and classify the method in `Steering/config/methods.py`.
2. **Implement Extractor:** Subclass `BaseExtractor` in `Steering/extractors/` and implement `extract()`.
3. **Implement Steer Model:** Subclass `BaseSteerModel` in `Steering/steer_models/` and implement `hook_fn()` or override `generate()`.
4. **Register Classes:** Add classes to `EXTRACTOR_MAP` in `Steering/extractors/__init__.py` and `STEER_MAP` in `Steering/steer_models/__init__.py`.
5. **Backend Type Mapping:** Add the backend mode (`hf`, `sae`, or `tl`) in `_get_model_type_for_method()` inside `Steering/pipeline.py`.
6. **Verify:** Test using the pipeline CLI:
   `python -m Steering.cli --config Configs/your_test_config.json`
