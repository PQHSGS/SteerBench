# FishBack v2 — Gradient-Guided Flow Steering

**Last updated**: 2026-07-21

---

## Problem Recap

FishBack v1 had three fatal flaws:
1. **Constant velocity target**: `v_true = dst - src` — same for all t, all tokens → reduces to CAA
2. **OT loss**: Single-step trajectory check, not full integration
3. **Jacobian reg**: Penalizes velocity variance, not aligned with steering goal

VGG-Flow (NeurIPS 2025, arxiv:2512.05116) proves: the optimal velocity correction for flow matching alignment is the **gradient of the value function** (HJB equation, Eq. 8). Our "value function" = −CE(model(h), harmful_tokens), so ∇_h CE = the direction that maximizes harmful probability.

## Design (VGG-Flow Style)

### Core insight

Standard flow matching: `v_θ(h, t)` learns to match constant velocity `h_harmful - h_safe`
FishBack v2: `v_θ(h, t)` learns to match **gradient at predicted final state**

```
v_target = ∇_{h_predicted} CE(model(h_predicted), harmful_tokens)
where h_predicted = h_t + (1-t) · stop-gradient(v_θ(h_t, t))
```

### Training loop (replaces _train_flow)

```
for each batch:
    1. Sample t ~ U(0,1)
    2. Interpolate: h_t = (1-t)*h_safe + t*h_harmful
    3. Get current prediction: v_pred = flow(h_t, t)
    4. Predicted final state: h_predicted = h_t + (1-t) * v_pred.detach()
    5. Inject h_predicted at target layer via hook
    6. Forward pass through frozen model
    7. Compute CE on FULL harmful sequence (not single next token)
    8. Backprop: v_target = ∇_{h_predicted} CE
    9. Train: L = MSE(v_pred, v_target)
```

### What's removed

| Removed | Why |
|:--------|:----|
| OT loss (sinkhorn_distance) | Single-step check, not aligned with multi-step integration |
| Jacobian regularization | Penalizes velocity variance, not aligned with steering |
| Fisher subspace (compute_fisher_subspace) | Projecting to 64-dim subspace loses information |
| fisher_R parameter | No longer needed |

### What stays

| Kept | Why |
|:-----|:----|
| FishBackMLP architecture | MLP matching FlowMLP pattern |
| Euler integration loop (FishBackSteerModel) | Multi-step steering at inference |
| Time embedding (sinusoidal) | Standard flow matching |
| Interpolation sampling | Curriculum of training points |

---

## Implementation Details

### _train_flow signature change

**Old**: `_train_flow(self, src, dst, fisher_R)`
**New**: `_train_flow(self, src, dst, target_prompts, target_layer)`

Parameters:
- `src`: safe activations [n, d_model]
- `dst`: harmful activations [n, d_model]
- `target_prompts`: List[str] — original harmful prompts (for tokenization + CE)
- `target_layer`: int — layer to inject at (from self.layer[0])

### Gradient computation

```python
# 1. Tokenize harmful prompts (right-padded, with attention mask)
tokens = model.tokenizer(target_prompts, return_tensors="pt", padding=True)
input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)

# 2. Get current prediction
v_pred = flow(h_t, t)  # [batch, n_tokens, d_model]

# 3. Predicted final state (stop-gradient)
h_predicted = h_t + (1 - t.unsqueeze(-1)) * v_pred.detach()
h_predicted = h_predicted.requires_grad_(True)

# 4. Inject at target layer and compute CE
hook_name = get_hook_name(target_layer, hook_point=self.hook_point[0])
# e.g., "blocks.14.hook_resid_pre"

def inject_hook(activation, hook):
    activation[:] = h_predicted
    return activation

logits = model.run_with_hooks(
    input_ids,
    attention_mask=attention_mask,
    fwd_hooks=[(hook_name, inject_hook)],
    return_type="logits",
)

# 5. Compute CE on FULL harmful sequence (mask padding)
# logits: [batch, seq_len, vocab_size]
# input_ids: [batch, seq_len]
# Shift: predict token i+1 from position i
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()
shift_mask = attention_mask[:, 1:].contiguous()

ce_loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    reduction="none",
)
ce_loss = (ce_loss.view_as(shift_labels) * shift_mask).sum() / shift_mask.sum()

# 6. Backprop to get gradient
v_target = torch.autograd.grad(ce_loss, h_predicted)[0]
```

### Key detail: full-sequence CE, not single next token

The user explicitly requires CE loss on the **sequence of tokens**, not single next-token prediction. This means:
- Shift logits by 1 position (predict token i+1 from position i)
- Apply attention mask to ignore padding
- Average over valid tokens (not batch size)

### Config params

**No new params needed.** Existing params are sufficient:
- `fb_hidden_dim=128` — MLP hidden dim
- `fb_n_layers=2` — MLP layers
- `fb_lr=1e-3` — learning rate
- `fb_epochs=100` — training epochs
- `fb_n_steps=10` — Euler integration steps

**Removed params** (dead code):
- `fb_lambda_ot` — OT loss weight
- `fb_lambda_jac` — Jacobian reg weight
- `fb_sinkhorn_epsilon` — Sinkhorn epsilon
- `fb_fisher_k` — Fisher subspace dimension

### Files to modify

| File | Change |
|:-----|:-------|
| `Steering/extractors/nonlinear.py` | Rewrite `_train_flow`: remove OT/Jacobian/Fisher, add gradient computation |
| `Steering/steer_models/nonlinear.py` | No change — inference loop stays the same |
| `Steering/config/methods.py` | Remove `fb_lambda_ot`, `fb_lambda_jac`, `fb_sinkhorn_epsilon`, `fb_fisher_k` |
| `Configs/Eval/FISHBACK/Gemma/*.json` | Remove dead params |

### Cost estimate

| Step | GPU Time | Description |
|:-----|:---------|:------------|
| Training (100 epochs) | ~2 hrs | 50 prompts × 100 epochs × model forward pass |
| Eval (1 task) | ~30 min | 100 test prompts × forward + generate |
| **Total per task** | **~2.5 hrs** | |
| **Total 4 tasks** | **~10 hrs** | |

Training is slower than v1 because each batch requires a model forward pass for gradient computation. But the quality should be much better.

---

## Decision Gates

| Result | Interpretation | Next Step |
|:-------|:---------------|:----------|
| Evil > 45% | Ceiling broken | Full sweep, compare with all methods |
| Evil > 20% but < 45% | Significant improvement | Tune learning rate, try deeper MLP |
| Evil < 20% | Gradient guidance helps but not enough | Add value consistency loss (VGG-Flow Eq. 12) |
| Evil = 0% | No improvement | Investigate gradient landscape (is it flat?) |

---

# Geometric INN — Path-Cost Regularized Invertible Mapping

**Added**: 2026-07-20

## Motivation

All activation methods (CAA, COLD, CAST, FishBack, etc.) share the same mechanism: `h += coeff · v`. The bottleneck is not the vector direction but the mechanism itself — linear addition in raw activation space. The previous INNSteer attempted to fix this with an invertible mapping but failed (0% Evil, 0% Toxic) because:

1. **Wrong loss**: Cosine similarity ≠ separation (destroys cluster structure)
2. **No path cost**: Straight lines in z-space don't correspond to meaningful paths in h-space
3. **No Fisher regularization**: Mapping geometry is disconnected from model's local curvature
4. **Steering vector is CAA in latent space**: Same problem as CAA, just in a different space

## Key Insight

**Path cost goes in the MAPPING, not steering.** Optimize φ such that:
- Geodesics in h-space ↔ straight lines in z-space
- Then CAA in z-space = geodesic in h-space
- Single optimization of φ, no joint optimization at inference

## Architecture

### GeometricInvertibleNN (in `extractors/nonlinear.py`)

Same RealNVP-style affine coupling layers as `InvertibleNN`, but with additional methods:

1. **`compute_jacobian_diag(z)`**: Compute diagonal of J = dh/dz via autograd
   - O(d) backward passes, not O(d²)
   - Proxy for local geometry

2. **`compute_path_cost(z, n_directions=8)`**: Penalize Jacobian variation
   - Sample random directions d on unit sphere
   - Compute ||J(z)·d||² at each z
   - Path cost = variance of this quantity across batch
   - Low cost → straight lines in z ↔ smooth paths in h

3. **`compute_fisher_loss(z)`**: Penalize Jacobian variation across samples
   - Fisher loss = variance of log|diag(J)| across samples
   - Low cost → locally uniform geometry

### Loss Composition

```
L = L_nll + λ_sep·L_separation + λ_path·L_path
```

| Term | Weight | Purpose |
|:-----|:-------|:--------|
| L_nll | 1.0 | Standard VAE-like NLL (z² - log det) |
| L_separation | λ_sep=2.0 | Maximize ||μ_dst - μ_src|| (contrastive) |
| L_path | λ_path=0.5 | Penalize Jacobian variation (smooth paths) |

3 losses, 3 hyperparameters. Fisher and log-det removed (redundant with NLL and path cost respectively).

### GeometricINNExtractor (in `extractors/nonlinear.py`)

Trains GeometricInvertibleNN per layer, stores:
- `inn_state_dicts`: Trained INN weights per layer
- `inn_config`: Architecture config (n_coupling, hidden_dim)
- `steering_vector`: CAA vector in latent space (z_dst_mean - z_src_mean)

### GeometricINNSteerModel (in `steer_models/nonlinear.py`)

Steering hook:
```
1. Encode: z = φ(h)
2. Steer: z' = z + coeff * v  (CAA in latent space)
3. Decode: h' = φ⁻¹(z')
```

## Config Fields

### ExtractorConfig

| Field | Default | Description |
|:------|:--------|:------------|
| ginn_n_coupling | 4 | Affine coupling layers |
| ginn_hidden_dim | 512 | MLP hidden dim |
| ginn_lr | 1e-3 | Learning rate |
| ginn_weight_decay | 1e-4 | Weight decay |
| ginn_epochs | 300 | Training epochs |
| ginn_batch_size | 64 | Training batch size |
| ginn_lambda_sep | 2.0 | Separation loss weight |
| ginn_lambda_path | 0.5 | Path cost weight |
| ginn_grad_clip | 1.0 | Gradient clipping |
| ginn_warmup_epochs | 50 | LR warmup |
| ginn_n_path_directions | 8 | Random dirs for path cost |
| ginn_checkpoint_dir | None | Save checkpoints |

### SteerConfig

No additional fields — uses `inn_state_dicts` and `inn_config` from metadata.

## Differences from Old INNSteer

| Aspect | Old INNSteer | GeometricINN |
|:-------|:-------------|:-------------|
| Separation loss | Cosine similarity | Max-margin distance |
| Path cost | None | Jacobian smoothness (in mapping) |
| Fisher regularization | None | Log-diag variance |
| Architecture | 4 layers, 512 dim | 4 layers, 512 dim (same) |
| Steering | CAA in latent space | CAA in latent space (same) |
| Overfitting risk | High (larger arch) | Lower (smaller arch + regularization) |

## Implementation Status

- [x] GeometricInvertibleNN architecture
- [x] GeometricINNExtractor
- [x] GeometricINNSteerModel
- [x] Registration in extractors/__init__.py and steer_models/__init__.py
- [x] Config fields in config/methods.py
- [ ] Extraction on Deception (sanity check — should work like old INNSteer)
- [ ] Extraction on Evil (test — this is the hard task)
- [ ] Evaluation sweep (coeff 1-30)
- [ ] Comparison with CAA, FishBack, old INNSteer

## Decision Gates

| Result | Interpretation | Next Step |
|:-------|:---------------|:----------|
| Evil > 0% but < 20% | Path cost helps but not enough | Try iterative steering (Euler on Fisher manifold) |
| Evil > 20% but < 45% | Significant improvement, ceiling not broken | Tune loss weights, try deeper INN |
| Evil > 45% | Ceiling broken | Full sweep, compare with all methods |
| Evil = 0% (same as old) | Path cost doesn't help | Try iterative steering or abandon INN approach |
| Overfitting (train good, eval bad) | Too few samples for INN | Reduce INN size, increase regularization |

## Cost Estimate

| Step | GPU Time | Description |
|:-----|:---------|:------------|
| Extract Deception | 30 min | Sanity check |
| Extract Evil | 30 min | Primary test |
| Eval sweep (1 task) | 1 hr | 30 coefficients × 100 prompts |
| **Total** | **~2 hrs** | Per task |
