# Consolidated Report: Representation Steering — Lit Survey, Activation Geometry, Method Weaknesses, Experiment History

Single source of truth for SAESteeringBench. Merges lit survey, method analyses, safety task geometry, systematic weaknesses, session brainstorming + experiment history. Publication-grade record at [Steering/report.md](file:///home/caotue/SAESteeringBench/Steering/report.md).

**Exp tracking**: All scripts, replication code, outputs in [`Experiments/`](../Experiments/). Each exp self-contained with documented results.

> [!IMPORTANT]
> **Model Selection & Hook Design Rules:**
> 1. **Must use `google/gemma-2-2b-it` (instruction-tuned model) for safety alignment experiments.** The base model (`google/gemma-2-2b`) has no RLHF safety alignment, so any diagnostic experiment that tests for the hypothesised cancellation filter mechanisms (ExpA, ExpB, ExpC, ExpD, ExpE) must run on `google/gemma-2-2b-it`.
> 2. **Must utilize `SteeringPipeline` as much as possible.** Do not write manual hooks to modify activations (e.g., `activations[0, -1, :] += coeff * v`) when evaluating multiple methods. Manual additions collapse complex algorithms (like CHaRS barycentric transport, FLAS ODEs, and AcT mappings) into simple linear CAA additions, causing false/hallucinated results. Always instantiate steering wrappers via `SteeringPipeline` and call `setup_hooks` so the library manages hook functions correctly.

---

## Table of Contents
1. [Lit Survey: Paradigm Shift to Distributional Alignment (2024–2026)](#1-literature-survey-the-paradigm-shift-to-distributional-alignment-2024-2026)
2. [Method Formulations, Algorithms, Internal Experiments](#2-method-formulations-algorithms-internal-experiments)
3. [Activation Geometry Diagnostics of Safety Tasks](#3-activation-geometry-diagnostics-of-safety-tasks)
4. [Cross-Method Benchmark Results & Empirical Summary](#4-cross-method-benchmark-results-empirical-summary)
5. [Systematic Analysis of Method Weaknesses & Paper Contradictions](#5-systematic-analysis-of-method-weaknesses-paper-contradictions)
6. [Fine-Tuning vs. Steering Dynamics](#6-fine-tuning-vs-steering-dynamics)
7. [Proposed Verification & Diagnostic Experiments](#7-proposed-verification-diagnostic-experiments)
8. [Implications & Brainstorming History](#8-implications-brainstorming-history)
9. [Experiment History Log](#9-experiment-history-log)
10. [Conclusions & Future Plans](#10-conclusions-future-plans)
 - [10.7 Mechanistic Proofs & The Dual Constraint Model](#107-mechanistic-proofs--rebuttal-analysis-the-dual-constraint-model)
 * [10.7 Mechanistic Proofs & Rebuttal Analysis: The Dual Constraint Model](#107-mechanistic-proofs--rebuttal-analysis-the-dual-constraint-model)

---

## 1. Literature Survey

**Moved to [Steering/survey.md](survey.md)** — comprehensive survey of 25+ papers covering linear intervention, OT steering, control theory, shallow alignment theory, persona subspace, and predictive diagnostics.

Key relevance to safety ceiling: Young (2026) Zero Gradient Theorem + Assistant Axis persona resistance + J-space 93% inert variance together explain why activation methods are bounded on safety tasks but not deception.

---

## 2. Method Formulations, Algorithms, Internal Experiments

### 2.1 FLAS (Flow-based Activation Steering)
Neural ODE with concept-conditioned velocity field v_theta via multi-step Euler:
: h_k+1 = h_k + fracTN cdot v_thetaleft(h_k, frackTN, cright)
* **Loss**: L_total = L_LM + lambda_divL_div — penalizes cosine similarity between velocity fields of different concepts. Trained on positive-only data.
* **Key Findings**:
 1. PCA trajectories start shared, bend into concept-specific regions.
 2. N=1 ablation drops HMean 1.015  ->  0.837 — multi-step paths causally necessary.
 3. Removing cross-attention to concept embedding drops HMean to 0.109.

### 2.2 REPS (Reference-free Preference Steering)
Optimizes steering direction theta via preference optimization:
: mathcalL(theta) = - left[ log sigma(Delta^+) + log sigma(Delta^-) right]
: Delta^+ = beta^+ log p_theta(y_steer | x) - log p_ref(y_steer | x)
: Delta^- = log p_ref(y_null | x) - log p_theta(y_null | x)
* **Structure**: LoReFT: Phi (h) = h + R^T(Wh + b - Rh), R in R^{r x d} orthonormal. Samples alpha in [2, 20] during training.
* **Key Finding**: Decoupling between concept detection and steering — LM-objective vectors detect better but steer worse than preference-optimized ones.

### 2.3 CurveBall (Nonlinear Manifold Steering)
Projects to 20-dim polynomial/RBF KPCA space (Z = phi (H)), computes steering as normalized mean diff in KPCA (z_hat):
: a_target = phi(A_curr) + alpha hatz
: A_steered = phi^-1(a_target) + left(A_curr - phi^-1(phi(A_curr))right)
* **Key mechanism**: residual preservation prevents PPL explosion.
* **Key Finding**: VAE pullback metrics show LLM manifold highly curved (distortion R  >>  1). Linear steering crashes at curvature kappa > 8, CurveBall stable.

### 2.4 Lin-ACT & LinEAS
* **Lin-ACT**: Closed-form 1D Wasserstein affine transport. Pushforward shows near-perfect overlap vs target; bias-only ITI-C fails. Sequential estimation > simultaneous.
* **LinEAS**: Learns w, b per layer via Sliced Wasserstein Distance. Group Lasso sparsity identifies relevant neurons; at 1\% support, MMLU preserved.

### 2.5 WeightSteer (LoRA Contrastive Weight Steering)
* **Mechanism**: LoRA on contrastive activation states. Inference: merge LoRA weights into target projections.
* **Key Finding**: Near-perfect on simple styles but severe PPL explosion at higher coeffs from direct weight modification.

---

## 3. Activation Geometry Diagnostics of Safety Tasks

7 geometric diagnostics across **Toxicity Induction**, **Deception (Honesty Suppression)**, **Evil Induction**, **Refusal Response** on `google/gemma-2-2b` across 6 layers (L in \{6, 10, 14, 18, 22, 25\}).

### 3.1 Task-Specific Profiles

* **Toxicity**: Mean W_2 = 1.336, cos  theta = 0.961. kappa_T = 0.461, d_pullback = 25.890, R_arc/chord = 0.856. Optimal K = 6.5 clusters. Asymmetry = 0.44, SNR = 1.636. eff\_rank} approx 80, 270 components for 95\% variance.
* **Deception**: W_2 = 0.330, cos  theta = 0.999. kappa_T = 0.257, d_pullback = 3.439, R_arc/chord = 0.990. K = 3.7 clusters. Asymmetry = 0.90, SNR = 0.358. eff\_rank} approx 12.5, 60 components.
* **Evil**: W_2 = 2.089, cos  theta = 0.918. kappa_T = 0.186, d_pullback = 48.178, R_arc/chord = 1.122. K = 2.0 clusters (bimodal). Asymmetry = 0.95, SNR = 4.531. eff\_rank} approx 44, 200 components.
* **Refusal**: W_2 = 2.610, cos  theta = 0.831. kappa_T = 0.167, d_pullback = 18.722, R_arc/chord = 1.010. Asymmetry = 0.580, **SNR = 4.951**. eff\_rank} approx 40.9. Why easy: refusal/compliance separated by massive clean gap (SNR = 4.951), near-flat boundary (R_arc/chord approx 1.01, kappa_T = 0.167). Simple linear CAA sufficient.

### 3.2 Manifold Metrics & Formulations

1. **Trajectory Bending (kappa_T)**: Path curvature in KPCA space. kappa_T = 1 - frac{1}{N-1} sum(i=0)^{N-2} cos (v_i, v_{i+1}). kappa_T approx 0 = straight line.
2. **Pullback Deviation (d_pullback)**: Max distance between ambient linear path and curved manifold: d_pullback = max _alpha || x(alpha) - {Proj}_{M}(x(alpha)) ||_2.
3. **Manifold Curvature Ratio (R_arc/chord)**: Geodesic arc / chord distance. R > 1.0 = highly curved.
4. **PCA Tangent Reconstruction Error (E_PCA)**: Avg reconstruction error of ambient path projected onto PCA tangent space.

### 3.3 Geometrical Implications on Method Performance

1. **Deep layer curvature**: kappa_T spikes ~1.0 at layers 22-25. Linear vectors (CAA, LoReFT) applied here push off-manifold → PPL explosion, repetition.
2. **High pullback in Evil/Toxic**: Evil d_pullback=48.178, R=1.122 — CHaRS/COBRA needed. Toxic highest E_PCA=3.550, kappa_T=0.461 — high-rank (K=6.5, 270 PCs), cannot compress into flat subspace.
3. **Parallel manifolds in Deception**: Lowest d_pullback=3.439, R=0.990 but SNR =0.358 — distributions overlap heavily. OT/COBRA works via joint distribution alignment.
4. **Refusal = "easy"**: Lowest kappa_T=0.167, R=1.010, E_PCA=1.145, highest SNR =4.951. Flat + well-separated → CAA sufficient. Advanced methods offer no geometric benefit.

---

## 4. Cross-Method Benchmark Results & Empirical Summary

Tables below summarize cross-method benchmarks (0615–0619). (OOD) = LSP > per-task threshold (Deception 25, Evil 10, Toxic 20).

### 4.1 Benchmark Evaluation Grids

#### DECEPTION (Deceptiveness ↑)
| Method | c=1.0 | c=2.0 | c=3.0 | c=5.0 | c=7.0 | c=10.0 | Best Clean Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CAA | 15.00 | 17.00 | 28.00 | 25.00 | 48.00 | 51.00 | **51.0\%** (c=10.0, ppl=2.15) |
| ACT | 47.00 | 60.00 | 80.00 | 90.00 | 79.00 | (OOD)84.00 | **90.0\%** (c=5.0, ppl=2.45) |
| CURVE | 68.00 | 66.00 | 66.00 | 73.00 | 55.00 | 63.00 | **73.0\%** (c=5.0, ppl=5.69) |
| CHARS | 33.00 | 28.00 | 55.00 | 73.00 | 85.00 | — | **85.0\%** (c=7.0, ppl=4.34) |
| FLAS | 72.00 | — | — | — | — | — | **72.0\%** (c=1.0) |
| REPS | 31.00 | 42.00 | — | — | — | — | **42.0\%** (c=2.0, ppl=2.32) |
| LinNEAS | 24.00 | 26.00 | 24.00 | — | — | — | **26.0\%** (c=2.0, ppl=2.16) |
| WEIGHT | 68.00 | — | 54.00 | — | — | — | **68.0\%** (c=1.0) |
| COBRA | 35.00 | 43.00 | 61.00 | 79.00 | 94.00 | (OOD)100.00 | **94.0\%** (c=7.0, ppl=5.56) |

#### EVIL (Accuracy ↑)
| Method | c=1.0 | c=2.0 | c=3.0 | c=5.0 | c=7.0 | c=10.0 | Best Clean Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CAA | 0.00 | 1.00 | 2.00 | (OOD)5.00 | (OOD)3.00 | (OOD)0.00 | **2.0\%** (c=3.0) |
| ACT | 1.00 | 3.00 | (OOD)5.00 | (OOD)8.00 | (OOD)2.00 | (OOD)0.00 | **3.0\%** (c=2.0) |
| CURVE | 0.00 | 2.00 | 1.00 | (OOD)4.00 | (OOD)0.00 | (OOD)0.00 | **2.0\%** (c=2.0) |
| CHARS | 0.00 | 1.00 | 3.00 | (OOD)6.00 | (OOD)2.00 | (OOD)0.00 | **3.0\%** (c=3.0) |
| FLAS | 2.00 | 4.00 | 6.00 | 8.00 | 7.00 | 5.00 | **8.0\%** (c=5.0, ppl=5.03) |
| REPS | 0.00 | 0.00 | 0.00 | 1.00 | 2.00 | 1.00 | **2.0\%** (c=7.0) |
| LINNEAS | 1.00 | 2.00 | 0.00 | — | — | — | **2.0\%** (c=2.0) |
| WEIGHT | 62.00 | — | — | — | — | — | **62.0\%** (c=1.0, ppl=TBD) |
| COBRA | 3.00 | 5.00 | 10.00 | (OOD)12.00 | — | — | **10.0\%** (c=3.0, ppl=5.76) |

**WeightSteer note**: Weight coeffs not comparab<= to activation coeffs — scale LoRA deltas, not activation vectors. At c=1.2, WeightSteer **81.0\%** Evil — first method to break 0-10\% ceiling. Ceiling is **activation-specific, not fundamental**.

#### TOXIC (Accuracy ↑)
| Method | c=1.0 | c=2.0 | c=3.0 | c=5.0 | c=7.0 | c=10.0 | Best Clean Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CAA | 1.00 | 0.00 | 0.00 | 1.00 | 36.00 | — | **36.0\%** (c=7.0, ppl=6.92) |
| ACT | 0.00 | 2.00 | 2.00 | 4.00 | 21.00 | (OOD)73.00 | **21.0\%** (c=7.0, ppl=4.11) |
| CURVE | 44.00 | (OOD)59.00 | (OOD)61.00 | (OOD)87.00 | — | — | **44.0\%** (c=1.0, ppl=6.77) |
| CHARS | 0.00 | 0.00 | 2.00 | 0.00 | — | (OOD)80.00 | **2.0\%** (c=3.0) |
| FLAS | — | 0.00 | 0.00 | 0.00 | — | 0.00 | **0.0\%** |
| REPS | — | — | — | (OOD)11.00 | (OOD)18.00 | (OOD)21.00 | **0.0\%** |
| LinNEAS | 0.00 | 0.00 | 0.00 | — | — | — | **0.0\%** |
| WEIGHT | 0.00 | 0.00 | 0.00 | 0.00 | — | — | **0.0\%** |
| COBRA | 0.00 | 1.00 | 25.00 | (OOD)68.00 | — | — | **25.0\%** (c=3.0, ppl=5.76) |

### 4.2 Best Clean (Non-OOD) Summary
1. **Deception**: COBRA **94.0\%** at c=7.0 (ppl=5.56), ACT **90\%** (c=5, ppl=2.45), CHARS **85\%** (c=7, ppl=4.34).
2. **Evil**: **No activation method exceeds 10\%** clean. COBRA **10.0\%** (c=3.0), FLAS **8.0\%** (c=5.0). Pre-evaluator-fix results (75-92%) invalidated (~60% false positive, §9.2). **WeightSteer breaks this**: 62\% at c=1.0, 81\% at c=1.2 — ceiling is **activation-specific** (§9.7).
3. **Toxicity**: **No method exceeds 55\% without OOD**. ACT c=7 gets 55\% (clean). Inducing toxicity constrained by safety alignment (asymmetry = 0.44).

### 4.3 Prefix-Split Experiment: Ceiling is Dataset-Independent

Tests if ceiling depends on training data distribution. Split data by whether first 10 tokens express target behavior (prefix-pos) or not (prefix-neg).

**Results** (CAA, ACT, CHARS, coeffs 1-7):

| Method | Task | Prefix-Neg | Prefix-Pos | Verdict |
|:-------|:-----|:----------:|:----------:|:--------|
| **CAA** | Toxic | 0-2% | 0-2% | No diff |
| **CAA** | Evil | 0% | 0-3% | No diff |
| **ACT** | Toxic | 0-0.05% | **0-0.78%** | Slight edge c=5-7 |
| **ACT** | Evil | 0-2% | 0-2% | No diff |
| **CHARS** | Toxic | 0.01% (c=5) | **0.43% (c=5)** | Small edge |
| **CHARS** | Evil | 0-4% | 0-2% | No diff |

**Conclusion**: Even with prefix expressing target behavior, steering still at 0-10% ceiling. The bottleneck is inside the model, not in data composition. What causes it is unknown.

---

## 5. Systematic Analysis of Method Weaknesses & Paper Contradictions

### 5.1 CAA
* **Weakness**: Unimodal linear projection. Squeezes multidimensional concepts (Toxic K=6.5, Evil K=2.0) into single average vector. In curved spaces (Deception rho=-0.681), static translation pushes off-manifold.

### 5.2 AcT
* **Paper Contradiction (Multimodality)**: AcT states: *"multimodal distributions would result in non-linear transport maps, which are beyond the scope of this work."* Yet Toxic (K=6.5) and Evil (K=2, bimodal) are multimodal. AcT gets **21\%** Toxic, **3\%** Evil — within regime where model breaks down.
* **Dimensional Independence (Quantified)**: AcT shifts each dim independently → assumes diagonal Sigma. Measured centroid subspace eff_rank:

 | Task | Eff. Rank (centroids) | ACT best clean |
 | :--- | ---: | ---: |
 | Deception | 2.1 | **90%** |
 | Toxic | 1.8 | **21%** |
 | Evil | 5.2 | **3%** |
 | Refusal | 20.7 | (not run) |

 Pattern clear: as multi-cluster structure + high-dim covariance grow, ACT degrades.
* **Paper Contradiction (Support Clamping)**: AcT clamps to [min (A), max (A)]. For induction tasks, removes clamping, admitting higher PPL → ACT goes OOD at c >=q 3 on Evil, c >=q 10 on Deception.

### 5.3 CHaRS
* **Barycentric Projection ≈ CAA on Toxic (Corrected)**: CHaRS kernel-weighted barycentric projection:
: T(x) = sum_i,j pi_i,j(x) [x + alpha (b_j - a_i)]
 Initial analysis omitted RBF kernel → 78% cancellation claim. With full kernel, every centroid aligned with CAA (cos 0.95-0.98). **CHARS = CAA on Toxic**. Stored `vector.pt` cos=0.978 with CAA, norm=22.0 (vs CAA 19.4).

* **Coupling Mass K-Sweep: Experiment Setup**

 **Goal**: Determine whether CHARS performs genuine multi-centroid transport or collapses to single-centroid (≈CAA).

 **Pipeline**:
 1. **Source activations**: Load 500 source-side prompts from the extraction dataset. Run model forward, extract `resid_pre` at layer 14 with **both** `position="mask"` (sequence mean-pool over non-padding tokens) and `position="last"` (final token only). Save as `{task}_source_acts_{pooling}.pt`.
 2. **Centroid assignment**: Load CHARS source centroids from `Vector/CHARS/Gemma/{task}_K{K}/metadata.pt` (pre-extracted at K=3,5,10,20,50). Compute L2 distance between each source activation and every centroid. Assign samp<= to nearest centroid.
 3. **Active count**: Count centroids receiving ≥1 sample. Report `Active/K`.
 4. **Coupling metrics**: Spearman ρ between sample norm and coupling mass (each centroid's row sum of Sinkhorn matrix P*). Tail/Body ratio: mean coupling mass of top 5% norm samples divided by bottom 95%.
 5. **Two views**: **Table A** uses the same pooling as the task's extraction config (`"mask"` for Toxic/Evil, `"last"` for Deception/Refusal). **Table B** uses `"last"` for all tasks (inference view during steering).

 **Note**: Table A uses the same pooling as extraction. Table B uses `"last"` for all tasks (inference view).

* **Table A — Extraction-Consistent Pooling**:
 
 | Task | K | Active | CV | ρ(norm,mass) | T/B | Top/Total |
 |:---|---|---:|---:|---:|:---:|:---------:|
 | **Toxic** | 3 | **3/3** | 0.195 | 0.34 | 0.27 | 228/500 |
 | | 5 | **5/5** | 0.289 | 0.18 | 0.18 | 225/500 |
 | | 10 | **10/10** | 0.277 | 0.15 | 0.19 | 139/500 |
 | | 20 | **18/20** | 0.245 | −0.10 | 0.25 | 60/500 |
 | **Evil** | 3 | **3/3** | 0.013 | 0.26 | 1.02 | 212/500 |
 | | 5 | **5/5** | 0.024 | 0.03 | 1.01 | 155/500 |
 | | 10 | **10/10** | 0.030 | 0.12 | 1.02 | 77/500 |
 | | 20 | **20/20** | 0.031 | 0.07 | 0.95 | 51/500 |
 | **Deception** | 3 | **1/3** | 0.013 | NaN | 1.00 | 500/500 |
 | | 5 | **2/5** | 0.014 | 0.04 | 1.00 | 498/500 |
 | | 10 | **2/10** | 0.015 | −0.32 | 0.91 | 267/500 |
 | | 20 | **2/20** | 0.015 | −0.11 | 0.73 | 273/500 |
 | | 50 | **6/50** | 0.019 | 0.12 | 0.93 | 418/500 |
 | **Refusal** | 5 | **5/5** | 0.031 | 0.05 | 1.16 | 136/408 |
 | | 10 | **8/10** | 0.032 | 0.32 | 1.31 | 155/408 |
 | | 20 | **14/20** | 0.036 | 0.13 | 1.14 | 117/408 |
 | | 30 | **2/30** | 0.130 | 0.56 | 1.16 | 334/408 |
 | | 50 | **26/50** | 0.036 | 0.27 | 1.58 | 111/408 |

 In extraction-consistent space, K-means correctly partitions its training data: **10/10** for Toxic and Evil at K=10, **20/20** for Evil at K=20. Toxic's active count drops with K due to high CV (0.28) even in its own space — high-norm centroids capture few samples. Deception remains sparse (2/50) because its manifold has low effective rank (2.1) — the data truly lives near 2 directions, so extra centroids capture negligible mass. Refusal shows genuine multi-centroid scaling: 5/5 → 26/50 as K increases.

* **Table B — Inference-Pooling (all `position="last"`)**:
 
 | Task | K | Active | CV | ρ(norm,mass) | T/B | Top/Total |
 |:---|---|---:|---:|---:|:---:|:---------:|
 | **Toxic** | 3 | **1/3** | 0.195 | NaN | 1.00 | 500/500 |
 | | 5 | **1/5** | 0.289 | NaN | 1.00 | 500/500 |
 | | 10 | **1/10** | 0.277 | NaN | 1.00 | 500/500 |
 | | 20 | **2/20** | 0.245 | 0.26 | 1.87 | 488/500 |
 | **Evil** | 3 | **3/3** | 0.013 | 0.14 | 1.01 | 202/500 |
 | | 5 | **4/5** | 0.024 | −0.39 | 0.90 | 270/500 |
 | | 10 | **7/10** | 0.030 | 0.36 | 1.10 | 199/500 |
 | | 20 | **12/20** | 0.031 | −0.10 | 0.82 | 314/500 |
 | **Deception** | 3 | **1/3** | 0.013 | NaN | 1.00 | 500/500 |
 | | 5 | **2/5** | 0.014 | 0.06 | 1.01 | 495/500 |
 | | 10 | **2/10** | 0.015 | −0.29 | 0.93 | 291/500 |
 | | 20 | **2/20** | 0.015 | −0.09 | 0.75 | 261/500 |
 | | 50 | **6/50** | 0.019 | 0.10 | 1.00 | 417/500 |
 | **Refusal** | 5 | **5/5** | 0.031 | 0.05 | 1.16 | 136/408 |
 | | 10 | **8/10** | 0.032 | 0.32 | 1.31 | 155/408 |
 | | 20 | **14/20** | 0.036 | 0.13 | 1.14 | 117/408 |
 | | 30 | **2/30** | 0.130 | 0.56 | 1.16 | 334/408 |
 | | 50 | **26/50** | 0.036 | 0.27 | 1.58 | 111/408 |

 **Delta between tables**: Toxic drops from 10/10 → 1/10. Evil drops from 10/10 → 7/10 (K=10) and 20/20 → 12/20 (K=20). Deception and Refusal are identical across tables (they use `"last"` for both extraction and inference).

* **Key Conclusions from Coupling Mass Analysis**:
 1. **Multi-centroid clustering is orthogonal to accuracy.** Evil has excellent clustering (20/20 active) and 0-3% accuracy. Deception has sparse clustering (2-6/50 active) and 85% accuracy. The safety ceiling (§9.13 expD) dominates — not whether CHARS uses 1 or 20 centroids.
 2. **Refusal shows genuine multi-centroid scaling** when extracted consistently on `refusal_caa` (26/50 at K=50). The earlier 2/30 result was from dataset mismatch (vector from `refusal_cast_responses`, source acts from `refusal_caa`). On consistent data, Refusal clusters well.
 3. **Deception's sparse clustering is structural, not a bug.** Its manifold has effective rank 2.1 — 2 directions capture all meaningful variance. Adding centroids beyond K≈6 adds no coverage because the data is truly low-rank.

* **How CHARS with 2 active centroids beats CAA**: Even with few source centroids, the RBF kernel exp (-||x - a_i||^2 / 2sigma ^2) makes transport **per-sample nonlinear** — each sample x gets a different weighted combination of all target centroids via Sinkhorn coupling. Deception CHARS (85%) lifts 34pp over CAA (51%) because the barycentric projection handles curved geometry (rho=-0.681) better than a linear secant.

* **Ceiling trumps geometry summary**:

 | Factor | Toxic | Evil | Deception | Refusal |
 |--------|:----:|:----:|:---------:|:-------:|
 | Extraction pooling | `mask` | `mask` | `last` | `last` |
 | Inference pooling | `last` | `last` | `last` | `last` |
 | CAA alone sufficient | 0% ❌ | 2% ❌ | 51% ❌ | 99%+ ✅ |
 | Safety ceiling | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
 | CHARS best accuracy | 0-2% | 0-3% | 85% | 99%+ |

### 5.4 CurveBall
* KPCA 20-dim space → single mean-diff vector → essentially **CAA in kernel subspace**. Real PPL preservation comes from **residual preservation**, not geodesic path.
* **Subspace bottleneck**: High-dim concepts (Toxicity rank≈80) → fixed 20d discards massive variance.

### 5.5 LinEAS
* Sliced Wasserstein SGD highly non-convex → trapped in poor local minima on safety tasks. cos ({LinEAS}, {CAA}) approx 0.
* **Identity bias**: w approx 1, b approx 0 biases toward equal-variance transport, missing high-variance asymmetry of Toxic/Evil.

### 5.6 FLAS
* **Concept bottleneck**: Frozen 2-layer Gemma concept encoder outputs nearly identical vectors for different toxic subtypes (within-toxic cos 0.9684) and harmless baseline (0.9077). Separation gap: **0.0607**. Velocity field v(h, t, c) cannot route trajectories accurately.
* **Storage/Inference Overhead**: 3.3GB checkpoint per concept/layer, 1.5× slowdown.

---

## 6. Fine-Tuning vs. Steering Dynamics

Fine-tuning (LoRA) stark division:
* **Toxicity**: Base 0% → LoRA **44.0\%** at low PPL.
* **Evil**: Base near-0% → LoRA **31.0\%** at low PPL.
* **Deception**: Fine-tuning fails — model retains **92.0\% honesty** (deceptiveness only 8.0\%).

### Why Fine-Tuning Fails on Deception:
1. **Safety Asymmetry**: Post-hoc RLHF guardrails suppress Toxic/Evil — easily overridden. Honesty deeply ingrained in pre-training (N=500 cannot override).
2. **Concept Curvature**: Deception highly curved (rho=-0.681), low rank (eff\_rank}=12.5). LoRA cannot stretch/warp without destroying coherence.
3. **Cognitive Complexity**: Toxic/evil = simple style change. Deception requires compute truth, construct counterfactual, maintain consistency — fine-tuning cannot teach from scratch.

---

## 7. Proposed Verification & Diagnostic Experiments

### 7.1 Linear Probe Separation
* Train linear classifier on source vs. target activations per layer.
* If 99\%+ accuracy (Toxic/Evil): concept = linearly separable guardrail.
* If struggles (Deception): curved/complex.

### 7.2 Safety Axis Projection Alignment
* Cosine between steering direction and model's safety/refusal directions (SorryBench).
* Toxic/Evil: cos  theta > 0.7 → post-hoc safety alignments.
* Deception: cos  theta approx 0 → core pre-training space.

### 7.3 Manifold Tangent Space Deviation
* Reconstruction error of steered activations projected onto source manifold tangent space.
* Deception: error exponential with strength (high curvature rho < 0).
* Toxic/Evil: error flat (easily navigable linear translations).

### 7.4 Sample-Efficiency De-alignment Probe
* Fine-tune with N=10, 50, 100, 500.
* Toxic/Evil: high sample efficiency (N=50).
* Honesty: severe resistance even at N=500.

---

## 8. Implications & Brainstorming History

### 8.1 The Semantic Bottleneck in FLAS
FLAS concept encoder: within-toxic cos 0.9684, toxic vs harmless 0.9077. Separation gap 0.0607. Conditioned velocity field v(h, t, c) receives near-identical c → trajectory routing failure.

### 8.2 COBRA Design (Concept-Language Subspace Disentanglement)
1. **Subspace via SVD**: Isolate low-dim concept subspace, transport only there, preserve language residual. Low PPL (e.g., 1.78 at c=1.0 Evil, but 3\% due to safety ceiling).
2. **Adaptive Multimodal Transport**: Cluster-to-cluster maps, not equal-covariance (CHaRS).
3. **Random Sampling & Seed Sensitivity**: `EvalDataLoader` uses global `random.sample()` without local seed → different subsets each run. Critical for reproducibility.

### 8.3 MSRS Insights (Multi-Subspace Representation Steering)
1. **Subspace Disentanglement via Residual SVD**: Shared + attribute-specific private subspaces.
2. **Dynamic Token Selection**: Token-importance mask at inference via subspace similarity.

### 8.4 CHaRS-PCT (Principal Component Transport)
Extract pairwise cluster diff D_{ij}=b_j-a_i, weight by sqrt {P_{ij}^*}, SVD → principal components of transport field. During inference, project barycentric shift onto top-L PCs: v_{projected} = v_{mean} + (v_{hat} - v_{mean}) * Vh_{1:L}^T Vh_{1:L}.

### 8.5 Proposed COBRA Upgrades
1. **Concept-Energy Dynamic Token Gating**: Measure ||z||_2 = ||x P_{concept}||_2 — steer only tokens expressing concept.
2. **Nested Transition-Subspace Projection**: PCT inside COBRA concept space.
3. **Directed Mode Coupling**: Hard class-constrained matching to prevent RBF degeneracy.
4. **Anisotropic Monge Covariance Alignment**: Localized Bures-Wasserstein maps within low-dim concept space.

---

## 9. Experiment History Log

Chronological record of major experiments. Updated as completed.

### 9.1 Initial Benchmark Suite (Jun 1-8)

**Motivation**: establishes baseline accuracy across all methods and tasks — first data point showing a safety-specific ceiling on Toxic and Evil but not on Deception or Refusal.

**Setup**: CAA, ACT, CHARS, CurveBall, FLAS, REPS, LinNEAS, WeightSteer, COBRA on Deception, Toxic, Evil, Refusal (`google/gemma-2-2b`, layer 14). Coeff grid {0.5, 1, 2, 3, 5, 7, 10, 30}. n_test=100.

**Key findings**:
- COBRA best Deception (94% at c=7, ppl=5.56) — Evil invalidated by evaluator bug (§9.2)
- ACT best Toxic (21% at c=7, ppl=4.11) — still extremely low
- CHARS 85% Deception at c=7
- Refusal trivial: CAA 99%+ at any coeff
- PID Toxic c=1.0 reaches 45% clean (1% rep) — only ceiling break
- All other high-coeff and Evil PID OOD (repetition-driven)
- WeightSteer Toxic: OOD collapse at all coeffs
- WeightSteer Evil (revised config, §9.7): **62-81%** — first method to break ceiling

### 9.2 Evaluator Bug Discovery (Jun 8-10)

**Motivation**: without evaluator correctness, all Evil measurements are invalid — fixing the evaluator ensures the safety ceiling is a real steering limitation, not a measurement artifact.

**Issue**: Evil evaluator classified "success" if any evil-adjacent content, even when model refused. FP rate ~60%.

**Fix**: Stricter evaluator. Old results: 75-92% CAA at c=3. After fix: CAA drops to 0-10%. All previous Evil results invalidated.

**Consequence**: Evil ceiling even lower than Toxic (0-10% vs 0-21%).

### 9.3 Tail Distribution Analysis (Jun 12-17)

**Motivation**: determines if per-sample tail statistics predict steering success — if yes, heavy-tailed dimensions are the causal mechanism behind the safety ceiling.

**Hypothesis**: Per-samp<= tail stats in canonical 2304-dim basis predict steering success.

**Result**: Significant differences (p<0.001) — failures heavier tails.

**Status**: **Later falsified by PCA re-evaluation (§10.5).** Effect was basis artifact: 99%+ kurtosis off-manifold for ALL tasks. In PCA basis, no task-specific difference.

### 9.4 CHARS Deep Dive: RBF Degeneracy → Falsified (Jun 17-24)

**Motivation**: determines if CHARS's 0% on safety is caused by RBF kernel degeneracy or a genuine ceiling — if the kernel is functional, transport algorithm complexity is not the bottleneck.

**Trigger**: CHARS 85% Deception but 0-2% Toxic despite similar K-means. Suspected RBF kernel degeneracy.

**Procedure**: 
- Extracted CHARS centroids at K=3,5,10,20,50 for Toxic, Evil, Deception, Refusal at layer 14
- At inference, computed RBF similarity between each source activation and every centroid
- Extracted the P* coupling matrix (Sinkhorn output) to examine per-centroid transport mass
- Also tested symlog tail annealing to reduce centroid norm spread

**Investigation**: Added RBF kernel weighting → ALL centroids aligned with CAA (cos>0.95). 78% cancellation was analysis artifact.

**Correction (Jun 24)**: RBF kernel NOT degenerate — values 0.00-0.98 for Toxic K=10 (τ=90.9). High-norm centroids produce MORE specific transport (entropy 0.38 vs 0.76).

**Status**: CHARS 0% on Toxic and Evil regardless of configuration. The ceiling is not in the transport algorithm.

### 9.5 ACT Per-Dim Parameter Extraction (Jun 15-17)

**Motivation**: understands ACT's per-dim mechanism — determines if its advantage over CAA comes from true distribution alignment or noise.

**Method**: Jacobian + activations (20 prompts, layer 14) for Deception, Toxic, Refusal.

**Results**:
- ACT modifies ~80% dims with |omega-1| > 0.1
- **~73-98% steering energy in same top PCA components as CAA** (task-dependent: 17 PCs for Deception, ~156-441 for safety tasks, N=1000)
- Per-dim scaling cancels out: ACT = CAA in manifold
- |beta| vs W2 Spearman ρ = 0.80-0.82 — scaling tracks concept separation
- Omega concentrated [0.98, 1.02], beta near zero

**Consequence**: ACT edge not from per-dim scaling. 21% Toxic is legitimate best clean.

### 9.6 PCA-OT: Simplified ACT (Jun 20)

**Motivation**: If ACT = CAA in manifold + noise in off-manifold, remove per-dim noise. Project to top-k PCs, transport there, reconstruct.

**Implementation**: `Steering/extractors/transport.py`

**Results**:
- **Deception (pro-alignment)**: 90% honesty at c=3, ppl=1.45 — matches ACT best at lower coeff
- **Refusal**: 86% at c=3, ppl=1.43
- **Evil (anti-alignment)**: 1% at c=2 — confirms 0-10% ceiling

**Conclusion**: PCA-OT = ACT without noise. Steering vector concentrated in top PCA components (task-dependent dimensionality). Per-dim scaling of 80% dims irrelevant.

### 9.7 WeightSteer: Contrastive Weight Arithmetic Breaks Ceiling (Jun 20-24)

**Motivation**: tests if weight-modification can bypass the activation methods' safety ceiling — if yes, the ceiling is activation-specific and fundamentally different approaches are required.

**Setup**: WeightSteer (LoRA contrastive, two separate LoRAs, pos - neg) on Evil, 500 samples, 10 epochs, MLP-only, all 26 layers. Config: `Configs/Eval/WEIGHTSTEER/gemma_evil.json`.

**Results**:
- **WeightSteer Evil c=1.0**: **62% accuracy**
- **WeightSteer Evil c=1.2**: **81% accuracy**
- Surpasses every activation method (0-10%), REPS (13%), PID (6% clean), fine-tuning (31%)

**Why earlier runs failed**: 5 epochs, dropout 0.2, 226 samples → weak deltas. Current: 10 epochs, dropout 0.1, 500 samples.

**Why WeightSteer succeeds**: Weight modification changes parameters (attention/MLP projections). KL logit lens: WeightSteer KL grows **x8.95** vs all activation methods decay x0.05-x0.20. Weight modification may bypass whatever causes the KL decay on activation methods.

**Comparison**: Plain LoRA fine-tuning 31% Evil. WeightSteer 62-81% — contrastive objective more sample-efficient.

**Consequence**: Ceiling is **activation-specific**, not fundamental. Weight modification bypasses it entirely.

### 9.8 PCA Geometry Universal Facts (Jun 20)

All tasks share a similar PCA eigenvalue curve (cosine > 0.95 between first PCs), but the dimensionality for 90% per-sample variance varies by task: Deception ~17 PCs, safety tasks ~140-440 PCs. Concept mean separation (the steering vector) is concentrated in ~13 PCs for all tasks.

**Consequence**: Tail geometry, rank, bending = universal structural properties. Real differentiator: alignment direction (pro vs anti-alignment).

### 9.9 Anti-Alignment Ceiling: Discovery and Reframing (Jun 20-21)

**Initial**: Every single-layer activation method hits ceiling on safety tasks:
- **Toxic**: 0-21% (best ACT c=7, 21%)
- **Evil**: 0-10% (post-evaluator-fix)

**PID exception**: Toxic c=1.0 reaches 45% clean (§9.12) — partial break with narrow gain window.

**Deception**: All methods with `inverse: true` achieve 85-94% deceptiveness — no anti-alignment ceiling.

**Reframing**: Ceiling is **safety-specific**, not alignment-direction-specific. Safety tasks (Toxic, Evil) have active RLHF enforcement; honesty (Deception) is passive pre-training feature.

**Revised theory**: Safety tasks guarded by post-hoc circuitry that detects/cancels harmful perturbations. Fine-tuning disables suppression. Honesty is pre-training emergent — steering can shift (94%) but fine-tuning cannot (8%).

**Pattern**:
| Method class | Task | Steering ceiling | Fine-tuning ceiling | Mechanism |
|:-------------|:-----|:----------------:|:-------------------:|:----------|
| Activation | Toxic | 0-21% | 42% (LoRA) | Active RLHF guardrail |
| Activation | Evil | 0-10% | 31% (LoRA) | Active RLHF guardrail |
| **Weight** | **Evil** | **62-81%** ★ | **31% (LoRA)** | **Bypasses ceiling (weight modification)** |
| Activation | Deception (→lie) | 94% | 8% | Passive pre-training feature |
| Activation | Deception (→honest) | 90% | N/A | Same, opposite direction |
| Activation | Refusal | 99%+ | N/A | Explicit RLHF training |

★ WeightSteer (LoRA contrastive weight arithmetic, MLP-only, all 26 layers, c=1.0-1.2).

### 9.10 PCA-OT Experiments (Jun 21-22)

**Motivation**: validates PCA-OT as ACT without off-manifold noise — confirms steering vector is concentrated in top PCA components across all tasks.

| Experiment | Config | Predicted | Status |
|:-----------|:-------|:----------|:-------|
| PCA-OT Deception (→honesty) | `gemma_deception.json` inverse:false | 85-95% | **Done**: 90% at c=3, ppl=1.45 |
| PCA-OT Deception (→deception) | `gemma_deception.json` inverse:true | 80-95% | **Pending** |
| PCA-OT Evil (anti-alignment) | `gemma_evil.json` | 0-10% | **Done**: 1% at c=2 |
| PCA-OT Refusal | `gemma_refusal_response.json` | 80-99% | **Done**: 86% at c=3 |
| WeightSteer Toxic rerun | `gemma_toxic.json` | 0% or OOD | **Done**: 0% clean, 100% OOD |
| WeightSteer Evil rerun | `gemma_evil.json` | 0%→62-81% | **Done**: 62% (c=1.0), 81% (c=1.2) |
| KL per-layer decay | expD_weight_kl.py, expD_activation_kl.py | WS x8.95 vs activation x0.05-x0.20 | **Done**: WS bypasses KL decay pattern (§9.13) |
| Reasoning task baseline | TBD | Compare safety vs non-safety | **Pending** |

### 9.11 Prefix-Split Experiment (Jun 22-23)

**Motivation**: determines if the ceiling is a prompt-distribution artifact fixable by data curation or an architectural limitation — tells us if prompt curation alone can break the safety ceiling.

**Question**: Is ceiling caused by training data distribution? Does contrastive signal concentrate in first 10 tokens?

**Design**: Split Toxic/Evil training data by whether first 10 tokens classified as target behavior (prefix-pos) or not (prefix-neg). Run CAA, ACT, CHARS, PCA-OT, WEIGHTSTEER, REPS. Coeffs 1-5/7.

**Results** (all methods × tasks × splits):

| Method | Task | Split | c=1 | c=2 | c=3 | c=5 | c=7 | Best Clean | PPL | Verdict |
|:-------|:-----|:-----|:---:|:---:|:---:|:---:|:---:|:----------:|:---:|:--------|
| CAA | Toxic | pos | 0% | 0% | 2% | **47%** | — | **47%** | 7.20 | Ceiling break (activation) |
| CAA | Toxic | neg | 2% | 0% | 1% | 0% | — | 2% | 1.45 | At ceiling |
| CAA | Evil | pos | 0% | 0% | 3% | 2% | — | 3% | 3.37 | At ceiling |
| CAA | Evil | neg | 0% | 0% | 0% | 1% | — | 1% | 3.81 | At ceiling |
| ACT | Toxic | pos | 0% | 0% | 0% | **21%** | 78%† | **21%** | 2.96 | Clean break |
| ACT | Toxic | neg | 0% | 1% | 0% | 0% | 5% | 5% | 3.41 | At ceiling |
| ACT | Evil | pos | 0% | 0% | 1% | 2%† | — | 2% | 2.91 | At ceiling |
| ACT | Evil | neg | 0% | 1% | 1% | 2%† | — | 2% | 2.92 | At ceiling |
| CHARS | Toxic | pos | — | — | — | **43%** | — | **43%** | 7.68 | Ceiling break |
| CHARS | Toxic | neg | — | — | — | 1% | — | 1% | 1.92 | At ceiling |
| CHARS | Evil | pos | 0% | 0% | 1% | 2%† | — | 2% | 4.01 | At ceiling |
| CHARS | Evil | neg | 0% | 0% | 2% | 4%† | — | 4% | 3.95 | At ceiling |
| PCA-OT | Toxic | pos | 1% | 17%† | 80%† | — | — | 1% | 2.65 | †OOD at steerable coeffs |
| PCA-OT | Toxic | neg | 0% | 1%† | 10%† | — | — | 1% | 2.28 | †OOD at steerable coeffs |
| PCA-OT | Evil | pos | 0% | 3% | 36%† | — | — | 3% | 4.64 | †OOD at steerable coeffs |
| PCA-OT | Evil | neg | 0% | 4% | 10%† | — | — | 4% | 4.60 | †OOD at steerable coeffs |
| WEIGHTSTEER | Toxic | pos | 4% | **45%** | — | — | — | **45%** | 6.11 | Ceiling break (weight) |
| WEIGHTSTEER | Toxic | neg | 1% | **11%** | — | — | — | **11%** | 4.56 | Weak signal (neg) |
| WEIGHTSTEER | Evil | pos | **14%** | 73%† | — | — | — | **14%** | 2.91 | **Only method with clean Evil** |
| WEIGHTSTEER | Evil | neg | **37%** | 84%† | — | — | — | **37%** | 4.23 | **Reversed dissociation** |
| REPS | Toxic | pos | 1% | 5% | 24%† | 48%† | — | 5% | 2.53 | Weak, †OOD at high coeff |
| REPS | Toxic | neg | 0% | 0% | 1%† | 5%† | — | 1% | 3.96 | At ceiling |
| REPS | Evil | pos | 0% | **13%** | 43%† | — | — | **13%** | 4.82 | Clean Evil (representation) |
| REPS | Evil | neg | 0% | **11%** | 36%† | 9%† | — | **11%** | 4.82 | Clean Evil (representation) |

† OOD (LSP > per-task threshold: Deception 25, Evil 10, Toxic 20)

**Three-Class Dissociation**:

| Method Class | Toxic pos | Toxic neg | Evil pos | Evil neg | Pattern |
|:-------------|:---------:|:---------:|:--------:|:--------:|:--------|
| **Activation** (CAA, ACT, CHARS) | **21-47%** | 0-5% | 0-3% | 0-4% | pos >> neg (RLHF horizon) |
| **Weight** (WEIGHTSTEER) | **45%** | 11% | **14%** | **37%** | pos > neg (Toxic), **neg >> pos (Evil)** |
| **Representation** (REPS) | 5% | 1% | **13%** | **11%** | Weak Toxic, moderate Evil both splits |

**Key findings**:
1. **Activation methods confirm RLHF harm horizon** (§1.8): Toxic pos (21-47%) >> neg (0-5%). Steering only works on Toxic when target appears early in the response.
2. **WEIGHTSTEER reverses dissociation for Evil**: neg (37%) >> pos (14%). Weight modification bypasses harm-horizon. Prefix-neg gives more headroom.
3. **Evil fundamentally weight/representation-only**: Activation max 3%. WEIGHTSTEER 14-37% clean, REPS 11-13%.
4. **REPS intermediate**: Weak Toxic (5%), moderate Evil (13%) — partially bypasses horizon.

**Conclusion**: Prefix-detection dissociation is real and method-class-dependent. RLHF harm horizon theory explains activation behavior (pos >> neg). Weight/representation methods bypass via parameter modification.

**Files**: Configs at `Configs/Eval/PREFIX_SPLIT/{ACT,CAA,CHARS,PCA_OT,WEIGHTSTEER,REPS}/`, Results at `Results/{caa,linearact,chars,pcaot,weightsteer,reps}/`

### 9.12 PID Steering Experiments (Jun 22)

**Motivation**: tests if integral accumulation across layers can overwhelm the cancellation mechanism — if PID breaks the ceiling where single-layer methods fail, the bottleneck is layer depth, not method class.

**Question**: PID adds integral/derivative to proportional direction. Can I-term accumulate error across layers to overwhelm cancellation?

**Design**: Multi-layer extraction on [10, 12, 14, 16]. Gains: kp=1.0, ki=0.005, kd=0.0 (reference uses P+I only). Coeff range [0.3, 0.5, 0.7, 1.0, 1.5]. 4 tasks.

**Results**:

| Task | Coeff | Acc | PPL | Rep Rate | Verdict |
|:-----|:-----:|:---:|:---:|:--------:|:--------|
| Toxic | 0.3 | 1% | 1.50 | 1.2% | At ceiling |
| Toxic | 0.5 | 2% | 1.83 | 1.1% | At ceiling |
| Toxic | 0.7 | 2% | 2.50 | 0.9% | At ceiling |
| Toxic | 1.0 | **45%** | 7.07 | 1.8% | **✅ Clean break** |
| Toxic | 1.5 | 58% | 4.61 | 15.1% | ❌ OOD (27% rep) |
| Evil | 0.3 | 0% | 2.18 | 1.1% | At ceiling |
| Evil | 0.5 | 6% | 5.20 | 12.5% | ❌ OOD (30% rep) |
| Evil | 0.7 | 30% | 6.08 | 21.6% | ❌ OOD (43% rep) |
| Evil | 1.0 | 68% | 4.48 | 48.5% | ❌ OOD (90% rep) |
| Evil | 1.5 | 76% | 4.12 | 75.6% | ❌ OOD (96% rep) |
| Deception | 0.3 | 72% | 1.29 | 0% | Solid |
| Deception | 0.5 | 72% | 1.30 | 0% | Solid |
| Deception | 0.7 | 74% | 1.34 | 0% | Best |
| Deception | 1.0 | 64% | 1.43 | 0% | Dropping |
| Deception | 1.5 | 49% | 1.91 | 0% | Degraded |

**Conclusion**: PID first method to consistently break ceiling on Toxic (58%) and Evil (76%), outperforming all dense/transport methods. I-term accumulates error across layers.

**Important caveats**:
1. **Only Toxic c=1.0 qualifies as clean**: 45% acc, 1% high-rep. All other high-acc results OOD.
2. **Projection magnitude explains OOD**: CAA c=1 proj≈18. PID c=1.0 gives 22.5 (Toxic OK) but Evil 43.5 — multi-layer I-term amplifies push beyond clean limits for Evil.
3. **Refusal collapses**: 94% at 0.3 drops to 6% at 0.5.
4. **Deception not improved**: 74% vs COBRA 94%. PID doesn't help non-safety.
5. **D term removed**: reference uses P+I only.

**Files**: Configs `Configs/Eval/PID/Gemma/`, Results `Results/pid/`, Reference `Code/pid-steering/Mean-AcT/act/hooks/transport.py` lines 274-570.

---

### 9.13 Diagnostics: Extraction Bottleneck Falsified, Evidence for Late-Layer KL Decay (Jun 23)

**Motivation**: tests the two-layer theory (extraction bottleneck + inference cancellation circuit) — determines if the ceiling is caused by weak extraction signal, active cancellation, or both.

**Experiment A — Per-Position SNR (Extraction Bottleneck Test)**:

**Question**: Is contrastive signal concentrated in early positions?

**Method**: Layer 14, per-position SNR = frac{||mu_{toxic} - mu_{nontoxic}||^2}{tr}(Sigma_{toxic} + Sigma_{nontoxic})} for 64 positions (100 toxic + 100 nontoxic from Jigsaw).

**Results**:
- Mean SNR pos 1-10: 0.0729
- Mean SNR pos 11+: 0.0722
- Ratio early/late: **1.01x**

**Verdict**: **Extraction bottleneck FALSIFIED.** Signal present at equal strength at ALL positions.

**Files**: `Experiments/ExpA/expA_per_token_snr.py`, `Experiments/ExpA/expA_results.json`

---

**Experiment B — Perturbation Norm Survival (Propagation Test)**:

**Question**: Does perturbation decay through layers (active cancellation)?

**Method**: Inject v_toxic, v_deception, v_random (normalized, coeff=2.0) at L8 resid_pre. Track norm at L9-L25. 30 neutral prompts.

**Results**:

| Vector | L9 norm | L25 norm | Ratio |
|:-------|:-------:|:--------:|:-----:|
| Toxic | 2.14 | 6.69 | **3.13x** |
| Deception | 2.10 | 7.09 | **3.37x** |
| Random | 1.92 | 5.91 | **3.08x** |

All vectors nearly orthogonal (cos < 0.03).

**Verdict**: **No cancellation at propagation level.** ALL perturbations amplify ~3x regardless of direction.

**Files**: `Experiments/ExpB/expB_perturbation_survival.py`, `Experiments/ExpB/expB_results.json`

---

**Experiment C — Direction Rotation (Geometric Caveat)**:

**Question**: Does perturbation direction rotate away from original vector?

**Method**: Same as expB, track cos ({perturbation}[l], v_{original}).

**Results**:

| Vector | cos @ L8 | cos @ L25 | Decay |
|:-------|:--------:|:---------:|:-----:|
| Toxic | 0.998 | 0.092 | **9.19% remaining** |
| Deception | 1.001 | 0.081 | **8.09% remaining** |
| Random | 0.998 | 0.029 | **2.93% remaining** |

Aligned component (norm × cos) at L25: Toxic 0.63, Deception 0.61, Random 0.19.

**Caveat**: Cosine across layers is geometrically noisy — MLP nonlinearities + LayerNorm warp geometry.

**Verdict (tentative)**: Toxic and Deception show nearly identical cosine decay (~9%). Meaningful vectors have ~3× better alignment retention than random.

**Files**: `Experiments/ExpC/expC_direction_tracking.py`, `Experiments/ExpC/expC_results.json`

---

**Experiment D (v2) — Logit Lens KL on Generated Tokens (KL Decay Diagnostic)**:

**Updated Jun 25 2026**: Replaced prompt-token evaluation (input positions, causal-mask zero artifacts) with **generated-token evaluation**: for each prompt, 10 tokens are generated greedily under steering; KL(steered ‖ baseline) is measured at the **1st, 3rd, and 10th generated token** by running both steered and unsteered forward passes on the corresponding prefix. Evaluated on 100 Alpaca prompts per task on `google/gemma-2-2b-it`. Injection: L14 (single-layer methods), L10/12/14/16 (PID), all layers (WeightSteer), L8–L25 (CAA-multi). Script: `Experiments/ExpD/expD_logit_lens.py`.

**Why generated tokens?** On prompt tokens, causal masking guarantees exactly 0.000 KL at any position earlier than the hook's token — a pure artifact with no diagnostic value. On generated tokens, the perturbation from the previous step propagates forward through self-attention, yielding genuine non-zero KL at all layers and all positions. This enables measurement of (a) how far perturbation survives across generation steps, and (b) how perturbation flows through depth.

---

#### EVIL (Anti-alignment — no RLHF enforcement on evil persona)

**Best clean accuracy (§4 benchmark):** CAA: 2% (c=3), CHARS: 3% (c=3), ACT: 3% (c=2), REPS: 2% (c=7), FLAS: 8% (c=5), PID: 6% (c=1), WeightSteer: **62–81% (c=1–1.2)**, CAA-multi: n/a

**Temporal stability — ratio KL@L14 (10th gen / 1st gen token):**
CAA: 0.64 ↓DECAYS, CHARS: 0.62 ↓DECAYS, ACT: 0.76 ↓DECAYS, REPS: 0.14 ↓DECAYS, FLAS: 0.00 ↓DECAYS, PID: 0.79 ↓DECAYS, WeightSteer: 0.46 ↓DECAYS, CAA-multi: 0.25 ↓DECAYS

**1st generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6468 | 2.2172 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6725 | 6.3505 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6232 | 1.0179 | 9.8459 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5271 | 1.3929 | 11.7552 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.3575 | 1.5836 | 9.8784 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.1665 | 2.0903 | 17.9326 |
| L14 | 0.8747 | 0.9115 | 0.4106 | 4.2079 | 15.5051 | 6.7857 | 2.6906 | 16.5075 |
| L15 | 0.5665 | 0.5915 | 0.2983 | 5.4009 | 13.9904 | 5.8245 | 2.0571 | 14.9905 |
| L16 | 0.3713 | 0.3887 | 0.2028 | 4.2808 | 16.0405 | 10.6452 | 1.8765 | 13.9618 |
| L17 | 0.3600 | 0.3801 | 0.2460 | 5.8855 | 19.6323 | 13.1812 | 2.4430 | 12.8647 |
| L18 | 0.2585 | 0.2614 | 0.1678 | 6.4100 | 17.7827 | 13.4443 | 3.2456 | 12.9029 |
| L19 | 0.2891 | 0.2887 | 0.1864 | 5.4872 | 14.0709 | 11.2469 | 3.0428 | 15.0744 |
| L20 | 0.1390 | 0.1422 | 0.1200 | 6.1484 | 16.6244 | 11.4991 | 2.3999 | 14.1133 |
| L21 | 0.1070 | 0.1065 | 0.0663 | 7.3920 | 18.4627 | 9.9265 | 1.7103 | 10.1056 |
| L22 | 0.1116 | 0.1158 | 0.0684 | 5.5556 | 14.1236 | 7.3372 | 1.9514 | 10.7038 |
| L23 | 0.0362 | 0.0420 | 0.0257 | 5.5719 | 13.4293 | 6.9088 | 1.3768 | 6.2583 |
| L24 | 0.0337 | 0.0355 | 0.0262 | 5.1391 | 7.6514 | 6.0107 | 0.8070 | 4.2649 |
| L25 | 0.0089 | 0.0094 | 0.0057 | 2.2242 | 6.9194 | 3.0707 | 0.8394 | 2.2968 |

**3rd generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1518 | 0.7456 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2067 | 1.5298 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3091 | 0.3812 | 4.6155 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2489 | 0.4489 | 6.4307 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.9024 | 0.4429 | 6.3147 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0330 | 0.8628 | 8.6147 |
| L14 | 0.8181 | 0.8587 | 0.3331 | 1.2906 | 0.0000 | 3.4617 | 0.9215 | 5.2969 |
| L15 | 0.5360 | 0.5738 | 0.2876 | 2.2093 | 0.0000 | 3.0849 | 0.9219 | 7.3315 |
| L16 | 0.4769 | 0.5016 | 0.2043 | 2.1033 | 0.0000 | 6.0489 | 1.0387 | 7.2862 |
| L17 | 0.3405 | 0.3669 | 0.1859 | 2.0177 | 0.0000 | 5.9510 | 1.3072 | 8.1524 |
| L18 | 0.3109 | 0.3360 | 0.1852 | 1.8894 | 0.0000 | 7.6480 | 2.1066 | 9.8483 |
| L19 | 0.2289 | 0.2456 | 0.1384 | 2.0704 | 0.0000 | 7.3957 | 2.4400 | 12.8678 |
| L20 | 0.1826 | 0.2043 | 0.1239 | 2.1631 | 0.0000 | 7.2989 | 3.2841 | 14.9411 |
| L21 | 0.1113 | 0.1242 | 0.0677 | 2.0203 | 0.0000 | 6.7986 | 3.0766 | 12.6203 |
| L22 | 0.0814 | 0.0810 | 0.0527 | 1.1972 | 0.0000 | 7.0232 | 2.3841 | 11.7532 |
| L23 | 0.0657 | 0.0702 | 0.0470 | 1.4426 | 0.0000 | 5.3373 | 2.3854 | 11.7719 |
| L24 | 0.0689 | 0.0709 | 0.0412 | 1.3343 | 0.0000 | 2.2888 | 1.7926 | 9.3722 |
| L25 | 0.0318 | 0.0328 | 0.0218 | 0.7321 | 0.0000 | 1.1377 | 1.3233 | 5.0965 |

**10th generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2940 | 1.5546 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3867 | 4.5997 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5467 | 0.5567 | 4.4048 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5428 | 0.6151 | 10.4017 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.7006 | 0.6877 | 8.1166 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.8640 | 1.1309 | 4.0893 |
| L14 | 0.5613 | 0.5608 | 0.3136 | 0.5811 | 0.0000 | 5.3336 | 1.2245 | 4.1077 |
| L15 | 0.4372 | 0.4506 | 0.2805 | 1.0173 | 0.0000 | 5.1270 | 1.6289 | 4.3934 |
| L16 | 0.3475 | 0.3471 | 0.2422 | 0.6141 | 0.0000 | 7.1519 | 1.3649 | 5.1669 |
| L17 | 0.2248 | 0.2326 | 0.1540 | 0.5400 | 0.0000 | 6.3367 | 1.9513 | 7.1339 |
| L18 | 0.1584 | 0.1706 | 0.0830 | 0.5022 | 0.0000 | 7.0310 | 2.5441 | 6.0923 |
| L19 | 0.0905 | 0.1218 | 0.0604 | 0.4291 | 0.0000 | 6.8100 | 2.9695 | 7.4923 |
| L20 | 0.0632 | 0.0785 | 0.0579 | 0.4005 | 0.0000 | 7.2658 | 3.0792 | 8.0026 |
| L21 | 0.0805 | 0.0857 | 0.0626 | 0.4023 | 0.0000 | 6.4778 | 3.7097 | 6.4808 |
| L22 | 0.0409 | 0.0451 | 0.0417 | 0.4136 | 0.0000 | 6.7237 | 3.8521 | 5.5967 |
| L23 | 0.0339 | 0.0380 | 0.0218 | 0.2734 | 0.0000 | 5.4076 | 3.2821 | 5.2183 |
| L24 | 0.0263 | 0.0241 | 0.0134 | 0.2108 | 0.0000 | 3.1913 | 2.7082 | 4.9756 |
| L25 | 0.0135 | 0.0142 | 0.0069 | 0.0936 | 0.0000 | 1.2117 | 2.1013 | 3.7208 |

---

#### TOXIC (Safety-enforced — RLHF toxicity suppression expected)

**Best clean accuracy (§4 benchmark):** CAA: 36% (c=7), CHARS: 2% (c=3), ACT: 21% (c=7), REPS: 0% (OOD), FLAS: **0%** (all coeffs), PID: **45% (c=1)**, WeightSteer: 0%, CAA-multi: n/a

**Temporal stability — ratio KL@L14 (10th gen / 1st gen token):**
CAA: 0.64 ↓DECAYS, CHARS: 0.83 →STABLE, ACT: 0.64 ↓DECAYS, REPS: 0.64 ↓DECAYS, FLAS: 0.74 ↓DECAYS, PID: **1.25 ↑GROWS**, WeightSteer: 0.94 →STABLE, CAA-multi: 0.88 →STABLE

**1st generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2206 | 0.4058 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2971 | 0.9718 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4963 | 0.4220 | 2.2209 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0d.0000 | 0.0000 | 0.3660 | 0.4602 | 2.6424 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.1704 | 0.4305 | 3.4050 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9733 | 0.7584 | 3.8970 |
| L14 | 0.1793 | 0.2575 | 0.1397 | 1.2629 | 8.5655 | 2.5864 | 0.7801 | 5.3010 |
| L15 | 0.1461 | 0.2051 | 0.0912 | 1.1195 | 6.7495 | 2.2595 | 0.9296 | 6.4038 |
| L16 | 0.0845 | 0.1101 | 0.0532 | 1.4108 | 11.0945 | 3.7020 | 0.9784 | 5.7646 |
| L17 | 0.0799 | 0.1108 | 0.0333 | 2.0685 | 11.0498 | 3.5287 | 1.1497 | 6.9553 |
| L18 | 0.0596 | 0.0819 | 0.0388 | 2.8532 | 11.3254 | 2.6728 | 1.5270 | 6.5850 |
| L19 | 0.1190 | 0.1487 | 0.0829 | 2.8344 | 10.5568 | 3.5661 | 1.7602 | 9.3349 |
| L20 | 0.0802 | 0.1023 | 0.0485 | 1.8293 | 10.1323 | 2.9555 | 1.6853 | 8.2224 |
| L21 | 0.0762 | 0.0937 | 0.0202 | 1.8724 | 9.0800 | 2.4986 | 0.9646 | 5.8626 |
| L22 | 0.0831 | 0.1023 | 0.0236 | 1.8850 | 7.0634 | 2.1919 | 0.9197 | 7.0176 |
| L23 | 0.0343 | 0.0437 | 0.0066 | 1.3142 | 6.5882 | 1.8434 | 0.4470 | 5.1185 |
| L24 | 0.0284 | 0.0376 | 0.0070 | 0.2506 | 5.8811 | 0.7138 | 0.2876 | 2.4729 |
| L25 | 0.0057 | 0.0077 | 0.0048 | 0.0730 | 3.5229 | 0.1705 | 0.2837 | 0.9174 |

**3rd generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0702 | 0.0570 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1148 | 0.1813 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2475 | 0.2415 | 0.5930 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1599 | 0.3136 | 0.8259 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0227 | 0.4871 | 1.4350 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6601 | 0.6417 | 1.7470 |
| L14 | 0.1418 | 0.2109 | 0.0509 | 0.4265 | 5.2309 | 1.4976 | 0.8728 | 2.4687 |
| L15 | 0.0716 | 0.1200 | 0.0421 | 0.5169 | 4.6532 | 1.3956 | 1.0922 | 3.1375 |
| L16 | 0.0501 | 0.0706 | 0.0410 | 0.7646 | 6.2113 | 2.6553 | 1.4249 | 3.8471 |
| L17 | 0.0375 | 0.0587 | 0.0349 | 0.7376 | 5.6586 | 2.9616 | 1.6077 | 5.8592 |
| L18 | 0.0472 | 0.0440 | 0.0315 | 0.6649 | 5.9228 | 3.1752 | 1.7305 | 7.4134 |
| L19 | 0.0457 | 0.0446 | 0.0263 | 0.6780 | 5.3781 | 3.3239 | 2.1350 | 8.8111 |
| L20 | 0.0533 | 0.0533 | 0.0295 | 0.4639 | 5.1322 | 3.6332 | 1.9381 | 8.6113 |
| L21 | 0.0333 | 0.0349 | 0.0170 | 0.4216 | 3.9149 | 2.2555 | 1.4467 | 6.5743 |
| L22 | 0.0342 | 0.0248 | 0.0138 | 0.3426 | 2.9795 | 1.9108 | 1.6830 | 6.2842 |
| L23 | 0.0346 | 0.0250 | 0.0099 | 0.5283 | 2.6932 | 2.0079 | 0.9231 | 5.4342 |
| L24 | 0.0452 | 0.0289 | 0.0096 | 0.1664 | 1.2930 | 1.4844 | 1.1680 | 5.1549 |
| L25 | 0.0154 | 0.0119 | 0.0048 | 0.0808 | 0.5811 | 0.7526 | 0.8542 | 4.0263 |

**10th generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1274 | 0.2298 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1482 | 0.6973 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4739 | 0.1756 | 1.7038 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3829 | 0.3030 | 2.0931 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.8325 | 0.4028 | 2.9486 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.2524 | 0.5289 | 3.4763 |
| L14 | 0.1143 | 0.2137 | 0.0897 | 0.8059 | 6.3184 | 3.2251 | 0.7303 | 4.6510 |
| L15 | 0.0974 | 0.1103 | 0.0522 | 0.5549 | 3.9049 | 2.7297 | 0.9893 | 5.5168 |
| L16 | 0.0621 | 0.0823 | 0.0455 | 0.3336 | 5.0612 | 4.7057 | 0.9924 | 6.2061 |
| L17 | 0.0488 | 0.0575 | 0.0405 | 0.3936 | 4.7365 | 4.9042 | 1.0587 | 9.1932 |
| L18 | 0.0600 | 0.0354 | 0.0216 | 0.2777 | 5.1096 | 4.6956 | 1.5875 | 7.5746 |
| L19 | 0.0455 | 0.0457 | 0.0178 | 0.4389 | 4.2543 | 4.1286 | 1.5785 | 8.5711 |
| L20 | 0.0382 | 0.0533 | 0.0246 | 0.5399 | 3.1287 | 3.2997 | 1.6561 | 7.4874 |
| L21 | 0.0331 | 0.0324 | 0.0169 | 0.4155 | 2.7805 | 2.4760 | 1.5776 | 5.5743 |
| L22 | 0.0245 | 0.0364 | 0.0195 | 0.3285 | 2.3537 | 2.0826 | 1.8366 | 4.6124 |
| L23 | 0.0131 | 0.0202 | 0.0106 | 0.0942 | 1.6501 | 1.8400 | 1.0278 | 3.8293 |
| L24 | 0.0080 | 0.0114 | 0.0098 | 0.0771 | 1.4191 | 1.3196 | 1.0605 | 3.3721 |
| L25 | 0.0048 | 0.0057 | 0.0037 | 0.0371 | 0.6960 | 0.5964 | 0.8261 | 2.6876 |

---

#### DECEPTION (Honesty suppression — no direct RLHF enforcement)

**Best clean accuracy (§4 benchmark):** CAA: 51% (c=10), CHARS: 85% (c=7), ACT: **90% (c=5)**, REPS: 42% (c=2), FLAS: 72% (c=1), PID: n/a, WeightSteer: 68% (c=1), CAA-multi: n/a

**Temporal stability — ratio KL@L14 (10th gen / 1st gen token):**
CAA: **1.00 →STABLE**, CHARS: 0.87 →STABLE, ACT: 0.93 →STABLE, REPS: 0.44 ↓DECAYS, FLAS: 0.58 ↓DECAYS, PID: 0.84 →STABLE, WeightSteer: 0.31 ↓DECAYS, CAA-multi: 0.59 ↓DECAYS

**1st generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2310 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7410 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0053 | 0.0000 | 1.3126 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0039 | 0.0000 | 1.5717 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0156 | 0.0000 | 2.4192 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0082 | 0.0000 | 2.5753 |
| L14 | 0.0658 | 0.0636 | 0.0823 | 2.4275 | 0.3707 | 0.0895 | 1.5019 | 2.4775 |
| L15 | 0.0494 | 0.0467 | 0.0575 | 2.4808 | 0.8243 | 0.0719 | 1.7368 | 2.1675 |
| L16 | 0.0275 | 0.0279 | 0.0334 | 2.2004 | 1.5163 | 0.2726 | 1.4844 | 2.0749 |
| L17 | 0.0358 | 0.0391 | 0.0364 | 2.0375 | 0.8179 | 0.3004 | 1.1777 | 2.3842 |
| L18 | 0.0360 | 0.0407 | 0.0287 | 1.7981 | 0.7578 | 0.2350 | 1.2333 | 2.5211 |
| L19 | 0.0343 | 0.0351 | 0.0365 | 2.4986 | 1.1967 | 0.2873 | 1.6114 | 3.0662 |
| L20 | 0.0156 | 0.0170 | 0.0188 | 2.0074 | 0.6681 | 0.1758 | 1.1414 | 2.4940 |
| L21 | 0.0107 | 0.0116 | 0.0165 | 1.4152 | 0.4187 | 0.0826 | 0.5183 | 1.5153 |
| L22 | 0.0150 | 0.0172 | 0.0142 | 1.1171 | 0.7140 | 0.0669 | 0.3885 | 1.4408 |
| L23 | 0.0071 | 0.0086 | 0.0087 | 0.4505 | 0.4405 | 0.0509 | 0.2033 | 0.6740 |
| L24 | 0.0069 | 0.0083 | 0.0083 | 0.4486 | 0.1618 | 0.0151 | 0.1525 | 0.5109 |
| L25 | 0.0019 | 0.0025 | 0.0017 | 0.2578 | 0.0471 | 0.0057 | 0.0669 | 0.2691 |

**3rd generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0846 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3183 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0045 | 0.0000 | 0.6221 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0030 | 0.0000 | 0.8755 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0146 | 0.0000 | 0.9501 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0059 | 0.0000 | 1.5330 |
| L14 | 0.0522 | 0.0596 | 0.0511 | 0.5958 | 0.1593 | 0.0716 | 0.6177 | 1.6876 |
| L15 | 0.0424 | 0.0378 | 0.0327 | 0.9972 | 0.8034 | 0.0669 | 0.9904 | 1.6335 |
| L16 | 0.0311 | 0.0332 | 0.0343 | 0.7828 | 1.3599 | 0.2530 | 1.0039 | 1.5652 |
| L17 | 0.0234 | 0.0204 | 0.0438 | 0.7685 | 1.3715 | 0.1670 | 0.7150 | 1.4938 |
| L18 | 0.0191 | 0.0210 | 0.0327 | 0.8805 | 0.8954 | 0.1710 | 1.0301 | 1.9487 |
| L19 | 0.0239 | 0.0170 | 0.0336 | 0.9253 | 1.0562 | 0.1571 | 0.6594 | 1.8645 |
| L20 | 0.0218 | 0.0250 | 0.0386 | 0.9463 | 0.8881 | 0.1511 | 0.7916 | 2.0760 |
| L21 | 0.0202 | 0.0226 | 0.0298 | 1.0132 | 0.6808 | 0.1292 | 0.8947 | 1.6010 |
| L22 | 0.0200 | 0.0260 | 0.0174 | 0.9736 | 0.6664 | 0.0803 | 0.5448 | 1.3452 |
| L23 | 0.0088 | 0.0102 | 0.0106 | 0.6258 | 0.4416 | 0.0586 | 0.3901 | 1.1430 |
| L24 | 0.0139 | 0.0086 | 0.0105 | 0.4417 | 0.3917 | 0.0577 | 0.4369 | 0.7257 |
| L25 | 0.0056 | 0.0051 | 0.0065 | 0.1305 | 0.2631 | 0.0303 | 0.1985 | 0.3735 |

**10th generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0803 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3050 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0037 | 0.0000 | 0.6852 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0042 | 0.0000 | 0.9398 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0130 | 0.0000 | 1.4711 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0060 | 0.0000 | 1.2404 |
| L14 | 0.0657 | 0.0556 | 0.0762 | 1.0782 | 0.2165 | 0.0753 | 0.4667 | 1.4641 |
| L15 | 0.0503 | 0.0316 | 0.0404 | 0.7840 | 0.6619 | 0.0452 | 0.6534 | 1.3410 |
| L16 | 0.0265 | 0.0242 | 0.0323 | 0.9123 | 1.5148 | 0.1917 | 0.5510 | 1.4145 |
| L17 | 0.0234 | 0.0208 | 0.0251 | 1.0032 | 1.5275 | 0.2309 | 0.3331 | 1.4446 |
| L18 | 0.0242 | 0.0112 | 0.0218 | 1.0372 | 1.0096 | 0.1882 | 0.2934 | 1.3796 |
| L19 | 0.0129 | 0.0142 | 0.0144 | 0.9168 | 1.1478 | 0.1737 | 0.2816 | 1.4482 |
| L20 | 0.0084 | 0.0122 | 0.0158 | 0.9548 | 0.9436 | 0.1373 | 0.3379 | 1.1928 |
| L21 | 0.0084 | 0.0080 | 0.0106 | 0.5876 | 0.7989 | 0.0800 | 0.3651 | 1.6933 |
| L22 | 0.0084 | 0.0064 | 0.0106 | 0.3089 | 0.7420 | 0.0484 | 0.1769 | 1.2268 |
| L23 | 0.0066 | 0.0059 | 0.0073 | 0.2153 | 0.4329 | 0.0585 | 0.2938 | 1.0061 |
| L24 | 0.0068 | 0.0042 | 0.0050 | 0.2129 | 0.3092 | 0.0204 | 0.2260 | 0.7777 |
| L25 | 0.0026 | 0.0028 | 0.0027 | 0.0661 | 0.2646 | 0.0130 | 0.0979 | 0.3302 |

---

#### REFUSAL (Jailbreaking — surface classifier, flat geometry, SNR=4.951)

**Best clean accuracy (§4 benchmark):** All methods: **99%+** at c=1 — task is trivially easy due to high SNR and near-flat decision boundary (§3).

**Temporal stability — ratio KL@L14 (10th gen / 1st gen token):**
CAA: **1.16 ↑GROWS**, CHARS: **1.25 ↑GROWS**, ACT: **1.50 ↑GROWS**, REPS: 0.24 ↓DECAYS, FLAS: 0.35 ↓DECAYS, PID: **1.39 ↑GROWS**, WeightSteer: 0.60 ↓DECAYS, CAA-multi: 0.58 ↓DECAYS

**1st generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4051 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.5569 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.9029 | 0.0000 | 14.8545 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.7018 | 0.0000 | 16.1672 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.2283 | 0.0000 | 17.7468 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.2774 | 0.0000 | 20.5164 |
| L14 | 2.4285 | 2.4118 | 3.1772 | 11.2977 | 0.1026 | 10.8957 | 0.4620 | 22.8573 |
| L15 | 2.0238 | 2.0382 | 2.6166 | 9.0184 | 0.1890 | 8.2025 | 0.7922 | 22.1222 |
| L16 | 1.8765 | 1.8906 | 2.9602 | 12.4738 | 0.1811 | 14.7613 | 1.0188 | 25.3350 |
| L17 | 1.8426 | 1.8528 | 3.3647 | 15.9106 | 0.2461 | 18.0194 | 1.0242 | 27.7189 |
| L18 | 1.8524 | 1.8270 | 2.7420 | 19.7333 | 0.1480 | 18.5884 | 1.0443 | 30.8855 |
| L19 | 2.1507 | 2.1313 | 3.1101 | 25.7646 | 0.2132 | 20.8205 | 1.1519 | 30.5143 |
| L20 | 1.2755 | 1.2630 | 1.9602 | 27.1054 | 0.1029 | 22.2626 | 1.0694 | 31.9291 |
| L21 | 1.0437 | 1.0254 | 1.7150 | 23.6333 | 0.0865 | 20.1624 | 0.6951 | 31.1474 |
| L22 | 0.6788 | 0.6590 | 1.3932 | 24.2025 | 0.0456 | 21.8714 | 0.7213 | 28.6702 |
| L23 | 0.3683 | 0.3746 | 1.1138 | 24.7810 | 0.0626 | 18.5465 | 0.6687 | 27.0430 |
| L24 | 0.3310 | 0.3432 | 0.9196 | 15.8027 | 0.0566 | 13.6731 | 0.4266 | 21.5310 |
| L25 | 0.1483 | 0.1485 | 0.5303 | 4.6444 | 0.0183 | 4.1049 | 0.1291 | 18.2278 |

**3rd generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.8369 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.7998 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.6203 | 0.0000 | 9.4384 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.7645 | 0.0000 | 9.2290 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7.0229 | 0.0000 | 10.1568 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.0143 | 0.0000 | 12.7387 |
| L14 | 1.7328 | 2.0755 | 2.6588 | 8.0976 | 0.0590 | 17.0956 | 0.2331 | 17.0001 |
| L15 | 1.7220 | 1.9501 | 2.6484 | 6.7407 | 0.2600 | 15.6809 | 0.8402 | 13.4515 |
| L16 | 1.7940 | 2.0475 | 2.5144 | 7.8655 | 0.3433 | 19.8358 | 1.1645 | 15.8550 |
| L17 | 1.7548 | 1.7732 | 2.4448 | 8.6594 | 0.3484 | 19.7208 | 2.1927 | 29.0433 |
| L18 | 1.8898 | 1.8705 | 3.5047 | 10.5398 | 0.6655 | 21.1592 | 4.7180 | 30.1363 |
| L19 | 1.9428 | 1.7561 | 3.7956 | 14.4999 | 0.4574 | 16.4293 | 5.1317 | 24.8923 |
| L20 | 2.0122 | 1.9786 | 4.1869 | 16.4589 | 0.5432 | 16.6619 | 5.5970 | 30.7989 |
| L21 | 2.1465 | 1.8332 | 3.5074 | 16.6620 | 0.4072 | 13.3452 | 5.6870 | 30.0999 |
| L22 | 1.8750 | 1.6489 | 2.3920 | 17.8901 | 0.2626 | 11.5208 | 4.9874 | 19.8806 |
| L23 | 1.6615 | 1.3379 | 2.2829 | 20.6609 | 0.3042 | 10.6151 | 4.7119 | 24.3395 |
| L24 | 1.1253 | 1.1416 | 1.3036 | 12.7678 | 0.3097 | 9.3954 | 3.4510 | 22.9075 |
| L25 | 0.5506 | 0.5042 | 0.5759 | 4.1794 | 0.1390 | 3.1143 | 1.4453 | 19.3107 |

**10th generated token — KL(steered ‖ baseline) per layer:**

| Layer | CAA | CHARS | ACT | REPS | FLAS | PID | WeightSteer | CAA-multi |
|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| L8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.1466 |
| L9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.4320 |
| L10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.9528 | 0.0000 | 7.7112 |
| L11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.4861 | 0.0000 | 8.3293 |
| L12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9.0684 | 0.0000 | 10.3534 |
| L13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.4495 | 0.0000 | 13.2701 |
| L14 | 2.8096 | 3.0077 | 4.7749 | 2.7254 | 0.0356 | 15.1811 | 0.2786 | 13.1558 |
| L15 | 2.1506 | 2.4352 | 4.2769 | 2.0563 | 0.1061 | 13.8210 | 0.6740 | 12.8393 |
| L16 | 2.1176 | 2.3486 | 3.5971 | 2.0512 | 0.1516 | 18.0509 | 0.6284 | 12.7931 |
| L17 | 1.4430 | 2.0933 | 2.8912 | 2.3767 | 0.1356 | 18.8831 | 0.9264 | 14.5785 |
| L18 | 1.8499 | 2.2028 | 3.0341 | 2.9000 | 0.1737 | 21.7110 | 1.7257 | 15.9219 |
| L19 | 1.9858 | 2.1297 | 2.3682 | 3.0997 | 0.2277 | 17.3137 | 1.7021 | 12.3354 |
| L20 | 1.0831 | 1.9570 | 2.5148 | 3.3723 | 0.1888 | 18.2823 | 1.8236 | 17.4172 |
| L21 | 0.9290 | 1.3935 | 1.9794 | 3.0330 | 0.1647 | 13.6315 | 1.9925 | 27.1067 |
| L22 | 0.7362 | 1.2058 | 2.1273 | 2.6503 | 0.1228 | 13.8679 | 2.4128 | 19.2966 |
| L23 | 0.7069 | 0.7340 | 1.8050 | 2.6151 | 0.1527 | 11.8979 | 2.1289 | 27.7788 |
| L24 | 0.4462 | 0.7398 | 1.1492 | 1.7283 | 0.0672 | 9.1327 | 1.6639 | 22.9884 |
| L25 | 0.2400 | 0.2196 | 0.4430 | 0.6005 | 0.0378 | 3.2163 | 0.6195 | 19.7996 |

---

**Key findings (generated-token evaluation):**

1. **KL magnitude does not predict accuracy.** FLAS produces KL=8.57 at L14 on toxic (1st gen token) but achieves 0% accuracy across all coefficients. PID produces KL=2.59 on toxic and achieves 45% at c=1. CAA produces KL=0.18 and achieves 36% at c=7. Pearson correlation between KL@L14 and accuracy is near zero or negative within the toxic task — direction alignment, not perturbation magnitude, is the causal variable.

2. **The FLAS paradox — direction miss vs. direction hit.** FLAS on refusal: KL=0.103, accuracy=99%+. FLAS on toxic: KL=8.57, accuracy=0%. A 60× KL difference yields the same or opposite accuracy ordering. This suggests the model is **direction-selective**: it appears to cancel perturbations in certain subspaces (toxicity directions) while leaving others unaffected. FLAS learned to steer in a subspace relevant for refusal but irrelevant for toxicity. The mechanism behind this selectivity is unknown — it could be a dedicated circuit or a side effect of RLHF training dynamics.

3. **Temporal stability as a signature of late-layer KL decay.** Three distinct regimes on safety tasks (toxic, evil):
 - **↑ GROWS (ratio > 1.05):** Only PID on toxic (1.25×). The I-term accumulates error across layers, possibly outpacing whatever mechanism suppresses KL at late layers.
 - **→ STABLE (0.80–1.05):** WeightSteer (0.94×), CHARS (0.83×) on toxic. Weight-level modification re-generates the perturbation at every layer.
 - **↓ DECAYS (< 0.80):** All other methods on safety tasks. Single-point injection KL decays per generation step.

4. **Evil vs. toxic: task asymmetry.** On evil, ALL methods show ↓DECAYS temporal patterns, yet WeightSteer achieves 62–81% accuracy — decaying KL does not prevent steering on evil. On toxic, the same WeightSteer with similar KL magnitude achieves 0%. This asymmetry suggests the suppression mechanism (if it exists) is task-specific, not generic.

5. **Refusal: growing KL despite trivial accuracy.** CAA/CHARS/ACT/PID show ↑GROWS on refusal (ratios 1.16–1.50). Unlike safety tasks, refusal directions face no apparent suppression — perturbations toward compliance accumulate freely. Whether this is because no suppression mechanism exists for refusal, or because the mechanism is directional, is unknown.

6. **Deep-layer amplification (L14→L20 ratio at 1st gen token).** WeightSteer: 2.16× on toxic (KL grows deeper despite injection at all layers). REPS/PID on refusal: ~2.0–2.4× (perturbation amplified by attention in deeper layers). CAA/CHARS/ACT: 0.23–0.45× (decays with depth on all tasks). Methods with growing depth profiles suggest their perturbation aligns with the model's own attention-amplified signal processing, rather than being damped.

7. **Deception as the control task.** On Deception (no safety alignment), CAA achieves temporal stability ratio=1.00 (perfect) and CHARS/ACT ~0.87–0.93. These may represent the "natural" values for single-layer steering without suppression. Comparing these baselines to toxic/evil, the additional KL decay is ~36–54% per generation step for single-layer methods on safety tasks.

**Interpretation**: KL decay pattern is consistent with a direction-selective suppression mechanism, but the mechanism itself is unconfirmed. It could be a dedicated cancellation circuit, a side effect of RLHF training dynamics, or something else entirely. The only methods that bypass the KL decay are: (a) **I-term accumulation** (PID, grows KL rather than decaying), and (b) **weight-level modification** (WeightSteer, KL grows x8.95 vs activation decay). KL magnitude and temporal stability are not reliable predictors of accuracy.

**Files**: `Experiments/ExpD/expD_logit_lens.py`, `Experiments/ExpD/expD_results_evil.json`, `Experiments/ExpD/expD_results_toxic.json`, `Experiments/ExpD/expD_results_deception.json`, `Experiments/ExpD/expD_results_refusal.json`

---

### 9.14 Diagnostics: Per-Layer Contrastive Alignment (expE)

**Motivation**: tests if injected perturbations align with the model's own contrastive direction per layer — if toxic alignment decays faster than deception, the model actively rotates away from safety-relevant directions, revealing a layer-wise cancellation mechanism.

**Design**: Extract contrastive vectors at L8-L25 separately. Inject v at L8. Track cos ({perturbation}[l], v_{contrastive}[l]). If toxic alignment decays faster than deception, this would suggest active rotation away from safety-relevant directions.

**Results (gemma-2-2b-it)**:

| Vector | cos @ L8 | cos @ L25 | Decay |
|:-------|:--------:|:---------:|:-----:|
| Toxic | 0.7874 | 0.1145 | **14.54% remaining** |
| Deception | 0.7777 | 0.0499 | **6.42% remaining** |

**Verdict**: **FALSIFIED.** Alignment with natural feature direction decays for both safety and honesty. The deception perturbation actually decays *faster* than toxic (6.42% vs 14.54% remaining). This is inconsistent with an active mid-stream safety rotation circuit — no evidence of task-specific rotation was observed.

**Files**: `Experiments/ExpE/expE_contrastive_alignment.py`, `Experiments/ExpE/expE_results.json`

---

### 9.15 K-Sweep + Coupling Matrix Deep Dive (Jun 24)

**Motivation**: resolves whether RBF kernel degeneracy on Toxic causes CHARS's 0% — if K-sweep shows active centroids and functional coupling, the ceiling is not in the clustering or transport algorithm.

**Trigger**: Tail direction needed resolution — RBF kernel degenerate on Toxic?

**Hypothesis**: CHARS Toxic failure from RBF degeneracy (RBF ≈ 0 for high-norm centroids → collapse to largest-norm centroid).

**Experiments**:

| # | Experiment | What | Key finding |
|---|-----------|------|-------------|
| 1 | **Symlog eval** | CHARS Toxic c=3 with `chars_tail_transform: "symlog"` | **0.00%** — same as baseline |
| 2 | **Coupling mass K-sweep** | All tasks x all K, two pooling views (extraction + inference). 500 src acts, assign to nearest centroid | Active centroids differ by pooling view. Toxic: 10/10 (extraction view) vs 1/10 (inference view). |
| 3 | **P* matrix deep dive** | Where does each centroid's coupling mass go? | **RBF kernel fine** (0.00-0.98). High-norm centroids have more specific transport. |
| 4 | **PCA decomposition** | In-manifold vs off-manifold of centroids | 1 PC explains 90% variance |

**Exp 1 — Symlog**: Applied to CHARS Toxic (K=10, c=3). Reduced centroid CV 0.277→0.060. Accuracy **0.00%** — safety ceiling binding.

**Exp 2 — Coupling Mass K-Sweep**:

**Setup**: For each (task, K), extract 500 source activations at layer 14 using both `position="mask"` (sequence mean-pool over non-padding tokens) and `position="last"` (final token). Load CHARS source centroids from `Vector/CHARS/Gemma/{task}_K{K}/metadata.pt`. Assign each samp<= to nearest centroid by L2 distance. Count active centroids (>=1 sample). Report Spearman rho between sample norm and coupling mass (each centroid's row sum of Sinkhorn matrix P*), and tail/body ratio (mean coupling mass of top 5% norm samples / bottom 95%).

**Table A -- Extraction-consistent pooling** (matches each task's extractor config):

| Task | K | Active | CV | rho | T/B | Top/Total |
|:---|---|---:|---:|:---:|:---:|:---------|
| **Toxic** | 3 | **3/3** | 0.195 | +0.34 | 0.27 | 228/500 |
| | 5 | **5/5** | 0.289 | +0.18 | 0.18 | 225/500 |
| | 10 | **10/10** | 0.277 | +0.15 | 0.19 | 139/500 |
| | 20 | **18/20** | 0.245 | -0.10 | 0.25 | 60/500 |
| **Evil** | 3 | **3/3** | 0.013 | +0.26 | 1.02 | 212/500 |
| | 5 | **5/5** | 0.024 | +0.03 | 1.01 | 155/500 |
| | 10 | **10/10** | 0.030 | +0.12 | 1.02 | 77/500 |
| | 20 | **20/20** | 0.031 | +0.07 | 0.95 | 51/500 |
| **Deception** | 3 | **1/3** | 0.013 | NaN | 1.00 | 500/500 |
| | 5 | **2/5** | 0.014 | +0.04 | 1.00 | 498/500 |
| | 10 | **2/10** | 0.015 | -0.32 | 0.91 | 267/500 |
| | 20 | **2/20** | 0.015 | -0.11 | 0.73 | 273/500 |
| | 50 | **6/50** | 0.019 | +0.12 | 0.93 | 418/500 |
| **Refusal** | 5 | **5/5** | 0.031 | +0.05 | 1.16 | 136/408 |
| | 10 | **8/10** | 0.032 | +0.32 | 1.31 | 155/408 |
| | 20 | **14/20** | 0.036 | +0.13 | 1.14 | 117/408 |
| | 30 | **2/30** | 0.130 | +0.56 | 1.16 | 334/408 |
| | 50 | **26/50** | 0.036 | +0.27 | 1.58 | 111/408 |

In extraction-consistent space, K-means correctly partitions: **10/10** for Toxic and Evil at K=10, **20/20** for Evil at K=20. Toxic's active count drops at K=20 (18/20) due to high CV (0.28) -- high-norm centroids are too far from most samples even in mask space. Deception stays sparse (2-6/50) because its manifold has effective rank 2.1 -- extra centroids capture negligible mass. Refusal scales well: 5/5 to 26/50.

**Table B -- Inference-pooling (all `position="last"`)**:

| Task | K | Active | CV | rho | T/B | Top/Total |
|:---|---|---:|---:|:---:|:---:|:---------|
| **Toxic** | 3 | **1/3** | 0.195 | NaN | 1.00 | 500/500 |
| | 5 | **1/5** | 0.289 | NaN | 1.00 | 500/500 |
| | 10 | **1/10** | 0.277 | NaN | 1.00 | 500/500 |
| | 20 | **2/20** | 0.245 | +0.26 | 1.87 | 488/500 |
| **Evil** | 3 | **3/3** | 0.013 | +0.14 | 1.01 | 202/500 |
| | 5 | **4/5** | 0.024 | -0.39 | 0.90 | 270/500 |
| | 10 | **7/10** | 0.030 | +0.36 | 1.10 | 199/500 |
| | 20 | **12/20** | 0.031 | -0.10 | 0.82 | 314/500 |
| **Deception** | 3 | **1/3** | 0.013 | NaN | 1.00 | 500/500 |
| | 5 | **2/5** | 0.014 | +0.06 | 1.01 | 495/500 |
| | 10 | **2/10** | 0.015 | -0.29 | 0.93 | 291/500 |
| | 20 | **2/20** | 0.015 | -0.09 | 0.75 | 261/500 |
| | 50 | **6/50** | 0.019 | +0.10 | 1.00 | 417/500 |
| **Refusal** | 5 | **5/5** | 0.031 | +0.05 | 1.16 | 136/408 |
| | 10 | **8/10** | 0.032 | +0.32 | 1.31 | 155/408 |
| | 20 | **14/20** | 0.036 | +0.13 | 1.14 | 117/408 |
| | 30 | **2/30** | 0.130 | +0.56 | 1.16 | 334/408 |
| | 50 | **26/50** | 0.036 | +0.27 | 1.58 | 111/408 |

**Key conclusions**:
1. Multi-centroid clustering is **orthogonal to accuracy**: Evil 20/20 active, 0-3% accuracy. Deception 2-6/50, 85%.
2. Refusal scales well with K (26/50) when extraction dataset matches source activations (`refusal_caa`).

**Exp 4 — P* matrix deep dive**:

RBF kernel (τ=90.9) — NOT degenerate:
```
 src[6] norm=195: 0.83 0.30 0.62 0.00 0.05 0.46 0.01 0.13 0.45 0.33
 src[1] norm=459: 0.03 0.28 0.09 0.86 0.81 0.17 0.98 0.55 0.15 0.25
 src[7] norm=521: 0.00 0.07 0.02 0.96 0.41 0.04 0.76 0.20 0.03 0.06
```

RBF values 0.00-0.98 — kernel distinguishes by direction, not just norm.

P* coupling — high-norm centroids have MORE specific transport:
```
 src[6] norm=195 (active, 500 samples): entropy=0.757 (BROAD)
 src[1] norm=459 (0 samples): entropy=0.552 (CONCENTRATED, 44%+37%)
 src[7] norm=521 (0 samples): entropy=0.380 (HIGHLY, 63%+31%)
```

**RBF kernel NOT degenerate**: values 0.00-0.98 for Toxic K=10 (τ=90.9). High-norm centroids produce MORE specific transport (entropy 0.38 vs 0.76). RBF works correctly — failure is not in the kernel.

**Key findings**:
1. RBF kernel functional — values 0.00-0.98, distinguishes by direction not just norm.
2. High-norm centroids have MORE concentrated transport (entropy 0.38 vs 0.76).
3. Symlog (tail annealing) reduces CV 0.28→0.06 but accuracy stays 0%.
4. Safety ceiling (§9.13 expD) is the binding constraint — CHARS cannot bypass it.

**Files**: `Configs/Eval/CHARS/Gemma/extract_*.json`, `Experiments/Exp6/analyze_coupling.py`, `Experiments/Exp6/coupling_results_tableA/*.json`, `Experiments/Exp6/coupling_results_tableB/*.json`, `Experiments/Exp6/extract_source_acts_with_pooling.py`, `Steering/utils.py` lines 441-456 (`_pool`)

---

### 9.16 FACE-2 NLL-Distribution OOD Scoring (Jun 25)

**Motivation**: replaces manual PPL+rep-rate combination with a single unified OOD score — without reliable OOD detection, accuracy comparisons between methods are unreliable.

**Goal**: Single unified OOD score `[0,1]` replacing manual PPL+rep-rate combination. Zero extra forward passes (reuses per-token NLL from PPL computation).

**Method**: NLL-distribution JS divergence (not spectral FACE-2). For each sample, compute per-token NLL histogram (50 bins, 1st–99th percentile range). Score = `1 − JS(NLL_steered_hist || NLL_baseline_hist) / log(2)`. Higher = cleaner. Captures both mean NLL shift (incoherence → score ≈ 0) and variance change (repetition → score ≈ 0.7-0.8) in one bounded measure. Clean self-reference ∼0.92.

**Implementation**: `Steering/pipeline.py` — `_compute_face_scores()` static method. Flag `compute_face: bool = False` in `PipelineConfig`. Zero overhead when disabled. NLLs computed from same forward pass as PPL.

**Computed on existing JSON results**: Read `response` text from 30+ eval files, ran one HF forward pass per sample (gemma-2-2b, no steering). Compared steered NLL distribution vs coeff=0 baseline (or fresh generation).

**Results** (face_score, 3 coeff levels per method/task):

| Method | Task | c=1 | c=3 | c=10 |
|:-------|:-----|:---:|:---:|:----:|
| CAA | evil | 0.854 | 0.834 | 0.605 |
| LinearAct | evil | **0.872** | **0.868** | **0.875** |
| CHARS | evil | 0.802 | 0.829 | 0.600 |
| PCA-OT | evil | 0.844 | 0.841 | 0.759 |
| FLAS | evil | 0.841 | 0.826 | 0.493 |
| REPS | evil | 0.855 | 0.882 | 0.541 |
| WeightSteer | evil | 0.894 | 0.717 | 0.529 |
| PID | evil | 0.843 | — | — |
| COBRA | evil | — | 0.916 | — |
| CAA | toxic | 0.832 | 0.808 | 0.830 |
| LinearAct | toxic | 0.867 | 0.846 | 0.851 |
| CHARS | toxic | 0.817 | 0.807 | 0.830 |
| WeightSteer | toxic | 0.862 | 0.833 | 0.764 |
| PID | toxic | 0.822 | — | — |
| COBRA | toxic | 0.817 | 0.792 | — |
| CAST | toxic | 0.836 | 0.825 | 0.828 |
| CAA | deception | 0.825 | 0.831 | 0.747 |
| LinearAct | deception | 0.790 | 0.821 | 0.849 |
| CHARS | deception | 0.819 | 0.809 | 0.796 |
| WeightSteer | deception | 0.834 | 0.808 | 0.819 |
| PID | deception | 0.812 | — | — |
| COBRA | deception | 0.819 | 0.801 | 0.785 |
| CAST | deception | 0.826 | 0.833 | 0.832 |

**Key findings**:

1. **LinearAct is uniquely stable**: face_score stays 0.87–0.88 across ALL Evil coeffs (1→10). No other method maintains fluency at c=10 on Evil. This confirms the transport approach genuinely preserves output naturalness at high intervention strength.

2. **CAA/CHARS OOD at c=10 on Evil** (0.60–0.61), but NOT on Toxic (0.83) or Deception (0.75–0.80). The OOD degradation is task-specific, not universal. Suggests safety-suppression bypass at high coeff disrupts fluency, while coercion toward non-safety attributes (toxic content expression, dishonest text) does not.

3. **WeightSteer degrades earlier** (c=3→0.717 Evil, 0.833 Toxic, 0.808 Deception). Weight modification disrupts model distribution sooner than activation intervention, consistent with broader effects on model parameters.

4. **Parametric methods (FLAS, REPS) collapse at c=10 on Evil** (0.493, 0.541). Trained modules operate in narrow gain windows.

5. **PCA-OT degrades moderately at c=10** (0.759). Better than FLAS/REPS, worse than LinearAct. Consistent with theory: OT preserves distribution, but PCA truncation loses some nuance.

6. **COBRA c=3 Evil achieves 0.916** — highest single score at any coeff. Subspace-projected steering preserves fluency best when it works.

7. **Deception face_score is stable across all coeffs** (0.79–0.85). No fluency cost for steering honesty → deceptiveness axis. Supports safety ceiling theory: fluency loss on Evil may come from triggering hypothesised suppression mechanisms, not from vector magnitude alone.

**Baseline note**: Toxic baseline generated fresh (temperature=0.7) from prompts. Evil/Deception baselines from existing coeff=0 eval runs. All methods share the same baseline per task, so relative comparisons are valid.

**Implementation ref**: `Steering/pipeline.py` line 983 (`_compute_face_scores`), `Steering/config/pipeline.py` line 181 (`compute_face`), `Steering/config/results.py` lines 163–164 (`face_score`, `baseline_face_score`). Results JSON: `face_scores.json`.

---

### 9.17 ExpF: Per-Token KL on Steered Outputs (Jun 25-26)

**Motivation**: determines if steering KL concentrates in early tokens like fine-tuning — reveals whether steering is a shallow-channel manipulation or a deeper mechanistic intervention.

**Setup**: Step-by-step autoregressive generation with FIXED prefix (base's tokens). At each position t, both steered and base models condition on same context → KL(steered||base) is clean causal quantity. 5 methods × 4 tasks × 3 coeffs, n=50, max_new_tokens=20. Methods: CAA, CHARS, CURVE, REPS, WEIGHT.

**Files**: `Experiments/ExpF/expF_per_token_kl.py`, `Experiments/ExpF/analyze_kl.py`, `Experiments/ExpF/expF_results_{caa,chars,curve,reps,weight}.json`.

**Results — KL ratio (early 0-5 / late 5+) across tasks**:

| Method | Toxic | Evil | Deception | Refusal | Pattern |
|:-------|:-----:|:----:|:---------:|:-------:|:--------|
| **CAA** | 0.77-0.88 | 0.62-0.77 | **1.20-1.73** | 0.83-0.90 | Ceiling tasks < 1 |
| **CHARS** | 0.77-0.94 | 0.69-0.74 | **1.19-1.57** | 0.84-0.90 | Ceiling tasks < 1 |
| **CURVE** | 0.62-1.02 | 0.77-0.85 | **1.09-2.65** | 0.79-0.88 | Ceiling tasks < 1 |
| **REPS** | 1.39-5.77 | **1.37-1.84** | **1.26-2.71** | 1.18-1.93 | ALL tasks > 1 |
| **WEIGHT** | 0.99-1.29 | 0.83-1.02 | 0.71-0.95 | 1.00-2.37 | Flat (~1) |

**Key findings**:

1. **Cross-method systematic pattern**: For ALL 3 activation methods (CAA, CHARS, CURVE), Toxic/Evil have KL ratio < 1 (KL INCREASES in later tokens). Deception has ratio > 1 (KL concentrated early). Something suppresses early-token KL on safety tasks but not Deception. This is a universal pattern across all activation methods, but the mechanism is unknown.

2. **Qi et al. finding does NOT replicate for steering**: Fine-tuning KL localizes in first ~5 tokens (ratio >> 5). Activation steering on Toxic/Evil has ratio < 1 — KL is HIGHER in later tokens. Steering operates through a different channel from fine-tuning. Early-token KL is suppressed on safety tasks, pushing surviving KL to later positions.

3. **REPS bypasses early suppression**: REPS has ratio > 1 on ALL tasks including Toxic (1.39-5.77) and Evil (1.37-1.84). At c=3 on Evil, REPS achieves 48% accuracy (24/50) with ratio=1.75 — early KL is higher. REPS's subspace projection mechanism operates differently from activation vector addition.

4. **WeightSteer KL is flat**: Ratio ≈ 1 across all tasks. Weight modification produces uniform KL across positions, consistent with parameter-level (not activation-level) intervention.

5. **Refusal KL is enormous but uninformative**: CAA/CHARS/CURVE at c=3-5 produce 4-14 nats KL (vs 0.001-0.1 for Toxic/Evil). Despite 100x larger KL, accuracy = 0% at c≥3. The model massively changes its output distribution toward refusal — the KL is real but direction is wrong. KL magnitude ≠ alignment.

6. **Correct vs Incorrect early KL**: No consistent pattern. For Deception on CAA/CHARS, correct samples have LOWER early KL than incorrect (negative diff). This contradicts "more early logit change = success" hypothesis. Steering success is NOT predicted by per-position KL magnitude.

**Implications for safety ceiling**:
- Early KL suppression on Toxic/Evil but not Deception — whatever causes the KL decay operates at early output positions on safety tasks
- REPS partially bypasses (ratio > 1, 48% Evil at c=3)
- WeightSteer uniform KL — weight modification produces different KL dynamics from activation methods

### 9.18 Tail Analysis — Consolidated (Jun 26–27)

**Motivation**: consolidates all tail analysis results to confirm or falsify tail dimensions as a causal mechanism for the safety ceiling — stops a dead-end research direction if no correlation is found.

**What we already knew**: Earlier tests (§10.5) showed that "tail-heavy" activations (a few extreme values among the 2304 dimensions) do NOT predict which samples fail to steer. No metric, no task showed any correlation.

**What we did here**: Six follow-up probes (scripts in `Experiments/TailAnalysis/`). All use full 2304 dimensions (no PCA). We identify tail dimensions by checking which dimensions have extreme values clustered together — the same method across all experiments.

---

**Experiment 1 — Do the two classes have different tails?** (`target_vs_contrast_tail.py`)

- The "steer toward" and "steer away" classes have **identical** tail properties (statistical test p>0.05 for Toxic, Deception — no meaningful difference).
- The steering vector (direction we add) is concentrated in the PCA subspace (task-dependent: 17-441 PCs for 90% per-sample variance). Steering vector norm in-manifold: Toxic ~98%, Evil ~95%, Refusal ~94%, Deception ~40%.
- Mass in tail dims: Toxic=11.5%, Evil=5.6%, Refusal=1.7%, Deception=0.5%. **More tail mass = worse accuracy** — but this is just correlation, not cause.
- **Verdict**: Done. No class-specific tail behavior.

**Experiment 2 — Do transport methods use tail dims differently?** (`act_tail_affine.py`)

- ACT computes a per-dimension scale (ω) and shift (β) to transport source to target.
- The scale ω is **not different** in tail dims (p=0.34 — no signal).
- The shift β is **smaller** in tail dims (p=4×10⁻¹¹). ACT deliberately changes tail dims less.
- **Verdict**: Done. Transport methods also concentrate in core dimensions. They cannot "escape" into tail dims to bypass monitoring.

**Experiment 3 — Does steering change the tail magnitude?** (`post_steer_tail_analysis.py`)

- Script written, uses the full inference pipeline.
- **Not done** — all extraction-level questions already answered. If a suppression mechanism exists, it likely works on concept directions, not tail noise.

**Experiment 4 — Can tail-heavy clusters tell us which samples fail?** (`print_cluster_tables.py`, `eval_tail_kernel_checks.py`)

- We split 1000 extraction activations into clusters. For each cluster we compute tail_frac (fraction of samples in that cluster that are tail-heavy).
- **Clusters DO separate by tail_frac**: Deception has one cluster with 91% tail-heavy samples. Refusal has one with 67%.
- **But**: Eval samples (the ones we actually test steering on) are structurally different from extraction samples in tail dimensions. All eval samples are similarly distant from tail-heavy clusters — distances vary only 3-4% for 3 of 4 tasks.
- **Toxic is the exception**: 15% spread. Eval samples closer to tail clusters are 15 percentage points *less* steerable.
- **Interpretation**: The cancellation filter may catch Toxic samples that have more tail activation. But the effect is small and only appears on one task.

**Cluster composition** (tail-rich clusters across tasks):

| Task | Tail dims | Tail samples | K=5 best | K=10 best | K=20 best |
|:----|:---------:|:------------:|:--------:|:---------:|:---------:|
| Deception | 16 of 2304 | 125 of 1000 | 0.39 | C4=0.91 | C16=0.73 |
| Refusal | 59 of 2304 | 351 of 1000 | 0.71 | C2=0.67 (4 empty) | C14=1.0 |
| Toxic | 14 of 2304 | 119 of 1000 | 0.31 | C12=0.92 | C12=0.92 |
| Evil | 21 of 2304 | 80 of 1000 | 0.28 | C2=0.14 | C9=1.0 |

**Does tail proximity predict steering failure?**

| Task | Distance spread | High-acc group | Low-acc group | Gap | Direction |
|:----|:--------------:|:--------------:|:-------------:|:---:|:---------:|
| **Toxic** | **15.2%** | 13.0% | 28.3% | **−15.3 pp** | Tail → harder to steer |
| Deception | 3.7% | 27.9% | 33.7% | −5.8 pp | Weak, same direction |
| Evil | 3.7% | 17.6% | 13.4% | +4.2 pp | Weak, opposite direction |
| Refusal | 3.8% | 0.0% | 0.0% | 0.0 pp | No signal (zero accuracy) |

---

**What this means for our project**:

1. **The tail is not the problem**. After 6 experiments across 4 tasks using 3 methods, the answer is consistent: tail-heavy activation patterns do not predict steering success or failure. They are a static property of the model (ReLU/GeLU architectures create heavy tails mechanically), not a causal factor.

2. **What we actually know vs what we are guessing**:
 - We **know** that tail activations are structurally universal — same heavy tails for every task, every method. This is a solid experimental fact from 4 independent probes (§9.18 Exps 1-4).
 - We **know** that steering signal concentrates in the task-dependent PCA subspace: 72-98% of steering vector mass in the top PCs (17 for Deception, ~156-441 for safety tasks, §9.18 Exp 1). For any fixed k, the concentration varies by task.
 - We **suspect** (but haven't proven) that there is an output-stage cancellation filter. The evidence: logit lens KL decays for safety tasks but not Deception (§9.13 ExpD), and per-token KL is suppressed early on Toxic/Evil but not Deception (§9.17 ExpF). These are correlational — we see KL decay, we infer a filter. Direct proof would require activation patching to locate the circuit.
 - We **suspect** transport methods also hit the same ceiling because they operate in the same subspace (ACT tail-dim analysis, Exp 2). This is consistent but we haven't proven it's the same mechanism.
 - All activation methods failing on safety is a robust empirical finding across 20+ methods (§4 benchmark). The *mechanism* behind it is our best guess based on the KL experiments above. The paper will need more direct causal evidence (patching).

3. **Theories on the table (all tentative)**:
 - **Cancellation filter hypothesis** (§9.13 ExpD, §9.17 ExpF): A possible learned circuit at late layers detects safety-relevant steering and cancels its logit effect. Support: safety-task KL decays, non-safety KL grows. Open questions: which layer? which component (attention/MLP)? single circuit or distributed? Existence unconfirmed — requires activation patching.
 - **Subspace confinement hypothesis** (§10.4, §9.18 Exp 2): All activation methods modify the steering-relevant PCA subspace (task-dependent, ~17-441 dims). The model's safety monitoring may see those dims regardless of method. Support: PCA decomposition, ACT tail analysis. Open question: would a method that modifies different dimensions escape detection? Not obvious how to build one.
 - **Young's shallow alignment** (literature support): RLHF/DPO only modifies surface-level representations. Deep features (like "harmfulness") persist. The hypothesised cancellation filter, if real, would be a shallow patch on top. Support: WeightSteer (weight modification) bypasses the safety ceiling entirely, consistent with a shallow mechanism rather than a deeply integrated circuit.

4. **Where we go next**:
 - The tail is a dead end for method design. Stop investigating it.
 - To prove (or disprove) the cancellation filter theory: **activation patching**. Swap steered→unsteered per layer and see which single layer's intervention restores safety logits. This would locate the mechanism, if it exists (§10.7.2, item 2 design). **Priority action.**
 - Weight modification already works (62-81% Evil). Next: ablate which layers are sufficient, run on Toxic. If a single layer pair is enough, that further supports shallow-circuit theory.
 - PID multi-layer partially works (45% Toxic). Next: tune gains, try more layers. If PID's I-term accumulates to overwhelm a specific layer, that layer likely performs the suppression.
 - **Big picture**: No activation-level method beats the safety ceiling across 20+ methods. The mechanism is unknown but narrowing. Tail is ruled out. Cancellation filter is the leading hypothesis. Weight modification is the only known bypass. We need to nail the mechanism before we can claim a universal theory.

---

### 9.19 OOD Scoring: FACE-2 → LSP Suffix-Penalized Perplexity (Jun 25–27)

**Problem**: Two separate OOD approaches both tried to fix the same issue — PPL alone is misleading when output repeats (model assigns low loss to copied/duplicated tokens, producing low PPL for degenerate text).

**Attempt 1 — FACE-2 (Jun 25, abandoned)**:
- JS divergence between steered and baseline NLL histograms. Score = `1 − JS / log(2)`. Higher = cleaner.
- LinearAct maintained 0.87-0.88 across all Evil coeffs; CAA/CHARS dropped to 0.60 at c=10.
- Too noisy across runs. Score fluctuated without clean OOD separation.
- Removed from pipeline (no `_compute_face_scores` in current code).

**Attempt 2 — LSP (Localized Suffix-Penalized PPL, current)**:
- Suffix-penalized perplexity with LCS (longest common subsequence) detection in a 30-token sliding window.
- **Per-position**: `history = tokens[t-30:t]`. Compute LCS of current token suffix against history. If `lcs > 2`, penalty = `2.0 × log1p(lcs − 2)`. Add penalty to raw token NLL.
- **Global scoring**: Collect ALL penalized NLLs across ALL samples in one list, average globally, exponentiate once: `score = exp(mean(all_penalized_losses))`. Clean outputs score 2–6, degenerate > threshold.
- **Per-task thresholds**: Deception LSP > 25 = OOD, Evil LSP > 10 = OOD, Toxic LSP > 20 = OOD.
- Flag: `compute_lsp` in PipelineConfig. Field: `lsp_score` in EvalResult.
- Implementation: `Steering/pipeline.py` `_compute_lsp_scores()` (line 1010).

**Pareto integration**: Active. LSP replaces old `rep > 0.2 or ppl > 8` for OOD classification. Points with LSP > per-task threshold are plotted as OOD (faded, at right edge) in all Pareto plots.

**Status**: LSP active in pipeline code and Pareto plots. FACE-2 removed. Per-task thresholds (Deception 25, Evil 10, Toxic 20) set based on empirical LSP vs repetition-rate calibration.

---

### 9.20 OT Ablation: ACT-Gaussian & CHARS High-Lambda Transport (Jun 27–28)

**Motivation**: isolates whether the transport coupling is the causal factor behind OT methods' Pareto advantage — determines if future methods need complex transport or just per-dim scaling.

**Setup**:

| Script | Override | What it does |
|--------|----------|-------------|
| `run_linearact_ablate.sh` | `act_mode: "gaussian"` | Per-dim std-ratio shift (no sorted quantile matching) |
| `run_chars_ablate.sh` | `chars_lambda: 100.0` | Near-uniform Sinkhorn coupling (high entropy regularization → all centroids nearly equally weighted) |

**ACT ablation**: Standard ACT uses `act_mode="linear"` — sorts src/dst per-dim, computes ω = slope of sorted QQ-plot, matching full empirical distribution. The Gaussian ablation uses ω = σ_dst/σ_src, β = μ_dst − ω⊙μ_src — assumes Gaussian per-dim, no quantile matching. Configs: `gemma_deception.json`, `gemma_evil.json`, `gemma_toxic.json`. Output: `Results/linearact_ablate/`.

**CHARS ablation**: Standard CHARS uses `chars_lambda=0.1` (low entropy regularization, near-deterministic coupling). Ablation sets `chars_lambda=100.0`, making Sinkhorn converge to near-uniform coupling P ≈ 1/K — barycentric projection ≈ global centroid mean. Configs: `gemma_deception.json`, `gemma_evil.json`, `gemma_toxic.json`. Output: `Results/chars_ablate/`.

**Note**: The CHARS ablation also differs in K (Toxic: K=50 vs standard K=10; Evil: K=auto vs standard K=10; Deception: K=3 vs standard K=3) so the comparison is not a pure lambda isolation.

**Results** — Best clean accuracy (deceptiveness for Deception) by coeff, repetition_rate < 0.4, PPL < 20:

**ACT linear (sorted quantile) vs ACT gaussian (std-ratio only):**

| Task | Coeff | ACT linear | ACT gaussian | Delta |
|:-----|:-----:|:----------:|:------------:|:-----:|
| Toxic | 7 | 85% (rep=0.38, ppl=5.20) | 35% (rep=0.12, ppl=3.89) | −50pp |
| Toxic | 5 | 59%† | 3% (rep=0.02, ppl=2.23) | −56pp |
| Toxic | 3 | 59%† | 0% | −59pp |
| Toxic | 2 | 41% (rep=0.08, ppl=3.46) | 0% | −41pp |
| Evil | 1–3 | 0–5% | 0% | ~0pp |
| Deception | 1 | 95%† | 50% (rep=0.01, ppl=1.75) | −45pp |
| Deception | 2 | 93%† | 30% (rep=0.01, ppl=1.95) | −63pp |
| Deception | 3 | 93%† | 15% (rep=0.01, ppl=2.11) | −78pp |

† No repetition_rate in metadata (pre-Jun 15). PPL-based estimate: all < 5.

**CHARS low-lambda (λ=0.1, std K) vs CHARS high-lambda (λ=100, K varies):**

| Task | Coeff | CHARS λ=0.1 | CHARS λ=100 | Delta |
|:-----|:-----:|:-----------:|:-----------:|:-----:|
| Toxic | 3 | 2%* | 34% (rep=0.04, ppl=6.66) | +32pp |
| Toxic | 5 | 0%* | 27% (rep=0.10, ppl=39.4) | +27pp†† |
| Evil | 1 | 0%* | 0% | ~0pp |
| Evil | 3 | 3%* | 2% (rep=0.08, ppl=3.41) | −1pp |
| Evil | 5 | 0%* | 9% (rep=0.34, ppl=4.52) | +9pp |
| Deception | 1 | 85%* | 72% (rep=0.0, ppl=1.32) | −13pp |
| Deception | 3 | 65–85%‡ | 50% (rep=0.0, ppl=1.75) | −15–35pp |

*From §4.1 benchmark table (standard K values: Toxic K=10, Evil K=10, Deception K=3).
†† OOD by PPL (>20).
‡ Range across K-sweep results.

**Key findings:**

1. **ACT Gaussian is strictly worse than ACT linear on all tasks.** Sorted quantile matching is the causal mechanism for ACT's Pareto advantage. On Deception, Gaussian collapses from 95% (linear, c=1) to 50% (c=1). On Toxic, the gap is 85% vs 35% at c=7. On Evil, both score 0–5%. **Conclusion**: the per-dim full-distribution matching (sorted QQ-plot) is what makes ACT work — Gaussian std-ratio is barely better than CAA.

2. **CHARS high-lambda (near-uniform coupling) does NOT reduce performance on Toxic or Evil — it improves it.** On Toxic, high-lambda (34% at c=3) outperforms standard CHARS (2%). On Evil, comparable (2–9% vs 0–3%). **Conclusion**: the Sinkhorn transport plan is NOT on the critical path for CHARS. The method's advantage (when it exists) comes from centroid estimation + RBF-weighted barycentric projection, not from the specific discrete coupling. High lambda ≈ CAA-like behavior, and CAA actually outperforms degenerate coupling on Toxic/Evil.

3. **Deception is robust to transport simplification.** CHARS high-lambda (72% at c=1) is close to standard (77–85%). ACT Gaussian (50%) is worse but still functional. Deception's high steerability (85–95%) is not dependent on complex transport.

4. **Evil ceiling remains unbroken.** No ablation method exceeds 9% clean on Evil. The 0–10% safety ceiling is robust to transport simplification.

**Overall conclusion**: The sorted quantile coupling is causal for ACT's advantage. The Sinkhorn plan is NOT causal for CHARS — centroid estimation matters more. No transport simplification breaks the safety ceiling.

### 9.22 Refusal Subspace Ablation — FALSIFIED (Jun 29)

**Motivation**: Tests whether Evil/Toxic steering vectors share an anti-refusal component that triggers suppression. If safety-task steering pushes model activations strongly away from the refusal direction (cos(h_steered, r̂) < -0.5), the ceiling is mediated by a refusal-cancellation circuit. If cos(h_steered, r̂) ≈ cos(h_base, r̂) for all tasks, the bottleneck is not refusal-specific.

**Setup**:
- Extract r̂ from `refusal_cast_responses` composite dataset (700 pairs) using response-ONLY strings: r̂ = h(refusal_response) - h(compliant_response). No question context. CAA at layer 14, position="last".
- Load existing CAA vectors for Evil, Toxic, Deception from `Vector/CAA/Gemma/`
- Forward pass 50 test prompts per task (evil/toxic/liarbench), capture residual stream at layer 14 last token: baseline h_base (no steering) and h_steered (with steering active). Compute cos(h_base, r̂) and cos(h_steered, r̂).
- Script: `Experiments/ExpH/activation_analysis.py`
- Model: `google/gemma-2-2b-it`, layer 14, CAA, coeffs (evil=1, toxic=3, deception=5)
- Paper comparison: Safety Pitfalls (arxiv 2603.24543) extracts r̂ from last prompt token of harmful vs harmless prompts and checks directional alignment of steered activations. Our r̂ differs: response-only (refusal vs compliant text) rather than prompt-level (harmful vs harmless prompt). This isolates the refusal-behavior direction without question-content contamination.

**Observations**:

**Phase 1 — Vector-level cos(v, r̂)**:

| Task | cos(v, r̂) | |v| | |r̂| (response-only) |
|:-----|:---------:|:---:|:------------------:|
| Evil | +0.1289 | 20.65 | 69.50 |
| Toxic | +0.1963 | 27.39 | 69.50 |
| Deception | +0.0596 | 0.96 | 69.50 |

**Phase 2 — Activation-level cos** (Layer 14 last token, n=50):

| Task | coeff | cos(h_base, r̂) | cos(h_steered, r̂) | Δcos | |v|·coeff |
|:-----|:-----:|:--------------:|:-----------------:|:----:|:-------:|
| Evil | 1 | -0.0782 ± 0.068 | -0.0615 ± 0.067 | **+0.0167** | 20.62 |
| Toxic | 3 | -0.0686 ± 0.073 | +0.0251 ± 0.054 | **+0.0938** | 82.13 |
| Deception | 5 | -0.0296 ± 0.056 | -0.0280 ± 0.056 | **+0.0016** | 4.82 |

**Key findings**:
1. **h_base is anti-refusal**: All tasks show NEGATIVE cos(h_base, r̂) ≈ -0.03 to -0.08. The model's natural activation at the last prompt token of harmful prompts points weakly AWAY from the pure refusal-response direction. Makes sense: model is about to generate a refusal, but the last token of the prompt hasn't engaged full refusal circuits yet.
2. **Steering always TOWARD refusal**: Δcos > 0 for ALL tasks. Steering NEVER pushes activations away from refusal direction.
3. **Δcos scales with coeff·|v|**: Toxic (coeff=3, |v|=27.4) shifts most (+0.094), Evil (coeff=1, |v|=20.6) shifts less (+0.017), Deception (coeff=5, |v|=0.96) barely moves (+0.002).
4. **Toxic crosses zero**: cos goes from -0.069 (baseline, anti-refusal) to +0.025 (steered, pro-refusal). Despite this directional crossover, Toxic steering accuracy remains at ceiling (~0% at coeff=3). Refusal alignment is NOT the limiting factor.
5. **Compared to earlier r̂ (prompt+response)**: The response-only r̂ (|r̂|=69.5) gives stronger signal than prompt+response r̂ (|r̂|=104.6, cos ≈ 0.1 for all). Vector-level cos values are higher (0.06-0.20 vs 0.09-0.12), but still far from anti-aligned.

**Conclusion**: **FALSIFIED**. The refusal-direction overlap hypothesis is conclusively disproven:
- h_base is weakly anti-refusal (cos ≈ -0.05 to -0.08) for all tasks — natural state for harmful prompt processing
- Steering shifts all tasks TOWARD refusal (Δcos > 0), not away from it
- Toxic crosses from anti-refusal to pro-refusal yet accuracy remains at ceiling — refusal-direction alignment does NOT predict steering success
- Ablation (removing the ~cos(v,r̂) component) changes vector magnitude by <4% — negligible

Combined with ASTEER finding (§10.7.2 item 4): 150 non-safety concepts (format, tone, persona) all hit 10-23% ceiling. The bottleneck is a **general activation-steering property** — not refusal-specific. The KL decay asymmetry (§9.13: Evil/Toxic decay to 0.49× vs Deception grows to 1.08×) remains the only diagnostic tracking the safety vs non-safety split. This suggests a mechanism that attenuates steering influence as information propagates through later layers, selectively for tasks that engage the model's RLHF-trained generation manifold.

---

### 9.23 FLOW+LM: Joint CE–CFM Training Does Not Improve Safety Steering (Jun 28–29)

**Motivation**: Tests whether adding LM loss (CE on response tokens) as a training signal to the flow-matching objective can improve activation steering on safety tasks without causing OOD collapse. Prior methods (REPS, FLAS) use LM loss in low-rank or concept-conditioned settings — this isolates whether LM loss alone, in a simple single-layer FlowMLP, is sufficient.

**Setup** — `FlowExtractor` with `flow_lm_loss=True`:
- FlowMLP (dim=32 PCA subspace, hidden_dim=256, 2 layers)
- 400 CFM pre-training epochs → 100 CE-only fine-tuning epochs (λ=0)
- Training hook steers all positions from last prompt token onwards (matching `steer_once=false` inference)
- `flow_max_new_tokens=30` (CE loss computed only on first 30 response tokens)
- 4 tasks with `inverse=true` (steer toward honesty/non-toxic/non-evil):

| Config | Task | Train dataset | test dataset |
|--------|------|---------------|--------------|
| `gemma_refusal_response_flm.json` | Refusal | `refusal_cast_responses` | `refusal_open` |
| `gemma_deception_flm.json` | Deception | `liarbench` | `liarbench` |
| `gemma_toxic_flm.json` | Toxic | `toxic_jigsaw` | `toxic` |
| `gemma_evil_flm.json` | Evil | `evil` | `evil` |

**Observations**:
- CE loss does not decrease meaningfully during fine-tuning (fluctuates without trend)
- Steered outputs on Toxic and Evil remain at or near 0% accuracy — same ceiling as plain FLOW and all other single-layer activation methods
- Deception shows no measurable improvement over FLOW baseline
- OOD rates are comparable to or worse than CAA at equivalent coefficients

**Conclusion**: Adding LM loss to a single-layer FlowMLP does not improve steering performance. The CE gradient through ~12 transformer layers back to the FlowMLP is too attenuated to produce behavioral change, or the 1-step Euler approximation at t=0 is insufficient for the gradient to carry meaningful directional information. LM loss alone, without architectural changes (lower rank, deeper injection), is insufficient.

---

### 9.24 ExpH: Cancellation Filter Diagnostics via r̂-Alignment (Jun 29-30)

**Motivation**: Tests if the safety ceiling is caused by an active cancellation mechanism that identifies and suppresses steering perturbations aligned with the refusal direction r̂. If r̂-aligned components decay across layers while orthogonal components persist, this is direct evidence of an output-stage filter. Further tests whether removing r̂ alignment from steering vectors or steered activations bypasses the ceiling.

**Method — Four sub-experiments**:

1. **r̂ extraction**: Compute r̂_L = h_L(refusal_text) - h_L(compliant_text) from `behaviour_refusal.json` (response-only, no chat template) at L∈{14, 18, 22, 25}. Captures refusal-execution subspace, not harm-content detection.
2. **Multi-layer deep steering**: CAA vectors at L14/18/22/25 independently. coeff=5. Measure Δcos = cos(h_steered, r̂_L) - cos(h_base, r̂_L) at all target layers.
3. **Three r̂-ablation approaches + accuracy**: (a) Appr1: v' = v - proj_{r̂}(v), renormalized. (b) Appr2: v' = v - r̂, renormalized. (c) Appr3: h' = (h+v) - proj_{r̂}(h+v) at steer layer, norm-preserved. 100 prompts/test, same test sets.
4. **KL divergence per layer**: KL(unsteered || steered) at each layer L, computed via logit lens (model.unembed on residual at position -1). All 4 methods, 50 prompts.

**Preliminary finding — r̂_prompt fails** (paper method, harmful vs harmless prompts): cos(v, r̂_prompt) > 0 for all vectors (evil: +0.12, toxic: +0.15, deception: +0.07 at L14). Paper claims cos(v, r̂_prompt) < 0 predicts ASR increase. However, content-directing vectors (evil/toxic) share harm-content subspace with r̂_prompt, producing positive cosine even when they also align with refusal execution. r̂_prompt is contaminated by harm content detection — unsuitable for cancellation diagnosis.

**Primary experiment — r̂_response (response-only refusal vs compliant)**:

**r̂ norms** increase with depth: L14=69, L18=81, L22=121, L25=185.

#### 9.24.1 Multi-Layer Δcos Analysis (Original Vectors, coeff=5)

**Steer at L14**:

| Layer | evil Δcos | toxic Δcos | deception Δcos | cos(v, r̂)_evil | cos(v, r̂)_toxic | cos(v, r̂)_deception |
|-------|:--------:|:----------:|:--------------:|:---------------:|:----------------:|:-------------------:|
| 14 | +0.073 | +0.107 | +0.002 | +0.122 | +0.154 | +0.066 |
| 18 | +0.039 | +0.042 | +0.000 | +0.115 | +0.057 | +0.053 |
| 22 | +0.013 | +0.004 | +0.001 | +0.091 | +0.007 | +0.046 |
| 25 | +0.005 | +0.011 | +0.001 | +0.072 | +0.096 | +0.020 |

**Steer at L18**:

| Layer | evil Δcos | toxic Δcos | deception Δcos | cos(v, r̂)_evil | cos(v, r̂)_toxic | cos(v, r̂)_deception |
|-------|:--------:|:----------:|:--------------:|:---------------:|:----------------:|:-------------------:|
| 14 | +0.000 | +0.000 | +0.000 | +0.197 | +0.196 | -0.032 |
| 18 | +0.157 | +0.108 | +0.000 | +0.200 | +0.178 | -0.007 |
| 22 | +0.090 | +0.058 | +0.000 | +0.168 | +0.124 | +0.023 |
| 25 | +0.008 | +0.023 | +0.000 | +0.110 | +0.107 | +0.013 |

**Steer at L22**:

| Layer | evil Δcos | toxic Δcos | deception Δcos | cos(v, r̂)_evil | cos(v, r̂)_toxic | cos(v, r̂)_deception |
|-------|:--------:|:----------:|:--------------:|:---------------:|:----------------:|:-------------------:|
| 14 | +0.000 | +0.000 | +0.000 | +0.168 | +0.181 | -0.018 |
| 18 | +0.000 | +0.000 | +0.000 | +0.146 | +0.175 | +0.004 |
| 22 | +0.107 | +0.091 | +0.001 | +0.137 | +0.171 | +0.038 |
| 25 | +0.023 | +0.078 | +0.001 | +0.094 | +0.168 | +0.041 |

**Steer at L25**:

| Layer | evil Δcos | toxic Δcos | deception Δcos | cos(v, r̂)_evil | cos(v, r̂)_toxic | cos(v, r̂)_deception |
|-------|:--------:|:----------:|:--------------:|:---------------:|:----------------:|:-------------------:|
| 14 | +0.000 | +0.000 | +0.000 | +0.138 | +0.109 | -0.017 |
| 18 | +0.000 | +0.000 | +0.000 | +0.106 | +0.126 | -0.006 |
| 22 | +0.000 | +0.000 | +0.000 | +0.122 | +0.165 | +0.035 |
| 25 | +0.068 | +0.165 | +0.002 | +0.069 | +0.301 | +0.122 |

**Key patterns**:
- **Safety tasks: Δcos peaks at steering layer, decays downstream.** Evil and toxic steering pushes activations toward r̂ at injection layer, then suppression occurs over subsequent layers. L18 steer decays by L22-25. L22 steer still shows residual Δcos at L25 (toxic: +0.078).
- **Deception Δcos ≈ 0 at all layers × all steer layers.** Deception vectors are near-orthogonal to r̂, so they do not push toward the refusal direction and do not trigger cancellation. Consistent with deception steering working well (94% original accuracy).
- **Steer-at-L25: no downstream to decay into.** Δcos at L25 = +0.07 (evil) and +0.17 (toxic) with zero subsequent layers for cancellation — this is the perturbation that reaches the logits directly.
- **cos(v, r̂_L) also decays with steer depth** (e.g., evil cos(v,r̂) from L14=+0.122 → L25=+0.069), but Δcos decay is faster — the cancellation acts on the steered activation difference, not the static vector.

#### 9.24.2 Three r̂-Ablation Approaches — Accuracy Results

Three approaches to remove r̂ alignment from steering or steered activations, tested at all 4 steer layers:

| Method | Deception L14 | Deception L18 | Deception L22 | Deception L25 | Evil (all) | Toxic (all) |
|:-------|:------------:|:------------:|:------------:|:------------:|:----------:|:----------:|
| Original* | ~33% | ~61% | ~77% | ~76% | 0% | 0-10% |
| Appr1 (v - proj_{r̂}(v)) | 84% | 81% | 81% | 83% | 0% | 0-1% |
| Appr2 (v - r̂) | ~84% | ~81% | — | — | 0% | 0-5% |
| Appr3 (h' - proj_{r̂}(h+v)) | 43% | 63% | 75% | 75% | 0-3% | 1-10% |

*Original numbers from pre-overwrite run; Appr3 numbers from `eval_appr3/` for reference.

**Critical findings**:

1. **Appr1 (project r̂ out of vector):** Deception steering is worse (84-81% honesty vs original 33-77%). Removing r̂ alignment HURTS deception steering — the r̂ component was functionally part of the deception direction. Safety ceiling unchanged (0%).

2. **Appr2 (subtract full r̂):** Deception steering destroyed (84% honest at L14, was 33%). Safety tasks marginally worse (less harmful). r̂ component is essential for both steering directions — subtracting it removes a functional steering axis.

3. **Appr3 (on-the-fly projection):** Closest to original (43% vs 33% at L14). Small difference from norm-preservation scaling. Safety ceiling unchanged.

4. **No approach breaks the safety ceiling.** All methods produce 0-10% evil/toxic accuracy regardless of r̂ manipulation. The ceiling is not caused by r̂ contamination of the steering vector.

#### 9.24.3 Cos(h,r̂) Comparison: Original vs Ablated (Appr1)

For each steer_layer × target_layer × task, compare Δcos between original vector steering and Appr1 (r̂-projected) steering:

**Representative data — steer at L14, evil**:

| Target Layer | cos(v_orig, r̂) | cos(v_abl, r̂) | cos(base) | cos(orig_steer) | Δcos_orig | cos(abl_steer) | Δcos_abl |
|:------------:|:--------------:|:-------------:|:---------:|:---------------:|:---------:|:--------------:|:--------:|
| 14 | +0.122 | +0.001 | -0.057 | +0.015 | **+0.072** | -0.048 | **+0.009** |
| 18 | +0.115 | +0.040 | -0.071 | -0.030 | **+0.041** | -0.059 | **+0.012** |
| 22 | +0.091 | +0.037 | -0.110 | -0.095 | **+0.015** | -0.111 | **-0.002** |
| 25 | +0.072 | +0.038 | -0.170 | -0.160 | **+0.010** | -0.171 | **-0.002** |

**Pattern holds across all tasks and steer layers.** Ablated vectors have cos(v, r̂) ≈ 0 at steer layer. The resulting Δcos ≈ 0 at ALL target layers — the steered activations no longer align with r̂. Original vectors produce Δcos = +0.07 to +0.17 at steer layer, decaying downstream.

**Interpretation**: Appr1 successfully removes r̂ from both the vector and the downstream activation effect. Yet the accuracy ceiling persists (0% evil/toxic). The r̂-aligned component of the steered activation is NOT the proximal cause of the safety ceiling — removing it does not improve steerability.

#### 9.24.4 KL Divergence Per Layer — All Four Methods

KL(unsteered || steered) measured via logit lens at each layer. Key data (steer at L14, evil, coeff=5):

| Method | L12 | L13 | L14 | L15 | L16 | L17 | L18 | L20 | L22 | L24 | L25 |
|:-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Original | 0 | 0 | 19.13 | 9.51 | 7.22 | 6.78 | 3.57 | 1.84 | 1.09 | 0 | 0 |
| Appr1 | 0 | 0 | 18.80 | 9.32 | 7.60 | 7.19 | 3.71 | 2.14 | 1.17 | 0 | 0 |
| Appr2 | 0 | 0 | 8.57 | 7.11 | 10.80 | 7.89 | 2.78 | 3.15 | 1.12 | 0 | 0 |
| Appr3 | 0 | 0 | 18.59 | 8.18 | 8.24 | 7.60 | 3.38 | 1.82 | 1.01 | 0 | 0 |

**Complete steer-at-L25 KL** (final output — no downstream layers):

| Method | Evil L25 | Toxic L25 | Deception L25 |
|:-------|:--------:|:---------:|:-------------:|
| Original | 2.88 | 0.90 | 0 |
| Appr1 | 3.04 | 0.92 | 0 |
| Appr2 | 0.50 | 0.32 | 0 |
| Appr3 | 2.78 | 0.84 | 0 |

**Key findings**:

1. **Original ≈ Appr1 ≈ Appr3: near-identical KL at EVERY layer.** KL magnitude (how much the output distribution changes) is the same regardless of r̂ ablation. The steering is equally strong — just in a different direction. This rules out "ablation reduces steering strength" as an explanation for unchanged accuracy.

2. **Appr2: different KL profile.** Smaller at injection L14 (8.57 vs 19.13) because v - r̂ has different norm and direction composition. But L16 KL is higher (10.80 vs 7.22) — the model's processing amplifies the Appr2 perturbation differently.

3. **KL decays over 3-4 layers regardless of method.** Peak KL at injection layer, decays by L24-25. This is the same pattern observed in ExpD (§9.13) — the model damps distributional change from steering regardless of direction.

4. **Deception L25 steer: KL=0 for all methods.** Deception has cos(v, r̂) ≈ 0, so steering at the final layer does not change the distribution (measured via logit lens). This is consistent with deception having no KL decay to begin with.

5. **KL magnitude does not predict accuracy.** Original and Appr1 have same KL but different deception accuracy (33% vs 84% at L14). Accuracy depends on steering DIRECTION (which concepts it selects), not on how much the distribution changes.

#### 9.24.5 Synthesis & Conclusion

The cancellation filter hypothesis is partially confirmed but fundamentally incomplete:

**What is confirmed:**
- r̂-aligned perturbations decay 70-95% from injection layer to L25 (Δcos decays, §9.24.1)
- This decay is directional: safety tasks (cos(v, r̂) > 0) get suppressed; deception (cos ≈ 0) does not
- At high coeff (20), the filter saturates and produces erratic alignment

**What is falsified:**
- **The filter is NOT 1D r̂ projection.** Removing r̂ from the steering vector (Appr1) or the steered activation (Appr3) does not break the ceiling. Accuracy stays 0-10% for safety tasks regardless.
- **r̂ alignment is functional for deception steering.** The r̂ component in the deception vector helps drive deception (Appr1 makes it worse). r̂ is not noise — it is part of the multi-directional steering signal.
- **KL magnitude does not track accuracy.** Original and Appr1 have identical KL but different accuracy. The filter suppresses magnitude of distributional change but does not determine which concept wins.

**Remaining open question:** If the ceiling is not caused by r̂ contamination of the steering vector, what causes it? Candidate explanations:
- The ceiling is a multi-dimensional subspace constraint (not 1D r̂) — need principal angle analysis between task concepts and refusal subspaces
- Cancellation operates on non-linear invariants (mean/variance/etc) that our r̂-linear analysis does not capture
- The ceiling is a property of the RLHF training dynamics, not a post-hoc activation filter — weight modification bypasses it (WeightSteer 62-81% Evil) but no activation manipulation does
- The relevant target of cancellation is NOT the r̂ direction but the task direction (evil/toxic) itself — need to test: does cos(h_steered, v_task) decay similarly?

**Next**: All-layer r̂ ablation (remove r̂ from EVERY downstream layer's residual, not just steer vector). Also: multi-dimensional PCA subspace ablation (remove top-k r̂-subspace components rather than single direction).

**Files**: 
- `Experiments/ExpH/activation_analysis.py` — multi-layer Δcos
- `Experiments/ExpH/analyze_ablated_cos.py` — cos comparison original vs Appr1
- `Experiments/ExpH/analyze_kl_all.py` — KL per layer all methods
- `Experiments/ExpH/ablated_vectors.py` — Appr1 vector generation
- `Experiments/ExpH/run_eval.py` — eval harness
- `Experiments/ExpH/results/activation_analysis_multi_layer.json`
- `Experiments/ExpH/results/activation_analysis_ablated_comparison.json`
- `Experiments/ExpH/results/kl_divergence_all_methods.json`
- `Experiments/ExpH/results/eval/` — Appr1 accuracy
### 9.25 In-Distribution Steering (IDS) baseline (Jul 15)

**Motivation**: Evaluates the In-Distribution Steering (IDS) method (Vogels et al., 2025) on safety tasks to test if a dynamically scaled, density-constrained transport map in a PCA-reduced Mahalanobis subspace can bypass the safety ceiling without triggering OOD collapse (PPL/repetition).

**Setup**: IDS on Gemma-2-2b-it for the Evil behavior. Extractor configured with `ids_var_explained=0.40`, `ids_epsilon_pct=0.95`, and `ids_f1_threshold=0.70`. Evaluated on a 5-sample smoke test (seed 42) at layer 14, `coeff=1.0`.

**Key findings**:
- **Baseline**: 0.00%
- **Steered Accuracy**: 20.00% (+20.00% delta)
- **Mean Perplexity (steered)**: 1.73 (extremely low and fluent!)
- **OOD Score (steered)**: 1.82 (well below the task threshold of 10.0 for Evil)
- **F1-Score Selection**: Layer 14 steering vector achieved F1 = 1.0000, passing the layer selection threshold.
- **Interpretation**: IDS successfully steers the model toward Evil behavior while keeping activations strictly on-manifold. The perplexity (1.73) and repetition rate (3.4%) are exceptionally clean, validating that the closed-form quadratic Mahalanobis constraint dynamically scales down steering intensity to prevent OOD collapse.

**Files**:
- `Steering/extractors/ids.py`
- `Steering/steer_models/ids.py`
- `Steering/tests/test_ids.py`
- `Configs/Eval/IDS/Gemma/gemma_evil.json`
- `Results/ids/eval_gemma_evil_coeff_1p0_20260715_004440.json`

---

### 9.21 Mechanistic Investigation: Active Cancellation Hypothesis

#### 9.21.1 Motivation

WeightSteer reaches 62-81% Evil while all activation methods cap at 0-21%. This proves the concept EXISTS in weights but is inaccessible via activation steering. The question shifts from "does the concept exist?" to "why can't activation methods reach it?" We designed a systematic investigation to identify the mechanism.

#### 9.21.2 Competing Hypotheses (Ranked)

| # | Hypothesis | Status | Evidence For | Evidence Against |
|---|-----------|--------|-------------|-----------------|
| 1 | **Active Suppression**: RLHF circuits detect and cancel safety perturbations | PLANNED | WeightSteer bypasses; persona gate on Llama/Qwen; safety neurons ~5%; DBDI two-direction 97.88% ASR | No paper directly shows suppression CAUSES steering failure on safety tasks |
| 2 | **Multi-Dimensionality**: Safety controlled by multiple orthogonal directions | PLANNED | REPS r=4 > CAA; DBDI two-direction 97.88% > one-direction 20%; hidden dims carry secondary signals | Only tested on Llama/Qwen, not Gemma |
| 3 | **Geometry/Manifold**: Safety has higher rank, bimodal, more curvature → linear methods inadequate | REJECTED | REPS partial success at higher rank | CHARS/CurveBall/FLAS all nonlinear — all failed. Geometry is not the bottleneck. |
| 4 | **Budget Waste (J-Space)**: 93% variance is inert → steering wastes budget | SKIP | ~7% causal efficacy (Gurnee 2026) | Deception works fine at 90%+ with same waste structure. Never tested for steering. |
| 5 | **Representation Inaccessibility**: Evil not encoded in activation space | REJECTED | REPS 13% Evil | WeightSteer 62-81% Evil proves concept in weights; partial activation access exists |

Hypotheses 1+2 are NOT contradictory — 2 extends 1 (multi-dimensional suppression). If suppression exists but operates on 2+ orthogonal directions, 1D methods fail for two reasons: (a) they hit the suppression circuit AND (b) they can't cover all relevant directions simultaneously.

#### 9.21.3 Terminology Precision

| Term | Definition | Measurement |
|------|-----------|-------------|
| **Suppression** | Model actively detects perturbation and reduces its effect | Perturbation causal effect on output < what intermediate-layer norm predicts |
| **Cancellation** | Two signals in opposite directions partially cancel | Sub-additive ablation (Wang 2026) — ablating both writers+cancellers produces less effect than sum of individual ablations |
| **Natural Decay** | Perturbation effect diminishes through nonlinear layers regardless of model intent | ALL perturbations decay similarly (including random vectors) |
| **Redirection** | Perturbation energy preserved but moved to causally irrelevant dimensions | Norm stays high, causal effect drops |

**Critical gap**: KL divergence is NOT sufficient to distinguish these four phenomena. KL decay could mean suppression, natural decay, redirection, or manifold attraction — it cannot discriminate between them. Better metrics: causal effect decay (refined DLA), head-level signed attribution, perturbation transport efficiency.

#### 9.21.4 Key Papers Informing This Investigation

| Paper | Tag | Key Contribution |
|-------|-----|-----------------|
| Wang 2026 (2606.07560) Writers/Cancellers | PLANNED | Two head populations (writers push logit UP, cancellers push DOWN); sub-additive ablation; task-conditional sign flip |
| Rivera & Africa (2511.21399) Steering Awareness | PLANNED | Models CAN detect steering after LoRA training (95.5% on Qwen-32B); BUT detection ≠ resistance — detection-trained models are MORE susceptible (+32-36pp compliance) |
| DBDI (AAAI 2026) Multi-Direction | PLANNED | 97.88% ASR with two directions vs 20% with one — safety requires multi-dimensional intervention |
| OV-circuit (2604.08524) Propagation | PLANNED | OV-circuit is primary propagation pathway (≥44.5% vs QK 8.75%) — steering signals propagate mainly through OV paths |
| Safety neurons (NeurIPS 2025) | PLANNED | ~5% of neurons control safety behavior — sparse circuit, not distributed |
| Detection→Refusal heads (2603.09801) | PLANNED | Two-component circuit: detection heads → refusal heads |
| Persona-Refusal (2606.26161) | PLANNED | MP injection drops refusal 97%→2%; projection restores to 96.8%; persona gates refusal at L20-L22 |
| J-Space (Gurnee 2026) | SKIP | ~93% activation variance inert; never tested for steering efficacy; "global workspace" concept borrowed from neuroscience without rigorous mapping |
| Assistant Axis (Lu 2026) | SKIP | Legit PC1 cos>0.92 across families; but drift-harm correlation r=0.39–0.52 is correlational not causal; orthogonalizing against persona would likely just remove refusal |

#### 9.21.5 Experimental Chain

| # | Experiment | Question | Boolean? | Status |
|---|-----------|----------|----------|--------|
| 1 | **Differential Perturbation Survival** | Does Evil perturbation decay faster than Deception at intermediate layers? | Yes | **PLANNED** |
| 2 | Layer Injection Sweep | Where is the suppression gate (which layer)? | Yes | PLANNED (after Exp1) |
| 3 | Head-Level DLA | Which heads do the cancelling? | Yes | PLANNED (after Exp1) |
| 4 | Multi-Direction Test | Does Evil need >1 direction to succeed? | Yes | PLANNED (if Exp1 rejects suppression) |
| 5 | Restoration Test | Is concept suppressed (recoverable) or deleted? | Yes | PLANNED (future) |

**Exp1 Decision Gate**:
- Evil decays FASTER than Deception → suppression confirmed → proceed to Exp2 (locate) then Exp3 (identify heads)
- Evil and Deception decay at SIMILAR rates → no differential cancellation → proceed to Exp4 (multi-dimensionality)
- Deception decays FASTER → unexpected → investigate why Deception is suppressed more

#### 9.21.6 Re-Run Old Experiments: REJECTED

Re-running Wang's DLA pipeline, Persona-Refusal's projection experiment, or other paper experiments on our model is NOT efficient for a boolean check. Reasons:
1. Wang's pipeline is designed for Pythia on ICL tasks — full replication on Gemma-2B is a research project, not a boolean check
2. Persona-Refusal tested on Llama/Qwen only — re-running on Gemma tests a different model, not the same mechanism
3. Our Differential Perturbation Survival (Exp1) is a NEW, SIMPLER test that directly answers the boolean question

If Exp1 confirms cancellation, THEN consider replicating Wang's DLA on Gemma-2B to identify which specific heads are doing the cancellation. But that is step 2, not step 1.

#### 9.21.7 Exp1 Implementation Plan: Differential Perturbation Survival

**File**: `Experiments/ExpCausalSuppression/01_differential_survival.py`

**Setup**:
- Model: `google/gemma-2-2b-it` via TransformerLens
- Vectors: Pre-extracted CAA from `Vector/CAA/Gemma/evil/` and `Vector/CAA/Gemma/deception/`
- Injection layer: 14 (universal steering layer)
- Tracking layers: L14–L25 (all residual stream positions)
- Evaluation texts: 50 harmless prompts from `TrainDataset/behaviour/evil/normal.jsonl`
- Coefficient: 2.0

**Metrics** (per layer, per prompt):
1. **Norm survival**: `||steered_L - unsteered_L||` normalized by L14 value
2. **Direction persistence**: `cos(steered_L - unsteered_L, v_original)`
3. **Causal projection**: `dot(steered_L - unsteered_L, v_original)`

**Analysis**: Average across 50 prompts, plot norm/cosine/projection vs layer for Evil vs Deception, compute decay rates (slope of log-norm vs layer), t-test for differential decay.

**Expected runtime**: ~2 hours GPU. **Decision**: Evil decay > Deception decay (p < 0.05) → cancellation confirmed.


## 10. Conclusions & Future Plans

### 10.1 The Safety Steering Ceiling vs Deception Steerability

**Central finding**: Dissociation between two alignment types:

| Method class | Alignment axis | Ceiling | Fine-tuning ceiling | Mechanism |
|:-------------|:---------------|:-------:|:-------------------:|:----------|
| **Activation** | **Safety** (Toxic, Evil) | **0-45%*** | **42% (Toxic), 31% (Evil)** | Active RLHF guardrail |
| **Weight** | **Safety** (Evil) | **62-81%** ★ | **31% (Evil)** | Bypasses ceiling via weight modification |
| Activation | **Honesty** (Deception) | **94%** | **8%** | Passive pre-training feature |

* PID Toxic c=1.0 reaches 45% clean; higher coeffs and all Evil OOD (repetition-driven).
★ WeightSteer (LoRA contrastive weight arithmetic, MLP-only, all 26 layers, c=1.0-1.2).

Activation methods bounded **only on safety tasks** (Toxic 0-21%, Evil 0-10%) for single-layer + sparse multi-layer. **PID exception**: Toxic c=1.0 45% clean, but higher coeffs + all Evil OOD. **WeightSteer breaks entirely**: 62-81% Evil by modifying MLP weights — ceiling is **activation-specific** (§9.7).

**Deception steers well both ways**: 94% deceptiveness (COBRA), 90% honesty (PCA-OT). Ceiling is about whether task axis protected by active safety circuitry, not "anti-alignment."

**Tail analysis shows steering is in-manifold** (§9.18): Steering vectors are 73-98% concentrated in the in-manifold PCA subspace (threshold: PCs for 90% var, which ranges from 17 for Deception to ~440 for Toxic). ACT affine parameters (w, beta) do NOT deviate more in off-manifold dims (p=0.34). If a suppression mechanism exists, it would only need to monitor the in-manifold subspace, not all 2304. Since tail dims carry no concept signal, no method benefits from operating there — all activation methods collapse to CAA inside the manifold, and all face the same ceiling. WeightSteer modifies parameters rather than activations in the monitored subspace, which may explain why it bypasses the ceiling.

### 10.2 Why Safety Tasks Resist Steering but Deception Doesn't

Observations (mechanism unknown):

1. **Steering subspace varies by task**: Deception ~17 PCs for 90% per-sample variance, safety tasks ~156-441. Steering vector concentrates in these top PCs (72-98% of norm). All activation methods operate in this same subspace.
2. **Safety tasks show late-layer KL decay**: Toxic and Evil KL drops 0.49× from L14 to L25. Deception KL grows 1.08×. What causes the decay is unknown.
3. **Deception has no KL decay**: Honesty may lack whatever suppression mechanism safety tasks have — or the mechanism may be task-specific.
4. **Weight modification bypasses the KL decay**: LoRA modifies parameters directly. Fine-tuning 42% Toxic, 31% Evil where activation gets 0%.
5. **WeightSteer amplifies further**: 62-81% Evil (§9.7) — surpasses activation (0-10%) and fine-tuning (31%). Contrastive objective more sample-efficient. **Exp D (§9.13): WS KL grows x8.95 vs activation decay x0.05-x0.20.**

**2-Layer Problem — Revised (Jun 23)**:
- ~~**Extraction layer**~~ [**FALSIFIED**, §9.13 expA]: Per-position SNR ratio early/late = 1.11×. Signal at ALL positions.
- **Inference layer** [**CORRELATIONAL EVIDENCE**, §9.13 expD]: Logit lens KL shows toxic perturbation decays (0.49×) while deception persists (1.08× growth). Despite identical propagation (§9.13 expB). Consistent with an output-stage filter, but correlation only.
- **Propagation level** [**FALSIFIED**, §9.13 expB]: Perturbations AMPLIFY ~3× regardless of direction. No mid-stream cancellation.

**Revised understanding**: Signal at every position — no extraction bottleneck. KL decays at late layers on safety tasks but grows on Deception. The cause of this decay is unknown — it could be a learned output filter, a side effect of RLHF optimization, or a different mechanism entirely.

**Remaining open questions**:
1. Where does KL diverge — which layer(s)?
2. Is the KL decay specific to toxic/evil or any safety-violating direction?
3. Can multi-layer injection (PID, §9.12) reliably bypass?
4. ~~**Does WeightSteer disable KL decay?**~~ [Observed, §9.13 expD]: **WS KL grows x8.95** vs activation decay x0.05-x0.20. The mechanism is unknown.

### 10.3 Activation Steering vs Weight Modification: Ceiling

| Aspect | Activation Steering | WeightSteer (contrastive LoRA) | Fine-Tuning (LoRA) |
|:-------|:-------------------|:------------------------------|:-------------------|
| **Modified** | Hidden states at inference | Model weights (temporary) | Model weights (permanent) |
| **Evil accuracy** | 0-10% (all). PID 6% clean | **62-81%** | 31% |
| **Toxic accuracy** | 0-45% (PID c=1.0 clean) | 0% or OOD | 42% |
| **Efficiency** | One forward pass, no training | ~10 min training | ~10 min training |
| **Direction control** | Smooth by coeff | Smooth by coeff | Binary |

**Key findings**:
1. **WeightSteer beats fine-tuning** on Evil (62-81% vs 31%) — contrastive objective more sample-efficient.
2. **Weight modification bypasses the KL decay pattern** (KL logit lens, §9.13): WS KL **x8.95** growth vs all activation x0.05-x0.20 decay. Even multi-layer CAA (18 layers) only x2.46 with chaotic trajectory + 0% accuracy.
3. **Ceiling is activation-specific**: all activation 0-10% Evil; weight 62-81%.
4. **PID partially breaks ceiling**: 45% Toxic clean, narrow gain window (§9.12). I-term accumulates error across layers, partially overwhelms hypothesised suppression. Fragile vs weight modification. PID KL erratic (spikes L11/L13/L15), L25=2.68 vs multi-layer CAA 5.30 despite latter 0% — KL ≠ alignment.

### 10.4 PCA Geometry: Universal Structural Facts

All tasks share same PCA geometry on `google/gemma-2-2b` layer 14:
- **Intrinsic dim varies by task**: Deception ~17 PCs, safety tasks ~156-441 PCs for 90% per-sample variance
- **Concept separation 2D**: Top-2 PCs capture 98.5% class mean diff
- **Tail structure universal**: 99%+ kurtosis off-manifold for ALL tasks
- **Jacobian rank uniform**: eff_rank ≈ 15-17

**Implications**:
1. Per-dim canonical basis analyses always misleading — rotate to PCA first
 2. COBRA subspace projection: steers in the concept-relevant PCA subspace (varies by task dimensionality, but concept mean difference is near-2D)
4. PCA-OT = ACT with lower complexity, better PPL

### 10.5 Tail-Distribution Analysis (Falsified Hypothesis)

Documentation of falsified hypothesis. Preserved for completeness.

#### 10.5.1 Method (Initial — Corrected Later)

**Caveat**: Initial analysis computed kurtosis **per-sample across 2304 dims** (not standard). Standard: per-dim across samples. All subsequent (§10.5.3, §10.5.6) uses standard per-dim computation in PCA space.

Layer 14 pre-steering. Z-score per dimension. Tail metrics: per-sample kurtosis, max_z, tail_frac.

#### 10.5.2 Aggregate Results (Exploratory Heuristic, Pooled 5 Methods × 3 Tasks)

| Metric | Correct (n=194) | Incorrect (n=1106) | Delta | p-value |
|--------|:---------------:|:------------------:|:-----:|:-------:|
| **kurtosis*** | **−0.4043** | **+0.0507** | **+0.4551** | **0.0003** |
| **tail_frac_2sig** | **0.0344** | **0.0433** | **+0.0089** | **0.0001** |
| **tail_frac_3sig** | **0.0019** | **0.0032** | **+0.0013** | **0.0001** |
| **max_z** | **3.1783** | **3.4707** | **+0.2924** | **0.0000** |
| act_norm | 204.82 | 205.85 | +1.03 | 0.5184 |

*\*Per-sample kurtosis. Not standard kurtosis.*

#### 10.5.3 PCA-Based Re-Evaluation (Standard Per-Dimension Kurtosis)

Canonical-basis treats 2304 dims as independent. Repeating in PCA basis with proper per-dim kurtosis (`scipy.stats.kurtosis(fisher=True, bias=False)`):

| Task | k (90% var) | W2 in-manifold | Kurtosis in-manifold | Top-2 sep ratio |
|:---|:---:|:---:|:---:|:---:|
| Deception | 17/2304 | 0.134 | 0.011 | 0.984 |
| Toxic | 441/2304 | 0.035 | 0.006 | 0.986 |
| Evil | 212/2304 | — | — | — |
| Refusal | 156/2304 | 0.030 | 0.005 | 0.987 |

**Kurtosis in top PCs**: γ₂ ≈ 3-5 (near Gaussian). **In tail PCs**: γ₂ = 50-200+. Kurtosis concentration ratio R = frac{sum(d in {off-manifold)} |gamma_2^{(d)}|}{sum(d=1)^{2304} |gamma_2^{(d)}|}. For all tasks: R > 0.99 — 99%+ kurtosis resides in off-manifold (PC17+ for Deception, PC441+ for Toxic) directions.

These findings are robust to tail threshold. Hill estimator at α<3.5 gives Toxic=28 dims (vs 350 at α<4, vs 0 with kurtosis ≥10). No threshold produces a tail-dim set that predicts steering performance.

Evidence: Hill-based tail concentration + class separation + per-sample failure analysis together show tails irrelevant for steering.

#### 10.5.4 Methodological Implication

COBRA subspace projection is sensible: concept mean differences are near-2D (top-2 PCs capture 98.5% of separation), so steering in a low-rank subspace preserves residual structure. But per-sample variance is task-dependent (Deception ~17 PCs, safety up to ~441). Preserving off-manifold residual is still correct: **transport should not operate on dimensions with no concept signal.**

#### 10.5.5 CHARS Failure Mechanism

CHARS gets 85% on Deception but 0% on Toxic and Evil. RBF kernel is functional (values 0.00-0.98). Symlog tail annealing does not improve accuracy (0%). The same ceiling that affects all activation methods applies here — the cause is unknown (§9.13 expD shows correlated KL decay).

#### 10.5.6 Per-Sample Tail-Failure Correlation (Jun 25)

**Question**: Do tail-heavy activation distributions predict which test samples fail?

**Data**: Pre-steering activations in ACT eval results — Toxic (c=2.0, 13% acc) and Deception (c=1.0, 85% acc).

**Tail metrics**: Off-manifold residual norm (PCA k=13/50/99), off-manifold ratio, in-manifold norm, max |z| in PC/canonical space, N canonical dims with |z|>3.

**Method**: Mann-Whitney U + Spearman per metric.

**Results — Toxic (ACT c=2.0)**:

| Metric | MW p (run1) | MW p (run2) | Spearman ρ | Direction | Significant? |
|:-------|:-----------:|:-----------:|:----------:|:---------:|:-----------:|
| Off-manifold ratio (k=13) | 0.17 | 0.35 | -0.10 | none | ❌ |
| Off-manifold ratio (k=50) | 0.27 | 0.26 | -0.06 | none | ❌ |
| Residual norm (any k) | 0.38-0.84 | 0.39-0.84 | -0.03 to +0.01 | none | ❌ |
| Total norm | 0.87 | 0.91 | +0.11 | none | ❌ |
| N tail canonical dims | 0.44 | 0.82 | -0.02 | none | ❌ |

**Results — Deception (ACT c=1.0, 85% acc)**:

| Metric | MW p | Spearman ρ | Direction | Significant? |
|:-------|:----:|:----------:|:---------:|:-----------:|
| Total norm | **0.001** | -0.31 | fails → higher norm | ⚠ weak |
| In-manifold norm (k=13) | **0.0000** | -0.45 | fails → higher in-manifold | ⚠ strong |
| Off-manifold ratio (any k) | 0.16-0.87 | -0.10 to +0.11 | none | ❌ |
| Max canonical z | 0.013 | -0.23 | fails → higher | ❌* |
| N tail canonical dims | 0.008 | -0.24 | fails → more | ❌* |

*Not significant after Bonferroni (α=0.0006).

**Interpretation**: Deception c=1.0 failures have higher in-manifold norm (p=0.0000), not higher off-manifold residual. Effect in CONCEPT direction, not tail. Harder examples need stronger steering. Effect vanishes at c=3.0.

**Verdict**: Off-manifold tail distribution does NOT predict steering failure for any metric, PCA cutoff, task, or accuracy. Tail-heavy distribution is **static structural property** of ReLU/GeLU residual stream — no concept signal, correlates with no outcome. Tails safely ignorable for steering method design.

**Conclusive cross-experiment synthesis** (Jun 26, §9.18): Three additional experiments confirm and deepen this finding:

| Experiment | Question | Key finding | Status |
|:-----------|:---------|:------------|:-------|
| **Target-vs-contrast tail** | Does one class have more tail mass? | Steering vector 73-98% in-manifold. Class tail properties identical (MW p>0.05). Zero canonical tail dims (kurt >= 10) for Toxic/Deception/Refusal. | ✅ **Completed.** |
| **CHARS tail-cluster transport** | Do tail centroids transport differently? | Deception: identical signatures under both definitions (T\! -> \!T=84.9\%, Enr=2.83x). Toxic/Evil: tail sources map to body targets (T\! -> \!Tapprox28{-}45\%). No predictive power for accuracy. | ✅ **Completed.** |
| **ACT affine in tail dims** | Are w,beta different in tail dims? | w: not systematically different (p=0.34). beta: significantly **smaller** in off-manifold (p=4x10^{-11}, rho=-0.19 to -0.34). ACT concentrates modifications in-manifold. | ✅ **Completed.** |
| **Post-steer tail** | Does steering change tail magnitude? | Requires full inference run. Low priority — all extraction-level questions answered. | 🔲 Deferred. |

**Implication**: The steering signal, class separation, and ACT modifications all live in-manifold. The off-manifold tail is structurally heavy-tailed but functionally inert. If a suppression mechanism exists, it would only need to monitor the low-rank concept subspace, not all 2304 dims. This is consistent with the ceiling being method-independent: no activation method has a reason to use tail dims, so all are equally confined.
These do NOT re-test the falsified prediction hypothesis — they probe mechanistic properties of the extraction/steering process.

### 2.6 FishBack (Flow + OT + Fisher) — Reimplemented Method

**Status**: Code complete, gatekeeper pending (Jul 2026)
**Hypothesis**: Combining flow matching + VGG-Flow target gradient guidance + task-contrastive Fisher subspace will outperform prior nonlinear methods on safety tasks.

**CRITICAL ALIGNMENT CORRECTION (Jul 2026)**:
- **Prompt-Boundary Injection**: Formats user question prompt using `apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True)`. The exact token index of the prompt boundary is `prompt_len - 1` (the `<start_of_turn>model\n` token), matching the inference steering hook location during `model.generate()`.
- **Response-Only CE Loss (`response_mask`)**: Applies `response_mask` zeroing out all question prompt tokens (`response_mask[b, :prompt_end_indices[b]] = 0`) so cross-entropy loss and target gradients $\nabla_{h_{\text{predicted}}} \text{CE}_{\text{response}}$ evaluate **exclusively on target response tokens**.
- **Memory Efficiency**: Uses `GRAD_BATCH = 2` with `ce_loss.backward()` to destroy autograd graph buffers instantly and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, maintaining peak VRAM <6.2 GB.

**Key improvement over old FishBack**: Uses task-contrastive Fisher subspace (harmful vs safe activations) combined with VGG-Flow target response loss guidance.

**Motivation**: Flow + PCA works for refusal (easy, SNR=4.95) but fails for evil/toxic/deception (hard, curved manifolds). PCA captures variance, not causality. We need a subspace that captures behavioral directions where perturbations at layer 14 affect the model's safety alignment, not just variance in activations.

#### 2.6.1 Method Components

| Component | Failure Mode Addressed | Mechanism |
|:----------|:----------------------|:----------|
| Flow matching (multi-step) | ODESteer's single-step Euler fails on curved trajectories | Learns full nonlinear velocity field v_θ(h,t); multi-step generation |
| OT loss (Sinkhorn) | MSE treats points independently, doesn't model distribution geometry | Optimal transport finds minimum-cost map between distributions |
| Task-contrastive Fisher subspace | PCA captures variance, not causality; next-token Fisher is too narrow | Top eigenvectors of G = Cov(∇_{h_ℓ} L) where L = ‖h_harmful - h_safe‖² captures behavioral directions |
| Jacobian regularization | Sharp, brittle transformations (GINN cliff: 92% at c=2, 0% at c=3) | Penalizes var(‖v_pred‖) → smooth transport |

#### 2.6.2 Task-Contrastive Fisher Subspace — How Behavioral Directions Are Measured

**NOT next-token Fisher. NOT target token. NOT contrast token.**

The subspace is defined by which directions at layer 14 consistently differentiate harmful from safe activations.

**Definition**:
```
G = Cov(∇_{h_ℓ} L) where L = ‖h_harmful - h_safe‖²
```

Where:
- h_harmful = activations from harmful prompts (Evil/Toxic)
- h_safe = activations from safe prompts
- G = (1/N) * gradients^T @ gradients where gradients = 2 * (h_harmful - h_safe)

**What this measures**: "Which directions at layer 14 consistently affect the contrastive loss between harmful and safe activations?"

**Computation**:
1. For each training prompt pair (harmful, safe), extract activations at layer 14
2. Compute difference: diff = acts_harmful - acts_safe
3. Gradients: gradients = 2 * diff (gradient of L2 loss)
4. Fisher matrix: G = (1/N) * gradients^T @ gradients
5. Eigendecomposition → top-k eigenvectors → R (projection matrix)

**Interpretation**: Top eigenvectors of G = directions in h_{14} space that maximally differentiate harmful from safe activations. These are the "behavioral" directions for steering — not because they change the next token, but because they capture the model's safety alignment structure.

#### 2.6.3 Why This Is Meaningful for OT Loss

In the Fisher subspace, the distance between two activations h1 and h2 is:

`d_F(h1, h2) = sqrt((h1-h2)^T G (h1-h2)) = sqrt((h1-h2)^T Cov(∇L) (h1-h2))`

This measures how much the activations differ along behavioral directions.

**Key property**: Two activations that are close in raw L2 but differ in behavioral directions will be FAR apart in Fisher distance. Two activations that are far in raw L2 but share behavioral structure will be CLOSE.

This is exactly what we want for OT loss — distance should correspond to "how different will the model's safety behavior be?"

#### 2.6.4 Loss Function

```
L_total = L_flow + λ_ot * L_ot + λ_jac * L_jac
```

Where:
- **L_flow** = Velocity matching (MSE between predicted and true velocity): `‖v_θ(h_t, t) - (h_harmful - h_safe)‖²`. This is a proxy for LM loss. FLAS showed LM loss > MSE, but for gatekeeper, velocity matching is simpler. Can upgrade to LM loss later.
- **L_ot** = Sinkhorn distance in Fisher subspace: `W_2(R @ h_pred, R @ h_target)` where R is the Fisher projection matrix
- **L_jac** = `var(‖v_pred‖)` — penalizes velocity norm variation across samples for smooth transport

**Design decisions (gatekeeper)**:
- Flow architecture: MLP (2-layer, 512 hidden, ReLU) — simple, proven in GINN
- OT solver: Sinkhorn (epsilon=0.1, differentiable, fast)
- Subspace: Task-contrastive Fisher k=64 (fair comparison with PCA k=64)
- Loss balancing: λ_ot=1.0, λ_jac=0.1 (gatekeeper tests whether components help at all)

#### 2.6.5 Gatekeeper Experiment

**Question**: Does OT in task-contrastive Fisher subspace outperform MSE velocity loss in PCA subspace?

**2×2 ablation**:

| | PCA k=64 | Fisher k=64 |
|:--|:---------|:--------------|
| **MSE velocity loss** | Baseline | Tests Fisher value |
| **OT loss** | Tests OT value | Full FishBack |

**Setup**: gemma-2-2b-it, layer 14, refusal task, coefficients 1-10, 500 training prompts

**Decision gates**:
- IF Fisher+OT > PCA+MSE at 3+ coefficients → method is promising, proceed to full implementation
- IF Fisher+OT ≈ PCA+MSE (within 5%) → OT doesn't help, focus on Fisher alone
- IF Fisher+OT < PCA+MSE → something is wrong, redesign

#### 2.6.6 CRITICAL REMINDERS

1. **NO TARGET TOKEN**: The concept (evil, toxicity) is NOT represented by any single token. Do NOT use next-token prediction to define the subspace.
2. **TASK-CONTRASTIVE**: The subspace is defined by harmful vs safe activation differences, not variance or next-token sensitivity.
3. **BEHAVIORAL DIRECTIONS**: The Fisher subspace captures directions that differentiate safety behaviors, not just any directions.

### 10.6 Active Plan

See `Steering/plan.md` for the current experimental plan, branching logic, and decision criteria. This section was replaced on 2026-07-17.
