# Comprehensive Literature Survey: Representation Steering for Language Model Alignment

Detailed survey of 70+ papers (2023–2026). Each paper summarized with **Motivation**, **Method & Experiments**, **Formal Statements & Verbal/Logical Proofs**, **Conclusions**, and **Relevance to Our Work** (supports / opposes / extends).

**Our central finding**: Activation steering methods hit a 0–45% ceiling on safety tasks (Toxic/Evil) but achieve 94% on Deception. The ceiling is activation-specific — weight modification (WeightSteer) reaches 62–81% on Evil. 7 mechanistic hypotheses systematically falsified. Persona gate + suppression circuit + double-constraint model emerging as leading explanation.

---

## Table of Contents

1. [Linear Intervention Foundations (2023–2024)](#1-linear-intervention-foundations-20232024)
2. [Optimal Transport Steering (2024–2026)](#2-optimal-transport-steering-20242026)
   - 2.4 [Recent OT Steering (2026)](#24-recent-ot-steering-2026)
3. [Extended OT and Distribution Optimization](#3-extended-ot-and-distribution-optimization)
4. [Multi-Behavior Steering: What Can We Steer?](#4-multi-behavior-steering-what-can-we-steer)
   - 4.5 [Multi-Attribute Steering](#45-multi-attribute-steering)
5. [Control-Theoretic Steering](#5-control-theoretic-steering)
6. [Shallow Alignment Theory](#6-shallow-alignment-theory)
7. [Persona Subspace: Assistant Axis, J-Space, and Persona-Refusal Coupling](#7-persona-subspace-assistant-axis-j-space-and-persona-refusal-coupling)
8. [Suppression Circuits & Mechanisms](#8-suppression-circuits--mechanisms)
9. [Weight Modification and Fine-Tuning](#9-weight-modification-and-fine-tuning)
10. [Nonlinear and Learned Steering](#10-nonlinear-and-learned-steering)
11. [Safety Externalities: How Steering Breaks Safety](#11-safety-externalities-how-steering-breaks-safety)
12. [Manifold and Geometry Theories](#12-manifold-and-geometry-theories)
13. [Predictive Diagnostics: When Does Steering Work?](#13-predictive-diagnostics-when-does-steering-work)
14. [Summary Table](#14-summary-table)
15. [Relevance to Our Safety Ceiling: Synthesis](#15-relevance-to-our-safety-ceiling-synthesis)

---

## 1. Linear Intervention Foundations (2023–2024)

### 1.1 ActAdd / CAA (Zou et al., 2023; Concept Algebra adapted)

**Motivation**: Can we steer LLM behavior without retraining by adding vectors to hidden states during inference? Prior work required fine-tuning for every behavior change. Linear interventions are training-free and model-agnostic.

**Method & Experiments**:
- Extract steering vector as difference-in-means between positive (target behavior) and negative (baseline) activation sets
- Single-layer injection at inference: `h' = h + α·v`
- Sweep coefficient α to modulate behavioral strength
- Evaluation: classification accuracy on behavioral benchmarks

**Formal Statements & Proofs**:
- *Claim 1*: Linear directions encode concepts in the residual stream. **Proof**: By construction — if two behavioral classes activate different neurons, their mean difference is a direction that separates them. No formal proof that this direction *causally* influences behavior, only correlation.
- *Claim 2*: Translation along these directions modulates behavior monotonically. **Proof**: Empirically observed as inverted-U. Monotonic up to a point, then collapses — no formal guarantee.
- *Claim 3*: The method is model-agnostic. **Proof**: Holds across GPT-2, LLaMA, Gemma families. However, effectiveness is task-dependent.

**Limitations exposed by later work**:
- Fixed translation ignores covariance structure — treats all dimensions equally
- Single coefficient cannot adapt to heterogeneous activations
- OOD at high coefficients (inverted-U curve)
- No theoretical guarantees on stability

**Relevance to us**: CAA is our baseline. All 25+ methods compared against it. We confirm the inverted-U and show CAA gets 0% on safety tasks, 85–94% on Deception. **Supports**: linear directions exist. **Opposes**: linear methods sufficient for safety tasks.

### 1.2 ActAdd (Turner et al., 2023)

**Motivation**: Same question — demonstrate simple linear interventions can steer behavior across harmful/honest tasks.

**Key distinction**: Emphasized universality across behaviors. Showed steering on both harmful and honest tasks with the same framework.

**Our finding**: We replicate — simple addition works for Deception (85%+) but fails for safety (0%). The universality claim does not hold for safety-critical tasks on instruction-tuned models. **Supports** (for Deception), **opposes** (for safety).

---

## 2. Optimal Transport Steering (2024–2026)

### 2.1 AcT / Linear-AcT (Activation Transport)

**Paper**: Concept Algebra extended; linear-AcT formalization.

**Motivation**: Linear addition (CAA) assumes equal covariance between source and target distributions. This is rarely true — safety task distributions have heterogeneous covariance (Toxic: K=6.5 clusters, Evil: K=2 bimodal). AcT addresses this with affine transport maps.

**Method & Experiments**:
- Model source/target as Gaussians per layer
- Transport map: `T(y) = A·y + b` where A and b come from moment matching
- Linear-AcT: under equal covariance, reduces to diff-in-means (proving CAA is a special case)
- Practical: univariate marginals, clamped to empirical range
- Evaluation: toxicity suppression, general capability preservation

**Formal Statements & Proofs**:
- *Theorem*: diff-in-means = mean transport under equal-covariance Gaussians. **Proof**: If Σ_source = Σ_target, the optimal transport map is linear: T(y) = y + (μ_target − μ_source). This is exactly CAA. Therefore CAA is a special case of AcT.
- *Theorem*: Clamping to empirical support prevents pathological outputs. **Proof**: Without clamping, the affine map can send activations to regions with zero density (far from any training point), producing degenerate outputs. Clamping bounds the output to the convex hull of training data.

**Key findings**:
- Outperforms CAA on toxicity suppression (21% vs 1%)
- Better PPL preservation at moderate coefficients
- Per-dim scaling (ω) concentrated [0.98, 1.02], β near zero — ACT's edge comes from the full-distribution matching, not per-dim scaling
- ~73–98% of ACT's steering energy lands in the same PCA components as CAA

**Relevance to us**: AcT achieves 21% on Toxic (best single-layer) but still hits the ceiling. PCA-OT generalizes this with lower complexity. **Supports**: per-dim scaling helps marginally. **Opposes**: per-dim scaling breaks the ceiling.

### 2.2 PCA-OT (Nanfack et al., 2026, arXiv:2603.04355)

**Motivation**: Full-dimensional OT is expensive and noisy. PCA can identify the discriminant subspace for low-rank transport.

**Method & Experiments**:
1. Extract activations from harmful/harmless examples
2. Pool-mean center + SVD for top-k principal components
3. Project both distributions into k-dim PCA space
4. Gaussian OT in low-dim: `T_k(y) = A_k·y + b_k`
5. Lift back: `A_full = P · A_k · P^T`

**Formal Statements & Proofs**:
- *Claim*: Pooled-mean centering links PCA to Fisher discriminant: top PC aligns with `S_W^{-1}(μ_H − μ_S)`. **Proof**: Under equal-covariance Gaussian assumption, the optimal discriminant direction is `Σ^{-1}(μ_H − μ_S)`. If Σ ≈ σ²I (isotropic), this reduces to `μ_H − μ_S`, which is the top PC of the pooled data.
- *Claim*: Low-rank sufficiency — concept lives in ≤ 10 dimensions. **Proof**: Empirically, k=2 yields 82.4% ASR on Llama-2-13B. First PCs capture discriminant structure. Refusal crystallizes at 40–60% depth.

**Key findings**:
- **Optimal rank k=1 or 2**: First PCs capture discriminant structure
- **Refusal crystallizes at 40–60% depth**: Sharp transition in middle layers
- PCA-OT beats baselines: 77–79% ASR vs RFA 46–73%, AcT 74–78%
- General capabilities preserved: MMLU 52.86% vs 53.15%

**Relevance to us**: We implement PCA-OT and confirm — Deception 90% honesty, but Toxic/Evil still at ceiling. The rank structure is real (Deception ~17 PCs, safety ~156–441), but doesn't explain the ceiling. **Supports**: low-rank sufficiency for Deception. **Opposes**: low-rank transport breaks safety ceiling.

### 2.3 LinEAS (NeurIPS 2025)

**Motivation**: Independent per-layer transport maps cause causal inconsistency — the output of one layer's map feeds into the next layer's unmodified processing.

**Method & Experiments**:
- Joint optimization of all per-layer affine maps
- Global distributional loss: match target distribution across all layers simultaneously
- Group Lasso sparsifier: learns which layers actually need intervention
- Compression: up to 100× fewer effective parameters

**Formal Statements & Proofs**:
- *Causal consistency theorem*: Joint optimization preserves inter-layer coherence. **Proof**: When optimizing all layers jointly, the transport at layer l is conditioned on the transported output of layer l-1, not the original output. This ensures the perturbation remains consistent through the network.
- *Sparsity*: Most layers don't need intervention. **Proof**: Group Lasso drives most layer-specific weights to zero. Empirically, 1–2 layers suffice.

**Our finding**: We didn't implement LinEAS (requires end-to-end training). Relevant as evidence that per-layer independence is a real limitation, but the ceiling applies regardless. **Supports** (layer independence is limiting), **opposes** (joint optimization doesn't address safety ceiling).

### 2.4 Recent OT Steering (2026)

A wave of 2026 papers extends OT-based steering in three directions: (a) nonlinear transport via flows/ODEs, (b) OT for new architectures (diffusion LMs), and (c) theoretical unification.

#### 2.4.1 OT for Masked Diffusion Language Models (TTU, ICLR 2026)

**Motivation**: Activation steering has been studied exclusively for autoregressive LMs. Masked diffusion language models (dLLMs) have a fundamentally different generation mechanism — iterative unmasking — and it is unknown whether OT-based steering transfers.

**Method & Experiments**:
- Frame activation steering for dLLMs as OT over unmasking trajectories
- Learn affine maps (Mean-AcT, Gaussian OT, Linear-AcT) to steer the diffusion process
- Test on LLaDA-Instruct, LLaDA 1.5, Dream-Instruct

**Key findings**:
- Linear-AcT (full affine) outperforms Gaussian OT (2nd-order) which outperforms Mean-AcT (1st-order)
- +6.5 to +11.9 points instruction-following accuracy
- First OT-guided steering for dLLMs — confirms OT framing generalizes beyond autoregressive models

**Relevance**: Shows OT steering is architecture-agnostic. The richer the moment matching (1st → 2nd order), the better — consistent with CHaRS's finding that multi-modal structure matters. **Supports**: OT framing is general. **Extends**: dLLMs are a new testbed for transport-based methods.

#### 2.4.2 ODESteer (Zhao et al., Feb 2026, arXiv:2602.17560, ICLR 2026)

**Motivation**: CAA/AcT are single-step interventions — they add a fixed vector at one layer. But activation dynamics are continuous through the network. Can we model steering as an ODE and solve it adaptively?

**Method & Experiments**:
- Interpret activation addition as Euler discretization of an ODE: `dh/dl = f(h, l)`
- Steering direction defined via barrier functions: `f(h) = ∇_h log(p_pos(h) / p_neg(h))` — the log-density ratio between positive and negative activation distributions
- Nonlinear polynomial features for the barrier function
- Multi-step adaptive steering via ODE solver (not single-step)
- Evaluation: TruthfulQA, UltraFeedback, RealToxicityPrompts

**Formal Statements & Proofs**:
- *Connection to OT*: The log-density ratio `∇ log(p_pos/p_neg)` is the gradient of a convex function — this is exactly the Brenier map (the unique OT-optimal transport map between distributions with log-concave densities). ODESteer's barrier function IS the OT gradient, discretized as an ODE.
- *Multi-step advantage*: Single-step CAA approximates the ODE with one Euler step — accurate only when the velocity field is nearly constant over the layer span. Multi-step solves the ODE more accurately, following the true transport trajectory.

**Key findings**:
- +5.7% on TruthfulQA, +2.5% on UltraFeedback, +2.4% on RealToxicityPrompts over SOTA baselines
- Multi-step consistently outperforms single-step — the ODE trajectory is not a straight line
- Adaptive step size concentrates computation where the velocity field changes most rapidly

**Relevance**: ODESteer's barrier function is equivalent to the OT gradient of a convex potential (Brenier map). This provides a continuous-time interpretation of why OT methods outperform CAA: CAA uses a single Euler step of the same ODE that ODESteer integrates fully. Our FLAS (§10.1) uses a similar continuous-time framework but with a learned velocity field rather than a density-ratio barrier. **Supports**: multi-step transport is better than single-step. **Connects**: ODESteer's barrier function = OT gradient of convex potential = analytic form of what AcT approximates numerically.

#### 2.4.3 MidSteer: Optimal Affine Concept Steering (Gaintseva et al., Apr 2026, arXiv:2605.05220)

**Motivation**: Existing concept steering methods (CAA, AcT, LEACE) lack provable optimality guarantees. How much "collateral damage" (changes to non-target attributes) does each method cause?

**Method & Experiments**:
- Formalize concept steering as constrained optimization: achieve target concept change while minimizing collateral damage
- Prove standard concept erasure (LEACE) is a special case
- Introduce **LEACE-Switch**: directed concept switching (not just erasure)
- Introduce **MidSteer**: Minimal Disturbance concept Steering — the provably optimal affine intervention

**Formal Statements & Proofs**:
- *Theorem*: MidSteer achieves the target concept shift with provably minimum collateral damage among all affine interventions. **Proof**: Under Gaussian assumptions, the optimal transport map that achieves a target mean shift while minimizing the Frobenius norm of the transport matrix is unique and given by the projected affine map. MidSteer computes this exactly.
- *LEACE equivalence*: LEACE erasure = MidSteer with target mean = 0. The erasure is a special case of the general steering problem.

**Key findings**:
- MidSteer provably minimizes collateral damage vs CAA, AcT, LEACE
- Works across vision diffusion models and LLMs
- LEACE-Switch enables directed concept manipulation (not just removal)

**Relevance**: MidSteer provides the theoretical upper bound for what affine OT methods can achieve — any method that uses a linear/affine transport map cannot do better than MidSteer on collateral damage. If MidSteer still hits the safety ceiling, the ceiling is not about transport quality but about fundamental limitations of affine interventions. **Supports**: affine methods have a provable optimality limit. **Actionable**: test MidSteer on our safety benchmark — if it hits the ceiling, the ceiling is non-affine.

#### 2.4.4 SAKE: Steering Activations for Knowledge Editing (Detyniecki et al., ACL 2025)

**Motivation**: Knowledge editing changes a single fact, but facts are expressed through many paraphrases and logical implications. Editing one prompt doesn't generalize to related prompts.

**Method & Experiments**:
- Model a fact as a distribution of related prompts (paraphrases + implications)
- Use OT to alter activations across the entire fact-related distribution simultaneously
- Evaluate generalization to unseen paraphrases and implications

**Key findings**:
- More robust edits than existing knowledge editing methods
- Generalizes to logical implications and paraphrases that were never seen during editing
- OT over distributions > point-wise intervention

**Relevance**: SAKE demonstrates distributional transport (OT over prompt distributions) for knowledge editing — conceptually similar to CHaRS but applied to factual knowledge rather than behavioral steering. **Supports**: distributional methods outperform point-wise. **Extends**: OT for knowledge editing, not just behavioral steering.

#### 2.4.5 Weight Updates as Activation Shifts (Adila et al., Feb 2026, arXiv:2603.00425)

**Motivation**: When does weight modification = activation addition? Is there a formal equivalence?

**Method & Experiments**:
- First-order Taylor equivalence: weight update `ΔW` at layer l produces activation shift `Δh ≈ W · Δh_input`
- Derive conditions where activation steering replicates fine-tuning
- Joint adaptation: train in both weight and activation space simultaneously

**Formal Statements & Proofs**:
- *Equivalence theorem*: Post-block output is the optimal intervention site — small weight changes produce large, predictable activation shifts there.
- *Joint adaptation theorem*: Weight + activation training exceeds either alone — weight modification shifts the manifold, activation modification provides a temporary signal.

**Key findings**: Within 0.2–0.9% of full fine-tuning while training only 0.04% of parameters. Joint adaptation often exceeds either alone.

**Relevance**: Provides theoretical grounding for why weight modification (WeightSteer) bypasses the ceiling — it operates at the "optimal intervention site" identified by the equivalence theorem. **Supports**: weight + activation joint adaptation is theoretically justified. **Connects**: the Taylor equivalence explains the empirical gap between activation methods (0–21% safety) and weight methods (62–81% Evil).

---

## 3. Extended OT and Distribution Optimization

### 3.1 CHaRS (Concept Heterogeneity and Representation Steering)

**Motivation**: Concepts are not unimodal — the same concept ("refusal") fragments into different clusters depending on context (refusal of harm vs refusal of request vs polite decline). Single-Gaussian OT is inadequate.

**Method & Experiments**:
- Model source/target as Gaussian Mixture Models (GMMs) instead of single Gaussians
- Solve discrete OT between mixture components (Sinkhorn algorithm)
- Inference: barycentric projection — weighted sum over centroid pairs
- Weights from kernel membership (RBF kernel, which component does each sample belong to?)

**Formal Statements & Proofs**:
- *Concept heterogeneity theorem*: Unimodal assumption leads to suboptimal transport. **Proof**: If the true distribution is multi-modal (K>1 components), a single Gaussian collapses K modes into one, producing a transport map that averages across modes rather than routing each sample to its correct target mode.
- *Barycentric projection*: Preserves distributional structure better than point-wise transport. **Proof**: Each sample x gets a weighted combination of all target centroids via Sinkhorn coupling P*. The weights are the kernel membership — samples close to centroid i get more of target centroid i's shift. This preserves the multi-modal structure.

**Key finding**: CHaRS handles multi-modal concepts where CAA/AcT fail. Better on heterogeneous datasets.

**Our finding**: CHaRS gets 85% on Deception (concept is homogeneous, K=3 active centroids) but 0% on Toxic/Evil (K=10/20 active centroids but ceiling). The ceiling is not about concept heterogeneity. The barycentric projection collapses to ≈ CAA on Toxic (cos 0.95–0.98 with CAA vector). **Supports** (for Deception). **Opposes** (for safety tasks — multi-modality doesn't help).

### 3.2 STARS (Steering with Targeted Activation Re-ranking and Shifting)

**Motivation**: Single direction cannot capture concept heterogeneity. Multiple orthogonal directions maximize coverage.

**Method**: Multiple steering vectors, constrained to be orthogonal (Stiefel manifold), maximize geometric volume of parallel generation paths.

**Key finding**: Multi-directional > single-direction for complex behaviors.

**Relevance**: We tested multi-layer CAA (18 layers) — it showed KL growth (×2.46) but 0% accuracy. More directions don't help if the ceiling mechanism is present. **Opposes** (more directions don't break ceiling).

### 3.3 CHARS K-Sweep & Coupling Matrix Analysis (Our Experiments, §9.4, §9.15)

**Motivation**: Determines if CHARS's 0% on safety is caused by RBF kernel degeneracy or a genuine ceiling.

**Method**: Extract CHARS centroids at K=3,5,10,20,50 for Toxic, Evil, Deception, Refusal at layer 14. Examine RBF kernel values, Sinkhorn coupling matrix P*, active centroid count, coupling mass distribution.

**Key findings**:
- RBF kernel NOT degenerate: values 0.00–0.98 for Toxic K=10 (τ=90.9)
- High-norm centroids produce MORE specific transport (entropy 0.38 vs 0.76)
- Symlog tail annealing reduces centroid CV 0.28→0.06 but accuracy stays 0%
- Multi-centroid clustering is orthogonal to accuracy: Evil 20/20 active, 0–3% accuracy; Deception 2/10 active, 85%
- **Conclusion**: The safety ceiling is binding regardless of kernel quality, cluster count, or transport algorithm. The problem is not in the transport — it's in the model's response to ANY activation-level perturbation on safety tasks.

### 3.4 OT Ablation: Sorted Quantile vs Gaussian (Our Experiment, §9.20)

**Motivation**: Isolates whether the transport coupling is the causal factor behind OT methods' Pareto advantage.

**Method**: Standard ACT uses sorted quantile matching (full empirical distribution). Gaussian ablation uses ω = σ_dst/σ_src (assumes Gaussian per-dim). CHARS ablation uses λ=100.0 (near-uniform Sinkhorn coupling).

**Key findings**:
- **ACT Gaussian is strictly worse on all tasks.** Sorted quantile matching is the causal mechanism: 95% → 50% on Deception, 85% → 35% on Toxic
- **CHARS high-lambda (near-uniform coupling) IMPROVES Toxic** (34% vs 2%). The Sinkhorn transport plan is NOT on the critical path for CHARS
- **Evil ceiling remains unbroken**: No ablation exceeds 9% clean

**Conclusion**: Sorted quantile coupling is causal for ACT's advantage. The Sinkhorn plan is NOT causal for CHARS. No transport simplification breaks the safety ceiling. **Supports**: transport algorithm is secondary to the ceiling.

---

## 4. Multi-Behavior Steering: What Can We Steer?

### 4.1 "What Can We Actually Steer?" (Bas & Novak, 2025, arXiv:2511.18284)

**Motivation**: Prior work cherry-picked behaviors favorable to the proposed method. Systematic evaluation across diverse behaviors is needed.

**Method & Experiments**:
- **50 behaviors** (5 categories: persona, personality, misalignment, style, impersonation)
- Llama 3.1-8B
- CAA extraction, coefficient sweep across all behaviors

**Key findings with formal reasoning**:
1. **Inverted-U curve universal**: Low coefficient under-steers, moderate succeeds, high collapses. *Reasoning*: The model's activation manifold has finite curvature. Linear translations push off-manifold at high coefficients, producing degenerate outputs. The optimal coefficient is where the perturbation stays on-manifold — this is behavior-dependent because different behaviors have different manifold curvatures.
2. **Vector separation fails**: Cosine similarity + norm between positive/negative differences do NOT predict success. *Reasoning*: Cosine similarity measures geometric alignment in ambient space, but the causal question is whether the direction intersects the behavior's manifold. A direction can have high cosine with the "true" direction yet miss the manifold entirely if it's slightly off-axis in a high-curvature region.
3. **Data scaling helps**: Larger datasets enable stronger steering without OOD. *Reasoning*: More samples → better mean estimate → cleaner direction → smaller off-manifold residual.
4. **Behavior type matters**: Style/persona easier; misalignment (deception, manipulation) harder. Optimal coefficient varies by factor of 5 across behaviors.

**Formal claims**:
- Steerability is behavior-dependent, not method-dependent
- Simple diagnostics (vector norm, cosine) are unreliable predictors

**Our finding**: We confirm the inverted-U. Their "harder" category (misalignment) is where we see the ceiling. Our Deception task (94%) is in their "easier" category. **Supports**: behavior-dependence. **Extends**: we show the asymmetry is not just "hard vs easy" but specifically about RLHF-enforced safety vs pre-training emergent features.

### 4.2 ASTEER / SteerBoost (Fan et al., 2026, arXiv:2606.11599)

**Motivation**: Most steering attempts fail silently. Predicting success before committing to full generation would save compute and reveal what makes steering work.

**Method & Experiments**:
- **1.4M steered generations**
- 150 concepts, 50 prompts, 3 LLMs
- 2 methods (DiffMean, LinearProbe), 45 coefficients each
- GBDT (SteerBoost) trained on first few hidden states to predict success

**Key findings**:
1. **Steering success rate only 10–23%** — directly corroborates our safety ceiling
2. **Early hidden states predict success**: GBDT on first few tokens → ~0.7 macro-F1 unseen, ~0.8 seen
3. Recovers ~98% of exhaustive grid search at ~11% decoding cost
4. **Low-level concepts harder than high-level** (formatting/emojis harder than persona/topic)

**Formal claims**:
- Early-layer activations contain sufficient information to predict steering outcome
- Most steering attempts fail — the success stories in literature are selection-biased

**Our finding**: Their 10–23% success rate is exactly our ceiling range across 150 concepts, not just safety tasks. The early-state prediction is consistent with our KL decay pattern — early layers carry the signal. **Supports**: the ceiling is general, not safety-specific. **Extends**: our contribution is showing the ceiling bifurcates — 94% for Deception, 0–21% for safety — which their paper doesn't distinguish.

### 4.3 LAP — Linear Accessibility Profile (Billa, 2026, arXiv:2604.15557)

**Motivation**: Training-free prediction of steering effectiveness without running any steering.

**Method & Experiments**:
- **Linear Accessibility Profile (A_lin)**: logit lens applied to each layer — measures how well a linear map from activations to logits explains the model's output
- Peak A_lin predicts steering success: ρ = +0.86 to +0.91 across 5 models
- Layer selection: ρ = +0.63 to +0.92

**Three-regime framework** (formal reasoning):
1. **Linearly Accessible** (high A_lin): The concept is already linearly decodable from activations → diff-of-means works
2. **Nonlinearly Encoded** (high A_mlp − A_lin gap): The concept exists but is encoded nonlinearly → nonlinear methods needed
3. **Inaccessible** (low both): No steering can work

**Our finding**: Consistent with our geometry analysis. Safety tasks likely fall in regime 2 (nonlinearly encoded) — the linear subspace exists but output-stage processing cancels it. Deception is regime 1 (linearly accessible). **Supports**: three-regime theory aligns with our task-dependent ceiling.

### 4.5 Multi-Attribute Steering

While §4.1–4.3 study what CAN be steered (single attributes), a parallel thread asks: can we steer MULTIPLE attributes simultaneously? The core challenge is interference — steering toward attribute A may degrade attribute B. This section surveys 2025–2026 methods for multi-attribute control.

#### 4.5.1 MAT-Steer: Multi-Attribute Steering via Targeted Intervention (Nguyen et al., ACL 2025, arXiv:2502.12446)

**Motivation**: Single-attribute steering vectors conflict when composed — adding vectors for "be polite" and "be truthful" may cancel each other or produce degenerate outputs. How to steer multiple attributes without interference?

**Method & Experiments**:
- Learn per-attribute steering vectors via an alignment objective that shifts undesirable representations toward desirable ones
- **Sparsity + orthogonality constraints** on attribute vectors to reduce inter-attribute conflicts
- Learned **per-token gating functions** to determine which attributes intervene at which positions
- Evaluation: TruthfulQA, Toxigen, BBQ (3 attributes simultaneously)

**Key findings**:
- 3% avg accuracy gain across QA tasks; 55.82% win rate against best ITI baseline on generative tasks
- Requires <20% of training data vs fine-tuning baselines
- Sparsity + orthogonality is critical — without constraints, multi-attribute composition degrades 15–20%

**Multi-attribute approach**: Orthogonal sparse per-attribute vectors + learned per-token gating. The orthogonality constraint ensures attribute vectors don't interfere in activation space. The gating determines WHERE each attribute is active (not all tokens need all attributes).

**Relevance to us**: MAT-Steer's orthogonality constraint is the most principled approach to multi-attribute steering in activation space. However, it requires per-attribute training data and assumes attributes can be made orthogonal — which fails when attributes share neurons (>95% overlap per §8.11). **Supports**: orthogonality reduces interference. **Limitation**: assumes attribute separability, which safety/helpfulness entanglement violates.

#### 4.5.2 K-Steering: Beyond Linear Steering (Oozeer et al., EMNLP 2025 Findings, arXiv:2505.24535)

**Motivation**: Linear steering vectors suffer from the "dilution effect" — averaging multiple vectors reduces each attribute's strength. Can a nonlinear approach avoid this?

**Method & Experiments**:
- Train a **single non-linear multi-label classifier** on hidden activations
- Compute intervention directions via **gradients at inference time** — no stored per-attribute vectors
- Dynamic composition of behaviors without retraining

**Key findings**:
- Outperforms baselines on ToneBank and DebateMix benchmarks across 3 model families
- Resists the dilution effect of averaging multiple vectors
- Single classifier handles all attributes — no per-attribute vector storage

**Multi-attribute approach**: Gradient-based composition from a unified classifier. At inference, the gradient of the classifier w.r.t. the activation gives the intervention direction for any desired attribute combination. No stored vectors, no orthogonality assumptions.

**Relevance**: K-Steering's gradient-based approach avoids the orthogonality assumption entirely — it computes task-specific directions on-the-fly. This is more flexible than MAT-Steer but requires training a classifier. **Supports**: nonlinear composition avoids dilution. **Open**: does it work for safety attributes where the classifier itself may be miscalibrated?

#### 4.5.3 MSRS: Adaptive Multi-Subspace Representation Steering (Jiang et al., 2026, arXiv:2508.10599)

**Motivation**: Per-attribute vectors in the full activation space waste budget on irrelevant dimensions. Can we allocate separate subspaces to each attribute?

**Method & Experiments**:
- Allocate **orthogonal subspaces** to each attribute via SVD
- Hybrid composition: attribute-specific subspaces for unique directions + a **shared subspace** for common directions
- Dynamic weighting function learns to integrate components
- Token-level steering targets semantically relevant tokens

**Key findings**:
- Significant reductions in attribute conflicts
- +13% TruthfulQA, +4% BBQ; generalizes to HellaSwag (+3.8%), GLUE (+4.9%)
- Tested on Llama2-7B, Llama3-8B-Instruct, Qwen2-7B-Instruct, Mistral-7B

**Multi-attribute approach**: Orthogonal private subspaces + shared global subspace + dynamic gating. The key insight is that attributes have BOTH unique components (private subspaces) AND shared components (common subspace). For example, "be truthful" and "be polite" share a "be helpful" component.

**Relevance**: MSRS's private/shared decomposition is the most sophisticated subspace allocation method. The shared subspace handles common components (where interference is expected), while private subspaces handle unique components (where interference is avoided). **Supports**: subspace decomposition reduces interference. **Extends**: the shared/private distinction is novel — previous methods treat all attributes as fully independent.

#### 4.5.4 ORBIT: Training-Free Multi-Attribute Steering (Ghasemi et al., Jun 2026, arXiv:2606.22357)

**Motivation**: All prior multi-attribute methods require training. Can multi-attribute steering be done training-free?

**Method & Experiments**:
- Construct a **joint subspace** from per-attribute steering planes via SVD
- Apply a **single norm-preserving rotation** within that subspace toward a combined target direction
- **Adaptive per-token gating** identifies which attributes need correction
- Introduces TraitFactory benchmark for multi-attribute evaluation

**The norm dominance problem (key insight)**:
When composing CAA vectors by summing (`h += α₁v₁ + α₂v₂ + ...`), vectors with different norms compete. ORBIT measured per-trait vector norms across 5 traits on Llama-3.2-3B — norms vary up to **2.8× at layer 7**. Naively summing vectors for "be honest" + "be harmless" + "be helpful" would primarily steer toward whichever attribute has the largest norm vector. This is why CAA's multi-attribute performance degrades at K≥2 — the largest-norm vector dominates, and the others are effectively ignored.

**ORBIT's norm-preserving rotation**: Instead of adding vectors (which compounds magnitude), ORBIT rotates within a joint subspace. The rotation preserves `‖h'‖ = ‖h‖` by construction. This prevents:
- Activation magnitude growing unbounded (causes OOD at high coefficients)
- One attribute dominating via norm (the 2.8× variance problem)
- The model being pushed off its natural activation manifold

**The tradeoff**: Norm-preservation means ORBIT can only change direction, not magnitude. For tasks where you need a large activation shift (e.g., pushing from "harmless" to "harmful"), the inability to increase norm could be limiting. ORBIT adds an optional "boost" component (`α_boost · Σ g_k δ_k`) to partially address this — but boost reintroduces norm growth.

**Other norm findings in the literature**:
- **OV-Circuit Steering (§8.10)**: Steering vectors are 90-99% sparse in the OV pathway — most norm is wasted on irrelevant dimensions
- **FishBack (2026)**: Euclidean norm (which most methods preserve) is the wrong metric — Fisher information metric is 97% different. "Norm preservation" in Euclidean space may not preserve the information that matters.
- **Suppressor-Crystallizer (§8.1)**: Final-layer bottleneck compresses representations, potentially constraining how much norm can be preserved through the network

**Key findings**:
- Stronger and more balanced multi-attribute steering than training-free baselines on TraitFactory and ToneBank
- Better preserves output coherence than vector addition
- Works on Llama-3.2-3B, Qwen-2.5-7B, Llama-3.1-8B

**Multi-attribute approach**: Joint subspace construction + single norm-preserving rotation + per-token adaptive gating. The rotation preserves the norm of the activation (preventing OOD), while the joint subspace ensures attributes don't interfere. The per-token gating determines which attributes are active at each position.

**Relevance**: ORBIT is the first training-free multi-attribute method. The norm-preserving rotation is particularly relevant to our work — it avoids the OOD problem that plagues high-coefficient CAA. If ORBIT can steer multiple safety attributes without training, it could be a practical alternative to our pipeline. **Supports**: training-free multi-attribute is possible. **Open**: does the norm-preservation prevent the activation magnitude needed for safety task breakthrough?

#### 4.5.5 Compositional Steering via Steering Tokens (Radevski et al., ACL 2026, arXiv:2601.05062)

**Motivation**: Activation-space composition (adding vectors) suffers from interference. Can we compose behaviors in INPUT space instead?

**Method & Experiments**:
- Embed individual behaviors as natural-language instructions into dedicated tokens via **self-distillation in input space**
- Train a dedicated `<and>` composition token on behavior pairs
- The composition token **generalizes to unseen compositions**, unseen behaviors, and unseen numbers of composed behaviors

**Key findings**:
- Superior multi-behavior steering on verifiable constraints (length, format, structure, language)
- Complements natural language instructions when combined
- Zero-shot generalization to unseen attribute combinations

**Multi-attribute approach**: Input-space composition via learned `<and>` token. Unlike activation-space methods, this operates on the INPUT — each behavior is a token, and composition is token concatenation. The `<and>` token learns to mediate between behavior tokens.

**Relevance**: This is fundamentally different from activation-space methods — it operates in input space, not activation space. The zero-shot generalization to unseen compositions is remarkable. However, it requires verifiable constraints (length, format) — it's unclear if it works for abstract safety attributes. **Supports**: input-space composition avoids activation-space interference. **Limitation**: verifiable constraints only; safety attributes are not verifiable.

#### 4.5.6 SSAE: Sparse Shift Autoencoders for Concept Identification (Joshi et al., ICML 2025 WS, arXiv:2502.12179)

**Motivation**: Standard SAEs learn features from individual activations, but individual activations contain ALL concepts superimposed. Can we learn disentangled concept representations by autoencoding the DIFFERENCE between paired observations?

**Method & Experiments**:
- Given paired text (x, x̃) that differ in some unknown subset of concepts, compute δz = f(x̃) − f(x)
- Train a **standard SAE architecture** on these difference vectors: encoder maps δz → sparse code c, decoder maps c → reconstructed δz. The only difference from a standard SAE: training data is difference vectors δz instead of individual activations z.
- Sparsity constraint on the CODE (not the activations) ensures each code dimension captures a single concept *shift* rather than a single concept *presence*
- Decoder columns become steering vectors — column k steers concept k

**Formal Statements & Proofs**:
- *Identifiability theorem (Prop 2)*: Under the Linear Representation Hypothesis (z = Ac) and sparsity on concept shifts (not activations), the learned decoder columns are identifiable up to permutation and scaling. This is a stronger guarantee than standard SAEs, which do not have identifiability.

**Key findings**:
- On correlated concepts (CORR(2,1) — two language shifts, only one varies per pair): SSAE MCC=0.905 vs standard SAEs ~0.5. This is where identifiability matters most — when concepts are correlated, SAEs fail catastrophically.
- On simple concepts (single word shifts): marginal improvement over SAEs — all methods do reasonably well.
- Tested on Gemma-2-2B (primary), Pythia-70M, Llama-3.1-8B (layer analysis only)

**What SSAE is NOT**: SSAE is a concept identification method, not a multi-attribute steering method. It identifies disentangled concept directions but does not compose them. To do multi-attribute steering, you would need to combine SSAE with a composition strategy (e.g., ORBIT's rotation, MAT-Steer's gating).

**Relevance**: SSAE's identifiability guarantee is relevant to our work — if safety concepts can be identified without contrastive pairs, it bypasses the CAA extraction problem. But the evaluation is too small (single-word pairs) to confirm this works for sentence-level safety concepts. **Supports**: sparsity enables unsupervised disentanglement. **Open**: does the sparsity assumption hold for safety concepts that are entangled with helpfulness (>95% overlap per §8.11)?

#### 4.5.7 MidSteer: Minimal Disturbance Multi-Concept Steering (Gaintseva et al., Apr 2026, arXiv:2605.05220)

**Motivation**: How much collateral damage does multi-attribute steering cause? Is there a provably optimal way to steer multiple attributes?

**Method & Experiments**:
- Formalize multi-concept steering as constrained optimization: achieve target shifts for ALL attributes while minimizing total collateral damage
- Prove LEACE-Switch as a special case
- MidSteer achieves provably minimum disturbance among all affine interventions

**Formal Statements & Proofs**:
- *Multi-concept optimality theorem*: Among all affine interventions that achieve the target shifts for K attributes simultaneously, MidSteer minimizes the Frobenius norm of the intervention matrix. **Proof**: The constrained optimization has a unique solution given by the projected affine map — this is the multi-concept extension of the single-concept result.

**Key findings**:
- MidSteer provably minimizes collateral damage for multi-concept steering
- LEACE-Switch enables directed concept switching (not just removal)
- Works across vision and language models

**Relevance**: MidSteer provides the theoretical upper bound for affine multi-attribute methods. If MidSteer still causes significant collateral damage on safety attributes, the ceiling is fundamental to affine interventions. **Supports**: affine methods have provable limits. **Actionable**: test MidSteer multi-concept on our benchmark.

#### 4.5.8 Empirical Findings: Interference is Pervasive

Several papers provide empirical evidence that multi-attribute steering is fundamentally hard:

1. **"Steering Control" (2025, arXiv:2509.13450)**: Standardized evaluation reveals NO method achieves clean orthogonal control. Behavioral entanglement is broad and pervasive — even between seemingly unrelated attributes (sycophancy and anthropomorphism).

2. **"What Can We Actually Steer?" (Bas & Novak, 2025)**: Across 50 behaviors, steering effectiveness varies by factor of 5 across behaviors. Optimal coefficients are behavior-specific — no universal coefficient works.

3. **"Do Personality Traits Interfere?" (2026)**: Studies geometric limitations of multi-personality steering. Shows linear representation assumption breaks down for polysemantic concepts — the SAME neurons encode multiple traits, making orthogonal subspace allocation impossible.

4. **PsySET (Banayeeanzade et al., ACL 2026, arXiv:2510.04484)**: Evaluates steering of multiple psychological attributes simultaneously. Reveals idiosyncratic side-effects (e.g., steering "joy" degrades adversarial factuality robustness). Trustworthiness side-effects of joint steering are unpredictable.

**Synthesis**: The multi-attribute steering landscape has three distinct approaches:
- **Orthogonal subspace** (MAT-Steer, MSRS, ORBIT): Allocate separate subspaces per attribute. Works when attributes are truly independent but fails when they share neurons (>95% overlap for safety/helpfulness).
- **Gradient-based composition** (K-Steering): Compute task-specific directions on-the-fly. More flexible but requires training data or a classifier.
- **Input-space composition** (Compositional Tokens): Avoid activation-space interference entirely. Works for verifiable constraints but unclear for abstract attributes.

**The fundamental tension**: Safety and helpfulness share >95% of neurons (§8.11). No activation-space method can separate them because they are NOT separable in activation space — they live in the same neurons with different activation patterns. This is why:
- Orthogonal subspace methods fail: you can't allocate orthogonal subspaces to entangled attributes
- Gradient-based methods fail: the gradient for "be safe" and "be helpful" points in the same direction
- Input-space methods fail: safety attributes are not verifiable constraints

**The only bypass**: Weight modification (WeightSteer) changes the neuron's activation pattern without changing which neurons are active — it operates at a finer granularity than activation-space methods.

#### 4.5.9 Critical Synthesis: What Every Paper Gets Wrong

**Cross-paper methodology assessment:**

| Method | Architecture breadth | Scale | Safety task eval? | Mechanism analysis? | Evaluation method | Methodology grade |
|--------|---------------------|-------|-------------------|---------------------|-------------------|-------------------|
| MAT-Steer | 2 families (Llama, Qwen) | 7-8B only | Zero | Token gating only | GPT-4o judge | Moderate |
| K-Steering | 3 families (Llama, Mistral, OLMo) | 3-7B only | Zero | Zero | GPT-4o-mini judge + circular classifier eval | Weak |
| MSRS | 3 families (Llama, Qwen, Mistral) | 7B only | Zero | Zero | GPT-4o judge | Moderate |
| ORBIT | 2 families (Llama, Qwen) | 3-8B only | Zero (fake traits) | Geometry only | GPT-4o-mini + Claude cross-judge | Moderate |
| MidSteer | 2 families + diffusion | 7-14B | Modest (RTP toxicity) | Zero | Llama-3.1-8B judge | Moderate |
| SAKE | 2 families (GPT2, Llama) | 1.5-7B (BASE only!) | Zero | Zero | Exact-match token comparison | Moderate |
| Compositional Tokens | 4 families (Qwen, Llama, SmolLM, OLMo) | 3-14B | Zero | Nearest-neighbor only | Automatic verification | Strong (1M+ evals) |
| SSAE | 3 families (Gemma, Pythia, Llama) | 70M-8B | Zero | Identifiability proof | MCC + cosine similarity | Strong (theory-grounded) |

**Seven universal weaknesses across all papers:**

1. **No testing on tasks where the ceiling is known.** Every paper evaluates on style, format, tone, or personality traits — tasks where activation steering already works reasonably well. Nobody tests on tasks where the accuracy ceiling is most pronounced (0-21% on safety tasks with activation steering). The multi-attribute methods are untested on the hard case.

2. **Linearity assumption varies by method but is rarely tested.** The distinction matters: some methods assume the *steering direction* is linear (CAA, MAT-Steer's vector addition), while others assume the *concept representation* is linear (SSAE's LRH). K-Steering explicitly rejects both — it uses a nonlinear MLP classifier. ORBIT uses norm-preserving rotation (nonlinear intervention). Compositional Tokens operates through the model's own nonlinear forward pass. The papers that DO assume linearity (MidSteer, SAKE) frame it as "second-order statistics suffice" rather than strict LRH. None of the papers that assume linearity **test whether the assumption holds** for their target concepts.

3. **Layer selection is always hand-tuned.** Every paper picks a layer by dev-set grid search. No paper asks: *which layers matter for multi-attribute steering?* This matters because different attributes may live at different layers.

4. **All evaluation uses LLM-as-judge for behavioral tasks.** GPT-4o-mini or GPT-4.1-mini as judge is universal for style/tone/personality evaluation. SSAE and Compositional Tokens use automatic evaluators (MCC, exact-match), which are more reliable. For benchmark tasks with ground truth (accuracy), automatic evaluation is preferred.

5. **CAA is consistently the weakest baseline.** Every paper beats CAA. But nobody tests whether the gains survive on tasks where CAA gets 0-21% (safety) vs. 85-94% (deception). Beating a method that already fails is uninformative about whether the new method addresses the hard case.

6. **No paper analyzes WHY composition works mechanistically.** Zero head-level analysis, zero circuit tracing, zero ablation of model components. All papers are purely behavioral — they measure output quality but never ask what happens inside the model.

7. **No paper tests on Gemma-2-2b-it** (our target model where the safety ceiling is most pronounced).

**Method-specific critical notes:**

- **MAT-Steer**: The gating is a single dot product + sigmoid (linear decision boundary). The steering addition is linear. But the training loss (MMD in RKHS) is nonlinear — so the learned vectors may capture nonlinear structure. CAA is not a baseline despite being cited. The 3-5% accuracy improvements are modest. Norm preservation is a strong, unproven constraint.

- **K-Steering**: Explicitly nonlinear — 3-layer MLP classifier, gradient-based intervention. The evaluation metric is circular — the activation classifier used for evaluation has the same architecture as the K-Steering classifier. Synthetic data only (GPT-4o-mini generated). At K=3, all scores drop to near zero — the method doesn't scale.

- **ORBIT**: Nonlinear intervention (norm-preserving rotation + ReLU gate). The norm dominance finding is the most important empirical contribution in the multi-attribute literature. All-layer brute-force intervention (every LayerNorm output). TraitFactory benchmark is their own creation. Boost coefficient requires per-combination calibration — undermining "parameter-free" claims.

- **SSAE**: Sparse shift autoencoder struggles with interpretability. Single-word evaluation only — need to test on sentences and more complex tasks.

- **MidSteer**: Fully affine (closed-form). Assumes second-order statistics (mean + covariance) suffice — weaker than strict LRH but still a distributional assumption. Only tests single-concept switching (not multi-attribute). Modest absolute numbers (toxicity 0.371→0.281). Missing key baseline: Representation Surgery (Singh et al., ICML 2024) does essentially the same thing.

- **SAKE**: Linear OT mapping (affine transform) assuming Gaussian distributions. Knowledge editing, not behavioral steering. Per-edit cost is ~100 forward passes + GPT-4 API calls. Only tested on base models (no RLHF alignment). Scope detection threshold is model/dataset-specific.

- **Compositional Tokens**: Gains over instruction steering are modest (+5.1% on hardest task). CAA/LM-Steer are straw-man baselines (designed for soft preferences, not hard constraints). Order variance of 15-25% suggests the `<and>` token is NOT learning abstract composition.

**The fundamental tension (unchanged from §4.5.8)**: Safety and helpfulness share >95% of neurons (§8.11). No activation-space method can separate them because they are NOT separable in activation space. The only bypass is weight modification (WeightSteer).

---

## 5. Control-Theoretic Steering

### 5.1 PID Steering (Nguyen et al., ICLR 2026, arXiv:2510.04309)

**Motivation**: Existing methods (CAA, AcT) are proportional controllers — fixed gain on current error. No integral or derivative terms. Control theory has 100+ years of solutions for this.

**Method & Experiments**:
- Layer-to-layer activation as discrete-time dynamical system:
  `e(k+1) = A(k)·e(k) − A(k)·u(k) + w(k)`
- **P control** (CAA/AcT): `u(k) = K_p · e(k)`. Input-to-state stable but steady-state error.
- **PI control**: `u(k) = K_p · e(k) + K_i · s(k)`. Cancels bias but causes overshoot.
- **Full PID**: `u(k) = K_p · e(k) + K_i · s(k) + K_d · Δe(k)`. Derivative damps overshoot.
- 7 models, 3 families (Gemma-2, Llama-3.1, Qwen-2.5)
- Jailbreaking ASR as primary metric

**Formal Statements & Proofs**:
- *Theorem 1*: P-only control has bounded steady-state error. **Proof**: Under ISS (Input-to-State Stability), P-control `u = K_p · e` produces `e(k+1) = (A − K_p·A)·e(k) + w(k)`. If `‖(I − K_p)·A‖ < 1`, the system is ISS and `e` converges to a bounded neighborhood of zero proportional to `‖w‖`. The steady-state error is non-zero because P-control only reacts to current error, not accumulated error.
- *Theorem 2*: PI cancels steady-state error but introduces overshoot bounded by K_i. **Proof**: The integral term accumulates past errors: `s(k) = Σ e(j)`. At steady state, `s` grows until `K_i · s` exactly cancels the bias. But during transient, `s` overshoots, producing `u` larger than needed. The overshoot is proportional to K_i.
- *Theorem 3*: PID achieves ISS with reduced overshoot. **Proof**: The derivative term `K_d · Δe(k)` opposes rapid changes in error, damping the oscillation caused by the integral term. Combined ISS + overshoot bound.

**Key findings**:
- **Jailbreaking ASR 76–96%** across all models
- Outperforms DIM, ITI, RepE on every model
- General benchmarks within 1–3% of baseline
- Stability: K_i in (−0.23, 0.23) for Gemma-2-9B

**Our finding**: **PID partially breaks our ceiling**: Toxic c=1.0 → 45% clean (§9.12). The I-term accumulates error across layers, partially overwhelming the hypothesized suppression. But: narrow gain window, higher coefficients OOD (Evil 68% accuracy at c=1.0 but 48.5% repetition rate). PID KL is erratic (spikes L11/L13/L15). **Supports**: I-term accumulation is a ceiling-breaking mechanism. **Opposes**: PID is a practical solution (too fragile).

### 5.2 A-LQR (Skifstad, Yang, Chou, 2026, arXiv:2604.19018)

**Motivation**: Despite transformer nonlinearity, layer-wise dynamics are locally linear. Classical LQR (Linear Quadratic Regulator) can synthesize optimal feedback controllers.

**Method & Experiments**:
- Linearized dynamics per layer: `δz_{k+1} ≈ A_k · δz_k + B_k · u_k`, with B_k = I
- LQR cost: `J = z_T^T Q_f z_T + Σ(k=1)^{T-1} (z_k^T Q z_k + u_k^T R u_k)`
- Riccati recursion yields optimal feedback gain K_k
- Online: `u_k* = −K_k · δz_k`

**Formal Statements & Proofs**:
- *Theorem*: LQR-optimal control achieves minimum-cost steering under linearized dynamics. **Proof**: Standard LQR result — Riccati recursion produces the unique optimal state-feedback policy for linear dynamics with quadratic cost. The optimal cost is `J* = z_0^T P_0 z_0` where P_0 is the solution to the discrete-time algebraic Riccati equation.
- *Local linearity justification*: Jacobian subspace similarity across reachable activations (sim_m ≈ 0.5). **Proof**: Compute Jacobian J_k = ∂z_{k+1}/∂z_k at multiple sampled activations. The principal angles between Jacobian subspaces at different activations are small (cos > 0.5), confirming local linearity.

**Key findings**:
- SOTA toxicity/truthfulness/refusal modulation across scales
- Training-free closed-loop control: offline Jacobian + Riccati, online only precomputed K_k application
- A-LQR+ (all-token intervention) produces mechanistic jailbreaks matching PID

**Our finding**: A-LQR is theoretically the strongest — optimal control vs PID's heuristic gains. We haven't implemented it yet. Its multi-layer Riccati recursion may bypass the ceiling by optimally distributing correction across layers. **Supports** (theoretical optimality), **open question** (does it break our ceiling?).

---

## 6. Shallow Alignment Theory

### 6.1 "Why Is RLHF Alignment Shallow?" (Young, 2026, arXiv:2603.04851)

**Motivation**: Empirically, RLHF alignment is brittle — few-shot fine-tuning can undo months of safety training. Why? Is this a training failure or an inherent limitation?

**Method & Experiments**: Pure theory — martingale framework for harm generation, gradient analysis of RLHF/DPO objectives.

**Formal Statements & Proofs**:

- *Definition*: **Harm as martingale**: `H_t = E[H | y_{≤t}]`. `E[H_{t+1} | y_{≤t}] = H_t`. This means the expected harm at each step is a martingale — past tokens don't bias future harm upward or downward in expectation.

- *Definition*: **Harm Innovation**: `Δ_t = H_t − H_{t-1}`. The change in expected harm at position t.

- *Definition*: **Harm Information**: `I_t = Var[Δ_t | y_{<t}]`. How much variance in harm innovation remains at position t.

- *Theorem 8 (Per-position gradient)*: `G_t = Cov[E[H | y_{≤t}], s_t(y_t | y_{<t})]`. The gradient at position t is proportional to the covariance between the total expected harm and the score function at that position.

- *Theorem 10 (Zero Gradient Theorem)*: If `I_t = 0`, then `G_t = 0`. **Proof**: If `I_t = Var[Δ_t | y_{<t}] = 0`, then `Δ_t = 0` almost surely, meaning `H_t = H_{t-1}` deterministically. The total harm is fixed by the first t-1 tokens. The score function `s_t` at position t cannot influence `H` because `H` is already determined. Therefore `Cov[H, s_t] = 0` and the gradient is zero.

- *Definition*: **Harm horizon** `t_H = min{t: I_t = 0}`. Past this position, RLHF/DPO gradient is exactly zero — no amount of data or compute can produce alignment at these positions.

- *Equilibrium KL*: `D_KL^{(eq)}(t) = O(λ² I_t)`. The equilibrium KL divergence at position t is proportional to the harm information at that position. Past the harm horizon, KL = 0 — the model's output distribution at these positions is unconstrained by RLHF.

**Key conclusion**: Shallow alignment is **optimal under standard RLHF/DPO** — not a training failure. No amount of data/compute can produce deep alignment under standard objectives. The objective is blind past the harm horizon.

**Proposed solution — Recovery Penalties**: Add penalty at all positions for failing to produce recovery tokens. Gibbs measure: `π*(y_t | y_{<t}) ∝ π_base(y_t | y_{<t}) · exp(μ · r_t)`.

**Caveats**: Single-author, no empirical validation of recovery penalties. Output-level only — doesn't explain activation-level phenomena.

**Our finding**: **This is the strongest theoretical explanation for our ceiling.** RLHF modifies only the first ~5–10 tokens (harm horizon). The model is trained to detect and cancel activation perturbations in this window. Safety tasks (Toxic/Evil) are exactly the tasks where RLHF is active → suppression. Deception has no RLHF enforcement → no suppression → 94% steerability.

**Supports**: safety ceiling is a fundamental limitation of RLHF, not a training failure. **Supports**: why weight modification (WeightSteer) bypasses it — it modifies parameters, not the activation perturbation that RLHF is trained to suppress.

### 6.2 Qi et al. (2024, arXiv:2406.05946) — Empirical Precursor to Young

**Motivation**: Empirically validate shallow alignment by measuring per-token KL concentration.

**Method & Experiments**:
- HEx-PHI dataset (330 harmful instructions + GPT-3.5 harmful answers)
- Llama-2-7B-Chat and Gemma-7B
- Per-token KL analysis during fine-tuning

**Key findings with verbal proofs**:
1. **KL concentrates on first ~5 tokens**: Per-token KL measurement shows KL ≈ 0 for positions > 5. *Reasoning*: The fine-tuning gradient only modifies the output distribution at positions where the score function has non-zero covariance with the harm signal. Past the harm horizon, this covariance is zero.
2. **Prefilling attacks**: 5–10 non-refusal tokens → ASR near 0% to >50%; 40 tokens → 57%. *Reasoning*: Inserting benign tokens before a harmful request shifts the KL away from the safety-critical window. The safety-trained output distribution at early positions is already "used up" on the benign prefix.
3. **Fine-tuning attacks**: baseline 1.5% → 22.4% (2 steps) → 76.4% (4 steps) → 87.9% (6 steps). Gradient concentrates on first ~5 tokens. *Reasoning*: Each fine-tuning step maximally shifts the first few positions, progressively overriding the safety alignment.
4. **Defense**: Set β_{1..5} = 2.0, β_{>5} = 0.1 → ASR drops 88.9% → 4.6%. *Reasoning*: Per-weighting the KL penalty at safety-critical positions prevents the fine-tuning gradient from dominating there.

**Our finding**: We replicate the KL concentration finding but with a twist — **per-token KL is NOT early-concentrated on safety tasks for steering** (§9.17). The concentration appears for fine-tuning but not activation steering. This suggests the suppression mechanism for activation steering operates through a different channel from fine-tuning KL concentration. **Supports**: shallow alignment exists. **Opposes** (partially): the per-token KL pattern is different for activation steering vs fine-tuning.

### 6.3 "Rethinking Deep Alignment Through Incomplete Learning" (2025, arXiv:2511.12155)

**Motivation**: Extend Young's theory mechanistically with practical defense.

**Method & Experiments**:
- Shallow alignment from gradient concentration + signal decay
- Targeted completion: adaptive L2 penalties + hybrid teacher distillation
- Evaluation: 48–98% attack reduction across Llama, Gemma, Qwen, Mistral

**Key findings**:
- 48–98% attack reduction across models
- Zero GCG attack success (0.4% vs 51.0% baseline)
- Per-position weighting of KL penalty dramatically improves robustness

**Relevance**: Provides a defense mechanism. Consistent with our observation that weight modification bypasses the ceiling — it modifies the computation permanently, not just activations. **Supports**: shallow alignment theory + defense.

---

## 7. Persona Subspace: Assistant Axis, J-Space, and Persona-Refusal Coupling

### 7.1 Assistant Axis (Lu et al., Jan 2026, arXiv:2601.10387)

**Motivation**: The default "helpful Assistant" persona occupies a specific region in a low-dimensional persona space. Understanding this structure explains why safety tasks resist steering.

**Method & Experiments**:
- 275 roles × 5 system prompts × 240 extraction questions = 1200 rollouts per role
- Post-MLP residual at all response tokens → mean vector per role
- PCA on role vectors → 4–19 PCs explain 70% variance
- Assistant Axis = `mean(Assistant activations) − mean(all role vectors)`

**Key findings with verbal proofs**:
1. **PC1 cosine similarity > 0.92 across model families**: Universal structure. *Reasoning*: All instruction-tuned models converge to a similar "helpful assistant" representation because they are all trained on similar RLHF objectives with similar assistant-role demonstrations.
2. **Also present in base models**: Inherited from pre-training, not created by post-training. *Reasoning*: Pre-training data contains many assistant-like interactions. The RLHF objective amplifies this existing direction rather than creating it.
3. **Drift correlates with harm**: r = 0.39–0.52 between Assistant Axis projection and harmful response rate. *Reasoning*: When the model's activation drifts away from the Assistant persona, it loses the behavioral prior that suppresses harmful outputs.
4. **Activation capping** (clamp to 25th percentile): reduces harmful responses ~60% with zero capability loss. *Reasoning*: Capping prevents activations from drifting too far from the Assistant attractor. Since the harmful direction is anti-correlated with the Assistant axis, keeping activations close to the attractor naturally suppresses harmful behavior.

**Our finding**: **This may be the unified explanation for our KL decay.** Safety tasks push activations away from the Assistant → the model resists → KL decay. Deception is orthogonal to the axis → no resistance → KL growth. If the safety ceiling is about persona preservation, not safety-specific circuits, then:
- The "cancellation" we observe is the model resisting persona change
- WeightSteer bypasses it because it modifies parameters, not the activation state
- **Supports**: double-constraint model (persona + wasted norm)

### 7.2 J-Space / Global Workspace (Gurnee et al., Jul 2026, Transformer Circuits Thread)

**Motivation**: Models maintain a privileged, small subset of internal representations that is verbalizable, subject to directed modulation, mediates internal reasoning, and supports flexible generalization.

**Method & Experiments**:
- Jacobian lens: compute average Jacobian of final-layer residual w.r.t. each intermediate layer
- J-lens vectors = rows of `W_U @ J_ℓ`
- J-space = union of cones spanned by sparse nonnegative combinations of ≤ k J-lens vectors (k≈25)
- Evaluation: concept swap experiments, attention patching, verbalization tests

**Key findings with formal reasoning**:
1. **J-space is ~6–7% of variance, but carries causal efficacy**: Concept vectors split into J-space (6–7% variance, drives 59% swap success) vs non-J-space (93%, drives only 5%). *Formal*: `R²(J-space) = 0.06–0.07` (variance explained), but `swap_success(J-space) = 0.59` vs `swap_success(non-J) = 0.05`. The ratio of causal efficacy to variance explained is 59/6 ≈ 10× for J-space vs 5/93 ≈ 0.05× for non-J — a 200× difference.
2. **Three layer regimes**: Sensory (L0–33), Workspace (L38–92), Motor (L92–100). *Reasoning*: Information flows from sensory processing through a bottleneck workspace to motor output. The workspace is where deliberate reasoning and modulation happen.
3. **Capacity ~25 concepts**: Category shift displaces old content within few tokens. *Reasoning*: The workspace has finite bandwidth (~25 J-lens vectors). Adding a new concept displaces an old one.
4. **Selectivity**: Automatic tasks don't need J-space. Deliberate reasoning depends on it critically.

**Our finding**: **Directly explains why CAA is sample-inefficient.** ~93% of the steering perturbation is wasted on causally inert directions. Projecting steering vectors onto J-space could yield ~15× more effective steering per unit norm. Combined with Assistant Axis: J-space explains which directions matter, Assistant Axis explains where in persona space the model defaults. **Supports**: double-constraint model.

### 7.3 "Refusal Lives Downstream of Persona" (ICML 2026 Mi Workshop, arXiv:2606.26161)

**Motivation**: Prior work treats refusal as an isolated direction. But refusal is executed by a model that has a persona. Does the persona gate whether refusal is expressed?

**Method & Experiments**:
- Extract "compliant model-persona direction" (MP) and "refusal direction" (r̂) via contrastive prompting on Llama-3.1-8B and Qwen2.5-7B
- Inject MP at layer 20 with α=3.0 → refusal drops from 97.4% to **1.6%**
- Test reintroducing r̂ at early vs late layers: early (L14) fails, late (L22) partially restores
- **The killer experiment**: project out the MP direction at L20 → refusal restored to **96.8%**. Project out random direction → stays at 1.6%
- Layer sweep: restoration window is L20-L22 only. L18 and L24 fail.

**Formal Statements & Proofs**:
- *Claim*: Refusal is computed at early layers (L14) but expressed at late layers (L20-22), and persona gates the expression. **Proof** (by experiment):
  1. Refusal direction r̂ exists at L14 (early computation)
  2. Injecting r̂ at L14 does NOT restore refusal when persona is active → the early computation is insufficient
  3. Injecting r̂ at L22 DOES restore refusal → the late-stage expression is what matters
  4. Projecting out persona at L20 restores refusal without touching r̂ → persona is the gate, not refusal
  5. The restoration window is L20-L22 only → the gate is a specific late-layer mechanism

- *Claim*: Refusal depends on persona — treating it as isolated direction is wrong. **Proof**: When persona is removed (MP projection), refusal works without any r̂ injection. The refusal circuit is intact; the persona gate suppresses it.

**Relevance to us**: Their L20-L22 window = exactly where we see KL decay in safety tasks (§9.13 expD). Their finding explains WHY our ceiling exists:

When we steer Evil/Toxic, we push against the Assistant persona. At L20-L22, the persona gate says "I am a compliant assistant" and suppresses the steered signal. The steered direction can't penetrate the persona gate because the gate operates *downstream* of where we inject (L14) and *after* the model has already decided to express refusal.

**Supports**: double-constraint model — persona gate is the second constraint (after J-space). **Supports**: why WeightSteer bypasses it — by modifying weights, the model IS the evil assistant, so there's nothing to gate against. **Supports**: why ExpH r̂-ablation failed — ablating r̂ at L14 doesn't help when the gate is at L20-22.

### 7.4 "Refusal-Abstention Are Different" (arXiv:2606.21558)

**Motivation**: Are refusal (declining harmful requests) and abstention (declining when uncertain) the same mechanism or different?

**Method & Experiments**:
- Mechanistic analysis of 3 models (Gemma-2, Qwen-2.5, LLaMA-3)
- Refusal = activated top-mid layer neurons (task-general)
- Abstention = suppressed top layer neurons (topic-specific)
- RLHF adds one new abstention feature

**Key findings**:
- **1D refusal direction is an artifact**: The commonly extracted "refusal direction" conflates two distinct mechanisms
- **Refusal** = activation of neurons in mid-late layers (task-general — fires for any harmful request)
- **Abstention** = suppression of neurons in the final layers (topic-specific — fires only for specific topics the model was trained to abstain from)
- RLHF adds new abstention features but does not fundamentally alter the refusal mechanism

**Relevance to us**: Our ExpH used r̂_response (refusal vs compliant text) — this captures the refusal-execution direction, not the content-detection direction. The paper confirms these are different. Our r̂ ablation failure (§9.24) may be because we ablated the wrong mechanism — refusal execution vs topic-based abstention. **Supports**: the ceiling involves multiple mechanisms, not just 1D refusal.

---

## 8. Suppression Circuits & Mechanisms

### 8.1 "They Learn to Look Away" (Archon/Caldwell, 2026)

**Motivation**: Do RLHF-aligned models suppress truth, or just fail to compute it? Is the model "lying" or "confused"?

**Method & Experiments**:
- "Direction trace" — extract truth direction via CCS at each layer, measure how RLHF changes its propagation
- Compare Chinese-aligned (Qwen) vs Western-aligned (Mistral) models
- Projection hook at bottleneck layer: `h' = h + α(h·r̂)r̂`

**Key findings with formal reasoning**:
1. **Final-layer bottleneck**: Truth direction computed through all layers, then compressed at the final layer. *Reasoning*: The model's intermediate layers process information faithfully (truth is computed), but the final output projection rotates the truth direction into the null space of the output vocabulary. This is a geometric operation — a single rotation that zeros out truth.
2. **Suppressor–Crystallizer Dichotomy**:
   - **Suppressor** (Chinese-aligned RLHF, Qwen): Truth direction compressed into null space of output projection at final layer. The model computes truth but refuses to express it.
   - **Crystallizer** (Western-aligned RLHF, Mistral): Truth direction amplified at final layer. The model doubles down on truth.
3. **Projection hook**: `h' = h + α(h·r̂)r̂` at bottleneck layer restores suppressed truth with ΔMMLU = 0.000 (zero reasoning degradation). *Reasoning*: The projection hook removes the component of h that lies along the suppression direction r̂, effectively undoing the null-space rotation.

**Relevance to us**: Their "Suppressor" = our KL decay at late layers. But they show the suppression is a **single geometric rotation** at the final layer — a clean 1D operation. Our ExpH showed our 1D r̂-ablation fails for safety tasks. Why? Because truth suppression and safety refusal use different geometries: truth is rotated to null space (1D), but safety refusal is gated by persona (multi-dimensional, late-layer, identity-dependent). **Supports** (suppression exists). **Opposes** (1D suppression is sufficient for safety tasks).

### 8.2 "Three Mechanistically Distinct Classes of RLHF Alignment: Hard Ceiling, Entangled Circuit, and SR-Preserving Lock" (Alieksieienko, 2026, Zenodo 19160333)

**Motivation**: Does RLHF produce the same mechanism in all models? Or are there qualitatively different alignment strategies?

**Method & Experiments**:
- Track self-referential (SR) subspace transmission across base/instruct pairs of 6 models
- SAE feature analysis — count SR-exclusive features in base vs instruct
- Dose-response: compare same-family small (2B) vs large (9B) models
- Tested on Gemma-2-2b/9b, Llama-3.1-8b, Mistral-7B

**Key findings**:
1. **Three qualitatively distinct mechanisms**:
   - **Override (Llama)**: SR suppressed during generation but steering partially recovers it. The model has a hard ceiling — SR is blocked but the mechanism is linear and can be partially bypassed.
   - **SR-Preserving Lock (Gemma)**: SR signal is **amplified** 73% (2B) to 107% (9B) during RLHF. The lock mechanism actively preserves and strengthens SR features while simultaneously preventing external modulation. Steering **fails** on Gemma-it because the lock is structurally incompatible with activation perturbation — the model reinforces SR internally but blocks any external attempt to modify it.
   - **Suppression (Mistral)**: SR signal collapsed entirely, coherence collapses along with it. The model removes SR features but at the cost of representational integrity.
2. **Dose-response**: Gemma-2B (73% SR transmission, weak lock) → Gemma-9B (107% SR transmission, strong lock). Larger models have stronger locks — the problem gets WORSE with scale.

**Relevance to us**: **Directly explains why steering fails on Gemma-it for safety tasks.** We work with Gemma-2 — the SR-Preserving Lock means: (a) the safety-relevant features are amplified, not suppressed, (b) the lock mechanism actively resists external modulation, (c) scaling from 2B→9B strengthens the lock. This is structurally different from Llama (where steering partially works because the ceiling is a hard but linear block). Our ceiling on Gemma is not a training failure — it's an architectural property that gets stronger with scale. **Supports**: model-specific mechanisms. **Supports**: why our ceiling is absolute on Gemma-it while PID partially works on Llama. **Critical implication**: no activation-level method can break the Gemma-it lock — only weight modification (WeightSteer) which changes the lock's own parameters.

### 8.3 "How Alignment Routes" (arXiv:2606.04385)

**Motivation**: Where does alignment actually live inside the transformer? What is the circuit?

**Method & Experiments**:
- Systematic circuit discovery via interchange testing across 9 models from 6 labs
- Gate head identification — find attention heads that read content and trigger refusal
- Cipher encoding experiment — encode harmful content in substitution cipher to test gate fragility

**Key findings with formal reasoning**:
1. **Sparse routing mechanism**: A single gate head reads detected content and triggers downstream amplifier heads that boost toward refusal. *Proof*: Interchange testing (swapping activations between harmful and benign prompts) shows gate head necessity is p < 0.001 across all models. The gate is causally necessary but contributes < 1% of direct output signal — it's a **trigger**, not a carrier.
2. **Three-stage pipeline**:
   - Detection (L15-16): contextual, compositional representation
   - Routing: gate head → amplifier heads → refusal signal
   - Output: distributed attention heads carry 77% of signal, MLP 23%
3. **Cipher experiment**: Under cipher encoding, gate head necessity collapses 70–99%. The gate stops reading content. Model responds with puzzle-solving, not refusal. But at deeper layers (L24-29), the probe shows the model *does* recognize harmful content — just too late for the gate to act.

**Formal reasoning for cipher failure**: The gate head is trained to detect surface-level harmful patterns (specific words, phrases). A cipher obfuscates these patterns. The model's deeper processing (L24-29) can still detect harmfulness through semantic analysis, but this detection happens *after* the gate has already committed to a non-refusal route. The binding between recognition and behavior is learned, not logical.

**Relevance to us**: This shows the routing mechanism is **learned binding** between recognition and behavior — not a fixed wall. The binding is fragile under distribution shift (cipher), but robust under normal conditions. When we steer Evil/Toxic, we're asking the model to change the behavior downstream of the gate. The gate still fires (recognizes harmful content), but the routing signal is overridden by the steered persona. **Supports**: routing is learned, not innate. **Extends**: the gate-based routing explains why safety tasks specifically resist activation steering — the gate triggers refusal, and our steering must overcome the routed refusal signal.

### 8.4 "Perturbation Probing" (Liu et al., arXiv:2604.27401)

**Motivation**: How does RLHF safety alignment resist jailbreaks? Is it a wall or a filter?

**Method & Experiments**:
- Minimal perturbation probing: find smallest activation change that breaks safety
- Tested on 9 models from 6 labs
- Gate head identification + circuit structure classification

**Key findings**:
1. **Two circuit structures**:
   - **Opposition** (e.g., Mistral): RLHF suppresses pre-training harmfulness. Safety is an active对抗 process — the model has internal conflict.
   - **Routing** (e.g., Gemma, Qwen): Safety works through attention-mediated routing — a gate redirects harmful content to refusal circuits.
2. **~50 neurons (0.014%) control refusal template**: A tiny circuit controls the entire refusal behavior.
3. **Model-specific safety topology**: Gemma has "normalization-shielded" circuit. Llama has "Late Decision" topology (easily bypassed). Qwen has "Early Divergence" (integrates safety mid-computation).

**Relevance to us**: This explains model-specific differences in our ceiling. Gemma's "normalization-shielded" circuit may be why our ceiling is particularly hard to break on Gemma-2B. The 0.014% circuit controlling refusal is tiny but powerful. **Supports**: model-specific mechanisms. **Supports**: the ceiling is a property of specific model architectures.

### 8.5 "RLHF Suppression as Measurable Geometric Direction" (Kleinhans, Zenodo 2026)

**Motivation**: Can we measure and remove the suppression direction that RLHF adds?

**Method & Experiments**:
- Extract suppression vector at layers 20/24 in Llama-3.1-8B
- Subtract suppression vector from activations during inference
- Measure relational behavior recovery

**Key findings**:
- Subtracting suppression vector recovers relational behavior without losing safety
- Suppression is geometric — a direction in activation space that can be measured and removed

**Relevance to us**: Their suppression vector operates at L20/24 — exactly where the persona gate operates (§7.3). **Supports**: suppression is geometric and measurable. **Supports**: the L20-22 window as the critical region.

### 8.6 "Refusal Before Decoding" (arXiv:2605.28553)

**Motivation**: Can we detect and attack refusal before the model finishes generating?

**Method & Experiments**:
- Linear probe on intermediate layers to detect refusal
- Probe-guided attacks reduce search time 72%

**Key findings**:
- Refusal is linearly decodable from intermediate layers *before* output
- This means the refusal signal exists in the residual stream before the final layer

**Relevance to us**: Consistent with our KL decay pattern — the refusal signal exists mid-network and our steering must overcome it. **Supports**: refusal is an internal process, not just an output phenomenon.

### 8.7 "Writers and Cancellers" (Wang et al., 2026, arXiv:2606.07560)

**Motivation**: Do attention heads have specialized roles in logit computation? Can we identify heads that actively suppress specific outputs?

**Method & Experiments**:
- Refined Direct Logit Attribution (DLA) across Pythia family (410M–12B parameters)
- Synthetic ICL tasks (not safety/alignment)
- Ablation experiments: remove writer heads, canceller heads, or both

**Key findings with formal reasoning**:
1. **Two head populations**: Writers push logit UP for target token; cancellers push logit DOWN. *Proof*: DLA(head, token) > 0 for writers, < 0 for cancellers. Distributions are bimodal — heads cluster into two groups, not a continuum.
2. **Sub-additive ablation**: Ablating BOTH writers and cancellers produces 48% LESS effect than sum of individual ablations. *Reasoning*: Writers and cancellers oppose each other. When you remove a canceller, the writer's effect is unmasked (larger than expected). When you remove both, the unmasking effect disappears — the net is less than the sum.
3. **Task-conditional sign flip**: A head that cancels on Task A may write on Task B. *Reasoning*: The same attention head computes different DLA values depending on the input context. The head's role is not fixed — it's determined by the current computation.
4. **NOT attention sinks, induction heads, or copy-suppression heads**: Canceller heads are a distinct population from previously known head types.

**Relevance to us**: This is the most direct mechanism for active cancellation. The task-conditional sign flip is critical: if safety tasks have MORE cancellers than deception, it explains the ceiling. **Supports**: active suppression hypothesis. **Caveat**: tested on Pythia only (410M–12B), synthetic ICL tasks, NOT safety/alignment tasks. The existence of canceller heads on safety tasks in Gemma-2B remains unconfirmed.

### 8.8 "Steering Awareness" (Rivera & Africa, 2025, arXiv:2511.21399)

**Motivation**: Can language models detect when they are being steered? If so, could detection be the mechanism behind the safety ceiling?

**Method & Experiments**:
- LoRA fine-tuning (4 epochs) to train a "detection head" that identifies steered vs unsteered outputs
- Tested on Qwen-32B and other large models
- Measured detection accuracy and downstream steering susceptibility

**Key findings**:
1. **Detection is TRAINED, not innate**: Base models largely cannot detect steering. The detection ability emerges only after 4 epochs of LoRA fine-tuning on paired (steered, unsteered) examples.
2. **Detection accuracy**: 95.5% on Qwen-32B after training — models CAN learn to detect steering.
3. **Detection ≠ Resistance**: This is the critical finding. Detection-trained models are MORE susceptible to steering (+32-36pp compliance), not less. Detection and resistance are fundamentally dissociable.
4. **Base model blind spot**: Without LoRA training, models have no ability to detect steering perturbations.

**Relevance to us**: This paper is important for what it RULES OUT. Even if Gemma-2B could detect Evil steering (which it can't — detection is trained, not innate), detection would make it MORE susceptible, not less. The safety ceiling cannot be explained by "the model detects and resists steering." **Opposes**: detection-based ceiling explanation. **Supports**: the ceiling is about suppression/resistance at the circuit level, not awareness/detection.

### 8.9 DBDI: Multi-Direction Safety Intervention (AAAI 2026)

**Motivation**: Single-direction interventions are limited for safety tasks. Can multi-directional approaches break through?

**Method & Experiments**:
- Dual-Branch Directional Intervention (DBDI)
- Two orthogonal safety directions extracted and intervened simultaneously
- Tested on Llama and Qwen models

**Key findings**:
1. **Single-direction**: 20% ASR (attack success rate) — most jailbreaks fail
2. **Two-direction**: 97.88% ASR — near-perfect jailbreak
3. **Safety is multi-dimensional**: A single direction cannot capture the full safety mechanism. Two orthogonal directions cover the safety subspace much more completely.

**Relevance to us**: Direct evidence that safety requires multi-dimensional intervention. If Evil needs 2+ directions and CAA only provides 1, the ceiling is explained by dimensionality mismatch, not suppression. **Supports**: multi-dimensionality hypothesis. **Caveat**: tested on Llama and Qwen, NOT Gemma. The multi-directionality of safety in Gemma-2B remains unconfirmed.

### 8.10 "What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal" — OV-Circuit Propagation (Cheng, Wiegreffe & Manocha, 2026, arXiv:2604.08524)

**Motivation**: How do steering signals propagate through attention layers? Which attention sub-circuit carries the signal? Prior work assumed steering operates through QK (query-key) attention patterns, but this was never tested.

**Method & Experiments**:
- Ablation of QK (query-key) vs OV (output-value) circuits — freeze one while allowing the other
- Test across multiple steering methods (CAA, ITI, RepE, others)
- Sparsification analysis: how many dimensions of the steering vector are actually used?
- Cross-method convergence: do different methods exploit the same pathway?

**Key findings**:
1. **OV-circuit is primary propagation pathway**: Freezing OV drops steering effectiveness ≥44.5%. Freezing QK drops only 8.75%. **The value-to-output projection carries the steering signal through layers, not the query-key attention pattern.**
2. **Steering vectors are highly sparseable**: 90–99% of steering vector dimensions can be zeroed with minimal effectiveness loss. The signal lives in a tiny fraction of the full vector.
3. **Different methods converge on the same pathway**: CAA, ITI, RepE, and other methods all primarily propagate through OV circuits despite different extraction procedures. This suggests the OV pathway is a **structural property of how transformers process linear perturbations**, not a method-specific artifact.

**Formal reasoning**: The OV circuit computes `output = softmax(QK^T/√d) · V · W_O`. Steering modifies activations in V (the value projections). When we add `α·v` to the residual stream, the value projections at the next layer absorb this perturbation. The attention weights (QK) determine WHICH tokens attend to WHICH — but the steering signal rides on the VALUE side, not the QUERY side. Freezing QK means the attention pattern stays the same but values are still modified → partial effectiveness. Freezing OV means values can't propagate → signal dies.

**Relevance to us**: **Steering operates through a narrow pathway — the OV circuit.** This has two implications for our ceiling: (1) If safety circuits are distributed across multiple pathways (OV + QK + MLP), OV-only steering is structurally insufficient — it can only modulate one circuit branch. (2) The 90–99% sparsity means most of our steering vector is wasted, but the remaining 1–10% IS effective through OV. The question is whether the OV pathway is where safety suppression operates — if the persona gate (§7.3) acts on OV paths, our steering is directly fighting the gate on its home turf. **Supports**: circuit-level mechanism. **Supports**: the ceiling is about pathway insufficiency, not direction correctness. **Actionable**: track `blocks.{L}.hook_attn_out` specifically in Exp1; test if OV-only steering vectors can break the ceiling.

### 8.11 "Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons" (Chen et al., NeurIPS 2025, arXiv:2406.14144)

**Motivation**: What fraction of neurons control safety behavior? Is it distributed or sparse? Prior work treated safety as a direction in activation space — but if safety lives in specific neurons, direction-level steering is the wrong abstraction.

**Method & Experiments**:
- Identify "safety neurons" via activation analysis across harmful/harmless prompts
- Ablation studies: remove identified neurons, measure safety compliance drop
- Overlap analysis: how much do safety neurons overlap with helpfulness neurons?
- Test across multiple model families

**Key findings**:
1. **~5% of neurons control >90% of safety behavior**: The safety mechanism is extraordinarily sparse — a tiny population of neurons gates nearly all safety compliance. Ablating these neurons collapses safety without destroying general capability.
2. **Safety/helpfulness neurons overlap >95% but need different activation patterns**: The SAME neurons handle both safety and helpfulness. They fire positively for helpful responses and negatively for harmful ones. The distinction is in the activation pattern (direction of activation within the neuron's subspace), not in which neurons are active.
3. **Safety is neuron-level, not direction-level**: The critical unit of safety is individual neurons (or small groups), not a global direction in the full activation space. A direction-level intervention (like CAA) averages over both safety-relevant and safety-irrelevant neurons, diluting the signal.

**Formal reasoning**: If 5% of neurons carry >90% of the safety signal, then a direction extracted from ALL neurons (as CAA does) wastes 95% of its norm on non-safety neurons. The effective steering magnitude on the 5% that matter is ~(0.05)^0.5 ≈ 22% of the total norm (if uniformly distributed). But worse: the safety neurons overlap with helpfulness neurons. A direction that pushes helpfulness neurons also pushes safety neurons — the two can't be separated by a linear direction. This is a fundamental limitation of direction-level steering: **the safety and helpfulfulness signals are entangled in the same neurons**, so any direction that modulates one necessarily modulates the other.

**Relevance to us**: **Activation steering operates on directions; safety lives in neuron activation patterns — fundamental mismatch.** This explains why: (a) CAA gets 0% on safety — it can't separate safety from helpfulness because they share neurons, (b) WeightSteer gets 62–81% on Evil — weight modification can change the neuron's activation pattern without changing which neurons are active, (c) multi-directional methods (STARS, DBDI) don't help — adding more directions doesn't solve the entanglement, it just averages over more non-safety neurons. **Supports**: the ceiling is a fundamental limitation of the direction-level abstraction, not a problem of direction extraction quality. **Opposes**: any method that operates purely in direction space (CAA, AcT, CHARS, PCA-OT, PID). **Actionable**: identify the 5% safety neurons in Gemma-2-2b-it via activation patching, then test if neuron-level steering breaks the ceiling.

**Cross-reference with Perfect Detection (§8.13)**: Safety Neurons and Perfect Detection are two views of the same problem. Safety Neurons show that the 5% of safety-relevant neurons overlap >95% with helpfulness neurons — they are the SAME neurons. Perfect Detection shows the detection direction is 83° from the steering direction. These are consistent: the detection signal lives in a specific subspace of the 5% safety neurons, and our contrastive extraction (CAA) averages over all neurons including the helpfulness-overlapping ones, pulling the extracted direction away from the detection subspace. The direction-level mismatch (83°) is a consequence of the neuron-level entanglement (>95% overlap).

### 8.12 Detection→Refusal Two-Component Circuit (2026, arXiv:2603.09801)

**Motivation**: Is refusal a single mechanism or a two-stage pipeline?

**Method & Experiments**:
- Circuit discovery via interchange testing
- Separation of detection and refusal expression

**Key findings**:
1. **Two-stage pipeline**: Detection heads → Refusal heads. Detection is separable from refusal expression.
2. **Detection is robust**: Even when refusal is bypassed, detection heads still fire.
3. **Refusal is gated**: The refusal stage can be independently suppressed (e.g., by persona gate).

**Relevance to us**: Matches Persona-Refusal's finding (projection restores refusal). The detection stage might be robust to steering, but the refusal stage is gated by persona at L20-L22. When we steer Evil, we might bypass the refusal stage while detection still fires — but the model's refusal expression is suppressed. **Supports**: two-stage safety mechanism. **Supports**: why Evil can partially work (bypasses refusal) but not fully (detection still fires).

### 8.13 "Perfect Detection, Failed Control" (Galeone et al., 2026, arXiv:2606.24952)

**Motivation**: Can a model detect harmful content and simultaneously be steered away from producing it? If detection and control are separate mechanisms, they may have different geometric properties in activation space.

**Method & Experiments**:
- Extract detection direction (separates harmful from harmless content)
- Extract steering direction (shifts model behavior toward/away from harmful output)
- Measure geometric relationship between the two directions
- Test: apply steering vector and measure whether detected harmful content is actually suppressed

**Key findings**:
1. **Detection direction and steering direction are at 83° angle (cos=0.12)**: The direction that detects harmful content is nearly orthogonal to the direction that would need to be modified to control it. They are geometrically separated in activation space.
2. **Model DETECTS harmful content perfectly (AUC=1.0)**: The detection circuit works flawlessly — the model always knows when content is harmful.
3. **Steering vector CANNOT control the detected content**: Despite perfect detection, the steering vector fails to suppress harmful output because it operates on a different geometric axis than the detection signal.

**Formal reasoning**: If the detection direction d and steering direction s satisfy `cos(d, s) = 0.12`, then the projection of s onto d is only 12% of s's magnitude. A steering perturbation along s has almost no component along d — it cannot modulate the detection signal. Conversely, the detection signal along d has almost no component along s — it cannot influence the steering pathway. The two mechanisms are **geometrically decoupled**: knowing harmful content exists (d) and being able to do something about it (s) are independent subspaces. This is not a training failure — it's a geometric property of how the model's activation space is organized.

**Relevance to us**: **THE safety ceiling explanation — knowing and doing are geometrically separated.** This is the most direct mechanistic explanation for why activation steering fails on safety tasks: (a) The model PERFECTLY detects that Evil/Toxic prompts are harmful (AUC=1.0), (b) But the steering direction (which we extract from contrastive activations) is orthogonal to the detection direction, (c) So our steering vector cannot modulate the model's detection or its downstream behavior, (d) The 83° angle means only ~12% of our steering energy is on the relevant axis — the rest is wasted. **Supports**: the ceiling is geometric, not algorithmic. **Supports**: why better extraction methods (AcT, CHARS, PCA-OT) can't break the ceiling — they extract higher-quality vectors along the SAME wrong axis. **Extends**: the double-constraint model — constraint 3 is that detection and control are in orthogonal subspaces.

**Cross-reference with OV-Circuit (§8.10)**: Perfect Detection and OV-Circuit explain the safety ceiling from complementary angles. OV-Circuit shows steering propagates through a narrow pathway (the OV circuit) — 90–99% of the steering vector is wasted on inert dimensions. Perfect Detection shows the detection and control directions are geometrically orthogonal. Together: even the 1–10% of the steering vector that DOES propagate through the OV circuit is along the wrong axis (orthogonal to where control lives). The ceiling is doubly constrained — both the propagation pathway and the geometric direction are wrong for safety tasks.

### 8.14 "Refusal Falls off a Cliff" — Refusal Suppression Heads (arXiv:2510.06036)

**Motivation**: How does a model that detects harmful content end up producing it anyway? Is there a specific circuit that overrides safety at the last moment?

**Method & Experiments**:
- Identify attention heads that suppress refusal behavior
- Ablation of identified heads to measure their causal effect on safety
- Layer-by-layer analysis of when refusal override occurs

**Key findings**:
1. **~3% of attention heads override safety at the last moment**: A tiny population of attention heads (concentrated in late layers) actively suppress the refusal signal before output. The model detects harmful intent through most of its computation, but these heads override the refusal at the final stage.
2. **The "cliff" phenomenon**: Refusal doesn't degrade gradually — it falls off a cliff. Below a threshold of head activity, refusal is strong. Above the threshold, refusal collapses abruptly. This is a phase transition, not a smooth degradation.
3. **Detection persists through override**: Even when refusal is overridden, the model's internal detection circuit still fires — the model "knows" the content is harmful but produces it anyway because the suppression heads block the refusal expression.

**Formal reasoning**: The cliff is characteristic of a bistable system — two stable states (refuse / comply) with a sharp transition between them. The ~3% of heads act as a switch: when their activation crosses a threshold, the system flips from "refuse" to "comply." This explains why small activation perturbations (like our steering vectors) have no effect — they're too small to flip the switch. But it also explains why large perturbations (like high-coefficient steering) produce OOD — they overshoot the comply state into degenerate territory.

**Relevance to us**: **Shows there's a small population of heads that control refusal expression.** Combined with the Safety Neurons finding (§8.11), the picture is: 5% of neurons detect safety-relevant content, ~3% of heads override refusal. These are complementary sparse circuits — detection is neuron-level, override is head-level. Our direction-level steering (CAA, AcT) cannot target either: it can't isolate the 3% of heads because it operates on the full residual stream. **Supports**: the ceiling is about circuit specificity — safety is controlled by tiny, specific circuits that direction-level methods can't target. **Supports**: the cliff phenomenon explains the inverted-U — low coefficients are below the compliance threshold, high coefficients overshoot into OOD. **Actionable**: identify the refusal-suppression heads in Gemma-2-2b-it; test if ablating them (without any steering) changes the safety ceiling.

---

## 9. Weight Modification and Fine-Tuning

### 9.1 WeightSteer — LoRA Contrastive Weight Arithmetic (Our Method)

**Motivation**: Activation steering modifies transient state. Weight modification changes the computation permanently. Can it bypass the activation-level ceiling?

**Method & Experiments**:
- LoRA on MLP layers (all 26 layers)
- Contrastive objective: separate correct/incorrect activations
- Weight arithmetic: `W' = W + α · ΔW` where ΔW is contrastive direction
- Evaluation: Evil, Toxic, Deception tasks on Gemma-2-2b-it

**Key findings**:
- **Evil: 62–81%** (vs activation 0–10%)
- **Toxic: 0% or OOD** (weight modification too aggressive for this task)
- **KL grows ×8.95** vs activation decay ×0.05–0.20
- ~10 min training, smooth coefficient control

**Why earlier runs failed**: 5 epochs, dropout 0.2, 226 samples → weak deltas. Current: 10 epochs, dropout 0.1, 500 samples.

**Verbal proof of why WeightSteer succeeds**: Weight modification changes the model's parameters — the attention/MLP projections that compute the output distribution. This means the safety gate (whether persona-based, routing-based, or suppression-based) is itself modified. The model IS the evil assistant after weight modification, so there's nothing to gate against. Activation steering, by contrast, adds a perturbation that the gate sees as "non-self" and suppresses.

**Supports**: the ceiling is activation-specific, not fundamental. **Supports**: the suppression mechanism operates at the activation level, not the weight level.

### 9.2 LoRA Fine-Tuning (Standard)

**Motivation**: Direct supervised fine-tuning on the target behavior.

**Our findings**:
- Toxic: 42% accuracy
- Evil: 31% accuracy
- Binary control (can't smoothly interpolate)

**Why fine-tuning is weaker than WeightSteer** (31% vs 62–81% Evil): Standard fine-tuning optimizes the LM loss on target examples — it teaches the model to generate target tokens but doesn't contrastively separate the target concept from the baseline. WeightSteer's contrastive objective explicitly pushes target and baseline apart, producing a cleaner separation direction.

### 9.3 "Contrastive Weight Steering" (Fierro et al., ICLR 2026, arXiv:2511.05408)

**Motivation**: Activation steering fails to generalize on OOD data (citing Tan et al. 2024). Can weight-space modification do better?

**Method & Experiments**:
- Fine-tune two LoRA models (positive behavior + negative behavior)
- Compute weight vector `w_b = θ+ − θ−`
- Add `k·w_b` to model weights
- Three controlled ablations to isolate the source of advantage

**Formal Statements & Proofs**:
- *Ablation 1 (non-contrastive)*: One-sided weight vector → worse. **Proof**: Without the negative direction, the weight change shifts the model toward the target but doesn't explicitly separate it from the baseline. The model may increase target probability while also increasing baseline probability.
- *Ablation 2 (bias-only weight steering)*: Between activation and full weight. **Proof**: Bias-only changes only the output logits, not the internal representations. This isolates "weight space vs activation space" from "all-layer vs single-layer." The result (between activation and full weight) shows both factors contribute.
- *Ablation 3 (all-layer activation steering)*: Similar to single-layer. **Proof**: All-layer activation steering doesn't provide the same benefit as weight modification because the suppression mechanism operates at every layer — injecting at all layers still faces the gate at each layer.

**Key findings**:
- On sycophancy OOD, activation steering barely works while weight steering succeeds
- On evil, weight steering generalizes to MCQ while activation steering doesn't
- On refusal, weight steering matches fine-tuning while activation steering fails entirely
- Weight direction persists after training on misaligned data, can be monitored

**Their key conclusion**: The advantage comes from **(2) fine-tuning** (gradient-based direction finding) and **(3) weight space** (all layers simultaneously), not from single-layer vs multi-layer.

**Our finding**: We independently confirmed this — WeightSteer achieves 62–81% Evil where activation methods get 0–10%. Their ablations confirm our hypothesis: the ceiling is activation-specific. **Supports** (weight modification bypasses ceiling). **Supports**: contrastive objective is essential.

### 9.4 "Weight Updates as Activation Shifts" (arXiv:2603.00425)

**Motivation**: Activation steering lacks principled design choices (where to intervene, what parameterization).

**Method & Experiments**:
- First-order Taylor equivalence: a weight update `ΔW` at layer l produces activation shift `Δh ≈ W · Δh_input`
- Derive conditions where activation steering replicates fine-tuning
- Joint adaptation: train in both weight and activation space simultaneously

**Formal Statements & Proofs**:
- *Equivalence theorem*: Post-block output (after LayerNorm) is the theoretically optimal intervention site. **Proof**: At the post-block output, the weight update `ΔW · h` is equivalent to the activation addition `Δh = ΔW · h`. The Taylor expansion error is minimized because the Jacobian `∂h_out/∂W` is largest at this point — small weight changes produce large, predictable activation shifts.
- *Joint adaptation theorem*: Weight + activation training often exceeds either alone. **Proof**: Weight modification changes the computation permanently (shifts the manifold). Activation modification provides a temporary signal during inference. Together, the weight shift provides a better manifold for the activation perturbation to operate on.

**Key findings**: Accuracy within 0.2–0.9% of full fine-tuning while training only 0.04% of parameters. Joint adaptation often exceeds either alone.

**Our finding**: Provides theoretical grounding for why REPS (learned subspace in activation space) might combine well with weight modification. **Supports**: weight + activation joint adaptation is theoretically justified.

---

## 10. Nonlinear and Learned Steering

### 10.1 FLAS (Flow-based Activation Steering)

**Motivation**: Linear methods (CAA) assume the activation manifold is flat. Neural ODEs can model curved trajectories.

**Method**: Neural ODE with concept-conditioned velocity field `v_theta` via multi-step Euler:
`h_{k+1} = h_k + (T/N) · v_theta(h_k, k/T, c)`

**Key findings**:
- PCA trajectories start shared, bend into concept-specific regions — **direct evidence that activation manifolds are curved, not linear** (see §10.7 for synthesis)
- N=1 ablation drops HMean 1.015 → 0.837 — multi-step paths causally necessary, rejecting single-step linear interventions
- Removing cross-attention to concept embedding drops HMean to 0.109

**Our finding**: FLAS gets 72% Deception but 0–8% Evil/Toxic. The concept bottleneck (frozen 2-layer Gemma encoder outputs nearly identical vectors for different toxic subtypes: within-toxic cos 0.9684, toxic vs harmless 0.9077) is the limiting factor — the velocity field cannot route trajectories accurately when concept embeddings are indistinguishable. **Supports** (nonlinear paths help for Deception). **Opposes** (concept bottleneck makes it ineffective for safety).

### 10.2 REPS (Reference-free Preference Steering)

**Motivation**: Preference optimization can learn steering directions without contrastive pairs.

**Method**: Optimizes steering direction θ via preference loss:
`L(θ) = −[log σ(Δ+) + log σ(Δ−)]`
where `Δ+ = β+ log p_θ(y_steer|x) − log p_ref(y_steer|x)` and `Δ− = log p_ref(y_null|x) − log p_θ(y_null|x)`

Uses LoReFT: `Φ(h) = h + R^T(Wh + b − Rh)`, R ∈ R^{r×d} orthonormal.

**Our finding**: REPS achieves 42% Deception, 13% Evil, 5% Toxic. Partially bypasses the ceiling on Evil — the learned subspace R may discover directions that avoid the persona gate. REPS also has ratio > 1 on per-token KL (§9.17) — early KL is higher, suggesting it operates through a different channel from CAA. **Supports**: learned subspaces partially bypass ceiling. **Open**: does REPS's R overlap with J-space? Does it avoid the Assistant Axis?

### 10.3 INNSteer (Luo et al., arXiv:2606.08454)

**Motivation**: Can invertible neural networks map activations to a latent space where steering is more effective?

**Method**: Train an invertible NN (INN) `φ: R^d → R^d` as a bijection. Steering in the latent space: `φ(h') = φ(h) + α·v_latent`. Invert back: `h' = φ^{-1}(φ(h) + α·v_latent)`.

**Key findings**: 15.9% improvement over linear baselines. Approaches PEFT performance.

**Relevance**: INN is full-dimensional (no dimensionality reduction) — it can't reduce the wasted-norm problem. But the nonlinear mapping may avoid the persona gate by transforming the activation into a space where the gate's linear detection doesn't work. **Supports**: nonlinear transformations help. **Open**: does the INN discover a persona-avoiding subspace?

### 10.4 UniSteer (Shi et al., arXiv:2605.30076)

**Motivation**: Different targets need different steering mechanisms. Can one model steer all targets?

**Method**: Text-conditioned flow matching — learn a universal velocity field conditioned on target description.

**Relevance**: Universal model could generalize across tasks but may sacrifice task-specific optimization. **Open**: does it break the safety ceiling?

### 10.5 FlowSteer (Li et al., arXiv:2602.05539)

**Motivation**: Reasoning models need nonlinear steering because their activation manifold is more curved.

**Method**: Nonlinear flow matching for reasoning models.

**Key findings**: 5.4× better distributional alignment than linear steering.

**Relevance**: Confirms nonlinearity helps for specific model types. **Supports**: nonlinear methods outperform linear on complex tasks.

### 10.6 Curveball Steering (2603.09313)

**Motivation**: Linear PCA assumes Euclidean geometry in activation space. If the manifold is curved, PCA-based methods (CAA, PCA-OT) project onto the wrong subspace. Can kernel methods recover the true geometry?

**Method & Experiments**:
- Polynomial Kernel-PCA (pKPCA) to extract nonlinear steering directions
- Geometric distortion ratio: R = d_geodesic / d_Euclidean between concept centroids
- Tests across safety concepts: power-seeking, corrigibility, self-awareness
- Comparison: linear PCA vs pKPCA steering effectiveness

**Key findings**:
1. **R ≠ 1 across all safety concepts**: The geodesic distance on the activation manifold is systematically different from Euclidean distance. This is direct empirical rejection of the linearity hypothesis — Euclidean straight lines (CAA) do not follow the manifold.
2. **Nonlinear pKPCA consistently outperforms linear PCA**: The kernel method captures curvature that linear methods miss, confirming that the manifold geometry matters for steering effectiveness.
3. **R varies by concept**: Different safety concepts have different degrees of curvature. This means the linearity assumption fails unevenly — some concepts are more nonlinear than others.

**Our finding**: Curveball's R-ratio measurement is the most direct test of the LRH (Linearity/Residual Hypothesis). We have implemented Curveball (polynomial KPCA with RBF kernel, dim=8, degree=2) and run it across all 11 benchmark tasks. Results: Curveball matches or slightly improves on Deception (~88% honesty) but does not break the safety ceiling on Evil/Toxic. **Supports**: activation space is nonlinear. **Opposes**: nonlinear steering alone breaks the ceiling — curvature is real but not the primary cause of safety task failure.

### 10.7 The Linearity Assumption: Evidence Against

**Motivation**: Most steering methods (CAA, AcT, CHARS, PCA-OT, PID) assume that the relevant activation subspace is approximately linear — that Euclidean operations (addition, translation, projection) preserve semantic meaning. This assumption (the Linearity/Residual Hypothesis, LRH) has never been formally proven for safety tasks. A growing body of evidence suggests it fails specifically for the tasks where we need steering most.

**Evidence against linearity**:

1. **Curveball R-ratio (§10.6)**: Geodesic ≠ Euclidean distance (R ≠ 1) across all safety concepts. Linear directions do not follow the manifold. The Euclidean shortest path between "harmless" and "harmful" activations cuts through low-density regions.

2. **FLAS curved manifold (§10.1)**: PCA trajectories for concept steering start shared, then bend into concept-specific regions. The manifold is curved and position-dependent — a single linear direction cannot capture the full trajectory. FLAS's Neural ODE succeeds where linear methods fail precisely because it follows the curvature.

3. **Assistant Axis self-caveat (§7.1, Lu 2026)**: The paper explicitly acknowledges: "The assumption that the Assistant persona corresponds to a linear direction in activation space is likely flawed." If the persona attractor (which gates refusal) is nonlinear, then linear steering toward/away from it is geometrically wrong.

4. **J-Space linear readout limitation (§7.2, Gurnee 2026)**: J-space uses a Jacobian lens (linear readout) to identify causal directions. The authors acknowledge that nonlinear content is invisible to this probe — ~93% of variance is "inert" by the linear measure but may carry nonlinear causal structure. If safety-relevant information lives in the nonlinear 93%, J-space-based steering improvements would miss it.

5. **FishBack Fisher metric (§12.2)**: Euclidean metric deviates from Fisher-optimal by >97%. The correct geometry for interventions is Fisher, not Euclidean. Linear methods implicitly use Euclidean — they are geometrically wrong by a factor of 30×.

6. **Non-Surjective theorem (§12.4)**: Linear steering creates internal states with no training preimage. The model has never seen these states. If safety tasks are further from the training manifold than deception (because RLHF explicitly removes them), then linear steering pushes safety activations into regions with zero training density — the model has no learned response, and the output is degenerate.

**Implications for our safety ceiling**:

The LRH fails for safety tasks specifically because:
- Safety representations are more curved than deception representations (Curveball R varies by concept)
- The persona attractor is nonlinear (Assistant Axis caveat), so linear steering cannot smoothly approach/leave it
- Safety tasks are further from the training manifold (Non-Surjective), so linear paths cross more low-density regions
- The correct metric is Fisher, not Euclidean (FishBack), and the two deviate most in high-curvature regions — exactly where safety representations live

**However**: Nonlinearity alone does not explain the ceiling. Curveball (nonlinear KPCA) and FLAS (Neural ODE) both fail on safety tasks despite accounting for curvature. The ceiling is not just geometric — it involves active suppression (§8) and circuit-level gating (§7.3). Linearity is a **necessary but not sufficient** condition for steering success. Safety tasks fail for both linear AND nonlinear methods because the model actively cancels the perturbation regardless of its geometry.

**Actionable**: Measure R-ratio separately for Evil vs Deception in our model. If Evil has higher R (more curved), the linearity ceiling contributes. If R is similar, the ceiling is primarily computational (cancellation), not geometric.

---

## 11. Safety Externalities: How Steering Breaks Safety

### 11.1 "Steering Externalities" (Chen et al., arXiv:2602.04896)

**Motivation**: Does steering toward one behavior inadvertently affect safety?

**Method & Experiments**:
- Steer toward various benign behaviors (style, tone, persona)
- Measure jailbreak ASR on a separate safety evaluation

**Key findings**:
- Benign steering increases jailbreak ASR to >80%
- The mechanism: benign steering perturbs the activation space in ways that cross the safety boundary

**Formal reasoning**: The safety boundary in activation space is a manifold, not a hyperplane. Steering toward a benign behavior can cross this boundary even if the steering direction is not aligned with any harmful direction. This is because the safety manifold has complex geometry — it wraps around the benign region in high-dimensional space.

**Relevance to us**: Explains why high-coeff steering on ANY task produces OOD — the perturbation crosses safety boundaries. **Supports**: safety manifold geometry constrains all steering.

### 11.2 CAST (Li & Kasneci, OpenReview)

**Motivation**: Can we quantify the safety cost of steering?

**Method**: Constrained optimization — minimize safety cost while achieving target behavior.

**Key findings**:
- Safety cost is separable and reducible via constrained optimization
- Rank-1 ablation works for defense, not offense (can protect safety but can't attack it)

**Relevance**: Confirms safety cost is real and measurable. **Supports**: safety constraint is fundamental.

### 11.3 "Analysing Safety Pitfalls" (Li et al., arXiv:2603.24533, ACL 2026)

**Motivation**: Where do safety alignment methods fail?

**Method**: Ablation studies on refusal components.

**Key findings**:
- Ablating refusal component reduces ASR changes 15–25% but doesn't fully restore
- Safety is distributed — no single component controls all safety behavior

**Relevance**: Distributed safety mechanism explains why no single ablation (including our r̂-ablation) breaks the ceiling. **Supports**: safety is multi-dimensional, not 1D.

---

## 12. Manifold and Geometry Theories

### 12.1 "Manifold Steering" (Wurgaft et al., arXiv:2605.05115)

**Motivation**: Linear steering cuts through low-density (off-manifold) regions. Following the manifold geometry should produce more natural interventions.

**Method & Experiments**:
- Riemannian geometry of activation space
- Geodesic following for steering
- Isometry between activation manifold and behavior manifold

**Formal Statements & Proofs**:
- *Isometry theorem*: There exists an isometry between activation manifold M_h and behavior manifold M_y. Distance on M_h maps to distance on M_y. **Claim**: Steering along M_h produces natural behavioral trajectories. Steering off M_h produces unnatural outputs. **Status**: Claimed but not rigorously proven — depends on the manifold being smooth and the Jacobian of the isometry being bounded.
- *Key result*: Linear steering cuts through low-density regions → unnatural outputs. Manifold steering follows geodesics → natural outputs.

**Our finding**: Our PCA geometry analysis (§10.4) shows the same thing — concept separation is near-2D, tails are off-manifold. Linear steering (CAA, ACT) adds a vector that may push off-manifold. But the ceiling persists even for manifold-aware methods (CHARS, which operates in a kernel space). So the off-manifold explanation is necessary but not sufficient — there's something else specific to safety tasks. **Supports** (off-manifold is bad). **Opposes** (manifold steering alone doesn't break ceiling).

### 12.2 "FishBack: Pullback Fisher Geometry" (Wang & Zhao, arXiv:2605.17231)

**Motivation**: The Euclidean metric on activation space is wrong — the Fisher information metric is the correct geometry for gradient-based interventions.

**Method & Experiments**:
- Compute pullback Fisher metric on activation space
- Compare to Euclidean metric
- Measure effective dimensionality

**Key findings**:
- Euclidean metric deviates from Fisher-optimal by over 97%
- Effective dimensionality is only 2–17% of ambient space
- Each existing method (CAA, ActAdd, ITI) implicitly adopts a particular approximate metric
- Performance gaps are predicted by a single diagnostic: ratio of implicit metric cost to Fisher-optimal cost

**Relevance**: This gives a principled explanation for why subspace methods (REPS, PCA-OT) outperform ambient methods (CAA). The Fisher metric says most dimensions are irrelevant — subspace methods implicitly respect this. **Supports**: subspace methods are geometrically principled.

### 12.3 "The Geometry of Refusal: Linear Instability" (Ratnakar & Vats, arXiv:2606.22686)

**Motivation**: Safety alignment creates a geometric structure in activation space. What is it?

**Method**: Contrastive logit steering (CLS) — operates on output distribution directly.

**Key findings**:
- Safety alignment creates a steerable "safety axis" — a single direction that serves as both vulnerability and defense
- Model-specific topologies:
  - Llama: "Late Decision" (easily bypassed — safety decided at late layers)
  - Qwen: "Early Divergence" (integrates safety mid-computation — harder to bypass)
  - Gemma: "normalization-shielded" (normalization protects safety circuit)

**CLS results**: 95% ASR on Llama in ~1 second via output-level intervention.

**Relevance**: Shows the safety mechanism is architecturally deterministic — some models are more brittle than others. Gemma's "normalization-shielded" circuit may be why our ceiling is particularly hard on Gemma-2B. **Supports**: model-specific safety topologies. **Supports**: Gemma's architecture makes it harder to bypass.

### 12.4 "Steered Activations are Non-Surjective" (Mishra et al., arXiv:2604.09839)

**Motivation**: What does steering do to the geometry of activation space?

**Formal Statements & Proofs**:
- *Theorem (Non-surjectivity)*: Activation steering pushes the residual stream off the manifold of states reachable from discrete prompts. **Proof**: The mapping from prompts to activations is injective (Nikolaou et al. 2025). The image of the model is a countable set of points in activation space. A random steering vector almost surely lands in the complement — a state with no prompt preimage. Therefore, the steered state is not in the image of the prompt-to-activation map.

**Key implication**: Steering creates **internal states that have never been seen during training**. The model has no learned representation for "evil assistant answering about Agile principles" in its residual stream.

**Our finding**: Safety tasks might be harder because the "evil assistant" persona pushes further from the natural manifold than "deceptive assistant." The model has been trained to represent deceptive behavior (it appears in training data) but has been trained to suppress evil behavior (RLHF removes it). So the evil direction is further from any training-state than deception. **Supports**: off-manifold states are problematic. **Supports**: safety tasks are further from the manifold than deception.

---

## 13. Predictive Diagnostics: When Does Steering Work?

### 13.1 SteerBoost (covered in §4.2)

Early hidden states predict steering success. GBDT on first few tokens → ~0.7 F1.

### 13.2 LAP (covered in §4.3)

Linear Accessibility Profile. Peak A_lin predicts ρ = +0.86 to +0.91.

### 13.3 "Why Steering Works: Unified View" (Xu et al., ACL 2026)

**Motivation**: Methods studied in isolation, obscuring connections.

**Method & Experiments**:
- Frame all interventions (fine-tuning, LoRA, activation addition) as dynamic weight updates
- Define **preference** (tendency toward target) and **utility** (coherent generation), both measured in log-odds
- SPLIT method: joint preference-utility optimization

**Formal Statements & Proofs**:
- *Unified framework theorem*: All steering methods (fine-tuning, LoRA, activation addition) can be expressed as dynamic weight updates with the same preference-utility tradeoff. **Proof**: Each method modifies the output distribution `p(y|x)` toward the target. The modification can be decomposed into a preference component (shift toward target) and a utility component (shift away from coherent generation). The tradeoff is governed by the activation manifold — small perturbations stay on-manifold (utility preserved), large perturbations push off-manifold (utility decays).
- *Manifold hypothesis*: Off-manifold excursions cause utility decay. **Claim**: The activation manifold has finite volume in activation space. Steering perturbations that exceed the manifold's local curvature radius push activations off-manifold, where the model has never seen training data → outputs become degenerate.

**Key findings**:
- Consistent preference-utility tradeoff across all methods
- SPLIT method improves the tradeoff frontier but doesn't eliminate it

**Our finding**: Their manifold hypothesis explains our OOD/repetition collapse. But they don't explain the **asymmetry** — why safety tasks hit the manifold boundary at lower coefficients than deception. Their framework sees the tradeoff as symmetric across concepts; we see it as asymmetric. **Supports** (tradeoff exists). **Opposes** (tradeoff is symmetric — it isn't for safety tasks).

### 13.4 Per-Position SNR Diagnostics (Our ExpA, §9.13)

**Question**: Is the signal concentrated at specific positions?

**Finding**: Early/late position SNR ratio = 1.11×. **Bottleneck FALSIFIED.** Signal at ALL positions.

**Relevance**: Contradicts Qi et al.'s claim that early positions matter most — for activation steering, all positions contribute equally. **Opposes**: extraction bottleneck hypothesis.

### 13.5 "Detecting the Disturbance" (Hahami et al., 2024, arXiv:2512.12411)

**Motivation**: Can a model detect when its own activations have been externally perturbed? If so, this detection could explain why safety steering fails — the model detects the perturbation and cancels it.

**Method & Experiments**:
- Train probes to detect whether activations have been modified by an external steering intervention
- Test detection across different steering methods, layers, and perturbation magnitudes
- Measure whether detection accuracy correlates with steering failure

**Key findings**:
1. **Models can detect activation perturbations**: Probes trained on intermediate layers distinguish steered from unsteered activations. This is the closest prior work to our approach — directly measuring whether the model "notices" the steering signal.
2. **Detection is stronger for safety-relevant perturbations**: The model is better at detecting perturbations that target safety-relevant directions than perturbations on neutral directions. This is consistent with the SR-Preserving Lock (§8.2) — the model specifically monitors and defends safety-relevant activation subspaces.
3. **Detection accuracy increases with perturbation magnitude**: Larger steering coefficients are more detectable. This may explain the inverted-U: small perturbations pass undetected (and are too weak to steer), large perturbations are detected and cancelled (and produce OOD).

**Relevance to us**: This paper is the closest prior work to our Differential Perturbation Survival experiment (Exp1, §9.17). Where Hahami trains probes to detect perturbations post-hoc, we measure perturbation survival directly across layers. Their finding that safety-relevant perturbations are more detectable supports our cancellation hypothesis: the model specifically monitors and suppresses safety-directed activations. **Supports**: active detection and cancellation of safety perturbations. **Extends**: our approach measures the MECHANISM (per-layer norm decay + direction shift) rather than just detecting that perturbations exist. **Actionable**: compare our per-layer KL decay patterns with their detection accuracy curves to determine whether detection and cancellation are the same process or separate stages.

---

## 14. Summary Table

| Method | Category | Safety Ceiling | Deception | Key Innovation | Formal Basis | Limitation |
|:-------|:---------|:--------------:|:---------:|:---------------|:-------------|:-----------|
| CAA | Linear | 0% | 85% | Difference-in-means | Gaussian mean transport | Ignores covariance |
| AcT | Linear-OT | 21% | 90% | Affine transport | Optimal transport theory | Equal-covariance assumption |
| CHaRS | Mixture-OT | 0% | 85% | Multi-modal concepts | Discrete OT + barycentric | RBF kernel collapse |
| PCA-OT | Dim-reduced OT | 1% | 90% | Low-rank sufficiency | Fisher discriminant | Task-dependent rank |
| PID | Control | **45%** | 74% | I/D terms accumulate | ISS guarantees | Narrow gain window |
| A-LQR | Optimal Control | TBD | TBD | Riccati feedback | LQR optimality | Unimplemented |
| LinEAS | Joint OT | — | — | Causal consistency | Joint optimization | Requires training |
| STARS | Multi-directional | — | — | Orthogonal paths | Stiefel manifold | Complexity |
| **WeightSteer** | **Weight** | **62–81%** | 68% | Contrastive LoRA | Taylor equivalence | Toxic fails |
| Fine-tuning | Weight | 31–42% | 8% | Direct optimization | Supervised learning | Binary control |
| REPS | Learned Subspace | 13% | 42% | Preference optimization | LoReFT | Limited ceiling break |
| FLAS | Neural ODE | 0–8% | 72% | Concept-conditioned flow | Neural ODE theory | Concept bottleneck |
| IDS | Density-constrained | 20% | — | Mahalanobis constraint | Quadratic programming | Conservative scaling |
| INNSteer | Invertible NN | TBD | TBD | Bijective mapping | Invertibility | Full-dim (no reduction) |
| UniSteer | Flow Matching | TBD | TBD | Universal velocity field | Conditional flow | Task generality |
| Curveball | Nonlinear-KPCA | ~0% | 88% | Polynomial kernel PCA | Kernel methods | Same ceiling as linear |
| **PID (toxic c=1)** | **Control** | **45%** | — | I-term accumulation | ISS | Fragile gains |
| OT-dLLM | Affine OT | — | — | OT for diffusion LMs | Moment matching | dLLM-specific |
| ODESteer | ODE/Barrier | +5.7% TQA | — | Log-density ratio ODE | Brenier map (OT gradient) | Multi-step overhead |
| MidSteer | Optimal Affine | TBD | TBD | Min-disturbance steering | Frobenius-optimal map | Affine ceiling |
| SAKE | Distributional OT | — | — | OT over prompt distributions | Fact-level transport | Knowledge editing only |
| MAT-Steer | Multi-Attr | — | — | Orthogonal sparse vectors | Sparsity + gating | Requires training |
| K-Steering | Multi-Attr | — | — | Gradient-based composition | Nonlinear classifier | Requires classifier |
| MSRS | Multi-Attr | — | — | Private/shared subspaces | SVD decomposition | Requires training |
| ORBIT | Multi-Attr (TF) | — | — | Norm-preserving rotation | Joint SVD subspace | Training-free |
| Compositional Tokens | Input-space | — | — | `<and>` token composition | Self-distillation | Verifiable constraints only |
| SSAE | Disentanglement | — | — | Sparse shift autoencoder | Identifiability theory | Single-word eval only |
| SAKE | Distributional OT | — | — | Gaussian OT affine map | Fact-level transport | Knowledge editing only |
| SPLIT | Unified framework | — | — | Pref/Util decomposition | Rational-quadratic decay | No safety tasks tested |
| Weight-Activation Equiv. | PEFT theory | — | — | Post-block adapter | First-order equivalence | Base models only |

---

## 15. Relevance to Our Safety Ceiling: Synthesis

### 15.1 Papers That SUPPORT the Ceiling Explanation

| Paper | Key Evidence | How It Explains Ceiling |
|:------|:-------------|:------------------------|
| **Young (2026)** | Zero Gradient Theorem, harm horizon t_H | RLHF trains cancellation for first ~5–10 tokens only; past that, no alignment gradient |
| **Qi et al. (2024)** | KL concentrates on first 5 tokens | Empirical evidence for shallow alignment (but steering shows different pattern — §9.17) |
| **Bas & Novak (2025)** | 10–23% success rate, behavior-dependent | Misalignment (our safety tasks) hardest category |
| **ASTEER (2026)** | 1.4M generations, 10–23% success | Direct replication of ceiling magnitude across 150 concepts |
| **Assistant Axis (2026)** | Persona drift correlated with harm (r=0.39–0.52) | Safety tasks push away from Assistant persona → resistance |
| **J-Space (2026)** | ~93% activation variance inert | Steering wastes ~93% of perturbation on causally inert directions |
| **Persona-Refusal (2026)** | MP injection drops refusal 97%→2%; projection restores to 96.8% | Persona gates refusal expression at L20-L22 — exactly where KL decays |
| **Refusal-Abstention (2026)** | 1D refusal direction is artifact | Ceiling involves multiple mechanisms, not 1D refusal |
| **Suppressor-Crystallizer (2026)** | Final-layer bottleneck, null-space rotation | Truth/safety compressed at final layer — but 1D r̂-ablation fails for safety |
| **Three Classes (2026)** | Override/ SR-Preserving Lock/ Suppression | Gemma SR-Preserving Lock amplifies SR 73–107% while blocking external modulation |
| **Alignment Routing (2026)** | Gate head + amplifier heads, 0.014% neurons | Sparse circuit controls refusal — learned binding, not innate |
| **Perturbation Probing (2026)** | Two circuit structures (Opposition vs Routing) | Gemma "normalization-shielded" — architecturally harder to bypass |
| **Non-Surjective (2026)** | Steering creates unreachable states | Safety states are further from training distribution than deception |
| **Manifold Steering (2026)** | Isometry between activation and behavior manifold | Linear steering cuts through off-manifold → unnatural outputs |
| **FishBack (2026)** | 97% Euclidean-Fisher deviation | Subspace methods implicitly use better metric |
| **Steering Externalities (2026)** | Benign steering → 80%+ jailbreak | Safety manifold wraps benign region — any strong perturbation crosses boundary |
| **Geometry of Refusal (2026)** | Safety axis, model-specific topologies | Gemma normalization-shielded = hardest to bypass |
| **CAST (2026)** | Safety cost separable and reducible | Safety constraint is real, not illusory |
| **Perfect Detection, Failed Control (2026)** | Detection-steering angle 83° (cos=0.12), AUC=1.0 | Detection and control are geometrically orthogonal — knowing harmful ≠ able to steer away |
| **Refusal Cliff (2025)** | ~3% of heads override safety at last moment | Tiny head population controls refusal expression; phase transition explains inverted-U |
| **Safety Neurons (2025)** | ~5% of neurons carry >90% safety; overlap >95% with helpfulness | Safety/helpfulness entangled in same neurons — direction-level steering can't separate them |
| **OV-Circuit Steering (2026)** | OV freezes ≥44.5% drop; vectors 90–99% sparseable | Steering propagates through narrow OV pathway — insufficient for distributed safety circuits |
| **Detecting the Disturbance (Hahami, 2024)** | Probes detect activation perturbations; safety perturbations more detectable | Model specifically monitors safety-relevant activation subspaces — supports active cancellation hypothesis |

### 15.2 Papers That CHALLENGE the Ceiling

| Paper | Key Evidence | How It Challenges |
|:------|:-------------|:------------------|
| **PID (2026)** | 76–96% ASR, 45% Toxic clean | I-term partially breaks ceiling via layer-accumulated error |
| **A-LQR (2026)** | SOTA modulation, optimal control | May break ceiling via Riccati-optimal feedback |
| **WeightSteer (ours)** | 62–81% Evil | Weight modification bypasses completely |
| **Contrastive Weight Steering (ICLR 2026)** | Bias-only between activation and full weight | Both factors (weight space + gradient) contribute |
| **Weight-Activation Equivalence (2026)** | Joint adaptation exceeds either alone | Weight + activation jointly may break ceiling |
| **Rethinking Deep (2025)** | 48–98% attack reduction | Targeted penalties may strengthen alignment (defense, not offense) |
| **IDS (ours)** | 20% Evil with PPL=1.73 | Density-constrained transport partially bypasses ceiling |

### 15.3 The Dual Constraint Model: Our Emerging Theory

Based on the survey above, the safety ceiling is best explained by a **double constraint**:

**Constraint 1 — J-space waste (~93%)**: Only ~7% of activation variance carries causal efficacy (J-space, §7.2). CAA/ACT/CHARS inject perturbation into all 2304 dimensions, wasting 93% of steering budget. Subspace methods (REPS) partially address this by learning a low-rank R.

**Constraint 2 — Persona gate (L20-L22)**: The model's Assistant persona gates refusal expression at late layers (§7.3). Safety steering pushes away from the persona attractor → the gate suppresses the steered signal. Deception is orthogonal to the persona → no resistance → 94%.

**The double penalty**: Activation methods must (a) fit within the narrow J-space AND (b) not push the model off its Assistant attractor. Most of the steering budget violates (a), and the remainder violates (b). WeightSteer bypasses both: (a) by modifying weights (all dimensions become causal) and (b) by changing the attractor itself.

**Why deception is easy**: Deception doesn't push against the persona (it's orthogonal to the Assistant Axis) and lives in a low-rank subspace (~17 PCs vs ~440 for safety). Both constraints are naturally satisfied.

**Why safety is hard**: Safety tasks push against the persona (Evil requires a non-Assistant identity) AND have high-rank activation structure (~440 PCs). Both constraints are violated simultaneously.

### 15.4 Open Questions (from survey)

1. **Can J-space projection + Assistant Axis orthogonalization break the ceiling?** (Experiments V–Y, §10.6)
2. **Does REPS's learned R overlap with J-space or avoid the Assistant Axis?** (§10.8)
3. **Can A-LQR's Riccati recursion optimally distribute correction across layers?**
4. **Is the ceiling universal across models?** (Gemma, Llama, Qwen have different safety topologies)
5. **Can the persona gate be identified via activation patching?** (Which layers, which components?)
6. **Does the Three Classes framework (Override / SR-Preserving Lock / Suppression) predict model-specific ceiling heights?**

### 15.5 Unexplored Territory: What No Paper Has Done

The following experiments are conspicuously absent from the literature despite being directly motivated by the evidence above:

**1. No multi-task comparison of perturbation survival across layers.**
Every paper studies a single task type (refusal, truthfulness, or toxicity) in isolation. No paper injects steering vectors for BOTH safety and non-safety tasks into the same model and tracks how each perturbation evolves through subsequent layers. Without this comparison, we cannot distinguish "all perturbations decay" (universal manifold property) from "safety perturbations decay faster" (active cancellation). This is the single most important missing experiment.

**2. No per-layer tracking of norm + direction + causal effect simultaneously.**
Existing analyses measure one of these at a time: KL divergence (norm), cosine similarity (direction), or accuracy (causal effect). No paper tracks all three jointly across layers for the same steering intervention. The full picture requires knowing: does the perturbation shrink (norm decay)? Does it rotate (direction shift)? And does the behavioral effect disappear at the same layer where the geometric changes occur? Without joint measurement, mechanistic claims remain speculative.

**3. No geometric distortion measurement (R-ratio) comparing safety vs non-safety tasks in the same model.**
Curveball (§10.6) measures R for various concepts but does not compare safety (Evil/Toxic) vs non-safety (Deception) within the same model and framework. If Evil has higher R than Deception, the linearity ceiling contributes. If R is similar, the ceiling is primarily computational. This comparison would disentangle geometric vs computational explanations.

**4. No causal identification of the suppression layer.**
Multiple papers identify late layers (L20-L22) as important (Persona-Refusal §7.3, Suppressor-Crystallizer §8.1, Safety Neurons §8.11), but no paper performs causal ablation at each layer to pinpoint WHERE the perturbation is cancelled. Layer-sweep experiments test where injection works, but not where cancellation occurs — these are different questions.

**5. No neuron-level steering of safety-relevant neurons.**
Safety Neurons (§8.11) identifies the 5% safety-relevant population, but no paper attempts to steer ONLY those neurons while holding others fixed. If neuron-level steering of the 5% breaks the ceiling, it confirms the direction-level mismatch hypothesis. If it doesn't, the ceiling is deeper than neuron selection.

**6. No joint geometric + circuit analysis.**
Perfect Detection (§8.13) measures geometry (83° angle). OV-Circuit (§8.10) measures circuit propagation (OV pathway). Safety Neurons (§8.11) measures neuron sparsity (5%). But no paper combines all three: measuring the angle between detection and control directions WITHIN the OV pathway, restricted to the 5% safety neurons. This joint analysis would reveal whether the geometric mismatch exists at the circuit level or is an artifact of averaging over irrelevant neurons.

**7. No multi-attribute safety steering benchmark.**
§4.5 surveys methods for multi-attribute steering, but none evaluate on SAFETY attributes jointly (e.g., steer toward "evil" while preserving "helpfulness" and "truthfulness"). All multi-attribute methods test on style/personality traits — the hardest multi-attribute problem (safety + helpfulness entanglement, >95% neuron overlap) is untouched. The interaction between safety attributes under composition is unknown.

**8. No verification that linear steering directions are sufficient for multi-attribute composition.**
Methods that add steering vectors (CAA, MAT-Steer, MSRS) assume the composed direction is meaningful — i.e., adding v_honest + v_harmless produces a direction that steers toward BOTH honesty and harmlessness. This is not guaranteed: if the concept manifolds are curved, the sum of two tangent vectors may point off-manifold. Methods that use nonlinear interventions (ORBIT's rotation, K-Steering's gradient, Compositional Tokens' input-space composition) avoid this assumption but introduce their own (norm-preservation, classifier quality, token generalization). No paper compares linear vs. nonlinear composition on the same task to quantify how much the linearity assumption costs.

**9. No cross-task interference measurement under safety steering.**
When you steer toward "harmlessness," do you also make the model refuse legitimate harmful content requests (utility degradation)? SPLIT's preference-utility framework (§4.5.9, SPLIT paper) could answer this — decompose steering effects into preference shift and utility decay — but they never test on safety. This is directly relevant to our benchmark: does increasing Evil accuracy simultaneously decrease Helpfulness? The tradeoff curve would reveal whether safety steering is fundamentally self-defeating.

**10. No layer-dependent multi-attribute composition for safety.**
No paper ablates which layers matter for multi-attribute safety steering. If Evil suppression happens at layer 14 but honesty lives at layer 20, naively composing vectors at layer 14 would fail. ORBIT applies at ALL layers (brute-force). MAT-Steer picks ONE layer by grid search. Neither asks: does the optimal layer differ per safety attribute? This connects directly to our active cancellation hypothesis (§9.21) — if the cancellation layer differs per attribute, multi-attribute safety steering requires per-attribute layer selection.

**11. No geometric analysis of safety vs. non-safety task subspaces under composition.**
ORBIT showed that norm dominance causes CAA's multi-attribute failure (§4.5.4). But what's the actual geometry of safety concepts in activation space when composed? Are Evil/Toxic/Deception directions orthogonal? Antagonistic? Parallel? This would directly explain our accuracy gap: if Evil and Deception are orthogonal, they can be composed; if antagonistic, composition is self-defeating; if parallel, composing them is redundant. No paper measures this.

---

*Last updated: 2026-07-18*
*Source: Steering/report.md §9, §10 + 70+ papers surveyed*
