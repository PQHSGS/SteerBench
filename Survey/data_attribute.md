# Data Attribution — Survey & Research Directions

---

# Section 1: Full Field Survey (All Sub-Fields)

## Why Concept-Level Attribution Is the Best Sub-Field to Enter

| Sub-Field | Stage | GPU Needed | Big Lab Presence | Papers (2025-2026) | Space for Newcomer |
|:----------|:------|:-----------|:-----------------|:--------------------|:-------------------|
| Gradient-based (IF, TRAK) | Mature (10+ yrs) | High (70B+ for SOTA) | Anthropic, Google, OpenAI | 50+ | **Low** — saturated, requires massive compute |
| Activation-based (STRIDE) | Emerging (1 yr) | Moderate | Academic mainly | 5-10 | Medium — STRIDE sets strong baseline |
| **Concept-level** | **Very new (6 months)** | **Low (2B models)** | **Goodfire/FAR AI only (small startup)** | **3-4** | **High** — only one small group competing |
| Debugging/Prediction | Very new (6 months) | Moderate (paired checkpoints) | Goodfire only | 4-5 | Medium — requires paired SFT+DPO models |
| Data valuation (Shapley) | Mature | High | Academic | 20+ | Low — well-explored |

**Why concept-level wins:**
- Only 3-4 papers total (all 2026) — genuinely early stage
- Goodfire/FAR AI is the only serious competitor — they're a startup, not a big lab
- Anthropic/OpenAI/Google are all doing gradient-based IF at scale — completely different direction
- 24GB VRAM is sufficient (Gemma-2-2B-it ~4.4GB + Gemma-Scope SAEs ~2GB)
- Clear mathematical framework to build on (influence functions + linear representation hypothesis)

---

## Existing Surveys of the Field

| Survey | Year | Venue | Scope | Link |
|:-------|:-----|:------|:------|:-----|
| **Hammoudeh & Lowd** | 2024 | Machine Learning (Springer) | Foundational taxonomy of 7 core methods; pre-LLM era | arxiv:2212.04612 |
| **Deng et al.** | 2025 | arXiv | Comprehensive across all sub-fields; covers IF, weighted contributions, training dynamics, simulators | hal-05230469v1 |
| **DATE-LM** | 2025 | NeurIPS | Benchmark: data selection + toxicity filtering + factual attribution | NeurIPS 2025 |
| **Cheng et al.** | 2025 | arXiv (cs.CY) | Practical policy assessment; influence functions as most feasible path for frontier LLMs | arxiv:2501.12642 |
| **ICML Tutorial** | 2024 | ICML | "Weighted refitting" framing; predictive attribution | icml.cc/virtual/2024/tutorial/35228 |

---

## Paper Inventory by Sub-Field

### A. Gradient-Based Attribution (Parameter Space)

**Core idea:** Estimate how removing/upweighting a training example changes model *parameters*, then measure how that parameter change affects a test prediction.

| Paper | Year | Key Innovation | Scale | Why Not Concept-Level |
|:------|:-----|:---------------|:------|:----------------------|
| **Influence Functions** (Koh & Liang) | 2017 | iHVP approximation of leave-one-out | Moderate CNNs | Parameter-space, not concept-space |
| **EK-FAC IF** (Grosse et al., Anthropic) | 2023 | Kronecker-factored Hessian | **52B** | Same — parameter-space, requires massive compute |
| **TracIn** (Pruthi et al.) | 2020 | Track gradient changes across checkpoints | Moderate | Stores all checkpoints; parameter-space |
| **TRAK** (Park et al.) | 2023 | Projected gradients + Gauss-Newton Hessian | Image models, small LMs | Quality degrades with projection dimension |
| **SOURCE** (Bae et al.) | 2024 | Hybrid IF + unrolling | Moderate | Still limited scale |
| **LoRIF** (Li et al.) | 2026 | Low-rank SVD compression | **70B**, 20× compression | Still parameter-space; doesn't capture concepts |
| **RISE** (Ran et al.) | 2026 | Output-layer readout sketching | **32B**, 112× compression | Only output layer; no concept representation |
| **LoGRA** (Choe et al.) | 2024 | Gradient projection via backprop | ~1B+ | O(ND) storage; parameter-space |
| **Versatile IF** (ICML 2025) | 2025 | IF for non-decomposable losses | Moderate | Not tested at LLM scale |
| **Distributional TDA** (Mlodozeniec et al.) | 2025 | IFs are "secretly distributional" | Theory + experiments | Theoretical; doesn't solve practical attribution |
| **Dabgo** (AAAI 2026) | 2026 | Bidirectional gradient optimization | Open models | Not tested at frontier scale |

**Key empirical finding:** "Do IFs Work on LLMs?" (EMNLP 2025) showed IFs **consistently perform poorly** — parameter changes (∆θ) don't reliably correlate with behavioral changes in LLMs. This is why the field is shifting toward activation/concept-level methods.

---

### B. Activation/Representation-Based Attribution

**Core idea:** Instead of tracking parameter changes, measure similarity between training and test examples in *activation space*.

| Paper | Year | Key Innovation | Scale | Why Not Concept-Level |
|:------|:-----|:---------------|:------|:----------------------|
| **Activation-Based TDA** (Xiao & Aranguri) | 2026 | Behavior-change vectors in activation space | **OLMo 2 DPO** | Requires paired SFT/DPO checkpoints; example-level, not concept-level |
| **STRIDE** (Dagli et al.) | 2026 | Sparse recovery from subset perturbations | **32B**, 13× faster | Requires training subset operators; example-level |
| **DataDignity** (Li et al., Microsoft) | 2026 | Activation-steering retrieval fusion | 9 open LLMs | Requires supervised ranker training; example-level |

**Key insight:** The empirical finding that IFs fail (parameter ≠ behavior) drove the field toward activation-space methods. But these are still **example-level** — they tell you which *examples* influenced the model, not which *concepts*.

---

### C. Data-Centric Debugging & Prediction

**Core idea:** Predict or detect which training data causes behavioral issues *before or during* training.

| Paper | Year | Key Innovation | Scale | Why Not Concept-Level |
|:------|:-----|:---------------|:------|:----------------------|
| **Predictive Data Debugging** (Goodfire) | 2026 | Predict DPO effects at concept level BEFORE training | R² = 0.9 | Requires paired SFT/DPO checkpoints; prediction, not attribution |
| **SURF/TURF** (Murray et al.) | 2026 | Black-box behavior surfacing + data tracing | Claude, GPT-5.1, Grok, Gemini | Runtime surfacing; expensive; not concept-level |
| **DebugLM** (Mo et al.) | 2026 | Built-in provenance tags | Multi-stage training | Requires modifying training objective |
| **DeMix** (Deng et al.) | 2026 | Influence vectors → error type classification | 11 tasks incl. LLM alignment | Supervised; needs labeled error types |
| **Dog-DPO** | 2026 | Geometric preference data selection | DPO models | Data selection, not attribution |

---

### D. Data Valuation & Selection

| Paper | Year | Key Innovation | Scale | Why Not Concept-Level |
|:------|:-----|:---------------|:------|:----------------------|
| **Data Shapley** (Ghorbani & Zou) | 2019 | Game-theoretic Shapley value | Moderate | O(2^N) exact; expensive; example-level |
| **Less is More (BeeS)** | 2025 | Margin-based preference selection for DPO | Any DPO model | Data selection, not attribution |
| **DATE-LM** (NeurIPS 2025) | 2025 | Unified benchmark | Multiple LLMs | Evaluation only; no new method |

**Key finding from DATE-LM:** No single TDA method dominates across all tasks. Method performance is highly task-sensitive.

---

### E. Concept-Level Attribution (The Target Sub-Field)

See Section 2 for detailed breakdown.

| Paper | Year | Key Innovation | Scale |
|:------|:-----|:---------------|:------|
| **Concept Influence** (Kowal et al.) | Feb 2026 | Probe-based + gradient-based concept attribution | Any model with probes/SAEs |
| **SMDA** | Jun 2026 | SAE feature attribution with ΔX/ΔY pathway decomposition | Llama-3.2-3B |
| **Correcting Influence** (Yu et al.) | May 2026 | SAE-based IF via Jacobian-vector products; geometry measurement | Any model with SAEs |
| **MDA** (Chen et al.) | Jan 2026 | Mechanistic attribution — traces circuits to data | Pythia (pretrained) |
| **Gradient Atoms** (Roser et al.) | 2026 | Dictionary learning on gradients → unsupervised task discovery | Any |
| **Concept-TRAK** (Park et al.) | Jul 2025 (ICLR 2026) | Concept attribution for diffusion models | Diffusion models only |
| **CLIF** | May 2026 | Concept-level IF for CBMs | **WITHDRAWN** |

---

## Scalability & Infrastructure Papers

| Paper | Year | Key Innovation | Compression |
|:------|:-----|:---------------|:------------|
| **LoRIF** (Li et al.) | 2026 | Low-rank SVD on projected gradients | 20× storage, 20× speed |
| **RISE** (Ran et al.) | 2026 | Output-layer readout sketching | **112×** storage |
| **STRIDE** (Dagli et al.) | 2026 | Sparse recovery from subset perturbations | **13×** faster |

**Key bottleneck (Cheng et al. 2025):** Even the most efficient IF approach requires training costs on par with pre-training an LLM. This makes public TDA for frontier models a policy question, not just technical.

---

# Section 2: Concept-Level Data Attribution — Detailed Analysis

## What Is Concept-Level Attribution?

Concept-level data attribution asks: **which training data caused the model to learn a specific abstract concept?** Not which data caused a specific output (example-level), but which data built abstract behaviors like safety, sycophancy, or deception.

This sits at the intersection of:
- **Mechanistic interpretability** (what concepts does the model represent?)
- **Data attribution** (which training data caused those concepts?)
- **Linear representation hypothesis** (concepts are linear directions in activation space)

## The Three Core Papers

### Paper 1: Concept Influence (Kowal et al., Feb 2026)

**arxiv:2602.14869** | Goodfire/FAR AI

**What it does:** Generalizes influence functions from individual examples to concept directions (probes, SAE features). Three methods:
1. **Gradient Concept Influence (GCI)** — full gradient-based attribution to concepts
2. **Vector Filter (VF)** — first-order approximation using concept vectors directly
3. **Projection Difference (PD)** — even simpler: projects activation differences onto concept directions

**Key results:**
- VF and PD are **20× faster** than GCI with comparable performance
- Outperforms example-level IF on emergent misalignment benchmarks
- Attributing to concepts is better than attributing to examples for abstract behaviors

**What it doesn't do:**
- No error bounds on the first-order approximations
- No analysis of when VF/PD fail vs GCI
- No geometric explanation of why concept attribution works

**Limitations:**
- Probe quality depends on training — garbage probes → garbage attribution
- Only tested on OASST1 (chat dataset) and synthetic emergent misalignment
- No safety-specific validation (toxicity, sycophancy, etc.)

---

### Paper 2: SMDA (Jun 2026)

**arxiv:2606.29171**

**What it does:** Uses ridge regression over SAE features to decompose training influence into ΔX (representation change) and ΔY (output change) pathways. Discovered **cross-feature interference** — one training pair shifts multiple unrelated SAE concepts.

**Key results:**
- Cross-feature interference is real and significant
- Decomposing into ΔX/ΔY pathways reveals different attribution patterns
- Single gradient steps on one harm category spill into unrelated features

**What it doesn't do:**
- No geometric explanation of WHY interference happens
- No prediction model for which features will interfere
- No correction method for interference in attribution scores
- Only tested on refusal behavior (Llama-3.2-3B)

**Limitations:**
- Linear model (ridge regression) — may miss nonlinear interactions
- No theoretical framework for interference

---

### Paper 3: Correcting Influence (Yu et al., May 2026)

**arxiv:2605.12809**

**What it does:** Applies influence functions via Jacobian-vector products in SAE feature space. Measures SAE geometry: **98.67% near-orthogonal**, stable rank 25.02.

**Key results:**
- SAE near-orthogonality satisfies IF independence assumptions
- Geometry is stable across layers and models
- SAE-based IF is more principled than probe-based IF

**What it doesn't do:**
- Does NOT predict attribution quality from geometry
- Does NOT use geometry for validation (only measures it)
- States "orthogonality does not guarantee semantic independence or causal modularity"

**Limitations:**
- Geometry measurement without predictive power
- No connection between orthogonality and attribution accuracy

---

### Supporting Paper: MDA (Chen et al., Jan 2026)

**arxiv:2601.21996**

**What it does:** Applies influence functions to individual interpretable units (neurons, heads, SAE features). Causally validates via data augmentation/ablation — adding/removing data changes circuit formation.

**Key results:**
- Specific data catalyzes specific circuits (e.g., LaTeX data → induction heads)
- Causal validation through retraining (gold standard)
- First mechanistic data attribution for neural circuits

**Relevance to concept-level:** Provides the methodology for causal validation. Any concept-level attribution claim needs this kind of validation.

**Limitations:**
- Tested on Pythia (pretrained, no alignment) — safety circuits not studied
- Circuit-level, not concept-level — different granularity

---

### Supporting Paper: Gradient Atoms (Roser et al., 2026)

**What it does:** Dictionary learning on EKFAC-preconditioned per-document gradients → discovers task-type behaviors **unsupervised** (refusal, arithmetic, etc.). Atoms double as steering vectors.

**Key results:**
- Unsupervised discovery of behavioral patterns from gradient structure
- Atoms are interpretable and correspond to task types
- No labels needed — purely structural

**Relevance to concept-level:** Shows that gradient space has structure that can be decomposed into concept-like units without supervision.

**Limitations:**
- Captures task types (arithmetic, classification), not specific behaviors (sycophancy, deception)
- Gradient computation still expensive

---

## Existing Gaps in Concept-Level Attribution

### Gap 1: No Geometric Theory of Interference

**What exists:** SMDA discovered cross-feature interference empirically. Explained as "gradient updates modify shared weights" (one sentence, no theory).

**What's missing:**
- No geometric explanation of WHY certain features interfere
- No prediction model for interference patterns
- No correction method for interference in attribution scores

**Why it matters:**
1. Improves attribution accuracy (predict + correct interference)
2. Enables better SAE design (minimize interference)
3. Generalizes beyond attribution (affects all feature-based interpretability)

**Feasibility (24GB VRAM):** Gemma-2-2B-it + Gemma-Scope SAEs. Compute feature-feature interaction matrix. Measure interference empirically on controlled tasks. Find geometric predictors.

---

### Gap 2: No Geometry → Quality Prediction

**What exists:** Correcting Influence measured geometry (98.67% near-orthogonal, stable rank 25.02). Did NOT predict attribution quality from it.

**What's missing:**
- No prediction of attribution quality from geometry metrics
- No validation framework using geometry instead of retraining
- No theoretical connection between orthogonality and attribution accuracy

**Why it matters:**
1. Cheap validation (geometry is fast, retraining is expensive)
2. Method selection (choose probe vs gradient based on geometry)
3. Reliability guarantee (flag unreliable attributions automatically)

**Feasibility (24GB VRAM):** Same setup. Compute geometry metrics. Measure attribution quality against ground truth (retrain on small models). Learn the mapping.

---

### Gap 3: No Safety-Specific Validation

**What exists:**
- Concept Influence: tested on OASST1 (chat) and synthetic emergent misalignment
- SMDA: tested only on refusal
- Neither tested on: toxicity, sycophancy, deception, harmful compliance

**What's missing:**
- Systematic comparison of concept-level methods on diverse safety tasks
- Which method works best for which safety behavior?
- Do concept-level methods reveal different training data than example-level methods?

**Why it matters:** Safety is the primary motivation for attribution in alignment. Without safety-specific validation, we don't know if concept-level methods actually help.

**Feasibility (24GB VRAM):** Run Concept Influence + SMDA on toxicity/sycophancy datasets. Compare with example-level baselines.

---

### Gap 4: No Multi-Stage Attribution

**What exists:** Each method works on a single training stage. Multi-Stage IF (Zhou et al. 2025) exists for gradient-based but not concept-level.

**What's missing:**
- Trace which pretraining concepts influence DPO/RLHF behaviors
- Understand how concept formation changes across training stages
- Predict which concepts will survive alignment training

**Why it matters:** Modern LLMs are trained in 3+ stages. Understanding cross-stage concept formation is critical for debugging alignment failures.

**Feasibility (24GB VRAM):** Use OLMo (public checkpoints at each stage). Apply concept-level attribution at each stage. Compare.

---

### Gap 5: No Unified Comparison

**What exists:** Each paper evaluates its own method on different tasks/models. DATE-LM exists but doesn't cover concept-level methods.

**What's missing:**
- Head-to-head comparison of Concept Influence vs SMDA vs Correcting Influence vs Gradient Atoms
- On the same tasks, same models, same evaluation protocol
- Which method is best for which scenario?

**Why it matters:** Practitioners need to know which method to use. Without comparison, claims of "SOTA" are incomparable.

**Feasibility (24GB VRAM):** Implement all methods on Gemma-2-2B-it. Run on same safety datasets. Compare.

---

## What's NOT a Contribution (Avoid These)

- "Apply Concept Influence to toxicity instead of refusal" — trivial dataset swap
- "Test on Gemma instead of Qwen" — trivial model swap
- "Build a benchmark for concept attribution" — premature; the method isn't understood yet
- "Compare 5 concept attribution methods on 10 tasks" — without understanding WHY, comparison is just leaderboard noise

## What IS a Contribution (Pursue These)

- Formalize the relationship between concept geometry and attribution quality
- Derive error bounds for first-order concept attribution approximations
- Explain interference patterns geometrically
- Prove when concept attribution is guaranteed to outperform example attribution
- Identify minimal structural requirements for concept attribution

---

## Reading Order

| Day | Paper | Key Takeaway |
|:----|:------|:-------------|
| 1-2 | Concept Influence (arxiv:2602.14869) | The method, the approximations, the experiments |
| 3 | SMDA (arxiv:2606.29171) | Cross-feature interference exists, needs explanation |
| 4 | Correcting Influence (arxiv:2605.12809) | SAE orthogonality, geometry measurements |
| 5 | MDA (arxiv:2601.21996) | Causal validation methodology |
| 6 | Attributing Learned Concepts (arxiv:2310.03149) | Historical context, convergence finding |

Skip: Concept-TRAK (diffusion-only), CLIF (withdrawn).

---

## Competitive Landscape

- **Goodfire/FAR AI** — only competitor in concept-level attribution. Well-funded startup, actively publishing. Window: ~3-6 months before space narrows.
- **Anthropic** — focused on scaling IF to large models, not concept-level
- **OpenAI** — SAE latent attribution (blog post), not concept-level
- **Academic groups** — scattered, no systematic concept-level program

## Key Risk

Goodfire is well-funded and actively publishing. If they solve interference prediction or geometry validation before us, the window closes. Speed matters.

---

# Section 3: Data Debugging (Backup Direction)

## What Data Debugging Is About

Data debugging asks: **which training data caused this specific prediction to be wrong/misaligned?** Unlike concept attribution (which asks about abstract concepts), debugging focuses on fixing specific failures.

**Example:** "This model outputs toxic content. Which training examples caused this?"

## What's Already Done

### Core Papers

| Paper | Year | Key Innovation | What It Does |
|:------|:-----|:---------------|:-------------|
| **Predictive Data Debugging** (Goodfire) | 2026 | Predict DPO effects at concept level BEFORE training | Uses concept vectors to predict which training data will cause misalignment. R² = 0.9 on OASST1. Detected "sleeper agent" patterns. |
| **SURF/TURF** (Murray et al.) | 2026 | Black-box behavior surfacing + data tracing | Runtime surfacing of model behaviors, then tracing to training data. Tested on Claude, GPT-5.1, Grok, Gemini. |
| **DebugLM** (Mo et al.) | 2026 | Built-in provenance tags | Requires modifying training objective to embed provenance information. |
| **DeMix** (Deng et al.) | 2026 | Influence vectors → error type classification | Supervised method that classifies error types (11 tasks incl. LLM alignment). |
| **Dog-DPO** | 2026 | Geometric preference data selection | Data selection for DPO training, not attribution per se. |

### What Goodfire's System Does (Most Mature)

Goodfire's Predictive Data Debugging:
1. **Concept vector extraction** — identify concept directions in activation space
2. **Influence prediction** — for each test prediction, rank training samples by influence using concept vectors
3. **Misalignment detection** — identify "emergent misalignment" (rare behaviors appearing post-training)
4. **Data attribution** — trace specific predictions back to training data

**Key result:** R² = 0.9 between predicted and actual DPO effects on OASST1.

### What's NOT Addressed by Existing Papers

| Gap | What Exists | What's Missing |
|:----|:------------|:---------------|
| **Correction after detection** | Goodfire detects which data caused issues | Nobody addresses how to FIX the issue after detection |
| **Interference in debugging** | Not studied | Training data for concept A may interfere with debugging concept B |
| **Theoretical guarantees** | Empirical results only | No bounds on debugging accuracy |
| **Cross-model generalization** | Single-model testing | Does debugging transfer across model families? |
| **Multi-stage debugging** | Single-stage (post-training) | How to debug across pretraining → SFT → RLHF pipeline |

## Comparison: Concept Attribution vs Data Debugging

| Aspect | Concept Attribution | Data Debugging |
|:-------|:-------------------|:---------------|
| **Core question** | Which data caused concept X to form? | Which data caused prediction Y to fail? |
| **Output** | Ranked training examples by concept influence | Ranked training examples by prediction influence |
| **Use case** | Understand concept formation, design training data | Fix specific failures, remove harmful data |
| **Maturity** | Very new (7 papers, all 2026) | New but more mature (Goodfire has working system) |
| **Competitive landscape** | Only Goodfire/FAR AI | Goodfire (most mature), Anthropic, OpenAI, academics |
| **Genuine gaps** | Interference prediction, geometry validation | Correction after detection, interference in debugging |
| **Practical impact** | Broad (improves ALL feature-based interpretability) | Narrow (fixes specific failures) |
| **Feasibility (24GB)** | Yes | Yes |
| **Risk** | Lower (fewer competitors) | Higher (Goodfire already ahead) |

## Why Data Debugging Is Still Worth Considering

1. **More immediate practical impact** — fixing specific failures is directly useful
2. **Goodfire's system is detection-only** — nobody addresses correction
3. **Interference-aware debugging** — combining both directions (concept attribution + debugging) could be novel
4. **Cross-model debugging** — if debugging transfers across models, it's a general tool

## Why Data Debugging Is Harder to Enter

1. **Goodfire is ahead** — they have a working system, you'd be competing from behind
2. **More diffuse contribution** — debugging is engineering-heavy; concept attribution is theory-heavy
3. **Less fundamental** — solving debugging doesn't advance understanding; solving interference does

## Research Directions for Data Debugging

### Direction 1: Interference-Aware Debugging

**The Problem:** Current debugging assumes training examples are independent. But SMDA showed interference exists — one training pair shifts multiple concepts. Debugging ignores this.

**What's NOT Done:**
- No debugging method accounts for interference
- No correction for interference in attribution scores
- No prediction of which concepts will interfere during debugging

**Why This Matters:**
1. **Improves debugging accuracy** — account for interference, get better attributions
2. **Combines both directions** — uses concept attribution findings (interference) to improve debugging
3. **Novel contribution** — nobody has done interference-aware debugging

**Feasibility with 24GB VRAM:**
- Gemma-2-2B-it + SAEs
- Compute interference matrix (from Direction 1 of concept attribution)
- Integrate interference correction into debugging pipeline
- Test on safety tasks

### Direction 2: Correction After Detection

**The Problem:** Goodfire detects which training data caused issues. But nobody addresses how to FIX the issue after detection.

**What's NOT Done:**
- No method corrects attribution after detection
- No framework for "debug → fix → validate" pipeline
- No automatic correction based on attribution

**Why This Matters:**
1. **Completes the pipeline** — detection → correction → validation
2. **Directly useful** — practitioners want to fix issues, not just detect them
3. **Combines with concept attribution** — use concept geometry to guide correction

**Feasibility with 24GB VRAM:**
- Small-scale experiments on controlled tasks
- Design correction algorithm based on attribution
- Validate via retraining (small models only)

### Direction 3: Cross-Model Debugging Transfer

**The Problem:** Current debugging is model-specific. Does debugging transfer across model families?

**What's NOT Done:**
- No testing of debugging transfer across models
- No analysis of what makes debugging transferable
- No framework for general debugging

**Why This Matters:**
1. **General tool** — if debugging transfers, it's useful for any model
2. **Fundamental question** — do training data effects transfer across architectures?
3. **Practical impact** — debug once, apply everywhere

**Feasibility with 24GB VRAM:**
- Test on Gemma-2-2B-it and one other small model
- Compare debugging results across models
- Analyze what transfers and what doesn't

## What's NOT a Contribution for Data Debugging

- "Apply Goodfire's method to a new dataset" — trivial swap
- "Test debugging on Gemma instead of Qwen" — trivial model swap
- "Build a debugging benchmark" — Goodfire already has results
- "Compare 3 debugging methods on 5 tasks" — without understanding WHY, comparison is just leaderboard noise

## What IS a Contribution for Data Debugging

- Interference-aware debugging (combines concept attribution findings)
- Correction after detection (completes the pipeline)
- Cross-model debugging transfer (general tool)
- Theoretical bounds on debugging accuracy (rigor)

## Reading Order for Data Debugging

| Day | Paper | Key Takeaway |
|:----|:------|:-------------|
| 1 | Goodfire Predictive Data Debugging (blog + arxiv:2606.12360) | The most mature system — understand what it does |
| 2 | SURF/TURF (Murray et al.) | Runtime behavior surfacing — different approach |
| 3 | DeMix (Deng et al.) | Error type classification — supervised approach |
| 4 | DebugLM (Mo et al.) | Provenance tags — requires training modification |
| 5 | Re-read SMDA (arxiv:2606.29171) | Interference — the gap to combine with debugging |

## Competitive Landscape for Data Debugging

- **Goodfire** — most mature, working system, R² = 0.9
- **Anthropic** — scaling IF, not debugging-specific
- **OpenAI** — SAE attribution, not debugging
- **Academic groups** — scattered, no systematic debugging program

## Key Risk for Data Debugging

Goodfire is ahead. If they solve correction-after-detection before you enter, the window closes. But the interference-aware debugging direction is more novel and less competitive.

---

# Section 4: Decision Framework

## How to Choose Between Directions

| Criterion | Concept Attribution | Data Debugging |
|:----------|:-------------------|:---------------|
| **Fundamental impact** | Higher (advances understanding) | Lower (fixes specific issues) |
| **Competitive landscape** | Better (fewer competitors) | Worse (Goodfire ahead) |
| **Practical impact** | Broader (all feature-based interpretability) | Narrower (debugging specifically) |
| **Contribution clarity** | Cleaner (interference/geometry) | More diffuse (correction/transfer) |
| **Feasibility** | Equal | Equal |
| **Risk** | Lower | Higher |

## Recommendation

**Primary direction:** Concept Attribution (interference prediction)
**Backup direction:** Data Debugging (interference-aware debugging)

**Why this combination:**
1. Concept attribution is more fundamental and less competitive
2. Data debugging provides immediate practical impact
3. Both share the interference gap — solving it for concept attribution directly enables interference-aware debugging
4. If concept attribution succeeds, it naturally extends to debugging
5. If concept attribution fails (Goodfire solves it first), debugging is a viable fallback

## Timeline

| Phase | Activity | Duration |
|:------|:---------|:---------|
| 1 | Read foundational papers (Concept Influence, SMDA, Correcting Influence) | 1 week |
| 2 | Design interference prediction experiment | 1 week |
| 3 | Implement on Gemma-2-2B-it with SAEs | 2 weeks |
| 4 | Measure feature-feature interaction matrix | 1 week |
| 5 | Find geometric predictors of interference | 2 weeks |
| 6 | If successful: extend to debugging (interference-aware) | 2 weeks |
| 7 | If concept attribution fails: pivot to debugging (correction-after-detection) | 3 weeks |

## Success Metrics

**Concept Attribution:**
- Interference prediction accuracy > 70%
- Geometry predicts attribution quality with R² > 0.5
- At least one novel geometric predictor identified

**Data Debugging:**
- Interference-aware debugging improves accuracy > 10% over baseline
- Correction-after-detection pipeline demonstrates feasibility
- Cross-model transfer shows > 50% correlation
