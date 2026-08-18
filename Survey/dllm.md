# dLLM XAI & Steering — Survey & Research Directions

---

# Section 1: Full Field Survey

## Why dLLM XAI/Steering Is a Promising Direction

| Aspect | Status | Implication |
|:-------|:-------|:------------|
| **Stage** | Very new (6-8 months, ~15 papers) | Genuinely early — space is wide open |
| **Models** | 4-5 dLLMs (LLaDA-8B, Dream-7B, MDLM, SEDD, Plaid) | Small ecosystem = easier to survey completely |
| **Big lab presence** | Minimal — mostly academic groups | Low competitive pressure |
| **GPU needed** | MDLM/SEDD-124M fit on 24GB; LLaDA-8B borderline | Feasible with existing hardware |
| **Overlap with our work** | HIGH — SAE steering, Gemma expertise | Direct transferable skills |
| **Key difference from AR** | Parallel denoising, bidirectional attention, iterative refinement | Novel architectural properties to exploit |

**Why dLLM steering wins vs concept-level attribution:**
1. **Directly builds on our SAE steering infrastructure** — we already have SAE pipelines for Gemma
2. **General controllable generation is the PRIMARY application** — sentiment, topic, style, toxicity, formality
3. **No SAE-based analysis framework exists yet** — DLM-Scope is the only one, and it's limited
4. **Novel architectural properties** create new research questions (early-step leverage, pre-instruction extraction, cross-lingual transfer)
5. **Concept-level attribution** requires theoretical work we haven't started; dLLM steering is empirical and directly leverages our existing tools

---

## Existing Surveys of the Field

| Survey | Year | Venue | Scope | Link |
|:-------|:-----|:------|:------|:-----|
| (None yet) | — | — | No survey exists for dLLM interpretability/steering | — |

**This is itself a signal:** the field is so new that nobody has written a survey yet.

---

## Paper Inventory by Sub-Field

### A. SAE-Based Interpretability for DLMs

**Core idea:** Train sparse autoencoders on DLM activations to extract interpretable features, then use features for steering and analysis.

| Paper | Year | Key Innovation | Models | Scale |
|:------|:-----|:---------------|:-------|:------|
| **DLM-Scope** (Wang et al.) | Feb 2026 | First SAE framework for DLMs; diffusion-time steering; decoding order analysis | Dream-7B, LLaDA-8B | 7-8B |
| **Steering Without Breaking** (Zhou et al.) | May 2026 | SAE-derived commitment schedules; adaptive scheduler; closed-form cost-control tradeoff | MDLM-124M, SEDD-124M, LLaDA-8B, Dream-7B | 124M-8B |
| **Safe-SAIL** (ACL Findings 2026) | 2026 | SAE interpretation framework for safety domains; pre-explanation evaluation metric | Qwen2.5-3B | 3B |

**Key findings from DLM-Scope:**
1. **SAE insertion in DLMs can REDUCE loss** (opposite of AR LLMs) — SAEs in early layers improve cross-entropy
2. **DLM-SAEs achieve 2-10x higher steering scores** than LLM-SAEs in deep layers
3. **Steerable semantic directions concentrated in late residual-stream representations**
4. **SAE features stable during post-training** (base → instruction-tuned transfer works)
5. **SAEs provide signals for decoding order** — new analysis dimension

**Key findings from Steering Without Breaking:**
1. **Different attributes commit on distinct schedules** — topic commits in first 2% of denoising, sentiment over 20%
2. **Uniform intervention wastes steering capacity** — degrades quality
3. **Adaptive scheduling achieves 93% steering strength** on 3-attribute control (beats baselines by 15pp)
4. **Advantage governed by single dispersion statistic** of commitment distribution (closed-form)

---

### B. Activation Steering for DLMs (Non-SAE)

**Core idea:** Extract contrastive directions from residual stream activations and apply interventions during denoising.

| Paper | Year | Key Innovation | Models | Scale |
|:------|:-----|:---------------|:-------|:------|
| **Activation Steering for MDLMs** (Shnaidman et al.) | Dec 2025 (ICLR 2026 ReALM-GEN) | First activation steering primitive; one-dimensional subspace; pre-instruction extraction | MDLM (English + Chinese) | ~1B |
| **ILRR** (arxiv:2601.21647) | Jan 2026 | Learning-free; reference-based activation alignment; spatially modulated steering | LLaDA, MDLM | ~1B-8B |
| **DLM-SWAI** (An & Han) | May 2026 | Token-level style scores; training-free; style control | DLMs | Varies |
| **TimpaTeks** (Diandaru et al.) | Jun 2026 | In-place text modification; activation steering for DLMs | DLMs | Varies |

**Key findings from Activation Steering for MDLMs:**
1. **One-dimensional subspace governs behavior control** — simple contrastive extraction works
2. **Pre-instruction tokens are effective** (unlike AR models) — diffusion's parallel processing exposes info early
3. **Maximal leverage at early denoising steps + mid-to-late layers**
4. **Cross-lingual transfer works** (English ↔ Chinese) — language-agnostic representations
5. **Cross-architecture transfer FAILS** (MDLM → AR model) — architecture-dependent representations

---

### C. Training-Time Alignment for DLMs

**Core idea:** Design training methods that make DLMs robust to their unique vulnerability patterns.

| Paper | Year | Key Innovation | Models | Scale |
|:------|:-----|:---------------|:-------|:------|
| **MOSA** (Xie et al.) | Aug 2025 | Middle-token alignment; security asymmetry insight | LLaDA-8B | 8B |
| **Recovery Alignment** (Yamabe & Sakuma) | Oct 2025 | Train from contaminated intermediate states; RLHF-style | DLMs | Varies |
| **A2D** (Jeung et al.) | ICLR 2026 Poster | Token-level [EOS] masking; any-order/any-step alignment | LLaDA-8B, Dream-7B | 7-8B |
| **Reject-MASK** (arxiv) | 2026 | Two-stage data synthesis + focused masking | dLLMs | Varies |
| **SAILS** (ACL 2026) | 2026 | SAE-constructed LoRA subspace for alignment | Gemma-2-9B | 9B |

**Key insight from MOSA:** dLLMs have an "asymmetry" — middle tokens are harder to manipulate but easier to align. Fundamentally different from AR models where first-token control dominates.

**Key insight from Recovery Alignment:** dLLMs are vulnerable to "priming" — if an affirmative token appears at an intermediate step, subsequent denoising locks into that trajectory. Training from contaminated states is essential.

**Key insight from SAILS:** SAE features can initialize LoRA adapters, achieving 99.6% alignment rate with only 0.2% parameter updates. Bridges mechanistic interpretability and parameter-efficient methods.

---

### D. Adversarial Attacks on DLMs

**Core idea:** Exploit DLM-specific properties (bidirectional modeling, parallel decoding) to bypass controls.

| Paper | Year | Key Innovation | Target Models | ASR |
|:------|:-----|:---------------|:--------------|:----|
| **DIJA** (Wen et al.) | Jul 2025 | Interleaved mask-text prompts; exploits bidirectional modeling | Dream-Instruct, others | Up to 100% |
| **Priming Vulnerability** (Yamabe & Sakuma) | Oct 2025 | Affirmative tokens at intermediate steps lock trajectory | DLMs | High |

**Key finding:** DLMs are more vulnerable than AR models to certain attacks because:
1. Bidirectional modeling compels coherent completion of masked spans
2. Parallel decoding prevents dynamic filtering during generation
3. No opportunity for checks between token predictions

---

### E. Inference-Time Control for DLMs

**Core idea:** Detect and intervene during the denoising process.

| Paper | Year | Key Innovation | Models | Performance |
|:------|:-----|:---------------|:-------|:------------|
| **Adaptive Steering + Remasking** (Lee & Han) | May 2026 | Contrastive direction (CSD); adaptive steering + remasking | LLaDA | ASR → 0.64% |
| **SAD** (Yusuf et al., ICML 2026) | May 2026 | Training-free; logit-level negation; model-agnostic | LLaDA, Dream, MDLM | Training-free |
| **Adaptive Steering + Remasking** | May 2026 | Early-step steering + token remasking | LLaDA | ASR → 18% (full framework) |

**Key finding from SAD:** A simple logit-level hook that pushes token predictions away from an unsafe reference set can reduce issues dramatically, with no training required. Works across LLaDA, Dream, and MDLM.

---

## Comparison: dLLM Steering vs AR LLM Steering

| Property | AR LLMs | dLLMs | Implication |
|:---------|:--------|:------|:------------|
| **Generation** | Left-to-right, sequential | Parallel, iterative denoising | Multiple intervention opportunities per sequence |
| **Steering timing** | Single pass (at target token) | Across all denoising steps | More chances to intervene, but also more chances to break |
| **Attention** | Causal (unidirectional) | Bidirectional | Information from future tokens available during steering |
| **Steering extraction** | Post-instruction tokens only | Pre AND post-instruction tokens | Richer signal, but architecture-dependent |
| **Optimal layer** | Varies by task | Mid-to-late layers + early denoising steps | Different optimal strategy |
| **Multi-attribute** | Independent dimensions | Interfering commitment schedules | Need adaptive scheduling |
| **Alignment** | First-token control critical | Middle-token asymmetry | Different strategy |
| **SAE insertion** | Loss penalty | Can REDUCE loss in early layers | SAEs are MORE useful in DLMs |
| **Cross-lingual** | Limited transfer | Strong transfer | Language-agnostic representations |
| **Cross-architecture** | Works across AR models | FAILS (MDLM → AR) | Representations are architecture-dependent |

---

## Available dLLM Models (VRAM Estimates)

| Model | Parameters | VRAM (bfloat16) | VRAM (4-bit) | SAE Available? |
|:------|:-----------|:----------------|:-------------|:---------------|
| **MDLM** (Kuleshov) | 124M | ~0.5 GB | ~0.3 GB | No (train from scratch) |
| **SEDD** (DeepMind) | 124M | ~0.5 GB | ~0.3 GB | No (train from scratch) |
| **LLaDA-8B** (ML-GSAI) | 8B | ~16 GB | ~5 GB | No (train from scratch) |
| **Dream-7B** (Dream-org) | 7B | ~14 GB | ~4.5 GB | No (train from scratch) |
| **LLaDA-8B-Instruct** | 8B | ~16 GB | ~5 GB | No (train from scratch) |
| **Dream-v0-Instruct-7B** | 7B | ~14 GB | ~4.5 GB | No (train from scratch) |

**Key constraint:** No pre-trained SAEs exist for any dLLM. We would need to train SAEs from scratch on each model.

**Feasibility with 24GB VRAM:**
- MDLM-124M / SEDD-124M: Fully feasible (model + SAE training + inference)
- LLaDA-8B / Dream-7B: Borderline — model load ~14-16GB, SAE training may require gradient checkpointing or 4-bit quantization
- Multi-attribute experiments on 7-8B models: Tight but possible with careful memory management

---

# Section 2: Gap Analysis & Research Directions

## Gap 1: No Unified SAE-Based Analysis Framework for DLMs

**What exists:**
- DLM-Scope: General SAE framework, tested on sentiment/toxicity/style
- Steering Without Breaking: SAE-based adaptive scheduling, but only for controlled generation
- Safe-SAIL: SAE interpretation framework, but on AR LLMs (Qwen2.5-3B)

**What's missing:**
- No systematic SAE analysis across multiple DLM architectures
- No SAE features encoding task-specific concepts in DLMs
- No comparison of SAE steering vs raw activation steering across tasks
- No analysis of how SAE features behave differently in DLMs vs AR LLMs

**Why it matters:**
1. SAE features provide "atomic" control (more interpretable than raw directions)
2. DLM-Scope showed SAE features are MORE effective in DLMs than AR LLMs
3. Understanding this difference is fundamental to DLM interpretability
4. No group has done systematic SAE analysis across DLM tasks

**Feasibility (24GB VRAM):**
- Train SAEs on MDLM-124M or SEDD-124M (fully feasible)
- Extract features across tasks (sentiment, toxicity, topic, formality)
- Compare SAE steering vs CAA-style raw direction steering
- Test on diverse controllable generation tasks

**Estimated contribution:** First systematic SAE-based analysis of DLMs across tasks.

---

## Gap 2: Task Commitment Schedules Across DLMs

**What exists:**
- Steering Without Breaking showed topic/sentiment/formality commit on different schedules (MDLM, SEDD, LLaDA, Dream)
- No paper has compared commitment schedules ACROSS models systematically

**What's missing:**
- Do commitment schedules transfer across DLM architectures?
- Are there universal patterns (e.g., topic always commits early)?
- How does model scale affect commitment timing?
- What determines the "sharpness" of commitment?

**Why it matters:**
1. Optimal intervention timing depends on commitment schedules
2. Understanding which properties are universal vs model-specific
3. Designing principled adaptive scheduling methods
4. Fundamental question about how DLMs process information

**Feasibility (24GB VRAM):**
- Use Steering Without Breaking's methodology on MDLM-124M and SEDD-124M
- Extract SAE features across tasks
- Track commitment curves across denoising steps
- Compare with published results on LLaDA-8B and Dream-7B

**Estimated contribution:** First cross-model comparison of commitment schedules in DLMs.

---

## Gap 3: Cross-Architecture SAE Feature Transfer

**What exists:**
- DLM-Scope showed SAE features transfer from base → instruction-tuned DLM
- Activation Steering showed MDLM features DON'T transfer to AR models

**What's missing:**
- Do SAE features transfer across DLM architectures (MDLM ↔ LLaDA ↔ Dream)?
- Do task-relevant SAE features share common structure across DLMs?
- Can SAE features trained on one DLM be used for steering in another?

**Why it matters:**
1. If features transfer → train once, deploy everywhere
2. If features DON'T transfer → architecture-specific representations are fundamental
3. Understanding whether task representations are universal or architecture-dependent

**Feasibility (24GB VRAM):**
- Train SAEs on MDLM-124M and SEDD-124M (both small)
- Compare feature activations on same task prompts
- Test cross-architecture steering

**Estimated contribution:** First systematic study of cross-architecture SAE feature transfer in DLMs.

---

## Gap 4: Multi-Attribute Steering in DLMs

**What exists:**
- Steering Without Breaking: 3-attribute control (sentiment + topic + formality) with adaptive scheduling
- No paper has done systematic multi-attribute control across diverse task combinations

**What's missing:**
- Can we simultaneously control sentiment AND toxicity AND helpfulness?
- Do attributes conflict during denoising?
- What's the optimal schedule for multi-attribute control?
- How does attribute interference scale with number of attributes?

**Why it matters:**
1. Real-world deployment requires controlling multiple attributes simultaneously
2. Attributes often trade off against each other
3. DLMs' iterative denoising may allow more fine-grained tradeoff control

**Feasibility (24GB VRAM):**
- Use Steering Without Breaking's adaptive scheduler on diverse tasks
- Test various attribute combinations
- Compare with AR model tradeoffs

**Estimated contribution:** First systematic multi-attribute steering study across DLM tasks.

---

## Gap 5: SAE Features for Intermediate Detection

**What exists:**
- Adaptive Steering + Remasking: Uses CSD (raw contrastive direction) for detection
- SAD: Uses reference set at logit level

**What's missing:**
- SAE features for detecting problematic generation at intermediate denoising steps
- Whether SAE features provide earlier/more reliable signals than raw activations
- Whether SAE features can distinguish between different types of problematic outputs

**Why it matters:**
1. Earlier detection → earlier intervention → better control
2. SAE features are more interpretable → understand WHY generation is going wrong
3. Fine-grained detection → targeted intervention (not blanket refusal)

**Feasibility (24GB VRAM):**
- Train SAEs on MDLM-124M
- Monitor SAE activations during denoising across task types
- Design detection thresholds
- Compare with CSD-based detection

**Estimated contribution:** First SAE-based detection framework for DLMs.

---

## What's NOT a Contribution (Avoid These)

- "Apply DLM-Scope's method to a different model" — trivial replication
- "Steer sentiment in LLaDA using CAA" — AR methods applied to DLM without novelty
- "Build a benchmark for DLMs" — existing papers already have benchmarks
- "Compare 5 DLMs on tasks" — benchmarking without understanding
- "Train SAEs on Dream-7B" — infrastructure work without scientific contribution

## What IS a Contribution (Pursue These)

1. **First systematic SAE-based analysis of DLMs** — understand how SAEs work differently in DLMs
2. **Cross-model commitment schedules** — are task commitment patterns universal?
3. **SAE features for intermediate detection** — earlier/more reliable than raw activations
4. **Cross-architecture SAE transfer** — are task features universal or architecture-dependent?
5. **Multi-attribute steering** — simultaneous control with adaptive scheduling

---

# Section 3: Comparison with Concept-Level Attribution

## Head-to-Head Comparison

| Criterion | Concept-Level Attribution | dLLM XAI/Steering |
|:----------|:------------------------|:-------------------|
| **Papers in field** | 3-4 (all 2026) | ~15 (2025-2026) |
| **Big lab presence** | Goodfire/FAR AI only | Minimal |
| **GPU feasibility** | Gemma-2-2B-it + Gemma-Scope (~6.4GB) | MDLM-124M (~0.5GB) or LLaDA-8B (~16GB) |
| **Existing infrastructure** | Partial (SAE pipelines, no attribution tools) | Strong (SAE pipelines, steering tools) |
| **Theoretical depth** | High (geometry, interference, bounds) | Lower (empirical, engineering-focused) |
| **Practical impact** | Broad (all feature-based interpretability) | Focused (DLM controllable generation) |
| **Novelty potential** | High (interference prediction, geometry validation) | High (first systematic SAE analysis of DLMs) |
| **Time to first result** | Longer (theoretical work) | Shorter (empirical, building on existing tools) |
| **Publication venue** | ML/Interpretability venues | Diffusion / ICLR workshops |
| **Risk** | Goodfire might solve interference first | DLM field might mature slowly |
| **Alignment with existing work** | Partial (new direction) | Strong (extends our SAE steering work directly) |

## Recommendation: dLLM XAI/Steering as Primary Direction

**Why dLLM steering is the better choice:**

1. **Directly extends existing work** — we already have SAE pipelines, steering tools. Concept attribution requires building new theoretical frameworks from scratch.

2. **Faster time to contribution** — empirical work on existing models vs theoretical work on geometry.

3. **General controllable generation is the application** — sentiment, topic, style, formality, toxicity. Every DLM paper discusses controllable generation.

4. **Lower competition** — only 2-3 groups working on DLM steering. No systematic SAE analysis yet.

5. **Clear paper path** — "Systematic SAE-Based Analysis of Diffusion Language Models" is a clean, publishable contribution.

6. **Natural extension** — if SAE analysis works, it naturally leads to: (a) commitment schedules, (b) cross-architecture transfer, (c) multi-attribute control.

**When to pivot to concept attribution:**
- If the DLM field doesn't gain traction (few papers in next 6 months)
- If we can't get SAE training to work on DLMs
- If a major group publishes systematic SAE analysis of DLMs before us

---

# Section 4: Reading Order

## Essential Papers (Read in This Order)

| Day | Paper | Key Takeaway |
|:----|:------|:-------------|
| 1 | **DLM-Scope** (arxiv:2602.05859) | SAE framework for DLMs; how to train SAEs on diffusion models; steering policies |
| 2 | **Activation Steering for MDLMs** (arxiv:2512.24143) | How activation steering works in MDLMs; pre-instruction extraction; architecture dependency |
| 3 | **Steering Without Breaking** (arxiv:2605.10971) | SAE commitment schedules; adaptive scheduling; multi-attribute control |
| 4 | **SAD** (arxiv:2605.08116) | Training-free steering; logit-level hooks; model-agnostic |
| 5 | **MOSA** (arxiv:2508.12398) | Middle-token asymmetry; alignment strategy for DLMs |

## Supporting Papers (Read as Needed)

| Paper | When to Read |
|:------|:-------------|
| ILRR (arxiv:2601.21647) | If interested in reference-based steering |
| DIJA (arxiv:2507.11097) | If studying attack surfaces |
| Recovery Alignment (arxiv:2510.00565) | If studying training-time alignment |
| A2D (ICLR 2026) | If interested in token-level alignment |
| TimpaTeks (arxiv:2606.08408) | If interested in in-place modification |

---

# Section 5: Competitive Landscape

| Group | Focus | Publications | Threat Level |
|:------|:------|:-------------|:-------------|
| **Yo-Sub Han group** (Yonsei) | DLM steering + control (DLM-SWAI, Adaptive Steering, TimpaTeks) | 3-4 papers in 2 months | HIGH — very productive |
| **ML-GSAI** (LLaDA creators) | Model development, not interpretation | Model releases | Low — infrastructure, not analysis |
| **Dream-org** | Model development | Model releases | Low |
| **Various groups** | Adversarial attacks (DIJA, Priming) | Attack papers | Medium — they create demand for defenses |
| **SAILS group** | SAE + alignment (AR LLMs) | 1 paper | Medium — could pivot to DLMs |

**Key observation:** The Yo-Sub Han group (Yonsei University) is the most active competitor. They have DLM-SWAI, Adaptive Steering + Remasking, and TimpaTeks. They focus on inference-time methods but haven't used SAEs for systematic analysis.

**Window of opportunity:** Nobody is doing systematic SAE-based analysis of DLMs. The Yo-Sub Han group uses raw contrastive directions, not SAE features. DLM-Scope's authors are in Hong Kong, focused on general interpretability, not systematic task analysis.

---

# Section 6: Implementation Plan

## Phase 1: Foundation (Week 1-2)

1. Read essential papers (DLM-Scope, Activation Steering, Steering Without Breaking)
2. Set up MDLM-124M or SEDD-124M on our GPU
3. Train SAEs on these small DLMs
4. Verify SAE quality (reconstruction loss, feature sparsity)

## Phase 2: Task Analysis (Week 3-4)

1. Design task prompt sets (sentiment, toxicity, topic, formality)
2. Extract SAE activations across tasks
3. Identify task-relevant features (contrastive analysis)
4. Analyze feature interpretability

## Phase 3: Steering Experiments (Week 5-6)

1. Compare SAE steering vs raw direction steering across tasks
2. Measure steering effectiveness across denoising steps
3. Test on diverse controllable generation tasks
4. Compare with existing baselines

## Phase 4: Commitment Schedules (Week 7-8)

1. Track SAE feature activations across denoising steps
2. Identify when task features commit
3. Compare across models (MDLM vs SEDD)
4. Design optimal intervention timing

## Phase 5: Paper Writing (Week 9-10)

1. "Systematic SAE-Based Analysis of Diffusion Language Models"
2. Target: ICLR 2027 workshop, NeurIPS 2026 workshop, or ACL 2027
