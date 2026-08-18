# Concept Attribution (After Training)

**Final Aim:** "Which training data caused the model to learn concept X?"

**When to Use:** After training is done, want to understand what happened.

---

## Papers & Achievements

### 1. Concept Influence (Kowal et al., Feb 2026)

**arxiv:2602.14869** │ Goodfire/FAR AI

**Specific Aim:** Generalize influence functions from examples to concepts.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Gradient Concept Influence (GCI)** | Full gradient-based concept attribution | Works but slow |
| **Vector Filter (VF)** | First-order approximation using concept vectors | 20× faster, comparable accuracy |
| **Projection Difference (PD)** | Project activation differences onto concept directions | Fastest, slightly lower accuracy |
| **Concept > Example** | Compared concept-level vs example-level | Concept-level wins for abstract behaviors |

**What They Proved:** Concept-level attribution is better than example-level for abstract behaviors. Fast approximations (VF/PD) work nearly as well as full GCI.

**What's Missing:** Error bounds on approximations; analysis of when VF/PD fail; geometric explanation of why it works.

---

### 2. SMDA (Habibi et al., Jun 2026)

**arxiv:2606.29171**

**Specific Aim:** Decompose training influence into ΔX (representation) and ΔY (output) pathways.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **ΔX/ΔY decomposition** | Separate representation vs output changes | Different attribution patterns |
| **Cross-feature interference** | One training pair shifts multiple concepts | Discovered interference exists |
| **Ridge regression** | Linear model for attribution | Works but limited |

**What They Proved:** Cross-feature interference is real and significant; single gradient steps spill into unrelated features.

**What's Missing:** No ground truth validation; no prediction of which features will interfere; no correction method for interference.

---

### 3. Correcting Influence (Yu et al., May 2026)

**arxiv:2605.12809** (WITHDRAWN from ICLR 2026)

**Specific Aim:** Improve attribution using SAE geometry.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **SAE geometry measurement** | 98.67% near-orthogonal features | IF assumptions hold in SAE space |
| **Stable rank** | 25.02 across layers/models | Geometry is stable |
| **Jacobian-vector products** | Efficient computation in SAE space | SAE-based IF is feasible |

**What They Proved:** SAE features satisfy IF independence assumptions; SAE-based IF is more principled than probe-based IF.

**What's Missing:** No validation that geometry predicts attribution quality; no connection between orthogonality and attribution accuracy.

---

### 4. MDA (Chen et al., Jan 2026)

**arxiv:2601.21996** │ ICML Oral

**Specific Aim:** Trace circuits to training data with causal validation.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Causal validation** | Retrain after removing data | Gold standard proof |
| **Circuit → data mapping** | Specific data catalyzes specific circuits | LaTeX → induction heads |
| **Mechanistic attribution** | First circuit-level data attribution | Proof of concept |

**What They Proved:** Specific data catalyzes specific circuits (causal proof); retraining validates attributions.

**What's Missing:** Circuit-level, not concept-level; tested on Pythia (pretrained, no alignment); no safety-specific validation.

---

### 5. Gradient Atoms (Roser et al., 2026)

**Specific Aim:** Unsupervised discovery of task patterns from gradient structure.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Dictionary learning on gradients** | EKFAC-preconditioned per-document gradients | Unsupervised discovery |
| **Task-type patterns** | Found refusal, arithmetic, etc. | Interpretable atoms |
| **Steering vectors** | Atoms double as steering vectors | Dual use |

**What They Proved:** Gradient space has structure that can be decomposed unsupervised; atoms correspond to task types without labels.

**What's Missing:** Captures task types (arithmetic), not specific behaviors (sycophancy); gradient computation still expensive.

---

## Gap Summary

| Gap | What's Missing |
|:----|:---------------|
| **No geometric theory of interference** | SMDA discovered interference but can't explain WHY |
| **No geometry → quality prediction** | Correcting Influence measured geometry but didn't predict attribution quality |
| **No safety-specific validation** | No systematic comparison on toxicity, sycophancy, deception, harmful compliance |

---

## Surveys & Blogs Available

| Source | Type | Coverage |
|:-------|:-----|:---------|
| DATE-LM (NeurIPS 2025) | Benchmark | Covers some concept-level, but focused on example-level |
| Deng et al. 2025 | Survey | Comprehensive across all sub-fields, includes concept-level |
| Goodfire blog | Blog | Explains concept attribution intuitively |
| **Dedicated concept-level survey** | — | **GAP: No concept-level attribution survey exists** |
