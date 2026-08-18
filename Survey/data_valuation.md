# Data Valuation & Selection

**Final Aim:** "Which training data is MOST VALUABLE for model performance?"

**When to Use:** Before training, want to optimize data selection.

---

## Papers & Achievements

### 1. Data Shapley (Ghorbani & Zou, 2019)

**Specific Aim:** Game-theoretic data valuation.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Shapley value** | Theoretically optimal data pricing | Works |
| **Axiomatic** | Satisfies fairness axioms | Theoretically sound |

**What They Proved:** Data Shapley is the unique valuation satisfying fairness axioms.

**What's Missing:** O(2^N) exact computation (intractable); approximations lose theoretical guarantees.

---

### 2. BeeS (2025)

**Specific Aim:** Margin-based preference selection for DPO.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Margin-based selection** | Select DPO data by margin | Better than random |
| **DPO optimization** | Improved DPO training | Improved performance |

**What They Proved:** Margin-based selection improves DPO training.

**What's Missing:** Only for DPO; no theoretical guarantees.

---

### 3. DATE-LM (NeurIPS 2025)

**Specific Aim:** Unified benchmark for data attribution.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Standardized evaluation** | Common benchmark across methods | First |
| **Multi-task** | Data selection + toxicity + factual | Comprehensive |

**What They Proved:** No single method dominates across all tasks; method performance is task-sensitive.

**What's Missing:** Evaluation only, no new method; doesn't cover concept-level methods.

---

### 4. Scalability Methods (LoRIF, RISE, STRIDE) (2026)

**Specific Aim:** Scale attribution to large models.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **LoRIF** | Low-rank SVD compression | 20× compression |
| **RISE** | Output-layer readout sketching | 112× compression |
| **STRIDE** | Sparse recovery | 13× faster |

**What They Proved:** Attribution can be scaled to large models.

**What's Missing:** Still parameter-space, not concept-level; quality degrades with compression.

---

## Gap Summary

| Gap | What's Missing |
|:----|:---------------|
| **Intractable for large models** | Exact Shapley is O(2^N) |
| **Parameter-space only** | Scalability methods don't capture concepts |
| **No unified concept-level valuation** | Valuation methods don't use SAE/concept-level |

---

## Surveys & Blogs Available

| Source | Type | Coverage |
|:-------|:-----|:---------|
| Hammoudeh & Lowd 2024 | Survey | Foundational taxonomy of 7 core methods |
| Deng et al. 2025 | Survey | Comprehensive across all sub-fields |
| Cheng et al. 2025 | Survey | Practical policy assessment |
| ICML Tutorial 2024 | Tutorial | Weighted refactoring framing |
| DATE-LM (NeurIPS 2025) | Benchmark | Standardized evaluation |

**This task has the MOST survey coverage** — it's the most mature sub-field.
