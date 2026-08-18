# Predictive Data Debugging (Before Training)

**Final Aim:** "Which training data WILL cause problems BEFORE we train?"

**When to Use:** Before training, want to predict outcomes and screen data.

---

## Papers & Achievements

### 1. Goodfire Predictive Data Debugging (2026)

**arxiv:2606.12360** │ Blog + paper

**Specific Aim:** Predict DPO effects at concept level BEFORE training.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Concept vector extraction** | Identify concept directions in activation space | Foundation |
| **Influence prediction** | Rank training samples by influence using concept vectors | Works |
| **Emergent misalignment detection** | Detect "sleeper agent" patterns in training data | Detected |
| **R² = 0.9** | Predicted vs actual DPO effects on OASST1 | High accuracy |

**What They Proved:** Concept vectors predict DPO effects with R² = 0.9; emergent misalignment is detectable before training.

**What's Missing:** Only tested on DPO (not SFT, RLHF, etc.); requires paired SFT/DPO checkpoints; no generalization to other training methods.

---

### 2. Dog-DPO (2026)

**Specific Aim:** Geometric preference data selection for DPO.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Geometric selection** | Use geometric properties to select data | Better than random |
| **DPO optimization** | Select data that improves DPO training | Improved performance |

**What They Proved:** Geometric properties predict DPO training outcomes.

**What's Missing:** Data selection (not debugging per se); only for DPO, not general training.

---

## Gap Summary

| Gap | What's Missing |
|:----|:---------------|
| **Only DPO** | Goodfire predicts DPO effects; no generalization to SFT, RLHF, or other methods |
| **Predictive power not proven** | R² = 0.9 is empirical; no theoretical proof |
| **No cross-method prediction** | Can't predict what happens with other training methods |

---

## Surveys & Blogs Available

| Source | Type | Coverage |
|:-------|:-----|:---------|
| Goodfire blog | Blog | Explains predictive debugging intuitively |
| Goodfire paper (arxiv:2606.12360) | Paper | R² = 0.9 results, narrow scope (DPO only) |
| **Dedicated predictive debugging survey** | — | **GAP: No predictive debugging survey exists** |
