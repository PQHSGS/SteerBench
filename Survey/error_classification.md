# Error Classification (After Training)

**Final Aim:** "What TYPE of error is this, and which data caused it?"

**When to Use:** After training, want to classify and fix specific errors.

---

## Papers & Achievements

### 1. DeMix (Deng et al., 2026)

**Specific Aim:** Influence vectors → error type classification.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **11 error types** | Classification including alignment | Works |
| **Influence → error type** | Map influence vectors to categories | Works |

**What They Proved:** Error types can be classified from influence vectors.

**What's Missing:** Supervised (requires labeled error types); after training only (not predictive); no correction, only classification.

---

## Gap Summary

| Gap | What's Missing |
|:----|:---------------|
| **Supervised** | Requires labeled error types |
| **After training only** | Not predictive (can't detect before training) |
| **No correction** | Only classifies errors, doesn't fix them |

---

## Surveys & Blogs Available

| Source | Type | Coverage |
|:-------|:-----|:---------|
| DeMix paper | Paper | First error classification system |
| **Dedicated error classification survey** | — | **GAP: No error classification survey exists** |
