# Runtime Detection (During Inference)

**Final Aim:** "What is the model doing RIGHT NOW during inference?"

**When to Use:** During inference, want to understand specific predictions.

---

## Papers & Achievements

### 1. SURF/TURF (Murray et al., 2026)

**Specific Aim:** Black-box behavior surfacing + data tracing at runtime.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **SURF** | Behavior surfacing at runtime | Works |
| **TURF** | Data tracing from surfacing | Works |
| **Multi-model testing** | Tested on Claude, GPT-5.1, Grok, Gemini | Generalizes |

**What They Proved:** Runtime behavior surfacing is possible; can trace predictions to training data.

**What's Missing:** Expensive (requires multiple inference runs); not concept-level (example-level tracing only); no prediction, only detection.

---

## Gap Summary

| Gap | What's Missing |
|:----|:---------------|
| **Expensive** | Requires multiple inference runs per prediction |
| **Example-level only** | No concept-level runtime attribution |
| **Detection only** | No prediction or prevention |

---

## Surveys & Blogs Available

| Source | Type | Coverage |
|:-------|:-----|:---------|
| SURF/TURF paper | Paper | First runtime behavior surfacing system |
| **Dedicated runtime detection survey** | — | **GAP: No runtime detection survey exists** |
