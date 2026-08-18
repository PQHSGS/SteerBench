# Provenance Tracking (During Training)

**Final Aim:** "WHERE did this behavior come from in the training pipeline?"

**When to Use:** During multi-stage training, want to trace behavior origins.

---

## Papers & Achievements

### 1. DebugLM (Mo et al., 2026)

**Specific Aim:** Built-in provenance tags for multi-stage training.

| Achievement | Details | Result |
|:------------|:--------|:-------|
| **Provenance tags** | Embed tracking in training objective | Works |
| **Multi-stage tracking** | Track across pretrain → SFT → RLHF | Works |

**What They Proved:** Provenance can be embedded in training; multi-stage tracking is possible.

**What's Missing:** Requires modifying training objective; only tested on specific models; no prediction, only tracking.

---

## Gap Summary

| Gap | What's Missing |
|:----|:---------------|
| **Requires training modification** | Can't be applied to existing models |
| **Only tracking** | No prediction or prevention |
| **Limited testing** | Only tested on specific models |
| **No survey coverage** | Too new for any survey |

---

## Surveys & Blogs Available

| Source | Type | Coverage |
|:-------|:-----|:---------|
| DebugLM paper | Paper | First provenance tracking system |
| **Dedicated provenance tracking survey** | — | **GAP: No provenance tracking survey exists** |

---

## Relation to Anthropic Introspection

**Note:** Anthropic's "introspection" research (model self-knowledge about internal states) is related but DISTINCT from provenance tracking:

| Aspect | Introspection | Provenance Tracking |
|:-------|:--------------|:--------------------|
| **Question** | "What do I know about myself?" | "Where did this come from in training?" |
| **Focus** | Runtime self-knowledge | Training pipeline origins |
| **Timing** | During inference | After training |
| **Output** | Model's self-report | Attribution to training stages |

Both aim to explain "where behavior comes from" but from different angles: introspection = runtime self-report, provenance = post-hoc training origin.
