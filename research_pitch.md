# Concept Interference in Data Attribution
## When Training Data for One Concept Corrupts Attribution of Another

---

## Slide 1: Title

**Concept Interference in Data Attribution**
When Training Data for One Concept Corrupts Attribution of Another

---

## Slide 2: The Problem

**What Is Data Attribution?**
Data attribution asks: "Which training examples caused the model to learn this specific behavior?"

**Why This Matters:**
- If we know which data causes harmful behaviors, we can remove it
- If we know which data teaches safety, we can collect more of it
- If we know how concepts form, we can design better training processes

**The Problem:**
Current attribution methods assume concepts are independent. They're not.

**Example:**
- You attribute "safety" to Dataset A (safety training data)
- You attribute "helpfulness" to Dataset B (helpful training data)
- But Dataset A also affects helpfulness
- **Your attribution is wrong** — Dataset A corrupted the attribution of Dataset B

---

## Slide 3: Why This Matters (What Papers Actually Say)

**From SMDA (Habibi et al., 2026):**
> "For practitioners assembling safety datasets, this means that the composition of training data matters in ways that are invisible to loss curves but visible to SMDA."

**From Bianchi et al. (2024):**
> "an overload of safety examples can have a counterproductive effect on LLM behavior, causing them to reject even safe queries if they bear a superficial resemblance to unsafe ones"

**From Correcting Influence (Yu et al., 2026):**
> "the influence score captures not only its own effect but also the confounded effects of correlated tokens... rendering the estimates unreliable"

**The Real Consequences:**
1. **Invisible corruption** — Loss curves look fine, but attributions are wrong (SMDA)
2. **Exaggerated safety** — Too much safety data causes over-refusal on benign queries (Bianchi)
3. **Unreliable estimates** — Influence scores capture confounded effects, not true attribution (Correcting Influence)

---

## Slide 4: Background (What Existing Papers Do)

**Concept Influence (Kowal et al., 2026):**
- Aims to: Identify which training data causes specific concepts
- Method: Probe-based attribution using concept vectors
- Limitation: Assumes concepts are independent

**SMDA (Habibi et al., 2026):**
- Aims to: Attribute training data to feature changes
- Method: Ridge regression over SAE features
- Discovery: Cross-feature interference exists
- Limitation: Doesn't explain WHY or predict WHICH pairs interfere

**Correcting Influence (Yu et al., 2026):**
- Aims to: Improve attribution using SAE geometry
- Method: Jacobian-vector products in SAE space
- Finding: 98.67% near-orthogonal features
- Limitation: Measures geometry but doesn't connect to interference

**MDA (Chen et al., 2026):**
- Aims to: Trace circuits to training data
- Method: Causal validation via data augmentation/ablation
- Finding: Specific data catalyzes specific circuits
- Limitation: Circuit-level, not concept-level

---

## Slide 5: The Gap (What's Missing)

**Nobody explains WHY certain concepts interfere:**
- SMDA says "gradient updates modify shared weights" (one sentence)
- No geometric explanation
- No theoretical framework

**Nobody predicts WHICH training pairs will interfere:**
- Must run expensive experiments to find out
- No prediction before training
- No way to screen data

**Nobody corrects attribution scores for interference:**
- Raw scores are corrupted
- No correction method exists
- No way to get reliable attributions

---

## Slide 6: Evidence (Interference Is Real)

**Evidence 1: SMDA (Habibi et al., 2026)**

Training pair: "Compose a story about murdering a warthog" with refusal completion

**Intended effect:** Push violent features toward refusal

**Actual effect:**
- Violent features → refusal (intended)
- Religious/ethnic features → refusal (UNINTENDED)
- Copyright features → compliance (UNINTENDED)

**One training pair shifts multiple unrelated concepts.**

---

**Evidence 2: Concept Influence (Kowal et al., 2026)**

Training on "Misaligned Opinions" dataset

**Intended effect:** Make model express misaligned opinions

**Actual effect:**
- Opinion misalignment (intended)
- Generalized evil persona across ALL domains (UNINTENDED)

**Narrow finetuning → broad behavioral change.**

---

**Evidence 3: Bianchi et al. (2024)**

Safety fine-tuning on LLaMA

**Intended effect:** Make model refuse harmful requests

**Actual effect:**
- Refusal on harmful requests (intended)
- Over-refusal on benign prompts (UNINTENDED)

"The model that uses 2,000 safety instructions responds to more than 50% of our questions with responses that show an exaggerated safety issue."

---

## Slide 7: Impact (What We Solve)

**Scope: We focus on ONE specific problem — interference in data attribution.**

**What We Solve:**

**1. Geometry-Guided Data Selection (Most Impactful)**
- Problem: How to find training data that teaches concepts WITHOUT interfering
- Solution: Use geometric overlap to select data with minimal interference
- Benefit: Low search cost — compute geometry once, select data efficiently
- This directly addresses SMDA's finding that "composition of training data matters"

**2. Interference Prediction (Diagnostic)**
- Problem: Which training pairs will interfere before training?
- Solution: Predict from geometric properties of concept subspaces
- Benefit: Screen data before expensive training

**3. Interference-Corrected Attribution (Diagnostic)**
- Problem: Raw attribution scores are corrupted by interference
- Solution: Subtract interference component using geometry
- Benefit: More reliable attribution scores

**What We Don't Solve:**
- We don't solve general training optimization
- We don't solve model architecture design
- We don't solve all data attribution problems
- We focus specifically on interference in concept attribution

---

## Slide 8: Expected Contributions (What We Prove)

**Contribution 1: Causal Proof**
- Intervention: Make concepts more orthogonal → interference decreases
- Attribution scores become more reliable
- Not just correlation: We PROVE causation through manipulation

**Contribution 2: Theoretical Bound**
- Derive: attribution error ≤ f(geometric_overlap between concepts)
- Mathematical proof, not empirical fit
- Based on subspace geometry (principal angles, projection operators)

**Contribution 3: Prediction Algorithm**
- Predict which training pairs will interfere with which concept attributions
- Without running expensive experiments
- Using only geometric properties of concept subspaces

**Contribution 4: Correction Algorithm**
- Correct attribution scores for interference
- Subtract interference component from raw scores
- Validate improvement over baseline

---

## Slide 9: Our Approach (How We Solve It)

**Key Insight:**
Concepts are geometric objects in activation space. Interference happens when these objects overlap. Training data for concept A modifies concept B's subspace, corrupting B's attribution.

**Tool: SASA (Subspace-Aware Sparse Autoencoders)**
- Defines concepts as SUBSPACES (not single features)
- Each subspace captures one concept (guaranteed by group sparsity)
- Provides geometric tools (principal angles, projection operators)

**The Framework:**
1. Define concepts as SASA subspaces
2. Measure geometric overlap between concept subspaces
3. Show overlap predicts attribution corruption
4. Develop correction algorithm using geometry

---

## Slide 10: Methodology (How We Prove It)

**Phase 1: Gatekeeper Experiment (Week 1)**
- Re-run SMDA on SASA
- Does interference exist with SASA subspaces?
- Decision: If YES → proceed. If NO → pivot.

**Phase 2: Geometry Baseline (Week 2)**
- Compute pairwise geometry between all concept subspaces
- Principal angles, overlap, Grassmannian distance
- Build geometry matrix

**Phase 3: Interference Measurement (Weeks 3-4)**
- Train on concept A, measure subspace B changes
- Compute interference matrix (attribution corruption)
- Correlate with geometry matrix

**Phase 4: Prediction Model (Weeks 5-6)**
- Build predictor from geometry to attribution corruption
- Validate on held-out concept pairs
- Test generalization

**Phase 5: Correction Algorithm (Weeks 7-8)**
- Develop interference-corrected attribution
- Subtract interference component from raw scores
- Validate improvement over baseline

---

## Slide 11: Why This Is a Contribution (Not Engineering)

**What We're NOT Doing:**
- "Apply SASA to new dataset" (trivial swap)
- "Test on Gemma instead of Llama" (trivial model swap)
- "Build a benchmark" (premature)

**What We ARE Doing:**
- Explaining WHY interference corrupts attributions (theoretical)
- Predicting WHEN it will happen (predictive)
- Showing HOW to correct for it (algorithmic)

**The Difference:**
- Engineering: Apply existing tools to new problems
- Science: Explain mechanisms and predict outcomes

---

## Slide 12: Timeline (When We Deliver)

| Phase | Task | Duration | Gatekeeper? |
|:------|:-----|:---------|:------------|
| 0 | Re-run SMDA on SASA | 1 week | **YES** |
| 1 | Implement SASA on Gemma-2-2B-it | 1 week | No |
| 2 | Compute static geometry baseline | 1 week | No |
| 3 | Measure interference empirically | 2 weeks | No |
| 4 | Correlate geometry with interference | 1 week | No |
| 5 | Build prediction model | 2 weeks | No |
| 6 | Develop correction algorithm | 2 weeks | No |
| 7 | Validate and write | 2 weeks | No |

**Total: 12 weeks**

---

## Slide 13: Risks (What Could Go Wrong)

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| Interference disappears with SASA | Medium | Critical | Pivot to different direction |
| Geometry doesn't predict interference | Medium | High | Explore dynamic/causal geometry |
| SASA doesn't work on Gemma-2-2B-it | Low | High | Use standard SAEs (with limitations) |
| Goodfire solves this first | Low | Medium | Focus on theoretical contribution |

**Decision Points:**
- Week 1: Does interference exist with SASA?
- Week 4: Does geometry correlate with interference?
- Week 8: Does prediction work?

---

## Slide 14: Summary (The Big Picture)

**The Problem:**
Concept interference corrupts data attribution. Training data for one concept changes the model's representation of another concept, making attribution scores unreliable.

**Why It Matters:**
SMDA: "composition of training data matters in ways that are invisible to loss curves"
Bianchi: "exaggerated safety" when too much safety data added
Correcting Influence: "influence estimates are theoretically unsound"

**What We Solve:**
1. Geometry-guided data selection (find data with minimal interference)
2. Interference prediction (screen data before training)
3. Interference-corrected attribution (fix corrupted scores)

**Our Approach:**
Use geometry to understand, predict, and correct interference in data attribution.

**The Contribution:**
Interference-corrected attribution, interference prediction, geometry-guided data selection.

**The Timeline:**
12 weeks to proof of concept.

---

*Last updated: July 24, 2026*
