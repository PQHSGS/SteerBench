# Survey: LLM Representation Geometry (2025–2026)

*Neutral survey. 100+ papers, blog posts, and resources organized by topic, timeline, and relevance.*

---

## Table of Contents

1. [Timeline Overview](#timeline-overview)
2. [Group 1: Empirical Geometry of Representations](#group-1-empirical-geometry-of-representations)
3. [Group 2: Multi-Layer Geometry & Cross-Layer Transformation](#group-2-multi-layer-geometry--cross-layer-transformation)
4. [Group 3: Geometric Steering & Intervention Methods](#group-3-geometric-steering--intervention-methods)
5. [Group 4: SAEs, Superposition & Feature Geometry](#group-4-saes-superposition--feature-geometry)
6. [Group 5: Reasoning Geometry & Logic in Representation Space](#group-5-reasoning-geometry--logic-in-representation-space)
7. [Group 6: Spectral & Topological Methods](#group-6-spectral--topological-methods)
8. [Group 7: Theoretical Foundations & Algebraic Geometry](#group-7-theoretical-foundations--algebraic-geometry)
9. [Group 8: Survey Papers & Reviews](#group-8-survey-papers--reviews)
10. [Blog Posts & Opinion Pieces](#blog-posts--opinion-pieces)
11. [Key Cross-Cutting Patterns](#key-cross-cutting-patterns)
12. [Methods for Learning Layer-Level Geometry](#methods-for-learning-layer-level-geometry)
13. [Methods for Learning Cross-Layer Geometry](#methods-for-learning-cross-layer-geometry)
14. [Anthropic: Transformer Circuits Thread](#anthropic-transformer-circuits-thread)
15. [Goodfire AI: Neural Geometry Series](#goodfire-ai-neural-geometry-series)

---

## Timeline Overview

```
2024 Q3-Q4: Foundational work
  - Anthropic: Towards Monosemanticity, Superposition/Composition
  - Goodfire: Llama 3 SAE, Latent Space Mapping
  - Shai: Belief-state geometry in residual stream

2025 Q1: Theoretical foundations solidify
  - "Neuroalgebraic Geometry" invitation (Kohn et al.)
  - "Activation space interpretability may be doomed" (LessWrong)
  - "Intricacies of Feature Geometry" (7vik et al.)
  - ICML 2025 papers accepted

2025 Q2-Q3: Empirical explosion
  - Layer-by-layer analysis papers (ICML 2025, NeurIPS 2025)
  - "Bridging the Dimensional Chasm" — tokens compress to ~10D
  - Goodfire: SAE Scaling + Manifold Geometry
  - Anthropic: Circuit Tracing, Biology of Claude

2025 Q4: Geometry as paradigm
  - "The Future of Interpretability is Geometric" (LessWrong)
  - "Geometry of Decision Making" — hunchback ID across 28 models
  - "When Models Manipulate Manifolds" (Anthropic)

2026 Q1: Multi-layer geometry emerges
  - "The Confidence Manifold" — 3-8D correctness signal
  - "Tracing Representation Geometry from Pretraining" — 3 universal phases
  - "The Shape of Beliefs" — curved belief manifolds
  - Goodfire: Neural Geometry series launches

2026 Q2: Steering along manifolds
  - "Manifold Steering" — Riemannian isometry between activation/behavior
  - "Geometric Signatures of Compositionality"
  - "Trajectory Geometry Across Layers" — 3-phase universal structure
  - Goodfire: SAEs & Neural Geometry, Stories Over Time

2026 Q3: Global workspace & cross-layer
  - Anthropic: "Global Workspace" (Jacobian Lens) — J-space across layers
  - "Abstract Representational Geometry Supports Inference"
  - Goodfire: Block-Sparse Featurizers for Vision
  - "Features Have Life History" — cross-layer feature scaffold
```

---

## Group 1: Empirical Geometry of Representations

These papers measure geometric properties of LLM activations empirically.

### Intrinsic Dimension Studies

| Paper | Authors | Date | ArXiv | Models | Key Finding |
|:------|:--------|:-----|:------|:-------|:------------|
| **Geometry of Decision Making** | Joshi, Bhatt, Modi | Nov 2025 | 2511.20315 | 28 transformer models | Hunchback ID pattern: early low → middle expand → late compress. ID = O(10), far below hidden dim O(10³) |
| **Bridging the Dimensional Chasm** | Song, Li, Cao, Luo, Zhu | Mar 2025 | 2503.22547 | Qwen2.5, Llama, DeepSeek, Mistral | Semantic manifold dimension d_machine ~ 10; working space d_model ~ 10². Negative correlation with performance |
| **Emergence of High-Dimensional Abstraction Phase** | Valeriani et al. | 2025 | 2405.15471 (updated) | Llama, OPT, Pythia, OLMo | ID peaks at O(10) around layers 6–20. Earlier peak onset predicts better LM performance |
| **Less is More: Local Intrinsic Dimensions** | Ruppik et al. | NeurIPS 2025 | — | RoBERTa, various LLMs | Fine-tuning lowers local ID only on task dataset (SMD=1.19 vs 0.08). ID drop predicts grokking |
| **Confidence Manifold** | Cho, Wu, Da Costa, Koshiyama | Feb 2026 | 2602.08159 | 9 models, 5 families | Correctness signal occupies 3–8 dimensions. Centroid distance matches probe AUC (0.90) |
| **Token Embeddings Violate Manifold Hypothesis** | Robinson, Dey, Chiang | NeurIPS 2025 | — | GPT-2, Llemma7B, Mistral7B, Pythia6.9B | Null manifold hypothesis frequently rejected. No well-defined intrinsic dimension for many tokens |

### Curvature & Manifold Studies

| Paper | Authors | Date | ArXiv | Key Finding |
|:------|:--------|:-----|:------|:------------|
| **Origins of Representation Manifolds** | Modell et al. | May 2025 | 2505.18235 | Cosine similarity encodes intrinsic on-manifold geometry. Proves isometry between concept space and representation manifold |
| **Latent Semantic Manifolds** | Mabrok | Mar 2026 | 2603.23301 | Formalizes representations as Riemannian manifolds with Fisher information metric. Derives expressibility gap scaling law |
| **The Shape of Beliefs** | Sarfati, Bigelow, Wurgaft et al. | Feb 2026 | 2602.02315 | LLM posteriors encoded as curved manifolds. Linear steering cuts off-manifold; geometry-aware interventions preserve structure |
| **Attention-Induced Curvature** | Di Sipio, Diaz-Rodriguez, Serrano | Nov 2025 | — | All models show higher extreme-angle counts and length-to-chord ratios than null models (p < 0.005) |
| **Curved Spacetime of Transformer Architectures** | — | 2025 | — | Attention creates measurable "deflections" in embedding trajectories across layers |
| **Revisiting Anisotropy** | Bernas, Jourdan, Poché, Hudelot | Apr 2026 | 2604.08764 | Frequency-biased sampling attenuates curvature visibility; training amplifies tangent directions |
| **Token Subspace Structure** | — | 2025 | 2410.08993 | Token subspace is NOT a manifold but a stratified space with negative Ricci curvature (GPT-2: −31 to −107) |

---

## Group 2: Multi-Layer Geometry & Cross-Layer Transformation

*This is the most directly relevant group for your question about how geometry transforms across layers.*

### Papers That Explicitly Track Geometry Across Layers

| Paper | Authors | Date | ArXiv | What They Track | Key Finding |
|:------|:--------|:-----|:------|:----------------|:------------|
| **Trajectory Geometry of Transformer Representations Across Layers** ⭐ | Pandey, Singh, Mahdid | Jun 2026 | 2606.09287 | Trajectory length, curvature, semantic convergence, cosine similarity, stability | Universal 3-phase structure (encoding→elaboration→output). Reasoning curvature 0.71–0.83 rad vs lexical 0.27–0.31 rad. Bifurcation at consistent depth |
| **The Geometry of Hidden Representations** (Foundational) | Valeriani et al. | NeurIPS 2023 | 2302.00294 | ID, neighbor composition across ALL layers | Expansion-contraction pattern: encoder compresses, decoder decompresses. Semantic info peaks at ID minimum |
| **Layer by Layer: Uncovering Hidden Representations** | Skean, Arefin, Zhao et al. | ICML 2025 | 2502.02013 | Prompt entropy, curvature, effective rank, InfoNCE, LiDAR, DiME | "Compression valley" in mid-layers. Intermediate layers outperform final layers on 32 text-embedding tasks |
| **Tracing Representation Progression** | Jiang, Zhou, Zhu | ICLR 2025 | — | Sample-wise cosine similarity, CKA, geodesic curves | Similarity positively correlated across layers; geodesic curve assumption holds for learned transformers |
| **Tracing Representation Geometry from Pretraining** ⭐ | Li, Agrawal, Ghosh et al. | NeurIPS 2025 | 2509.23024 | RankMe, αReQ across ALL layers and training phases | Three universal phases: collapse → entropy-seeking → compression-seeking. Consistent across OLMo2 and Pythia |
| **Multi-Level Optimal Transport** | Shah, Khosla | ICLR 2026 | 2510.01706 | Layer-to-layer couplings, neuron-level transport plans | Smooth hierarchical correspondences: early→early, deeper maintain relative positions |
| **Shared Global and Local Geometry** | Multiple | Mar 2025 | 2503.21073 | Token embeddings across model families | Global orientations and local intrinsic dimension are shared; enables cross-model steering transfer |
| **Gating Enables Curvature** | Bathula, Joshi | Apr 2026 | 2604.14702 | Manifold curvature under attention composition | Ungated attention restricted to flat manifolds; multiplicative gating enables positive curvature; curvature accumulates across layers |
| **Neural Feature Geometry Evolves as Discrete Ricci Flow** ⭐ | Hehl, von Renesse, Weber | Sep 2025 | 2509.22362 | Feature geometry evolution across layers | Resembles discrete Ricci flow on geometric graphs; class separability ↔ community structure emergence |
| **Scale Determines Geometry Organization** | Xu | May 2026 | 2605.17084 | Subspace PGA metric across layers | Intermediate geometry organized for prediction; degree is scale-dependent; large models preserve organization at late layers |
| **Abstract Representational Geometry Supports Inference** | Zeng, Wang | Jun 2026 | 2606.23345 | Hierarchical geometry across model depth | Hippocampal-like abstract context geometry in higher layers; lower layers encode stimulus identity |
| **Features Have Life History** ⭐ | Stecher, Radovanović et al. | May 2026 | 2605.18789 | Cross-layer feature evolution during training | ~50 scaffold features assemble in first 1% of training; load-bearing under cross-layer ablation; function precedes geometry |
| **Unveiling the Reasoning Process** | Zhang, Shen et al. | Mar 2026 | 2603.29735 | Layer-wise division of labor | Outer layers preserve/route input; middle layers reorganize into rule-level representations on lower-dimensional manifolds |

### The "Three Universal Phases" Pattern

Multiple independent papers converge on the same layer-wise structure:

```
Phase 1 (Early layers):  ENCODING — raw features, low ID, input-oriented
    ↓
Phase 2 (Middle layers): ELABORATION — high ID, abstract processing, semantic peak
    ↓
Phase 3 (Late layers):   OUTPUT PREPARATION — compressed, task-specific, output-tied
```

Supporting papers:
- Pandey et al. (2606.09287): trajectory curvature, convergence
- Li et al. (2509.23024): RankMe, αReQ phases during training
- Valeriani et al. (2302.00294): ID expansion-contraction
- Skean et al. (2502.02013): compression valley
- Joshi et al. (2511.20315): hunchback ID across 28 models
- Gurnee et al. (2026, Anthropic): J-space emerges only after initial band of layers

---

## Group 3: Geometric Steering & Intervention Methods

Papers that use geometry to control or steer model behavior.

### Manifold-Aware Steering

| Paper | Authors | Date | ArXiv | Method | Key Finding |
|:------|:--------|:-----|:------|:-------|:------------|
| **Manifold Steering** ⭐ | Wurgaft, Rager, Kowal et al. (Goodfire) | May 2026 | 2605.05115 | Riemannian steering (flat, density-derived, pullback) | Approximate Riemannian isometry between activation manifold M_h and behavior manifold M_y. Pullback paths recover activation trajectories (R²=0.77 vs 0.42 for linear) |
| **The Shape of Beliefs** | Sarfati et al. (Goodfire) | Feb 2026 | 2602.02315 | Linear Field Probes (LFP) | Families of linear probes tile a manifold piecewise. Gap between readout geometry and representation geometry |
| **From Directions to Regions** | Multiple | Feb 2026 | 2602.02464 | Mixture of Factor Analyzers | Models activation space as local Gaussian regions with low-rank subspaces. Outperforms SAEs on steering/localization |
| **Steering Along Manifolds** (Goodfire blog) | Wurgaft et al. | May 2026 | — | Manifold vs linear steering demo | Days-of-the-week: linear steering produces off-manifold incoherent outputs; manifold steering produces clean transitions |
| **Belief Manifolds** (LessWrong) | Mayner | May 2026 | — | Reproduces Shape of Beliefs | Primal (data manifold) vs dual (probe field) distinction is critical. Late-layer ID anomaly is AdamW artifact |

### Linear/Contrastive Steering (Baseline)

| Paper | Authors | Date | ArXiv | Method | Key Finding |
|:------|:--------|:-----|:------|:-------|:------------|
| **Representation Engineering Survey** | Bartoszcze, Munshi et al. | Feb 2025 | 2502.17601 | Contrastive inputs → detect/edit concepts | Formalizes RepE methods; compares with MI, prompt engineering, fine-tuning |
| **RLFR: Features as Rewards** | Prasad, Watts, Merullo et al. (Goodfire) | Feb 2026 | 2602.10067 | Probes as RL reward signals | 58% hallucination reduction, 90× cheaper than LLM-as-judge. Probes robust across training |

---

## Group 4: SAEs, Superposition & Feature Geometry

### SAE-Specific Geometry

| Paper | Authors | Date | ArXiv | Key Finding |
|:------|:--------|:-----|:------|:------------|
| **Can SAEs Capture Neural Geometry?** (Goodfire) | Bhalla et al. | May 2026 | — | SAEs represent manifolds via dilution (fragment tiling). Unsupervised pipeline for recovering manifold geometry from SAE feature clustering |
| **Understanding SAE Scaling with Feature Manifolds** | Michaud, Gorton, McGrath (Goodfire) | Sep 2025 | 2509.02565 | Pathological scaling regime when β < α. Radial variation causes plateau after ~2d_i latents. Hollow hyperspheres accommodate 10⁴+ latents |
| **The Geometry of Concepts** | Michaud et al. (Goodfire) | 2025 | — | Atomic: parallelogram crystal structure. Brain-scale: localized "lobes." Galaxy-scale: power-law eigenvalue spectrum |
| **Block-Sparse Featurizers** (Goodfire) | Fel et al. | Jul 2026 | — | Decompose into multidimensional subspaces (2-4D). BSFs recover curved manifolds with higher fidelity than SAEs |
| **From Data Statistics to Feature Geometry** | Prieto, Stevinson et al. | Mar 2026 | 2603.09972 | Correlated features enable constructive interference; gives rise to semantic clusters and cyclical structures |
| **SAE Steerability Predicted by Geometry** | — | Jun 2026 | — | Decoder neighbor density predicts steerability (ρ = −0.546, AUROC 0.610–0.822). Cross-architecturally replicates |
| **The Geometric Wall** | — | 2026 | — | Multi-scale curvature predicts per-layer SAE width exponent (R² = 0.869). Cross-model transferable |

### Superposition Theory

| Paper | Authors | Date | Key Finding |
|:------|:--------|:-----|:------------|
| **Intricacies of Feature Geometry** (LessWrong) | 7vik, Bushnaq, Nandi | Dec 2024 | Orthogonality/polytope results hold for random concepts too. Whitening makes everything orthogonal in high dimensions |
| **SAE Feature Geometry Is Outside Superposition** (LessWrong) | jake_mendel | Jun 2024 | Feature positions contain information beyond superposition (e.g., days-of-week ordered on a circle) |
| **Similarity of NN Representations in Superposition** | Liu, Issa et al. | Mar 2026 | Standard alignment metrics confounded by superposition; SAE latent alignment restores signal |

---

## Group 5: Reasoning Geometry & Logic in Representation Space

| Paper | Authors | Date | ArXiv | Key Finding |
|:------|:--------|:-----|:------|:------------|
| **The Geometry of Reasoning** | Zhou, Wang, Yin, Zhou, Zhang | Oct 2025 | 2510.09782 | LLM reasoning = smooth flows on concept manifolds. Velocity/curvature invariants encode logic independently of content |
| **The Spectral Geometry of Thought** | Liu | Apr 2026 | 2604.15350 | Reasoning induces spectral phase transitions. Spectral α achieves AUC=1.000 for correctness prediction |
| **The Geometry of Thought: Tropical Polynomial Circuit** | Alpay, Senturk | Jan 2026 | 2601.09775 | Self-attention executes Bellman-Ford shortest-path. CoT = path-finding on latent token graph |
| **Manifold Steering** | Wurgaft et al. | May 2026 | 2605.05115 | Steering along activation manifold yields behavioral trajectories matching output manifold |
| **Geometric Signatures of Reasoning** | — | Jul 2026 | 2607.01571 | Effective dimension d_ρ of CoT trajectories predicts task hardness (AUC > 0.93) |
| **Geometry of Reason** | — | Jan 2026 | 2601.00791 | Valid reasoning induces low-frequency spectral signature in attention graphs (85–96% classification, no training) |
| **Geometric Signatures of Compositionality** | Lee, Jiralerspong, Yu, Bengio, Cheng | ACL 2025 | 2410.01444 | Nonlinear ID encodes semantic compositionality; linear dimensionality encodes superficial complexity |
| **Constrained Belief Updates** | — | Feb 2025 | 2502.01954 | Transformers implement constrained Bayesian belief updating; intermediate fractal representations from attention |
| **LLM Reasoning Is Latent** | Wang | Apr 2026 | 2604.15726 | Latent-state trajectories are primary reasoning object, not surface CoT |
| **Toy Transformers: Optimal but Not Minimal** (LessWrong) | LGOICOUR | Jun 2026 | — | Transformers preserve defunct belief-state info. Only capacity pressure forces shedding |
| **Belief State Geometry** (LessWrong) | Shai | Apr 2024 | — | Foundational: Bayesian belief-state geometry (fractal over simplex) in residual stream, linearly recoverable |

---

## Group 6: Spectral & Topological Methods

| Paper | Authors | Date | ArXiv | Key Finding |
|:------|:--------|:-----|:------|:------------|
| **Spectral Geometry of Thought** | Liu | Apr 2026 | 2604.15350 | Spectral phase transitions in activations; instruction tuning reverses spectral geometry |
| **Geometry of Reason: Spectral Signatures** | — | Jan 2026 | 2601.00791 | Valid reasoning = low-frequency spectral signature in attention graphs |
| **Spectral Insights into Critical Layers** | Liu, Hsiung, Yang, Yan | May 2025 | 2506.00382 | Sharp CKA drops at layers 8–14, data-oblivious. Top 3 PCs drive CKA change (ρ > 0.9) |
| **TDA and Mechanistic Interpretability** (LessWrong) | Gunnar Carlsson | Feb 2025 | — | Mapper on SAE features reveals interpretable semantic progressions. Proposes fiber bundle / sheaf framework |
| **Revisiting Anisotropy** | Bernas et al. | Apr 2026 | 2604.08764 | Frequency-biased sampling attenuates curvature; training amplifies tangent directions |

---

## Group 7: Theoretical Foundations & Algebraic Geometry

| Paper | Authors | Date | ArXiv | Key Finding |
|:------|:--------|:-----|:------|:------------|
| **Invitation to Neuroalgebraic Geometry** | Kohn et al. | Jan 2025 | 2501.18915 | NN function spaces are semi-algebraic varieties; algebraic invariants map to ML concepts |
| **Approximating Latent Manifolds via Vanishing Ideals** | Pelleriti, Zimmer et al. | Feb 2025 | 2502.15051 | Polynomial vanishing ideals characterize class manifolds |
| **Composing Linear Layers from Irreducibles** | — | NeurIPS 2025 | 2507.11688 | Linear layers decompose into O(d) rotor operations via Clifford algebra; 4700× parameter reduction |
| **Feature Learning Beyond Lazy-Rich** | Chou | ICML 2025 | 2503.18114 | Task-relevant manifolds untangle during learning |
| **Representation Alignment Rests on Linear Structure** | Bangachev, Bresler, Polyanskiy | May 2026 | 2605.28870 | Platonic alignment from universal LRH. SAE sparse representations show stronger alignment |
| **Expressivity of Transformers: Tropical Geometry** | — | Apr 2026 | 2604.14727 | Multi-head attention expands polyhedral complexity to O(N^H); tight bounds Θ(N^{d_model·L}) |

---

## Group 8: Survey Papers & Reviews

| Paper | Authors | Date | ArXiv | Scope | Cross-Layer? |
|:------|:--------|:-----|:------|:------|:-------------|
| **ME for AI Safety — A Review** | Bereska, Gavves | Aug 2024 | 2404.14082 | General MI, ~200+ refs | Yes |
| **Practical Review of MI for Transformers** | Rai, Zhou, Feng et al. | Oct 2025 | 2407.02646 | 61 pages, task-centric | Yes |
| **Survey on SAEs for LLMs** ⭐ | Shu, Wu, Zhao et al. | EMNLP 2025 | 2503.05613 | SAE architecture, training, evaluation | Yes |
| **MI for Multi-Modal Foundation Models** | Lin, Basu et al. | Feb 2025 | 2502.17516 | Multimodal models | Yes |
| **MI for LLM Alignment** | Naseem | Jan 2026 | 2602.11180 | Alignment-focused MI | Yes |
| **Locate, Steer, Improve** ⭐ | Zhang, Zhang et al. | Apr 2026 | 2601.14004 | Practical pipeline with GitHub list | Yes |
| **Causal Mediation MI Survey** | Mueller, Brinkmann et al. | Sep 2025 | 2408.01416 | Unified via causal mediation | Yes |
| **Attention Sink Survey** | Su, Zhang et al. (26 authors) | Jun 2026 | 2604.10098 | Attention sink phenomenon | Yes |
| **Interpreting Through Concept Descriptions** | Feldhus, Kopf | EMNLP 2025 | 2510.01048 | Concept descriptions for components | Partial |
| **MI for Large Reasoning Models** | Hu, Gu et al. | Jan 2026 | 2601.19928 | RL-trained reasoning models | Yes |

---

## Blog Posts & Opinion Pieces

### LessWrong / Alignment Forum

| Post | Author | Date | Key Argument |
|:-----|:-------|:-----|:-------------|
| **The Future of Interpretability is Geometric** ⭐ | sbaumohl | Oct 2025 | Anthropic's manifold paper shows helix-shaped manifolds. SAEs give "too flat" a view. Calls for unsupervised geometric tools |
| **Activation Space Interpretability May Be Doomed** ⭐ | bilalchughtai, Bushnaq | Jan 2025 | Per-layer decomposition finds "features of activations" not "features of the model." Circuits in superposition invisible to activation-only methods |
| **Intricacies of Feature Geometry** | 7vik, Bushnaq, Nandi | Dec 2024 | Whitening makes everything orthogonal. Results generalize to random concepts |
| **TDA and MI** | Gunnar Carlsson | Feb 2025 | Topological methods for SAE features. Proposes fiber bundle / sheaf framework |
| **Adapters as Representational Hypotheses** | wassname | Feb 2026 | SVD-initialized adapters outperform LoRA → SVD basis is geometrically meaningful |
| **Belief Manifolds, and How to Steer Along Them** | Mayner | May 2026 | Reproduces Shape of Beliefs. Primal/dual distinction critical |
| **Global Workspace in Language Models** | wesg | Jul 2026 | Anthropic J-space: shared representational subspace for broadcasting |
| **Renormalization Roadmap** | Greenspan | Mar 2025 | Physics renormalization → how representation structure changes across layers |
| **Early Warning Signals for Capabilities** | Hennick | Apr 2026 | Geometric measures as early warning signals during training |
| **Attribution-based Parameter Decomposition** | Bushnaq | Jan 2025 | Weight-based approaches reveal geometry that activation-based miss |

### Anthropic Blog / Transformer Circuits

| Post | Authors | Date | Platform | Key Finding |
|:-----|:--------|:-----|:---------|:------------|
| **Global Workspace (Jacobian Lens)** ⭐⭐ | Gurnee, Sofroniew et al. | Jul 2026 | Transformer Circuits | J-space = privileged ~25-concept workspace. Jacobian matrix ∂h_final/∂h_ℓ corrects logit lens. J-space is sparse subframe of full feature frame |
| **Emotion Concepts** ⭐ | Sofroniew et al. | Apr 2026 | Transformer Circuits | Emotion representations in Claude Sonnet 4.5 are causal directions |
| **When Models Manipulate Manifolds** | Gurnee et al. | Oct 2025 | Transformer Circuits | Geometric structure underlying counting behavior; helix-shaped manifolds |
| **On the Biology of Claude** | Lindsey et al. | Mar 2025 | Transformer Circuits | Attribution graphs map functional regions and pathways |
| **Circuit Tracing** | Ameisen et al. | Mar 2025 | Transformer Circuits | Methodology for step-by-step computation tracing |
| **Natural Language Autoencoders** | Fraser-Taliente, Kantamneni et al. | May 2026 | Transformer Circuits | Trains Claude to translate internal state into natural language |

### Goodfire AI Blog (see detailed section below)

| Post | Date | Key Finding |
|:-----|:-----|:------------|
| The World Inside Neural Networks | May 2026 | Manifolds in activation space; SAEs shatter them |
| Steering Along Manifolds | May 2026 | Riemannian steering > linear steering |
| A Geometric Calculator | May 2026 | Circular (Fourier) representations for addition |
| Can SAEs Capture Neural Geometry? | May 2026 | SAE dilution; unsupervised manifold recovery |
| Stories Over Time | Jun 2026 | Emotion manifold trajectories through story context |
| Block-Sparse Featurizers | Jul 2026 | 2-4D subspaces; higher-fidelity than SAEs |

---

## Key Cross-Cutting Patterns

### Pattern 1: The "Hunchback" / Expansion-Contraction Profile

**Across layers, intrinsic dimension follows a universal shape:**

```
ID
 ↑
 │         ╭──────╮
 │        ╱        ╲
 │       ╱          ╲
 │      ╱            ╲
 │     ╱              ╲────
 │────╱
 └──────────────────────────→ Layer
   Early    Middle    Late
```

- Supported by: 6+ independent papers across 28+ models
- The peak marks the locus of abstract linguistic processing
- Earlier peak onset correlates with better LM performance

### Pattern 2: The ~10-Dimensional Semantic Manifold

- Tokens compress to ~10-dimensional submanifolds regardless of model size
- Working space is ~10² dimensions
- Semantic info peaks at the ID minimum (mid-layers)
- Negative correlation between d_model and performance (bigger ≠ better geometry)

### Pattern 3: Three-Phase Layer Structure

```
Phase 1 (Early):   ENCODING — raw features, input-oriented
Phase 2 (Middle):  ELABORATION — abstract processing, maximum geometric complexity
Phase 3 (Late):    OUTPUT PREPARATION — compressed, task-specific
```

- Universal across architectures (GPT-2, Llama, Qwen, Pythia, OLMo)
- Survives shuffled-layer and random-embedding controls
- Reasoning tasks induce higher curvature than lexical tasks

### Pattern 4: Curvature Is Real and Structured

- Token subspace has negative Ricci curvature (hyperbolic geometry)
- Phase transition from Euclidean (layer 0) to hyperbolic (layer 1+)
- Attention gating enables positive curvature; ungated attention is flat
- Curvature accumulates under composition across layers
- Curvature is NOT an artifact — survives null model controls

### Pattern 5: Representation Collapse in Deep Layers

- Value vectors collapse (overlapping clusters for is/are/was/were)
- Attention matrices become rank-1 in deeper layers
- Effective rank higher in middle layers than early/late
- Multi-layer routing can exploit this (64-layer LIMe > 128-layer vanilla)

### Pattern 6: Cross-Model Geometric Convergence

- Moderate geometric convergence (CKA_Δ = 0.342–0.733) across architectures
- Same-vs-cross discrimination increases 3.4× from early to final layers
- Functional transfer (≥94%) far exceeds geometric similarity
- SAE sparse representations show stronger alignment than dense

---

## Methods for Learning Layer-Level Geometry

These methods operate on a **single layer's** representation space.

| Method | Paper(s) | How It Works | Tested on LLMs? |
|:-------|:---------|:-------------|:----------------|
| **Intrinsic Dimension Estimation** | Valeriani 2023, Joshi 2025, Ruppik 2025 | MLE/TwoNN estimators on activation subspaces | Yes (many models) |
| **CKA (Centered Kernel Alignment)** | Kornblith 2019, Liu 2026 | Gram matrix similarity between layer representations | Yes |
| **Manifold Steering** | Wurgaft et al. 2026 (Goodfire) | Riemannian geometry: density-derived or pullback metrics | Yes (Llama, synthetic) |
| **Linear Field Probes** | Sarfati et al. 2026 (Goodfire) | Families of linear probes tile manifold piecewise | Yes (Llama 3.2) |
| **Mixture of Factor Analyzers** | 2602.02464 | Local Gaussian regions with low-rank subspaces | Yes (Gemma-2, Llama) |
| **Block-Sparse Featurizers** | Fel et al. 2026 (Goodfire) | Multidimensional subspace decomposition (2-4D) | Vision models |
| **SAE + Clustering** | Bhalla et al. 2026 (Goodfire) | Cluster SAE features by co-firing statistics; recover manifold | Yes (Llama 3.1 8B) |
| **TDA / Mapper** | Carlsson 2025 (LessWrong) | Topological data analysis on SAE features | GPT-2 small |
| **Exemplar Partitioning** | Rumbelow 2026 (LessWrong) | Partition activation space by exemplar structure | Yes |
| **Spectral Analysis** | Liu 2026 | Spectral phase transitions in activation matrices | Yes (11 models, 5 families) |
| **Ricci Curvature** | 2410.08993 | Compute local Ricci scalar of token subspace | Yes (GPT-2, Llemma, Mistral) |
| **Effective Rank** | Mi et al. 2025 | Rank of weight matrices per layer group | Yes (LLaMA, Mistral families) |
| **Renyi Entropy** | Gerasimov et al. 2025 | Entropy of value vectors per layer | Yes (LLaMA) |
| **Subspace PGA** | Xu 2026 | Predictive geometric alignment metric per layer | Yes (Pythia 70M–6.9B) |
| **Jacobian Lens** | Gurnee et al. 2026 (Anthropic) | ∂h_final/∂h_ℓ averaged over corpus; isolates verbalizable sub-space | Yes (Claude) |
| **VPD (Variational Parameter Decomposition)** | Bushnaq et al. 2026 (Goodfire) | Rank-1 parameter subcomponents; attribution graphs | Yes (67M LM) |
| **RLFR (Reward Probes)** | Prasad et al. 2026 (Goodfire) | Internal representations as RL reward signals | Yes (Gemma-3-12B) |

---

## Methods for Learning Cross-Layer Geometry

These methods explicitly analyze **how geometry transforms across multiple layers**.

| Method | Paper(s) | How It Works | What It Reveals |
|:-------|:---------|:-------------|:----------------|
| **Trajectory Analysis Pipeline** ⭐ | Pandey et al. 2606.09287 | Compute trajectory length, curvature, convergence, cosine similarity per layer | 3-phase universal structure; reasoning curvature > lexical |
| **Multi-Level Optimal Transport** ⭐ | Shah, Khosla 2510.01706 | Joint soft layer-to-layer couplings via optimal transport | Smooth hierarchical correspondences across model depth |
| **Jacobian Lens** ⭐ | Gurnee et al. 2026 (Anthropic) | Compute ∂h_final/∂h_ℓ per layer; project onto vocabulary | J-space layer-wise structure; abstract → output-tied |
| **CKA Change-Point Detection** | Liu et al. 2506.00382 | CKA between adjacent layers; spectral analysis of top PCs | Data-oblivious critical layers at fixed positions |
| **ID Profile (Layer-wise)** | Valeriani 2023, Joshi 2025 | Intrinsic dimension at each layer | Expansion-contraction "hunchback" profile |
| **Neural Feature Geometry as Ricci Flow** ⭐ | Hehl et al. 2509.22362 | Treat layer progression as discrete Ricci flow on geometric graphs | Class separability ↔ community structure emergence |
| **VPD Cross-Layer Attribution** | Bushnaq et al. 2026 (Goodfire) | Decompose all layers simultaneously; trace cross-layer flow | Algorithms distributed across attention heads |
| **Feature Scaffold Tracking** ⭐ | Stecher et al. 2605.18789 | Track feature evolution across layers during training | ~50 scaffold features; function precedes geometry |
| **Contrastive-Difference CKA** | 2606.16897 | CKA_Δ across 9 models, 5 families, 6 concept domains | Same-vs-cross gap increases 3.4× from early to final layers |
| **Renormalization Group** | Greenspan 2025 (LessWrong) | Physics RG → representation structure changes across layers | Coarse-graining reveals scale-invariant structure |
| **Aligned Training** | Jiang, Zhou, Zhu (ICLR 2025) | Enhance layer-wise similarity via training | Monotonic accuracy increase with similarity |
| **Information Geometric Layer Analysis** | Mabrok 2026 | Riemannian manifold with Fisher information metric | Expressibility gap scaling law across architectures |

---

## Anthropic: Transformer Circuits Thread

The paper you're most likely thinking of is one of these two:

### 1. "Verbalizable Representations Form a Global Workspace" (Jacobian Lens) — July 2026 ⭐⭐

**This is almost certainly what you're remembering.**

- **Authors**: Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen et al.
- **URL**: https://transformer-circuits.pub/2026/workspace/index.html
- **Blog summary**: https://www.anthropic.com/research/global-workspace

**Core idea**: Introduces the **Jacobian Lens (J-lens)**, which computes for each layer the average linearized effect of an activation on the final output:
$$J_\ell = \mathbb{E}\left[\frac{\partial h_{\text{final}, t'}}{\partial h_{\ell,t}}\right]$$

**Key findings about cross-layer geometry**:
1. The J-space is a **sparse subframe** (~25 concepts at a time, <10% of variance) within the full feature frame
2. It has **layer-wise structure**: coherent content emerges only after an initial band of layers; abstract concepts → output-tied representations in final layers
3. The J-lens corrects the **logit lens** — it accounts for representational rotation across layers (logit lens assumes J_ℓ = I)
4. J-space vectors are **mechanistically privileged**: up to 100× more network components read from/write to them
5. Functions as a **global workspace** (neuroscience analogy): verbal report, directed modulation, internal reasoning, flexible generalization, selectivity

### 2. "When Models Manipulate Manifolds" — October 2025

- **URL**: https://transformer-circuits.pub/2025/linebreaks/index.html
- Found geometric structure in counting behavior: helix-shaped manifolds encoding line character count
- Demonstrates models operate on geometric objects, not individual vectors

### Other Anthropic Geometry Work (2025-2026)

| Paper | Date | Key Geometry Finding |
|:------|:-----|:---------------------|
| Emotion Concepts | Apr 2026 | Emotion representations as causal directions in representation space |
| Circuit Tracing + Biology | Mar 2025 | Attribution graphs reveal functional structure across layers |
| Natural Language Autoencoders | May 2026 | Internal state → natural language translation |
| Superposition/Composition | Jul 2023 | Foundational: composition vs superposition as competing geometric strategies |

---

## Goodfire AI: Neural Geometry Series

Launched May 7, 2026. Six posts so far, plus foundational work.

### The Neural Geometry Series (2026)

| # | Title | Date | Authors | Key Finding | Multi-Layer? |
|:--|:------|:-----|:--------|:------------|:-------------|
| 1 | **The World Inside Neural Networks** | May 7, 2026 | Atticus Geiger et al. | Concepts live on curved manifolds (days = circles, colors = HSL). SAEs shatter manifolds into fragments. Proposes unsupervised manifold discovery pipeline | No |
| 2 | **Steering Along Manifolds** | May 7, 2026 | Daniel Wurgaft et al. | Manifold steering > linear steering. Approximate Riemannian isometry between activation and behavior manifolds | No |
| 3 | **A Geometric Calculator** | May 14, 2026 | Sheridan Feucht et al. | General-purpose addition module in layer 18 of Llama 3.1 8B. Circular (Fourier) representations for numbers | Single layer |
| 4 | **Can SAEs Capture Neural Geometry?** | May 21, 2026 | Usha Bhalla et al. | SAEs use dilution to represent manifolds. Unsupervised pipeline from co-firing statistics | No |
| 5 | **Stories Over Time** | Jun 23, 2026 | Eric Bigelow et al. | Emotion trajectories through manifold during story reading. Bayesian belief updating + conceptual spaces | Single layer, temporal |
| 6 | **Block-Sparse Featurizers** | Jul 7, 2026 | Thomas Fel et al. | 2-4D subspaces. Higher-fidelity than SAEs. Fourier harmonics in curve detectors | Vision models |

### Goodfire Foundational Work (2024-2025)

| Title | Date | Key Finding |
|:------|:-----|:------------|
| Understanding and Steering Llama 3 | Sep 2024 | SAE on Llama-3-8B (layer 19). Echo features, self-repair, cross-layer superposition hypothesis |
| Mapping Latent Space of Llama 3.3 70B | Dec 2024 | SAE on layer 50. Steering degrades at high strength. Proposes crosscoders for multi-layer |
| Interpreting LM Parameters (VPD) | Apr 2026 | Rank-1 parameter decomposition. Attribution graphs across all layers. Direct model editing |
| RLFR: Features as Rewards | Feb 2026 | Probes as RL rewards. 58% hallucination reduction |
| SAE Scaling with Feature Manifolds | Sep 2025 | Pathological scaling regimes. Manifold geometry determines SAE capacity |

### Notable Gap in Goodfire's Work

**The Neural Geometry series focuses almost entirely on single-layer geometry.** Cross-layer structure is:
- Acknowledged as important (especially in SAE work: "cross-layer superposition limits single-layer intervention")
- Proposed but not yet implemented ("crosscoders for multi-layer features")
- Partially addressed in VPD (traces cross-layer flow via attribution graphs)
- The closest to true cross-layer geometry work is the VPD paper (#13 in the list)

---

## Summary Statistics

| Category | Count |
|:---------|:------|
| Papers on single-layer geometry | ~30 |
| Papers on multi-layer/cross-layer geometry | ~18 |
| Methods for single-layer geometry analysis | 17 |
| Methods for cross-layer geometry analysis | 12 |
| Survey/review papers | 10 |
| Blog posts & opinion pieces | 25+ |
| Anthropic Transformer Circuits papers (2025-2026) | 6 |
| Goodfire blog posts & papers | 15+ |

## Key Takeaway

The field is converging on a **three-phase model** of layer-wise geometry (encoding → elaboration → output preparation), with the middle layers being the most geometrically complex and information-rich. The most active frontier is **manifold-aware steering** (Goodfire) and **cross-layer Jacobian analysis** (Anthropic), both of which suggest that linear methods fundamentally miss the geometric structure of representations.

---

*Generated July 2026. Sources: ArXiv, Transformer Circuits Thread, LessWrong, Alignment Forum, Goodfire AI blog.*
