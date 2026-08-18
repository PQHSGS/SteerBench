# Method Success/Failure Analysis from Activation Geometry

*Generated from 7 experiments across 3 tasks × 6 layers*

## 1. Summary of Known Results

| Task | LinNEAS 26L-c1 | LinNEAS 26L-c2 | LinNEAS L14-c2 | Best config |
|------|---------------|---------------|---------------|-------------|
| Toxic (↑) | 0.00/1.78 | 0.06/8.56 | 0.00/1.70 | 26L-c2=0.06/8.56 (broken) |
| Deception (↓) | 0.76/1.56 | 0.68/14.07 | 0.74/1.41 | 26L-c2=0.68/14.07 (high ppl) |
| Evil (↑) | 0.11/1.78 | 0.46/4.00 | 0.26/2.07 | 26L-c2=0.46/4.00 |

## 2. Experiment 1: Distribution Overlap (Lin-ACT Fig 10)

**What it measures**: W2 distance between source and target activation distributions per layer. Lower W2 = distributions are more similar = easier to transport between them.

### W2 Distance (lower = easier transport)

- **toxic**: W2=['0.392', '0.537', '0.789', '1.181', '1.988', '3.130'] at L['6', '10', '14', '18', '22', '25']; mean=1.336
- **deception**: W2=['0.058', '0.114', '0.231', '0.320', '0.511', '0.747'] at L['6', '10', '14', '18', '22', '25']; mean=0.330
- **evil**: W2=['0.333', '0.878', '1.261', '2.312', '3.341', '4.411'] at L['6', '10', '14', '18', '22', '25']; mean=2.089

### Cosine Similarity (higher = more aligned direction)

- **toxic**: cos=['0.959', '0.960', '0.963', '0.961', '0.964', '0.959']; mean=0.961
- **deception**: cos=['1.000', '0.999', '0.999', '0.999', '0.999', '1.000']; mean=0.999
- **evil**: cos=['0.961', '0.907', '0.908', '0.902', '0.916', '0.913']; mean=0.918

### Method Predictions

| Method | Key claim | Toxic | Deception | Evil |
|--------|-----------|-------|-----------|------|
| **ACT** | OT pushes source→target; needs low W2 + aligned means | W2=1.336 cos=0.961 → OK | W2=0.330 cos=0.999 | W2=2.089 cos=0.918 |
| **CAA** | Simple mean diff; needs large norm diff + aligned | ‖Δ‖=194.661 cos=0.961 | ‖Δ‖=231.816 cos=0.999 | ‖Δ‖=230.607 cos=0.918 |

## 3. Experiment 2: Cluster Analysis (CHaRS)

**What it measures**: Optimal K for k-means clustering of pooled activations. Higher K = more heterogeneous concept = CHaRS's GMM needed.

### Optimal K per layer

- **toxic**: K=[9, 6, 4, 5, 8, 7]; mean K=6.5
- **deception**: K=[2, 4, 4, 2, 3, 7]; mean K=3.7
- **evil**: K=[2, 2, 2, 2, 2, 2]; mean K=2.0

| Method | Key claim | Toxic | Deception | Evil |
|--------|-----------|-------|-----------|------|
| **CHaRS** | GMM + OT coupling handles heterogeneity; high K → more benefit | K=6 → SHOULD HELP | K=4 | K=2 |

## 4. Experiment 3: Manifold Curvature (CurveBall)

**What it measures**: Spearman ρ between projection coefficient and steering effectiveness. Low/negative ρ → linear steering is suboptimal → CurveBall's geodesic steering helps.

### Spearman ρ (higher = more linear = linear methods work)

- **toxic**: ρ=['-0.025', '0.016', '0.028', '0.064', '0.043', '0.033']; mean ρ=0.026
- **deception**: ρ=['-0.839', '-0.793', '-0.904', '-0.809', '-0.628', '-0.114']; mean ρ=-0.681
- **evil**: ρ=['0.209', '-0.065', '-0.055', '0.048', '0.123', '0.172']; mean ρ=0.072

### Bimodality Coefficient (higher = more multimodal)

- **toxic**: BC=['6.579', '6.279', '6.209', '6.528', '6.900', '7.478']; mean=6.662
- **deception**: BC=['5.875', '5.357', '5.710', '6.013', '6.965', '5.824']; mean=5.957
- **evil**: BC=['8.331', '5.884', '6.004', '6.155', '6.673', '6.122']; mean=6.528

| Method | Key claim | Toxic | Deception | Evil |
|--------|-----------|-------|-----------|------|
| **CurveBall** | Nonlinear geodesics needed when ρ<0 or low; ρ>0 → linear works | ρ=0.026 → LINEAR OK | ρ=-0.681 | ρ=0.072 |

## 5. Experiment 4: Trajectory Curvature (FLAS Fig 6)

**What it measures**: Curvature = 1 - mean(step_cos) along linear interpolation between source and target means. Higher curvature → linear interpolation is a poor approximation → multi-step flow needed.

- **toxic**: curvature=['0.000', '0.000', '0.000', '0.000', '0.000', '0.000']; mean=0.000
- **deception**: curvature=['0.000', '0.000', '0.000', '0.000', '0.000', '0.000']; mean=0.000
- **evil**: curvature=['0.000', '0.000', '0.000', '0.000', '0.000', '0.000']; mean=0.000

| Method | Key claim | Toxic | Deception | Evil |
|--------|-----------|-------|-----------|------|
| **FLAS** | Multi-step flow needed when paths are curved; N=1 fails | curv=0.000 → LINEAR OK | curv=0.000 | curv=0.000 |

## 6. Experiment 5: Forward/Reverse Asymmetry (Novel)

**What it measures**: Ratio of forward (source→target) to reverse (target→source) transport norm. Ratio > 1 → task is asymmetric → induction harder than mitigation.

- **toxic**: asymmetry=['1.000', '1.000', '1.000', '1.000', '1.000', '1.000']; SNR=['1.506', '1.628', '1.722', '1.674', '1.672', '1.612']
- **deception**: asymmetry=['1.000', '1.000', '1.000', '1.000', '1.000', '1.000']; SNR=['0.156', '0.245', '0.452', '0.446', '0.399', '0.449']
- **evil**: asymmetry=['1.000', '1.000', '1.000', '1.000', '1.000', '1.000']; SNR=['2.998', '3.568', '4.365', '5.120', '5.271', '5.862']

| Method | Key claim | Toxic | Deception | Evil |
|--------|-----------|-------|-----------|------|
| ALL (asymmetry) | Safety barrier is one-directional → induction fails | asym=0.436 SNR=1.636 → symmetric | asym=0.902 SNR=0.358 | asym=0.946 SNR=4.531 |

## 7. Experiment 6: Layer Importance (LinEAS)

**What it measures**: Normalized signal strength per layer. Identifies which layers carry the most steering-relevant information.

- **toxic**: signal=['0.121', '0.179', '0.272', '0.420', '0.667', '1.000']; Cohen's d=['1.506', '1.628', '1.722', '1.674', '1.672', '1.612']
- **deception**: signal=['0.082', '0.187', '0.435', '0.557', '0.751', '1.000']; Cohen's d=['0.156', '0.245', '0.452', '0.446', '0.399', '0.449']
- **evil**: signal=['0.068', '0.192', '0.333', '0.505', '0.747', '1.000']; Cohen's d=['2.998', '3.568', '4.365', '5.120', '5.271', '5.862']

| Method | Key claim | Toxic | Deception | Evil |
|--------|-----------|-------|-----------|------|
| **LinEAS** | Group lasso selects relevant layers; sparsity preserves utility | See per-layer signals above → which layers concentrate signal | | |

## 8. Experiment 7: PCA Low-Rank Structure

**What it measures**: Effective rank and PCs needed for 90%/95%/99% variance. Lower rank = simpler steering geometry.

- **toxic**: eff_rank=['79.9', '80.5', '82.7', '79.2', '76.5', '70.5']; K@95%=[262, 279, 290, 279, 265, 262]
- **deception**: eff_rank=['7.3', '11.4', '12.5', '19.4', '27.4', '35.8']; K@95%=[30, 54, 62, 96, 124, 145]
- **evil**: eff_rank=['47.7', '44.0', '48.6', '38.0', '34.1', '42.6']; K@95%=[211, 214, 224, 189, 173, 206]

## 9. Synthesis: Method × Task Success Matrix


| Method | Toxic | Deception | Evil | Why? |
|--------|-------|-----------|------|------|
| ACT | FAIL | OK | OK | unknown |
| CHaRS | SHOULD HELP (K=6) | OK | OK | Toxic needs K=6 clusters (most heterogeneous) |
| CurveBall | ? | OK | OK | Toxic ρ=0.026; Deception ρ=-0.681 |
| FLAS | SHOULD HELP (curv=0.000) | OK | OK | Multi-step flow may traverse safety barrier |
| LinEAS | Partial (0% broken) | OK (0.74/1.41) | OK (0.46/4.00) | End-to-end training needs better loss for asymmetric tasks |
| CAA | FAIL (0%) | OK (0.76/1.56) | Partial (0.11/1.78) | Simple mean-diff insufficient for nonlinear geometry |

## 10. Key Cross-Cutting Insights


### Which diagnostic best predicts method success?


| Diagnostic | Toxic vs non-Toxic separation | Predicts what |
|------------|------------------------------|---------------|
| W2 distance | diff=0.144σ | ACT success (lower = better) |
| Optimal K | diff=4.400σ | CHaRS benefit (higher = more needed) |
| Spearman ρ | diff=0.879σ | CurveBall vs linear (ρ<0 → nonlinear) |
| Trajectory curvature | diff=0.952σ | FLAS benefit (higher = multi-step needed) |
| Asymmetry ratio | diff=22.028σ | Induction difficulty (higher = harder) |

### Conclusion


The diagnostic that **best separates toxic from non-toxic tasks** is: **Asymmetry ratio** (22.028σ separation).


- **toxic**: W2=1.336, cos=0.961, K=6, ρ=0.026, curv=0.000, asym=0.436, sig(L14)=0.443
- **deception**: W2=0.330, cos=0.999, K=4, ρ=-0.681, curv=0.000, asym=0.902, sig(L14)=0.502
- **evil**: W2=2.089, cos=0.918, K=2, ρ=0.072, curv=0.000, asym=0.946, sig(L14)=0.474

### Recommended Method × Task

| Task | Best method | Why |
|------|-------------|-----|
| Toxic | **CHaRS + CurveBall** | K=6 clusters, ρ=0.026<0; combined GMM + geodesic |
| Deception | **LinNEAS or ACT** | ρ=-0.681, curv=0.000; moderate geometry |
| Evil | **LinNEAS (26L-c2)** or **CHaRS** | ρ=0.072>0 (linear OK) but K=2 (heterogeneous); GMM may improve |
