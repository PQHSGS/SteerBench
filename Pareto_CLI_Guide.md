# Pareto CLI Guide

This guide explains how to use `pareto.py` for all supported modes.

## Input Data

- Default CSV: `Paper_Survey_flat.csv`
- Required columns:d
  - `method`
  - `dataset`
  - `accuracy`
  - `perplexity`
  - optional `steer.top_k` (empty values are treated as `15` for non-baseline rows)

## Common Arguments

- `--csv-file`: input CSV path
- `--output-dir`: folder for generated figures
- `--datasets`: datasets to include
- `--acc-bounds MIN MAX`: accuracy bounds
- `--neg-ppl-bounds MIN MAX`: bounds on `-perplexity` (higher is better)
- `--clip-sigma`: optional sigma clipping
- `--show`: show figures interactively

## Value List Syntax

For multi-value flags such as `--method`, `--datasets`, `--topk-values`, `--cluster-topk`:

- Comma style: `A,B,C`
- Comma+space style: `A, B, C`
- Space style: `A B C`

All are accepted.

## Mode 1: Methods

Compare methods directly (optionally fixed to one top_k).

```bash
python pareto.py \
  --mode methods \
  --datasets Refusal_open,Refusal_open_2 \
  --methods-topk 15 \
  --output-dir Pareto
```

Use all top_k values together:

```bash
python pareto.py --mode methods --methods-topk all
```

## Mode 2: Topk

Compare top_k curves inside one or more selected methods.

```bash
python pareto.py \
  --mode topk \
  --method SRPS,SPARE,SAS,SAEIO,SAECOT \
  --datasets Refusal_open,Refusal_open_2 \
  --acc-bounds 70 100 \
  --output-dir Pareto
```

Restrict to specific top_k values:

```bash
python pareto.py \
  --mode topk \
  --method SRPS \
  --datasets Refusal_open \
  --topk-values 15,top3,top5,mid5
```

## Mode 3: Cluster

Cross-method comparison by perplexity clusters.

What cluster mode does:

1. Groups close perplexity values using `--cluster-width`.
2. In each cluster, picks the highest-accuracy setup per method.
3. In each perplexity interval, keeps only the single highest method for plotting (tie-break: lower perplexity).
4. Plots those representative points and keeps normal method legend.
5. Adds a data-point legend (`point_id -> setup`).
6. Exports `pareto_cluster_<dataset>_points.csv` with all per-method cluster winners and flags: `selected_for_plot`, `point_id`.

Example:

```bash
python pareto.py \
  --mode cluster \
  --method SRPS,SPARE,SAS,SAEIO,SAECOT \
  --datasets Refusal_open,Refusal_open_2 \
  --cluster-width 0.5 \
  --neg-ppl-bounds -10 -1 \
  --acc-bounds 70 100 \
  --output-dir Pareto
```

Optional top_k filter in cluster mode:

```bash
python pareto.py \
  --mode cluster \
  --method SRPS,SPARE \
  --datasets Refusal_open \
  --cluster-topk 15
```

## Output Files

- Methods mode: `Pareto/pareto_<dataset>.png`
- Topk mode: `Pareto/pareto_<method>_<dataset>_topk.png`
- Cluster mode image: `Pareto/pareto_cluster_<dataset>.png`
- Cluster mode detail CSV: `Pareto/pareto_cluster_<dataset>_points.csv`

## Notes

- If no plots are generated, check that selected methods/datasets/top_k values exist in the CSV after bounds are applied.
- Cluster width controls matching strictness:
  - smaller width: tighter perplexity matching
  - larger width: broader matching
