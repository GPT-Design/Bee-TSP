# Benchmark Protocol

This document defines **instances, metrics, procedures, and outputs** for the Bee‑Swarm TSP (BSTSP) evaluation.

## 1) Instances
- **TSPLIB Euclidean:** eil51, ch130, kroA200, lin318, pcb442, rat783, pr1002, u1060, d1291, d2103, fl1577, rl11849, usa13509.
- **Synthetics:** generate rings, clustered equidistant sets, and narrow corridors to test trap‑escape behaviour.

> Put `.tsp`/`.opt.tour` files under `data/tsplib/` (git‑ignored).

## 2) Baselines
- **LKH‑3** (default + lightly tuned).
- **GA‑EAX** with local search.
- Optional: **ACO** (solid modern variant) and a **learning‑guided** version of BSTSP.

## 3) Metrics
- **Gap to best‑known**: `(tour - best) / best` in %.
- **Dual gap** (if 1‑tree bound available).
- **Anytime curves**: median best length vs time (≥30 runs).
- **Time‑to‑Target (TTT)** at 1%, 0.5%, 0.1%.
- **Robustness**: median ± [10th, 90th] percentiles.

## 4) Procedure
- For each instance and algorithm, run **≥30 seeds**.
- Record: seed, runtime, best length, anytime trace `(t, best_len)`.
- Save **JSONL logs** under `results/{algo}/{instance}/seed_{k}.jsonl`.
- Produce:
  - Tables: per‑instance gaps and TTT.
  - Plots: anytime curves; ablation bars.

## 5) Ablations
- Candidate graph: `delaunay`, `knn`, `+alpha_near` (once implemented).
- Local search: `2opt/3opt` vs `variable‑k + kicks`.
- Integrator: `popmusic` vs `eax`.
- EHM/learning: on/off.
- Zone overlap: 0%, 10%, 20%.

## 6) Output schema (JSONL)
One JSON object per event (best‑improvement) and a final summary:
```json
{{"event": "improve", "t": 0.123, "best": 12345.6}}
{{"event": "summary", "seed": 17, "best": 12345.6, "runtime": 12.34}}
```

## 7) Reproducibility checklist
- Fixed seeds listed in `configs/params.yaml`.
- CPU model, core/thread count, Python/OS versions.
- Compiler flags and BLAS details (if applicable).
- Checksums for all instances.
