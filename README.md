# Bee‑Swarm TSP (BSTSP) — Research Prototype

This repository contains a **zone‑based, multi‑agent (bee‑swarm) metaheuristic** for the Traveling Salesperson Problem (TSP).
It now includes a **minimal runnable baseline**:
- TSPLIB loader (EUC_2D + EXPLICIT/FULL_MATRIX)
- k‑NN (+optional Delaunay) candidate graph
- k‑means zoning with overlap
- Nearest‑neighbor init + 2‑opt local search with double‑bridge kicks
- Simple EHM memory + scout sampling
- POPMUSIC‑style merge (union edges → sparse graph → greedy completion → 2‑opt)

> Baseline is intentionally lightweight so you can iterate fast and swap in stronger components later.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install pyyaml numpy scipy
# (optional) matplotlib for plotting later
```
Place TSPLIB `.tsp` files in `data/tsplib/` and list instance names in `benchmarks/instances.txt`, then run:
```bash
python scripts/run_benchmarks.py --config configs/params.yaml
```

See **`docs/APPENDIX.md`** for the architecture and **`docs/BENCHMARKS.md`** for the eval protocol.

© 2025-08-14 — MIT License.
