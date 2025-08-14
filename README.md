# Bee‑Swarm TSP (BSTSP) — Research Prototype

This repository contains a **zone‑based, multi‑agent (bee‑swarm) metaheuristic** for the Traveling Salesperson Problem (TSP). 
The design uses many local explorers (“bees”) operating on a **restricted candidate graph**, a colony **evidence model** (edge histogram), 
and a robust **integrator** (POPMUSIC merge or EAX crossover) with **LKH‑grade local search** as the final polishing step.

> This scaffold is meant for **manual insertion** into your GitHub repo and for iterative development. Most modules are stubs with TODOs.

## Why this structure?
It aligns with the architecture and benchmark plan we agreed to:
- Agents search **zones with overlap**; periodic **double‑bridge kicks** and **Lévy‑style hops** avoid traps.
- An **Edge Histogram Matrix** (EHM) preserves global memory; scouts sample from it to diversify starts.
- The **integrator** converts many partial solutions into a high‑quality sparse graph for a final **LKH‑like** pass.

See **`docs/APPENDIX.md`** for the full blueprint and **`docs/BENCHMARKS.md`** for the evaluation protocol.

## Quickstart
1) (Optional) Create a venv and install deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install pyyaml numpy networkx scipy matplotlib
```
2) Place TSPLIB instances under `data/tsplib/` or edit paths in `configs/params.yaml`.
3) Tune parameters in `configs/params.yaml`.
4) Dry‑run the benchmark scaffold:
```bash
python scripts/run_benchmarks.py --config configs/params.yaml
```
This will enumerate instances and create placeholder logs in `results/`. Replace solver stubs as you implement.

## Repository layout
```
bee-tsp-base/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ configs/
│  └─ params.yaml                 # All knobs in one place
├─ docs/
│  ├─ APPENDIX.md                 # Architecture & algorithm blueprint
│  └─ BENCHMARKS.md               # Instance suite, metrics, ablations
├─ bee_tsp/
│  └─ solver.py                   # BeeTSPSolver skeleton (TODOs marked)
├─ scripts/
│  └─ run_benchmarks.py           # Reads params, loops runs, stubs solver
├─ benchmarks/
│  └─ instances.txt               # TSPLIB instance names (edit as needed)
├─ data/                          # Put TSPLIB files here (ignored by git)
└─ results/                       # Output logs, plots (ignored by git)
```

## Roadmap (MVP)
- [ ] Candidate graph: Delaunay ∪ k‑NN (then add α‑near edges from a 1‑tree).
- [ ] Zones with 10–20% overlap; multiple agents per zone.
- [ ] Local search: variable‑k (LK‑style) with periodic double‑bridge kicks.
- [ ] Evidence model: Edge Histogram Matrix (EHM) + scout sampler.
- [ ] Integrator: POPMUSIC merge → final LK pass (or EAX → LK).
- [ ] Benchmarks: anytime curves, TTT, ablations.

## Acknowledgements
- Based on your **Bee TSP Proposal** and the collaborative appendix we derived from it (see `docs/APPENDIX.md`).

---

© 2025-08-14 — MIT License (see `LICENSE`).
