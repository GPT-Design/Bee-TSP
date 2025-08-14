# Appendix: Architecture & Benchmark Plan for the Bee‑Swarm TSP

**Scope.** This appendix turns the Bee‑Swarm design—zone‑based agents with budgets, Lévy‑style hops, and a central integrator—
into a reproducible algorithm with plug‑in components, parameters, and a benchmark protocol.

## A. Algorithm blueprint (hybridized)

**Core idea.** Many “bees” discover high‑quality local structure on a **restricted candidate graph**, then an **integrator** merges tours; 
a final **LKH‑grade** local search polishes the best tour.

### A1. Candidate graph (mandatory)
- Per‑node candidate set \(C(i)\) of size \(k\) (typ. 20–40).
- Euclidean: Delaunay neighbors ∪ k‑nearest neighbors.
- Next step: add **α‑near** edges from a 1‑tree (Held–Karp) bound.

### A2. Bee agents (explorers)
Operate on a **zone** (with overlap) and a budget.
- **Moves:** 2‑opt/3‑opt improving moves restricted to \(C(\cdot)\).
- **Kicks:** periodic double‑bridge (ruin‑and‑recreate) to escape basins.
- **Medium hops:** Lévy‑like restarts on stagnation.
- **Outputs:** complete (or partial) tour and its edge set.

### A3. Evidence model (colony memory)
- **Edge Histogram Matrix (EHM):** edge frequencies in top‑quality tours.
- **Scout sampling:** sample new starts from EHM (with smoothing); immediately local‑search.

### A4. Integrator (stitching)
Two options:
- **POPMUSIC‑style merging:** union edges from many tours → sparse graph → final **LKH** pass.
- **EAX crossover:** combine pairs of good tours; **LKH**‑polish the offspring.

### A5. Optional “learning waggle”
- Light model (e.g., gradient‑boosted trees) predicts edge usefulness; used as a prior within \(C(\cdot)\).

## B. Formal scoring & energy

### B1. Dual‑gap energy (when HK is available)
\[
\mathcal{E}(\text{tour}) = \text{length}(\text{tour}) - w_{\text{HK}}(\pi),
\]
where \(w_{\text{HK}}(\pi)\) is the 1‑tree lower bound with node penalties \(\pi\). Prefer moves that reduce \(\mathcal{E}\).

### B2. Population entropy (for exploration)
Let \(f_e\) be edge frequency. Define
\[
H = -\sum_e \big[f_e\log f_e + (1-f_e)\log(1-f_e)\big].
\]
Score edges by local gain minus \(\lambda\) times an uncertainty bonus from this entropy.

## C. Zones & budgets

### C1. Zoning
- Partition by k‑means (coords) **or** graph partitioning on the k‑NN distance graph.
- **Overlap halo:** 10–20% of nodes around each zone.
- **Targets:** zone size 50–300.

### C2. Budgets
- Step budget: max improving moves or 2–5×|zone|.
- Time budget: per‑agent soft cap.
- Restart: Lévy‑style jump on stagnation.

## D. Pseudocode (high‑level)

```text
Input: Coordinates/distances, n, params
Build candidate sets C(i) for all i
Zones ← PartitionNodes(n, overlap=τ)

EHM ← empty
BestPool ← ∅

repeat until wall_time or target_gap:
    parallel for zone in Zones:
        for agent in 1..A_zone:
            T ← ConstructInitialTour(zone, C)      # greedy + randomization
            T ← LocalSearch_LK(T, C)               # variable-k-opt + kicks
            BestPool ← UpdatePool(T)
            EHM ← UpdateEdgeHistogram(T)
            if stagnation:
                T ← ScoutFromEHM(EHM, C)
                T ← LocalSearch_LK(T, C)

    # Integrator
    G_sparse ← BuildSparseGraph(BestPool, EHM)
    if mode = POPMUSIC: T_global ← MergeTours(G_sparse)
    else:               T_global ← EAX_Combine(BestPool)
    T_global ← LocalSearch_LK(T_global, C)
    Best ← UpdateBest(T_global)

return Best
```

## E. Parameters (defaults)
- Candidate size \(k\): 30 (small \(n\): 20; large \(n\): 40).
- Zone overlap \(\tau\): 15%.
- Agents/zone: 2–8.
- Kick period: every 200–400 improving moves (or ~0.2 s).
- Scout rate: 10–20% of agents per outer loop.
- EHM smoothing \(\epsilon\): 0.02–0.05.
- Integrator: POPMUSIC edge cap/node 8–16; EAX pairs/round 5–20.
- Learning prior warm‑up: ~100 tours.

## F. Benchmark protocol (publication‑ready)

### F1. Instance suite
- TSPLIB Euclidean: eil51, ch130, kroA200, lin318, pcb442, rat783, pr1002, u1060, d1291, d2103, fl1577, rl11849, usa13509.
- Stress synthetics: rings, clustered equidistant points, narrow corridors.

### F2. Baselines
- LKH‑3 (default + tuned), GA‑EAX; optionally ACO and learning‑guided variant.

### F3. Metrics
- Relative gap to best‑known (or Held–Karp bound).
- Anytime curves (median over ≥30 runs).
- Time‑to‑Target at 1%, 0.5%, 0.1%.
- Robustness: median ± [P10, P90].

### F4. Ablations
- Candidate graph variants; LK depth with/without kicks.
- Integrator: POPMUSIC vs EAX.
- EHM/learning on/off; zone overlap ∈ {0%, 10%, 20%}.

### F5. Reproducibility
- Seeds, hardware/threads, compilers, instance checksums, param file, raw logs.
