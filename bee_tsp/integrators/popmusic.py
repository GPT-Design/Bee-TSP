# === begin: popmusic_integrate (REPLACE THIS WHOLE FUNCTION) ===
from typing import List, Dict, Tuple
from collections import defaultdict
# --- safe imports ---
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("[POPMUSIC] numpy is required. Install with: pip install numpy") from e

def popmusic_integrate(
    pool: List[List[int]],
    dist,
    candidate_adj: Dict[int, List[int]],
    config: Dict,
    delta_phi: Dict[Tuple[int, int], float] = None
) -> List[int]:
    """
    POPMUSIC-style integration:
    - Treat pool[0] as elitist seed when config['elitist_seed'] is True (default),
      so you can put the Savings tour first and 'only improve it'.
    - Merge edges with per-node cap (config key: merge_edge_cap_per_node).
    - Use a small local 2-opt repair between rounds (config key: repair_time_ms).
    - All parameters read from `config` with safe defaults.
    """
    if not pool:
        raise ValueError("popmusic_integrate: empty pool")

    # --- config (read from params.yaml) ---
    parents_per_round = int(config.get('parents_per_round', 6))
    merge_cap = int(config.get('merge_edge_cap_per_node', 8))
    rounds = int(config.get('rounds', 3))
    repair_ms = int(config.get('repair_time_ms', 300))
    elitist = bool(config.get('elitist_seed', True))  # default True

    def tour_len(T: List[int]) -> float:
        n = len(T)
        return sum(dist.d(T[i], T[(i+1) % n]) for i in range(n))

    # If given only one parent, just repair it and return
    if len(pool) == 1:
        return _two_opt_repair(pool[0], dist, time_ms=repair_ms)

    # Choose base (Savings) tour: either pool[0] (elitist) or best among pool
    base = pool[0][:] if elitist else min(pool, key=tour_len)[:]
    best = base[:]
    best_len = tour_len(best)

    # POPMUSIC rounds
    for _ in range(rounds):
        # rank parents (excluding base) by length
        others = pool[1:] if elitist else [p for p in pool if p is not base]
        ranked = sorted(others, key=tour_len)
        take = max(1, min(parents_per_round - 1, len(ranked)))
        parents = [base] + ranked[:take]

        # build edge union with bias for base edges
        union = defaultdict(list)  # u -> list[(weight, v, dist)]
        def add_edge(u: int, v: int, w: int):
            d = dist.d(u, v)
            union[u].append((w, v, d))
            union[v].append((w, u, d))

        for pi, par in enumerate(parents):
            w = 2 if (elitist and pi == 0) else 1  # double-weight base edges
            n = len(par)
            for i in range(n):
                u, v = par[i], par[(i + 1) % n]
                add_edge(u, v, w)

        # cap edges per node by (weight desc, distance asc)
        sparse_adj: Dict[int, List[int]] = {}
        for u, nbrs in union.items():
            nbrs.sort(key=lambda x: (-x[0], x[2]))
            sparse_adj[u] = [v for _, v, _ in nbrs[:merge_cap]]

        # reconstruct a tour from sparse graph, then repair
        cand = _construct_from_sparse(sparse_adj, len(base), dist)
        cand = _two_opt_repair(cand, dist, time_ms=repair_ms)
        L = tour_len(cand)
        if L < best_len:
            best, best_len = cand, L
            base = best[:]   # keep improving the base
    return best

# --- helpers kept local to avoid cross-file dependencies ---
def _construct_from_sparse(sparse_adj: Dict[int, List[int]], n: int, dist) -> List[int]:
    """
    Greedy stitching from sparse adjacency; fills gaps if disconnected.
    """
    if not sparse_adj:
        return list(range(n))  # degenerate fallback

    used = [False] * n
    # start from node with max degree or 0 if absent
    start = max(sparse_adj.keys(), key=lambda u: len(sparse_adj.get(u, []))) if sparse_adj else 0
    tour = [start]
    used[start] = True

    while len(tour) < n:
        u = tour[-1]
        cands = [v for v in sparse_adj.get(u, []) if not used[v]]
        if cands:
            v = min(cands, key=lambda w: dist.d(u, w))
        else:
            # jump to nearest unused overall
            v = min((j for j in range(n) if not used[j]), key=lambda j: dist.d(u, j))
        used[v] = True
        tour.append(v)
    return tour

def _two_opt_repair(tour: List[int], dist, time_ms: int = 300) -> List[int]:
    """
    Simple bounded 2-opt with a rough time/iteration cap.
    (If you already have a high-performance 2-opt elsewhere, feel free to call it instead.)
    """
    import time as _t
    n = len(tour)
    if n < 4:
        return tour[:]
    pos = [0] * n
    for i, c in enumerate(tour):
        pos[c] = i

    def delta(i, k):
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[k], tour[(k + 1) % n]
        return (dist.d(a, c) + dist.d(b, d)) - (dist.d(a, b) + dist.d(c, d))

    start_t = _t.time()
    improved = True
    while improved and ( (_t.time() - start_t) * 1000.0 < time_ms ):
        improved = False
        for i in range(n - 1):
            for k in range(i + 2, n - (0 if i > 0 else 1)):
                if delta(i, k) < 0.0:
                    # reverse segment (i+1 ... k)
                    if i + 1 <= k:
                        tour[i + 1:k + 1] = reversed(tour[i + 1:k + 1])
                    else:
                        seg = list(reversed(tour[k + 1:i + 1]))
                        tour[k + 1:i + 1] = seg
                    # rebuild pos (cheap and safe here)
                    for idx, c in enumerate(tour):
                        pos[c] = idx
                    improved = True
                    break
            if improved:
                break

    return tour[:]
# === end: popmusic_integrate ===
