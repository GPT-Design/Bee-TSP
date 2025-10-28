"""
Full Edge Assembly Crossover (EAX) - Robust Implementation
- Strict invariants (degree-2, permutation)
- Safe E-set reconstruction (no singletons fabricated)
- Proper multi-cycle patching via 2-edge bridges
- Offspring sanity checks + length smell test + sane fallback

Public entrypoint: eax_integrate(pool, dist, candidate_adj, config)
"""

from __future__ import annotations
import math
import random
import time
from typing import List, Dict, Set, Tuple, Iterable
from collections import defaultdict, deque


# =========================
#         GUARDRAILS
# =========================

class TSPGuard:
    """Cheap, strong checks to prevent returning garbage tours."""
    def __init__(self, dist, n: int, instance_tag: str = ""):
        self.dist = dist
        self.n = n
        self.instance_tag = instance_tag or getattr(dist, "name", "")
        self._baseline_len = None  # computed lazily

    def validate_permutation(self, tour: List[int], label: str = "tour") -> None:
        if not tour or len(tour) != self.n:
            raise AssertionError(f"[{label}] bad length: {len(tour) if tour else 0} != {self.n}")
        if len(set(tour)) != self.n:
            raise AssertionError(f"[{label}] duplicate/missing cities")
        lo, hi = min(tour), max(tour)
        if lo < 0 or hi >= self.n:
            raise AssertionError(f"[{label}] city id out of range: [{lo},{hi}] vs [0,{self.n-1}]")

    def length(self, tour: List[int]) -> float:
        n = len(tour)
        total = 0.0
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            d = self.dist.d(a, b)
            if not math.isfinite(d) or d < 0:
                raise AssertionError(f"[LEN] nonfinite/neg distance d({a},{b})={d}")
            total += d
        return total

    def smell_test(self, L: float) -> None:
        """Flag absurd blowups (wrong metric, broken splice, etc.)."""
        if self._baseline_len is None:
            # Identity tour as crude baseline (exercises dist)
            baseline = list(range(self.n))
            self._baseline_len = max(1.0, self.length(baseline))
        # Allow slack; >20× baseline is almost surely nonsense.
        if L > 20.0 * self._baseline_len:
            raise AssertionError(
                f"[SANITY] length {L:.3g} too large vs baseline {self._baseline_len:.3g}"
            )


# =========================
#         API
# =========================

def eax_integrate(
    pool: List[List[int]],
    dist,
    candidate_adj: Dict[int, List[int]],
    config: Dict,
) -> List[int]:
    """
    EAX integration with robust invariants and safe fallbacks.
    - pool: list of parent tours (permutations of 0..n-1)
    - dist: object exposing d(i,j) -> nonnegative distance
    - candidate_adj: node -> list of candidate neighbor nodes (for 2-opt restriction)
    - config: dict with keys:
        parents_per_round (int, default 4)
        offspring_per_round (int, default 2)
        repair_time_ms (int, default 200)
        max_ab_cycles (int, default 10)
    """
    print(f"[EAX_ENTRY] Starting with {len(pool)} tours")
    if not pool or len(pool) < 2:
        return pool[0][:] if pool else list(range(100))

    n = len(pool[0])
    guard = TSPGuard(dist, n, instance_tag=getattr(dist, "name", ""))

    # Config
    top_k = int(config.get("parents_per_round", 4))
    num_offspring = int(config.get("offspring_per_round", 2))
    repair_time_ms = int(config.get("repair_time_ms", 200))
    max_cycles_to_try = int(config.get("max_ab_cycles", 10))

    # Validate & sort parents
    sane_parents: List[Tuple[float, List[int]]] = []
    for idx, p in enumerate(pool):
        try:
            guard.validate_permutation(p, f"parent[{idx}]")
            L = guard.length(p)
            guard.smell_test(L)  # also exercises metric
            sane_parents.append((L, p))
        except AssertionError as e:
            # Skip broken parent
            print(f"[EAX] Dropping invalid parent[{idx}]: {e}")

    if len(sane_parents) < 2:
        # If everything is broken, fail loudly
        raise AssertionError("[EAX] Not enough sane parents to proceed")

    sane_parents.sort(key=lambda t: t[0])
    parents = [p for _, p in sane_parents[:max(2, top_k)]]

    # Generate offspring
    best_offspring = None
    best_len = float("inf")

    for _ in range(num_offspring):
        p1, p2 = random.sample(parents, 2)
        child = eax_full_recombine(
            p1, p2, dist, candidate_adj, repair_time_ms, max_cycles_to_try
        )

        try:
            guard.validate_permutation(child, "offspring")
            Lc = guard.length(child)
            guard.smell_test(Lc)
        except AssertionError as e:
            print(f"[EAX] Offspring rejected: {e}")
            continue

        if Lc < best_len:
            best_len, best_offspring = Lc, child

    if best_offspring is not None:
        return best_offspring

    # Fallback: return best sane parent if no acceptable child
    print("[EAX] No valid offspring; returning best sane parent")
    return parents[0][:]


# =========================
#     EAX CORE LOGIC
# =========================

def eax_full_recombine(
    parent1: List[int],
    parent2: List[int],
    dist,
    candidate_adj: Dict[int, List[int]],
    repair_time_ms: int,
    max_cycles: int
) -> List[int]:
    """
    Full EAX with AB-cycle detection and E-set manipulation.
    Uses capped, de-duplicated AB-cycle enumeration to avoid RAM blow-ups.
    Enforces full coverage (deg==2 for every node) before reconstruction.
    """
    n = len(parent1)

    # Step 1: Build union graph adjacency
    union_adj = build_union_graph(parent1, parent2)

    # Step 2: Find up to 'max_cycles' unique AB-cycles (streamed)
    print("[EAX] Enumerating AB-cycles…")
    ab_cycles = find_ab_cycles(union_adj, n, cap=max_cycles)
    if not ab_cycles:
        # No cycles found - parents too similar, return better parent
        len1 = tour_length(parent1, dist)
        len2 = tour_length(parent2, dist)
        print("[EAX] No AB-cycles; returning better parent.")
        return parent1[:] if len1 <= len2 else parent2[:]
    avg_len = sum(len(c) for c in ab_cycles) / max(1, len(ab_cycles))
    print(f"[EAX] AB-cycles: kept={len(ab_cycles)} (cap={max_cycles}), avg_len≈{avg_len:.1f}")

    # Step 3: Try different cycle combinations
    best_tour = None
    best_length = float('inf')

    # Try up to min(len(ab_cycles), max_cycles) random subsets
    num_attempts = min(len(ab_cycles), max_cycles)

    # Parent bound for mild pruning
    p1_len = tour_length(parent1, dist)
    p2_len = tour_length(parent2, dist)
    parent_ub = 2.0 * max(p1_len, p2_len) + 1000.0

    for _ in range(num_attempts):
        # Select random subset of cycles (1 to len(ab_cycles))
        k = random.randint(1, len(ab_cycles))
        selected_cycles = random.sample(ab_cycles, k)

        # Build E-set by applying selected cycles (XOR semantics)
        e_set = apply_cycles_to_eset(parent1, selected_cycles)

        # Degree-2 invariant on E-set: **every node must be degree 2**
        try:
            assert_degree_two(e_set, n, allow_zero=False)
        except AssertionError as e:
            print(f"[EAX] Skipping subset (E-set deg!=2): {str(e)[:120]}")
            continue

        # Convert E-set to cycles, then merge if needed
        cycles = ecycle_components(e_set, n)
        if not cycles:
            continue
        if len(cycles) == 1:
            offspring = cycles[0]
        else:
            offspring = merge_cycles_greedy(cycles, dist)

        # Hard permutation check BEFORE 2-opt (avoid IndexError)
        if not (len(offspring) == n and len(set(offspring)) == n and min(offspring) == 0 and max(offspring) == n-1):
            print(f"[EAX] Skipping subset (offspring not a full permutation): "
                  f"len={len(offspring)}, unique={len(set(offspring))}")
            continue

        # Repair with 2-opt
        offspring = repair_tour_2opt(offspring, dist, candidate_adj, repair_time_ms)

        # Evaluate
        length = tour_length(offspring, dist)
        if length < best_length:
            best_length = length
            best_tour = offspring

        # Optional pruning
        if best_length < parent_ub:
            parent_ub = best_length

    if best_tour is None:
        print(f"[EAX_WARN] All offspring invalid/none improved; returning better parent")
        return parent1[:] if p1_len <= p2_len else parent2[:]

    return best_tour

# =========================
#     AB-CYCLE FINDING
# =========================

def build_union_graph(parent1: List[int], parent2: List[int]) -> Dict[int, Dict[str, Set[int]]]:
    """
    Returns: {node: {'p1': {neighbors from P1}, 'p2': {neighbors from P2}}}
    """
    n = len(parent1)
    union = defaultdict(lambda: {'p1': set(), 'p2': set()})

    for i in range(n):
        u, v = parent1[i], parent1[(i + 1) % n]
        union[u]['p1'].add(v)
        union[v]['p1'].add(u)
    for i in range(n):
        u, v = parent2[i], parent2[(i + 1) % n]
        union[u]['p2'].add(v)
        union[v]['p2'].add(u)

    return union

def _walk_alternating_cycle(
    start: int,
    nxt: int,
    label: str,
    union_adj: Dict[int, Dict[str, Set[int]]],
) -> List[Tuple[int, int, str]]:
    """
    Walk an alternating-labeled cycle starting from (start -> nxt) with edge 'label'.
    Avoid immediate backtracking; prefer closing to 'start' when possible.
    """
    cycle: List[Tuple[int, int, str]] = []
    cur = start
    prev = None
    cur_label = label

    # Local guard to avoid very long drifts
    steps = 0
    max_steps = 2 * len(union_adj) + 10

    while steps <= max_steps:
        steps += 1
        cycle.append((cur, nxt, cur_label))
        cur, prev = nxt, cur
        next_label = 'p2' if cur_label == 'p1' else 'p1'

        # Prefer not to immediately backtrack
        candidates = [nb for nb in union_adj[cur][next_label] if nb != prev]
        if not candidates:
            # If forced, allow closing to start
            if start in union_adj[cur][next_label] and prev != start:
                candidates = [start]
            else:
                return []

        # If we can close to start and we have ≥3 edges already, do so
        if start in candidates and len(cycle) >= 3:
            cycle.append((cur, start, next_label))
            return cycle

        # Otherwise take the first candidate deterministically
        nxt = candidates[0]
        cur_label = next_label

    # Safety: bail on runaway walks
    return []

def find_ab_cycles(
    union_adj: Dict[int, Dict[str, Set[int]]],
    n: int,
    cap: int
) -> List[List[Tuple[int, int, str]]]:
    """
    Streamed, de-duplicated AB-cycle enumeration with a hard cap.
    Returns up to 'cap' unique cycles (each as [(u,v,label), ...]).
    """
    seen: Set[frozenset] = set()  # signatures for dedup
    out: List[List[Tuple[int, int, str]]] = []

    def signature(cycle: List[Tuple[int, int, str]]) -> frozenset:
        # undirected edge with label
        return frozenset(((min(u, v), max(u, v), lab) for (u, v, lab) in cycle))

    total_found = 0
    for start in range(n):
        for first_label in ('p1', 'p2'):
            # Iterate neighbors deterministically for stability
            for nxt in sorted(union_adj[start][first_label]):
                cyc = _walk_alternating_cycle(start, nxt, first_label, union_adj)
                if cyc and len(cyc) >= 4:
                    total_found += 1
                    sig = signature(cyc)
                    if sig not in seen:
                        seen.add(sig)
                        out.append(cyc)
                        if len(out) >= cap:
                            print(f"[EAX] AB-cycle cap reached: kept={len(out)}, seen_total≈{total_found}")
                            return out
    if total_found > len(out):
        print(f"[EAX] AB-cycles enumerated: kept={len(out)} / raw≈{total_found} (dedup applied)")
    return out

# =========================
#   E-SET → CYCLES/Tour
# =========================

def apply_cycles_to_eset(
    parent1: List[int],
    cycles: List[List[Tuple[int, int, str]]]
) -> Set[Tuple[int, int]]:
    """
    Apply selected AB-cycles to parent1 to create the E-set using XOR (toggle) semantics.

    Start from all P1 edges in parent1.
    For each edge (u,v,lab) in the selected cycles:
      - Toggle the undirected edge (min(u,v), max(u,v)) in the E-set,
        regardless of label. This implements the symmetric-difference effect
        across multiple cycles and prevents degree-1 artifacts when cycles overlap.
    """
    n = len(parent1)

    # Start with all P1 edges from parent1
    eset: Set[Tuple[int, int]] = set()
    for i in range(n):
        u, v = parent1[i], parent1[(i + 1) % n]
        if u > v:
            u, v = v, u
        eset.add((u, v))

    # Toggle helper
    def toggle(edge: Tuple[int, int]):
        if edge in eset:
            eset.remove(edge)
        else:
            eset.add(edge)

    # For each selected cycle, toggle each edge
    # (p1 edges: remove if present / re-add if encountered twice;
    #  p2 edges: add if absent / remove if encountered twice)
    for cycle in cycles:
        for u, v, _lab in cycle:
            a, b = (u, v) if u < v else (v, u)
            toggle((a, b))

    return eset

def assert_degree_two(eset: Set[Tuple[int, int]], n: int, allow_zero: bool) -> None:
    """Ensure each node has deg==2 (or deg in {0,2} if allow_zero)."""
    deg = [0] * n
    for u, v in eset:
        deg[u] += 1
        deg[v] += 1
    bad = []
    for i, d in enumerate(deg):
        if allow_zero:
            if d not in (0, 2):
                bad.append((i, d))
        else:
            if d != 2:
                bad.append((i, d))
    if bad:
        raise AssertionError(f"[E-SET] Degree violations (node,deg): {bad[:10]}... total={len(bad)}")


def ecycle_components(eset: Set[Tuple[int, int]], n: int) -> List[List[int]]:
    """
    Decompose E-set into disjoint cycles by adjacency walk.
    No node fabrication; only nodes with deg==2 are in cycles.
    """
    adj = defaultdict(list)
    for u, v in eset:
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * n
    cycles: List[List[int]] = []

    for start in range(n):
        if start not in adj or len(adj[start]) != 2 or visited[start]:
            continue
        # walk cycle
        cyc = []
        cur = start
        prev = None
        while True:
            cyc.append(cur)
            visited[cur] = True
            a, b = adj[cur]
            nxt = a if a != prev else b
            prev, cur = cur, nxt
            if cur == start:
                break
        # Close check
        if start not in adj[cyc[-1]]:
            raise AssertionError("[E-CYCLE] cycle does not close")
        cycles.append(cyc)

    # Optional sanity: ensure no leftover deg-2 nodes unvisited
    for node in adj:
        if len(adj[node]) == 2 and not visited[node]:
            raise AssertionError(f"[E-CYCLE] missed deg-2 node {node}")

    return cycles


def merge_cycles_greedy(cycles: List[List[int]], dist) -> List[int]:
    """
    Merge multiple cycles into a single tour using 2-edge bridges
    (reopen both cycles and reconnect crosswise with best gain).
    """
    if not cycles:
        return []
    if len(cycles) == 1:
        return cycles[0][:]

    # Copy and sort to encourage stable, short merges first
    work = [c[:] for c in cycles]
    work.sort(key=len)
    A = work.pop(0)

    def best_bridge(A: List[int], B: List[int]) -> Tuple[str, int, int, float]:
        best_mode = "straight"
        best_i = 0
        best_j = 0
        best_gain = math.inf
        m, n = len(A), len(B)
        for i in range(m):
            a1, a2 = A[i], A[(i + 1) % m]
            for j in range(n):
                b1, b2 = B[j], B[(j + 1) % n]
                d0 = dist.d(a1, a2) + dist.d(b1, b2)
                d1 = dist.d(a1, b1) + dist.d(a2, b2)
                d2 = dist.d(a1, b2) + dist.d(a2, b1)
                g1 = d1 - d0
                g2 = d2 - d0
                if g1 < best_gain:
                    best_gain, best_mode, best_i, best_j = g1, "straight", i, j
                if g2 < best_gain:
                    best_gain, best_mode, best_i, best_j = g2, "cross", i, j
        return best_mode, best_i, best_j, best_gain

    while work:
        B = work.pop(0)
        mode, i, j, _ = best_bridge(A, B)
        # splice cycles A and B according to mode
        if mode == "straight":
            A1 = A[: i + 1]
            A2 = A[i + 1 :]
            B1 = B[: j + 1]
            B2 = B[j + 1 :]
            A = A1 + list(reversed(B1)) + A2 + B2
        else:  # "cross"
            A1 = A[: i + 1]
            A2 = A[i + 1 :]
            B1 = B[j + 1 :]
            B2 = B[: j + 1]
            A = A1 + B1 + A2 + list(reversed(B2))

    return A

# =========================
#        LOCAL OPT
# =========================

def repair_tour_2opt(
    tour: List[int],
    dist,
    candidate_adj: Dict[int, List[int]],
    time_budget_ms: int,
) -> List[int]:
    """Candidate-restricted, time-bounded 2-opt improvement."""
    if not tour or len(tour) < 4:
        return tour[:]

    t = tour[:]
    n = len(t)
    pos = [0] * n
    for i, city in enumerate(t):
        pos[city] = i

    start_ms = time.time() * 1000
    improved = True

    while improved and (time.time() * 1000 - start_ms) < time_budget_ms:
        improved = False
        for i in range(n):
            if (time.time() * 1000 - start_ms) >= time_budget_ms:
                break

            a = t[i]
            b = t[(i + 1) % n]
            cand = candidate_adj.get(a, [])[:8]  # slightly larger beam helps
            for ck in cand:
                k = pos[ck]
                if k == i or k == (i + 1) % n or k == (i - 1) % n:
                    continue
                c = t[k]
                d = t[(k + 1) % n]

                old_cost = dist.d(a, b) + dist.d(c, d)
                new_cost = dist.d(a, c) + dist.d(b, d)
                delta = new_cost - old_cost

                if delta < -1e-9:
                    if i < k:
                        t[i + 1 : k + 1] = reversed(t[i + 1 : k + 1])
                    else:
                        t[k + 1 : i + 1] = reversed(t[k + 1 : i + 1])
                    # rebuild positions quickly (O(n) but bounded by time budget)
                    for idx, city in enumerate(t):
                        pos[city] = idx
                    improved = True
                    break
            if improved:
                break

    return t


# =========================
#       UTILITIES
# =========================

def tour_length(tour: List[int], dist) -> float:
    if not tour:
        return float("inf")
    s = 0.0
    n = len(tour)
    for i in range(n):
        s += dist.d(tour[i], tour[(i + 1) % n])
    return s
