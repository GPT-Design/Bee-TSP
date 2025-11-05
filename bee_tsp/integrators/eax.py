"""
Full Edge Assembly Crossover (EAX) - Safe, bounded Mark-1 implementation
- Strict AB-cycle validation (alternation, closure, parent-membership, continuity)
- Vertex-disjoint cycle selection (prevents parity breakage)
- Robust E-set toggling against P1 edge set only
- Degree==2 enforcement before converting E-set to a tour
- Safe E-set → tour (components merged) + guarded 2-opt repair
- Memory and time caps to avoid blow-ups on large instances (e.g., pr2392, pla33810)
"""

from __future__ import annotations
import random, time
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple

# =========================
# Public API
# =========================

def eax_integrate(
    pool: List[List[int]],
    dist,
    candidate_adj: Dict[int, List[int]],
    config: Dict
) -> List[int]:
    """
    Integrate a pool using EAX and return the best offspring (or best parent on failure).
    """
    print(f"[EAX_ENTRY] Starting with {len(pool)} tours")
    if not pool or len(pool) < 2:
        return pool[0][:] if pool else []

    n = len(pool[0])

    # Config (conservative Mark-1 defaults)
    top_k          = int(config.get("parents_per_round", 4))
    num_offspring  = int(config.get("offspring_per_round", config.get("num_offspring", 3)))
    repair_time_ms = int(config.get("repair_time_ms", 300))
    max_ab_cycles  = int(config.get("max_ab_cycles", 10))

    # sort parents by length
    ranked  = sorted(pool, key=lambda t: tour_length(t, dist))
    parents = ranked[:max(2, min(top_k, len(ranked)))]

    # run multiple offspring attempts
    best_child, best_len = None, float("inf")
    for _ in range(max(1, num_offspring)):
        p1, p2 = random.sample(parents, 2)
        child = eax_full_recombine(
            p1, p2, dist, candidate_adj,
            repair_time_ms=repair_time_ms,
            max_cycles=max_ab_cycles
        )
        if not validate_tour(child, n):
            continue
        L = dist.tour_length(child)
        if L < best_len:
            best_len, best_child = L, child

    # acceptance / fallback
    if best_child is None:
        # No valid child this round → choose the best available parents deterministically
        p_sorted = sorted(parents, key=lambda t: dist.tour_length(t))
        pA = p_sorted[0]
        pB = p_sorted[1] if len(p_sorted) > 1 else p_sorted[0]
        return _fallback_polish_best_parent(pA, pB, dist, candidate_adj, repair_time_ms)


    # otherwise return the best valid child
    return best_child

def _fallback_polish_best_parent(
    p1: List[int],
    p2: List[int],
    dist,
    candidate_adj: Dict[int, List[int]],
    repair_time_ms: int
) -> List[int]:
    """
    When cycles are scarce/invalid, do something productive:
    take the better parent and give it a quick polish.
    Prefer a single POPMUSIC pass if available; else 2-opt.
    """
    best_par = p1 if tour_length(p1, dist) <= tour_length(p2, dist) else p2
    best_len = tour_length(best_par, dist)

    # Try POPMUSIC if available
    try:
        from bee_tsp.integrators.popmusic import popmusic_integrate
        print("[EAX→POP] No usable EAX cycles; polishing best parent with POPMUSIC (1 round)")
        cfg = {
            "parents_per_round": 2,
            "merge_edge_cap_per_node": 6,
            "rounds": 1,
            "repair_time_ms": max(200, min(400, repair_time_ms)),
        }
        child = popmusic_integrate([best_par], dist, candidate_adj, cfg)
        if validate_tour(child, len(best_par)):
            child_len = tour_length(child, dist)
            if child_len < best_len:
                print(f"[EAX→POP] POPMUSIC improved: {int(best_len)} → {int(child_len)}")
                return child
            else:
                print("[EAX→POP] POPMUSIC made no improvement")
    except Exception as e:
        # POPMUSIC unavailable or failed; continue to 2-opt
        print(f"[EAX→POP] POPMUSIC unavailable/failed ({type(e).__name__}): {e}")

    # Minimal local polish (2-opt) fallback
    print("[EAX→2OPT] Using 2-opt fallback polish")
    child2 = repair_tour_2opt(best_par, dist, candidate_adj, repair_time_ms)
    if validate_tour(child2, len(best_par)):
        child2_len = tour_length(child2, dist)
        if child2_len < best_len:
            print(f"[EAX→2OPT] Improved: {int(best_len)} → {int(child2_len)}")
            return child2

    # No improvement
    return best_par

def eax_full_recombine(
    parent1: List[int],
    parent2: List[int],
    dist,
    candidate_adj: Dict[int, List[int]],
    repair_time_ms: int,
    max_cycles: int
) -> List[int]:
    """
    Safe EAX with budgeted enumeration, vertex-disjoint selection,
    and a POPMUSIC/2-opt fallback when cycles are scarce or invalid.
    """
    import time

    n = len(parent1)

    # --- per-pair guard so big-n can't stall here
    pair_t0 = time.time() * 1000.0
    pair_budget_ms = 2500  # local guard; keep modest

    # --- 1) Enumerate AB-cycles (bounded)
    ab_cycles = enumerate_ab_cycles(
        parent1, parent2, n,
        cap_report=10_000,
        log_cap=20,
        time_budget_ms=3000
    )
    if (not ab_cycles) or ((time.time() * 1000.0 - pair_t0) > pair_budget_ms):
        return _fallback_polish_best_parent(parent1, parent2, dist, candidate_adj, repair_time_ms)

    # --- 2) Sanitize + keep only vertex-disjoint cycles
    ab_cycles = _sanitize_ab_cycles(ab_cycles, parent1, parent2, n, max_len=2048)
    if (not ab_cycles) or ((time.time() * 1000.0 - pair_t0) > pair_budget_ms):
        return _fallback_polish_best_parent(parent1, parent2, dist, candidate_adj, repair_time_ms)

    selected_cycles = _filter_vertex_disjoint_cycles(ab_cycles, n, max_cycles)
    if (not selected_cycles) or ((time.time() * 1000.0 - pair_t0) > pair_budget_ms):
        print(f"[EAX] No usable cycles; POPMUSIC polish (selected=0)")
        return _fallback_polish_best_parent(parent1, parent2, dist, candidate_adj, repair_time_ms)

    # --- helper: greedy E-set builder starting from P1 (degree-safe)
    def _build_eset_greedy_from_p1(parent1_tour, cycles_labeled, n_nodes, cap):
        def norm(a, b): return (a, b) if a < b else (b, a)
        # seed with P1
        eset_g: set[tuple[int, int]] = set()
        deg = [0] * n_nodes
        for i in range(n_nodes):
            u, v = parent1_tour[i], parent1_tour[(i + 1) % n_nodes]
            e = norm(u, v)
            if e not in eset_g:
                eset_g.add(e)
                deg[u] += 1
                deg[v] += 1  # P1 is a tour → deg == 2 across vertices

        chosen = 0
        for cyc in selected_cycles:  # try in given order; upstream may have sorted
            if chosen >= cap:
                break
            ok = True
            add_edges: list[tuple[int, int]] = []
            remove_edges: list[tuple[int, int]] = []
            for (u, v, lab) in cyc:
                e = norm(u, v)
                if lab == 'p1':
                    if e not in eset_g:
                        ok = False; break
                    remove_edges.append(e)
                else:  # 'p2'
                    add_edges.append(e)
            if not ok:
                continue
            # degree feasibility
            for (u, v) in remove_edges:
                if deg[u] - 1 < 0 or deg[v] - 1 < 0:
                    ok = False; break
            if ok:
                for (u, v) in add_edges:
                    if deg[u] + 1 > 2 or deg[v] + 1 > 2:
                        ok = False; break
            if not ok:
                continue
            # apply
            for (u, v) in remove_edges:
                if (u, v) in eset_g:
                    eset_g.discard((u, v))
                    deg[u] -= 1; deg[v] -= 1
                else:
                    # Shouldn't happen if checks above passed
                    ok = False; break
            if not ok:
                continue
            for (u, v) in add_edges:
                if u > v: u, v = v, u
                if (u, v) not in eset_g:
                    eset_g.add((u, v))
                    deg[u] += 1; deg[v] += 1
            chosen += 1
        return eset_g

    # --- 3) Naive E-set (XOR) + strict degree-2; rescue if needed
    try:
        eset = apply_cycles_to_eset(parent1, parent2, selected_cycles)
        assert_degree_two(eset, n, allow_zero=False)
    except AssertionError:
        # Greedy rescue
        eset = _build_eset_greedy_from_p1(parent1, selected_cycles, n, max_cycles)
        try:
            assert_degree_two(eset, n, allow_zero=False)
            print("[EAX] Greedy E-set rescue: 2-regular achieved.")
        except AssertionError:
            # Single-cycle salvage
            print("[EAX] Greedy rescue empty; trying single-cycle fallback.")
            eset = set()
            for cyc in selected_cycles[:5]:
                single = apply_cycles_to_eset(parent1, parent2, [cyc])
                try:
                    assert_degree_two(single, n, allow_zero=False)
                    eset = single
                    print("[EAX] Single-cycle rescue: 2-regular achieved.")
                    break
                except AssertionError:
                    continue
            if not eset:
                print("[EAX] Unable to form 2-regular E-set; POPMUSIC fallback.")
                return _fallback_polish_best_parent(parent1, parent2, dist, candidate_adj, repair_time_ms)

    # --- 4) Build offspring from E-set + hard validity gate
    offspring = eset_to_tour(eset, n, dist)
    if not validate_tour(offspring, n):
        print("[EAX] Invalid tour produced from E-set (perm check failed) → POPMUSIC fallback.")
        return _fallback_polish_best_parent(parent1, parent2, dist, candidate_adj, repair_time_ms)

    # --- 5) Quick polish + re-validate (budgeted)
    offspring = repair_tour_2opt(offspring, dist, candidate_adj, repair_time_ms)
    if not validate_tour(offspring, n):
        return _fallback_polish_best_parent(parent1, parent2, dist, candidate_adj, repair_time_ms)

    # --- 6) Return better of offspring vs best parent
    best_par = _best_parent(parent1, parent2, dist)
    return offspring if dist.tour_length(offspring) <= dist.tour_length(best_par) else best_par

# =========================
# AB-cycle enumeration & validation
# =========================

def _build_parent_adj(parent: List[int]) -> Dict[int, Set[int]]:
    n = len(parent)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        u, v = parent[i], parent[(i + 1) % n]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_in(adj: Dict[int, Set[int]], u: int, v: int) -> bool:
    return v in adj.get(u, ()) or u in adj.get(v, ())

def _union_labelled(parent1: List[int], parent2: List[int]) -> Dict[int, Dict[str, Set[int]]]:
    """Union adjacency with labels 'p1' and 'p2'."""
    n = len(parent1)
    U = {i: {'p1': set(), 'p2': set()} for i in range(n)}
    for i in range(n):
        a, b = parent1[i], parent1[(i + 1) % n]
        U[a]['p1'].add(b); U[b]['p1'].add(a)
    for i in range(n):
        a, b = parent2[i], parent2[(i + 1) % n]
        U[a]['p2'].add(b); U[b]['p2'].add(a)
    return U

def _canon_cycle_sig(cyc: List[Tuple[int,int,str]]) -> Tuple:
    """
    Canonical signature for cycle dedup (rotation + reversal invariance).
    Represent as sequence of triples, pick lexicographically smallest rotation/orientation.
    """
    if not cyc: return tuple()
    seq = cyc[:]
    # normalize undirected edge orientation (ordered endpoints)
    seq = [(min(u,v), max(u,v), lab) for (u,v,lab) in seq]
    # all rotations both directions
    cand = []
    m = len(seq)
    for s in range(m):
        cand.append(tuple(seq[s:]+seq[:s]))
    rev = list(reversed(seq))
    for s in range(m):
        cand.append(tuple(rev[s:]+rev[:s]))
    return min(cand)

def enumerate_ab_cycles(
    parent1: List[int],
    parent2: List[int],
    n: int,
    cap_report: int = 10000,
    log_cap: int = 20,
    time_budget_ms: int = 3000  # hard cap to avoid stalls
) -> List[List[Tuple[int,int,str]]]:
    """
    Enumerate SIMPLE AB-cycles (no vertex revisits except closing to start).
    Budgeted by raw-cycle cap and wall-clock time to prevent stalls.
    """
    U = _union_labelled(parent1, parent2)
    raw_cycles: List[List[Tuple[int,int,str]]] = []
    seen_sigs: Set[Tuple] = set()

    per_node_start_cap = 12
    max_depth = 4096
    t0 = time.time() * 1000.0

    print("[EAX] Enumerating AB-cycles…")
    for s in range(n):
        if (time.time()*1000.0 - t0) >= time_budget_ms:
            break
        for first_parent in ('p1', 'p2'):
            nbrs = list(U[s][first_parent])
            random.shuffle(nbrs)
            nbrs = nbrs[:per_node_start_cap]
            for nb in nbrs:
                cyc: List[Tuple[int,int,str]] = []
                cur = s
                next_parent = first_parent
                depth = 0
                used_vertices: Set[int] = {s}
                while depth < max_depth:
                    if (time.time()*1000.0 - t0) >= time_budget_ms:
                        break
                    depth += 1
                    picked = None
                    for v in U[cur][next_parent]:
                        if v == s and len(cyc) >= 3:
                            picked = v
                            break
                        if v not in used_vertices:
                            picked = v
                            break
                    if picked is None:
                        break

                    cyc.append((cur, picked, next_parent))
                    if picked == s and len(cyc) >= 4:
                        sig = _canon_cycle_sig(cyc)
                        if sig not in seen_sigs:
                            seen_sigs.add(sig)
                            raw_cycles.append(cyc[:])
                            if len(raw_cycles) >= cap_report:
                                kept = min(len(raw_cycles), log_cap)
                                avg_len = sum(len(c) for c in raw_cycles[:kept]) / kept
                                print(f"[EAX] AB-cycles enumerated: kept={kept} / raw≈{len(raw_cycles)} (dedup applied)")
                                print(f"[EAX] AB-cycles: kept={kept} (cap={log_cap}), avg_len≈{avg_len:.1f}")
                                return raw_cycles[:log_cap]
                        break

                    used_vertices.add(picked)
                    cur = picked
                    next_parent = 'p2' if next_parent == 'p1' else 'p1'

    kept = min(len(raw_cycles), log_cap)
    if len(raw_cycles) >= log_cap:
        avg_len = sum(len(c) for c in raw_cycles[:kept]) / kept if kept else 0.0
        print(f"[EAX] AB-cycles enumerated: kept={kept} / raw≈{len(raw_cycles)} (dedup applied)")
        print(f"[EAX] AB-cycles: kept={kept} (cap={log_cap}), avg_len≈{avg_len:.1f}")
    else:
        print(f"[EAX] AB-cycles: kept={len(raw_cycles)} (cap={log_cap})")
    return raw_cycles[:log_cap]

def _validate_ab_cycle(
    cyc: List[Tuple[int,int,str]],
    p1_adj: Dict[int, Set[int]],
    p2_adj: Dict[int, Set[int]],
) -> bool:
    """
    Valid AB-cycle:
      - len >= 4 and even
      - labels strictly alternate globally
      - each edge exists in its stated parent
      - continuity: consecutive edges share exactly one endpoint
      - closure: last edge touches first start vertex
      - SIMPLE: each vertex used by the cycle has degree 2 in the cycle
      - LABEL BALANCE per vertex: exactly one 'p1' and one 'p2' incidence
    """
    m = len(cyc)
    if m < 4 or (m % 2) != 0:
        return False

    # global alternation
    for i in range(1, m):
        if cyc[i][2] == cyc[i-1][2]:
            return False

    deg_local: Dict[int, int] = defaultdict(int)
    lbl_count: Dict[int, Counter] = defaultdict(Counter)

    for i in range(m):
        u, v, lab = cyc[i]
        if lab == 'p1':
            if not _edge_in(p1_adj, u, v): return False
        else:
            if not _edge_in(p2_adj, u, v): return False

        # continuity
        u2, v2, _ = cyc[(i + 1) % m]
        shared = ((u == u2) + (u == v2) + (v == u2) + (v == v2))
        if shared == 0:
            return False

        # local undirected degree + per-vertex label counts
        deg_local[u] += 1; deg_local[v] += 1
        lbl_count[u][lab] += 1; lbl_count[v][lab] += 1

    # closure
    s0 = cyc[0][0]
    ul, vl, _ = cyc[-1]
    if not (s0 == ul or s0 == vl):
        return False

    # simple: deg==2; label balance: one 'p1' and one 'p2' at each used vertex
    for v, d in deg_local.items():
        if d != 2:
            return False
        if not (lbl_count[v]['p1'] == 1 and lbl_count[v]['p2'] == 1):
            return False

    return True

def _sanitize_ab_cycles(
    ab_cycles: List[List[Tuple[int,int,str]]],
    parent1: List[int],
    parent2: List[int],
    n: int,
    max_len: int | None = None,
) -> List[List[Tuple[int,int,str]]]:
    """Trim very long cycles (keep alternation), drop invalid ones."""
    p1_adj = _build_parent_adj(parent1)
    p2_adj = _build_parent_adj(parent2)

    cleaned = []
    for cyc in ab_cycles:
        c = cyc
        if max_len is not None and len(c) > max_len:
            want = max_len if (max_len % 2) == 0 else (max_len - 1)
            if want >= 4:
                c = c[:want]
        if _validate_ab_cycle(c, p1_adj, p2_adj):
            cleaned.append(c)
    return cleaned

def _filter_vertex_disjoint_cycles(
    cycles: List[List[Tuple[int,int,str]]],
    n: int,
    max_pick: int
) -> List[List[Tuple[int,int,str]]]:
    """
    Greedy pick of vertex-disjoint cycles to avoid parity clashes at vertices.
    """
    picked = []
    used: Set[int] = set()
    # sort longer first (usually more impactful)
    cycles_sorted = sorted(cycles, key=lambda c: -len(c))
    for cyc in cycles_sorted:
        verts = set()
        ok = True
        for (u, v, _) in cyc:
            if u in used or v in used:
                ok = False; break
            verts.add(u); verts.add(v)
        if ok:
            picked.append(cyc)
            used |= verts
            if len(picked) >= max_pick:
                break
    return picked

# =========================
# E-set construction and checks
# =========================

def apply_cycles_to_eset(
    parent1: List[int],
    parent2: List[int],
    cycles: List[List[Tuple[int,int,str]]]
) -> Set[Tuple[int,int]]:
    """
    Start from P1 edges; for each cycle edge:
      - if labeled 'p1': remove that edge from the set
      - if labeled 'p2': add that edge to the set
    Using normalized undirected edges (min, max).
    """
    n = len(parent1)
    def norm(a, b): return (a, b) if a < b else (b, a)

    eset: Set[Tuple[int,int]] = set()
    # seed with P1
    for i in range(n):
        u, v = parent1[i], parent1[(i+1) % n]
        eset.add(norm(u, v))

    # toggle with cycles
    for cyc in cycles:
        for (u, v, lab) in cyc:
            e = norm(u, v)
            if lab == 'p1':
                eset.discard(e)
            else:  # 'p2'
                eset.add(e)
    return eset

def assert_degree_two(eset: Set[Tuple[int,int]], n: int, allow_zero: bool=False) -> None:
    """
    Verify each vertex has degree exactly 2 (or exactly 0 if allow_zero).
    """
    deg = [0]*n
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

def _can_add_cycle_safely(eset: set[tuple[int,int]], cyc_edges: list[tuple[int,int]], n: int) -> bool:
    """Check if adding this cycle’s edges keeps all vertex degrees ≤ 2."""
    deg = [0]*n
    for u,v in eset:
        deg[u]+=1; deg[v]+=1
    for u,v in cyc_edges:
        deg[u]+=1; deg[v]+=1
        if deg[u] > 2 or deg[v] > 2:
            return False
    return True

def _build_eset_greedy(cycles_edges: list[list[tuple[int,int]]], n: int, cap: int) -> set[tuple[int,int]]:
    """
    Greedy degree-safe E-set: try cycles in order; add only if all degrees stay ≤ 2.
    cycles_edges: each item is a list of undirected edges (u,v) with u!=v, u<v normalization preferred
    """
    eset: set[tuple[int,int]] = set()
    chosen = 0
    for cyc in cycles_edges:
        if chosen >= cap:
            break
        if _can_add_cycle_safely(eset, cyc, n):
            for e in cyc:
                u,v = e
                if u > v:  # normalize
                    u,v = v,u
                eset.add((u,v))
            chosen += 1
    return eset

def validate_tour(tour: List[int], n: int) -> bool:
    return bool(tour) and len(tour)==n and min(tour)==0 and max(tour)==n-1 and len(set(tour))==n

# =========================
# E-set → tour
# =========================

def eset_to_tour(eset: Set[Tuple[int,int]], n: int, dist) -> List[int]:
    """
    Convert E-set to a Hamiltonian tour.
    Precondition: degree==2 everywhere.
    Strategy: build adjacency; walk one cycle; if multiple cycles (shouldn’t happen with deg==2),
    connect components greedily.
    """
    if not eset:
        return list(range(n))  # fallback

    # adjacency
    adj = [[] for _ in range(n)]
    for a,b in eset:
        adj[a].append(b)
        adj[b].append(a)

    # detect components (should be 1 with deg==2, but guard anyway)
    seen = [False]*n
    comps = []
    for s in range(n):
        if seen[s]: continue
        if not adj[s]: continue
        stack=[s]; seen[s]=True; comp=[s]
        while stack:
            u=stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v]=True; stack.append(v); comp.append(v)
        comps.append(comp)

    if not comps:
        return list(range(n))

    if len(comps)==1:
        # walk the simple cycle
        start = comps[0][0]
        tour=[start]
        prev = None
        cur = start
        while True:
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0]!=prev else (nbrs[1] if len(nbrs)>1 else None)
            if nxt is None:
                break
            if nxt == start and len(tour)==n:
                return tour
            tour.append(nxt)
            prev, cur = cur, nxt
            if len(tour) > n+5:  # paranoia
                break
        # fallback join if something odd
        return _join_components_greedy([tour], dist, n)

    # multiple components (shouldn’t with deg==2), connect greedily
    return _join_components_greedy(comps, dist, n)

def _join_components_greedy(comps: List[List[int]], dist, n: int) -> List[int]:
    """Greedily connect component sequences by nearest endpoints."""
    if not comps:
        return list(range(n))
    tour = comps[0][:]
    for comp in comps[1:]:
        # pick best insertion between ends (cheap heuristic)
        best = None
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i+1)%len(tour)]
            for u in comp:
                # try splice: a-u + u-b
                gain = (dist.d(a,u) + dist.d(u,b)) - dist.d(a,b)
                if (best is None) or (gain < best[0]):
                    best = (gain, i, u)
        _, i, u = best
        tour = tour[:i+1] + [u] + tour[i+1:]
        # append remaining nodes of comp naively next to u
        rest = [x for x in comp if x != u]
        tour = tour[:i+2] + rest + tour[i+2:]
    # if duplicates happened (paranoia), dedupe sequentially then fill gaps
    tour = _repair_perm_paranoid(tour, n)
    return tour

def _repair_perm_paranoid(t: List[int], n: int) -> List[int]:
    seen=set(); out=[]
    for x in t:
        if 0<=x<n and x not in seen:
            out.append(x); seen.add(x)
    # fill missing
    for i in range(n):
        if i not in seen: out.append(i)
    return out[:n]

# =========================
# 2-opt repair (guarded)
# =========================

def repair_tour_2opt(tour: List[int], dist, candidate_adj: Dict[int, List[int]], time_budget_ms: int) -> List[int]:
    if not tour or len(tour) < 4:
        return tour
    t0 = time.time() * 1000.0
    n  = len(tour)
    tour = tour[:]

    pos = [0]*n
    for i, c in enumerate(tour):
        if not (0 <= c < n):
            return tour
        pos[c] = i

    improved=True
    while improved and (time.time()*1000.0 - t0) < time_budget_ms:
        improved=False
        for i in range(n):
            if (time.time()*1000.0 - t0) >= time_budget_ms: break
            a = tour[i]; b = tour[(i+1)%n]
            cand = candidate_adj.get(a, ())
            # small cap per iteration
            for city_k in cand[:8]:
                k = pos[city_k]
                if k==i or k==(i+1)%n or k==(i-1)%n: continue
                c = tour[k]; d = tour[(k+1)%n]
                delta = (dist.d(a,c) + dist.d(b,d)) - (dist.d(a,b) + dist.d(c,d))
                if delta < -1e-6:
                    if i < k:
                        tour[i+1:k+1] = reversed(tour[i+1:k+1])
                    else:
                        tour[k+1:i+1] = reversed(tour[k+1:i+1])
                    # rebuild pos only for affected segment (simple but safe: rebuild full)
                    for idx, city in enumerate(tour):
                        pos[city] = idx
                    improved=True
                    break
            if improved: break
    return tour

# =========================
# Utilities
# =========================

def tour_length(tour: List[int], dist) -> float:
    if not tour: return float("inf")
    n = len(tour)
    s = 0.0
    for i in range(n):
        s += dist.d(tour[i], tour[(i+1)%n])
    return s

def _best_parent(p1, p2, dist):
    return p1[:] if tour_length(p1, dist) <= tour_length(p2, dist) else p2[:]
