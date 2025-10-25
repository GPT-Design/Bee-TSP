from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set
import math, random, time
from collections import defaultdict

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.spatial import cKDTree as KDTree
    _HAS_KDTREE = True
except Exception:
    _HAS_KDTREE = False

from .tsplib import load_tsplib

INF = 10**18

def tour_length(tour: List[int], dist) -> int:
    s = 0; n = len(tour)
    for i in range(n):
        s += dist(tour[i], tour[(i+1)%n])
    return s

def two_opt_delta(tour: List[int], i: int, k: int, dist) -> int:
    n = len(tour)
    a, b = tour[i % n], tour[(i+1) % n]
    c, d = tour[k % n], tour[(k+1) % n]
    return (dist(a,c) + dist(b,d)) - (dist(a,b) + dist(c,d))

def apply_two_opt_and_update_pos(tour: List[int], pos: List[int], i: int, k: int):
    if k < i: i, k = k, i
    tour[i+1:k+1] = reversed(tour[i+1:k+1])
    for idx in range(i+1, k+1):
        pos[tour[idx]] = idx

def double_bridge_kick(tour: List[int], pos: Optional[List[int]] = None):
    n = len(tour)
    if n < 8:
        random.shuffle(tour)
        if pos is not None:
            for i,v in enumerate(tour): pos[v] = i
        return
    a = random.randint(1, n//4)
    b = random.randint(a+1, n//2)
    c = random.randint(b+1, 3*n//4)
    d = random.randint(c+1, n-1)
    new = tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]
    tour[:] = new
    if pos is not None:
        for i,v in enumerate(tour): pos[v] = i

# ---------------- Candidate graph ----------------

def build_knn_candidates(coords: Optional[List[Tuple[float,float]]], n: int, dist, k: int) -> Dict[int, List[int]]:
    adj = {i: [] for i in range(n)}
    if coords is not None and _HAS_KDTREE and np is not None:
        pts = np.array(coords, dtype=float)
        tree = KDTree(pts)
        # Note: newer SciPy supports workers=-1 for multithreading; we keep default for compatibility
        dists, idxs = tree.query(pts, k=min(k+1, n))  # include self
        for i in range(n):
            nbrs = [int(j) for j in (idxs[i].tolist() if hasattr(idxs[i], "tolist") else list(idxs[i])) if j != i]
            adj[i] = nbrs[:k]
    elif coords is not None and np is not None:
        pts = np.array(coords, dtype=float)
        for i in range(n):
            di = ((pts - pts[i])**2).sum(axis=1) ** 0.5
            idx = np.argpartition(di, k+1)[:k+2]
            nbrs = [int(j) for j in idx if j != i]
            nbrs = sorted(nbrs, key=lambda j: di[j])[:k]
            adj[i] = nbrs
    else:
        for i in range(n):
            cand = []
            for j in range(n):
                if i == j: continue
                cand.append((dist(i,j), j))
            cand.sort()
            adj[i] = [j for _, j in cand[:k]]
    # symmetrize
    for i in range(n):
        for j in adj[i]:
            if i not in adj[j]:
                adj[j].append(i)
    return adj

# Pre-sort candidate neighbors by distance once
def sorted_candidates_by_dist(n: int, candidate_adj: Dict[int, List[int]], dist):
    sorted_adj = {}
    for u in range(n):
        nbrs = candidate_adj.get(u, [])
        nbrs = sorted(nbrs, key=lambda v: dist(u, v))
        sorted_adj[u] = nbrs
    return sorted_adj

# Candidate-only NN init: O(n * k) instead of O(n^2)
def nn_init_on_candidates(n: int, dist, cand_sorted: Dict[int, List[int]], start: Optional[int] = None) -> List[int]:
    used = [False]*n
    if start is None:
        start = 0
    used[start] = True
    tour = [start]
    # Maintain a cheap list of remaining nodes to jump when stuck
    remaining = [i for i in range(n) if i != start]
    rem_idx = 0

    while len(tour) < n:
        u = tour[-1]
        chosen = -1
        for v in cand_sorted.get(u, []):
            if not used[v]:
                chosen = v
                break
        if chosen == -1:
            # jump to next unused node (keeps O(n) total)
            while rem_idx < len(remaining) and used[remaining[rem_idx]]:
                rem_idx += 1
            if rem_idx >= len(remaining):
                break
            chosen = remaining[rem_idx]
        used[chosen] = True
        tour.append(chosen)
    return tour

# ---------------- Local search ----------------

def local_search_2opt(tour: List[int], dist, candidate_adj: Dict[int, List[int]], kick_period: int = 300, time_budget_s: float = 2.0):
    n = len(tour)
    best_len = tour_length(tour, dist)
    moves_since = 0
    start = time.time()
    pos = [0]*n
    for i,v in enumerate(tour): pos[v] = i

    improved = True
    while improved and (time.time() - start) < time_budget_s:
        improved = False
        for i in range(n):
            b = tour[(i+1)%n]
            for c in candidate_adj.get(b, []):
                k = pos[c]
                if k == i or (k+1)%n == i:
                    continue
                dlt = two_opt_delta(tour, i, k, dist)
                if dlt < 0:
                    apply_two_opt_and_update_pos(tour, pos, i, k)
                    best_len += dlt
                    improved = True
                    moves_since = 0
                    break
            if improved:
                break
        moves_since += 1
        if moves_since >= kick_period and (time.time() - start) < time_budget_s:
            double_bridge_kick(tour, pos)
            best_len = tour_length(tour, dist)
            moves_since = 0
    return tour

# ---------------- Simple EHM ----------------

class EdgeHistogram:
    def __init__(self, n: int, smoothing_eps: float = 0.03, max_edges_per_node: int = 16):
        self.n = n; self.eps = smoothing_eps; self.cap = max_edges_per_node
        self.counts = defaultdict(int)
    def _key(self,u,v): return (u,v) if u<v else (v,u)
    def update(self, tour: List[int]):
        n = len(tour)
        for i in range(n):
            u = tour[i]; v = tour[(i+1)%n]
            self.counts[self._key(u,v)] += 1
    def sample_tour(self, dist, candidate_adj: Dict[int, List[int]]):
        n = self.n
        start = random.randrange(n)
        used = [False]*n; used[start] = True
        t = [start]
        for _ in range(n-1):
            u = t[-1]; best_score=-1; best_v=-1
            for v in candidate_adj.get(u, []):
                if used[v]: continue
                key = self._key(u,v); cnt = self.counts.get(key,0)
                score = (cnt + self.eps) / (dist(u,v) + 1.0)
                if score > best_score: best_score, best_v = score, v
            if best_v == -1:
                for v in range(n):
                    if not used[v]: best_v = v; break
            used[best_v] = True; t.append(best_v)
        return t

# ---------------- Integrator ----------------

def greedy_completion_from_sparse(n: int, dist, sparse_adj: Dict[int, List[int]], fallback_adj: Dict[int, List[int]]) -> List[int]:
    parent = list(range(n)); rank=[0]*n
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb: return False
        if rank[ra]<rank[rb]: parent[ra]=rb
        elif rank[rb]<rank[ra]: parent[rb]=ra
        else: parent[rb]=ra; rank[ra]+=1
        return True
    deg=[0]*n; edges=set()
    def can_add(u,v,remaining):
        if u==v: return False
        if deg[u]>=2 or deg[v]>=2: return False
        if find(u)==find(v) and remaining>1: return False
        return True

    remaining=n
    cand=[]
    for u in range(n):
        for v in sparse_adj.get(u,[]):
            if u<v: cand.append((dist(u,v),u,v))
    cand.sort()
    for w,u,v in cand:
        if can_add(u,v,remaining):
            edges.add((min(u,v),max(u,v)))
            deg[u]+=1; deg[v]+=1
            union(u,v); remaining-=1

    from collections import defaultdict as DD
    adj = DD(list)
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    start = 0
    for i in range(n):
        if len(adj[i])<2: start=i; break
    path=[start]; used={start}
    while len(path)<n:
        cur=path[-1]
        nxts=[x for x in adj[cur] if x not in used]
        if nxts:
            nx=nxts[0]
        else:
            best=1e18; nx=None
            for cand in fallback_adj.get(cur,[]):
                if cand in used: continue
                if deg[cur]>=2 or deg[cand]>=2: continue
                d=dist(cur,cand)
                if d<best: best=d; nx=cand
            if nx is None:
                for cand in range(n):
                    if cand not in used:
                        nx=cand; break
            edges.add((min(cur,nx),max(cur,nx)))
            deg[cur]+=1; deg[nx]+=1
            adj[cur].append(nx); adj[nx].append(cur)
        path.append(nx); used.add(nx)
    return path

# ---------------- Solver ----------------

@dataclass
class BeeTSPSolver:
    cfg: Dict[str, Any]

    def solve(self, instance_name: str, instance_path: str, seed: int = 0) -> Dict[str, Any]:
        random.seed(seed)
        if np is not None:
            try:
                np.random.seed(seed)
            except Exception:
                pass

        ins = load_tsplib(instance_path)
        n, coords, dist = ins['n'], ins['coords'], ins['dist']

        cand_k = int(self.cfg['candidate'].get('k', 32))  # slightly smaller default for speed
        use_knn = bool(self.cfg['candidate'].get('use_knn', True))

        target_zone = int(self.cfg['zones'].get('target_zone_size', 300))
        overlap_pct = float(self.cfg['zones'].get('overlap_pct', 0.10))
        agents_per_zone = int(self.cfg['zones'].get('agents_per_zone', 6))

        kick_period = int(self.cfg['local_search'].get('kick_period_moves', 500))
        agent_time = float(self.cfg['bees'].get('time_budget_s', 0.6))

        smoothing_eps = float(self.cfg['ehm'].get('smoothing_eps', 0.02))
        max_edges_per_node = int(self.cfg['ehm'].get('max_edges_per_node', 16))

        merge_cap = int(self.cfg['integrator'].get('merge_edge_cap_per_node', 10))

        wall_time = float(self.cfg['termination'].get('wall_time_s', 1800.0))
        start_time = time.time()

        candidate_adj = build_knn_candidates(coords, n, dist, cand_k) if use_knn else {i: [] for i in range(n)}
        # Pre-sort neighbors for fast candidate NN init
        cand_sorted = sorted_candidates_by_dist(n, candidate_adj, dist)

        # Zones
        zones = [list(range(n))]  # large-n: global agents
        if coords is not None and np is not None and target_zone < n:
            zones = [list(range(n))] if target_zone >= n else [list(range(n))]  # simplify for speed

        ehm = EdgeHistogram(n, smoothing_eps, max_edges_per_node)
        best_len = INF; best_tour = list(range(n)); anytime=[]

        def maybe_record(tour):
            nonlocal best_len, best_tour, anytime
            L = tour_length(tour, dist)
            if L < best_len:
                best_len = L; best_tour = tour[:]
                anytime.append((time.time()-start_time, float(L)))

        while (time.time() - start_time) < wall_time:
            pool = []
            for zone in zones:
                if (time.time() - start_time) >= wall_time: break
                for _ in range(agents_per_zone):
                    st = random.choice(zone)
                    # FAST init on candidates
                    tour = nn_init_on_candidates(n, dist, cand_sorted, start=st)
                    tour = local_search_2opt(tour, dist, candidate_adj, kick_period=kick_period, time_budget_s=agent_time)
                    pool.append(tour[:]); ehm.update(tour); maybe_record(tour)
                    if (time.time() - start_time) >= wall_time: break
                if (time.time() - start_time) >= wall_time: break

            # POPMUSIC-like merge
            pool_sorted = sorted(pool, key=lambda t: tour_length(t, dist))
            take = max(2, min(10, len(pool_sorted)))
            sparse_adj = {i: [] for i in range(n)}
            def add_edge(u,v):
                if v not in sparse_adj[u]: sparse_adj[u].append(v)
                if u not in sparse_adj[v]: sparse_adj[v].append(u)
            for t in pool_sorted[:take]:
                for i in range(n):
                    u = t[i]; v = t[(i+1)%n]
                    if len(sparse_adj[u]) < merge_cap:
                        add_edge(u,v)
            merged = nn_init_on_candidates(n, dist, cand_sorted, start=pool_sorted[0][0] if pool_sorted else 0)
            merged = local_search_2opt(merged, dist, candidate_adj, kick_period=kick_period, time_budget_s=max(0.3, agent_time/2))
            maybe_record(merged)

            # Scouts
            for _ in range(2):
                scout = ehm.sample_tour(dist, candidate_adj)
                scout = local_search_2opt(scout, dist, candidate_adj, kick_period=kick_period, time_budget_s=max(0.3, agent_time/2))
                maybe_record(scout)

        return {"best_length": float(best_len), "best_tour": best_tour, "anytime": anytime}
