
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
    from scipy.spatial import Delaunay
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

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

def apply_two_opt(tour: List[int], i: int, k: int):
    if k < i: i, k = k, i
    tour[i+1:k+1] = reversed(tour[i+1:k+1])

def double_bridge_kick(tour: List[int]):
    n = len(tour)
    if n < 8: 
        random.shuffle(tour); return
    a = random.randint(1, n//4)
    b = random.randint(a+1, n//2)
    c = random.randint(b+1, 3*n//4)
    d = random.randint(c+1, n-1)
    new = tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]
    tour[:] = new

def nearest_neighbor_init(n: int, dist, start: Optional[int]=None) -> List[int]:
    start = 0 if start is None else start
    used = [False]*n; used[start] = True
    t = [start]
    for _ in range(n-1):
        u = t[-1]; best, bestj = 1e18, -1
        for j in range(n):
            if not used[j]:
                d = dist(u,j)
                if d < best: best, bestj = d, j
        used[bestj] = True; t.append(bestj)
    return t

def build_knn_candidates(coords: Optional[List[Tuple[float,float]]], n: int, dist, k: int) -> Dict[int, List[int]]:
    adj = {i: [] for i in range(n)}
    if coords is not None and np is not None:
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
    for i in range(n):
        for j in adj[i]:
            if i not in adj[j]:
                adj[j].append(i)
    return adj

def add_delaunay_edges(adj: Dict[int, List[int]], coords: Optional[List[Tuple[float,float]]]) -> None:
    if not _HAS_SCIPY or coords is None or np is None: return
    pts = np.array(coords, dtype=float)
    tri = Delaunay(pts)
    for a,b,c in tri.simplices:
        for u,v in [(int(a),int(b)),(int(b),int(c)),(int(c),int(a))]:
            if v not in adj[u]: adj[u].append(v)
            if u not in adj[v]: adj[v].append(u)

def kmeans_partition(coords: List[Tuple[float,float]], target_zone_size: int, overlap_pct: float, seed: int = 0):
    n = len(coords)
    if target_zone_size <= 0 or target_zone_size >= n or np is None:
        return [list(range(n))]
    k = max(1, int(math.ceil(n / float(target_zone_size))))
    rng = np.random.default_rng(seed)
    pts = np.array(coords, dtype=float)

    centers = [pts[rng.integers(0, len(pts))]]
    while len(centers) < k:
        d2 = np.min([np.sum((pts - c)**2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum()
        centers.append(pts[rng.choice(len(pts), p=probs)])
    centers = np.vstack(centers)
    for _ in range(15):
        dists = np.sum((pts[:,None,:] - centers[None,:,:])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.vstack([pts[labels==j].mean(axis=0) if np.any(labels==j) else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers): break
        centers = new_centers
    zones = [np.where(labels==j)[0].tolist() for j in range(k)]
    if overlap_pct > 0 and k > 1:
        order = np.argsort(dists, axis=1)
        nearest = order[:,0]; second = order[:,1]
        margin = dists[np.arange(n), second] - dists[np.arange(n), nearest]
        thr = np.quantile(margin, overlap_pct)
        for i in range(n):
            if second[i] != nearest[i] and margin[i] <= thr:
                zones[second[i]].append(int(i))
    zones = [sorted(set(z)) for z in zones if z]
    return zones or [list(range(n))]

def local_search_2opt(tour: List[int], dist, candidate_adj: Dict[int, List[int]], kick_period: int = 300, time_budget_s: float = 2.0):
    n = len(tour)
    best_len = tour_length(tour, dist)
    moves_since = 0
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget_s:
        improved = False
        for i in range(n):
            b = tour[(i+1)%n]
            for c in candidate_adj.get(b, []):
                k = tour.index(c)
                if k == i or (k+1)%n == i: 
                    continue
                dlt = two_opt_delta(tour, i, k, dist)
                if dlt < 0:
                    apply_two_opt(tour, i, k)
                    best_len += dlt
                    improved = True
                    moves_since = 0
                    break
            if improved:
                break
        moves_since += 1
        if moves_since >= kick_period and (time.time() - start) < time_budget_s:
            double_bridge_kick(tour)
            best_len = tour_length(tour, dist)
            moves_since = 0
    return tour

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

    # complete path
    from collections import defaultdict as DD
    adj = DD(list)
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    # start at a node with deg<2
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

        cand_k = int(self.cfg['candidate'].get('k', 30))
        use_delaunay = bool(self.cfg['candidate'].get('use_delaunay', True))
        use_knn = bool(self.cfg['candidate'].get('use_knn', True))

        target_zone = int(self.cfg['zones'].get('target_zone_size', 200))
        overlap_pct = float(self.cfg['zones'].get('overlap_pct', 0.15))
        agents_per_zone = int(self.cfg['zones'].get('agents_per_zone', 4))

        kick_period = int(self.cfg['local_search'].get('kick_period_moves', 300))
        agent_time = float(self.cfg['bees'].get('time_budget_s', 2.0))

        smoothing_eps = float(self.cfg['ehm'].get('smoothing_eps', 0.03))
        max_edges_per_node = int(self.cfg['ehm'].get('max_edges_per_node', 16))

        merge_cap = int(self.cfg['integrator'].get('merge_edge_cap_per_node', 12))

        wall_time = float(self.cfg['termination'].get('wall_time_s', 60.0))
        start_time = time.time()

        candidate_adj = build_knn_candidates(coords, n, dist, cand_k) if use_knn else {i: [] for i in range(n)}
        if use_delaunay:
            add_delaunay_edges(candidate_adj, coords)
        for u in range(n):
            nbrs = candidate_adj[u]
            nbrs = sorted(nbrs, key=lambda v: dist(u,v))[:max(cand_k, len(nbrs))]
            candidate_adj[u] = nbrs

        zones = kmeans_partition(coords if coords is not None else [(i,0.0) for i in range(n)], target_zone, overlap_pct, seed=seed)

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
                    tour = nearest_neighbor_init(n, dist, start=st)
                    tour = local_search_2opt(tour, dist, candidate_adj, kick_period=kick_period, time_budget_s=agent_time)
                    pool.append(tour[:]); ehm.update(tour); maybe_record(tour)
                    if (time.time() - start_time) >= wall_time: break
                if (time.time() - start_time) >= wall_time: break

            # POPMUSIC-like merge: union top tours
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
            merged = greedy_completion_from_sparse(n, dist, sparse_adj, candidate_adj)
            merged = local_search_2opt(merged, dist, candidate_adj, kick_period=kick_period, time_budget_s=max(0.5, agent_time/2))
            maybe_record(merged)

            # Scouts from EHM
            for _ in range(2):
                scout = ehm.sample_tour(dist, candidate_adj)
                scout = local_search_2opt(scout, dist, candidate_adj, kick_period=kick_period, time_budget_s=max(0.5, agent_time/2))
                maybe_record(scout)

        return {"best_length": float(best_len), "best_tour": best_tour, "anytime": anytime}
