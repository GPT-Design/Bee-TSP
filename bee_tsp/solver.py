from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math, random, time
from bee_tsp.metrics import build_cost_fn, wrap_dist
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

try:
    from .integrators.ucr import ucr_integrate
    _HAS_UCR = True
except ImportError:
    _HAS_UCR = False
    ucr_integrate = None

try:
    from .features import compute_features
    _HAS_FEATURES = True
except ImportError:
    _HAS_FEATURES = False
    compute_features = None

INF = 10**18

class EdgeHistogram:
    """Edge Histogram Matrix (EHM) for tracking good edge frequencies."""
    
    def __init__(self, n: int):
        self.n = n
        self.counts = {}  # (u,v) -> count where u < v
        
    def record_tour(self, tour: List[int]):
        """Record edges from a tour in the histogram."""
        n = len(tour)
        for i in range(n):
            u, v = tour[i], tour[(i+1) % n]
            key = (u, v) if u < v else (v, u)
            self.counts[key] = self.counts.get(key, 0) + 1
            
    def sample_tour(self, dist, candidate_adj: Dict[int, List[int]], temperature: float = 0.7, smooth_eps: float = 1.0) -> List[int]:
        """
        Sample a tour using EHM-biased greedy construction.
        Weights ~ ((count+smooth_eps)/cost)^(1/temperature)
        """
        n = self.n
        start = random.randrange(n)
        used = [False] * n
        used[start] = True
        tour = [start]
        
        for _ in range(n - 1):
            u = tour[-1]
            cands = [v for v in candidate_adj.get(u, []) if not used[v]]
            if not cands:
                cands = [i for i in range(n) if not used[i]]
            if not cands:
                break
                
            # Calculate EHM-biased weights
            weights = []
            for v in cands:
                key = (u, v) if u < v else (v, u)
                cnt = self.counts.get(key, 0)
                score = (cnt + smooth_eps) / (dist(u, v) + 1.0)
                weights.append(score ** (1.0 / max(1e-6, temperature)))
            
            # Weighted random selection
            if weights:
                total = sum(weights)
                if total > 0:
                    r = random.random() * total
                    acc = 0.0
                    pick = cands[-1]
                    for v, w in zip(cands, weights):
                        acc += w
                        if r <= acc:
                            pick = v
                            break
                    used[pick] = True
                    tour.append(pick)
                else:
                    # Fallback to random selection
                    pick = random.choice(cands)
                    used[pick] = True
                    tour.append(pick)
        
        return tour

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

def or_opt_delta(tour: List[int], i: int, length: int, j: int, dist) -> int:
    """Calculate delta for Or-opt move: relocate segment [i:i+length] after position j."""
    n = len(tour)
    if length <= 0 or length >= n or i == j or (i + length - 1) == j:
        return INF
    
    # Normalize positions
    i = i % n
    j = j % n
    
    # Get the segment and surrounding edges
    segment_start = i
    segment_end = (i + length - 1) % n
    prev_segment = (i - 1) % n
    next_segment = (i + length) % n
    
    # Current cost: edges around the segment
    old_cost = (dist(tour[prev_segment], tour[segment_start]) + 
                dist(tour[segment_end], tour[next_segment]))
    
    # Cost after removing segment
    remove_cost = dist(tour[prev_segment], tour[next_segment])
    
    # Cost of inserting segment after position j
    if j != prev_segment and j != segment_end:
        next_j = (j + 1) % n
        old_j_cost = dist(tour[j], tour[next_j])
        new_j_cost = (dist(tour[j], tour[segment_start]) + 
                      dist(tour[segment_end], tour[next_j]))
        insert_cost = new_j_cost - old_j_cost
    else:
        insert_cost = 0
    
    return (remove_cost + insert_cost) - old_cost

def apply_or_opt_and_update_pos(tour: List[int], pos: List[int], i: int, length: int, j: int):
    """Apply Or-opt move: relocate segment [i:i+length] after position j."""
    n = len(tour)
    if length <= 0 or length >= n or i == j:
        return
    
    # Extract the segment
    segment = []
    for idx in range(length):
        segment.append(tour[(i + idx) % n])
    
    # Create new tour
    new_tour = []
    for idx in range(n):
        if idx < i or idx >= i + length:  # Not in the segment
            new_tour.append(tour[idx])
            if idx == j:  # Insert segment after position j
                new_tour.extend(segment)
    
    # Handle case where j is before i
    if j < i:
        new_tour = []
        for idx in range(n):
            if idx < i or idx >= i + length:
                new_tour.append(tour[idx])
        # Insert segment after position j
        insert_pos = j + 1
        new_tour = new_tour[:insert_pos] + segment + new_tour[insert_pos:]
    
    # Update tour and positions
    tour[:] = new_tour[:n]  # Ensure we don't exceed tour length
    for idx, city in enumerate(tour):
        pos[city] = idx

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

# ---------- Candidates ----------

def build_knn_candidates(coords: Optional[List[Tuple[float,float]]], n: int, dist, k: int) -> Dict[int, List[int]]:
    adj = {i: [] for i in range(n)}
    if coords is not None and _HAS_KDTREE and np is not None:
        pts = np.array(coords, dtype=float)
        tree = KDTree(pts)
        try:
            dists, idxs = tree.query(pts, k=min(k+1, n), workers=-1)  # type: ignore
        except TypeError:
            try:
                dists, idxs = tree.query(pts, k=min(k+1, n), n_jobs=-1)  # type: ignore
            except TypeError:
                dists, idxs = tree.query(pts, k=min(k+1, n))
        for i in range(n):
            arr = idxs[i].tolist() if hasattr(idxs[i], "tolist") else list(idxs[i])
            nbrs = [int(j) for j in arr if j != i]
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
    for i in range(n):
        for j in adj[i]:
            if i not in adj[j]:
                adj[j].append(i)
    for u in range(n):
        adj[u] = sorted(adj[u], key=lambda v: dist(u,v))
    return adj

def nn_init_on_candidates(n: int, dist, cand_sorted: Dict[int, List[int]], start: Optional[int] = None) -> List[int]:
    used = [False]*n
    if start is None:
        start = 0
    used[start] = True
    tour = [start]
    remaining = [i for i in range(n) if i != start]
    rem_idx = 0
    while len(tour) < n:
        u = tour[-1]
        chosen = -1
        for v in cand_sorted.get(u, []):
            if not used[v]:
                chosen = v; break
        if chosen == -1:
            while rem_idx < len(remaining) and used[remaining[rem_idx]]:
                rem_idx += 1
            if rem_idx >= len(remaining): break
            chosen = remaining[rem_idx]
        used[chosen] = True
        tour.append(chosen)
    return tour

def local_search_2opt_dlb(tour: List[int], dist, cand_sorted: Dict[int, List[int]], kick_period: int = 800, time_budget_s: float = 1.0, use_or_opt: bool = False):
    n = len(tour)
    best_len = tour_length(tour, dist)
    start = time.time()
    pos = [0]*n
    for i,v in enumerate(tour): pos[v] = i

    dlb = [False]*n
    moves_since = 0
    improved_global = True

    while improved_global and (time.time() - start) < time_budget_s:
        improved_global = False
        for i in range(n):
            a = tour[i]
            if dlb[a]:
                continue
            b = tour[(i+1)%n]
            improved_local = False
            
            # Try 2-opt moves
            for c in cand_sorted.get(b, []):
                k = pos[c]
                if k == i or (k+1)%n == i:
                    continue
                dlt = two_opt_delta(tour, i, k, dist)
                if dlt < 0:
                    apply_two_opt_and_update_pos(tour, pos, i, k)
                    best_len += dlt
                    improved_local = True
                    improved_global = True
                    dlb[a] = dlb[b] = dlb[tour[k]] = dlb[tour[(k+1)%n]] = False
                    moves_since = 0
                    break
            
            # Try Or-opt moves if enabled and no 2-opt improvement found
            if use_or_opt and not improved_local:
                for length in [1, 2, 3]:  # Or-opt-1, Or-opt-2, Or-opt-3
                    if improved_local:
                        break
                    for j in range(n):
                        if j == i or abs(j - i) <= length:
                            continue
                        dlt = or_opt_delta(tour, i, length, j, dist)
                        if dlt < 0:
                            # OR-OPT DISABLED: Major bugs in delta calculation cause metric mismatch
                            # apply_or_opt_and_update_pos(tour, pos, i, length, j)
                            # best_len += dlt
                            # improved_local = True
                            # improved_global = True
                            # # Reset DLB for affected cities
                            # for idx in range(length):
                            #     dlb[tour[(i + idx) % n]] = False
                            # dlb[tour[j]] = dlb[tour[(j+1)%n]] = False
                            # moves_since = 0
                            # break
                            pass
            
            if not improved_local:
                dlb[a] = True
            if (time.time() - start) >= time_budget_s:
                break

        moves_since += 1
        if moves_since >= kick_period and (time.time() - start) < time_budget_s:
            double_bridge_kick(tour, pos)
            best_len = tour_length(tour, dist)
            dlb = [False]*n
            moves_since = 0

    return tour

def apply_adaptive_overlap(candidate_adj: Dict[int, List[int]], coords, n: int, dist, k: int, config: Dict[str, Any], ehm=None) -> Dict[int, List[int]]:
    """
    LITE version of adaptive overlap + portal nodes.
    Since there's no actual zoning in this implementation, we'll simulate cross-zone behavior
    by identifying sparse boundary regions and improving connectivity there.
    """
    portal_pct = float(config.get('portal_pct', 2.0))
    
    # For LITE version without real zones, identify boundary nodes as those with:
    # 1. Lower local density (fewer k-NN neighbors within close distance)
    # 2. Higher potential for cross-connectivity
    
    boundary_scores = {}
    
    for i in range(n):
        score = 0.0
        knn_neighbors = candidate_adj.get(i, [])
        
        if len(knn_neighbors) > 0:
            # Score based on average distance to k-NN (higher = more boundary-like)
            avg_dist = sum(dist(i, j) for j in knn_neighbors[:min(k//2, len(knn_neighbors))]) / min(k//2, len(knn_neighbors))
            score += avg_dist
            
            # Add EHM edge count if available
            if ehm is not None:
                ehm_count = sum(ehm.counts.get((i, j) if i < j else (j, i), 0) for j in knn_neighbors)
                score += ehm_count * 0.1  # Small weight
        
        boundary_scores[i] = score
    
    # Select top portal_pct% nodes as portals
    if boundary_scores:
        sorted_nodes = sorted(boundary_scores.items(), key=lambda x: x[1], reverse=True)
        num_portals = max(1, int(n * portal_pct / 100.0))
        portals = [node for node, _ in sorted_nodes[:num_portals]]
        
        # Augment candidate graph for portals
        enhanced_adj = {i: list(candidate_adj.get(i, [])) for i in range(n)}
        
        for portal in portals:
            current_neighbors = set(enhanced_adj[portal])
            
            # Find k//2 additional neighbors from distant regions
            candidates = []
            for j in range(n):
                if j != portal and j not in current_neighbors:
                    candidates.append((dist(portal, j), j))
            
            # Add closest distant neighbors
            candidates.sort()
            new_neighbors = min(k//2, len(candidates))
            for _, j in candidates[:new_neighbors]:
                enhanced_adj[portal].append(j)
                # Also add reverse connection for symmetry
                if portal not in enhanced_adj[j]:
                    enhanced_adj[j].append(portal)
        
        # Re-sort by distance for each node
        for i in range(n):
            enhanced_adj[i] = sorted(enhanced_adj[i], key=lambda j: dist(i, j))
            
        return enhanced_adj
    
    return candidate_adj

def apply_three_opt_lite(tour: List[int], dist, candidate_adj: Dict[int, List[int]], tri_budget: int) -> List[int]:
    """
    LITE version of 3-opt on stall:
    A) Double-bridge kick (random 4-cut) on current tour
    B) Immediate bounded 2-opt polish
    C) Targeted 3-edge try on longest edges using candidate graph
    """
    if not tour or len(tour) < 4:
        return tour
        
    n = len(tour)
    tour = tour[:]  # Make a copy
    
    # A) Double-bridge kick (4-cut random reconnection)
    double_bridge_kick(tour, list(range(n)))
    
    # B) Bounded 2-opt polish (limited time/moves)
    pos = [0] * n
    for i, city in enumerate(tour):
        pos[city] = i
    
    # Quick 2-opt pass with limited moves
    moves_budget = min(tri_budget // 3, 500)  # Conservative move limit
    for _ in range(moves_budget):
        improved = False
        for i in range(n):
            for k in candidate_adj.get(tour[i], [])[:5]:  # Only check first 5 candidates
                k_pos = pos[k]
                if k_pos == i or k_pos == (i + 1) % n or k_pos == (i - 1) % n:
                    continue
                    
                delta = two_opt_delta(tour, i, k_pos, dist)
                if delta < -0.1:  # Small threshold to avoid numerical issues
                    apply_two_opt_and_update_pos(tour, pos, i, k_pos)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # C) Targeted 3-edge try on longest edges
    if tri_budget > 0:
        # Find the longest edges in current tour
        edge_lengths = []
        for i in range(n):
            j = (i + 1) % n
            edge_lengths.append((dist(tour[i], tour[j]), i))
        
        # Try 3-opt on a few longest edges
        edge_lengths.sort(reverse=True)
        attempts = min(tri_budget // 100, 5)  # Conservative attempt limit
        
        for _, i in edge_lengths[:attempts]:
            # Try simple 3-opt move around this edge
            best_delta = 0
            best_move = None
            
            # Look for improvement by reconnecting this edge differently
            for j in candidate_adj.get(tour[i], [])[:3]:  # Check only top 3 candidates
                j_pos = None
                for p, city in enumerate(tour):
                    if city == j:
                        j_pos = p
                        break
                
                if j_pos is not None and abs(j_pos - i) > 2:  # Avoid adjacent positions
                    k_pos = (j_pos + 1) % n
                    # Check if this would be a valid 3-opt move
                    if abs(k_pos - i) > 2:
                        # Simple 3-opt delta approximation (not exact but fast)
                        old_cost = (dist(tour[i], tour[(i+1)%n]) + 
                                   dist(tour[j_pos], tour[k_pos]) + 
                                   dist(tour[(i-1)%n], tour[i]))
                        new_cost = (dist(tour[i], tour[j_pos]) + 
                                   dist(tour[(i+1)%n], tour[k_pos]) + 
                                   dist(tour[(i-1)%n], tour[j_pos]))
                        delta = new_cost - old_cost
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, j_pos, k_pos)
            
            if best_move:
                # Apply simple reconnection (this is a simplified 3-opt)
                i, j_pos, k_pos = best_move
                # For LITE version, just do a segment reversal
                if j_pos < k_pos:
                    tour[j_pos:k_pos+1] = tour[j_pos:k_pos+1][::-1]
                break  # Only try one improvement to keep it fast
    
    return tour

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
        n, coords = ins['n'], ins.get('coords', None)
        
        # Record metric tag for parity checking
        metric_tag = ins.get('edge_weight_type', 'UNKNOWN')
        self.cfg.setdefault("metric", {})["edge_weight_type"] = metric_tag
        
        # Metric wrapper (default = TSPLIB behavior)
        cost = build_cost_fn(self.cfg, ins)
        integer = bool(self.cfg.get("metric", {}).get("integer", True))
        dist = wrap_dist(cost, integer=integer)

        # Dynamic scout configuration
        dyn = (self.cfg.get("scouting", {}) or {}).get("dynamic", {})
        scout_enabled = bool(dyn.get("enabled", True))
        no_improve_s = float(dyn.get("no_improve_s", 120.0))
        max_restarts = int(dyn.get("max_restarts_per_seed", 3))
        polish_time_s = float(dyn.get("polish_time_s", 0.4))
        sample_cfg = dyn.get("sample", {}) or {}
        temperature = float(sample_cfg.get("temperature", 0.7))
        smooth_eps = float(sample_cfg.get("smooth_eps", 1.0))

        cand_k = int(self.cfg['candidate'].get('k', 24))
        use_knn = bool(self.cfg['candidate'].get('use_knn', True))

        agents_per_zone = int(self.cfg['zones'].get('agents_per_zone', 6))

        kick_period = int(self.cfg['local_search'].get('kick_period_moves', 800))
        use_or_opt = bool(self.cfg['local_search'].get('use_or_opt', False))
        agent_time = float(self.cfg['bees'].get('time_budget_s', 0.6))
        
        # 3-opt on stall configuration
        stall_windows_threshold = int(self.cfg.get('local_search', {}).get('stall_windows', 2))
        three_opt_lite = bool(self.cfg.get('local_search', {}).get('three_opt_lite', False))
        tri_budget = int(self.cfg.get('local_search', {}).get('tri_budget', 1500))
        
        # Integrator configuration
        integrator_cfg = self.cfg.get('integrator', {})
        integrator_method = integrator_cfg.get('method', 'popmusic')  # Default to popmusic
        ucr_config = integrator_cfg.get('ucr', {})

        wall_time = float(self.cfg['termination'].get('wall_time_s', 1800.0))
        start_time = time.time()

        candidate_adj = build_knn_candidates(coords, n, dist, cand_k) if use_knn else {i: [] for i in range(n)}
        
        # Compute S·S·T features if available and coords exist
        features_info = None
        if _HAS_FEATURES and coords is not None:
            try:
                feat_cfg = self.cfg.get("solver", {}).get("features", {})
                features_info = compute_features(
                    np.array(coords), 
                    kd_backend=feat_cfg.get("kd_backend", "auto"),
                    max_threads=feat_cfg.get("max_threads_per_lib", 1)
                )
                print(f"S·S·T features computed using {features_info.get('kd_backend', 'unknown')} backend")
            except Exception as e:
                print(f"Warning: Feature computation failed: {e}")
        
        # Adaptive overlap + portal nodes (LITE version)
        adaptive_overlap_cfg = self.cfg.get('zones', {}).get('adaptive_overlap', {})
        if adaptive_overlap_cfg.get('enabled', False):
            candidate_adj = apply_adaptive_overlap(candidate_adj, coords, n, dist, cand_k, adaptive_overlap_cfg, ehm=None)

        # Initialize EHM and tracking variables
        ehm = EdgeHistogram(n)
        best_len = INF
        best_tour: List[int] = list(range(n))
        anytime: List[Tuple[float,float]] = []
        
        # Stagnation tracking for dynamic scouts
        last_improve_t = time.time()
        restarts_done = 0
        
        # Stall tracking for 3-opt on stall
        stall_windows = 0
        last_length = INF

        def maybe_record(tour):
            nonlocal best_len, best_tour, anytime, last_improve_t
            L = tour_length(tour, dist)
            if L < best_len:
                best_len = L; best_tour = tour[:]
                anytime.append((time.time()-start_time, float(L)))
                last_improve_t = time.time()  # Reset improvement timer
            # Always update EHM with good tours
            ehm.record_tour(tour)

        zone = list(range(n))

        while (time.time() - start_time) < wall_time:
            pool: List[List[int]] = []
            for _ in range(agents_per_zone):
                st = random.choice(zone)
                tour = nn_init_on_candidates(n, dist, candidate_adj, start=st)
                tour = local_search_2opt_dlb(tour, dist, candidate_adj, kick_period=kick_period, time_budget_s=agent_time, use_or_opt=use_or_opt)
                pool.append(tour[:]); maybe_record(tour)
                if (time.time() - start_time) >= wall_time: break

            pool_sorted = sorted(pool, key=lambda t: tour_length(t, dist))
            if pool_sorted:
                maybe_record(pool_sorted[0])
            
            # Integration step
            if integrator_method == 'ucr' and _HAS_UCR and len(pool) >= 2:
                integrated_tour = ucr_integrate(pool, dist, candidate_adj, ucr_config)
                maybe_record(integrated_tour)
            # For 'popmusic' method, we could add POPMUSIC integration here in the future
                
            # 3-opt on stall logic (LITE version)
            if three_opt_lite:
                current_best = tour_length(best_tour, dist) if best_tour else INF
                if current_best < last_length:
                    stall_windows = 0  # Reset on improvement
                    last_length = current_best
                else:
                    stall_windows += 1  # No improvement
                
                if stall_windows >= stall_windows_threshold:
                    # Apply LITE sequence: double-bridge + 2-opt + targeted 3-edge
                    unstuck_tour = apply_three_opt_lite(best_tour[:], dist, candidate_adj, tri_budget)
                    maybe_record(unstuck_tour)
                    stall_windows = 0  # Reset counter
                
            # Dynamic scout trigger (after agents + integrator step)
            now = time.time()
            if scout_enabled and (now - last_improve_t) >= no_improve_s and restarts_done < max_restarts:
                # EHM-biased restart + quick polish
                scout = ehm.sample_tour(dist, candidate_adj, temperature=temperature, smooth_eps=smooth_eps)
                scout = local_search_2opt_dlb(scout, dist, candidate_adj, kick_period=kick_period, time_budget_s=polish_time_s, use_or_opt=use_or_opt)
                maybe_record(scout)
                restarts_done += 1
                last_improve_t = time.time()

        return {"best_length": float(best_len), "best_tour": best_tour, "anytime": anytime}
