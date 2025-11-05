from __future__ import annotations

# --- stdlib ---
import random, time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

# --- project locals (hard deps for Mark 1) ---
from bee_tsp.tsplib import load_tsplib
from bee_tsp.distance import Distance
from bee_tsp.initializers import build_initial_tours

# --- optional integrators (guarded) ---
try:
    from bee_tsp.integrators.ucr import ucr_integrate as _ucr_integrate
    _HAS_UCR = True
except Exception:
    _ucr_integrate = None  # type: ignore
    _HAS_UCR = False

# --- optional adaptive module (guarded) ---
try:
    import bee_tsp.adaptive as _adaptive_mod  # noqa: F401 (may be unused until Mark 2)
    _HAS_ADAPTIVE = True
except Exception:
    _adaptive_mod = None  # type: ignore
    _HAS_ADAPTIVE = False

# --- optional features (guarded) ---
try:
    from bee_tsp.features import compute_features as _compute_features
    _HAS_FEATURES = True
except Exception:
    _compute_features = None  # type: ignore
    _HAS_FEATURES = False

# --- optional numeric backends (guarded) ---
try:
    import numpy as np  # noqa: F401
except Exception:
    np = None  # type: ignore

# --- optional: KDTree (scipy) with alias + fallback ---
try:
    from scipy.spatial import cKDTree as _CKDTREE
    KDTree = _CKDTREE  # runtime alias so references to KDTree resolve
    _HAS_KDTREE = True
except Exception:
    KDTree = None      # type: ignore
    _HAS_KDTREE = False

try:
    from scipy import spatial as _spatial  # noqa: F401
    _HAS_SCIPY = True
except Exception:
    _spatial = None  # type: ignore
    _HAS_SCIPY = False

# add this to silence “not accessed” without functional impact:
_HAS_SCIPY = bool(_HAS_SCIPY)  # no-op to mark access

# ---- module-level constants (safe to keep near the top) ----
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
                score = (cnt + smooth_eps) / (dist.d(u, v) + 1.0)
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
        s += dist.d(tour[i], tour[(i+1)%n])
    return s

def two_opt_delta(tour: List[int], i: int, k: int, dist) -> int:
    n = len(tour)
    a, b = tour[i % n], tour[(i+1) % n]
    c, d = tour[k % n], tour[(k+1) % n]
    return (dist.d(a,c) + dist.d(b,d)) - (dist.d(a,b) + dist.d(c,d))

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
    old_cost = (dist.d(tour[prev_segment], tour[segment_start]) +
                dist.d(tour[segment_end], tour[next_segment]))
    
    # Cost after removing segment
    remove_cost = dist.d(tour[prev_segment], tour[next_segment])
    
    # Cost of inserting segment after position j
    if j != prev_segment and j != segment_end:
        next_j = (j + 1) % n
        old_j_cost = dist.d(tour[j], tour[next_j])
        new_j_cost = (dist.d(tour[j], tour[segment_start]) +
                      dist.d(tour[segment_end], tour[next_j]))
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
                cand.append((dist.d(i,j), j))
            cand.sort()
            adj[i] = [j for _, j in cand[:k]]
    for i in range(n):
        for j in adj[i]:
            if i not in adj[j]:
                adj[j].append(i)
    for u in range(n):
        adj[u] = sorted(adj[u], key=lambda v: dist.d(u,v))
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
    best_len = dist.tour_length(tour)
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
            best_len = dist.tour_length(tour)
            dlb = [False]*n
            moves_since = 0

    return tour

@dataclass
class BeeTSPSolver:
    cfg: Dict[str, Any]

    def solve(self, instance_name: str, instance_path: str, seed: int = 0) -> Dict[str, Any]:
        np.random.seed(seed)
        random.seed(seed)

        ins = load_tsplib(instance_path)
        n, coords = ins['n'], ins.get('coords', None)

        # ---------- METRIC LOCK (Update_24R) ----------
        # Read TSPLIB tag and normalize a few aliases.
        tsplib_metric = str(ins.get("edge_weight_type", "EUC_2D")).upper()
        if tsplib_metric in ("EUC", "EUC2D"):
            tsplib_metric = "EUC_2D"
        elif tsplib_metric == "GEO_2D":
            tsplib_metric = "GEO"

        # Allow config to enforce a metric, if provided.
        config_metric = self.cfg.get("metric_tag")
        if config_metric:
            enforced = str(config_metric).upper()
            if enforced in ("EUC", "EUC2D"):
                enforced = "EUC_2D"
            elif enforced == "GEO_2D":
                enforced = "GEO"

            print(f"[METRIC_LOCK] Enforcing metric_tag='{enforced}' from config (overriding TSPLIB)")
            if tsplib_metric != enforced:
                print(f"[METRIC_LOCK] WARNING: TSPLIB says '{tsplib_metric}' but config enforces '{enforced}'")

            metric_tag = enforced
        else:
            metric_tag = tsplib_metric

        # Keep the legacy name if the rest of the code expects it.
        metric_name = metric_tag

        # Persist for downstream parity checks.
        self.cfg.setdefault("metric", {})["edge_weight_type"] = metric_tag
        # -----------------------------------------------

        # ===== ADAPTIVE ANALYSIS =====
        adaptive_enabled = bool(self.cfg.get('adaptive', {}).get('enabled', False))

        if adaptive_enabled:
            if _HAS_ADAPTIVE:
                try:
                    from bee_tsp.adaptive import (
                        analyze_instance_structure,
                        adaptive_delta_phi_config,
                        size_adaptive_params,
                    )

                    print("[ADAPTIVE] Analyzing instance structure...")
                    # coords, dist, n must already be defined
                    structure_metrics = analyze_instance_structure(coords, dist, n)

                    # Derive configs (each may return None)
                    adaptive_dphi_cfg = adaptive_delta_phi_config(structure_metrics) or {}
                    adaptive_size_cfg = size_adaptive_params(n) or {}

                    # Merge into solver config (non-destructive)
                    self.cfg.setdefault('delta_phi', {}).update(adaptive_dphi_cfg)
                    self.cfg.setdefault('adaptive_size', {}).update(adaptive_size_cfg)

                except Exception as e:
                    print(f"[ADAPTIVE] disabled (error: {e})")
            else:
                print("[ADAPTIVE] WARNING: adaptive module unavailable; skipping")
        # ===== END ADAPTIVE =====

        # Record metric tag for parity checking
        self.cfg.setdefault("metric", {})["edge_weight_type"] = metric_tag

        # STEP 2: Replace ad-hoc distance calls with unified Distance factory
        distance_factory = Distance(metric_tag, coords)
        dist = distance_factory  # Use the Distance object directly - bypasses old cost pipeline completely!
        print(f"[DISTANCE_FACTORY] Created unified distance calculator with metric={metric_tag}")
        print(f"[DISTANCE_FACTORY] BYPASSING old cost pipeline (build_cost_fn + wrap_dist) to ensure unified metric usage")
        
        # Initializer configuration with auto-selection
        init_cfg = self.cfg.get('initializers', {})
        K = init_cfg.get('K', 6)
        
        # Auto-select mix based on instance characteristics (if enabled)
        if 'mix' not in init_cfg or init_cfg.get('auto_select', True):
            # Import the auto_params function
            from bee_tsp.auto_params import choose_initializer_mix
            
            # Get the optimal mix for this instance
            mix = choose_initializer_mix(n, metric_tag)
            print(f"[AUTO_PARAMS] Auto-selected initializer mix for n={n}, metric={metric_tag}: {mix}")
        else:
            # Use manually specified mix from config
            mix = init_cfg.get('mix', ['nn'] * K)
            print(f"[INIT] Using manual mix from config: {mix}")
        
        # Update config with selected mix
        init_cfg['mix'] = mix
        self.cfg['initializers'] = init_cfg
        
        print(f"[DEBUG] K value: {K}")

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
        
        # 3-opt on stall configuration (currently not used)
        _stall_windows_threshold = int(self.cfg.get('local_search', {}).get('stall_windows', 2))
        _three_opt_lite = bool(self.cfg.get('local_search', {}).get('three_opt_lite', False))
        _tri_budget = int(self.cfg.get('local_search', {}).get('tri_budget', 1500))
        
        # Integrator configuration
        integrator_cfg = self.cfg.get('integrator', {})
        integrator_method = integrator_cfg.get('method', 'eax')  # Default to eax (change from popmusic)

        # Extract method-specific config and merge it
        # This allows nested structure: integrator.eax.parents_per_round, etc.
        method_specific_cfg = integrator_cfg.get(integrator_method, {})
        if method_specific_cfg:
            print(f"[INTEGRATOR] Using {integrator_method} with config: {method_specific_cfg}")
            # Merge method-specific params into integrator_cfg
            integrator_cfg = {**integrator_cfg, **method_specific_cfg}
        else:
            print(f"[INTEGRATOR] Using {integrator_method} with default/flat config")

        ucr_config = integrator_cfg.get('ucr', {})

        wall_time = float(self.cfg['termination'].get('wall_time_s', 1800.0))
        start_time = time.time()

        candidate_adj = build_knn_candidates(coords, n, dist, cand_k) if use_knn else {i: [] for i in range(n)}

        # Delta Phi enhancement
        delta_phi_enabled = bool(self.cfg.get('delta_phi', {}).get('enabled', False))
        delta_phi_dict = None

        if delta_phi_enabled:
            from bee_tsp.delta_phi import (
                compute_delta_phi_for_edges,
                filter_candidate_graph_by_delta_phi,
                add_high_delta_phi_edges
            )
    
            delta_phi_config = self.cfg.get('delta_phi', {})
    
            print("[ΔΦ] Computing edge quality metrics...")
            delta_phi_dict = compute_delta_phi_for_edges(
                n, dist, candidate_adj, delta_phi_config
            )
    
            print("[ΔΦ] Filtering candidate graph...")
            candidate_adj = filter_candidate_graph_by_delta_phi(
                candidate_adj, delta_phi_dict, n, dist, delta_phi_config
            )
    
            print("[ΔΦ] Adding high-quality edges...")
            candidate_adj = add_high_delta_phi_edges(
                candidate_adj, delta_phi_dict, n, dist, delta_phi_config
            )

        # ==== FEATURES (read-only; guarded) ==========================================
        feats = None
        try:
            feat_cfg = (self.cfg.get("features") or self.cfg.get("solver", {}).get("features") or {})
            if _HAS_FEATURES and bool(feat_cfg.get("enabled", False)):
                t0_ms = int(time.time() * 1000)
                feats = _compute_features(
                    coords,                               # pass raw coords; no NumPy required
                    kd_backend     = feat_cfg.get("kd_backend", "auto"),
                    max_threads    = int(feat_cfg.get("max_threads", 0) or 0),  # 0 = let lib decide
                    metric         = metric_tag,          # you already defined this earlier
                    time_budget_ms = int(feat_cfg.get("time_budget_ms", 400)),
                    seed           = int(seed),
                )
                if feat_cfg.get("log", True):
                    dt_ms = int(time.time() * 1000) - t0_ms
                    print(f"[FEATURES] ok in {dt_ms} ms · n={feats.get('n')} · nn_mean={feats.get('nn_mean')} · keys≈{sorted(list(feats.keys()))[:6]}")
            else:
                print("[FEATURES] skipped (disabled or unavailable)")
        except Exception as _e:
            print(f"[FEATURES] disabled (error: {type(_e).__name__}: {_e})")
            feats = None
        # ============================================================================
        # ---- MAIN SOLVER LOOP ---- Initialize EHM and tracking variables
        ehm = EdgeHistogram(n)
        best_len = INF
        best_tour: List[int] = list(range(n))
        anytime: List[Tuple[float,float]] = []
        
        # Stagnation tracking for dynamic scouts
        last_improve_t = time.time()
        restarts_done = 0
        
        # Stall tracking for 3-opt on stall - not currently in use
        stall_windows = 0
        last_length = INF

        def maybe_record(tour):
            nonlocal best_len, best_tour, anytime, last_improve_t
            L = dist.tour_length(tour)
            if L < best_len:
                best_len = L; best_tour = tour[:]
                anytime.append((time.time()-start_time, float(L)))
                last_improve_t = time.time()  # Reset improvement timer
            # Always update EHM with good tours
            ehm.record_tour(tour)

        zone = list(range(n))
        
        K = int(self.cfg['initializers'].get('K', 1))
        mix = self.cfg['initializers'].get('mix', ['nn'])
        init_budget_pct = float(self.cfg['initializers'].get('total_budget_pct', 0.05))
        init_budget_s = wall_time * init_budget_pct
        total_budget = wall_time * self.cfg.get('initializers', {}).get('total_budget_pct', 0.25)

        # BUILD INITIAL TOURS
        initial_tours = build_initial_tours(n, dist, candidate_adj, K, total_budget, mix)
        initial_tours.sort(key=lambda x: x[1])

        # Calculate best_length from initial tours (canonical TSPLIB integer)
        best_length = min(dist.tour_length(t) for t, _ in initial_tours)

        # Apply local search to initial tours and seed pool
        pool = []
        for initial_tour, _ in initial_tours:
            tour = local_search_2opt_dlb(initial_tour, dist, candidate_adj, 
                                kick_period=kick_period, 
                                time_budget_s=agent_time, 
                                use_or_opt=use_or_opt)
            maybe_record(tour)
            pool.append(tour[:])
            
            # Update best_length if local search improved
            tour_length = dist.tour_length(tour)
            if tour_length < best_length:
                best_length = tour_length

        # NOW start the main agent loop
        while (time.time() - start_time) < wall_time:
            iteration_tours: List[List[int]] = []  # Temp list for this iteration

            # Generate agent tours (exploration phase)
            for _ in range(agents_per_zone):
                st = random.choice(zone)

                # 80% warm-start from best pool tour, 20% build from scratch
                if pool and random.random() < 0.8:
                    best_pool_t = min(pool, key=lambda t: dist.tour_length(t))
                    tour = best_pool_t[:]
                    double_bridge_kick(tour, None)  # Perturb
                else:
                    tour = nn_init_on_candidates(n, dist, candidate_adj, start=st)

                tour = local_search_2opt_dlb(
                    tour, dist, candidate_adj,
                    kick_period=kick_period,
                    time_budget_s=agent_time,
                    use_or_opt=use_or_opt
                )

                # Quality gate: only keep tours within 5% of best
                tl = dist.tour_length(tour)
                if tl <= best_length * 1.05:
                    iteration_tours.append(tour[:])
                    if tl < best_length:
                        best_length = tl

                maybe_record(tour)
                if (time.time() - start_time) >= wall_time:
                    break

            # Add iteration tours to persistent pool
            pool.extend(iteration_tours)

            # Trim pool to top 30 tours if it gets too large
            if len(pool) > 50:
                pool = sorted(pool, key=lambda t: dist.tour_length(t))[:30]

            # Record best from pool
            if pool:
                best_pool_tour = min(pool, key=lambda t: dist.tour_length(t))
                maybe_record(best_pool_tour)

            # Integration step
            parents_needed = integrator_cfg.get('parents_per_round', 6)

            if len(pool) < parents_needed:
                # Skip if not enough tours
                pass  # Continue to next iteration
            else:
                # Enough tours - run integration
                if integrator_method == 'eax':
                    from bee_tsp.integrators.eax import eax_integrate
                    integrated_tour = eax_integrate(pool, dist, candidate_adj, integrator_cfg)
                    integrated_length = dist.tour_length(integrated_tour)
                    if integrated_length <= best_length * 1.05:
                        pool.append(integrated_tour)
                        maybe_record(integrated_tour)
                        best_length = min(best_length, integrated_length)

                elif integrator_method == 'popmusic':
                    from bee_tsp.integrators.popmusic import popmusic_integrate
                    # Sort pool (best first)
                    pool = sorted(pool, key=lambda t: dist.tour_length(t))
                    
                    print(f"[POPMUSIC] Running with {len(pool)} tours "
                        f"(parents={integrator_cfg.get('parents_per_round', 6)}, "
                        f"cap={integrator_cfg.get('merge_edge_cap_per_node', 8)}, "
                        f"rounds={integrator_cfg.get('rounds', 3)})")
                    
                    integrated_tour = popmusic_integrate(pool, dist, candidate_adj, integrator_cfg)
                    integrated_length = dist.tour_length(integrated_tour)
                    if integrated_length <= best_length * 1.05:
                        pool.append(integrated_tour)
                        maybe_record(integrated_tour)
                        best_length = min(best_length, integrated_length)
                
                elif integrator_method == 'ucr' and _HAS_UCR:
                    integrated_tour = _ucr_integrate(pool, dist, candidate_adj, ucr_config)
                    integrated_length = dist.tour_length(integrated_tour)
                    if integrated_length <= best_length * 1.05:
                        pool.append(integrated_tour)
                        maybe_record(integrated_tour)
                        best_length = min(best_length, integrated_length)
            # END integration step                       
                
            # Dynamic scout trigger (after agents + integrator step)
            now = time.time()
            if scout_enabled and (now - last_improve_t) >= no_improve_s and restarts_done < max_restarts:
                # EHM-biased restart + quick polish
                scout = ehm.sample_tour(dist, candidate_adj, temperature=temperature, smooth_eps=smooth_eps)
                scout = local_search_2opt_dlb(scout, dist, candidate_adj, kick_period=kick_period, time_budget_s=polish_time_s, use_or_opt=use_or_opt)
                maybe_record(scout)
                restarts_done += 1
                last_improve_t = time.time()

        # CRITICAL: Dual metric sentinel to catch any remaining metric drift
        try:
            dist.dual_metric_sentinel(best_tour, int(best_len))
        except AttributeError:
            pass  # Distance object might not have sentinel method in some tests
        except Exception as e:
            print(f"[METRIC_SENTINEL] WARNING: {e}")

        return {
            "instance_name": instance_name,
            "best_length": float(best_len),
            "best_tour": best_tour,
            "anytime": anytime,
            "metric": metric_tag,          # if not already present
            "seed": seed,                  # if not already present
            "features": feats,             # <-- add this line
        }