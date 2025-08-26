from __future__ import annotations
from typing import Dict, Any

def autoscale_by_n(n: int, base: Dict[str, Any]) -> Dict[str, Any]:
    """
    Size-aware parameter selection.
    Rules reflect current evidence across ~100..85k nodes and a tighter small/mid bracket (n<=6000).
    """
    s = dict(base)  # shallow copy
    solver = dict(s.get("solver", {}))

    cand = dict(solver.get("candidate", {}))
    zones = dict(solver.get("zones", {}))
    bees  = dict(solver.get("bees", {}))
    ls    = dict(solver.get("local_search", {}))
    integ = dict(solver.get("integrator", {}))
    term  = dict(solver.get("termination", {}))

    # -------- Heuristics --------
    if n <= 6000:
        # Small / lower-mid (e.g., fnl4461): global zone, deeper candidates, more polish
        k = 36
        zone_size = 999_999  # single zone
        overlap = 0.0
        agents = 6
        bee_time = 0.8
    elif n <= 10000:
        # Mid
        k = 28
        zone_size = 3000
        overlap = 0.005
        agents = 4
        bee_time = 0.6
    elif n <= 30000:
        # Upper-mid
        k = 26
        zone_size = 3000
        overlap = 0.005
        agents = 4
        bee_time = 0.6
    else:
        # Large / XL (e.g., pla85900)
        k = 24
        zone_size = 3000
        overlap = 0.003
        agents = 3
        bee_time = 0.5

    # Candidate defaults
    cand.setdefault("use_knn", True)
    cand.setdefault("use_delaunay", False)
    cand["k"] = int(k)

    # Zoning
    zones.setdefault("method", "kmeans")
    zones["target_zone_size"] = int(zone_size)
    zones["overlap_pct"] = float(overlap)
    zones["agents_per_zone"] = int(agents)

    # Bees
    bees["time_budget_s"] = float(bee_time)

    # Stable defaults
    ls.setdefault("kick_period_moves", 800)
    integ.setdefault("merge_edge_cap_per_node", 10)
    term.setdefault("wall_time_s", 1800)

    # Pack
    solver["candidate"] = cand
    solver["zones"] = zones
    solver["bees"] = bees
    solver["local_search"] = ls
    solver["integrator"] = integ
    solver["termination"] = term
    s["solver"] = solver
    return s

def autoscale_v2(base, feats):
    """
    Advanced autoscaling using S·S·T features for topology-aware parameter selection.
    """
    cfg = dict(base)  # shallow copy is OK for our keys
    n = feats["n"]; delta = feats["density_idx"]; a = feats["anisotropy"]
    h = feats["hull_frac"]; C = feats["cluster_score"]

    # size tiers (keep v1 logic, then apply deltas)
    if n <= 6000:    k, zone, ov, agents, bee = 36, None, 0.0, 6, 0.8
    elif n <= 10000: k, zone, ov, agents, bee = 28, 3000, 0.005, 4, 0.6
    elif n <= 30000: k, zone, ov, agents, bee = 26, 3000, 0.005, 4, 0.6
    elif n <= 60000: k, zone, ov, agents, bee = 26, 3000, 0.004, 4, 0.6  # new band (e.g., pla33810)
    else:            k, zone, ov, agents, bee = 24, 3000, 0.003, 3, 0.5

    # sparsity/topology deltas
    if delta < 0.9:                 # clustered
        k += 6; ov += 0.003
        if zone: zone = min(zone, 2500)
        cfg.setdefault("solver", {}).setdefault("routing", {})["cfrs"] = True
    if delta > 1.3:                 # diffuse/outliers
        k -= 4; ov = max(ov - 0.002, 0.0)
    if a > 8.0:                     # corridor
        k += 2
        cfg.setdefault("solver", {}).setdefault("zones", {})["stripe_partitions"] = True
    if h > 0.25:                    # ring/perimeter
        ov += 0.002
        cfg.setdefault("solver", {}).setdefault("traps", {})["bee_balls"] = True
        cfg["solver"].setdefault("candidate", {})["alpha_near_cross"] = 1.5
    if C > 0.35:                    # strong clusters
        cfg.setdefault("solver", {}).setdefault("routing", {})["cfrs"] = True

    # apply back
    sol = cfg.setdefault("solver", {})
    cand = sol.setdefault("candidate", {})
    cand["k"] = max(18, int(k))
    z = sol.setdefault("zones", {})
    z["target_zone_size"] = zone or 999999
    z["overlap_pct"] = float(min(ov, 0.01))
    z["agents_per_zone"] = agents
    sol.setdefault("bees", {})["time_budget_s"] = bee
    # always ensure delaunay + knn on small/mid
    cand.setdefault("use_knn", True)
    if n <= 6000: cand["use_delaunay"] = True
    return cfg
