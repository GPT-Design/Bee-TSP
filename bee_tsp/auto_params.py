"""
auto_params.py — Bee-TSP Mark 1 (Publication build)
Unified auto-configuration with strong safeguards, metric awareness, and clear seams
for future integrators/analysers (Mark 2).

Public API:
  - autoscale(instance_meta: dict) -> dict   # returns {"solver": <full config>}
  - choose_initializer_mix(n: int, metric_tag: str, adaptive: dict|None=None) -> [(name, weight)]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Iterable
import math

# -----------------------------
# Helpers & Normalization
# -----------------------------

def _norm_metric(metric: str | None) -> str:
    if not metric:
        return "UNKNOWN"
    m = metric.strip().upper()
    # Keep canonical TSPLIB labels; policy can specialize per metric.
    if m in ("EUC", "EUC2D", "EUC_2D"):
        return "EUC_2D"
    if m in ("ATT",):
        return "ATT"
    if m in ("CEIL_2D",):
        return "CEIL_2D"
    if m in ("GEO", "GEO_2D"):
        return "GEO"
    return m

def _tier_by_n(n: int) -> str:
    if n < 300:
        return "S"
    if n < 1200:
        return "M"
    if n < 5000:
        return "L"
    if n < 15000:
        return "XL"
    return "XXL"

def _safe_int(x: float, lo: int, hi: int) -> int:
    return max(lo, min(int(round(x)), hi))

# -----------------------------
# Defaults (evidence-based)
# -----------------------------

@dataclass(frozen=True)
class Defaults:
    # Candidate sets
    k_knn_S: int = 20
    k_knn_M: int = 30
    k_knn_L: int = 40
    k_knn_XL: int = 56
    k_knn_XXL: int = 64

    use_delaunay_small: bool = True      # ≤6k nodes
    use_delaunay_big: bool = False       # >6k: avoid RAM spikes

    # Zoning / partition
    zone_target_size: int = 600
    zone_overlap: float = 0.10           # 10%
    zone_cap: int = 256

    # Bees / agents
    bees_per_zone_S: int = 6
    bees_per_zone_M: int = 8
    bees_per_zone_L: int = 8
    bees_per_zone_XL: int = 8
    bees_per_zone_XXL: int = 8

    bee_ms_S: int = 150
    bee_ms_M: int = 200
    bee_ms_L: int = 250
    bee_ms_XL: int = 300
    bee_ms_XXL: int = 350

    # Local search (Mark 1: 2-opt primary)
    two_opt_ms_S: int = 150
    two_opt_ms_M: int = 200
    two_opt_ms_L: int = 300
    two_opt_ms_XL: int = 450
    two_opt_ms_XXL: int = 650

    # EAX integrator caps (hardened for large N)
    eax_max_ab_cycles_S: int = 16
    eax_max_ab_cycles_M: int = 32
    eax_max_ab_cycles_L: int = 64
    eax_max_ab_cycles_XL: int = 128
    eax_max_ab_cycles_XXL: int = 192

    eax_offspring_per_round: int = 2
    eax_parents_per_round: int = 6
    eax_repair_ms: int = 300

    # Initializer slots
    init_slots_S: int = 8
    init_slots_M: int = 10
    init_slots_L: int = 12
    init_slots_XL: int = 16
    init_slots_XXL: int = 20

    # Termination (ms)
    wall_ms_S: int = 15_000
    wall_ms_M: int = 60_000
    wall_ms_L: int = 180_000
    wall_ms_XL: int = 420_000
    wall_ms_XXL: int = 900_000

    # Restarts / pool
    max_restarts: int = 2
    keep_top_k_pool: int = 24
    unique_len_eps: float = 1e-6

D = Defaults()

# -----------------------------
# Candidate set policy
# -----------------------------

def _candidate_policy(n: int, metric: str) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    if tier == "S":
        k = D.k_knn_S
    elif tier == "M":
        k = D.k_knn_M
    elif tier == "L":
        k = D.k_knn_L
    elif tier == "XL":
        k = D.k_knn_XL
    else:
        k = D.k_knn_XXL

    use_delaunay = (n <= 6000 and D.use_delaunay_small) or (n > 6000 and D.use_delaunay_big)

    # Slightly higher K for ATT/CEIL_2D to combat rounding weirdness
    if metric in ("ATT", "CEIL_2D"):
        k = _safe_int(k * 1.05, 16, 96)

    policy = {
        # New fields (Mark 2 friendly)
        "type": "knn",
        "k": k,
        "augment_delaunay": bool(use_delaunay),
        "symmetric": True,
        "safeguards": {
            "max_total_edges": int(3.5 * n * k),
            "dedup_edges": True
        },
        # Back-compat for current solver:
        "use_knn": True,
        "use_delaunay": bool(use_delaunay),
        "use_alpha_near": False,
    }
    return policy

# -----------------------------
# Zones / partitioning
# -----------------------------

def _zones_policy(n: int, bbox: Tuple[float, float, float, float] | None) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    target = D.zone_target_size
    overlap = D.zone_overlap

    # XL/XXL nudge: smaller zones for locality
    if tier in ("XL", "XXL"):
        target = int(0.85 * target)

    est_zones = max(1, min(D.zone_cap, math.ceil(n / max(1, target))))

    return {
        # New
        "enabled": est_zones > 1,
        "est_count": est_zones,
        "target_size": target,
        "overlap": overlap,
        "shape": "grid",
        "bbox": bbox,
        # Back-compat (what solver.py currently reads)
        "method": "kmeans",                 # harmless placeholder; not used if zones disabled
        "target_zone_size": target,
        "overlap_pct": overlap,
        # "agents_per_zone" is plugged at assembly time from bees.per_zone
    }

# -----------------------------
# Bees / agents
# -----------------------------

def _bees_policy(n: int) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    if tier == "S":
        bees, ms = D.bees_per_zone_S, D.bee_ms_S
    elif tier == "M":
        bees, ms = D.bees_per_zone_M, D.bee_ms_M
    elif tier == "L":
        bees, ms = D.bees_per_zone_L, D.bee_ms_L
    elif tier == "XL":
        bees, ms = D.bees_per_zone_XL, D.bee_ms_XL
    else:
        bees, ms = D.bees_per_zone_XXL, D.bee_ms_XXL

    # Back-compat defaults (align with prior YAML fallback)
    scout_frac = 0.10 if tier in ("S", "M") else 0.12
    stagn_moves = 500 if tier in ("S", "M") else 1000
    levy_prob = 0.05 if tier in ("S", "M") else 0.08
    max_improve = 1200

    return {
        # New
        "per_zone": bees,
        "work_ms": ms,
        "parallel": True,
        "handoff_top_k": 2,
        "cap_total": max(16, bees * 64),
        # Back-compat
        "scout_fraction": scout_frac,
        "stagnation_moves": stagn_moves,
        "levy_jump_prob": levy_prob,
        "max_improving_moves": max_improve,
        "time_budget_s": ms / 1000.0,
    }

# -----------------------------
# Local Search policy
# -----------------------------

def _ls_policy(n: int, metric: str) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    if tier == "S":
        two_ms = D.two_opt_ms_S
    elif tier == "M":
        two_ms = D.two_opt_ms_M
    elif tier == "L":
        two_ms = D.two_opt_ms_L
    elif tier == "XL":
        two_ms = D.two_opt_ms_XL
    else:
        two_ms = D.two_opt_ms_XXL

    beam = 8 if metric in ("ATT", "CEIL_2D") else 6

    ls_new = {
        "two_opt": {"enabled": True, "time_ms": two_ms, "candidate_beam": beam, "accept_eps": 1e-12},
        "three_opt": {"enabled": False},
    }
    # Back-compat alias
    ls_old = {
        "use_lk": False,
        "max_k": 2,
        "double_bridge": True,
        "kick_period_moves": 500,
    }
    return {"_new": ls_new, "_old": ls_old}

# -----------------------------
# Integrator (EAX Mark 1)
# -----------------------------

def _integrator_policy(n: int, metric: str) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    if tier == "S":
        cap = D.eax_max_ab_cycles_S
    elif tier == "M":
        cap = D.eax_max_ab_cycles_M
    elif tier == "L":
        cap = D.eax_max_ab_cycles_L
    elif tier == "XL":
        cap = D.eax_max_ab_cycles_XL
    else:
        cap = D.eax_max_ab_cycles_XXL

    # Slightly tighter on huge ATT/CEIL_2D
    if metric in ("ATT", "CEIL_2D") and tier in ("XL", "XXL"):
        cap = int(cap * 0.9)

    return {
        # New
        "name": "EAX",
        "parents_per_round": D.eax_parents_per_round,
        "offspring_per_round": D.eax_offspring_per_round,
        "repair_time_ms": D.eax_repair_ms,
        "max_ab_cycles": cap,
        "guards": {"dedup_cycles": True, "cycle_cap": cap, "fallback_to_best_parent": True},
        # Back-compat
        "method": "eax",
    }

# -----------------------------
# Initializers (diversity)
# -----------------------------

def _initializers_policy(n: int, metric: str) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    if tier == "S":
        slots = D.init_slots_S
    elif tier == "M":
        slots = D.init_slots_M
    elif tier == "L":
        slots = D.init_slots_L
    elif tier == "XL":
        slots = D.init_slots_XL
    else:
        slots = D.init_slots_XXL

    if tier in ("L", "XL", "XXL"):
        mix = ["savings"] * (slots - 3) + ["nn", "nearest_insert", "nn"]
    else:
        base = ["nn", "nearest_insert", "savings", "nn"]
        mix = base + ["savings"] * max(0, slots - len(base))

    if metric in ("ATT", "CEIL_2D") and mix.count("nn") < 2:
        mix[-1] = "nn"

    return {"slots": slots, "methods": mix, "K": 6, "safeguards": {"max_total_inits": 64}}

# -----------------------------
# Termination / Restarts / Logging
# -----------------------------

def _termination_policy(n: int) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    wall = {
        "S": D.wall_ms_S,
        "M": D.wall_ms_M,
        "L": D.wall_ms_L,
        "XL": D.wall_ms_XL,
        "XXL": D.wall_ms_XXL,
    }[tier]
    return {
        # New
        "type": "wallclock_ms",
        "budget": wall,
        # Back-compat: many callers expect wall_time_s; run_large3 also overrides
        "wall_time_s": wall / 1000.0,
    }

def _restarts_policy() -> Dict[str, Any]:
    return {
        "max_restarts": D.max_restarts,
        "pool_keep_top_k": D.keep_top_k_pool,
        "unique_len_eps": D.unique_len_eps,
    }

def _logging_policy(n: int) -> Dict[str, Any]:
    return {
        "level": "INFO",
        "checkpoint_secs": 60 if n >= 5000 else 30,
        "print_parent_lengths": True,
        "print_cycle_stats": True,
    }

# -----------------------------
# Public entry: autoscale
# -----------------------------

def autoscale(instance_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a complete solver configuration from minimal instance metadata.
    Safe defaults if fields are missing. Single unified procedure (Mark 1).
    """
    name = (instance_meta.get("name") or "").strip()
    n = int(instance_meta.get("n") or 0)
    if n <= 0:
        raise ValueError("autoscale: instance size 'n' must be positive")

    metric = _norm_metric(instance_meta.get("metric"))
    bbox = instance_meta.get("bbox")

    candidate = _candidate_policy(n, metric)
    zones = _zones_policy(n, bbox)
    bees = _bees_policy(n)
    ls_both = _ls_policy(n, metric)
    integrator = _integrator_policy(n, metric)
    initializers = _initializers_policy(n, metric)
    termination = _termination_policy(n)
    restarts = _restarts_policy()
    logging = _logging_policy(n)

    # Assemble with back-compat mirrors
    solver = {
        "candidate": candidate,
        "zones": zones,
        "bees": bees,
        # New & old LS fields side-by-side
        "ls": ls_both["_new"],
        "local_search": ls_both["_old"],
        "integrator": integrator,
        "initializers": initializers,
        "termination": termination,
        "restarts": restarts,
        "logging": logging,
        "meta": {"name": name, "n": n, "metric": metric, "tier": _tier_by_n(n)},
        "safeguards": {
            "max_global_pool": 256,
            "max_memory_mb_hint": 8192 if n < 12000 else 16384 if n < 30000 else 24576,
            "deny_large_delaunay": (n > 6000),
        },
    }

    # Mirror a couple of legacy expectations:
    # - zones.agents_per_zone comes from bees.per_zone
    solver["zones"]["agents_per_zone"] = bees["per_zone"]

    return {"solver": solver}

# -----------------------------
# Initializer mix (back-compat for solver.py)
# -----------------------------

def normalize_weights(items: Iterable[tuple[str, float]]):
    total = sum(max(float(w), 0.0) for _, w in items) or 1.0
    return [(name, max(float(w), 0.0) / total) for name, w in items]

def choose_initializer_mix(n: int, metric_tag: str, adaptive: dict | None = None):
    """
    Back-compat wrapper expected by solver.py.
    Returns a weighted list of (initializer_name, weight) derived from _initializers_policy().
    """
    metric = _norm_metric(metric_tag)
    pol = _initializers_policy(n, metric)
    methods = pol.get("methods", [])
    if not methods:
        return [("nn", 1.0)]
    from collections import Counter
    cnt = Counter(methods)
    return normalize_weights(list(cnt.items()))

# -----------------------------
# Tiny self-test
# -----------------------------

if __name__ == "__main__":
    for meta in [
        {"name": "ch150", "n": 150, "metric": "EUC_2D"},
        {"name": "pr2392", "n": 2392, "metric": "ATT"},
        {"name": "pla33810", "n": 33810, "metric": "EUC_2D"},
        {"name": "pla33810_attish", "n": 33810, "metric": "CEIL_2D"},
    ]:
        cfg = autoscale(meta)["solver"]
        print(f"[AUTOSCALE] {meta['name']:>12s}  n={meta['n']:<6d}  tier={cfg['meta']['tier']:<3s}  "
              f"k={cfg['candidate']['k']:<3d} delaunay={cfg['candidate']['use_delaunay']!s:<5s}  "
              f"zones≈{cfg['zones']['est_count']:<3d}  EAX-cap={cfg['integrator']['max_ab_cycles']:<3d}  "
              f"inits={len(cfg['initializers']['methods'])}  wall={cfg['termination']['wall_time_s']:.0f}s")
