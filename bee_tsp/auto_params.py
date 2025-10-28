"""
auto_params.py — Bee-TSP Mark 1 (Publication build)
Unified auto-configuration with strong safeguards, metric awareness, and clear seams
for future integrators/analysers (Mark 2).

Design goals:
- Robust defaults for 2D TSPLIB-style instances (EUC_2D, CEIL_2D/ATT), but metric-agnostic where feasible.
- Scales from small N (~100) to very large N (>= 33k, e.g., pla33810) without memory blowups.
- Unified, evidence-informed procedure; no duplicated v1/v2 branches.
- Strict caps and “smell-test” style guards to prevent pathological settings.
- Modular seams: plug a richer Analyser and new Integrators in Mark 2 without changing callers.

Public API:
    - autoscale(instance_meta: dict) -> dict   # returns a full solver config

Expected solver consumer keys (stable surface):
    solver: {
        "candidate": {...},       # candidate set policy
        "zones": {...},           # spatial zoning / partition
        "bees": {...},            # agent swarm params
        "ls": {...},              # local search params (2-opt/3-opt caps)
        "integrator": {...},      # integrator selection + params (EAX Mark 1)
        "initializers": {...},    # tour initialisation mix
        "termination": {...},     # wallclock / iterations
        "restarts": {...},        # restart policy
        "logging": {...},         # verbosity + checkpoints
    }

instance_meta minimal fields (best effort; missing fields are handled):
    {
      "name": "pr2392",
      "n": 2392,
      "metric": "EUC_2D" | "CEIL_2D" | "ATT" | "GEO" | "CUSTOM" | None,
      # Optional (Mark 1 uses lightly; Mark 2 can deepen this):
      "bbox": (xmin, ymin, xmax, ymax) or None,
      "density_hint": float|None,        # points per area unit
      "anisotropy_hint": float|None,     # 0..1 (elongation)
      "clusters_hint": int|None,         # k-means-ish count
    }
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math


# -----------------------------
# Helpers & Normalization
# -----------------------------

def _norm_metric(metric: str | None) -> str:
    if not metric:
        return "UNKNOWN"
    m = metric.strip().upper()
    # Common aliases
    if m in ("EUC", "EUC2D", "EUC_2D"):
        return "EUC_2D"
    if m in ("CEIL_2D", "ATT", "EUC_2D_CEIL"):
        return "ATT"   # TSPLIB's CEIL_2D behaves like ATT for rounding contexts
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
# Default knobs (evidence-based)
# -----------------------------

@dataclass(frozen=True)
class Defaults:
    # Candidate sets
    k_knn_S: int = 20
    k_knn_M: int = 30
    k_knn_L: int = 40
    k_knn_XL: int = 56
    k_knn_XXL: int = 64

    use_delaunay_small: bool = True     # ≤6k: allow Delaunay augmentation
    use_delaunay_big: bool = False      # >6k: generally disable to save memory

    # Zoning / partition
    zone_target_size: int = 900         # target nodes per zone
    zone_overlap: float = 0.10          # 10% overlap to avoid cut artifacts
    zone_cap: int = 256                 # upper bound on number of zones

    # Bees / agents
    bees_per_zone_S: int = 6
    bees_per_zone_M: int = 8
    bees_per_zone_L: int = 10
    bees_per_zone_XL: int = 12
    bees_per_zone_XXL: int = 16

    bee_ms_S: int = 150                 # local budget per agent (ms)
    bee_ms_M: int = 200
    bee_ms_L: int = 250
    bee_ms_XL: int = 300
    bee_ms_XXL: int = 350

    # Local search (2-opt primary in Mark 1)
    two_opt_ms_S: int = 150
    two_opt_ms_M: int = 200
    two_opt_ms_L: int = 300
    two_opt_ms_XL: int = 450
    two_opt_ms_XXL: int = 650

    # EAX integrator caps (Mark 1 hardened)
    eax_max_ab_cycles_S: int = 16
    eax_max_ab_cycles_M: int = 32
    eax_max_ab_cycles_L: int = 64
    eax_max_ab_cycles_XL: int = 128
    eax_max_ab_cycles_XXL: int = 192   # conservative for 33k

    eax_offspring_per_round: int = 2
    eax_parents_per_round: int = 6
    eax_repair_ms: int = 250

    # Initializer mix slots (total # initial tours)
    init_slots_S: int = 8
    init_slots_M: int = 10
    init_slots_L: int = 12
    init_slots_XL: int = 16
    init_slots_XXL: int = 20

    # Termination / restarts
    wall_ms_S: int = 15_000
    wall_ms_M: int = 60_000
    wall_ms_L: int = 180_000
    wall_ms_XL: int = 420_000
    wall_ms_XXL: int = 900_000

    max_restarts: int = 2
    keep_top_k_pool: int = 24           # global tour pool cap to avoid RAM creep
    unique_len_eps: float = 1e-6        # dedup epsilon


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

    # Delaunay augmentation: valuable for small/mid Euclidean; risky for huge N
    use_delaunay = (n <= 6000 and D.use_delaunay_small) or (n > 6000 and D.use_delaunay_big)

    # ATT/CEIL_2D often benefits from slightly higher k due to rounding artifacts
    if metric in ("ATT",):
        k = _safe_int(k * 1.05, 16, 96)

    return {
        "type": "knn",
        "k": k,
        "augment_delaunay": bool(use_delaunay),
        "symmetric": True,
        "safeguards": {
            "max_total_edges": int(3.5 * n * k),   # hard RAM guard
            "dedup_edges": True
        }
    }


# -----------------------------
# Zones / partitioning
# -----------------------------

def _zones_policy(n: int, bbox: Tuple[float, float, float, float] | None) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    target = D.zone_target_size
    overlap = D.zone_overlap

    # XL/XXL: shrink target a bit to improve locality
    if tier in ("XL", "XXL"):
        target = int(0.85 * target)

    est_zones = max(1, min(D.zone_cap, math.ceil(n / max(1, target))))

    return {
        "enabled": est_zones > 1,
        "est_count": est_zones,
        "target_size": target,
        "overlap": overlap,
        "shape": "grid",             # seam for Mark 2 (k-d tree / adaptive)
        "bbox": bbox
    }


# -----------------------------
# Bees / agents
# -----------------------------

def _bees_policy(n: int) -> Dict[str, Any]:
    tier = _tier_by_n(n)
    if tier == "S":
        bees = D.bees_per_zone_S; ms = D.bee_ms_S
    elif tier == "M":
        bees = D.bees_per_zone_M; ms = D.bee_ms_M
    elif tier == "L":
        bees = D.bees_per_zone_L; ms = D.bee_ms_L
    elif tier == "XL":
        bees = D.bees_per_zone_XL; ms = D.bee_ms_XL
    else:
        bees = D.bees_per_zone_XXL; ms = D.bee_ms_XXL

    return {
        "per_zone": bees,
        "work_ms": ms,
        "parallel": True,
        "handoff_top_k": 2,        # hand best zone tours to integrator
        "cap_total": max(16, bees * 64)  # global safeguard
    }


# -----------------------------
# Local Search policy (Mark 1)
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

    # ATT rounding → allow small beam widen
    beam = 8 if metric == "ATT" else 6

    return {
        "two_opt": {
            "enabled": True,
            "time_ms": two_ms,
            "candidate_beam": beam,
            "accept_eps": 1e-12
        },
        "three_opt": {
            "enabled": False   # Mark 1: keep simple & predictable
        }
    }


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

    # Slightly tighter cap for extremely large ATT (rounding noise)
    if metric == "ATT" and tier in ("XL", "XXL"):
        cap = int(cap * 0.9)

    return {
        "name": "EAX",
        "parents_per_round": D.eax_parents_per_round,
        "offspring_per_round": D.eax_offspring_per_round,
        "repair_time_ms": D.eax_repair_ms,
        "max_ab_cycles": cap,
        # Hard safety rails expected by our EAX implementation:
        "guards": {
            "dedup_cycles": True,
            "cycle_cap": cap,
            "fallback_to_best_parent": True
        }
    }


# -----------------------------
# Initializers (diversity, large-N safe)
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

    # Evidence: Savings shines at large N; NN/NearInsert seed diversity is still useful.
    if tier in ("L", "XL", "XXL"):
        mix = ["savings"] * (slots - 3) + ["nn", "nearest_insert", "nn"]
    else:
        mix = ["nn", "nearest_insert", "savings", "nn"] + ["savings"] * (slots - 4)

    # ATT quirk: keep at least two NN seeds to reduce rounding traps
    if metric == "ATT" and mix.count("nn") < 2:
        mix[-1] = "nn"

    return {
        "slots": slots,
        "methods": mix,
        "K": 6,  # pool cap during init per method
        "safeguards": {"max_total_inits": 64}
    }


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
    return {"type": "wallclock_ms", "budget": wall}


def _restarts_policy() -> Dict[str, Any]:
    return {
        "max_restarts": D.max_restarts,
        "pool_keep_top_k": D.keep_top_k_pool,
        "unique_len_eps": D.unique_len_eps
    }


def _logging_policy(n: int) -> Dict[str, Any]:
    return {
        "level": "INFO",
        "checkpoint_secs": 60 if n >= 5000 else 30,
        "print_parent_lengths": True,
        "print_cycle_stats": True
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

    # Compose policies
    candidate = _candidate_policy(n, metric)
    zones = _zones_policy(n, bbox)
    bees = _bees_policy(n)
    ls = _ls_policy(n, metric)
    integrator = _integrator_policy(n, metric)
    initializers = _initializers_policy(n, metric)
    termination = _termination_policy(n)
    restarts = _restarts_policy()
    logging = _logging_policy(n)

    # Final assembly with global guards
    solver = {
        "candidate": candidate,
        "zones": zones,
        "bees": bees,
        "ls": ls,
        "integrator": integrator,
        "initializers": initializers,
        "termination": termination,
        "restarts": restarts,
        "logging": logging,
        "meta": {
            "name": name,
            "n": n,
            "metric": metric,
            "tier": _tier_by_n(n),
        },
        "safeguards": {
            # Hard ceilings; solver/integrator must honor these
            "max_global_pool": 256,
            "max_memory_mb_hint": 8192 if n < 12000 else 16384 if n < 30000 else 24576,
            "deny_large_delaunay": (n > 6000),
        }
    }

    return {"solver": solver}


# -----------------------------
# Optional: tiny self-test
# -----------------------------

if __name__ == "__main__":
    for meta in [
        {"name": "ch150", "n": 150, "metric": "EUC_2D"},
        {"name": "pr2392", "n": 2392, "metric": "ATT"},
        {"name": "pla33810", "n": 33810, "metric": "EUC_2D"},
    ]:
        cfg = autoscale(meta)
        s = cfg["solver"]
        print(f"[AUTOSCALE] {meta['name']:>8s} n={meta['n']:<6d} "
              f"tier={s['meta']['tier']:<3s}  "
              f"k={s['candidate']['k']:<3d} delaunay={s['candidate']['augment_delaunay']}  "
              f"zones≈{s['zones']['est_count']:<3d}  "
              f"EAX-cap={s['integrator']['max_ab_cycles']:<3d}  "
              f"inits={len(s['initializers']['methods'])}")
