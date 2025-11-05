#!/usr/bin/env python3
"""
bee_tsp.adaptive — Mark 1 minimal adaptive hooks (no heavy deps, pure Python)

Provides:
  - analyze_instance_structure(coords, dist, n) -> dict
  - adaptive_delta_phi_config(metrics) -> dict
  - size_adaptive_params(n) -> dict

These are intentionally lightweight. They give conservative hints that
won’t destabilize runs and are safe to ignore if not enabled.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

Number = float | int

def _bbox(coords: List[Tuple[Number, Number]]) -> tuple[float, float, float, float]:
    if not coords:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [float(x) for x, _ in coords]
    ys = [float(y) for _, y in coords]
    return (min(xs), min(ys), max(xs), max(ys))

def _safe_area(xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    return max(1e-12, w * h)

def analyze_instance_structure(coords: List[Tuple[Number, Number]],
                               dist,
                               n: int) -> Dict[str, Any]:
    """
    Very cheap geometry scan:
      - bbox, area, density ≈ n / area
      - anisotropy in [0,1]: 0 == square-ish, 1 == very elongated
      - coarse “stripe-ish”/“cluster-ish” flags (extremely heuristic)
    """
    xmin, ymin, xmax, ymax = _bbox(coords)
    area = _safe_area(xmin, ymin, xmax, ymax)
    w, h = max(0.0, xmax - xmin), max(0.0, ymax - ymin)

    # anisotropy: 0 (square) .. 1 (extremely elongated)
    if max(w, h) > 0:
        anisotropy = (max(w, h) - min(w, h)) / max(w, h)
    else:
        anisotropy = 0.0

    density = float(n) / area

    # coarse flags (do not impact solver unless you choose to)
    stripe_like = anisotropy > 0.75 and n > 2000
    cluster_like = (0.25 < anisotropy < 0.6) and n > 1000

    return {
        "n": int(n),
        "bbox": (xmin, ymin, xmax, ymax),
        "area": area,
        "width": w,
        "height": h,
        "anisotropy": anisotropy,          # 0..1
        "density": density,                # pts per area
        "stripe_like": bool(stripe_like),
        "cluster_like": bool(cluster_like),
    }

def adaptive_delta_phi_config(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a tiny ΔΦ tunable block. Mark 1 keeps this inert unless you
    wire ΔΦ into your pipeline. Safe, conservative values only.
    """
    aniso = float(metrics.get("anisotropy", 0.0))
    # Slightly larger angular bins if highly elongated
    phi_bins = 36 if aniso < 0.5 else 24
    return {
        "enabled": True,
        "phi_bins": phi_bins,          # if ignored elsewhere, no effect
        "radial_bins": 16,             # conservative default
        "min_sector_size": 8,
        "max_sector_size": 256
    }

def size_adaptive_params(n: int) -> Dict[str, Any]:
    """
    Light size-based hints you may choose to merge into cfg.
    They mirror what we already do in auto_params, but kept here so
    future Mark 2 analyzers can centralize policies.
    """
    if n < 300:
        tier = "S"
        k_boost = 0
        two_opt_ms = 150
    elif n < 1200:
        tier = "M"
        k_boost = 0
        two_opt_ms = 200
    elif n < 5000:
        tier = "L"
        k_boost = 0
        two_opt_ms = 300
    elif n < 15000:
        tier = "XL"
        k_boost = 4
        two_opt_ms = 450
    else:
        tier = "XXL"
        k_boost = 8
        two_opt_ms = 650

    return {
        "tier": tier,
        "candidate_k_boost": k_boost,   # add to KNN-k if you want
        "two_opt_time_ms_hint": two_opt_ms,
    }

__all__ = [
    "analyze_instance_structure",
    "adaptive_delta_phi_config",
    "size_adaptive_params",
]
