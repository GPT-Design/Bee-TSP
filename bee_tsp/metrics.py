from __future__ import annotations
from typing import Any, Dict, Callable, List
import math

def build_cost_fn(cfg: Dict[str, Any], ins: Dict[str, Any]) -> Callable[[int, int], float]:
    """
    Build a cost function based on config and instance data.
    Default behavior: use TSPLIB's built-in distance function (unchanged behavior).
    """
    metric_cfg = cfg.get("metric", {})
    metric_type = metric_cfg.get("type", "tsplib")
    
    if metric_type == "tsplib" or metric_type is None:
        # Default: use the TSPLIB distance function as-is
        return ins["dist"]
    
    # Future extension point for other metric types
    # elif metric_type == "euclidean_rounded":
    #     coords = ins["coords"]
    #     def euclidean_rounded(i: int, j: int) -> float:
    #         xi, yi = coords[i]; xj, yj = coords[j]
    #         return round(math.hypot(xi - xj, yi - yj))
    #     return euclidean_rounded
    
    # Fallback to TSPLIB
    return ins["dist"]

def wrap_dist(cost_fn: Callable[[int, int], float], integer: bool = True) -> Callable[[int, int], int]:
    """
    Wrap a cost function to return integers if needed.
    This maintains backward compatibility with existing code expecting int distances.
    """
    if integer:
        def int_dist(i: int, j: int) -> int:
            return int(cost_fn(i, j))
        return int_dist
    else:
        # Keep as float but cast to int for compatibility
        def float_dist(i: int, j: int) -> int:
            return int(cost_fn(i, j))
        return float_dist

def build_eval_fn(cfg: Dict[str, Any], ins: Dict[str, Any]) -> Callable[[List[int]], int]:
    """
    Returns eval_tour(tour) that sums edge costs using the same dist + integer
    rounding as the solver. 'tour' is a node index sequence (0-based or your internal).
    """
    cost = build_cost_fn(cfg, ins)
    integer = bool((cfg.get("metric") or {}).get("integer", True))
    dist = wrap_dist(cost, integer=integer)

    def eval_tour(tour: List[int]) -> int:
        s = 0
        n = len(tour)
        for i in range(n):
            s += dist(tour[i], tour[(i+1) % n])
        return int(s)
    return eval_tour