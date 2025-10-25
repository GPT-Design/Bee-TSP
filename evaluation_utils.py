#!/usr/bin/env python3
"""
Evaluation Utilities for TSPLIB Parity Checking
Per Update_Claude_12.txt - ensures consistent evaluation paths
"""

from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from bee_tsp.tsplib import load_tsplib
from bee_tsp.metrics import build_cost_fn, wrap_dist


def load_optimal_tour(opt_tour_path: str) -> List[int]:
    """Load optimal tour from TSPLIB .opt.tour file."""
    if not Path(opt_tour_path).exists():
        return []
    
    tour = []
    with open(opt_tour_path, 'r') as f:
        lines = f.readlines()
    
    tour_section = False
    for line in lines:
        line = line.strip()
        if line == 'TOUR_SECTION':
            tour_section = True
            continue
        elif line == 'EOF' or line == '-1':
            break
        
        if tour_section and line.isdigit():
            tour.append(int(line) - 1)  # Convert to 0-based indexing
    
    return tour


def evaluate_tour_tsplib(tour: List[int], instance_data: Dict[str, Any], config: Dict[str, Any]) -> int:
    """
    Evaluate a tour using the exact TSPLIB evaluation path.
    Returns integer tour length using the same method as single-hive solver.
    """
    if not tour or len(tour) < 2:
        return float('inf')
    
    # Use same evaluation path as BeeTSPSolver
    cost_fn = build_cost_fn(config, instance_data)
    integer = bool(config.get("metric", {}).get("integer", True))
    dist = wrap_dist(cost_fn, integer=integer)
    
    # Calculate tour length
    total_length = 0
    n = len(tour)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        total_length += dist(u, v)
    
    return int(total_length) if integer else total_length


def validate_parity_triad(instance_name: str, instance_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parity triad per Update_Claude_12.txt:
    - opt_eval_tsplib: Re-evaluate optimal tour with current eval path
    - metric_tag: Instance edge weight type
    - Returns validation results
    """
    # Load instance
    instance_data = load_tsplib(instance_path)
    metric_tag = instance_data.get('edge_weight_type', 'UNKNOWN')
    
    # Try to load optimal tour
    opt_tour_path = instance_path.replace('.tsp', '.opt.tour')
    optimal_tour = load_optimal_tour(opt_tour_path)
    
    result = {
        'instance_name': instance_name,
        'metric_tag': metric_tag,
        'has_opt_tour': len(optimal_tour) > 0,
        'opt_tour_length': len(optimal_tour) if optimal_tour else 0
    }
    
    if optimal_tour:
        # Evaluate optimal tour with TSPLIB path
        opt_eval_tsplib = evaluate_tour_tsplib(optimal_tour, instance_data, config)
        result['opt_eval_tsplib'] = opt_eval_tsplib
        
        # Known best values for validation
        known_best = {
            'pr2392': 378032,  # Known optimal for pr2392
            'fnl4461': 182566,  # Known best-known for fnl4461
            'pla33810': 66048945  # Known optimal for pla33810
        }
        
        if instance_name in known_best:
            expected = known_best[instance_name]
            result['expected_best'] = expected
            if optimal_tour:
                # Only do parity check if we have an optimal tour to evaluate
                result['parity_check'] = abs(opt_eval_tsplib - expected) <= 1  # Allow 1 unit tolerance
                result['parity_error'] = abs(opt_eval_tsplib - expected)
            else:
                # For instances without optimal tour, record best-known for reference
                result['parity_check'] = 'N/A (no optimal tour)'
                result['parity_error'] = 'N/A'
    
    return result


def create_tsplib_evaluator(instance_data: Dict[str, Any], config: Dict[str, Any]):
    """Create a TSPLIB-compatible distance function."""
    cost_fn = build_cost_fn(config, instance_data)
    integer = bool(config.get("metric", {}).get("integer", True))
    dist = wrap_dist(cost_fn, integer=integer)
    return dist


# Test function
if __name__ == "__main__":
    import yaml
    
    # Test with pr2392
    config = {
        'metric': {'integer': True},
        'candidate': {'k': 28}
    }
    
    result = validate_parity_triad('pr2392', 'data/tsplib/pr2392.tsp', config)
    print("Parity validation for pr2392:")
    for key, value in result.items():
        print(f"  {key}: {value}")