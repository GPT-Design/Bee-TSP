#!/usr/bin/env python3
"""
Debug Or-opt delta calculation to isolate the bug.
"""

from bee_tsp.tsplib import load_tsplib
from bee_tsp.metrics import build_cost_fn, wrap_dist
from bee_tsp.solver import or_opt_delta, apply_or_opt_and_update_pos, tour_length

def test_oropt_delta():
    # Load a simple instance
    ins = load_tsplib("data/tsplib/att48.tsp")
    cost = build_cost_fn({"metric": {"type": "tsplib", "integer": True}}, ins)
    dist = wrap_dist(cost, integer=True)
    
    # Simple test tour: 0, 1, 2, 3, 4, 5, 6, 7, ...
    tour = list(range(20))  # Use first 20 cities for simplicity
    n = len(tour)
    
    print("=== OR-OPT DEBUG ===")
    print(f"Initial tour: {tour[:10]}...")
    
    initial_length = tour_length(tour, dist)
    print(f"Initial tour length: {initial_length}")
    
    # Test a simple Or-opt move: move segment [2] after position 5
    i, length, j = 2, 1, 5
    
    print(f"\nTesting Or-opt move: move segment [{i}:{i+length}] after position {j}")
    print(f"Moving city {tour[i]} after city {tour[j]}")
    
    # Calculate delta
    delta = or_opt_delta(tour, i, length, j, dist)
    print(f"Calculated delta: {delta}")
    
    # Apply the move manually and verify
    tour_copy = tour[:]
    pos = list(range(n))
    for idx, city in enumerate(tour_copy):
        pos[city] = idx
    
    apply_or_opt_and_update_pos(tour_copy, pos, i, length, j)
    new_length = tour_length(tour_copy, dist)
    actual_delta = new_length - initial_length
    
    print(f"After move tour: {tour_copy[:10]}...")
    print(f"New tour length: {new_length}")
    print(f"Actual delta: {actual_delta}")
    
    print(f"\nComparison:")
    print(f"Calculated delta: {delta}")
    print(f"Actual delta: {actual_delta}")
    print(f"Match: {delta == actual_delta}")
    
    if delta != actual_delta:
        print("❌ OR-OPT DELTA CALCULATION IS WRONG!")
        print(f"Error: {delta - actual_delta}")
    else:
        print("✅ Or-opt delta calculation is correct")

if __name__ == "__main__":
    test_oropt_delta()