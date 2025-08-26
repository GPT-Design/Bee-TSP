#!/usr/bin/env python3
"""
Debug script to isolate the metric mismatch between solver and harness.
"""

from bee_tsp.tsplib import load_tsplib
from bee_tsp.metrics import build_cost_fn, wrap_dist, build_eval_fn
from bee_tsp import BeeTSPSolver

def load_opt_tour(opt_tour_path):
    """Load .opt.tour file and convert 1-based to 0-based indexing."""
    tour = []
    in_tour = False
    
    with open(opt_tour_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
                
            if line.upper().startswith("TOUR_SECTION"):
                in_tour = True
                continue
            if in_tour:
                if line == "-1" or line.upper() == "EOF":
                    break
                for tok in line.split():
                    if tok == "-1":
                        in_tour = False
                        break
                    if tok.isdigit():
                        tour.append(int(tok) - 1)  # Convert 1-based to 0-based
    return tour

def solver_tour_length(tour, dist):
    """Replicate solver's tour_length function exactly."""
    s = 0
    n = len(tour)
    for i in range(n):
        s += dist(tour[i], tour[(i+1)%n])
    return s

def main():
    # Test on att48
    tsp_path = "data/tsplib/att48.tsp"
    opt_path = "data/tsplib/att48.opt.tour"
    
    print("=== DEBUGGING METRIC MISMATCH ===")
    
    # Load instance and optimal tour
    ins = load_tsplib(tsp_path)
    opt_tour = load_opt_tour(opt_path)
    
    print(f"Instance: {tsp_path}")
    print(f"Edge weight type: {ins.get('edge_weight_type', 'UNKNOWN')}")
    print(f"Tour length: {len(opt_tour)}")
    
    # Test 1: Raw TSPLIB distance function
    raw_dist = ins['dist']
    raw_length = solver_tour_length(opt_tour, raw_dist)
    print(f"\n1. Raw TSPLIB dist: {raw_length}")
    
    # Test 2: Solver's metric setup (replicate exactly)
    solver_cfg = {
        "metric": {
            "type": "tsplib", 
            "integer": True,
            "edge_weight_type": ins.get('edge_weight_type', 'UNKNOWN')
        }
    }
    solver_cost = build_cost_fn(solver_cfg, ins)
    solver_dist = wrap_dist(solver_cost, integer=True)
    solver_length = solver_tour_length(opt_tour, solver_dist)
    print(f"2. Solver build_cost_fn + wrap_dist: {solver_length}")
    
    # Test 3: Harness eval_tour function  
    harness_eval = build_eval_fn(solver_cfg, ins)
    harness_length = harness_eval(opt_tour)
    print(f"3. Harness build_eval_fn: {harness_length}")
    
    # Test 4: Direct comparison of individual edges
    print(f"\n=== EDGE BY EDGE COMPARISON (first 5 edges) ===")
    for i in range(min(5, len(opt_tour))):
        u, v = opt_tour[i], opt_tour[(i+1) % len(opt_tour)]
        raw_edge = raw_dist(u, v)
        solver_edge = solver_dist(u, v)
        print(f"Edge {i}: ({u},{v}) -> raw={raw_edge}, solver={solver_edge}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Expected (best_known): 10628")
    print(f"Raw TSPLIB: {raw_length}")
    print(f"Solver method: {solver_length}")
    print(f"Harness method: {harness_length}")
    
    if raw_length != solver_length:
        print("❌ MISMATCH: Raw vs Solver distance functions differ!")
    if solver_length != harness_length:
        print("❌ MISMATCH: Solver vs Harness evaluation differs!")
    if harness_length == 10628:
        print("✅ Harness evaluation is correct")
    else:
        print("❌ Harness evaluation is wrong")

if __name__ == "__main__":
    main()