#!/usr/bin/env python3
"""
One-shot parity check to verify opt.tour evaluation matches best_known.
Usage: python quick_check_parity.py data/tsplib/att48.tsp data/tsplib/att48.opt.tour
"""

from bee_tsp.tsplib import load_tsplib
from bee_tsp.metrics import build_eval_fn
import sys

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
                # Accept multiple ids per line
                for tok in line.split():
                    if tok == "-1":
                        in_tour = False
                        break
                    if tok.isdigit():
                        tour.append(int(tok) - 1)  # Convert 1-based to 0-based
    
    return tour

def main():
    if len(sys.argv) != 3:
        print("Usage: python quick_check_parity.py <tsp_file> <opt_tour_file>")
        sys.exit(1)
    
    tsp_file = sys.argv[1]
    opt_file = sys.argv[2]
    
    # Use same config as solver/harness
    cfg = {"metric": {"type": "tsplib", "integer": True}}
    
    # Load instance and build eval function
    ins = load_tsplib(tsp_file)
    eval_tour = build_eval_fn(cfg, ins)
    
    # Load optimal tour (1-based → 0-based)
    tour = load_opt_tour(opt_file)
    
    # Evaluate the optimal tour
    opt_len = eval_tour(tour)
    
    print(f"OPT_EVAL: {opt_len}")
    
    # Additional debug info
    print(f"Instance: {tsp_file}")
    print(f"Tour length: {len(tour)} cities")
    print(f"Expected n: {ins['n']}")
    print(f"Edge weight type: {ins.get('edge_weight_type', 'UNKNOWN')}")

if __name__ == "__main__":
    main()