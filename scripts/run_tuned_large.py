#!/usr/bin/env python3
"""
Run large instances with tuned parameters from large_tune.yaml
Uses the exact CLI format requested with proper time budgets.
"""
import sys, argparse, subprocess, datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_all_three_instances import run_solver_test, load_best_known
import json

def get_tuned_config(base_time_budget):
    """Get tuned configuration based on large_tune.yaml parameters"""
    return {
        'candidate': {
            'k': 25,  # Will be overridden per instance
            'use_knn': True,
            'use_delaunay': True  # From large_tune.yaml
        },
        'zones': {
            'agents_per_zone': 8  # From large_tune.yaml (increased from 4)
        },
        'bees': {
            'time_budget_s': 2.0  # From large_tune.yaml
        },
        'local_search': {
            'kick_period_moves': 400,
            'use_or_opt': True  # From large_tune.yaml (enabled)
        },
        'scouting': {
            'dynamic': {
                'enabled': True,
                'no_improve_s': 8.0,
                'max_restarts_per_seed': 3,
                'polish_time_s': 0.6
            }
        },
        'integrator': {
            'method': 'ucr'  # From large_tune.yaml (changed from popmusic)
        },
        'termination': {
            'wall_time_s': base_time_budget
        }
    }

def main():
    # Parse arguments matching the CLI format requested
    instances = ['pr2392', 'fnl4461', 'pla33810']
    seeds = [42, 123, 456]
    
    # Time budgets as requested: pr2392 @ 10min, fnl4461 @ 30min, pla33810 @ 60-90min  
    budgets = {
        'pr2392': 600,   # 10 minutes
        'fnl4461': 1800, # 30 minutes  
        'pla33810': 3600 # 60 minutes (could extend to 90 if needed)
    }
    
    # Instance-specific k values
    k_values = {
        'pr2392': 30,   # Medium scale, can afford higher k
        'fnl4461': 25,  # Large scale, moderate k
        'pla33810': 20  # Very large scale, lower k for performance
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"tuned_large_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Tuned Large Instance Run ===")
    print(f"Timestamp: {timestamp}")
    print(f"Results directory: {results_dir}")
    print(f"Configuration: 8 agents/zone, Or-opt enabled, UCR integration, Delaunay triangulation")
    print(f"Time budgets: pr2392=10min, fnl4461=30min, pla33810=60min")
    print(f"Seeds: {seeds}")
    print()
    
    best_known = load_best_known("best_known.csv")
    all_results = []
    
    for instance in instances:
        print(f"[START] {instance}")
        
        # Instance files  
        tsp_path = f"data/tsplib/{instance}.tsp"
        opt_tour_path = f"data/tsplib/{instance}.opt.tour"
        if not Path(opt_tour_path).exists():
            opt_tour_path = None
            
        # Get configuration for this instance
        config = get_tuned_config(budgets[instance])
        config['candidate']['k'] = k_values[instance]
        
        print(f"[CONFIG] time_budget={budgets[instance]}s, k={k_values[instance]}, agents_per_zone=8")
        print(f"[CONFIG] or_opt=True, ucr=True, delaunay=True")
        
        try:
            # Run the test
            stats = run_solver_test(
                instance, tsp_path, opt_tour_path,
                seeds, budgets[instance], config, best_known
            )
            
            if stats is None:
                print(f"[ERROR] {instance} test returned None")
                continue
                
            all_results.append(stats)
            
            print(f"[RESULT] {instance}: {stats['best_gap_pct']:.2f}% gap, {stats['median_runtime']:.1f}s runtime")
            print(f"[RESULT] Best: {stats['best_found']:.0f}, Optimal: {stats['optimal_length']:.0f}")
            print(f"[RESULT] Parity: {stats['parity']}")
            
        except Exception as e:
            print(f"[ERROR] {instance} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Save combined results
    output_data = {
        'configuration': {
            'timestamp': timestamp,
            'instances': instances,
            'seeds': seeds,
            'budgets': budgets,
            'tuned_params': {
                'agents_per_zone': 8,
                'or_opt_enabled': True,
                'ucr_integration': True,
                'delaunay_triangulation': True
            }
        },
        'results': all_results
    }
    
    results_file = results_dir / f"tuned_large_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"=== FINAL SUMMARY ===")
    for stats in all_results:
        print(f"{stats['instance']:>10} (n={stats['n']:>5}): {stats['best_gap_pct']:>6.2f}% gap, "
              f"{stats['median_runtime']:>6.1f}s, parity={stats['parity']}")
    
    print(f"\nResults saved to: {results_file}")
    return len(all_results)

if __name__ == "__main__":
    completed = main()
    print(f"\nCompleted {completed}/3 instances successfully.")