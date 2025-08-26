#!/usr/bin/env python3
"""
Single instance runner for large tests using size-appropriate parameters.
"""
import sys, argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_all_three_instances import run_solver_test, load_best_known
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run single large instance test')
    parser.add_argument('--instance', required=True, choices=['pr2392', 'fnl4461', 'pla33810'])
    parser.add_argument('--time-budget', type=float, default=120.0)
    parser.add_argument('--candidate-k', type=int, default=25)
    parser.add_argument('--agent-time', type=float, default=8.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    instance_map = {
        'pr2392': ('pr2392', 'data/tsplib/pr2392.tsp', 'data/tsplib/pr2392.opt.tour'),
        'fnl4461': ('fnl4461', 'data/tsplib/fnl4461.tsp', None),
        'pla33810': ('pla33810', 'data/tsplib/pla33810.tsp', 'data/tsplib/pla33810.opt.tour')
    }
    
    instance_name, tsp_path, opt_tour_path = instance_map[args.instance]
    
    # Build solver configuration with size-appropriate parameters
    solver_config = {
        'candidate': {'k': args.candidate_k, 'use_knn': True},
        'zones': {'agents_per_zone': 4},
        'bees': {'time_budget_s': args.agent_time},
        'local_search': {'kick_period_moves': 400, 'use_or_opt': False},
        'scouting': {
            'dynamic': {
                'enabled': True,
                'no_improve_s': 8.0,
                'max_restarts_per_seed': 3,
                'polish_time_s': 0.6
            }
        },
        'termination': {'wall_time_s': args.time_budget}
    }
    
    # Load best known values
    best_known = load_best_known("best_known.csv")
    
    print(f"=== Single Large Instance Test: {instance_name} ===")
    print(f"Configuration: time_budget={args.time_budget}s, k={args.candidate_k}, agent_time={args.agent_time}s")
    print(f"Seeds: {args.seeds}")
    
    # Run the test
    stats = run_solver_test(
        instance_name, tsp_path, opt_tour_path, 
        args.seeds, args.time_budget, solver_config, best_known
    )
    
    if stats is None:
        print("Test failed!")
        return 1
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%y-%m-%d_%H%M")
        args.output = f"results/{timestamp}_single_{instance_name}.json"
    
    # Save results
    output_data = {
        'configuration': {
            'instance': instance_name,
            'time_budget_s': args.time_budget,
            'seeds': args.seeds,
            'solver_config': solver_config,
            'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': [stats]
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Instance: {stats['instance']} (n={stats['n']})")
    print(f"Best gap: {stats['best_gap_pct']:.2f}%")
    print(f"Best found: {stats['best_found']:.0f}")
    print(f"Optimal/Best known: {stats['optimal_length']:.0f}")
    print(f"Runtime: {stats['median_runtime']:.1f}s")
    print(f"Parity: {stats['parity']}")
    print(f"\nResults saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())