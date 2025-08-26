#!/usr/bin/env python3
"""
Test script to compare Bee-TSP solver performance against optimal tour files.
Tests att48 and ch130 instances with multiple seeds for statistical validity.
"""

import time
import json
import statistics
import argparse
from pathlib import Path
from bee_tsp import BeeTSPSolver
from bee_tsp.tsplib import load_tsplib

def parse_opt_tour(opt_tour_path):
    """Parse .opt.tour file to extract optimal tour and calculate its length."""
    tour = []
    optimal_length = None
    
    with open(opt_tour_path, 'r') as f:
        lines = f.readlines()
    
    # Extract optimal length from comment if available
    for line in lines:
        if line.startswith('COMMENT') and '(' in line and ')' in line:
            # Extract number from comment like "Optimum tour for kroA100 (21282)"
            try:
                comment = line.split('(')[1].split(')')[0]
                optimal_length = int(comment)
            except:
                pass
        elif 'Length' in line:
            # Extract from "Length 6110" format
            try:
                optimal_length = int(line.split('Length')[1].strip())
            except:
                pass
    
    # Parse tour section
    in_tour_section = False
    for line in lines:
        line = line.strip()
        if line == 'TOUR_SECTION':
            in_tour_section = True
            continue
        elif line == '-1' or line == 'EOF':
            break
        elif in_tour_section and line.isdigit():
            tour.append(int(line) - 1)  # Convert to 0-based indexing
    
    return tour, optimal_length

def calculate_tour_length(tour, dist_fn):
    """Calculate total tour length using distance function."""
    if not tour:
        return float('inf')
    
    total = 0
    n = len(tour)
    for i in range(n):
        total += dist_fn(tour[i], tour[(i+1) % n])
    return total

def run_solver_test(instance_name, tsp_path, opt_tour_path, seeds, time_budget=60.0, solver_config=None):
    """Run solver on instance with multiple seeds and compare to optimal."""
    
    print(f"\n=== Testing {instance_name} ===")
    
    # Parse optimal tour
    opt_tour, opt_length_from_file = parse_opt_tour(opt_tour_path)
    
    # Load instance to calculate actual optimal length
    ins = load_tsplib(str(tsp_path))
    dist_fn = ins['dist']
    
    # Calculate true optimal length from the tour
    true_opt_length = calculate_tour_length(opt_tour, dist_fn)
    
    print(f"Optimal tour length (calculated): {true_opt_length}")
    if opt_length_from_file:
        print(f"Optimal tour length (from file): {opt_length_from_file}")
    
    # Use provided config or create default
    if solver_config is None:
        solver_config = {
            'candidate': {'k': 30, 'use_knn': True},
            'zones': {'agents_per_zone': 4},
            'bees': {'time_budget_s': 1.2},
            'local_search': {'kick_period_moves': 400},
            'scouting': {
                'dynamic': {
                    'enabled': True,
                    'no_improve_s': 8.0,
                    'max_restarts_per_seed': 3,
                    'polish_time_s': 0.6
                }
            },
            'termination': {'wall_time_s': time_budget}
        }
    else:
        # Update termination time if not set
        solver_config['termination'] = solver_config.get('termination', {})
        solver_config['termination']['wall_time_s'] = time_budget
    
    solver = BeeTSPSolver(solver_config)
    
    # Run multiple seeds
    results = []
    for seed in seeds:
        print(f"  Running seed {seed}...", end=" ", flush=True)
        start_time = time.time()
        
        result = solver.solve(instance_name, str(tsp_path), seed=seed)
        runtime = time.time() - start_time
        
        best_length = result['best_length']
        gap = 100 * (best_length - true_opt_length) / true_opt_length
        
        results.append({
            'seed': seed,
            'length': best_length,
            'gap_pct': gap,
            'runtime': runtime,
            'improvements': len(result['anytime'])
        })
        
        print(f"Length: {best_length:.1f}, Gap: {gap:.2f}%, Time: {runtime:.1f}s")
    
    # Calculate statistics
    lengths = [r['length'] for r in results]
    gaps = [r['gap_pct'] for r in results]
    times = [r['runtime'] for r in results]
    
    stats = {
        'instance': instance_name,
        'optimal_length': true_opt_length,
        'num_runs': len(results),
        'best_found': min(lengths),
        'worst_found': max(lengths),
        'median_length': statistics.median(lengths),
        'mean_length': statistics.mean(lengths),
        'best_gap_pct': min(gaps),
        'worst_gap_pct': max(gaps),
        'median_gap_pct': statistics.median(gaps),
        'mean_gap_pct': statistics.mean(gaps),
        'median_runtime': statistics.median(times),
        'results': results
    }
    
    return stats

def main():
    """Run comprehensive test comparing solver to optimal solutions."""
    
    parser = argparse.ArgumentParser(description='Test Bee-TSP solver against optimal solutions')
    parser.add_argument('--time-budget', type=float, default=60.0,
                        help='Time budget per run in seconds (default: 60.0)')
    parser.add_argument('--seeds', type=int, nargs='*', default=[42, 123, 456, 789, 999],
                        help='Random seeds to test (default: 42 123 456 789 999)')
    parser.add_argument('--candidate-k', type=int, default=30,
                        help='Candidate graph k value (default: 30)')
    parser.add_argument('--agent-time', type=float, default=1.2,
                        help='Agent time budget in seconds (default: 1.2)')
    parser.add_argument('--scouts-stagnation', type=float, default=8.0,
                        help='Scout stagnation threshold in seconds (default: 8.0)')
    parser.add_argument('--max-restarts', type=int, default=3,
                        help='Max restarts per seed (default: 3)')
    parser.add_argument('--output', type=str, default='optimal_comparison_results.json',
                        help='Output file for results (default: optimal_comparison_results.json)')
    
    args = parser.parse_args()
    
    # Build solver configuration from CLI args
    solver_config = {
        'candidate': {'k': args.candidate_k, 'use_knn': True},
        'zones': {'agents_per_zone': 4},
        'bees': {'time_budget_s': args.agent_time},
        'local_search': {'kick_period_moves': 400},
        'scouting': {
            'dynamic': {
                'enabled': True,
                'no_improve_s': args.scouts_stagnation,
                'max_restarts_per_seed': args.max_restarts,
                'polish_time_s': 0.6
            }
        },
        'termination': {'wall_time_s': args.time_budget}
    }
    
    # Test instances (only those with both .tsp and .opt.tour files)
    test_instances = [
        ('att48', 'data/tsplib/att48.tsp', 'data/tsplib/att48.opt.tour'),
        ('ch130', 'data/tsplib/ch130.tsp', 'data/tsplib/ch130.opt.tour')
    ]
    
    print("Bee-TSP Solver vs Optimal Solutions Comparison")
    print("=" * 50)
    print(f"Configuration: time_budget={args.time_budget}s, k={args.candidate_k}, ")
    print(f"               agent_time={args.agent_time}s, scout_stagnation={args.scouts_stagnation}s, ")
    print(f"               max_restarts={args.max_restarts}, seeds={args.seeds}")
    
    all_results = []
    
    for instance_name, tsp_path, opt_tour_path in test_instances:
        if not Path(tsp_path).exists():
            print(f"Skipping {instance_name}: {tsp_path} not found")
            continue
        if not Path(opt_tour_path).exists():
            print(f"Skipping {instance_name}: {opt_tour_path} not found")
            continue
            
        stats = run_solver_test(instance_name, tsp_path, opt_tour_path, args.seeds, 
                                args.time_budget, solver_config)
        all_results.append(stats)
    
    # Print summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    print(f"{'Instance':<10} {'Optimal':<8} {'Best':<8} {'Median':<8} {'Worst':<8} {'Med Gap%':<8}")
    print("-" * 70)
    
    for stats in all_results:
        print(f"{stats['instance']:<10} "
              f"{stats['optimal_length']:<8.0f} "
              f"{stats['best_found']:<8.0f} "
              f"{stats['median_length']:<8.0f} "
              f"{stats['worst_found']:<8.0f} "
              f"{stats['median_gap_pct']:<8.2f}")
    
    # Save detailed results with configuration
    output_data = {
        'configuration': {
            'time_budget_s': args.time_budget,
            'seeds': args.seeds,
            'solver_config': solver_config
        },
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output}")
    
    # Performance analysis
    print("\nPERFORMANCE ANALYSIS:")
    print("-" * 30)
    for stats in all_results:
        print(f"\n{stats['instance']}:")
        print(f"  • Best gap: {stats['best_gap_pct']:.2f}% "
              f"(found optimal: {'YES' if stats['best_gap_pct'] < 0.01 else 'NO'})")
        print(f"  • Median gap: {stats['median_gap_pct']:.2f}%")
        print(f"  • Worst gap: {stats['worst_gap_pct']:.2f}%")
        print(f"  • Median runtime: {stats['median_runtime']:.1f}s")

if __name__ == '__main__':
    main()