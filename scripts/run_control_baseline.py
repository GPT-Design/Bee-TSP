#!/usr/bin/env python3
"""
Control/baseline runs for large instances with conservative parameters:
- agents=4, Or-opt=off, integrator=popmusic, modest k, workers=1
"""
import sys, datetime, json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_all_three_instances import run_solver_test, load_best_known

def get_baseline_config(time_budget, k_value):
    """Get conservative baseline configuration"""
    return {
        'candidate': {
            'k': k_value,
            'use_knn': True
            # No use_delaunay (baseline)
        },
        'zones': {
            'agents_per_zone': 4  # Baseline conservative
        },
        'bees': {
            'time_budget_s': min(time_budget * 0.15, 12.0)  # Modest agent time
        },
        'local_search': {
            'kick_period_moves': 400,
            'use_or_opt': False  # Explicitly disabled for baseline
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
            'method': 'popmusic'  # Baseline (not UCR)
        },
        'termination': {
            'wall_time_s': time_budget
        }
    }

def run_control_experiment(tag, instances, seeds, budgets, k_values, description):
    """Run control experiment with given parameters"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"control_{tag}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Control Run: {tag} ===")
    print(f"Description: {description}")
    print(f"Timestamp: {timestamp}")
    print(f"Configuration: agents=4, or_opt=False, integrator=popmusic, modest k")
    print(f"Workers: 1 (sequential execution)")
    print(f"Results directory: {results_dir}")
    print(f"Instances: {instances}")
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
        config = get_baseline_config(budgets[instance], k_values[instance])
        
        print(f"[CONFIG] time_budget={budgets[instance]}s, k={k_values[instance]}")
        print(f"[CONFIG] agents=4, or_opt=False, integrator=popmusic")
        print(f"[CONFIG] agent_time={config['bees']['time_budget_s']:.1f}s")
        
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
            
            print(f"[RESULT] {instance}: {stats['best_gap_pct']:.2f}% gap")
            print(f"[RESULT] Best: {stats['best_found']:.0f}, Target: {stats['optimal_length']:.0f}")
            print(f"[RESULT] Runtime: {stats['median_runtime']:.1f}s, Parity: {stats['parity']}")
            
        except Exception as e:
            print(f"[ERROR] {instance} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Save results
    output_data = {
        'experiment': {
            'tag': tag,
            'description': description,
            'timestamp': timestamp,
            'configuration': 'baseline_control'
        },
        'parameters': {
            'agents_per_zone': 4,
            'or_opt_enabled': False,
            'integrator_method': 'popmusic',
            'workers': 1,
            'budgets': budgets,
            'k_values': k_values
        },
        'instances': instances,
        'seeds': seeds,
        'results': all_results
    }
    
    results_file = results_dir / f"control_{tag}_results.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"=== CONTROL SUMMARY: {tag} ===")
    if all_results:
        for stats in all_results:
            print(f"{stats['instance']:>10} (n={stats['n']:>5}): {stats['best_gap_pct']:>6.2f}% gap, "
                  f"{stats['median_runtime']:>6.1f}s, parity={stats['parity']}")
    else:
        print("No results completed")
    
    print(f"\nResults saved to: {results_file}")
    return len(all_results), results_file

def main():
    # Define experiments
    experiments = [
        {
            'tag': 'baseline_short',
            'description': 'Baseline control with short time budgets (5-15 min)',
            'instances': ['pr2392', 'fnl4461', 'pla33810'],
            'seeds': [42, 123],
            'budgets': {
                'pr2392': 300,   # 5 minutes
                'fnl4461': 600,  # 10 minutes  
                'pla33810': 900  # 15 minutes
            },
            'k_values': {
                'pr2392': 25,   # Modest k for 2392 cities
                'fnl4461': 20,  # Modest k for 4461 cities
                'pla33810': 15  # Modest k for 33810 cities
            }
        },
        {
            'tag': 'baseline_medium',
            'description': 'Baseline control with medium time budgets (10-30 min)', 
            'instances': ['pr2392', 'fnl4461'],  # Skip pla33810 for medium run
            'seeds': [42, 123, 456],
            'budgets': {
                'pr2392': 600,   # 10 minutes
                'fnl4461': 1800  # 30 minutes
            },
            'k_values': {
                'pr2392': 25,   # Modest k
                'fnl4461': 20   # Modest k
            }
        }
    ]
    
    print("=== CONTROL BASELINE EXPERIMENTS ===")
    print("Running conservative baseline configurations")
    print("Parameters: agents=4, or_opt=False, integrator=popmusic, modest k, workers=1")
    print()
    
    total_completed = 0
    result_files = []
    
    for exp in experiments:
        completed, result_file = run_control_experiment(
            exp['tag'], exp['instances'], exp['seeds'], 
            exp['budgets'], exp['k_values'], exp['description']
        )
        total_completed += completed
        result_files.append(result_file)
        
        print(f"\nCompleted {completed}/{len(exp['instances'])} instances for {exp['tag']}")
        print("-" * 60)
        print()
    
    print(f"=== ALL CONTROL EXPERIMENTS COMPLETE ===")
    print(f"Total instances completed: {total_completed}")
    print("Result files:")
    for rf in result_files:
        print(f"  - {rf}")

if __name__ == "__main__":
    main()