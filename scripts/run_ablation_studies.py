#!/usr/bin/env python3
"""
Ablation studies - one knob at a time from baseline:
- oropt_only_on
- agents_8_only  
- ucr_only
- k_plus_4_only
- shorter_bee_time_only (2-3s)
"""
import sys, datetime, json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_all_three_instances import run_solver_test, load_best_known

def get_ablation_config(time_budget, k_value, ablation_type):
    """Get configuration with single ablation from baseline"""
    
    # Start with baseline configuration
    config = {
        'candidate': {
            'k': k_value,
            'use_knn': True
        },
        'zones': {
            'agents_per_zone': 4  # Baseline
        },
        'bees': {
            'time_budget_s': min(time_budget * 0.15, 12.0)  # Baseline agent time
        },
        'local_search': {
            'kick_period_moves': 400,
            'use_or_opt': False  # Baseline
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
            'method': 'popmusic'  # Baseline
        },
        'termination': {
            'wall_time_s': time_budget
        }
    }
    
    # Apply single ablation
    if ablation_type == 'oropt_only_on':
        config['local_search']['use_or_opt'] = True
        
    elif ablation_type == 'agents_8_only':
        config['zones']['agents_per_zone'] = 8
        
    elif ablation_type == 'ucr_only':
        config['integrator']['method'] = 'ucr'
        
    elif ablation_type == 'k_plus_4_only':
        config['candidate']['k'] = k_value + 4  # Increase k by 4
        
    elif ablation_type == 'shorter_bee_time_only':
        config['bees']['time_budget_s'] = min(2.5, time_budget * 0.05)  # 2-3s range
    
    return config

def run_ablation_experiment(ablation_type, description):
    """Run single ablation experiment"""
    
    # Use the requested budgets: 10/30/60-90 min by size
    instances = ['pr2392', 'fnl4461', 'pla33810']
    seeds = [42, 123, 456]
    budgets = {
        'pr2392': 600,   # 10 minutes
        'fnl4461': 1800, # 30 minutes
        'pla33810': 3600 # 60 minutes (could extend to 90 if needed)
    }
    k_values = {
        'pr2392': 25,   # Baseline k
        'fnl4461': 20,  # Baseline k
        'pla33810': 15  # Baseline k
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"ablation_{ablation_type}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Ablation Study: {ablation_type} ===")
    print(f"Description: {description}")
    print(f"Timestamp: {timestamp}")
    print(f"Budgets: pr2392=10min, fnl4461=30min, pla33810=60min")
    print(f"Workers: 1 (sequential, de-noised)")
    print(f"Results directory: {results_dir}")
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
            
        # Get configuration for this ablation
        config = get_ablation_config(budgets[instance], k_values[instance], ablation_type)
        
        print(f"[CONFIG] {ablation_type}: time_budget={budgets[instance]}s")
        print(f"[CONFIG] k={config['candidate']['k']}, agents={config['zones']['agents_per_zone']}")
        print(f"[CONFIG] or_opt={config['local_search']['use_or_opt']}, integrator={config['integrator']['method']}")
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
            
            # Calculate improvements per minute
            total_improvements = sum(r['improvements'] for r in stats['results'])
            total_runtime_min = stats['median_runtime'] / 60.0
            improvements_per_min = total_improvements / total_runtime_min if total_runtime_min > 0 else 0
            
            print(f"[RESULT] {instance}: {stats['best_gap_pct']:.2f}% gap")
            print(f"[RESULT] Best: {stats['best_found']:.0f}, Target: {stats['optimal_length']:.0f}")
            print(f"[RESULT] Runtime: {stats['median_runtime']:.1f}s, Improvements/min: {improvements_per_min:.1f}")
            print(f"[RESULT] Parity: {stats['parity']}")
            
        except Exception as e:
            print(f"[ERROR] {instance} failed: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # Save results
    output_data = {
        'experiment': {
            'type': 'ablation_study',
            'ablation': ablation_type,
            'description': description,
            'timestamp': timestamp
        },
        'parameters': {
            'budgets': budgets,
            'k_values': k_values,
            'workers': 1,
            'seeds': seeds
        },
        'instances': instances,
        'results': all_results
    }
    
    results_file = results_dir / f"ablation_{ablation_type}_results.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"=== ABLATION SUMMARY: {ablation_type} ===")
    if all_results:
        for stats in all_results:
            total_improvements = sum(r['improvements'] for r in stats['results'])
            total_runtime_min = stats['median_runtime'] / 60.0
            improvements_per_min = total_improvements / total_runtime_min if total_runtime_min > 0 else 0
            
            print(f"{stats['instance']:>10} (n={stats['n']:>5}): {stats['best_gap_pct']:>6.2f}% gap, "
                  f"{improvements_per_min:>4.1f} imp/min, parity={stats['parity']}")
    else:
        print("No results completed")
    
    print(f"\nResults saved to: {results_file}")
    return len(all_results), results_file

def main():
    # Define ablation experiments
    ablations = [
        ('oropt_only_on', 'Enable Or-opt local search only (baseline + or_opt=True)'),
        ('agents_8_only', 'Increase agents per zone to 8 only (baseline + agents=8)'),
        ('ucr_only', 'Use UCR integrator only (baseline + integrator=ucr)'),
        ('k_plus_4_only', 'Increase candidate k by 4 only (baseline + k+4)'),
        ('shorter_bee_time_only', 'Reduce agent time to 2-3s only (baseline + shorter_time)')
    ]
    
    print("=== ABLATION STUDIES ===")
    print("One parameter change at a time from baseline")
    print("Baseline: agents=4, or_opt=False, integrator=popmusic, modest k, 12s agent time")
    print("Budget: 10/30/60 min by instance size, workers=1")
    print()
    
    completed_experiments = []
    
    for ablation_type, description in ablations:
        print(f"Starting ablation: {ablation_type}")
        try:
            completed, result_file = run_ablation_experiment(ablation_type, description)
            completed_experiments.append((ablation_type, completed, result_file))
            print(f"Completed {completed}/3 instances for {ablation_type}")
        except Exception as e:
            print(f"FAILED ablation {ablation_type}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)
        print()
    
    print(f"=== ALL ABLATION STUDIES COMPLETE ===")
    for ablation_type, completed, result_file in completed_experiments:
        print(f"{ablation_type}: {completed}/3 instances -> {result_file}")

if __name__ == "__main__":
    # Ask which ablation to run
    ablations = [
        'oropt_only_on',
        'agents_8_only', 
        'ucr_only',
        'k_plus_4_only',
        'shorter_bee_time_only'
    ]
    
    if len(sys.argv) > 1:
        ablation = sys.argv[1]
        if ablation in ablations:
            descriptions = {
                'oropt_only_on': 'Enable Or-opt local search only',
                'agents_8_only': 'Increase agents per zone to 8 only',
                'ucr_only': 'Use UCR integrator only', 
                'k_plus_4_only': 'Increase candidate k by 4 only',
                'shorter_bee_time_only': 'Reduce agent time to 2-3s only'
            }
            run_ablation_experiment(ablation, descriptions[ablation])
        else:
            print(f"Unknown ablation: {ablation}")
            print(f"Available: {ablations}")
    else:
        print("Usage: python run_ablation_studies.py <ablation_type>")
        print(f"Available ablations: {ablations}")
        print("Or run main() for all ablations sequentially")
        main()