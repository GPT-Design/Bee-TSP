#!/usr/bin/env python3
"""
Quick ablation studies with shorter budgets for rapid testing.
One knob at a time from baseline.
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
        config['bees']['time_budget_s'] = 2.5  # 2.5s fixed
    
    return config

def run_quick_ablation(ablation_type, description):
    """Run quick ablation with shorter budgets"""
    
    # Quick test budgets - much shorter for rapid iteration
    instances = ['pr2392', 'fnl4461']  # Skip pla33810 for quick tests
    seeds = [42, 123]  # Reduced seeds
    budgets = {
        'pr2392': 180,   # 3 minutes
        'fnl4461': 300   # 5 minutes
    }
    k_values = {
        'pr2392': 25,   # Baseline k
        'fnl4461': 20   # Baseline k
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"quick_{ablation_type}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Quick Ablation: {ablation_type} ===")
    print(f"Description: {description}")
    print(f"Quick budgets: pr2392=3min, fnl4461=5min")
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
            
        # Get configuration for this ablation
        config = get_ablation_config(budgets[instance], k_values[instance], ablation_type)
        
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
            
            print(f"[RESULT] {stats['best_gap_pct']:.2f}% gap (vs baseline)")
            print(f"[RESULT] {improvements_per_min:.1f} improvements/minute")
            print(f"[RESULT] {stats['median_runtime']:.1f}s runtime, parity: {stats['parity']}")
            
        except Exception as e:
            print(f"[ERROR] {instance} failed: {e}")
        
        print()
    
    # Save results
    output_data = {
        'experiment': {
            'type': 'quick_ablation',
            'ablation': ablation_type,
            'description': description,
            'timestamp': timestamp
        },
        'results': all_results
    }
    
    results_file = results_dir / f"quick_{ablation_type}_results.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"=== QUICK SUMMARY: {ablation_type} ===")
    for stats in all_results:
        total_improvements = sum(r['improvements'] for r in stats['results'])
        total_runtime_min = stats['median_runtime'] / 60.0
        improvements_per_min = total_improvements / total_runtime_min if total_runtime_min > 0 else 0
        
        print(f"{stats['instance']:>8}: {stats['best_gap_pct']:>6.2f}% gap, "
              f"{improvements_per_min:>4.1f} imp/min, {stats['parity']}")
    
    return len(all_results)

def main():
    """Run all quick ablations"""
    ablations = [
        ('oropt_only_on', 'Enable Or-opt local search only'),
        ('agents_8_only', 'Increase agents per zone to 8 only'),
        ('ucr_only', 'Use UCR integrator only'),
        ('k_plus_4_only', 'Increase candidate k by 4 only'),
        ('shorter_bee_time_only', 'Reduce agent time to 2.5s only')
    ]
    
    print("=== QUICK ABLATION STUDIES ===")
    print("Fast iteration with shorter budgets")
    print("Baseline reference: pr2392 ~12.7% gap, fnl4461 ~34.9% gap")
    print()
    
    for ablation_type, description in ablations:
        print(f"Running: {ablation_type}")
        try:
            completed = run_quick_ablation(ablation_type, description)
            print(f"✓ {ablation_type}: {completed}/2 instances completed")
        except Exception as e:
            print(f"✗ {ablation_type} failed: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ablation = sys.argv[1]
        descriptions = {
            'oropt_only_on': 'Enable Or-opt local search only',
            'agents_8_only': 'Increase agents per zone to 8 only',
            'ucr_only': 'Use UCR integrator only', 
            'k_plus_4_only': 'Increase candidate k by 4 only',
            'shorter_bee_time_only': 'Reduce agent time to 2.5s only'
        }
        if ablation in descriptions:
            run_quick_ablation(ablation, descriptions[ablation])
        else:
            print(f"Available: {list(descriptions.keys())}")
    else:
        main()