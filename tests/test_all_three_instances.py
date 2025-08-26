#!/usr/bin/env python3
"""
Comprehensive test script for all three instances: att48, kroA100, ch130.
Tests Bee-TSP solver performance against optimal tour files with full configuration.
"""

import time
import json
import statistics
import argparse
import re
import csv
from datetime import datetime
from pathlib import Path
from bee_tsp import BeeTSPSolver
from bee_tsp.tsplib import load_tsplib
from bee_tsp.metrics import build_eval_fn

try:
    import yaml
except ImportError:
    yaml = None

def generate_timestamped_filename(base_name="Bee_TSP", extension="json"):
    """Generate timestamped filename in Japanese date format: YY-MM-DD_HHMM_<base_name>.<extension>"""
    now = datetime.now()
    timestamp = now.strftime("%y-%m-%d_%H%M")
    return f"{timestamp}_{base_name}.{extension}"

def load_best_known(filepath="best_known.csv"):
    """Load best known values and optimal tour availability from CSV file."""
    best_known = {}
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instance = row['instance'].strip()
                best_val = int(row['best_known'].strip())
                has_opt_tour = row.get('op.tour_available', 'N').strip().upper() == 'Y'
                best_known[instance] = {'best_known': best_val, 'has_opt_tour': has_opt_tour}
    except Exception as e:
        print(f"Warning: Could not load best_known.csv: {e}")
    return best_known

def parse_opt_tour(opt_tour_path):
    """Parse .opt.tour file to extract optimal tour and calculate its length."""
    tour = []
    optimal_length = None
    in_tour = False

    with open(opt_tour_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Length in comments or standalone lines
            m = re.search(r"Length\s*:?\s*([0-9]+)", line, re.IGNORECASE)
            if m and optimal_length is None:
                optimal_length = int(m.group(1))
            # Some files: "COMMENT ... (21282)"
            m2 = re.search(r"\(([0-9]+)\)", line)
            if m2 and optimal_length is None:
                optimal_length = int(m2.group(1))

            if line.upper().startswith("TOUR_SECTION"):
                in_tour = True
                continue
            if in_tour:
                if line == "-1" or line.upper() == "EOF":
                    break
                # accept multiple ids per line
                for tok in line.split():
                    if tok == "-1":
                        in_tour = False
                        break
                    if tok.isdigit():
                        tour.append(int(tok) - 1)  # 0-based

    return tour, optimal_length

# Removed calculate_tour_length - now using build_eval_fn for metric consistency

def run_solver_test(instance_name, tsp_path, opt_tour_path, seeds, time_budget=60.0, solver_config=None, best_known=None):
    """Run solver on instance with multiple seeds and compare to optimal."""
    
    print(f"\\n=== Testing {instance_name} ===")
    
    # Load instance first
    ins = load_tsplib(str(tsp_path))
    n = ins['n']
    
    # Parse optimal tour if available
    if opt_tour_path is not None:
        opt_tour, opt_length_from_file = parse_opt_tour(opt_tour_path)
    else:
        # Use best_known value and construct identity tour for evaluation
        opt_length_from_file = None
        opt_tour = list(range(n))  # Identity tour for metric consistency
    
    # Use the EXACT same metric path as the solver
    # The solver modifies self.cfg directly, so we replicate that exactly
    if solver_config is None:
        harness_cfg = {}
    else:
        harness_cfg = dict(solver_config)  # Copy to avoid modifying original
    
    # Replicate solver's metric setup exactly
    harness_cfg.setdefault("metric", {})["edge_weight_type"] = ins.get('edge_weight_type', 'UNKNOWN') 
    
    # Build evaluation function that matches solver exactly  
    eval_tour = build_eval_fn(harness_cfg, ins)
    
    # Calculate true optimal length using consistent evaluation
    if opt_tour_path is not None:
        # We have an optimal tour - evaluate it
        true_opt_length = eval_tour(opt_tour)
        
        # Parity check: verify harness evaluation matches best_known
        expected_best = best_known.get(instance_name, {}).get('best_known', 'N/A') if best_known else 'N/A'
        print(f"[PARITY] {instance_name}: opt_eval={true_opt_length} best_known={expected_best}")
        
        print(f"Instance size: {n} cities")
        print(f"Optimal tour length (calculated): {true_opt_length}")
        if opt_length_from_file:
            print(f"Optimal tour length (from file): {opt_length_from_file}")
        
        # Validate against best_known.csv if available
        if best_known and instance_name in best_known:
            if best_known[instance_name]['best_known'] == true_opt_length:
                print(f"* Validated against best_known.csv: {best_known[instance_name]['best_known']}")
            else:
                print(f"! Mismatch with best_known.csv: expected {best_known[instance_name]['best_known']}, got {true_opt_length}")
    else:
        # No optimal tour available - use best_known value
        if best_known and instance_name in best_known:
            true_opt_length = best_known[instance_name]['best_known']
            print(f"[BEST_KNOWN] {instance_name}: using best_known={true_opt_length}")
            print(f"Instance size: {n} cities")
            print(f"Best known length: {true_opt_length} (no optimal tour available)")
        else:
            print(f"ERROR: No optimal tour or best_known value available for {instance_name}")
            return None
    
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
            'improvements': len(result['anytime']),
            'anytime': result['anytime']  # Include anytime data for TTT calculations
        })
        
        print(f"Length: {best_length:.1f}, Gap: {gap:.2f}%, Time: {runtime:.1f}s, Improvements: {len(result['anytime'])}")
    
    # Calculate statistics
    lengths = [r['length'] for r in results]
    gaps = [r['gap_pct'] for r in results]
    times = [r['runtime'] for r in results]
    improvements = [r['improvements'] for r in results]
    
    # Percentiles
    def pctl(vals, q):
        if not vals: return None
        vals = sorted(vals)
        k = max(0, min(len(vals)-1, int(round((len(vals)-1)*q))))
        return vals[k]
    
    def time_to_target(anytime, target, optimal_value):
        # anytime: list[(t, best)] sorted by t
        # Parity guard: reject any L(t) < L_opt as metric bugs
        for t, best in sorted(anytime, key=lambda x: x[0]):
            if best < optimal_value:
                continue  # Skip impossible values (metric bug)
            if best <= target: 
                return t
        return None
    
    def time_to_optimal(anytime, optimal_value):
        # TTT@opt: Hit when L(t) == L_opt (exact equality only)
        # Parity guard: reject any L(t) < L_opt as metric bugs
        for t, best in sorted(anytime, key=lambda x: x[0]):
            if best < optimal_value:
                continue  # Skip impossible values (metric bug)
            if best == optimal_value: 
                return t
        return None
    
    # Percentiles
    p10_gap = pctl(gaps, 0.10)
    p90_gap = pctl(gaps, 0.90)
    
    # TTT calculation based on op.tour_available
    ttt = {}
    instance_info = best_known.get(instance_name, {}) if best_known else {}
    has_opt_tour = instance_info.get('has_opt_tour', False)
    opt_path = f"data/tsplib/{instance_name}.opt.tour"
    
    if has_opt_tour and Path(opt_path).exists():
        # TTT@opt (first timestamp achieving EXACT optimal length - equality only)
        hits = [time_to_optimal(r.get("anytime", []), true_opt_length) for r in results]
        hits = [h for h in hits if h is not None]
        ttt["ttt_opt_s"] = (sum(hits)/len(hits)) if hits else None
        
        # TTT@ε% using tight floor formula: T_ε = floor(L_opt * (1 + ε/100) + 1e-9)
        for pct in [0.1, 0.5, 1.0]:
            target = int(true_opt_length * (1.0 + pct/100.0) + 1e-9)  # floor() via int()
            hits = [time_to_target(r.get("anytime", []), target, true_opt_length) for r in results]
            hits = [h for h in hits if h is not None]
            key = f"ttt_{pct}pct_s"
            ttt[key] = (sum(hits)/len(hits)) if hits else None
    else:
        # Standard TTT @ {1%, 0.5%, 0.1%} using tight floor formula
        for pct in [0.1, 0.5, 1.0]:
            target = int(true_opt_length * (1.0 + pct/100.0) + 1e-9)  # floor() via int()
            hits = [time_to_target(r.get("anytime", []), target, true_opt_length) for r in results]
            hits = [h for h in hits if h is not None]
            key = f"ttt_{pct}pct_s"
            ttt[key] = (sum(hits)/len(hits)) if hits else None
    
    # Guardrail: detect impossible "negative gap"
    best_overall = min(lengths)
    instance_info = best_known.get(instance_name, {}) if best_known else {}
    has_opt_tour = instance_info.get('has_opt_tour', False)
    parity_status = "OK"
    
    if has_opt_tour and best_overall < true_opt_length:
        print(f"[WARN] METRIC_MISMATCH? best({best_overall}) < best_known({true_opt_length}) for {instance_name}")
        parity_status = "FAIL"
    
    stats = {
        'instance': instance_name,
        'n': n,
        'optimal_length': true_opt_length,
        'optimal_from_file': opt_length_from_file,
        'num_runs': len(results),
        'best_found': best_overall,
        'worst_found': max(lengths),
        'median_length': statistics.median(lengths),
        'mean_length': statistics.mean(lengths),
        'best_gap_pct': min(gaps),
        'worst_gap_pct': max(gaps),
        'median_gap_pct': statistics.median(gaps),
        'mean_gap_pct': statistics.mean(gaps),
        'p10_gap_pct': p10_gap,
        'p90_gap_pct': p90_gap,
        'median_runtime': statistics.median(times),
        'mean_improvements': statistics.mean(improvements),
        'parity': parity_status,
        'results': results,
        **ttt
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
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (default: timestamped file in results/ directory)')
    parser.add_argument('--config', type=str, help='YAML file to load solver config')
    
    args = parser.parse_args()
    
    # Load base config from YAML if provided
    base_config = {}
    if args.config:
        if yaml is None:
            print("ERROR: PyYAML not available. Install with: pip install pyyaml")
            return
        try:
            with open(args.config, 'r') as f:
                yaml_data = yaml.safe_load(f)
            base_config = yaml_data.get('solver', {})
            print(f"Loaded base config from: {args.config}")
        except Exception as e:
            print(f"ERROR loading config {args.config}: {e}")
            return
    
    # Build solver configuration from CLI args (CLI overrides YAML)
    solver_config = {
        'candidate': {
            'k': args.candidate_k, 
            'use_knn': True,
            **base_config.get('candidate', {})
        },
        'zones': {
            'agents_per_zone': 4,
            **base_config.get('zones', {})
        },
        'bees': {
            'time_budget_s': args.agent_time,
            **base_config.get('bees', {})
        },
        'local_search': {
            'kick_period_moves': 400,
            **base_config.get('local_search', {})
        },
        'scouting': {
            'dynamic': {
                'enabled': True,
                'no_improve_s': args.scouts_stagnation,
                'max_restarts_per_seed': args.max_restarts,
                'polish_time_s': 0.6,
                **base_config.get('scouting', {}).get('dynamic', {})
            },
            **{k: v for k, v in base_config.get('scouting', {}).items() if k != 'dynamic'}
        },
        'termination': {
            'wall_time_s': args.time_budget,
            **base_config.get('termination', {})
        }
    }
    
    # Override with CLI values (CLI takes precedence)
    solver_config['candidate']['k'] = args.candidate_k
    solver_config['bees']['time_budget_s'] = args.agent_time  
    solver_config['scouting']['dynamic']['no_improve_s'] = args.scouts_stagnation
    solver_config['scouting']['dynamic']['max_restarts_per_seed'] = args.max_restarts
    solver_config['termination']['wall_time_s'] = args.time_budget
    
    # Test instances - small, medium, and large scale
    test_instances = [
        # Small instances (< 200 cities)
        ('att48', 'data/tsplib/att48.tsp', 'data/tsplib/att48.opt.tour'),
        ('kroA100', 'data/tsplib/kroA100.tsp', 'data/tsplib/kroA100.opt.tour'),
        ('ch130', 'data/tsplib/ch130.tsp', 'data/tsplib/ch130.opt.tour'),
        # Large instances (1000+ cities)
        ('pr2392', 'data/tsplib/pr2392.tsp', 'data/tsplib/pr2392.opt.tour'),
        ('fnl4461', 'data/tsplib/fnl4461.tsp', None),  # No .opt.tour available
        ('pla33810', 'data/tsplib/pla33810.tsp', 'data/tsplib/pla33810.opt.tour')
    ]
    
    # Load best known values for validation
    best_known = load_best_known("best_known.csv")
    
    print("Bee-TSP Solver vs Optimal Solutions - Complete Test")
    print("=" * 60)
    print(f"Configuration: time_budget={args.time_budget}s, k={args.candidate_k},")
    print(f"               agent_time={args.agent_time}s, scout_stagnation={args.scouts_stagnation}s,")
    print(f"               max_restarts={args.max_restarts}, seeds={args.seeds}")
    print(f"Testing instances: {[t[0] for t in test_instances]}")
    
    # Show instance sizes
    sizes_info = []
    for name, tsp_path, _ in test_instances:
        try:
            ins = load_tsplib(tsp_path)
            sizes_info.append(f"{name}(n={ins['n']})")
        except:
            sizes_info.append(f"{name}(n=?)")
    print(f"Instance sizes: {', '.join(sizes_info)}")
    if best_known:
        print(f"Best known values loaded: {len(best_known)} instances")
    
    # Generate timestamped output filename and ensure results directory exists
    if args.output is None:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)  # Create results directory if it doesn't exist
        output_filename = results_dir / generate_timestamped_filename("Bee_TSP", "json")
    else:
        output_filename = Path(args.output)
    
    print(f"Results will be saved to: {output_filename}")
    
    all_results = []
    
    for instance_name, tsp_path, opt_tour_path in test_instances:
        if not Path(tsp_path).exists():
            print(f"Skipping {instance_name}: {tsp_path} not found")
            continue
        if opt_tour_path is not None and not Path(opt_tour_path).exists():
            print(f"Skipping {instance_name}: {opt_tour_path} not found")
            continue
            
        stats = run_solver_test(instance_name, tsp_path, opt_tour_path, args.seeds, 
                                args.time_budget, solver_config, best_known)
        all_results.append(stats)
    
    # Print summary report
    print("\\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print(f"{'Instance':<10} {'n':<4} {'Optimal':<8} {'Best':<8} {'Median':<8} {'[P10-P90]':<10} {'Best Gap%':<10} {'Med Gap%':<10} {'TTT Key':<8} {'Parity':<7}")
    print("-" * 93)
    
    for stats in all_results:
        p10_p90 = f"[{stats['p10_gap_pct']:.1f}-{stats['p90_gap_pct']:.1f}]" if stats['p10_gap_pct'] is not None else "N/A"
        
        # Display TTT@opt if available, else TTT@1%
        if 'ttt_opt_s' in stats and stats['ttt_opt_s'] is not None:
            ttt_display = f"{stats['ttt_opt_s']:.1f}s@opt"
        else:
            ttt_1pct = stats.get('ttt_1.0pct_s', 'N/A')
            if isinstance(ttt_1pct, float):
                ttt_display = f"{ttt_1pct:.1f}s@1%"
            else:
                ttt_display = "N/A"
        
        print(f"{stats['instance']:<10} "
              f"{stats['n']:<4} "
              f"{stats['optimal_length']:<8.0f} "
              f"{stats['best_found']:<8.0f} "
              f"{stats['median_length']:<8.0f} "
              f"{p10_p90:<10} "
              f"{stats['best_gap_pct']:<10.2f} "
              f"{stats['median_gap_pct']:<10.2f} "
              f"{ttt_display:<8} "
              f"{stats.get('parity', 'N/A'):<7}")
    
    # Save detailed results with configuration
    output_data = {
        'configuration': {
            'time_budget_s': args.time_budget,
            'seeds': args.seeds,
            'solver_config': solver_config,
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': all_results
    }
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\\nDetailed results saved to: {output_filename}")
    
    # Performance analysis
    print("\\nPERFORMANCE ANALYSIS:")
    print("-" * 50)
    for stats in all_results:
        print(f"\\n{stats['instance']} (n={stats['n']}):")
        found_optimal = stats['best_gap_pct'] < 0.01
        print(f"  • Best gap: {stats['best_gap_pct']:.2f}% "
              f"(found optimal: {'YES' if found_optimal else 'NO'})")
        print(f"  • Median gap: {stats['median_gap_pct']:.2f}% [{stats['p10_gap_pct']:.1f}% - {stats['p90_gap_pct']:.1f}%]")
        print(f"  • Worst gap: {stats['worst_gap_pct']:.2f}%")
        print(f"  • Median runtime: {stats['median_runtime']:.1f}s")
        print(f"  • Avg improvements: {stats['mean_improvements']:.1f}")
        
        # Time-to-target analysis
        ttt_results = []
        if 'ttt_opt_s' in stats and stats['ttt_opt_s'] is not None:
            # Show TTT@opt + windowed percentages
            ttt_results.append(f"opt: {stats['ttt_opt_s']:.1f}s")
            for pct in ['0.1', '0.5', '1.0']:
                key = f'ttt_{pct}pct_s'
                val = stats.get(key)
                if val is not None:
                    ttt_results.append(f"{pct}%: {val:.1f}s")
                else:
                    ttt_results.append(f"{pct}%: N/A")
        else:
            # Standard TTT analysis (windowed percentages)
            for pct in ['0.1', '0.5', '1.0']:
                key = f'ttt_{pct}pct_s'
                val = stats.get(key)
                if val is not None:
                    ttt_results.append(f"{pct}%: {val:.1f}s")
                else:
                    ttt_results.append(f"{pct}%: N/A")
        print(f"  • TTT: {' | '.join(ttt_results)}")

if __name__ == '__main__':
    main()