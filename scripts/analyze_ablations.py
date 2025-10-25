# scripts/analyze_ablations.py

import json
from pathlib import Path
from collections import defaultdict
import statistics

OPT_VALUES = {
    'pr2392': 378032,
    'fnl4461': 182566
}

def parse_results(results_dir="results", instance="pr2392"):
    opt_val = OPT_VALUES.get(instance, 378032)
    runs = defaultdict(list)
    
    for result_dir in Path(results_dir).glob(f"*_{instance}*/"):
        json_files = sorted(result_dir.glob("*.json"))
        if not json_files:
            continue
        
        yaml_files = list(result_dir.glob("*.yaml"))
        if yaml_files:
            import yaml
            with open(yaml_files[0]) as f:
                cfg = yaml.safe_load(f)
            config_name = identify_config(cfg['solver'])
        else:
            config_name = "unknown"
        
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            
            seed_num = data['seed_results'][0]['seed']
            jsonl_file = result_dir / f"seed_{seed_num}.jsonl"
            early_slope = None
            ttt_10pct = None
            
            if jsonl_file.exists():
                with open(jsonl_file) as f:
                    events = [json.loads(line) for line in f if line.strip()]
                    early_slope = calc_early_slope(events, opt_val)  # Changed
                    ttt_10pct = calc_ttt_10pct(events, opt_val)      # Changed
            
            for seed_result in data.get('seed_results', []):
                best_len = seed_result['best_length']
                gap_pct = 100 * (best_len - opt_val) / opt_val  # Changed
                
                runs[config_name].append({
                    'seed': seed_result['seed'],
                    'best_length': best_len,
                    'gap_pct': gap_pct,
                    'runtime': seed_result['runtime'],
                    'early_slope': early_slope,
                    'ttt_10pct': ttt_10pct
                })
    
    return runs

def identify_config(solver_cfg):
    """Identify config from solver parameters"""
    ec = solver_cfg['candidate']['edge_cap']
    pn = solver_cfg['candidate']['portals_per_node']
    K = solver_cfg['initializers']['K']
    budget = solver_cfg['initializers']['total_budget_pct']
    forbid = solver_cfg['local_search']['oropt_gate']['forbid_until_pct']
    rounds = solver_cfg['integrator']['rounds']
    hk = solver_cfg['hk1tree']['enabled']
    
    if ec == 8: return '01_edge_cap_8'
    elif ec == 12: return '01_edge_cap_12'
    elif ec == 16: return '01_edge_cap_16'
    elif pn == 3: return '02_portals_3'
    elif pn == 7: return '02_portals_7'
    elif pn == 10: return '02_portals_10'
    elif K == 4: return '03_k_init_4'
    elif budget == 0.10: return '04_init_budget_10'
    elif budget == 0.20: return '04_init_budget_20'
    elif forbid == 0.3: return '05_oropt_forbid_30'
    elif forbid == 0.7: return '05_oropt_forbid_70'
    elif rounds == 3: return '06_merge_3rounds'
    elif hk: return '07_hk_enabled'
    else: return '00_baseline'

def calc_early_slope(events, opt_val):
    """Δgap%/min in first 60s"""
    improves = [e for e in events if e.get('event') == 'improve' and e['t'] <= 60]
    if len(improves) < 2:
        return None
    
    first_gap = 100 * (improves[0]['best'] - opt_val) / opt_val
    last_gap = 100 * (improves[-1]['best'] - opt_val) / opt_val
    time_min = improves[-1]['t'] / 60
    
    return (first_gap - last_gap) / time_min if time_min > 0 else None

def calc_ttt_10pct(events, opt_val):
    """Time to reach ≤10% gap, or None"""
    threshold = opt_val * 1.10
    for e in events:
        if e.get('event') == 'improve' and e['best'] <= threshold:
            return e['t']
    return None

def summarize(runs, wall_s):
    """Generate comparison table"""
    baseline_median = None
    if '00_baseline' in runs:
        baseline_median = statistics.median([r['gap_pct'] for r in runs['00_baseline']])
    
    print(f"\n{'Config':<22} {'N':<3} {'Med%':<8} {'Best%':<8} {'Slope':<8} {'TTT@10%':<10} {'Status':<10}")
    print("-" * 85)
    
    for config in sorted(runs.keys()):
        gaps = [r['gap_pct'] for r in runs[config]]
        slopes = [r['early_slope'] for r in runs[config] if r['early_slope']]
        ttts = [r['ttt_10pct'] for r in runs[config] if r['ttt_10pct']]
        
        median_gap = statistics.median(gaps)
        best_gap = min(gaps)
        median_slope = statistics.median(slopes) if slopes else 0
        
        # TTT@10% display
        if wall_s < 300:
            ttt_display = "N/A"  # Too short to expect 10%
        elif ttts:
            ttt_display = f"{statistics.median(ttts):.1f}s"
        else:
            ttt_display = "✗"
        
        # Regression check
        if baseline_median and config != '00_baseline':
            if median_gap > baseline_median:
                status = "REGRESS"
            else:
                status = "✓"
        else:
            status = "-"
        
        print(f"{config:<22} {len(gaps):<3} {median_gap:>7.2f} {best_gap:>7.2f} "
              f"{median_slope:>7.2f} {ttt_display:<10} {status:<10}")

if __name__ == "__main__":
    import sys
    wall_s = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    instance = sys.argv[2] if len(sys.argv) > 2 else "pr2392"
    runs = parse_results(instance=instance)
    summarize(runs, wall_s)
    print(f"\nOptimal: {OPT_VALUES.get(instance, 378032)}, Wall time: {wall_s}s")