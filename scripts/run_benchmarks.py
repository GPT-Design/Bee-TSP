#!/usr/bin/env python3
import argparse, os, time, json, sys
from pathlib import Path

# Make project root importable even if you run this from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

from bee_tsp.solver import BeeTSPSolver
from output_app import StandardOutputManager

def load_config(path):
    if yaml is None:
        raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_instances(instances_file, tsplib_dir):
    names = []
    with open(instances_file, 'r') as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                names.append(s)
    paths = [str(Path(tsplib_dir) / f"{name}.tsp") for name in names]
    return list(zip(names, paths))

def run_once(solver, instance_name, instance_path, output_mgr, seed=0):
    """Run solver once with standardized output management."""
    start = time.time()
    result = solver.solve(instance_name, instance_path, seed=seed)
    runtime = time.time() - start
    
    # Save individual seed log using standardized format
    anytime_events = [{"event":"improve","t":float(t),"best":float(best)} 
                      for t, best in result.get("anytime", [])]
    summary_event = {"event":"summary","seed":seed,"best":float(result.get("best_length", 0.0)),"runtime":runtime}
    all_events = anytime_events + [summary_event]
    
    log_path = output_mgr.save_seed_log(seed, all_events)
    return result, runtime, str(log_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to YAML config (configs/params.yaml)')
    args = ap.parse_args()
    cfg = load_config(args.config)

    paths = cfg['paths']
    solver_cfg = cfg['solver']
    bench = cfg['benchmark']

    instances = load_instances(paths['instances_file'], paths['tsplib_dir'])
    solver = BeeTSPSolver(solver_cfg)

    for name, path in instances:
        print(f"[INFO] Instance: {name} -> {path}")
        
        # Use standardized output management
        output_mgr = StandardOutputManager(name, paths.get('results_dir', 'results'))
        
        # Save configuration for this instance
        instance_cfg = {
            "solver": solver_cfg,
            "benchmark": bench,
            "paths": paths,
            "instance": {"name": name, "path": path}
        }
        config_path = output_mgr.save_config(instance_cfg)
        
        # Track results across seeds
        seed_results = []
        
        for seed in bench.get('seed_list', [0]):
            res, rt, log = run_once(solver, name, path, output_mgr, seed=seed)
            seed_results.append({
                "seed": seed,
                "best_length": res.get('best_length', 0.0),
                "runtime": rt,
                "anytime_count": len(res.get("anytime", []))
            })
            print(f"  seed={seed}  best={res.get('best_length', 0.0):.1f}  runtime={rt:.2f}s  log={log}")
        
        # Save summary results
        summary_data = {
            "instance": name,
            "instance_path": path,
            "solver_config_summary": {
                "candidate_k": solver_cfg.get("candidate", {}).get("k", "unknown"),
                "wall_time_s": solver_cfg.get("termination", {}).get("wall_time_s", "unknown")
            },
            "seed_results": seed_results,
            "best_overall": min(r["best_length"] for r in seed_results),
            "avg_runtime": sum(r["runtime"] for r in seed_results) / len(seed_results),
            "total_seeds": len(seed_results)
        }
        
        results_path = output_mgr.save_results(summary_data)
        print(f"[INFO] Results saved to: {results_path}")
        print(f"[INFO] Config saved to: {config_path}")
        print(f"[INFO] Run directory: {output_mgr.get_run_directory()}")
        print()

if __name__ == "__main__":
    main()
