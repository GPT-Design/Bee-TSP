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

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def run_once(solver, instance_name, instance_path, out_dir, seed=0):
    start = time.time()
    result = solver.solve(instance_name, instance_path, seed=seed)
    runtime = time.time() - start
    ensure_dir(out_dir)
    log_path = Path(out_dir) / f"seed_{seed}.jsonl"
    with open(log_path, 'w') as w:
        for t, best in result.get("anytime", []):
            w.write(json.dumps({"event":"improve","t":float(t),"best":float(best)}) + "\n")
        w.write(json.dumps({"event":"summary","seed":seed,"best":float(result.get("best_length", 0.0)),"runtime":runtime}) + "\n")
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
        out_dir = Path(paths['results_dir']) / f"bstsp/{name}"
        for seed in bench.get('seed_list', [0]):
            res, rt, log = run_once(solver, name, path, out_dir, seed=seed)
            print(f"  seed={seed}  best={res.get('best_length', 0.0):.1f}  runtime={rt:.2f}s  log={log}")

if __name__ == "__main__":
    main()
