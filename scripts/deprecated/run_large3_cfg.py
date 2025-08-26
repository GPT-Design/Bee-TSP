#!/usr/bin/env python3
r"""
Config-aware runner for fnl4461, pla7397, pla85900.
Usage:
  python scripts/run_large3_cfg.py --config configs/params_z3000.yaml --seeds 5 --wall_small 1800 --wall_large 7200
You can also override tsplib/results dirs:
  python scripts/run_large3_cfg.py --config configs/params_z3000.yaml --tsplib_dir data/tsplib --results_dir results
"""
import argparse, os, sys, time, json
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_cfg(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_instance(inst_name, tsp_path, solver_cfg, results_dir, seeds, wall_override=None):
    from bee_tsp import BeeTSPSolver
    out_dir = Path(results_dir) / f"bstsp_large3/{inst_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(solver_cfg)
    if wall_override is not None:
        cfg = dict(cfg)
        term = dict(cfg.get("termination", {}))
        term["wall_time_s"] = float(wall_override)
        cfg["termination"] = term

    solver = BeeTSPSolver(cfg)
    for seed in seeds:
        t0 = time.time()
        res = solver.solve(inst_name, str(tsp_path), seed=seed)
        rt = time.time() - t0
        log = out_dir / f"seed_{seed}.jsonl"
        with open(log, "w", encoding="utf-8") as w:
            for t, best in res.get("anytime", []):
                w.write(json.dumps({"event":"improve","t":float(t),"best":float(best)}) + "\n")
            w.write(json.dumps({"event":"summary","seed":int(seed),"best":float(res.get("best_length", 0.0)),"runtime":rt}) + "\n")
        print(f"[{inst_name}] seed={seed} best={res.get('best_length', 0.0):.1f} time={rt:.1f}s -> {log}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--tsplib_dir", default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--wall_small", type=int, default=None)
    ap.add_argument("--wall_large", type=int, default=None)
    args = ap.parse_args()

    cfg_all = load_cfg(args.config)
    solver_cfg = cfg_all["solver"]
    paths = cfg_all.get("paths", {})
    bench = cfg_all.get("benchmark", {})

    tsplib_dir = Path(args.tsplib_dir or paths.get("tsplib_dir", "data/tsplib"))
    results_dir = args.results_dir or paths.get("results_dir", "results")
    seeds = list(range(args.seeds if args.seeds is not None else bench.get("runs", 5)))

    mapping = {
        "fnl4461": tsplib_dir / "fnl4461.tsp",
        "pla7397": tsplib_dir / "pla7397.tsp",
        "pla85900": tsplib_dir / "pla85900.tsp",
    }
    missing = [k for k,p in mapping.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing TSPLIB files: {', '.join(missing)} in {tsplib_dir}")

    default_wall = solver_cfg.get("termination", {}).get("wall_time_s", 1800)
    wall_small = args.wall_small if args.wall_small is not None else default_wall
    wall_large = args.wall_large if args.wall_large is not None else default_wall

    for inst in ["fnl4461", "pla7397"]:
        run_instance(inst, mapping[inst], solver_cfg, results_dir, seeds, wall_override=wall_small)
    run_instance("pla85900", mapping["pla85900"], solver_cfg, results_dir, seeds, wall_override=wall_large)

if __name__ == "__main__":
    main()
