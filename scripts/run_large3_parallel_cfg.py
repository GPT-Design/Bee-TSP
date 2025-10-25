#!/usr/bin/env python3
r"""
Parallel, config-aware runner for fnl4461, pla7397, pla85900.
Usage:
  python scripts/run_large3_parallel_cfg.py --config configs/params_z3000.yaml --seeds 8 --workers 4 --wall_small 1800 --wall_large 7200
"""
import argparse, os, sys, time, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_cfg(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_one(inst_name, tsp_path, cfg, seed):
    from bee_tsp import BeeTSPSolver
    t0 = time.time()
    solver = BeeTSPSolver(cfg)
    res = solver.solve(inst_name, str(tsp_path), seed=seed)
    rt = time.time() - t0
    return seed, res, rt

def run_family(inst, tsp_path, solver_cfg, seeds, out_dir, wall_override, workers):
    from copy import deepcopy
    cfg = deepcopy(solver_cfg)
    if wall_override is not None:
        cfg.setdefault("termination", {})
        cfg["termination"]["wall_time_s"] = float(wall_override)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, inst, tsp_path, cfg, s): s for s in seeds}
        for fut in as_completed(futs):
            seed, res, rt = fut.result()
            log = out_dir / f"seed_{seed}.jsonl"
            with open(log, "w", encoding="utf-8") as w:
                for t, best in res.get("anytime", []):
                    w.write(json.dumps({"event":"improve","t":float(t),"best":float(best)}) + "\n")
                w.write(json.dumps({"event":"summary","seed":int(seed),"best":float(res.get("best_length", 0.0)),"runtime":rt}) + "\n")
            print(f"[{inst}] seed={seed} best={res.get('best_length', 0.0):.1f} time={rt:.1f}s -> {log}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--tsplib_dir", default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
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

    # Run families
    run_family("fnl4461", mapping["fnl4461"], solver_cfg, seeds, Path(results_dir) / "bstsp_large3/fnl4461", wall_small, args.workers)
    run_family("pla7397", mapping["pla7397"], solver_cfg, seeds, Path(results_dir) / "bstsp_large3/pla7397", wall_small, args.workers)
    run_family("pla85900", mapping["pla85900"], solver_cfg, seeds, Path(results_dir) / "bstsp_large3/pla85900", wall_large, args.workers)

if __name__ == "__main__":
    main()
