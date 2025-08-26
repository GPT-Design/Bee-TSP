#!/usr/bin/env python3
"""
Parallel runner for large instances using multiple processes (Windows-safe).
Runs seeds concurrently to increase CPU utilization.
"""
import argparse, os, sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def run_one(inst_name, tsp_path, cfg, seed):
    # Import inside the worker
    from bee_tsp.solver import BeeTSPSolver
    solver = BeeTSPSolver(cfg)
    t0 = time.time()
    res = solver.solve(inst_name, str(tsp_path), seed=seed)
    rt = time.time() - t0
    return seed, res, rt

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsplib_dir", default="data/tsplib")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wall_small", type=int, default=1800)
    ap.add_argument("--wall_large", type=int, default=7200)
    args = ap.parse_args()

    # Load solver cfg (and tweak for large-n)
    cfg_path = Path("configs/params.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)["solver"]
    base["candidate"]["k"] = max(24, int(base["candidate"].get("k", 32)))
    base["candidate"]["use_delaunay"] = False
    base["zones"]["target_zone_size"] = max(300, int(base["zones"].get("target_zone_size", 300)))
    base["zones"]["overlap_pct"] = 0.0
    base["bees"]["time_budget_s"] = 0.6

    mapping = {
        "fnl4461": Path(args.tsplib_dir) / "fnl4461.tsp",
        "pla7397": Path(args.tsplib_dir) / "pla7397.tsp",
        "pla85900": Path(args.tsplib_dir) / "pla85900.tsp",
    }
    for k,p in mapping.items():
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    def run_family(inst, wall):
        out_dir = Path(args.results_dir) / f"bstsp_large3/{inst}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(base)
        cfg["termination"]["wall_time_s"] = float(wall)
        seeds = list(range(args.seeds))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, inst, mapping[inst], cfg, s): s for s in seeds}
            for fut in as_completed(futs):
                s = futs[fut]
                seed, res, rt = fut.result()
                log = out_dir / f"seed_{seed}.jsonl"
                with open(log, "w", encoding="utf-8") as w:
                    for t, best in res.get("anytime", []):
                        w.write(json.dumps({"event":"improve","t":float(t),"best":float(best)}) + "\n")
                    w.write(json.dumps({"event":"summary","seed":int(seed),"best":float(res.get("best_length", 0.0)),"runtime":rt}) + "\n")
                print(f"[{inst}] seed={seed} best={res.get('best_length', 0.0):.1f} time={rt:.1f}s  -> {log}")

    run_family("fnl4461", args.wall_small)
    run_family("pla7397", args.wall_small)
    run_family("pla85900", args.wall_large)

if __name__ == "__main__":
    main()
