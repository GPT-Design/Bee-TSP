#!/usr/bin/env python3
r"""
Auto-scaling parallel runner.
- Reads base YAML (paths + (optional) default solver params).
- Loads each TSPLIB instance to detect n and overrides solver params using autoscale rules.
- Writes resolved YAML per instance so you can reproduce exactly.
Usage:
  python scripts/run_auto_parallel.py --config configs/params_z3000.yaml --seeds 8 --workers 4 \
    --wall_small 1800 --wall_large 7200
"""
import argparse, os, sys, time, json, yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bee_tsp.tsplib import load_tsplib  # to get n
from bee_tsp import BeeTSPSolver
from bee_tsp.auto_params import autoscale_by_n  # our autoscaler
from output_app import StandardOutputManager

def run_one(inst_name, tsp_path, cfg, seed):
    t0 = time.time()
    solver = BeeTSPSolver(cfg["solver"])
    res = solver.solve(inst_name, str(tsp_path), seed=seed)
    rt = time.time() - t0
    return seed, res, rt

def dump_resolved_yaml(resolved: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params.yaml")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tsplib_dir", default=None)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--wall_small", type=int, default=None)
    ap.add_argument("--wall_large", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    paths = base_cfg.get("paths", {})
    tsplib_dir = Path(args.tsplib_dir or paths.get("tsplib_dir", "data/tsplib"))
    results_dir = Path(args.results_dir or paths.get("results_dir", "results"))
    bench = base_cfg.get("benchmark", {})
    seeds = list(range(args.seeds if args.seeds is not None else bench.get("runs", 5)))

    mapping = {
        "fnl4461": tsplib_dir / "fnl4461.tsp",
        "pla7397": tsplib_dir / "pla7397.tsp",
        "pla85900": tsplib_dir / "pla85900.tsp",
    }
    missing = [k for k,p in mapping.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing TSPLIB files: {', '.join(missing)} in {tsplib_dir}")

    def run_family(inst, wall_override: int|None):
        tsp_path = mapping[inst]
        ins = load_tsplib(str(tsp_path))
        n = int(ins["n"])

        # Use standardized output management
        output_mgr = StandardOutputManager(inst, str(results_dir))

        # Resolve solver params for this n
        resolved_all = autoscale_by_n(n, base_cfg)
        # Override wall_time if requested
        if wall_override is not None:
            resolved_all.setdefault("solver", {}).setdefault("termination", {})
            resolved_all["solver"]["termination"]["wall_time_s"] = float(wall_override)

        # Save configuration with autoscaling info
        instance_cfg = {
            **resolved_all,
            "autoscaling_info": {
                "instance_size_n": n,
                "original_wall_time": wall_override,
                "autoscaler_version": "autoscale_by_n"
            },
            "instance": {"name": inst, "path": str(tsp_path)}
        }
        config_path = output_mgr.save_config(instance_cfg)

        # Fan out seeds in parallel
        seed_results = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, inst, tsp_path, resolved_all, s): s for s in seeds}
            for fut in as_completed(futs):
                s = futs[fut]
                seed, res, rt = fut.result()
                
                # Save individual seed log using standardized format
                anytime_events = [{"event":"improve","t":float(t),"best":float(best)} 
                                  for t, best in res.get("anytime", [])]
                summary_event = {"event":"summary","seed":int(seed),"best":float(res.get("best_length", 0.0)),"runtime":rt}
                all_events = anytime_events + [summary_event]
                
                log_path = output_mgr.save_seed_log(seed, all_events)
                
                seed_results.append({
                    "seed": seed,
                    "best_length": res.get('best_length', 0.0),
                    "runtime": rt,
                    "anytime_count": len(res.get("anytime", []))
                })
                
                print(f"[{inst}] n={n} seed={seed} best={res.get('best_length', 0.0):.1f} time={rt:.1f}s -> {log_path}")
        
        # Save summary results
        summary_data = {
            "instance": inst,
            "instance_path": str(tsp_path),
            "instance_size_n": n,
            "autoscaled_params": {
                "candidate_k": resolved_all.get("solver", {}).get("candidate", {}).get("k", "unknown"),
                "wall_time_s": resolved_all.get("solver", {}).get("termination", {}).get("wall_time_s", "unknown"),
                "agents_per_zone": resolved_all.get("solver", {}).get("zones", {}).get("agents_per_zone", "unknown")
            },
            "seed_results": seed_results,
            "best_overall": min(r["best_length"] for r in seed_results),
            "worst_overall": max(r["best_length"] for r in seed_results),
            "avg_runtime": sum(r["runtime"] for r in seed_results) / len(seed_results),
            "total_seeds": len(seed_results)
        }
        
        results_path = output_mgr.save_results(summary_data)
        print(f"[{inst}] Summary saved to: {results_path}")
        print(f"[{inst}] Config saved to: {config_path}")
        print(f"[{inst}] Run directory: {output_mgr.get_run_directory()}")
        print()

    default_wall = base_cfg.get("solver", {}).get("termination", {}).get("wall_time_s", 1800)
    run_family("fnl4461", args.wall_small if args.wall_small is not None else default_wall)
    run_family("pla7397", args.wall_small if args.wall_small is not None else default_wall)
    run_family("pla85900", args.wall_large if args.wall_large is not None else default_wall)

if __name__ == "__main__":
    main()
