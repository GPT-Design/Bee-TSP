#!/usr/bin/env python3
from email import parser
import argparse, os, time, json, copy, sys
from pathlib import Path

# Ensure project root is importable when running from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import yaml
except Exception:
    yaml = None

from bee_tsp.solver import BeeTSPSolver
from scripts.output_app import StandardOutputManager

def read_tsplib_metric(tsp_path):
    """Read EDGE_WEIGHT_TYPE from TSPLIB file. Returns 'EUC_2D'|'CEIL_2D'|'ATT'|'GEO'"""
    with open(tsp_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('EDGE_WEIGHT_TYPE'):
                return line.split(':', 1)[1].strip().upper()
            if line.startswith('EOF'):
                break
    return 'EUC_2D'  # Default fallback

def load_base_config(config_path="configs/params.yaml"):
    cfg_path = Path(config_path)
    if yaml is not None and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        return base.get("solver", {})
            
    print("[CONFIG] WARNING: Using fallback config (YAML not loaded)")
    return {
        "candidate": {"k": 40, "use_delaunay": False, "use_knn": True, "use_alpha_near": False},
        "zones": {"method": "kmeans", "overlap_pct": 0.10, "target_zone_size": 300, "agents_per_zone": 6},
        "bees": {"scout_fraction": 0.10, "stagnation_moves": 500, "levy_jump_prob": 0.05, "max_improving_moves": 1200, "time_budget_s": 1.0},
        "local_search": {"use_lk": False, "max_k": 3, "double_bridge": True, "kick_period_moves": 500},
        "integrator": {"method": "popmusic", "merge_edge_cap_per_node": 10, "parents_per_round": 8, "rounds": 2},
        "ehm": {"smoothing_eps": 0.02, "max_edges_per_node": 16},
        "learning": {"enabled": False, "warmup_tours": 100, "model": "gbdt", "feature_set": ["distance","delaunay_adj","degree"]},
        "termination": {"wall_time_s": 1800, "target_gap_pct": 1.0},
    }

def run_instance(inst_name, tsp_path, solver_cfg, results_dir, seeds, wall_time, mode="SH"):
    """Run instance with standardized output management."""
    # Use standardized output management
    output_mgr = StandardOutputManager(inst_name, results_dir)

    # METRIC LOCK: Read EDGE_WEIGHT_TYPE from TSP file at startup (Update_24R fix)
    metric_tag = read_tsplib_metric(tsp_path)
    print(f"[METRIC_LOCK] {inst_name}: EDGE_WEIGHT_TYPE = {metric_tag} (from {tsp_path})")

    # Prepare solver config
    cfg = copy.deepcopy(solver_cfg)
    cfg["metric_tag"] = metric_tag  # Inject metric into solver config
    if "termination" not in cfg:
        cfg["termination"] = {}
    cfg["termination"]["wall_time_s"] = float(wall_time)
    
    # Save configuration for this instance
    instance_cfg = {
        "solver": cfg,
        "instance": {"name": inst_name, "path": str(tsp_path)},
        "run_parameters": {
            "seeds": seeds,
            "wall_time_s": wall_time,
            "mode": mode
        }
    }
    config_path = output_mgr.save_config(instance_cfg)
    
    solver = BeeTSPSolver(cfg)
    seed_results = []
    
    for seed in seeds:
        t0 = time.time()
        res = solver.solve(inst_name, str(tsp_path), seed=seed)
        rt = time.time() - t0
        
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
        
        print(f"[{inst_name}] seed={seed} best={res.get('best_length', 0.0):.1f} time={rt:.1f}s  -> {log_path}")
    
    # Save summary results
    summary_data = {
        "instance": inst_name,
        "instance_path": str(tsp_path),
        "wall_time_s": wall_time,
        "seed_results": seed_results,
        "best_overall": min(r["best_length"] for r in seed_results),
        "worst_overall": max(r["best_length"] for r in seed_results),
        "avg_runtime": sum(r["runtime"] for r in seed_results) / len(seed_results),
        "total_seeds": len(seed_results)
    }
    
    results_path = output_mgr.save_results(summary_data)

    # Append each seed result to global manifest
    print(f"[{inst_name}] Writing {len(seed_results)} entries to global manifest...")
    for seed_result in seed_results:
        # Create run data for manifest entry
        run_data = {
            "instance": inst_name,
            "metric": "tour_length",
            "mode": mode,
            "seed": seed_result["seed"],
            "wall_s": wall_time,
            "elapsed_s": seed_result["runtime"],
            "epochs_completed": 0,  # Not tracked in current implementation
            "epoch_s": 0,
            "tour_path": None,  # Would need tour file path
            "tour_sha256": None,
            "length_tsplib": seed_result["best_length"],
            "opt": 11849233.0 if inst_name == "pla33810" else None,  # Known optimal for pla33810
            "best_known": 11849233.0 if inst_name == "pla33810" else None,
            "parity": 0.0,
            "ttt10_s": None,  # Time-to-target metrics not available
            "ttt5_s": None,
            "early_slope": None,
            "improvements_per_min": None,
            "hive_attrib": {},
            "timestamp_utc": output_mgr.timestamp.isoformat()
        }

        # Create and append manifest entry
        manifest_entry = output_mgr.create_manifest_entry(run_data)
        output_mgr.append_to_global_manifest(manifest_entry)

    print(f"[{inst_name}] Manifest entries written to: {output_mgr.global_manifest_path}")
    print(f"[{inst_name}] Summary saved to: {results_path}")
    print(f"[{inst_name}] Config saved to: {config_path}")
    print(f"[{inst_name}] Run directory: {output_mgr.get_run_directory()}")
    return output_mgr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsplib_dir", default="data/tsplib")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds or single seed count")
    parser.add_argument("--wall_small", type=int, default=1800)
    parser.add_argument("--wall_large", type=int, default=7200)
    
    # New arguments to support specific instance runs
    parser.add_argument("--instance", help="Specific instance to run (e.g., pla33810)")
    parser.add_argument("--wall", type=int, help="Wall time for specific instance run")
    parser.add_argument("--mode", default="SH", help="Mode: SH (production), MH-Lite (experimental), MH-Full (experimental)")
    parser.add_argument("--seeds_list", nargs="+", type=int, help="Specific seeds list (e.g., 42 123 456)")
    parser.add_argument("--config", default="configs/params.yaml", help="Config file path for ablation testing")

    # NEW: Integrator override
    parser.add_argument("--integ", type=str, default=None, choices=["popmusic", "eax", "ucr"], help="Override integrator method (popmusic, eax, or ucr)")
     
    # NEW: ΔΦ toggle
    parser.add_argument("--delphi", type=str, default=None, choices=["on", "off"], help="Override ΔΦ enabled status (on or off)")

    # NEW: Adaptive toggle
    parser.add_argument("--adaptive", type=str, default=None, choices=["on", "off"], help="Enable adaptive parameter selection")
                    
    args = parser.parse_args()
    base_cfg = load_base_config(args.config)
    # CLI OVERRIDES
    if args.integ is not None:
        if 'integrator' not in base_cfg:
            base_cfg['integrator'] = {}
        integ_map = {'popmusic': 'popmusic', 'eax': 'eax', 'ucr': 'ucr'}
        base_cfg['integrator']['method'] = integ_map[args.integ]
        print(f"[CLI] Integrator override: {args.integ} -> {integ_map[args.integ]}")

    if args.delphi is not None:
        if 'delta_phi' not in base_cfg:
            base_cfg['delta_phi'] = {}
        base_cfg['delta_phi']['enabled'] = (args.delphi == 'on')
        print(f"[CLI] ΔΦ override: {args.delphi}")

    if args.adaptive is not None:
        if 'adaptive' not in base_cfg:
            base_cfg['adaptive'] = {}
        base_cfg['adaptive']['enabled'] = (args.adaptive == 'on')
        print(f"[CLI] Adaptive mode: {args.adaptive}")
    
    tsplib = Path(args.tsplib_dir)
    
    # Handle specific instance run vs. default batch run
    if args.instance:
        # Single instance mode
        instance_path = tsplib / f"{args.instance}.tsp"
        if not instance_path.exists():
            raise SystemExit(f"Missing TSPLIB file: {instance_path}")
        
        wall_time = args.wall if args.wall else (args.wall_large if args.instance == "pla85900" else args.wall_small)
        
        if args.seeds_list:
            seeds = args.seeds_list
        else:
            seeds = list(range(args.seeds))
        
        print(f"Running {args.instance} with mode={args.mode}, wall_time={wall_time}s, seeds={seeds}")
        run_instance(args.instance, instance_path, base_cfg, args.results_dir, seeds, wall_time, args.mode)
    else:
        # Default batch mode - run all three large instances
        mapping = {
            "fnl4461": tsplib / "fnl4461.tsp",
            "pla7397": tsplib / "pla7397.tsp",
            "pla85900": tsplib / "pla85900.tsp",
        }
        missing = [k for k,p in mapping.items() if not p.exists()]
        if missing:
            raise SystemExit(f"Missing TSPLIB files: {', '.join(missing)} in {tsplib}")
        seeds = list(range(args.seeds))
        for inst in ["fnl4461", "pla7397"]:
            run_instance(inst, mapping[inst], base_cfg, args.results_dir, seeds, args.wall_small)
        run_instance("pla85900", mapping["pla85900"], base_cfg, args.results_dir, seeds, args.wall_large)

if __name__ == "__main__":
    main()
