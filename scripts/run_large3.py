#!/usr/bin/env python3
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

# Try importing BeeTSPSolver from different solver names
try:
    from bee_tsp.solver import BeeTSPSolver
except Exception:
    # optional fallback if user kept solver_large_patch.py
    try:
        from bee_tsp.solver_large_patch import BeeTSPSolver  # type: ignore
    except Exception as e:
        raise

def load_base_config():
    cfg_path = Path("configs/params.yaml")
    if yaml is not None and cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        return base.get("solver", {})
    return {
        "candidate": {"k": 40, "use_delaunay": False, "use_knn": True, "use_alpha_near": False},
        "zones": {"method": "kmeans", "overlap_pct": 0.10, "target_zone_size": 300, "agents_per_zone": 6},
        "bees": {"scout_fraction": 0.10, "stagnation_moves": 500, "levy_jump_prob": 0.05, "max_improving_moves": 1200, "time_budget_s": 1.0},
        "local_search": {"use_lk": False, "max_k": 3, "double_bridge": True, "kick_period_moves": 500},
        "integrator": {"mode": "popmusic", "merge_edge_cap_per_node": 10, "parents_per_round": 8, "rounds": 2},
        "ehm": {"smoothing_eps": 0.02, "max_edges_per_node": 16},
        "learning": {"enabled": False, "warmup_tours": 100, "model": "gbdt", "feature_set": ["distance","delaunay_adj","degree"]},
        "termination": {"wall_time_s": 1800, "target_gap_pct": 1.0},
    }

def run_instance(inst_name, tsp_path, solver_cfg, results_dir, seeds, wall_time):
    out_dir = Path(results_dir) / f"bstsp_large3/{inst_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(solver_cfg)
    cfg["termination"]["wall_time_s"] = float(wall_time)
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
        print(f"[{inst_name}] seed={seed} best={res.get('best_length', 0.0):.1f} time={rt:.1f}s  -> {log}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsplib_dir", default="data/tsplib")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--wall_small", type=int, default=1800)
    ap.add_argument("--wall_large", type=int, default=7200)
    args = ap.parse_args()

    base_cfg = load_base_config()
    tsplib = Path(args.tsplib_dir)
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
