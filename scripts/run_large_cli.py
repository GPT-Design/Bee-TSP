#!/usr/bin/env python3
"""
Run selected large TSPLIB instances with size-based budgets and timestamped outputs.
It stages each instance into its own temp dir and invokes run_auto_parallel.py.

Usage (from repo root):
  python scripts/run_large_cli.py --instances pr2392 fnl4461 pla33810 \
    --tsplib_dir data/tsplib --config configs/params_z3000.yaml --seeds 42 123 456 \
    --workers 4 --results_dir results --tag auto
"""
import argparse, datetime, os, shutil, subprocess, sys, re
from pathlib import Path

def ts_now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def read_dimension(tsp_path: Path) -> int:
    # Fast parse of "DIMENSION : N" from .tsp; fallback 0 if missing.
    try:
        with open(tsp_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "DIMENSION" in line.upper():
                    m = re.search(r"(\d+)", line)
                    if m: return int(m.group(1))
    except Exception:
        pass
    return 0

def default_budget_for_n(n: int, m: int, l: int) -> int:
    # Map size to seconds: mid / large / xl (defaults 120/300/600)
    if n <= 5000:   return m   # e.g., pr2392
    if n <= 50000:  return l   # e.g., fnl4461, pla33810
    return max(l, 2*l)         # headroom for very large sets

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=["pr2392","fnl4461","pla33810"])
    p.add_argument("--tsplib_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", nargs="+", default=["42","123","456","789","999"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--tag", default="auto", help='"auto" → timestamp')
    # size-tier budgets (seconds)
    p.add_argument("--budget_mid",  type=int, default=120)  # ~1–2 min
    p.add_argument("--budget_large",type=int, default=300)  # ~5 min
    # extra: forward extra args to run_auto_parallel.py verbatim
    p.add_argument("--runner_extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    tag = ts_now() if args.tag == "auto" else args.tag
    results_root = Path(args.results_dir) / tag
    stage_root   = results_root / "tsplib_stage"
    results_root.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    runner = Path("test_all_three_instances.py")
    if not runner.exists():
        print(f"[ERROR] Missing {runner}.", file=sys.stderr); sys.exit(2)

    for inst in args.instances:
        tsp_src = Path(args.tsplib_dir) / f"{inst}.tsp"
        if not tsp_src.exists():
            print(f"[WARN] Missing {tsp_src}, skipping."); continue

        # stage this instance (copy .tsp and any .opt.tour if present)
        inst_stage = stage_root / inst
        inst_stage.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tsp_src, inst_stage / tsp_src.name)
        opt_src = tsp_src.with_suffix(".opt.tour")
        if opt_src.exists():
            shutil.copy2(opt_src, inst_stage / opt_src.name)

        n = read_dimension(tsp_src)
        wall = default_budget_for_n(n, args.budget_mid, args.budget_large)
        # call runner once per instance with a unique sub-tag
        subtag = f"{tag}-{inst}"
        log = (results_root / f"run_{inst}_{subtag}.log").as_posix()

        # Determine candidate k based on instance size
        if n <= 1000:
            k_val = 30
        elif n <= 5000:
            k_val = 25  
        else:
            k_val = 20
            
        # Calculate agent time based on wall time
        agent_time = min(wall * 0.2, 15.0)  # 20% of wall time, max 15s
        
        cmd = [
            py, str(runner),
            "--time-budget", str(wall),
            "--candidate-k", str(k_val),
            "--agent-time", str(agent_time),
            "--seeds", *args.seeds,
            "--output", str(results_root / f"{subtag}_results.json")
        ] + args.runner_extra

        print(f"[RUN] {inst} (n={n}) wall={wall}s -> tag={subtag}")
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            rc = proc.wait()
        if rc != 0:
            print(f"[ERROR] Runner failed for {inst} (rc={rc}). See {log}")
        else:
            print(f"[OK] {inst} complete. Log -> {log}")

    print(f"\nAll done. Results -> {results_root}")

if __name__ == "__main__":
    main()