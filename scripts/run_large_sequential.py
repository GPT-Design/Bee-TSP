#!/usr/bin/env python3
"""
Sequential runner for large instances with size-based budgets and timestamped outputs.
Uses our working single instance test harness.
"""
import argparse, datetime, subprocess, sys, re
from pathlib import Path

def ts_now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def read_dimension(tsp_path: Path) -> int:
    """Fast parse of DIMENSION from .tsp file"""
    try:
        with open(tsp_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "DIMENSION" in line.upper():
                    m = re.search(r"(\d+)", line)
                    if m: return int(m.group(1))
    except Exception:
        pass
    return 0

def get_size_params(n: int):
    """Get size-appropriate parameters"""
    if n <= 1000:
        return {"k": 30, "agent_time": 6.0, "budget": 120}
    elif n <= 3000:
        return {"k": 25, "agent_time": 8.0, "budget": 180}  
    elif n <= 10000:
        return {"k": 20, "agent_time": 10.0, "budget": 300}
    else:
        return {"k": 15, "agent_time": 12.0, "budget": 600}

def main():
    p = argparse.ArgumentParser(description="Run large instances sequentially")
    p.add_argument("--instances", nargs="+", default=["pr2392", "fnl4461", "pla33810"])
    p.add_argument("--tsplib_dir", default="data/tsplib")
    p.add_argument("--seeds", nargs="+", default=["42", "123"])
    p.add_argument("--results_dir", default="results")
    p.add_argument("--tag", default="auto", help='"auto" -> timestamp')
    p.add_argument("--budget_override", type=int, help="Override time budget for all instances")
    p.add_argument("--k_override", type=int, help="Override candidate k for all instances")
    
    args = p.parse_args()
    
    # Create timestamped results directory
    tag = ts_now() if args.tag == "auto" else args.tag
    results_root = Path(args.results_dir) / f"large_{tag}"
    results_root.mkdir(parents=True, exist_ok=True)
    
    py = sys.executable
    runner = Path("scripts") / "test_single_large.py"
    
    if not runner.exists():
        print(f"[ERROR] Missing {runner}")
        sys.exit(1)
    
    print(f"=== Large Instance Sequential Runner ===")
    print(f"Tag: {tag}")
    print(f"Results directory: {results_root}")
    print(f"Instances: {args.instances}")
    print(f"Seeds: {args.seeds}")
    print()
    
    for inst in args.instances:
        print(f"[START] {inst}")
        
        # Check if instance file exists
        tsp_path = Path(args.tsplib_dir) / f"{inst}.tsp"
        if not tsp_path.exists():
            print(f"[SKIP] Missing {tsp_path}")
            continue
            
        # Get instance size and appropriate parameters
        n = read_dimension(tsp_path)
        params = get_size_params(n)
        
        # Apply overrides if provided
        if args.budget_override:
            params["budget"] = args.budget_override
        if args.k_override:
            params["k"] = args.k_override
            
        # Build output path
        output_file = results_root / f"{inst}_{tag}_results.json"
        
        # Build command
        cmd = [
            py, str(runner),
            "--instance", inst,
            "--time-budget", str(params["budget"]),
            "--candidate-k", str(params["k"]),
            "--agent-time", str(params["agent_time"]),
            "--seeds"] + args.seeds + [
            "--output", str(output_file)
        ]
        
        print(f"[RUN] {inst} (n={n}) budget={params['budget']}s k={params['k']} agent_time={params['agent_time']}s")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=params["budget"] + 60)
            
            if result.returncode == 0:
                print(f"[OK] {inst} completed successfully")
                # Extract summary from output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "Best gap:" in line or "Runtime:" in line or "Parity:" in line:
                        print(f"     {line.split('=== SUMMARY ===')[-1].strip()}")
            else:
                print(f"[ERROR] {inst} failed (rc={result.returncode})")
                print(f"STDERR: {result.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {inst} exceeded {params['budget'] + 60}s")
        except Exception as e:
            print(f"[EXCEPTION] {inst} failed: {e}")
            
        print()
    
    print(f"All done. Results in: {results_root}")
    
    # List result files
    result_files = list(results_root.glob("*.json"))
    if result_files:
        print(f"Result files created: {len(result_files)}")
        for f in sorted(result_files):
            print(f"  - {f.name}")

if __name__ == "__main__":
    main()