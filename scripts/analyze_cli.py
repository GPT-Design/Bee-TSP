#!/usr/bin/env python3
import argparse, os, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> project root
AN = ROOT / "analyze_results.py"
CFG = ROOT / "configs" / "params.yaml"
BEST_DEFAULT = ROOT / "benchmarks" / "best_known.csv"

def guess_defaults():
    results = "results"
    algo = "bstsp"
    best = None
    out = "results/summary/results_summary.csv"
    if CFG.exists():
        try:
            import yaml
            cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
            results = cfg.get("paths",{}).get("results_dir", results)
            algo = cfg.get("benchmark",{}).get("algo", algo) or "bstsp"
            ttt = cfg.get("benchmark",{}).get("ttt_thresholds_pct", None)
        except Exception:
            pass
    if BEST_DEFAULT.exists():
        best = str(BEST_DEFAULT)
    return results, algo, best, out

def main():
    ap = argparse.ArgumentParser(description="Convenience CLI for analyze_results.py")
    ap.add_argument("--results", default=None)
    ap.add_argument("--algo", default=None)
    ap.add_argument("--best", default=None, help="CSV with columns: instance,best_known (optional)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not AN.exists():
        print(f"ERROR: analyze_results.py not found at {AN}", file=sys.stderr)
        sys.exit(1)

    results, algo, best, out = guess_defaults()
    if args.results: results = args.results
    if args.algo:    algo = args.algo
    if args.best is not None: best = args.best  # allow empty to disable
    if args.out:     out = args.out

    out_path = ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(AN), "--results", results, "--algo", algo, "--out", str(out)]
    if best:
        if Path(best).exists():
            cmd += ["--best", best]
        else:
            print(f"WARNING: best-known file not found: {best} (continuing without it)", file=sys.stderr)

    env = os.environ.copy()
    # Ensure project root importable for PyYAML if needed by analyze_results.py (not required though)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    print("Running:", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
