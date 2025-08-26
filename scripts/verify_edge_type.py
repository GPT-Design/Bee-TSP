#!/usr/bin/env python3
import argparse, os, sys, math, csv
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from bee_tsp.tsplib import load_tsplib
except Exception as e:
    print("ERROR: failed to import bee_tsp.tsplib.load_tsplib:", e, file=sys.stderr)
    sys.exit(2)

def read_best_from_csv(csv_path: Path, name: str):
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("instance","").strip().lower() == name.lower():
                    return int(float(row["best_known"]))
    except Exception as e:
        print(f"WARNING: failed to read {csv_path}: {e}", file=sys.stderr)
    return None

def parse_tour(path: Path):
    # TSPLIB .tour: read nodes between TOUR_SECTION and -1
    nodes = []
    in_sec = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            if ln.startswith("TOUR_SECTION"):
                in_sec = True; continue
            if not in_sec: continue
            if ln in ("-1","EOF"): break
            for tok in ln.split():
                if tok == "-1": in_sec = False; break
                nodes.append(int(tok))
            if not in_sec: break
    if not nodes:
        raise ValueError("No tour nodes parsed from TOUR_SECTION")
    return [v-1 for v in nodes]  # TSPLIB nodes are 1-based

def tour_length(dist, tour):
    s = 0
    n = len(tour)
    for i in range(n):
        a = tour[i]; b = tour[(i+1)%n]
        s += dist(a,b)
    return int(s)

def main():
    ap = argparse.ArgumentParser(description="Verify EDGE_WEIGHT_TYPE detection and (optionally) tour length.")
    ap.add_argument("--tsp", required=True, nargs="+", help="Path(s) to .tsp")
    ap.add_argument("--tour", help="Path to .opt.tour (optional)")
    ap.add_argument("--best", type=int, help="Known optimal/best length (optional if --bestcsv provided)")
    ap.add_argument("--bestcsv", help="CSV with columns instance,best_known")
    args = ap.parse_args()

    for tsp_path in args.tsp:
        tsp = Path(tsp_path)
        prob = load_tsplib(str(tsp))
        name = tsp.stem
        print(f"\n{name}: n={prob['n']} edge_weight_type={prob['edge_weight_type']}")

        if args.tour and len(args.tsp) > 1:
            print("NOTE: --tour provided but multiple TSPs given; the tour will be used only for the first TSP.")
            args.tsp = [tsp_path]

        if args.tour:
            tour = parse_tour(Path(args.tour))
            L = tour_length(prob["dist"], tour)
            target = None
            if args.best is not None:
                target = int(args.best)
            elif args.bestcsv:
                target = read_best_from_csv(Path(args.bestcsv), name)
            print(f"  tour length = {L}")
            if target is not None:
                gap = 100.0 * (L - target) / target if target > 0 else float('nan')
                print(f"  target (best_known) = {target}  gap = {gap:.3f}%")
            else:
                print("  (no target provided; use --best or --bestcsv to compare)")

if __name__ == "__main__":
    main()
