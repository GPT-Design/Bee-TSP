#!/usr/bin/env python3
# scripts/analyze_results.py
from __future__ import annotations
import argparse, json, sys, os, math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] manifest parse error line {ln}: {e}", file=sys.stderr)
    return rows

def read_anytime_log(jsonl_path: Path) -> Tuple[List[Tuple[float,float]], Optional[Dict[str,Any]]]:
    """Return (improve_events [(t,best)], summary_event or None)."""
    imp, summary = [], None
    if not jsonl_path.exists(): return imp, summary
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("event") == "improve":
                t = float(ev.get("t", 0.0))
                b = float(ev.get("best", float("inf")))
                imp.append((t, b))
            elif ev.get("event") == "summary":
                summary = ev
    return imp, summary

def time_to_target(improve_events: List[Tuple[float,float]],
                   target_len: float) -> Optional[float]:
    """First t where best<=target_len (seconds)."""
    if not improve_events: return None
    t0 = 0.0
    # events store seconds already in your logs
    best_seen = float("inf")
    tt = None
    for t,b in improve_events:
        best_seen = min(best_seen, b)
        if best_seen <= target_len:
            tt = t
            break
    return tt

def safe_float(x, default=None):
    try: return float(x)
    except Exception: return default

def compute_row_metrics(row: Dict[str,Any], logs_dir: Optional[Path]) -> Dict[str,Any]:
    """Derive TTT@{10,5}% and gap% if possible."""
    out = {}
    inst     = row.get("instance")
    seed     = row.get("seed")
    best_L   = safe_float(row.get("length_tsplib"))
    opt      = safe_float(row.get("opt"))
    bks      = safe_float(row.get("best_known"))
    elapsed  = safe_float(row.get("elapsed_s"))
    tourpath = row.get("tour_path")

    # choose target baseline: prefer opt, else best_known, else None
    baseline = opt if (opt is not None) else (bks if bks is not None else None)

    # derive log path heuristically from tour_path or results dir
    # e.g., .../<run_dir>/seed_<seed>_YY_MM_DD_HHMM.jsonl  OR seed_<seed>.jsonl
    seed_log = None
    if tourpath:
        p = Path(tourpath).with_suffix(".jsonl")  # same stem, .jsonl
        if p.exists(): seed_log = p
        else:
            # try sibling name seed_<seed>.jsonl in same dir
            cand = p.parent / f"seed_{seed}.jsonl"
            if cand.exists(): seed_log = cand
    if not seed_log and logs_dir:
        cand = logs_dir / f"seed_{seed}.jsonl"
        if cand.exists(): seed_log = cand

    # read anytime for TTT
    ttt10 = ttt5 = None
    if baseline is not None and seed_log and seed_log.exists():
        improves, _summary = read_anytime_log(seed_log)
        ttt10 = time_to_target(improves, baseline * 1.10)
        ttt5  = time_to_target(improves, baseline * 1.05)

    # gap%
    gap_pct = None
    if baseline is not None and best_L is not None:
        gap_pct = 100.0 * (best_L - baseline) / baseline

    out.update({
        "instance": inst,
        "seed": seed,
        "metric": row.get("metric"),
        "mode": row.get("mode"),
        "length_tsplib": best_L,
        "opt": opt,
        "best_known": bks,
        "gap_pct": gap_pct,
        "elapsed_s": elapsed,
        "ttt10_s": ttt10,
        "ttt5_s": ttt5,
        "tour_path": tourpath,
        "tour_sha256": row.get("tour_sha256"),
        "timestamp_utc": row.get("timestamp_utc"),
    })
    return out

def write_csv(rows: List[Dict[str,Any]], path: Path) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("instance,seed,metric,mode,length_tsplib,opt,best_known,gap_pct,elapsed_s,ttt10_s,ttt5_s,tour_path,tour_sha256,timestamp_utc\n")
        return
    cols = ["instance","seed","metric","mode","length_tsplib","opt","best_known","gap_pct",
            "elapsed_s","ttt10_s","ttt5_s","tour_path","tour_sha256","timestamp_utc"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if v is None: v = ""
                if isinstance(v, str):
                    # quote if comma inside
                    if "," in v or '"' in v:
                        v = '"' + v.replace('"','""') + '"'
                vals.append(str(v))
            f.write(",".join(vals) + "\n")

def summarize(rows: List[Dict[str,Any]]) -> str:
    by_inst: Dict[str, List[Dict[str,Any]]] = {}
    for r in rows:
        by_inst.setdefault(str(r.get("instance")), []).append(r)
    lines = []
    for inst, rs in sorted(by_inst.items()):
        n = len(rs)
        gaps = [r["gap_pct"] for r in rs if isinstance(r.get("gap_pct"), (int,float))]
        t10  = [r["ttt10_s"] for r in rs if isinstance(r.get("ttt10_s"), (int,float))]
        t5   = [r["ttt5_s"]  for r in rs if isinstance(r.get("ttt5_s"), (int,float))]
        def _stat(x): 
            if not x: return ("NA","NA","NA")
            xs = sorted(float(v) for v in x)
            med = xs[len(xs)//2]
            return (f"{xs[0]:.3g}", f"{med:.3g}", f"{xs[-1]:.3g}")
        gmin,gmed,gmax = _stat(gaps)
        t10min,t10med,t10max = _stat(t10)
        t5min,t5med,t5max    = _stat(t5)
        lines.append(f"{inst}: seeds={n}  gap% min/med/max= {gmin}/{gmed}/{gmax} ;  TTT10s min/med/max= {t10min}/{t10med}/{t10max} ;  TTT5s min/med/max= {t5min}/{t5med}/{t5max}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Analyze Bee-TSP results (TTT, gaps) from manifest.jsonl.")
    ap.add_argument("--manifest", default="results/manifest.jsonl", help="Path to manifest.jsonl")
    ap.add_argument("--results-root", default="results", help="Results root (to help locate seed logs)")
    ap.add_argument("--csv-out", default="results/analysis.csv", help="CSV output path")
    ap.add_argument("--txt-out", default="results/analysis.txt", help="Text summary output path")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rows = load_manifest(manifest)

    res_root = Path(args.results_root)
    analyzed: List[Dict[str,Any]] = []
    for row in rows:
        # only trust rows with a tour (reproducibility rule)
        if not row.get("tour_path") or not row.get("tour_sha256"):
            continue
        # attempt to infer the run directory to find seed logs
        tour_p = Path(row["tour_path"])
        logs_dir = tour_p.parent if tour_p.exists() else res_root
        analyzed.append(compute_row_metrics(row, logs_dir))

    out_csv = Path(args.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(analyzed, out_csv)

    out_txt = Path(args.txt_out)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summarize(analyzed) + "\n")

    print(f"[ANALYZE] Wrote {len(analyzed)} rows to {out_csv}")
    print(f"[ANALYZE] Summary -> {out_txt}")

if __name__ == "__main__":
    main()
