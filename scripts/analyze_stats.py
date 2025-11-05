#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Bee-TSP runs (minimal, robust).

- Accepts either a single run directory (containing seed_*.jsonl) or a results root (recursively scans).
- Reads run summary JSON and per-seed JSONL anytime logs.
- Resolves OPT via:
    1) data/tsplib/<instance>.opt.tour  (reads TOUR_LENGTH if present)
    2) data/best_known.csv  (columns: instance,best_known,op.tour_available)
    3) summary["best_known"]
- Computes per-seed: gap_pct, ttt5_s, ttt10_s, improvements, runtime_s.
- Writes: results/analysis_seeds.csv and results/analysis_instances.csv.
- If pandas unavailable, still writes CSVs with the stdlib csv module.

Usage examples:
  python scripts/analyze_stats.py --results results --tsplib data/tsplib --outdir results
  python scripts/analyze_stats.py --results results/25_11_03_2113_att48 --tsplib data/tsplib --outdir results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import glob
import numpy as np

# Optional (nice-to-have). If missing, we fall back to stdlib CSV writes.
try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


# -------------------------
# Best-known CSV loader
# -------------------------
def load_best_known_csv(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load CSV with columns: instance,best_known,op.tour_available
    Returns: {instance: {"best_known": int|None, "opt_tour_available": bool|None}}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path or not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                inst = (row.get("instance") or "").strip()
                if not inst:
                    continue
                bk_raw = (row.get("best_known") or "").strip()
                try:
                    bk_val = int(bk_raw) if bk_raw != "" else None
                except Exception:
                    bk_val = None
                opt_av_raw = (row.get("op.tour_available") or "").strip().upper()
                if opt_av_raw in ("1", "TRUE", "YES", "Y"):
                    opt_av = True
                elif opt_av_raw in ("0", "FALSE", "NO", "N"):
                    opt_av = False
                else:
                    opt_av = None
                out[inst] = {"best_known": bk_val, "opt_tour_available": opt_av}
    except Exception:
        pass
    return out


# -------------------------
# Run discovery / summary
# -------------------------
def is_run_dir(p: Path) -> bool:
    return p.is_dir() and (any(p.glob("seed_*_*.jsonl")) or any(p.glob("seed_*.jsonl")))

def discover_run_dirs(results_path: Path) -> List[Path]:
    """
    If `results_path` itself is a run dir, return [results_path].
    Else scan recursively for run dirs.
    """
    if results_path.exists() and is_run_dir(results_path):
        return [results_path]

    hits: List[Path] = []
    if results_path.exists() and results_path.is_dir():
        for q in results_path.rglob("*"):
            if is_run_dir(q):
                hits.append(q)
    hits.sort()
    return hits

def find_run_summary(run_dir: Path) -> Optional[Path]:
    """
    Guess the run's summary JSON as the *.json that is NOT a seed log and NOT a manifest.
    Prefer longest filename (your timestamped pattern).
    """
    cands: List[Path] = []
    for p in run_dir.glob("*.json"):
        name = p.name.lower()
        if name.startswith("seed_") or "manifest" in name:
            continue
        cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda x: len(x.name), reverse=True)
    return cands[0]

def list_seed_logs(run_dir: Path) -> List[Path]:
    logs = list(run_dir.glob("seed_*_*.jsonl"))
    if not logs:
        logs = list(run_dir.glob("seed_*.jsonl"))
    logs.sort()
    return logs


# -------------------------
# .opt.tour / OPT helpers
# -------------------------
def read_opt_tour_length(opt_path: Path) -> Optional[int]:
    """
    If .opt.tour has a TOUR_LENGTH header line, read that number and return it.
    Otherwise returns None (we won't recompute length here to avoid extra deps).
    """
    try:
        txt = opt_path.read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            s = line.strip()
            if s.upper().startswith("TOUR_LENGTH"):
                # supports "TOUR_LENGTH: 12345" or "TOUR_LENGTH 12345"
                parts = s.replace(":", " ").split()
                # last token should be the integer
                return int(parts[-1])
    except Exception:
        return None
    return None

def resolve_opt(instance: str,
                tsplib_root: Optional[Path],
                best_known_map: Optional[Dict[str, Dict[str, Any]]],
                summary_best_known: Optional[int]) -> Optional[int]:
    """
    Precedence for OPT:
      1) data/tsplib/<instance>.opt.tour (read TOUR_LENGTH)
      2) data/best_known.csv (best_known column)
      3) summary["best_known"]
      else: None
    """
    # 1) .opt.tour
    if tsplib_root:
        opt_path = tsplib_root / f"{instance}.opt.tour"
        if opt_path.exists():
            L = read_opt_tour_length(opt_path)
            if isinstance(L, int) and L > 0:
                return L

    # 2) best_known.csv
    if best_known_map and instance in best_known_map:
        bk = best_known_map[instance].get("best_known")
        if isinstance(bk, int) and bk > 0:
            return bk

    # 3) summary best_known
    if isinstance(summary_best_known, int) and summary_best_known > 0:
        return summary_best_known

    return None


# -------------------------
# Seed anytime parsing
# -------------------------
def read_seed_anytime_log(log_path: Path) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    """
    Returns (anytime_list, runtime_s) from seed_*.jsonl
    - improve events: {"event":"improve","t":seconds,"best":length}
    - summary row: {"event":"summary","runtime":seconds}
    """
    anytime: List[Tuple[float, float]] = []
    runtime: Optional[float] = None
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                ev = row.get("event")
                if ev == "improve":
                    t = float(row.get("t", 0.0))
                    b = float(row.get("best", 0.0))
                    anytime.append((t, b))
                elif ev == "summary":
                    rt = row.get("runtime")
                    if isinstance(rt, (int, float)):
                        runtime = float(rt)
    except Exception:
        pass
    anytime.sort(key=lambda x: x[0])
    return anytime, runtime

def compute_ttt(anytime: List[Tuple[float, float]], opt_len: int, threshold_pct: float) -> Optional[float]:
    """
    Time-To-Threshold (seconds): first t where best <= (1 + threshold_pct/100) * opt_len.
    Returns None if never achieved or invalid input.
    """
    if not anytime or not isinstance(opt_len, int) or opt_len <= 0:
        return None
    thresh = (1.0 + threshold_pct / 100.0) * float(opt_len)
    for (t, best) in anytime:
        if best <= thresh:
            return float(t)
    return None


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze Bee-TSP outputs into per-seed and per-instance CSVs.")
    ap.add_argument("--results", required=True, help="Run directory OR results root to sweep recursively")
    ap.add_argument("--tsplib", default=None, help="Path to TSPLIB root containing *.tsp / *.opt.tour")
    ap.add_argument("--best-known-csv", default="data/best_known.csv", help="CSV: instance,best_known,op.tour_available")
    ap.add_argument("--outdir", default="results", help="Output directory for analysis CSVs")
    args = ap.parse_args()

    results_root = Path(args.results).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tsplib_root = Path(args.tsplib).resolve() if args.tsplib else None
    bk_map = load_best_known_csv(Path(args.best_known_csv).resolve() if args.best_known_csv else None)

    run_dirs = discover_run_dirs(results_root)
    if not run_dirs:
        print(f"[WARN] No runs found under: {results_root}")
        return

    seed_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        summary_path = find_run_summary(run_dir)
        if summary_path and summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        else:
            summary = {}

        # Instance name: prefer summary, else guess from folder name
        instance = str(summary.get("instance") or run_dir.name.split("_", 2)[-1]).strip()
        summary_best_known = summary.get("best_known", None)

        # Decide OPT once for the run
        opt_len = resolve_opt(instance, tsplib_root, bk_map, summary_best_known)

        # Map seed -> (best_length, runtime) from summary if present
        summary_seeds: Dict[int, Dict[str, Any]] = {}
        for sr in summary.get("seed_results", []):
            try:
                s = int(sr.get("seed"))
            except Exception:
                continue
            summary_seeds[s] = {
                "best_length": sr.get("best_length"),
                "runtime": sr.get("runtime"),
            }

        # Parse each seed log
        for slog in list_seed_logs(run_dir):
            # seed id from filename: seed_<SEED>_...
            try:
                parts = slog.stem.split("_")
                seed_val = int(parts[1])
            except Exception:
                seed_val = -1

            anytime, runtime_log = read_seed_anytime_log(slog)
            sr = summary_seeds.get(seed_val, {})
            best_len = sr.get("best_length", None)
            runtime = sr.get("runtime", runtime_log)

            # Count improvements
            improvements = 0
            for ev in anytime:
                # we already only kept "improve" events; length = improvements
                improvements += 1
            # (If logs can contain other events, we could re-read raw JSON lines, but this is fine.)

            # ttt
            ttt5 = compute_ttt(anytime, int(opt_len), 5) if isinstance(opt_len, int) else None
            ttt10 = compute_ttt(anytime, int(opt_len), 10) if isinstance(opt_len, int) else None

            # gap
            if isinstance(best_len, (int, float)) and isinstance(opt_len, int) and opt_len > 0:
                gap_pct = (float(best_len) / float(opt_len) - 1.0) * 100.0
            else:
                gap_pct = np.nan

            seed_rows.append({
                "instance": instance,
                "seed": int(seed_val),
                "best_length": int(best_len) if isinstance(best_len, (int, float)) else np.nan,
                "opt": int(opt_len) if isinstance(opt_len, int) else np.nan,
                "gap_pct": float(gap_pct) if isinstance(gap_pct, (int, float)) else np.nan,
                "ttt10_s": float(ttt10) if isinstance(ttt10, (int, float)) else np.nan,
                "ttt5_s": float(ttt5) if isinstance(ttt5, (int, float)) else np.nan,
                "runtime_s": float(runtime) if isinstance(runtime, (int, float)) else np.nan,
                "improvements": int(improvements),
                "run_dir": str(run_dir),
            })

    # ---- Write analysis_seeds.csv ----
    seeds_csv = outdir / "analysis_seeds.csv"
    if pd is not None:
        df = pd.DataFrame(seed_rows)
        df.to_csv(seeds_csv, index=False)
        print(f"[WRITE] {seeds_csv}")
    else:
        if seed_rows:
            keys = list(seed_rows[0].keys())
            with seeds_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in seed_rows:
                    w.writerow(r)
        print(f"[WRITE] {seeds_csv} (wrote via stdlib; pandas not installed)")

    # ---- Aggregate per-instance and write analysis_instances.csv ----
    def _agg_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def col(vals, key):
            xs = [v.get(key) for v in vals]
            ys = [float(z) for z in xs if isinstance(z, (int, float)) and math.isfinite(float(z))]
            return ys

        gaps = col(rows, "gap_pct")
        t10  = col(rows, "ttt10_s")
        t5   = col(rows, "ttt5_s")

        agg = {
            "instance": rows[0]["instance"],
            "seeds": len(rows),
            "gap_min":  float(np.min(gaps)) if gaps else np.nan,
            "gap_med":  float(np.median(gaps)) if gaps else np.nan,
            "gap_max":  float(np.max(gaps)) if gaps else np.nan,
            "ttt10_min": float(np.min(t10)) if t10 else np.nan,
            "ttt10_med": float(np.median(t10)) if t10 else np.nan,
            "ttt10_max": float(np.max(t10)) if t10 else np.nan,
            "ttt5_min":  float(np.min(t5)) if t5 else np.nan,
            "ttt5_med":  float(np.median(t5)) if t5 else np.nan,
            "ttt5_max":  float(np.max(t5)) if t5 else np.nan,
        }
        return agg

    # group by instance
    by_inst: Dict[str, List[Dict[str, Any]]] = {}
    for r in seed_rows:
        by_inst.setdefault(r["instance"], []).append(r)

    inst_rows: List[Dict[str, Any]] = []
    for inst, rows in sorted(by_inst.items()):
        inst_rows.append(_agg_rows(rows))

    inst_csv = outdir / "analysis_instances.csv"
    if pd is not None:
        pd.DataFrame(inst_rows).to_csv(inst_csv, index=False)
        print(f"[WRITE] {inst_csv}")
    else:
        if inst_rows:
            keys = list(inst_rows[0].keys())
            with inst_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in inst_rows:
                    w.writerow(r)
        print(f"[WRITE] {inst_csv} (wrote via stdlib; pandas not installed)")

    # ---- Pretty console summaries (like your existing style) ----
    def fmt(x):
        if x is None:
            return "NA"
        try:
            xf = float(x)
            if not math.isfinite(xf):
                return "NA"
            return f"{xf:.1f}"
        except Exception:
            return "NA"

    for row in inst_rows:
        inst = row["instance"]
        seeds = row["seeds"]
        gmin, gmed, gmax = row["gap_min"], row["gap_med"], row["gap_max"]
        t10 = (row["ttt10_min"], row["ttt10_med"], row["ttt10_max"])
        t5  = (row["ttt5_min"],  row["ttt5_med"],  row["ttt5_max"])
        print(f"{inst}: seeds={seeds}  gap% min/med/max= {fmt(gmin)}/{fmt(gmed)}/{fmt(gmax)} ;  "
              f"TTT10s min/med/max= {fmt(t10[0])}/{fmt(t10[1])}/{fmt(t10[2])} ;  "
              f"TTT5s min/med/max= {fmt(t5[0])}/{fmt(t5[1])}/{fmt(t5[2])}")


if __name__ == "__main__":
    main()
