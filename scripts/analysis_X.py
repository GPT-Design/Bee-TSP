#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick stats (hard-coded OPT table, minimal deps).

- Scans results/ (or a specific run dir) for run folders.
- Reads summary JSON + seed_*.jsonl, computes gaps using TSPLIB_OPT only.
- Prints per-instance median gap with bootstrap 95% CI.
- Optional pairwise comparisons (Mann–Whitney if SciPy is present).
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Optional SciPy (for Mann–Whitney). If missing, we just skip p-values.
try:
    from scipy.stats import mannwhitneyu  # type: ignore
except Exception:
    mannwhitneyu = None  # type: ignore

# ---- Hard-coded OPT (edit/extend as you like) ----
TSPLIB_OPT: Dict[str, int] = {
    "att48": 10628,
    "berlin52": 7542,
    "ch130": 6110,
    "eil101": 629,
    "eil51": 426,
    "eil76": 538,
    "fnl4461": 182566,
    "gr137": 69853,
    "kroA100": 21282,
    "kroA200": 29368,
    "lin318": 42029,
    "pla7397": 23260728,
    "pla33810": 66048945,
    "pla85900": 142382641,
    "pr2392": 378032,
}

# ---------- helpers ----------
def is_run_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    # treat dir as a run if it has any seed logs
    return any(p.glob("seed_*.jsonl"))

def discover_run_dirs(root: Path) -> List[Path]:
    if is_run_dir(root):
        return [root]
    out: List[Path] = []
    if root.is_dir():
        for d in root.rglob("*"):
            if is_run_dir(d):
                out.append(d)
    out.sort()
    return out

def find_summary(run_dir: Path) -> Optional[Path]:
    # pick the longest-named *.json that isn’t a seed log
    cands = [p for p in run_dir.glob("*.json") if not p.name.startswith("seed_")]
    if not cands:
        return None
    cands.sort(key=lambda x: len(x.name), reverse=True)
    return cands[0]

def read_seed_anytime_log(seed_log: Path) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    anytime: List[Tuple[float, float]] = []
    runtime: Optional[float] = None
    try:
        for line in seed_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ev = row.get("event")
            if ev == "improve":
                t = float(row.get("t", 0.0))
                b = float(row.get("best", 0.0))
                anytime.append((t, b))
            elif ev == "summary":
                r = row.get("runtime")
                if isinstance(r, (int, float)):
                    runtime = float(r)
    except Exception:
        pass
    anytime.sort(key=lambda x: x[0])
    return anytime, runtime

def bootstrap_ci_median(x: np.ndarray, n=10_000, alpha=0.05, rng: Optional[np.random.Generator]=None) -> Tuple[float,float]:
    if rng is None:
        rng = np.random.default_rng()
    if len(x) == 0:
        return (np.nan, np.nan)
    meds = np.empty(n, dtype=float)
    m = len(x)
    for i in range(n):
        meds[i] = np.median(rng.choice(x, size=m, replace=True))
    lo, hi = np.percentile(meds, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def cliffs_delta(x: List[float], y: List[float]) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    greater = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj: greater += 1
            elif xi < yj: less += 1
    return (greater - less) / float(nx*ny)

# ---------- core ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="Run dir or results root")
    args = ap.parse_args()

    root = Path(args.results).resolve()
    runs = discover_run_dirs(root)
    if not runs:
        print(f"[WARN] No runs found under {root}")
        return

    # instance -> list of best lengths (per seed)
    best_by_instance: Dict[str, List[float]] = {}

    for run_dir in runs:
        # summary to get instance + seed bests
        summary_path = find_summary(run_dir)
        if not summary_path:
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        instance = str(summary.get("instance") or run_dir.name).split("_")[-1]
        if instance not in TSPLIB_OPT:
            # skip instances not in hard-coded table
            continue

        seed_results = summary.get("seed_results", [])
        if not seed_results:
            # as a fallback, try to infer best from logs — but in your pipeline,
            # summary should already have per-seed best_length
            pass

        for sr in seed_results:
            try:
                best_len = float(sr.get("best_length"))
            except Exception:
                continue
            best_by_instance.setdefault(instance, []).append(best_len)

    # Compute per-instance gaps (in %) using hard-coded OPT
    gaps_by_instance: Dict[str, List[float]] = {}
    for inst, bests in best_by_instance.items():
        opt = TSPLIB_OPT.get(inst)
        if not opt or opt <= 0:
            continue
        gaps = [ (b/opt - 1.0)*100.0 for b in bests ]
        gaps_by_instance[inst] = gaps

    # Print per-instance medians with bootstrap CIs
    print("=== RESULTS ===\n")
    for inst in sorted(gaps_by_instance.keys()):
        gaps = np.array(gaps_by_instance[inst], dtype=float)
        med = float(np.median(gaps)) if len(gaps) else np.nan
        lo, hi = bootstrap_ci_median(gaps, n=10_000, alpha=0.05)
        print(f"{inst}: Mdn={med:.1f}% [95% CI: {lo:.1f}%, {hi:.1f}%]  (n={len(gaps)})")

    # ---- Optional comparisons (edit as you like) ----
    def compare(a: str, b: str):
        xa = gaps_by_instance.get(a, [])
        xb = gaps_by_instance.get(b, [])
        if not xa or not xb:
            print(f"\n{a} vs {b}: insufficient data.")
            return
        delta = cliffs_delta(xa, xb)
        if mannwhitneyu is not None:
            U, p = mannwhitneyu(xa, xb, alternative="two-sided")
            print(f"\n{a} vs {b}: U={U}, p={p:.4f}, Cliff's Δ={delta:.3f}")
        else:
            print(f"\n{a} vs {b}: Cliff's Δ={delta:.3f} (SciPy missing → p-value skipped)")

    # Examples you mentioned:
    compare("eil76", "kroA200")

    # Example grouped comparison (EIL vs 'geometric' pile). Edit as desired.
    eil = []
    for k in ("eil51", "eil76", "eil101"):
        eil += gaps_by_instance.get(k, [])
    geo = []
    for k in ("berlin52", "kroA100", "kroA200"):
        geo += gaps_by_instance.get(k, [])
    if eil and geo:
        delta = cliffs_delta(eil, geo)
        if mannwhitneyu is not None:
            U, p = mannwhitneyu(eil, geo, alternative="two-sided")
            print(f"\nEIL vs Geometric: U={U}, p={p:.4f}, Δ={delta:.3f}")
        else:
            print(f"\nEIL vs Geometric: Δ={delta:.3f} (SciPy missing → p-value skipped)")
    else:
        print("\nEIL vs Geometric: insufficient data.")

if __name__ == "__main__":
    main()
