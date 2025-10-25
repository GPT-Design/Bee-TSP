#!/usr/bin/env python3
# tools/aggregate_ab_tables.py
import json, csv, statistics as stats
from pathlib import Path
import argparse

INSTANCES = {"pla33810": {3600,5400}, "pr2392": {600,900}, "fnl4461": {1800}}
MODES = {"SH","MH-Lite"}

def pctl(vals, q):
    if not vals: return None
    s = sorted(vals); k = (len(s)-1)*q/100
    i = int(k); d = k - i
    return s[i]*(1-d)+s[i+1]*d if i+1 < len(s) else s[i]

def read_manifest(path):
    for line in Path(path).read_text().splitlines():
        if line.strip(): yield json.loads(line)

def backfill_from_run_json(rec):
    # Optional: try to read the run summary JSON if something is missing
    tour_path = rec.get("tour_path")
    if not tour_path: return rec
    tp = Path(tour_path)
    if not tp.exists(): return rec
    run_json = tp.with_suffix(".json")
    if run_json.exists():
        try:
            j = json.loads(run_json.read_text())
            for k in ["wall_s","elapsed_s","epochs_completed","epoch_s","mode","instance"]:
                rec.setdefault(k, j.get(k))
        except Exception:
            pass
    return rec

def collect(records, include_experimental=False):
    by_group = {}   # (instance, wall_s, mode) -> list[rec]
    per_seed = []   # flat rows
    for r in records:
        r = backfill_from_run_json(r)
        inst = r.get("instance"); wall = r.get("wall_s"); mode = r.get("mode")
        seed = r.get("seed"); length = r.get("length_tsplib")
        if inst not in INSTANCES or wall not in INSTANCES[inst] or mode not in MODES:
            continue
        # Update_22: Hide MH rows by default unless --include-experimental
        if not include_experimental and mode in ["MH", "MH-Lite"]:
            continue
        key = (inst, wall, mode)
        by_group.setdefault(key, []).append(r)
        per_seed.append({
            "instance": inst, "wall_s": wall, "seed": seed, "mode": mode,
            "best_len_int": length,
            "ttt10_s": r.get("ttt10_s"), "ttt5_s": r.get("ttt5_s"),
            "early_slope": r.get("early_slope"),
            "improvements_per_min": r.get("improvements_per_min"),
            "hive_attrib": r.get("hive_attrib"), "parity": r.get("parity")
        })
    return by_group, per_seed

def median_safe(vals):
    vals = [v for v in vals if v is not None]
    return stats.median(vals) if vals else None

def write_master(by_group, out_csv):
    hdr = ["instance","wall_s","mode","best_len_med","best_len_p10","best_len_p90",
           "ttt10_s_med","ttt5_s_med","early_slope_med","improvements_per_min_med","n_runs"]
    rows=[]
    for (inst, wall, mode), recs in sorted(by_group.items()):
        if not recs: continue
        L   = [r.get("length_tsplib") for r in recs if r.get("length_tsplib") is not None]
        t10 = [r.get("ttt10_s")       for r in recs if r.get("ttt10_s")       is not None]
        t05 = [r.get("ttt5_s")        for r in recs if r.get("ttt5_s")        is not None]
        es  = [r.get("early_slope")   for r in recs if r.get("early_slope")   is not None]
        ipm = [r.get("improvements_per_min") for r in recs if r.get("improvements_per_min") is not None]
        rows.append({
            "instance": inst, "wall_s": wall, "mode": mode,
            "best_len_med": median_safe(L),
            "best_len_p10": pctl(L,10), "best_len_p90": pctl(L,90),
            "ttt10_s_med": median_safe(t10), "ttt5_s_med": median_safe(t05),
            "early_slope_med": median_safe(es),
            "improvements_per_min_med": median_safe(ipm),
            "n_runs": len(recs)
        })
    Path(out_csv).write_text("")  # ensure file exists even if no rows
    with open(out_csv, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)

def write_per_seed(per_seed, out_csv):
    hdr = ["instance","wall_s","seed","mode","best_len_int","ttt10_s","ttt5_s",
           "early_slope","improvements_per_min","hive_attrib","parity"]
    with open(out_csv, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(per_seed)

def print_go_nogo(by_group, gate_map=None):
    """
    Print GO/NO-GO per (instance, wall) comparing MH-Lite vs SH.
    gate_map: dict like {"pla33810": 1.0} meaning MH-Lite must beat SH by ≥1.0%.
    """
    import statistics as stats
    if gate_map is None:
        gate_map = {"pla33810": 1.0}  # long-wall gate (percent). Others default to 0.

    # median best_len per group
    med = {}
    for (inst, wall, mode), recs in by_group.items():
        L = [r.get("length_tsplib") for r in recs if r.get("length_tsplib") is not None]
        if L:
            med[(inst, wall, mode)] = stats.median(L)

    # pair SH vs MH-Lite
    pairs = sorted({(i, w) for (i, w, _) in med.keys()})
    for (inst, wall) in pairs:
        key_sh = (inst, wall, "SH")
        key_mh = (inst, wall, "MH-Lite")
        if key_sh in med and key_mh in med:
            sh = med[key_sh]; mh = med[key_mh]
            rel = (sh - mh) / sh * 100.0  # +% => MH-Lite better (shorter)
            gate = gate_map.get(inst, 0.0)
            verdict = "GO" if rel >= gate else "NO-GO"
            print(f"{inst} @{wall}s: MH-Lite vs SH = {rel:.3f}% -> {verdict}")


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"results/manifest.jsonl")
    ap.add_argument("--out-master", default=r"results/ab_master.csv")
    ap.add_argument("--out-per-seed", default=r"results/ab_per_seed.csv")
    # Update_22: Add flag to include experimental MH data
    ap.add_argument("--include-experimental", action="store_true",
                    help="Include experimental Multi-Hive (MH/MH-Lite) results")
    args = ap.parse_args()

    records = list(read_manifest(args.manifest))
    by_group, per_seed = collect(records, include_experimental=args.include_experimental)
    Path("results").mkdir(exist_ok=True)
    write_master(by_group, args.out_master)
    write_per_seed(per_seed, args.out_per_seed)

    if args.include_experimental:
        print("[aggregate] wrote", args.out_master, "and", args.out_per_seed, "(including experimental MH data)")
        # GO/NO-GO summary only when experimental data included
        print_go_nogo(by_group, gate_map={"pla33810": 1.0})
    else:
        print("[aggregate] wrote", args.out_master, "and", args.out_per_seed, "(SH production data only)")
        print("[aggregate] Use --include-experimental to include archived Multi-Hive results")