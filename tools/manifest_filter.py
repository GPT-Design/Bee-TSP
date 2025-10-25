#!/usr/bin/env python3
# tools/manifest_filter.py
import json, sys
from pathlib import Path

REQUIRED = ["instance","metric","mode","seed","wall_s","elapsed_s","epochs_completed","epoch_s",
            "tour_path","tour_sha256","length_tsplib","opt","best_known","parity","ttt10_s",
            "ttt5_s","early_slope","improvements_per_min","hive_attrib","timestamp_utc"]
METRICS = {"EUC_2D","CEIL_2D","ATT","GEO"}
MODES   = {"SH","MH-Lite","MH-Full"}

def to_int(x):
    if x is None: return None
    if isinstance(x,int): return x
    if isinstance(x,float) and x.is_integer(): return int(x)
    if isinstance(x,str) and x.strip().lstrip("+-").isdigit(): return int(x)
    raise ValueError

def to_float_or_none(x):
    if x is None: return None
    if isinstance(x,(int,float)): return float(x)
    if isinstance(x,str):
        s=x.strip().lower()
        if s in {"", "null", "none"}: return None
        return float(s)
    raise ValueError

def valid(rec):
    # required keys present
    if any(k not in rec for k in REQUIRED): return False, "missing required"
    try:
        if rec["metric"] not in METRICS: return False, "bad metric"
        if rec["mode"] not in MODES: return False, "bad mode"
        rec["seed"] = to_int(rec["seed"])
        rec["wall_s"] = to_int(rec["wall_s"])
        rec["epochs_completed"] = to_int(rec["epochs_completed"])
        rec["epoch_s"] = to_int(rec["epoch_s"])
        rec["length_tsplib"] = to_int(rec["length_tsplib"])
        # allow null for these
        rec["opt"] = None if rec["opt"] in ("", "null") else (to_int(rec["opt"]) if rec["opt"] is not None else None)
        rec["best_known"] = None if rec["best_known"] in ("", "null") else (to_int(rec["best_known"]) if rec["best_known"] is not None else None)
        rec["elapsed_s"] = to_float_or_none(rec["elapsed_s"])
        rec["ttt10_s"] = to_float_or_none(rec["ttt10_s"])
        rec["ttt5_s"]  = to_float_or_none(rec["ttt5_s"])
        rec["early_slope"] = to_float_or_none(rec["early_slope"])
        rec["improvements_per_min"] = to_float_or_none(rec["improvements_per_min"])
        # parity must be a string
        if not isinstance(rec["parity"], str): return False, "bad parity"
        # hive_attrib may be null or string
        if rec["hive_attrib"] is not None and not isinstance(rec["hive_attrib"], str): return False, "bad hive_attrib"
        # timestamp_utc must be a string
        if not isinstance(rec["timestamp_utc"], str): return False, "bad timestamp"
        # tour_path, tour_sha256 must be strings
        if not isinstance(rec["tour_path"], str) or not isinstance(rec["tour_sha256"], str): return False, "bad paths"
    except Exception:
        return False, "type conversion"
    return True, "ok"

def main():
    if len(sys.argv)<2:
        print("usage: manifest_filter.py results/manifest.jsonl"); sys.exit(2)
    src = Path(sys.argv[1])
    dst = src.with_suffix(".clean.jsonl")
    legacy = src.with_suffix(".legacy.jsonl")
    ok = bad = 0
    with src.open() as f, dst.open("w") as out_ok, legacy.open("w") as out_bad:
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            try:
                rec=json.loads(line)
            except Exception:
                out_bad.write(line+"\n"); bad+=1; continue
            is_ok, _ = valid(rec)
            (out_ok if is_ok else out_bad).write(json.dumps(rec)+"\n")
            ok += int(is_ok); bad += int(not is_ok)
    print(f"[manifest_filter] kept={ok} archived={bad}")
    print(f"[manifest_filter] wrote {dst} and {legacy}")
    print(">> To adopt: back up original then replace with the .clean.jsonl")

if __name__=="__main__": main()
