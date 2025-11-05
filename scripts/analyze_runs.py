import sys, json, math, glob, os
from statistics import median

# Extend as needed
TSPLIB_OPT = {
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
    # "pr2392": 378032,  # example if you want to add more later
}

def load_anytime(path):
    evts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event") == "improve":
                t = float(obj.get("t", 0.0))
                b = float(obj.get("best", math.inf))
                evts.append((t, b))
            elif obj.get("event") == "summary":
                # ensure final is included too
                t = float(obj.get("runtime", 0.0))
                b = float(obj.get("best", math.inf))
                evts.append((t, b))
    # sort by time just in case
    evts.sort(key=lambda x: x[0])
    return evts

def first_time_within(evts, target):
    for t, b in evts:
        if b <= target:
            return t
    return None

def summarize(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ("NA","NA","NA")
    xs.sort()
    return (f"{xs[0]:.1f}", f"{median(xs):.1f}", f"{xs[-1]:.1f}")

def main(results_dir):
    # infer instance from any seed file in subdirs like results/<stamp>_att48/
    subdirs = [d for d in glob.glob(os.path.join(results_dir, "*")) if os.path.isdir(d)]
    if not subdirs:
        print("No result subdirectories found.")
        return

    # pick all seed logs across subdirs; this is robust if you ran multiple stamps
    seed_logs = glob.glob(os.path.join(results_dir, "*", "seed_*_*.jsonl"))
    if not seed_logs:
        print("No seed jsonl logs found.")
        return

    # try to infer instance name from subdir suffix
    # e.g. results/25_11_03_1912_att48 -> "att48"
    def infer_instance(p):
        base = os.path.basename(os.path.dirname(p))
        return base.split("_")[-1] if "_" in base else base

    per_seed = []
    for log in seed_logs:
        inst = infer_instance(log)
        opt = TSPLIB_OPT.get(inst)
        if opt is None:
            # skip seeds for unknown OPTs
            continue

        evts = load_anytime(log)
        if not evts:
            continue

        final_t, final_b = evts[-1]
        gap = 100.0 * (final_b - opt) / opt

        t10 = first_time_within(evts, 1.10 * opt)
        t05 = first_time_within(evts, 1.05 * opt)

        per_seed.append({
            "instance": inst,
            "seed_file": os.path.basename(log),
            "final_best": int(final_b),
            "gap_percent": gap,
            "ttt10_s": t10,
            "ttt5_s": t05,
        })

    if not per_seed:
        print("No analyzable seeds (missing OPT or logs empty).")
        return

    # aggregate by instance
    by_inst = {}
    for row in per_seed:
        by_inst.setdefault(row["instance"], []).append(row)

    lines = []
    for inst, rows in by_inst.items():
        gaps = [r["gap_percent"] for r in rows]
        t10s = [r["ttt10_s"] for r in rows]
        t05s = [r["ttt5_s"] for r in rows]
        g_min,g_med,g_max = summarize(gaps)
        a10_min,a10_med,a10_max = summarize(t10s)
        a05_min,a05_med,a05_max = summarize(t05s)

        print(f"{inst}: seeds={len(rows)}  gap% min/med/max= {g_min}/{g_med}/{g_max} ;  TTT10s min/med/max= {a10_min}/{a10_med}/{a10_max} ;  TTT5s min/med/max= {a05_min}/{a05_med}/{a05_max}")

        for r in rows:
            lines.append(f'{inst},{r["seed_file"]},{r["final_best"]},{r["gap_percent"]:.3f},{"" if r["ttt10_s"] is None else f"{r["ttt10_s"]:.3f}"},{"" if r["ttt5_s"] is None else f"{r["ttt5_s"]:.3f}"}')

    out_csv = os.path.join(results_dir, "analysis.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("instance,seed_file,final_best,gap_percent,ttt10_s,ttt5_s\n")
        for L in lines:
            f.write(L+"\n")
    print(f"[analysis] wrote {out_csv}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "results")
