#!/usr/bin/env python3
import json, re, sys, math, hashlib, time
from pathlib import Path

RESULTS_DIR = Path("results")
BENCH_DIR   = Path("benchmarks")  # where <instance>.tsp lives

# ---- TSPLIB parsers (coords + tours) and integer metrics ----
def parse_tsp(tsp_path: Path):
    header, coords, in_coords = {}, [], False
    with tsp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("COMMENT"): continue
            if s.startswith("NODE_COORD_SECTION"):
                in_coords = True; continue
            if s.startswith("EOF"): break
            if in_coords:
                parts = s.split()
                if len(parts) >= 3:
                    try: _ = int(parts[0]); x = float(parts[1]); y = float(parts[2])
                    except ValueError: x = float(parts[0]); y = float(parts[1])
                    coords.append((x,y))
                continue
            if ":" in s:
                k,v = [x.strip() for x in s.split(":",1)]; header[k.upper()] = v
            else:
                parts = s.split()
                if len(parts)>=2: header[parts[0].upper()] = " ".join(parts[1:])
    metric = header.get("EDGE_WEIGHT_TYPE","EUC_2D").upper()
    return {"metric": metric, "coords": coords}

def parse_tour(tour_path: Path):
    tour, in_tour = [], False
    with tour_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("COMMENT"): continue
            if s.startswith("TOUR_SECTION"): in_tour=True; continue
            if s.startswith("EOF"): break
            if in_tour:
                if s=="-1": break
                for tok in s.split(): tour.append(int(tok))
    return tour

def d_euc(a,b): return int(round(math.hypot(a[0]-b[0], a[1]-b[1])))
def d_ceil(a,b): return int(math.ceil(math.hypot(a[0]-b[0], a[1]-b[1])))
def d_att(a,b):
    rij = math.sqrt(((a[0]-b[0])**2 + (a[1]-b[1])**2)/10.0)
    tij = int(round(rij))
    if tij < rij: tij += 1
    return tij

def tour_len(coords, tour_1based, metric):
    if not coords or not tour_1based: return None
    if metric=="EUC_2D": d = d_euc
    elif metric=="CEIL_2D": d = d_ceil
    elif metric=="ATT": d = d_att
    else: raise NotImplementedError(f"EDGE_WEIGHT_TYPE {metric}")
    t0 = [i-1 for i in tour_1based]
    total=0
    for i in range(len(t0)):
        a=coords[t0[i]]; b=coords[t0[(i+1)%len(t0)]]
        total += d(a,b)
    return total

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

# ---- manifest rebuild ----
TS_DIR_RX = re.compile(r"^\d{2}_\d{1,2}_\d{1,2}_\d{4}_.+$")  # YY_M_DD_HHMM_instance

def seeds_in(dirpath: Path):
    return sorted(int(m.group(1)) for m in
                  [re.search(r"seed_(\d+)\.jsonl$", p.name) for p in dirpath.glob("seed_*.jsonl")]
                  if m)

def main():
    out_path = RESULTS_DIR/"manifest.jsonl"
    if not RESULTS_DIR.exists():
        print("No results/ dir found", file=sys.stderr); sys.exit(2)

    lines = []
    for run_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir() and TS_DIR_RX.match(p.name)):
        base = run_dir.name
        # infer instance from dirname suffix or from JSON
        instance_hint = base.split("_", maxsplit=4)[-1]
        run_json = run_dir/(base + ".json")
        tour_path = run_dir/(base + ".tour")
        yaml_path = run_dir/(base + ".yaml")

        if not (run_json.exists() and tour_path.exists()):
            # skip partials
            continue

        # read run summary (Update 20)
        summary = json.loads(run_json.read_text())
        instance = summary.get("instance", instance_hint)
        metric   = summary.get("metric","EUC_2D")
        mode     = summary.get("mode","SH")
        wall_s   = int(summary.get("wall_s", 0) or 0)
        elapsed_s= float(summary.get("elapsed_s", 0.0) or 0.0)
        epochs_completed = int(summary.get("epochs_completed", 0) or 0)
        epoch_s  = int(summary.get("epoch_s", 0) or 0)
        parity   = summary.get("parity","NEQ_BEST")
        # optional per-run extras (may be absent)
        early_slope = summary.get("early_slope", None)
        improvements_per_min = summary.get("improvements_per_min", None)
        hive_attrib = summary.get("hive_attrib", None)
        opt = summary.get("opt", None)
        best_known = summary.get("best_known", None)

        # recompute authoritative integer length from tour if we have the .tsp
        tsp_path = BENCH_DIR / f"{instance}.tsp"
        tour = parse_tour(tour_path)
        metric_tag = str(summary.get("metric", "EUC_2D")).upper()

        length_int = None
        if tsp_path.exists():
            tsp = parse_tsp(tsp_path)
            coords = tsp["coords"]
            length_int = tour_len(coords, tour, metric_tag)
        else:
            # Fallback: trust controller's recorded integer length from the run summary
            fallback = summary.get("best_len_int") or summary.get("length_tsplib")
            if fallback is None:
                print(f"[rebuild][skip] missing .tsp and no best_len_int in {run_json}", file=sys.stderr)
                continue
            length_int = int(fallback)

        # each seed gets a line (values null if not per-seed)
        seeds = seeds_in(run_dir) or [0]
        for seed in seeds:
            rec = {
                "instance": instance,
                "metric": metric_tag,
                "mode": mode,
                "seed": int(seed),
                "wall_s": wall_s,
                "elapsed_s": elapsed_s,
                "epochs_completed": epochs_completed,
                "epoch_s": epoch_s,
                "tour_path": str(tour_path).replace("\\","/"),
                "tour_sha256": tour_sha,
                "length_tsplib": int(length_int) if length_int is not None else None,
                "opt": opt if (isinstance(opt,int) or opt is None) else None,
                "best_known": best_known if (isinstance(best_known,int) or best_known is None) else None,
                "parity": str(parity),
                "ttt10_s": None,
                "ttt5_s": None,
                "early_slope": early_slope if isinstance(early_slope,(int,float)) else None,
                "improvements_per_min": improvements_per_min if isinstance(improvements_per_min,(int,float)) else None,
                "hive_attrib": hive_attrib if (isinstance(hive_attrib,str) or hive_attrib is None) else None,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            lines.append(json.dumps(rec))

    if not lines:
        print("[rebuild] No standardized runs found to add.", file=sys.stderr)
        sys.exit(1)

    # Write/append: overwrite current manifest with rebuilt content
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rebuild] wrote {len(lines)} lines to {out_path}")

if __name__ == "__main__":
    main()
