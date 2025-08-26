#!/usr/bin/env python3
# (content abbreviated in this cell for brevity — full script retained)
import argparse, os, json, math, statistics
from pathlib import Path
from collections import defaultdict

def load_best_known(path):
    if not path: return {}
    m = {}
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        cols = {k:i for i,k in enumerate(header)}
        for line in f:
            if not line.strip(): continue
            parts = [p.strip() for p in line.strip().split(',')]
            try:
                name = parts[cols.get('instance',0)]
                val = float(parts[cols.get('best_known',1)])
            except Exception:
                continue
            m[name] = val
    return m

def read_run_jsonl(path):
    anytime = []
    best = None; runtime = None; seed = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            evt = obj.get('event')
            if evt == 'improve':
                t = float(obj.get('t', 0.0)); b = float(obj.get('best', float('inf')))
                anytime.append((t, b))
                best = b if best is None else min(best, b)
            elif evt == 'summary':
                seed = obj.get('seed', seed)
                runtime = float(obj.get('runtime', runtime or 0.0))
                b = obj.get('best', None)
                if b is not None:
                    b = float(b)
                    best = b if best is None else min(best, b)
    anytime.sort(key=lambda x: x[0])
    return {'seed': seed, 'best': best, 'runtime': runtime, 'anytime': anytime}

def earliest_time_to_target(anytime, target):
    for t, b in anytime:
        if b <= target:
            return t
    return None

def summarize_instance(instance_dir, best_known_val=None, thresholds=(1.0,0.5,0.1)):
    import statistics
    runs = []
    for p in sorted(Path(instance_dir).glob("seed_*.jsonl")):
        runs.append(read_run_jsonl(p))
    if not runs:
        return None

    bests = [r['best'] for r in runs if r['best'] is not None]
    rts   = [r['runtime'] for r in runs if r['runtime'] is not None]

    median_best = statistics.median(bests) if bests else None
    p10_best = statistics.quantiles(bests, n=10)[0] if len(bests) >= 10 else (min(bests) if bests else None)
    p90_best = statistics.quantiles(bests, n=10)[-1] if len(bests) >= 10 else (max(bests) if bests else None)
    median_rt = statistics.median(rts) if rts else None

    obs_best = min(bests) if bests else None
    ref = best_known_val if best_known_val is not None else obs_best

    ttt = {}
    if ref is not None:
        for th in thresholds:
            target = ref * (1.0 + th/100.0)
            times = []
            for r in runs:
                t_hit = earliest_time_to_target(r['anytime'], target)
                if t_hit is not None:
                    times.append(t_hit)
            ttt[th] = (statistics.median(times) if times else None)
    return {
        'median_best': median_best,
        'p10_best': p10_best,
        'p90_best': p90_best,
        'median_runtime': median_rt,
        'ref': ref,
        'ttt': ttt,
        'num_runs': len(runs),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--algo', default='bstsp')
    ap.add_argument('--best', default=None)
    ap.add_argument('--out', default='results_summary.csv')
    args = ap.parse_args()

    best_known = load_best_known(args.best) if args.best else {}
    algo_root = Path(args.results) / args.algo
    instances = [p.name for p in algo_root.iterdir() if p.is_dir()] if algo_root.exists() else []

    rows = []
    for inst in sorted(instances):
        s = summarize_instance(algo_root/inst, best_known.get(inst))
        if not s: 
            continue
        row = {
            'instance': inst,
            'num_runs': s['num_runs'],
            'ref_best': s['ref'],
            'median_best': s['median_best'],
            'p10_best': s['p10_best'],
            'p90_best': s['p90_best'],
            'median_runtime_s': s['median_runtime'],
            'ttt_1pct_s': s['ttt'].get(1.0) if s['ttt'] else None,
            'ttt_0.5pct_s': s['ttt'].get(0.5) if s['ttt'] else None,
            'ttt_0.1pct_s': s['ttt'].get(0.1) if s['ttt'] else None,
        }
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        import csv
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out} with {len(rows)} rows.")
    else:
        print("No runs found. Check your results directory.")

if __name__ == '__main__':
    main()
