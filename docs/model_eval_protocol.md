# MODEL\_EVAL\_PROTOCOL.md (drop in `docs/`)

## 0) Purpose

Compare **baseline vs new features** across models (GPT-5 lead, Claude, Gemini, DeepSeek) with **identical configs, seeds, metrics, and time caps**. Reject any claim that fails metric parity.

## 1) Canonical run matrix

Instances & caps:

* **Small:** att48, eil101, kroA100, ch130 → wall 90s
* **Mid:** gr137, pr2392 → wall 1800s
* **Large:** fnl4461, pla7397, pla33810, pla85900 → wall {1800, 1800, 3600, 7200}

Targets:

* If `.opt.tour` exists: report **TTT\@opt** and **TTT\@0.1%**.
* Else: **TTT\@1% / 0.5% / 0.1%**.

Seeds: `42 123 456 789 999` (all models must use these).

Env (each worker):

```
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## 2) Configs under test

* **Baseline** (frozen):

  ```yaml
  solver:
    candidate: { k: 24, use_knn: true, use_delaunay: true }
    local_search: { use_oropt: true, kick_period_moves: 420 }
    scouting: { dynamic: { enabled: true, no_improve_s: 12, max_restarts_per_seed: 1 } }
    metric: { type: tsplib, integer: true }
  integrator: { method: "popmusic" }
  ```
* **LITE features ON:** `zones.adaptive_overlap.enabled=true`, `local_search.three_opt_lite=true`, `integrator.method=ucr`.
* **Autoscaler v2** (when present): `features.kd_backend=auto`, `features.max_threads_per_lib=1`.

## 3) Output schema (one JSON per instance)

Each model writes `results/<model>/<instance>.json`:

```json
{
  "model": "gemini",
  "instance": "att48",
  "edge_weight_type": "EUC_2D",
  "kd_backend": "sklearn",
  "best_known": 10628,
  "best_overall": 10628,
  "median_best": 10647,
  "parity": "OK",            // OK or FAIL
  "ttt": {"opt": 3.8, "0.1%": 1.9, "0.5%": 0.9, "1.0%": 0.4},
  "seeds": [42,123,456,789,999],
  "resolved_params_path": "results/.../resolved_params.yaml",
  "distance_parity": "OK"    // from verify scripts
}
```

## 4) Metric parity (hard gate)

* Harness must call `build_eval_fn(cfg, ins)` from `bee_tsp/metrics.py` for all tour sums.
* **FAIL if** `.opt.tour` exists and `best_overall < best_known` (that’s a bug, not a win).
* Print once: `[PARITY] att48: opt_eval=10628 best_known=10628`.

## 5) Directory layout

```
results/
  gpt5/
  claude/
  gemini/
  deepseek/
```

Each contains one JSON per instance and a per-suite `summary.json`.

## 6) Acceptance (green/amber/red)

* **Green:** Parity OK on all “Y” instances; median TTT improves ≥15% vs baseline on ≥50% of instances; no regression on acceptance gates.
* **Amber:** Mixed; require ablation to attribute gains (e.g., turn off UCR or three\_opt\_lite).
* **Red:** Any parity FAIL or negative gap → reject run; fix & rerun.

## 7) Disagreement protocol

When two models disagree by >10% in TTT or >0.2% gap:

1. Confirm parity on both sides.
2. Re-run with same seeds, single worker (to remove scheduling noise).
3. Compare `resolved_params.yaml` and logged features (kd\_backend, autoscaler v2 flags).
4. If still divergent → UNKNOWN (no claim) until code diff explains it.

## 8) Report template (paste into update file)

```
### Multi-Model Report (Update N)
Instance  | Baseline TTT@1% | LITE TTT@1% (gpt5/claude/gemini/deepseek) | Parity (all) | Verdict
att48     | 12.3s           | 9.8 / 10.1 / 9.9 / 10.0                   | OK           | ✅ improve
kroA100   | 31.5s           | 28.2 / 27.9 / 28.5 / 28.1                 | OK           | ✅ improve
ch130     |  –              |  –                                        | OK           | hold (no change)
...
Summary: Green | Amber | Red
Notes: (any anomalies, reruns, config diffs)
```

---

## Tiny tools (optional)

**Aggregator (save `tools/aggregate_eval.py`)**

```python
import json, sys, glob, os, csv
rows = []
for model in ("gpt5","claude","gemini","deepseek"):
    for p in glob.glob(os.path.join("results", model, "*.json")):
        with open(p,"r") as f: r = json.load(f); r["model"]=model; rows.append(r)
with open("results/combined.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader(); w.writerows(rows)
print("Wrote results/combined.csv")
```

**Parity one-liner (PowerShell)**

```powershell
Get-ChildItem results\*\*.json | %{
  $j = Get-Content $_ | ConvertFrom-Json
  "{0,-8} {1,-10} parity:{2}" -f $j.model,$j.instance,$j.parity
}
```

---

If you want, I can also drop a 10-line “Ablation Checklist” you can paste into each update. Otherwise, this protocol + your update\_history flow will keep CC enthusiastic *and* correct—while Gemini/DeepSeek act as calm referees.
