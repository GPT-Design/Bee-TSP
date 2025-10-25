# scripts/job_runner.py

import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
from datetime import datetime
import os

class AblationRunner:
    def __init__(self, 
                 config_dir="configs/ablations_v2",
                 instance="pr2392",
                 wall_s=600,
                 n_runs=15,
                 seeds_start=42):
        self.config_dir = Path(config_dir)
        self.instance = instance
        self.wall_s = wall_s
        self.seeds = list(range(seeds_start, seeds_start + n_runs))
        self.base_cmd = ["python", "scripts/run_large3.py"]
    
    def run_all(self, max_workers=5):
        configs = sorted(self.config_dir.glob("*.yaml"))
        jobs = [(cfg, seed) for cfg in configs for seed in self.seeds]
        
        print(f"Jobs: {len(jobs)} ({len(configs)} configs × {len(self.seeds)} seeds)")
        print(f"Workers: {max_workers}, ETA: ~{len(jobs) * self.wall_s / max_workers / 3600:.1f} hours\n")
        
        completed = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._run_single, cfg, seed): (cfg, seed) 
                      for cfg, seed in jobs}
            
            for future in futures:
                cfg, seed = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    status = "✓" if result["returncode"] == 0 else "✗"
                    print(f"[{completed}/{len(jobs)}] {status} {cfg.stem} seed={seed}")
                except Exception as e:
                    completed += 1
                    print(f"[{completed}/{len(jobs)}] ✗ {cfg.stem} seed={seed} CRASH: {e}")
    
    def _run_single(self, config_path, seed):
        cmd = [
            "python", "scripts/run_large3.py",
            "--instance", self.instance,
            "--wall", str(self.wall_s),
            "--seeds_list", str(seed),
            "--config", str(config_path),
            "--mode", "SH"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "config": config_path.name,
            "seed": seed,
            "returncode": result.returncode,
            "stdout": result.stdout[-200:] if result.stdout else "",  # Last 200 chars
            "stderr": result.stderr[-200:] if result.stderr else ""
        }

if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        # Quick test: 2 configs, 2 seeds, 60s
        runner = AblationRunner(n_runs=2, wall_s=60)
        runner.run_all(max_workers=2)
    else:
        # Full run: 14 configs, 15 seeds, 600s
        runner = AblationRunner(n_runs=15, wall_s=600)
        runner.run_all(max_workers=5)
