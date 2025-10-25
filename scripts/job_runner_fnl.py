# scripts/job_runner_fnl.py

import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

class FNLTestRunner:
    def __init__(self, instance="fnl4461", wall_s=1800, n_runs=10, seeds_start=42):
        self.instance = instance
        self.wall_s = wall_s
        self.seeds = list(range(seeds_start, seeds_start + n_runs))
        
        # Test configs: baseline + 3 ablations
        self.configs = [
            "configs/params.yaml",  # baseline
            "configs/ablations_v2/01_edge_cap_16.yaml",
            "configs/ablations_v2/02_portals_10.yaml",
            "configs/ablations_v2/07_hk_enabled.yaml"
        ]
    
    def run_all(self, max_workers=4):  # 4 workers for 1800s runs
        jobs = [(Path(cfg), seed) for cfg in self.configs for seed in self.seeds]
        
        print(f"FNL4461 Test: {len(jobs)} jobs (4 configs × {len(self.seeds)} seeds)")
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
            "returncode": result.returncode
        }

if __name__ == "__main__":
    runner = FNLTestRunner(n_runs=10, wall_s=1800)
    runner.run_all(max_workers=4)