# scripts/config_generator.py

import yaml
from pathlib import Path
from copy import deepcopy

class AblationConfigGenerator:
    def __init__(self, baseline_path="configs/params.yaml"):
        with open(baseline_path) as f:
            self.baseline = yaml.safe_load(f)
    
    def generate_suite(self, output_dir="configs/ablations_v2"):
        """Generate ablation configs in your specified order"""
        configs = []
        
        # Baseline (for comparison)
        configs.append(("00_baseline", deepcopy(self.baseline)))
        
        # 1. Edge cap ablation
        for val in [8, 12, 16]:  # baseline is 10
            cfg = self._modify_nested(["solver", "candidate", "edge_cap"], val)
            configs.append((f"01_edge_cap_{val}", cfg))
        
        # 2. Portals per node
        for val in [3, 7, 10]:  # baseline is 5
            cfg = self._modify_nested(["solver", "candidate", "portals_per_node"], val)
            configs.append((f"02_portals_{val}", cfg))
        
        # 3. K-value (initializers)
        for val in [4]:  # baseline is 1, test 4
            cfg = self._modify_nested(["solver", "initializers", "K"], val)
            configs.append((f"03_k_init_{val}", cfg))
        
        # 4. Init budget
        for val in [0.10, 0.20]:  # baseline is 0.05
            cfg = self._modify_nested(["solver", "initializers", "total_budget_pct"], val)
            configs.append((f"04_init_budget_{int(val*100)}", cfg))
        
        # 5. Or-opt timing (forbid_until_pct)
        for val in [0.3, 0.7]:  # baseline is 0.5
            cfg = self._modify_nested(["solver", "local_search", "oropt_gate", "forbid_until_pct"], val)
            configs.append((f"05_oropt_forbid_{int(val*100)}", cfg))
        
        # 6. Merge schedule (POPMUSIC rounds)
        # baseline is 2 rounds (mid & end)
        # Test 3 rounds (mid, 3/4, end) - requires code change, just flag for now
        cfg = self._modify_nested(["solver", "integrator", "rounds"], 3)
        configs.append((f"06_merge_3rounds", cfg))
        
        # 7. HK 1-tree bound
        cfg = self._modify_nested(["solver", "hk1tree", "enabled"], True)
        configs.append((f"07_hk_enabled", cfg))
        
        # Write all
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for name, cfg in configs:
            out_path = Path(output_dir) / f"{name}.yaml"
            with open(out_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"✓ {out_path}")
        
        return configs
    
    def _modify_nested(self, path, value):
        """Modify nested dict via path like ['solver', 'candidate', 'edge_cap']"""
        cfg = deepcopy(self.baseline)
        current = cfg
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
        return cfg

if __name__ == "__main__":
    gen = AblationConfigGenerator()
    configs = gen.generate_suite()
    print(f"\n✓ Generated {len(configs)} ablation configs")