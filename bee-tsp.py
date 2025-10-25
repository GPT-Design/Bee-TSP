#!/usr/bin/env python3
"""
bee-tsp CLI Wrapper
Maps Update_Claude_18.txt CLI commands to multi_hive_solver.py interface
"""

import argparse
import sys
import os
from pathlib import Path


def create_mh_lite_config(args):
    """Create MH-Lite configuration from CLI arguments"""
    return {
        'solver': {
            'multi_hive': {
                'enabled': True,
                'profile': 'lite',  # MH-Lite profile
                'epoch_s': args.epoch_s,
                'hives': args.hives.split(',') if args.hives else ['H1', 'H3'],
                'profiles': {
                    'H1': {'delta_k': 0, 'bias': 'none', 'bee_ball_boost': 0.0, 'name': 'H1'},
                    'H3': {'delta_k': 4, 'bias': 'none', 'bee_ball_boost': 0.0, 'name': 'H3'}
                },
                'allocator': {
                    'method': args.allocator,
                    'min_slice_s': args.min_slice_s,
                    'improve_eps_pct': args.improve_eps_pct,
                    'ucb1_c': 1.0
                }
            },
            'candidate': {
                'use_delaunay': True,
                'k': 28  # k_base + 4 for SH Large-N profile
            },
            'zones': {
                'agents_per_zone': 4  # Large-N preset
            },
            'bees': {
                'time_budget_s': 2.5
            },
            'boundary_discipline': {
                'halo_fraction': args.halo,
                'portals_per_node': args.portals_per_node,
                'edge_cap': args.edge_cap
            },
            'local_search': {
                'use_or_opt': True,
                'oropt_gate': args.oropt_gate,
                'kick_period_moves': 400
            },
            'merge_schedule': args.merge_schedule,
            'performance': {
                'workers': args.workers,
                'threads_clamped': args.threads
            }
        },
        'integrator': {
            'method': 'popmusic'
        },
        'termination': {
            'wall_time_s': args.wall
        }
    }


def create_sh_config(args):
    """Create Single-Hive configuration from CLI arguments"""
    return {
        'solver': {
            'multi_hive': {
                'enabled': False  # Single-hive mode
            },
            'candidate': {
                'use_delaunay': True,
                'k': 28  # k_base + 4 for SH Large-N profile
            },
            'zones': {
                'agents_per_zone': 4  # Large-N preset
            },
            'bees': {
                'time_budget_s': 2.5
            },
            'boundary_discipline': {
                'halo_fraction': args.halo,
                'portals_per_node': args.portals_per_node,
                'edge_cap': args.edge_cap
            },
            'local_search': {
                'use_or_opt': True,
                'oropt_gate': args.oropt_gate,
                'kick_period_moves': 400
            },
            'merge_schedule': args.merge_schedule,
            'performance': {
                'workers': args.workers,
                'threads_clamped': args.threads
            }
        },
        'integrator': {
            'method': 'popmusic'
        },
        'termination': {
            'wall_time_s': args.wall
        }
    }


def main():
    parser = argparse.ArgumentParser(description='bee-tsp CLI - Maps to multi_hive_solver.py')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # run subcommand
    run_parser = subparsers.add_parser('run', help='Run TSP solver')
    
    # Core parameters
    run_parser.add_argument('--instance', required=True, help='Instance name (without .tsp)')
    run_parser.add_argument('--wall', type=int, required=True, help='Wall time in seconds')
    run_parser.add_argument('--mode', required=True, choices=['SH', 'MH-Lite'], 
                           help='Solver mode: SH (single-hive) or MH-Lite (multi-hive lite)')
    run_parser.add_argument('--seeds', required=True, help='Comma-separated random seeds')
    run_parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    run_parser.add_argument('--threads', type=int, default=1, help='Number of threads')
    
    # MH-Lite specific parameters
    run_parser.add_argument('--hives', help='Comma-separated hive names (MH-Lite only)')
    run_parser.add_argument('--epoch-s', type=float, default=120.0, help='Epoch duration in seconds')
    run_parser.add_argument('--min-slice-s', type=float, default=60.0, help='Minimum slice duration')
    run_parser.add_argument('--allocator', default='ucb1', help='Allocator method')
    run_parser.add_argument('--improve-eps-pct', type=float, default=0.2, help='Improvement epsilon percentage')
    
    # Configuration parameters
    run_parser.add_argument('--merge-schedule', default='mid,end', help='Merge schedule')
    run_parser.add_argument('--oropt-gate', default='late', help='Or-opt timing gate')
    run_parser.add_argument('--halo', type=float, required=True, help='Halo fraction')
    run_parser.add_argument('--portals-per-node', type=int, required=True, help='Portals per node')
    run_parser.add_argument('--edge-cap', type=int, required=True, help='Edge capacity')
    
    # Output options
    run_parser.add_argument('--output', help='Output JSON file')
    run_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.command != 'run':
        parser.print_help()
        return 1
    
    print(f"bee-tsp CLI wrapper - Running {args.mode} mode")
    print(f"Instance: {args.instance}, Wall time: {args.wall}s")
    
    # Set up environment per Update_17.txt
    os.environ['STRICT_TSPLIB_EVAL'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # Find instance file
    instance_path = f"data/tsplib/{args.instance}.tsp"
    if not Path(instance_path).exists():
        print(f"ERROR: Instance file not found: {instance_path}")
        return 1
    
    # Create configuration
    if args.mode == 'MH-Lite':
        import yaml
        config = create_mh_lite_config(args)
        config_file = f"{args.instance}_{args.mode.lower().replace('-', '_')}_{args.wall}s_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, indent=2)
        print(f"Created MH-Lite config: {config_file}")
    else:  # SH mode
        import yaml
        config = create_sh_config(args)
        config_file = f"{args.instance}_sh_{args.wall}s_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, indent=2)
        print(f"Created SH config: {config_file}")
    
    # Prepare output file
    if not args.output:
        args.output = f"{args.instance}_{args.mode.lower().replace('-', '_')}_{args.wall}s_results.json"
    
    # Build multi_hive_solver.py command
    cmd_args = [
        'python', 'multi_hive_solver.py',
        instance_path,
        '--config', config_file,
        '--time-budget', str(args.wall),
        '--output', args.output
    ]
    
    if args.mode == 'MH-Lite':
        cmd_args.extend(['--multi-hive'])
    
    if args.verbose:
        cmd_args.extend(['--verbose', '--log-events'])
    
    print(f"Executing: {' '.join(cmd_args)}")
    
    # Execute the command
    import subprocess
    try:
        result = subprocess.run(cmd_args, check=True)
        print(f"SUCCESS: Results saved to {args.output}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())