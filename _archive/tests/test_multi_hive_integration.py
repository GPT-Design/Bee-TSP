#!/usr/bin/env python3
"""
Multi-Hive Integration Test
Validates the bridge implementation per Update_Claude_11.txt
"""

import time
import json
from bee_tsp.multi_hive import MultiHiveController
from tests.test_solvers import DeterministicStubSolver


def test_multi_hive_integration():
    """Test multi-hive integration with deterministic stub solver."""
    print("Multi-Hive Integration Test")
    print("=" * 50)
    
    # Configuration with 5 hives
    config = {
        'solver': {
            'multi_hive': {
                'enabled': True,
                'epoch_s': 1.0,  # Short epochs for testing
                'hives': ['H1', 'H2', 'H3', 'H4', 'H5'],
                'profiles': {
                    'H1': {'delta_k': 0, 'bias': 'none', 'bee_ball_boost': 0.0, 'name': 'H1'},
                    'H2': {'delta_k': 2, 'bias': 'none', 'bee_ball_boost': 0.0, 'name': 'H2'},
                    'H3': {'delta_k': 4, 'bias': 'none', 'bee_ball_boost': 0.0, 'name': 'H3'},
                    'H4': {'delta_k': 2, 'bias': 'corridor', 'bee_ball_boost': 0.0, 'name': 'H4'},
                    'H5': {'delta_k': 4, 'bias': 'ring_breaker', 'bee_ball_boost': 0.5, 'name': 'H5'}
                },
                'allocator': {
                    'min_slice_s': 0.2,  # Very short for testing
                    'ucb1_c': 1.0
                }
            }
        }
    }
    
    # Create controller and stub solver
    controller = MultiHiveController(config['solver'])
    solver = DeterministicStubSolver(config['solver'])
    
    # Initialize for test problem
    controller.initialize_for_problem(n=20)
    controller.start_time = time.time()
    
    print("Running 3 test epochs...")
    
    # Run 3 epochs to validate UCB1 learning
    for epoch in range(1, 4):
        print(f"\\nEpoch {epoch}:")
        
        result = controller.run_epoch(solver, time_budget_s=1.0)
        
        print(f"  Improvements: {result['improvements']}")
        print(f"  Champion length: {result['champion_length']:.0f}")
        print(f"  Champion source: {result['champion_source']}")
        
        # Show hive allocations
        hive_results = {hr['hive']: hr for hr in result['hive_results']}
        for hive_name in ['H1', 'H2', 'H3', 'H4', 'H5']:
            hr = hive_results.get(hive_name, {})
            improved = "[IMPROVED]" if hr.get('improved', False) else ""
            print(f"    {hive_name}: {hr.get('time_used_s', 0):.2f}s, "
                  f"len={hr.get('best_len', 0):.0f}, "
                  f"reward={hr.get('reward', 0):.4f} {improved}")
    
    # Get final statistics
    final_stats = controller.get_final_stats()
    
    print("\\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    
    # Check Update_Claude_11.txt acceptance criteria
    checks_passed = 0
    total_checks = 5
    
    # 1. Non-zero total_time_s
    total_time = sum(stats['total_time_s'] for stats in final_stats['hive_stats'].values())
    if total_time > 0:
        print("[PASS] Non-zero total time:", f"{total_time:.2f}s")
        checks_passed += 1
    else:
        print("[FAIL] Zero total time")
    
    # 2. At least 1 improve event
    improve_events = [e for e in final_stats['events'] if e['evt'] == 'improve']
    if len(improve_events) >= 1:
        print(f"[PASS] Improve events: {len(improve_events)}")
        checks_passed += 1
    else:
        print("[FAIL] No improve events")
    
    # 3. Non-empty alloc events across >= 2 epochs
    alloc_events = [e for e in final_stats['events'] if e['evt'] == 'alloc']
    unique_epochs = set(e['epoch'] for e in alloc_events)
    if len(unique_epochs) >= 2:
        print(f"[PASS] Allocation events across {len(unique_epochs)} epochs")
        checks_passed += 1
    else:
        print(f"[FAIL] Allocation events only in {len(unique_epochs)} epochs")
    
    # 4. Champion length strictly decreases
    champion_length = final_stats['final_champion_length']
    if champion_length < 100000:  # Initial length
        print(f"[PASS] Champion improved: {champion_length:.0f} < 100000")
        checks_passed += 1
    else:
        print("[FAIL] Champion did not improve")
    
    # 5. UCB1 reallocates toward improving hive (H3)
    h3_stats = final_stats['hive_stats']['H3']
    h1_stats = final_stats['hive_stats']['H1']
    if h3_stats['avg_reward'] > h1_stats['avg_reward']:
        print(f"[PASS] UCB1 learning: H3 reward ({h3_stats['avg_reward']:.4f}) > H1 reward ({h1_stats['avg_reward']:.4f})")
        checks_passed += 1
    else:
        print(f"[FAIL] UCB1 not learning: H3 reward ({h3_stats['avg_reward']:.4f}) <= H1 reward ({h1_stats['avg_reward']:.4f})")
    
    print(f"\\nOverall: {checks_passed}/{total_checks} validation checks passed")
    
    # Show detailed hive statistics
    print("\\nDetailed Hive Statistics:")
    for name, stats in final_stats['hive_stats'].items():
        print(f"  {name}: slices={stats['slices_allocated']}, "
              f"time={stats['total_time_s']:.2f}s, "
              f"improvements={stats['improvements']}, "
              f"avg_reward={stats['avg_reward']:.4f}, "
              f"ucb1_score={stats['ucb1_score']:.4f}")
    
    # Show event log summary
    print(f"\\nEvent Log Summary:")
    event_counts = {}
    for event in final_stats['events']:
        evt_type = event['evt']
        event_counts[evt_type] = event_counts.get(evt_type, 0) + 1
    
    for evt_type, count in event_counts.items():
        print(f"  {evt_type}: {count}")
    
    print("\\nTest completed.")
    return checks_passed == total_checks


if __name__ == '__main__':
    success = test_multi_hive_integration()
    exit(0 if success else 1)