#!/usr/bin/env python3
"""
Test TTT target calculation with tight floor formula.
Verify att48 example: L_opt=10628
- 0.1% → floor(10628*1.001) = 10638
- 0.5% → 10681  
- 1.0% → 10734
"""

def calculate_ttt_target(optimal_length, pct):
    """Calculate TTT target using tight floor formula: T_ε = floor(L_opt * (1 + ε/100) + 1e-9)"""
    return int(optimal_length * (1.0 + pct/100.0) + 1e-9)

def test_att48_targets():
    print("=== TTT Target Calculation Test ===")
    print("Testing att48 (L_opt=10628)")
    
    optimal_length = 10628
    
    # Test cases from user specification
    test_cases = [
        (0.1, 10638),  # 0.1% → 10638
        (0.5, 10681),  # 0.5% → 10681 
        (1.0, 10734),  # 1.0% → 10734
    ]
    
    print(f"\nOptimal length: {optimal_length}")
    print("Percentage | Expected | Calculated | Match")
    print("-" * 40)
    
    all_correct = True
    for pct, expected in test_cases:
        calculated = calculate_ttt_target(optimal_length, pct)
        match = calculated == expected
        all_correct = all_correct and match
        
        print(f"{pct:8.1f}% | {expected:8d} | {calculated:10d} | {'OK' if match else 'FAIL'}")
        
        if not match:
            print(f"  Error: expected {expected}, got {calculated} (diff: {calculated - expected})")
    
    print("-" * 40)
    if all_correct:
        print("All TTT targets calculated correctly!")
    else:
        print("Some TTT targets are incorrect!")
    
    # Also test the formula breakdown
    print(f"\nFormula breakdown for att48:")
    for pct in [0.1, 0.5, 1.0]:
        raw_calc = optimal_length * (1.0 + pct/100.0)
        with_epsilon = raw_calc + 1e-9
        floored = int(with_epsilon)
        print(f"{pct}%: {optimal_length} * (1 + {pct}/100) + 1e-9 = {raw_calc:.10f} + 1e-9 = {with_epsilon:.10f} -> floor = {floored}")

if __name__ == "__main__":
    test_att48_targets()