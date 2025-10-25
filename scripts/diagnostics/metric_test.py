#!/usr/bin/env python3
"""
Test to verify if the solver is using wrong metric (CEIL_2D instead of EUC_2D)
for pr2392 which should be EUC_2D but might be incorrectly using CEIL_2D internally.
"""
import sys
import os
from pathlib import Path

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import math
from bee_tsp.tsplib import load_tsplib

def nint(x: float) -> int:
    return int(x + 0.5)

# Load pr2392
data = load_tsplib("benchmarks/pr2392.tsp")
coords = data["coords"]
expected_metric = data["edge_weight_type"]

print(f"File declares metric: {expected_metric}")
print(f"Number of nodes: {data['n']}")

# Test a few distance calculations manually
test_pairs = [(0, 1), (0, 100), (100, 200)]

print("\nDistance comparison:")
print("Pair  | EUC_2D (nint) | CEIL_2D (ceil) | Difference")
print("------|---------------|----------------|----------")

total_euc = 0
total_ceil = 0

for i, j in test_pairs:
    xi, yi = coords[i]
    xj, yj = coords[j]

    dist_raw = math.hypot(xi - xj, yi - yj)
    dist_euc = nint(dist_raw)    # EUC_2D method
    dist_ceil = math.ceil(dist_raw)  # CEIL_2D method

    print(f"({i:2},{j:3}) |     {dist_euc:8} |      {dist_ceil:8} |    {dist_ceil - dist_euc:+4}")

    total_euc += dist_euc
    total_ceil += dist_ceil

print(f"\nTotal for test edges:")
print(f"EUC_2D:  {total_euc}")
print(f"CEIL_2D: {total_ceil}")
print(f"CEIL is {((total_ceil - total_euc) / total_euc * 100):+.2f}% higher")

# Test what the loaded distance function returns
print(f"\nLoaded distance function test:")
print(f"d(0,1) = {data['dist'](0, 1)}")
print(f"Expected EUC_2D: {nint(math.hypot(coords[0][0] - coords[1][0], coords[0][1] - coords[1][1]))}")
print(f"Expected CEIL_2D: {math.ceil(math.hypot(coords[0][0] - coords[1][0], coords[0][1] - coords[1][1]))}")