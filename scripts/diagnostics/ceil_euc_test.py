#!/usr/bin/env python3
# CEIL/EUC sentinel test from Update_24R
import sys
import os

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import math
from pathlib import Path

def parse_tsp(p):
    H, C, flag = {}, [], False
    for s in Path(p).read_text().splitlines():
        s = s.strip()
        if not s or s.startswith("COMMENT"):
            continue
        if s.startswith("NODE_COORD_SECTION"):
            flag = True
            continue
        if s.startswith("EOF"):
            break
        if flag:
            parts = s.split()
            if len(parts) >= 3:
                try:
                    int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                except:
                    x = float(parts[0])
                    y = float(parts[1])
                C.append((x, y))
        elif ":" in s:
            k, v = [x.strip() for x in s.split(":", 1)]
            H[k.upper()] = v
    return H.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper(), C

def parse_tour(p):
    T, flag = [], False
    for s in Path(p).read_text().splitlines():
        s = s.strip()
        if not s:
            continue
        if s.startswith("TOUR_SECTION"):
            flag = True
            continue
        if s.startswith("EOF"):
            break
        if flag:
            if s == "-1":
                break
            T += [int(x) for x in s.split()]
    return T

def d_e(a, b):
    return int(round(math.hypot(a[0] - b[0], a[1] - b[1])))

def d_c(a, b):
    return int(math.ceil(math.hypot(a[0] - b[0], a[1] - b[1])))

# Test using pr2392 optimal tour
tsp = "benchmarks/pr2392.tsp"
tour = "data/tsplib/pr2392.opt.tour"

typ, coords = parse_tsp(tsp)
T = parse_tour(tour)
t0 = [i - 1 for i in T]  # Convert to 0-indexed

L_e = L_c = 0
for i in range(len(t0)):
    a = coords[t0[i]]
    b = coords[t0[(i + 1) % len(t0)]]
    L_e += d_e(a, b)
    L_c += d_c(a, b)

print("EDGE_WEIGHT_TYPE in tsp:", typ)
print("EUC_2D re-sum:", L_e, " | CEIL_2D re-sum:", L_c)
print("Difference:", L_c - L_e, "({:.2f}%)".format((L_c - L_e) / L_e * 100))

# Key insight: if current solver results (~486K) are closer to L_c than L_e,
# it suggests the solver is using CEIL_2D internally despite EUC_2D specification
print("\nComparison with current solver results (~486K):")
current_results = [484298, 484386, 486254, 486429, 486484, 486496, 486607]
avg_current = sum(current_results) / len(current_results)
print(f"Current average: {avg_current:.0f}")
print(f"Distance from EUC_2D: {abs(avg_current - L_e):.0f}")
print(f"Distance from CEIL_2D: {abs(avg_current - L_c):.0f}")

if abs(avg_current - L_c) < abs(avg_current - L_e):
    print("*** WARNING: Current results are closer to CEIL_2D than EUC_2D! ***")
    print("*** This suggests the solver is using wrong metric internally! ***")
else:
    print("Current results align with EUC_2D as expected.")