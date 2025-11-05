# test.py — minimal TSPLIB tour verifier (self-contained)
from pathlib import Path
from hashlib import sha256
from bee_tsp.tsplib import load_tsplib

# --- set these two paths ---
TSP  = "data/tsplib/pr2392.tsp"
TOUR = "results/25_10_28_2148_pr2392/seed_999_best.tour"

def read_tsplib_tour_1based(path: str) -> list[int]:
    ids, in_sec = [], False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("NAME","TYPE","DIMENSION","COMMENT","EOF")):
            continue
        if line.startswith("TOUR_SECTION"):
            in_sec = True
            continue
        if not in_sec:
            continue
        if line == "-1":
            break
        for tok in line.replace(",", " ").split():
            if tok.lstrip("+-").isdigit():
                ids.append(int(tok))
    return ids

def tour_len_int_1based(tour: list[int], dist) -> int:
    """Sum edges using 0-based indices expected by tsplib.dist."""
    s = 0
    prev = tour[-1] - 1
    for v in tour:
        u = v - 1
        s += dist(prev, u)
        prev = u
    return s

inst = load_tsplib(TSP)
dist = inst["dist"]        # TSPLIB-integer distance function
n    = inst["n"]
tour = read_tsplib_tour_1based(TOUR)

# basic checks
assert len(tour) == n, f"Tour length {len(tour)} != n={n}"
assert sorted(tour) == list(range(1, n+1)), "Tour is not a 1..n permutation"

L = tour_len_int_1based(tour, dist)
H = sha256(Path(TOUR).read_bytes()).hexdigest()

print("TSPLIB length:", L)
print("SHA256:", H)
