
#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import sys
from typing import List, Tuple, Dict

# ---------------------------
# TSPLIB parsing utilities
# ---------------------------

def parse_tsp(path: str) -> Dict:
    """
    Minimal TSPLIB .tsp parser for EUC_2D / CEIL_2D / ATT with NODE_COORD_SECTION.
    Returns dict with: name, type, dimension, edge_weight_type, coords: List[Tuple[float,float]]
    """
    header = {}
    coords: List[Tuple[float, float]] = []
    in_coords = False
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            raw = line.strip()
            if raw == '' or raw.startswith('COMMENT'):
                continue
            if raw.startswith('NODE_COORD_SECTION'):
                in_coords = True
                continue
            if raw.startswith('EOF'):
                break
            if in_coords:
                parts = raw.split()
                if len(parts) >= 3:
                    # TSPLIB indices are 1-based; we ignore the index and keep order
                    try:
                        _idx = int(parts[0])
                        x = float(parts[1]); y = float(parts[2])
                    except ValueError:
                        # Some files may have non-integer index; try without
                        x = float(parts[0]); y = float(parts[1])
                    coords.append((x, y))
                continue
            if ':' in raw:
                k, v = [x.strip() for x in raw.split(':', 1)]
                header[k.upper()] = v
            else:
                # key value without colon (rare)
                parts = raw.split()
                if len(parts) >= 2:
                    header[parts[0].upper()] = ' '.join(parts[1:])
    name = header.get('NAME', os.path.basename(path))
    ewtype = header.get('EDGE_WEIGHT_TYPE', 'EUC_2D').upper()
    dim = int(header.get('DIMENSION', len(coords)))
    if len(coords) != dim:
        # Some TSPLIB files list dim before coords; ensure consistency if coords parsed
        pass
    return {
        'name': name,
        'type': header.get('TYPE', 'TSP').upper(),
        'dimension': dim,
        'edge_weight_type': ewtype,
        'coords': coords
    }

def parse_tour(path: str, dimension_hint: int = None) -> List[int]:
    """
    Parse TSPLIB .tour with TOUR_SECTION. Returns 1-based city indices tour list (cycle implied).
    """
    tour: List[int] = []
    in_tour = False
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            raw = line.strip()
            if raw == '' or raw.startswith('COMMENT'):
                continue
            if raw.startswith('TOUR_SECTION'):
                in_tour = True
                continue
            if raw.startswith('EOF'):
                break
            if in_tour:
                if raw == '-1':
                    break
                # Some tours can list multiple indices per line
                parts = raw.split()
                for p in parts:
                    v = int(p)
                    tour.append(v)
    if dimension_hint is not None and len(tour) != dimension_hint:
        # Some tours may include only a path of dimension; typical .tour lists all cities once
        if len(tour) == 0:
            raise ValueError("Empty TOUR_SECTION parsed.")
    return tour

# ---------------------------
# TSPLIB integer distance functions
# ---------------------------

def dist_euc_2d(a: Tuple[float,float], b: Tuple[float,float]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(round(math.hypot(dx, dy)))

def dist_ceil_2d(a: Tuple[float,float], b: Tuple[float,float]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return int(math.ceil(math.hypot(dx, dy)))

def dist_att(a: Tuple[float,float], b: Tuple[float,float]) -> int:
    # TSPLIB pseudo-Euclidean (ATT)
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    rij = math.sqrt((dx*dx + dy*dy) / 10.0)
    tij = int(round(rij))
    if tij < rij:
        tij += 1
    return tij

def compute_length(coords: List[Tuple[float,float]], tour_1based: List[int], metric: str) -> int:
    metric = metric.upper()
    if metric == 'EUC_2D':
        dfunc = dist_euc_2d
    elif metric == 'CEIL_2D':
        dfunc = dist_ceil_2d
    elif metric == 'ATT':
        dfunc = dist_att
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE {metric} not implemented in this verifier (supports EUC_2D, CEIL_2D, ATT).")
    n = len(tour_1based)
    # Convert to 0-based
    tour0 = [i-1 for i in tour_1based]
    total = 0
    for i in range(n):
        a = coords[tour0[i]]
        b = coords[tour0[(i+1) % n]]
        total += dfunc(a, b)
    return total

# ---------------------------
# SHA256 helper
# ---------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="TSPLIB .tour verification (integer metrics).")
    ap.add_argument("--tsp", required=True, help="Path to TSPLIB .tsp file (coords).")
    ap.add_argument("--tour", required=True, help="Path to TSPLIB .tour file (tour order).")
    ap.add_argument("--instance", required=False, help="Instance name (for parity triad).")
    ap.add_argument("--metric", required=False, help="Override EDGE_WEIGHT_TYPE (default reads from .tsp).")
    ap.add_argument("--opt", type=int, default=None, help="Known optimal length (integer).")
    ap.add_argument("--best-known", type=int, default=None, help="Known best published length (integer).")
    ap.add_argument("--manifest", default=None, help="If set, append a JSON line with results to this file.")
    args = ap.parse_args()

    tsp = parse_tsp(args.tsp)
    coords = tsp['coords']
    if not coords:
        print("ERROR: No coordinates parsed from .tsp (expected NODE_COORD_SECTION).", file=sys.stderr)
        sys.exit(2)
    metric = (args.metric or tsp['edge_weight_type']).upper()
    tour = parse_tour(args.tour, dimension_hint=tsp['dimension'])
    if len(tour) != tsp['dimension']:
        print(f"WARNING: TOUR length {len(tour)} differs from DIMENSION {tsp['dimension']}. Proceeding.", file=sys.stderr)
    length = compute_length(coords, tour, metric)
    sha = sha256_file(args.tour)

    inst = args.instance or tsp['name']
    best_known = args.best_known
    opt = args.opt
    # Parity logic: OK only if opt provided and equals length; otherwise if best-known equals length, report MATCH_BEST.
    parity = "UNKNOWN"
    if opt is not None:
        parity = "OK" if length == opt else "FAIL"
    elif best_known is not None:
        parity = "MATCH_BEST" if length == best_known else "NEQ_BEST"

    # Print summary
    print("=== TSPLIB Verification Summary ===")
    print(f"Instance     : {inst}")
    print(f"Metric       : {metric}")
    print(f"Tour length  : {length}")
    print(f"SHA256(.tour): {sha}")
    if opt is not None:
        print(f"Opt provided : {opt}")
    if best_known is not None:
        print(f"Best-known   : {best_known}")
    print(f"Parity       : {parity}")

    # Parity triad (single-line)
    triad = {
        "instance": inst,
        "metric": metric,
        "length_tsplib": length,
        "opt": opt,
        "best_known": best_known,
        "parity": parity
    }
    print("PARITY_TRIAD:", json.dumps(triad, separators=(',',':')))

    # Manifest append (optional)
    if args.manifest:
        rec = {
            "instance": inst,
            "metric": metric,
            "tour_path": os.path.abspath(args.tour),
            "tour_sha256": sha,
            "length_tsplib": length,
            "opt": opt,
            "best_known": best_known,
            "parity": parity
        }
        with open(args.manifest, 'a', encoding='utf-8') as mf:
            mf.write(json.dumps(rec) + "\n")
        print(f"[manifest] appended to {args.manifest}")

if __name__ == "__main__":
    main()
