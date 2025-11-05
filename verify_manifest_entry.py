# verify_manifest_entry.py
from pathlib import Path
from hashlib import sha256
import json, sys
from bee_tsp.tsplib import load_tsplib

# USAGE:
#   python verify_manifest_entry.py data/tsplib/pla33810.tsp results/.../seed_999_best.tour results/manifest.jsonl
TSP, TOUR, MANIFEST = sys.argv[1], sys.argv[2], sys.argv[3]

def tour_len_1based(ids, dist):
    s, prev = 0, ids[-1]-1
    for v in ids:
        u = v-1
        s += dist(prev, u)
        prev = u
    return s

def read_tour_1based(p):
    ids, in_sec = [], False
    for raw in Path(p).read_text().splitlines():
        s = raw.strip()
        if s.startswith("TOUR_SECTION"): in_sec = True; continue
        if not in_sec: continue
        if s == "-1": break
        for tok in s.replace(",", " ").split():
            if tok.lstrip("+-").isdigit(): ids.append(int(tok))
    return ids

# recompute from TSPLIB
inst = load_tsplib(TSP)
ids  = read_tour_1based(TOUR)
calc = tour_len_1based(ids, inst["dist"])
h    = sha256(Path(TOUR).read_bytes()).hexdigest()

# find manifest row for this tour (by suffix match)
row = None
with open(MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("tour_path") and str(obj["tour_path"]).endswith(Path(TOUR).as_posix()):
            row = obj
            break

print("Recomputed length:", calc)
print("SHA256:", h)
if row is None:
    print("Manifest row: NOT FOUND for this tour_path")
else:
    ok_len  = (row.get("length_tsplib") == calc)
    ok_sha  = (row.get("tour_sha256") == h)
    print("Manifest length:", row.get("length_tsplib"), "OK" if ok_len else "MISMATCH")
    print("Manifest SHA256:", row.get("tour_sha256"), "OK" if ok_sha else "MISMATCH")
    print("MATCH:", ok_len and ok_sha)
