from pathlib import Path
import re
from bee_tsp.tsplib import load_tsplib

TSP  = "data/tsplib/pr2392.tsp"
TOUR = "results/25_10_28_2148_pr2392/seed_999_best.tour"  # change to any .tour you want

def read_1based_ids_and_comment_len(p: str):
    txt = Path(p).read_text()
    m = re.search(r"COMMENT\s*:\s.*length\s*=\s*([0-9]+)", txt)
    comment_len = int(m.group(1)) if m else None
    ids = []
    in_sec = False
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("TOUR_SECTION"): in_sec = True; continue
        if not in_sec: continue
        if s == "-1": break
        ids.extend(int(tok) for tok in s.replace(",", " ").split() if tok.lstrip("+-").isdigit())
    return ids, comment_len

def loop_len_1based(ids, dist):
    s = 0
    prev = ids[-1] - 1
    for v in ids:
        u = v - 1
        s += dist(prev, u)
        prev = u
    return s

inst = load_tsplib(TSP)
ids, comment_len = read_1based_ids_and_comment_len(TOUR)
calc_len = loop_len_1based(ids, inst["dist"])
print("COMMENT length:", comment_len)
print("Recomputed TSPLIB length:", calc_len)
print("MATCH:", comment_len == calc_len)