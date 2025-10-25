# tools/emit_tour_from_json.py
import json, sys
from pathlib import Path

def write_tour(dirpath: Path, basename: str, tour_1based):
    out = dirpath/f"{basename}.tour"
    with out.open("w") as f:
        f.write("NAME : " + basename + "\nTYPE : TOUR\nDIMENSION : {}\n".format(len(tour_1based)))
        f.write("TOUR_SECTION\n")
        for v in tour_1based: f.write(str(int(v))+"\n")
        f.write("-1\nEOF\n")
    print("[emit] wrote", out)

if __name__=="__main__":
    run_dir = Path(sys.argv[1])  # e.g., results/25_9_23_1226_pla33810
    base = run_dir.name
    js = run_dir/(base + ".json")
    data = json.loads(js.read_text())
    # adjust these keys to whatever your summary uses
    tour = data.get("best_tour_1based") or data.get("champion_tour") or data.get("tour_indices")
    if not tour:
        sys.exit("No tour array in " + str(js))
    write_tour(run_dir, base, tour)
