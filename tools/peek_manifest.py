# tools/peek_manifest.py
import json
from pathlib import Path
m = Path(r"results/manifest.jsonl")
rows = [json.loads(x) for x in m.read_text().splitlines() if x.strip()]
print(f"[manifest] {len(rows)} lines")
# Show the last few
for r in rows[-6:]:
    print({k:r.get(k) for k in ["instance","mode","wall_s","seed","length_tsplib","tour_path"]})
# Group count by (instance, wall, mode)
from collections import Counter
cnt = Counter((r.get("instance"), r.get("wall_s"), r.get("mode")) for r in rows)
print("\n[group counts]")
for k,v in sorted(cnt.items()):
    print(v, k)
