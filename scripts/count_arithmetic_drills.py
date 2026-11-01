"""How many arithmetic drills are actually in the pretraining mix?

A "drill" is a worked example: `17 + 26 = 43`, `12 * 7 = 84`.  The corpus has a
lot of mathematics, but mathematics prose (definitions, proofs, notation) is not
the same thing: exact arithmetic is learned from short input-output pairs with
verified answers, and SFT has 1.4M of those.

This samples documents per shard and counts:
  eq3  -- `a op b = c` (a drill with an explicit answer)
  eq2  -- `a op b` without an equals sign (an expression, not a drill)
"""
from __future__ import annotations

import gzip
import json
import random
import re
from collections import defaultdict
from pathlib import Path

MIX = Path("data/bilingual/mix")
PER_SHARD = 4000
EQ3 = re.compile(r"\b\d{1,4}\s*[+\-−*/×·÷]\s*\d{1,4}\s*=\s*-?\d{1,5}\b")
EQ2 = re.compile(r"\b\d{1,4}\s*[+\-−*/×·÷]\s*\d{1,4}\b")

stats = defaultdict(lambda: {"chars": 0, "eq3": 0, "eq2": 0, "docs": 0})
rng = random.Random(7)

for path in sorted(MIX.glob("*.jsonl.gz")):
    if "_chat." in path.name:
        continue
    lang, domain = path.name.split("_")[0], path.name.split("_", 1)[1].split(".")[0]
    keep = set(rng.sample(range(200_000), PER_SHARD))
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i not in keep:
                continue
            try:
                text = json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                continue
            s = stats[(lang, domain)]
            s["chars"] += len(text)
            s["docs"] += 1
            s["eq3"] += len(EQ3.findall(text))
            s["eq2"] += len(EQ2.findall(text))

total_chars = sum(v["chars"] for v in stats.values())
total_eq3 = sum(v["eq3"] for v in stats.values())
total_eq2 = sum(v["eq2"] for v in stats.values())

print(f"{total_chars/1e9:.2f} G chars sampled "
      f"({sum(v['docs'] for v in stats.values()):,} documents)")
print(f"drills (a op b = c): {total_eq3:,}  -> "
      f"{1e6 * total_eq3 / total_chars:.1f} per million characters")
print(f"bare expressions:    {total_eq2:,}  -> "
      f"{1e6 * total_eq2 / total_chars:.1f} per million characters")
print(f"drill share of documents: "
      f"{100 * sum(1 for v in stats.values() if v['eq3']) / max(1, len(stats)):.0f} %"
      " of (lang, domain) cells contain any")

print("\nper cell (drills per million chars):")
for (lang, dom), v in sorted(stats.items(), key=lambda kv: -kv[1]["eq3"]):
    if not v["chars"]:
        continue
    print(f"  {lang}/{dom:14s} {1e6 * v['eq3'] / v['chars']:8.1f}  "
          f"({v['eq3']:>6,} drills in {v['chars']/1e6:6.1f} M chars)")

# the SFT reservoir, for scale
print("\nfor comparison, the SFT arithmetic slice: 1.4M generated problems "
      "(scripts/gen_arithmetic_sft.py), answers computed and verified")
