"""Audit the quota tables: sums, and what the budget actually buys."""
import collections
import json
from pathlib import Path

from kronllm.bilingual import QUOTAS_EN, QUOTAS_RU

BUDGET = {"en": 4800e6, "ru": 5600e6}
BPT = {"en": 2.917, "ru": 4.964}

rows = [json.loads(l) for l in
        Path("data/bilingual/manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()]
seen = {}
for r in rows:
    seen[r["path"]] = r
avail = collections.Counter()
for r in seen.values():
    avail[(r["language"], r["domain"])] += int(r.get("bytes", 0) or 0)

tables = {"en": QUOTAS_EN, "ru": QUOTAS_RU}
for lang, table in tables.items():
    total = sum(table.values())
    print(f"\n=== {lang}: sum {total:.3f} "
          f"{'OK' if abs(total - 1.0) < 1e-9 else 'WRONG'}")
    print(f"{'domain':14s}{'quota':>7s}{'want MB':>10s}{'have MB':>10s}{'take MB':>10s}")
    got = 0
    for dom, q in sorted(table.items(), key=lambda x: -x[1]):
        want = BUDGET[lang] * q
        have = avail[(lang, dom)]
        take = min(want, have)
        got += take
        flag = "" if take >= want * 0.95 else f"  <- shortfall {take/1e6:.0f}/{want/1e6:.0f}"
        print(f"{dom:14s}{q:7.2f}{want/1e6:9.0f}{have/1e6:9.0f}{take/1e6:9.0f}{flag}")
    print(f"{'':14s}{'':7s}{'':9s}{'':9s}{got/1e6:9.0f}  total for {lang}")

en = min(BUDGET["en"], sum(avail[("en", d)] for d in QUOTAS_EN))
ru = min(BUDGET["ru"], sum(avail[("ru", d)] for d in QUOTAS_RU))
tok = {"en": en / BPT["en"], "ru": ru / BPT["ru"]}
tot = sum(tok.values())
print(f"\nbudget ceiling: EN {en/1e9:.2f} GB -> {tok['en']/1e9:.3f}B tokens, "
      f"RU {ru/1e9:.2f} GB -> {tok['ru']/1e9:.3f}B")
print(f"total {tot/1e9:.3f}B, EN {tok['en']/tot:.1%} / RU {tok['ru']/tot:.1%} "
      f"| steps {tot/16384:,.0f}")
