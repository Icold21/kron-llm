"""Rebuild corpus_report.json from the mix files (the build died after writing them)."""
import gzip
import json
from collections import defaultdict
from pathlib import Path

MIX = Path("data/bilingual/mix")
REPORT = Path("data/bilingual/corpus_report.json")

bytes_by = defaultdict(int)
docs_by = defaultdict(int)
for path in sorted(MIX.glob("*.jsonl.gz")):
    lang, domain = path.name[:-len(".jsonl.gz")].split("_", 1)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            bytes_by[(lang, domain)] += len(doc["text"].encode("utf-8"))
            docs_by[(lang, domain)] += 1

en = sum(v for k, v in bytes_by.items() if k[0] == "en")
ru = sum(v for k, v in bytes_by.items() if k[0] == "ru")
total = en + ru
bpt = {"en": 2.917, "ru": 4.964}
tok = {l: sum(v for k, v in bytes_by.items() if k[0] == l) / bpt[l] for l in ("en", "ru")}
ttok = sum(tok.values())

report = {
    "mixed": {l: {d: int(v) for (ll, d), v in bytes_by.items() if ll == l}
              for l in ("en", "ru")},
    "mixed_total_bytes": int(total),
    "ru_share": round(ru / max(1, total), 4),
    "documents": {l: sum(v for k, v in docs_by.items() if k[0] == l)
                  for l in ("en", "ru")},
    "budgets_mb": {"en": 4600, "ru": 5300},
    "dedupe": "exact sha256 + 2 KB prefix",
}
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"docs   {sum(docs_by.values()):,} (EN {report['documents']['en']:,} / "
      f"RU {report['documents']['ru']:,})")
print(f"text   {total/1e9:.2f} GB (EN {en/1e9:.2f} / RU {ru/1e9:.2f}), "
      f"RU byte share {ru/total:.1%}")
print(f"tokens {ttok/1e9:.3f} B, EN {tok['en']/ttok:.1%} / RU {tok['ru']/ttok:.1%}")
for l in ("en", "ru"):
    print(f"  {l}: " + ", ".join(f"{d} {v/1e6:.0f} MB"
                                 for (ll, d), v in sorted(bytes_by.items(), key=lambda x: -x[1])
                                 if ll == l))
