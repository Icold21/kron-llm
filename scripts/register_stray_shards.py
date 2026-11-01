"""Register the Habr and Stack Overflow shards that a killed fetch never recorded.

``fetch-hf`` writes the manifest entry only after the whole target is reached,
so killing it mid-way leaves a complete file on disk that no build will ever
read.  This truncates a possibly half-written last line, counts the documents
and records the shard properly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kronllm.bilingual import read_manifest, record_shard  # noqa: E402

SHARDS = ROOT / "data" / "bilingual" / "shards"
MANIFEST = ROOT / "data" / "bilingual" / "manifest.jsonl"

TODO = [
    ("hf_habr_ru_cs.jsonl", "ru", "cs",
     "Habr (HF georgiyozhegov/habr)", "Habr content, see dataset card"),
    ("hf_so_en_cs.jsonl", "en", "cs",
     "Stack Overflow (HF bigcode/stackoverflow-clean)", "CC-BY-SA-4.0"),
]

recorded = {Path(r.path).name for r in read_manifest(MANIFEST)}
for name, lang, domain, source, licence in TODO:
    path = SHARDS / name
    if not path.exists():
        print(f"[miss] {name}")
        continue
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n")
        path.write_bytes(raw[: cut + 1])
        print(f"[fix]  {name}: dropped {len(raw) - cut - 1} trailing bytes")
    docs = sum(1 for _ in path.open(encoding="utf-8"))
    if name in recorded:
        print(f"[skip] {name} already in the manifest ({docs:,} docs)")
        continue
    record_shard(path, MANIFEST, source=source, licence=licence, url="",
                 language=lang, domain=domain, docs=docs,
                 note="fetch interrupted after the target was reached; recorded by hand")
    print(f"[add]  {name}: {docs:,} docs, {path.stat().st_size/1e6:.0f} MB -> {lang}/{domain}")
