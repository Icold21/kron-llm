"""Deduplicate the mix files in place.

Pass 2 of the streaming build selected documents by SHA but wrote *every*
occurrence of a selected hash, so 19 % of the bytes were repeats of documents
already in the file.  This keeps the first occurrence and drops the rest.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path

MIX = Path("data/bilingual/mix")

seen: set[str] = set()
total_in = total_out = 0
docs_in = docs_out = 0
for path in sorted(MIX.glob("*.jsonl.gz")):
    tmp = path.with_suffix(".tmp")
    kept = dropped = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh, \
            gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            docs_in += 1
            doc = json.loads(line)
            text = doc["text"]
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            total_in += len(text.encode("utf-8"))
            if sha in seen:
                dropped += 1
                continue
            seen.add(sha)
            out.write(line + "\n")
            kept += 1
            docs_out += 1
            total_out += len(text.encode("utf-8"))
    os.replace(tmp, path)
    print(f"{path.name:34s} kept {kept:>7,}  dropped {dropped:>6,}")

print(f"\ndocuments {docs_in:,} -> {docs_out:,}")
print(f"text      {total_in/1e9:.2f} GB -> {total_out/1e9:.2f} GB")
