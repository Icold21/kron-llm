"""Fetch IETF RFCs as an English CS/engineering shard.

RFCs are the primary specification corpus of the internet: plain text, stable
URLs, and a licence that allows redistribution.  They fill the CS/engineering
quota, which Wikipedia categories alone cannot reach.

Usage::

    python scripts/fetch_rfc.py --first 1 --last 9000 --target-mb 120
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.bilingual import normalize, quality, record_shard

SHARDS = Path("data/bilingual/shards")
MANIFEST = Path("data/bilingual/manifest.jsonl")
BASE = "https://www.rfc-editor.org/rfc/rfc{}.txt"


def fetch(number: int, timeout: int = 30) -> str | None:
    req = urllib.request.Request(BASE.format(number),
                                 headers={"User-Agent": "kron-corpus/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    return raw.decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=9000)
    ap.add_argument("--target-mb", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = SHARDS / "rfc_en_cs.jsonl"
    target = int(args.target_mb * 1e6)
    started = time.time()
    kept = written = 0

    with open(out, "w", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, n): n
                       for n in range(args.first, args.last + 1)}
            for fut in as_completed(futures):
                number = futures[fut]
                raw = fut.result()
                if not raw or len(raw) < 2000:
                    continue
                text = normalize(raw)
                st = quality(text, "en", min_chars=2000,
                             max_symbol_ratio=0.15, max_repeat_ratio=0.30)
                if not st.ok:
                    continue
                doc = {"title": f"RFC {number}", "text": text, "language": "en",
                       "domain": "cs", "source": "IETF RFC Editor",
                       "licence": "IETF Trust (see RFC 5378)",
                       "url": BASE.format(number)}
                sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
                written += len(text.encode("utf-8"))
                if kept % 250 == 0:
                    sink.flush()
                    print(f"  {kept} RFCs, {written/1e6:.0f} MB, "
                          f"{time.time()-started:.0f}s", flush=True)
                if written >= target:
                    break
            # Drop the requests that are still queued: breaking out of
            # as_completed does not cancel them, and the pool would otherwise
            # keep fetching thousands of documents past the target.
            for fut in futures:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)

    print(f"[rfc] {kept} documents, {written/1e6:.0f} MB")
    if not kept:
        raise SystemExit("nothing written")
    shard = record_shard(out, MANIFEST, source="IETF RFC Editor",
                         licence="IETF Trust (see RFC 5378)",
                         url="https://www.rfc-editor.org/", language="en",
                         domain="cs", docs=kept,
                         note="primary internet specifications")
    print(f"[done] {kept} docs, {shard.bytes/1e6:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
