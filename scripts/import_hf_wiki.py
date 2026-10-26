"""Import a Hugging Face Wikimedia parquet dump as a corpus shard.

The dump mirrors (``dumps.wikimedia.org``) were transferring at 0.3-1.2 MB/s,
which makes a 5.6 GB file a multi-hour download.  The official
``wikimedia/wikipedia`` and ``wikimedia/wikisource`` datasets ship the same
articles as plain text parquet shards over the HF CDN, so there is no wikitext
to clean and the download is much faster.

Titles already covered by the curated shards are skipped, so the bulk shard is
disjoint from them.

Usage::

    python scripts/import_hf_wiki.py --dataset wikimedia/wikipedia \
        --config 20231101.ru --lang ru --domain encyclopedic --target-mb 2200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.bilingual import normalize, quality, record_shard

ROOT = Path("data/bilingual")
SHARDS = ROOT / "shards"
MANIFEST = ROOT / "manifest.jsonl"
CACHE = Path("data/hf_cache")


def download(dataset: str, name: str) -> Path:
    """Download one parquet file, retrying transient CDN disconnects."""
    last: Exception | None = None
    for attempt in range(6):
        try:
            return Path(hf_hub_download(dataset, name, repo_type="dataset",
                                        cache_dir=str(CACHE)))
        except Exception as exc:                      # noqa: BLE001
            last = exc
            wait = 5 * (attempt + 1)
            print(f"  [retry {attempt + 1}/6] {name}: {exc}; wait {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"could not download {name}") from last


def existing_titles(prefixes: tuple[str, ...]) -> set[str]:
    titles: set[str] = set()
    for prefix in prefixes:
        for path in SHARDS.glob(prefix):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        titles.add(json.loads(line).get("title", ""))
    return titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", default="20231101.ru")
    ap.add_argument("--site", default="wikipedia")
    ap.add_argument("--lang", default="ru", choices=["en", "ru"])
    ap.add_argument("--domain", default="encyclopedic")
    ap.add_argument("--target-mb", type=float, default=2200.0)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--licence", default="CC-BY-SA-4.0")
    ap.add_argument("--note", default="bulk import; curated shard titles skipped")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else \
        SHARDS / f"{args.site}_{args.lang}_bulk.jsonl"
    target = int(args.target_mb * 1e6)

    files = [f for f in list_repo_files(args.dataset, repo_type="dataset")
             if f.startswith(args.config + "/") and f.endswith(".parquet")]
    files.sort()
    print(f"[hf] {args.dataset} {args.config}: {len(files)} parquet files, "
          f"target {args.target_mb:.0f} MB")

    skip = existing_titles((f"{args.site}_{args.lang}_*.jsonl",))
    seen: set[str] = set()
    docs = 0
    written = 0
    started = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)

    # Resume: a killed import keeps its documents and is topped up.
    mode = "w"
    if out.exists() and out.stat().st_size > 0:
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                seen.add(doc["title"])
                written += len(doc["text"].encode("utf-8"))
        docs = len(seen)
        mode = "a"
        print(f"[resume] {docs:,} docs, {written/1e6:.0f} MB already in {out}")

    with open(out, mode, encoding="utf-8") as dst:
        for name in files:
            if written >= target:
                break
            path = download(args.dataset, name)
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=2000):
                data = batch.to_pydict()
                texts = data.get("text", [])
                titles = data.get("title") or [""] * len(texts)
                for title, text in zip(titles, texts):
                    title = (title or "").strip()
                    text = (text or "").strip()
                    if not text:
                        continue
                    if not title:
                        # Some datasets (cosmopedia) have no title column; use
                        # the opening words as a stable identifier.
                        title = " ".join(text.split()[:8])[:80]
                    if title in skip or title in seen:
                        continue
                    body = normalize(text)
                    if len(body) < args.min_chars:
                        continue
                    st = quality(body, args.lang, min_chars=args.min_chars,
                                 max_symbol_ratio=0.12, max_repeat_ratio=0.25)
                    if not st.ok:
                        continue
                    doc = {
                        "title": title,
                        "text": body,
                        "language": args.lang,
                        "domain": args.domain,
                        "source": f"{args.site} dump ({args.dataset})",
                        "licence": args.licence,
                        "url": f"https://{args.lang}.{args.site}.org/",
                        "note": args.note,
                    }
                    dst.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    seen.add(title)
                    docs += 1
                    written += len(body.encode("utf-8"))
                    if written >= target:
                        break
                if written >= target:
                    break
            print(f"  {name.split('/')[-1]}: {docs:,} docs total, "
                  f"{written/1e6:.0f} MB, {time.time()-started:.0f}s",
                  flush=True)

    print(f"[hf] stopping at {written/1e6:.0f} MB after {docs:,} docs")
    if not docs:
        raise SystemExit("nothing written")
    shard = record_shard(out, MANIFEST,
                         source=f"{args.dataset} ({args.site} dump)",
                         licence=args.licence,
                         url=f"https://{args.lang}.{args.site}.org/",
                         language=args.lang, domain=args.domain, docs=docs,
                         note=args.note)
    print(f"[done] {docs:,} docs, {shard.bytes/1e6:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
