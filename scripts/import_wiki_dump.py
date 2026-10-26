"""Import a MediaWiki ``pages-articles`` dump as a corpus shard.

The category-based fetchers in ``build_bilingual.py`` cover the domains that
matter for the science accent, but they cannot supply enough Russian. A full
dump is the cheapest high-quality source: CC BY-SA, one article per title, no
navigation residue.

Titles already present in the curated shards are skipped, so the bulk shard is
disjoint from them and a later MinHash pass is not needed for the dump itself.

Usage::

    python scripts/import_wiki_dump.py --dump data/dumps/ruwiki.xml.bz2 \
        --site wikipedia --lang ru --domain encyclopedic --target-mb 2600
"""
from __future__ import annotations

import argparse
import bz2
import json
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.bilingual import (clean_wikitext, normalize, quality,
                               record_shard)

ROOT = Path("data/bilingual")
SHARDS = ROOT / "shards"
MANIFEST = ROOT / "manifest.jsonl"


def _tag(elem: ET.Element) -> str:
    return elem.tag.rsplit("}", 1)[-1]


def existing_titles(site: str, lang: str) -> set[str]:
    """Titles already covered by the curated shards of this wiki."""
    titles: set[str] = set()
    for path in SHARDS.glob(f"{site}_{lang}_*.jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    titles.add(json.loads(line).get("title", ""))
    return titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--site", default="wikipedia",
                    choices=["wikipedia", "wikisource"])
    ap.add_argument("--lang", default="ru", choices=["en", "ru"])
    ap.add_argument("--domain", default="encyclopedic")
    ap.add_argument("--target-mb", type=float, default=2600.0)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else \
        SHARDS / f"{args.site}_{args.lang}_bulk.jsonl"
    target = int(args.target_mb * 1e6)
    skip = existing_titles(args.site, args.lang)
    print(f"[dump] {args.dump}: target {args.target_mb:.0f} MB, "
          f"{len(skip):,} titles already covered")

    started = time.time()
    docs = 0
    written = 0
    seen: set[str] = set()
    last_report = time.time()

    with bz2.open(args.dump, "rb") as fh, open(out, "w", encoding="utf-8") as dst:
        for event, elem in ET.iterparse(fh, events=("end",)):
            if _tag(elem) != "page":
                continue
            title = ns = None
            text = ""
            for child in elem:
                tag = _tag(child)
                if tag == "title":
                    title = (child.text or "").strip()
                elif tag == "ns":
                    ns = (child.text or "").strip()
                elif tag == "revision":
                    for rev in child:
                        if _tag(rev) == "text":
                            text = rev.text or ""
            elem.clear()

            if not title or ns != "0" or not text:
                continue
            if text.lstrip().lower().startswith("#redirect"):
                continue
            if title in skip or title in seen:
                continue

            body = clean_wikitext(text)
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
                "source": f"{args.site} dump (pages-articles)",
                "licence": "CC-BY-SA-4.0",
                "url": f"https://{args.lang}.{args.site}.org/",
                "note": "bulk dump import; curated titles skipped",
            }
            dst.write(json.dumps(doc, ensure_ascii=False) + "\n")
            seen.add(title)
            docs += 1
            written += len(body.encode("utf-8"))

            if time.time() - last_report > 30:
                rate = written / max(1, time.time() - started) / 1e6
                print(f"  {docs:,} docs, {written/1e6:.0f} MB, "
                      f"{rate:.1f} MB/s", flush=True)
                last_report = time.time()
            if written >= target:
                break

    print(f"[dump] stopping at {written/1e6:.0f} MB after {docs:,} docs "
          f"({time.time()-started:.0f}s)")
    if not docs:
        raise SystemExit("nothing written")

    shard = record_shard(out, MANIFEST,
                         source=f"{args.site} dump (pages-articles)",
                         licence="CC-BY-SA-4.0",
                         url=f"https://{args.lang}.{args.site}.org/",
                         language=args.lang, domain=args.domain, docs=docs,
                         note="bulk dump import; curated shard titles skipped")
    print(f"[done] {docs:,} docs, {shard.bytes/1e6:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
