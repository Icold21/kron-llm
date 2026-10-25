"""Import Rick and Morty episode transcripts from the Fandom wiki API.

The wiki's HTML pages are behind a 403, but ``api.php`` answers, and every
episode has a ``<Episode>/Transcript`` page.  Content is CC BY-SA, so it is a
cleaner source than a scraped dataset.

Usage::

    python scripts/import_rm_fandom.py --max-episodes 120
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.bilingual import append_jsonl, normalize, record_shard

API = "https://rickandmorty.fandom.com/api.php"
UA = {"User-Agent": "KronLLM-research/0.1 (corpus; github.com/Icold21/kron-llm)"}
ROOT = Path("data/bilingual")
SHARDS = ROOT / "shards"
MANIFEST = ROOT / "manifest.jsonl"

_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_FILE = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.I)
_RE_LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_RE_TAG = re.compile(r"<[^>]+>")
_RE_HEAD = re.compile(r"^=+.*?=+\s*$", re.M)


def api(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def clean_wikitext(text: str) -> str:
    prev = None
    while prev != text:                     # nested templates
        prev, text = text, _RE_TEMPLATE.sub("", text)
    text = _RE_FILE.sub("", text)
    text = _RE_LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = _RE_TAG.sub("", text)
    text = _RE_HEAD.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="humour")
    ap.add_argument("--max-episodes", type=int, default=120)
    ap.add_argument("--min-chars", type=int, default=1500)
    args = ap.parse_args()

    titles: list[str] = []
    cont: dict = {}
    while True:
        params = {"action": "query", "list": "categorymembers",
                  "cmtitle": "Category:Episodes", "cmlimit": 500,
                  "cmtype": "page", "format": "json", **cont}
        data = api(params)
        titles.extend(m["title"] for m in data["query"]["categorymembers"])
        cont = data.get("continue") or {}
        if not cont:
            break
    titles = titles[: args.max_episodes]
    print(f"[fandom] {len(titles)} episode pages")

    docs: list[dict] = []
    for i, title in enumerate(titles, 1):
        page = f"{title}/Transcript"
        try:
            data = api({"action": "parse", "page": page, "prop": "wikitext",
                        "format": "json"})
            raw = data["parse"]["wikitext"]["*"]
        except Exception:                      # missing page / parse error
            continue
        text = clean_wikitext(raw)
        if len(text) < args.min_chars:
            continue
        docs.append({"title": page, "text": text, "language": "en",
                     "domain": args.domain, "source": "Fandom wiki (api.php)",
                     "licence": "CC-BY-SA-3.0",
                     "url": "https://rickandmorty.fandom.com/wiki/" + page,
                     "note": "easter egg: series dialogue"})
        if i % 10 == 0:
            print(f"  {i}/{len(titles)} pages, {len(docs)} kept", flush=True)
        time.sleep(0.15)

    out = SHARDS / f"fandom_rm_{args.domain}.jsonl"
    if out.exists():
        out.unlink()
    n = append_jsonl(out, docs)
    shard = record_shard(out, MANIFEST, source="Fandom wiki (api.php)",
                         licence="CC-BY-SA-3.0",
                         url="https://rickandmorty.fandom.com/",
                         language="en", domain=args.domain, docs=n,
                         note="episode transcripts, easter egg")
    print(f"[done] {n} transcripts, {shard.bytes/1e6:.2f} MB -> {out.name}")


if __name__ == "__main__":
    main()
