"""Fetch textbooks and how-to pages from Wikibooks / Wikiversity, in parallel.

``build_bilingual.py fetch-wiki`` walks categories one request at a time.  That
is fine on a healthy network (0.3 s per request) and unusable on a bad one:
while this was written, a single category request took 5-45 s, so a depth-2
crawl of three shelves spent forty minutes before fetching a single page.

The same walk with eight threads finishes in minutes, because the cost is
latency, not throughput: Wikimedia answers a 50-title batch in the same time as
a single title.  The worker count is deliberately small and the user agent
identifies the project, so the crawl stays polite.

Usage::

    python scripts/fetch_textbooks.py --site wikibooks --lang en \
        --domain science --category "Chemistry,Physics,Biology" \
        --suffix _textbooks --target-mb 150
    python scripts/fetch_textbooks.py --site wikiversity --lang ru \
        --domain math --category "Математика" --target-mb 30
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_bilingual as bb  # noqa: E402
from kronllm.bilingual import quality, record_shard  # noqa: E402

SOURCE = {"wikibooks": "Wikibooks", "wikiversity": "Wikiversity",
          "wikipedia": "Wikipedia", "wikisource": "Wikisource"}
LICENCE = "CC-BY-SA-4.0"

_lock = threading.Lock()
_state = {"requests": 0, "errors": 0}


def api(site: str, lang: str, params: dict, tries: int = 3) -> dict:
    """One MediaWiki call, retried; counts requests for the progress line.

    The timeout is deliberately short: on a healthy link a category request
    returns in 0.3 s, and on the link this was written for (2026-10-30) a
    request either answers in a few seconds or is not going to answer at all.
    Waiting 60 s per attempt, three times, turns one dead category into three
    minutes of a worker's life.
    """
    base = "https://" + bb.WIKI_HOST[site].format(lang=lang) + "/w/api.php"
    query = {"format": "json", "formatversion": "2", **params}
    url = base + "?" + urllib.parse.urlencode(query)
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with _lock:
                _state["requests"] += 1
            return json.loads(bb._get(url, timeout=25, retries=1))
        except Exception as exc:  # noqa: BLE001
            last = exc
            with _lock:
                _state["errors"] += 1
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last))


def one_category(site: str, lang: str, cat: str) -> tuple[list[str], list[str]]:
    """Return ``(pages, subcategories)`` for one category, all continuations."""
    pages: list[str] = []
    subs: list[str] = []
    cont: str | None = None
    while True:
        params = {"action": "query", "list": "categorymembers", "cmtitle": cat,
                  "cmlimit": "500", "cmtype": "page|subcat"}
        if cont:
            params["cmcontinue"] = cont
        data = api(site, lang, params)
        for m in data.get("query", {}).get("categorymembers", []):
            if m.get("ns") == 14:
                subs.append(m["title"])
            elif m.get("ns") == 0:
                pages.append(m["title"])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return pages, subs


def collect(site: str, lang: str, roots: list[str], depth: int, limit: int,
            workers: int) -> list[str]:
    """Breadth-first category walk with one thread per category."""
    titles: list[str] = []
    seen_titles: set[str] = set()
    seen_cats: set[str] = set()
    frontier = [f"Category:{r}" for r in roots]
    for level in range(depth + 1):
        frontier = [c for c in frontier if c not in seen_cats]
        seen_cats.update(frontier)
        if not frontier or len(titles) >= limit:
            break
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda c: _safe_category(site, lang, c),
                                    frontier))
        nxt: list[str] = []
        for pages, subs in results:
            for t in pages:
                if t not in seen_titles and "|" not in t and len(titles) < limit:
                    seen_titles.add(t)
                    titles.append(t)
            nxt.extend(subs)
        print(f"  [level {level}] {len(frontier)} categories -> "
              f"{len(titles):,} titles, {len(nxt)} subcategories "
              f"({_state['requests']} requests, {_state['errors']} errors)",
              flush=True)
        frontier = nxt
    return titles


def _safe_category(site: str, lang: str, cat: str):
    try:
        return one_category(site, lang, cat)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {cat}: {exc}", flush=True)
        return [], []


def fetch_batch(site: str, lang: str, titles: list[str]) -> list[tuple[str, str]]:
    """Fetch up to 50 pages per request (the API limit for non-bots)."""
    out: list[tuple[str, str]] = []
    params = {"action": "query", "prop": "revisions", "rvslots": "main",
              "rvprop": "content", "titles": "|".join(titles),
              "redirects": "1"}
    try:
        data = api(site, lang, params)
    except Exception:  # noqa: BLE001
        return out
    for page in data.get("query", {}).get("pages", []):
        if page.get("missing"):
            continue
        revs = page.get("revisions") or []
        if not revs:
            continue
        content = revs[0].get("slots", {}).get("main", {}).get("content", "")
        if content:
            out.append((page["title"], content))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="wikibooks",
                    choices=["wikibooks", "wikiversity", "wikipedia",
                             "wikisource"])
    ap.add_argument("--lang", default="en", choices=["en", "ru"])
    ap.add_argument("--domain", default="science",
                    choices=["math", "science", "logic", "literature",
                             "encyclopedic", "cs", "ml", "humour"])
    ap.add_argument("--category", required=True,
                    help="comma-separated shelves; on Wikibooks a bare name "
                         "means Subject:<name>")
    ap.add_argument("--suffix", default="_textbooks")
    ap.add_argument("--target-mb", type=float, default=50.0)
    ap.add_argument("--min-chars", type=int, default=600)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-titles", type=int, default=0,
                    help="cap the crawl (0 = derived from --target-mb)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = bb.SHARDS / (f"{args.site}_{args.lang}_{args.domain}"
                       f"{args.suffix}.jsonl")
    if args.force:
        bb.reset_shard(out, bb.MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists (use --force to rebuild)")
        return

    roots = [c.strip() for c in args.category.split(",") if c.strip()]
    if args.site in bb.SUBJECT_SITES:
        roots = [r if r.startswith("Subject:") else f"Subject:{r}"
                 for r in roots]
    target = int(args.target_mb * 1e6)
    limit = args.max_titles or int(args.target_mb * 1e6 / 15_000 * 2.0) + 800

    print(f"[{args.site}/{args.lang}/{args.domain}] shelves {roots}, "
          f"up to {limit:,} titles, {args.workers} workers", flush=True)
    t0 = time.time()
    titles = collect(args.site, args.lang, roots, args.depth, limit, args.workers)
    print(f"[titles] {len(titles):,} in {time.time()-t0:.0f}s", flush=True)

    batches = [titles[i:i + 50] for i in range(0, len(titles), 50)]
    written = kept = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pool.map(
                lambda b: fetch_batch(args.site, args.lang, b), batches):
            for title, raw in chunk:
                text = bb.clean_wikitext(raw)
                if not quality(text, args.lang, min_chars=args.min_chars).ok:
                    continue
                doc = {"title": title, "text": text, "language": args.lang,
                       "domain": args.domain,
                       "source": f"{SOURCE[args.site]} (shelves: "
                                 f"{', '.join(r.replace('Subject:', '') for r in roots)})",
                       "licence": LICENCE,
                       "url": f"https://{args.lang}.{args.site}.org/wiki/"
                              f"{title.replace(' ', '_')}"}
                sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
                written += len(text.encode("utf-8"))
                if kept % 200 == 0:
                    sink.flush()
                    print(f"  {kept} docs, {written/1e6:.1f} MB, "
                          f"{_state['requests']} requests", flush=True)
            if written >= target:
                break
        sink.flush()

    if not kept:
        print("[warn] nothing passed the filters")
        if out.exists():
            out.unlink()
        return
    shard = record_shard(
        out, bb.MANIFEST, source=SOURCE[args.site], licence=LICENCE,
        url=f"https://{args.lang}.{args.site}.org/",
        language=args.lang, domain=args.domain, docs=kept,
        note=f"shelves={roots[:4]} depth={args.depth} threaded")
    print(f"[done] {kept:,} docs ({len(titles):,} titles fetched) -> "
          f"{out.name} ({shard.bytes/1e6:.1f} MB) in "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
