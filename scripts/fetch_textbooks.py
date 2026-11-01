"""Fetch textbooks and how-to pages from Wikibooks / Wikiversity, in parallel.

``build_bilingual.py fetch-wiki`` walks categories one request at a time.  That
is fine on a healthy network (0.3 s per request) and unusable on a bad one:
while this was written, a single category request took 5-45 s, and the link
dropped out entirely for minutes at a time.

Two properties make the difference here, and both are about *the link*, not
about speed:

**Patience.**  A request is retried until a deadline (``--request-deadline``,
20 min by default) with exponential backoff, not three times in an eighth of a
second.  Measured on 2026-10-30 the link answered ~1 request in 10 and then
worked normally for a burst: three attempts fail together, twenty rarely do.

**Resumability.**  The category walk is checkpointed after every level and the
already-fetched titles are read back from the output file, so a run that dies
after forty minutes of crawling resumes where it stopped instead of starting
over.  Same for a level: the frontier and the titles seen so far are on disk.

The worker count is deliberately small and the user agent identifies the
project, so the crawl stays polite: eight workers also happen to be what a
latency-bound fetch needs, because Wikimedia answers a 50-title batch in the
same time as a single title.

Usage::

    python scripts/fetch_textbooks.py --site wikibooks --lang en \
        --domain science --category "Chemistry,Physics,Biology" \
        --suffix _textbooks --target-mb 150
    python scripts/fetch_textbooks.py --site wikiversity --lang ru \
        --domain math --category "Математика" --target-mb 30
    python scripts/fetch_textbooks.py --site wikipedia --lang en \
        --domain science --category "Chemistry,Organic chemistry" --suffix _chem
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_bilingual as bb  # noqa: E402
from kronllm.bilingual import quality, record_shard  # noqa: E402

SOURCE = {"wikibooks": "Wikibooks", "wikiversity": "Wikiversity",
          "wikipedia": "Wikipedia", "wikisource": "Wikisource"}
LICENCE = "CC-BY-SA-4.0"

_lock = threading.Lock()
_state = {"requests": 0, "errors": 0, "docs": 0, "bytes": 0}


def api(site: str, lang: str, params: dict, deadline: float = 1200.0,
        timeout: float = 8.0) -> dict:
    """One MediaWiki call, retried until it succeeds or the deadline passes.

    The per-attempt timeout is deliberately short.  A healthy request returns in
    0.3-0.5 s; a dead one hangs until the client gives up, and with a 10 %
    success rate the cost of a request is almost entirely the *failures*.  At
    20 s an attempt costs 18 s on average and a category needs ten of them; at
    8 s the same category takes a third of the time, because the timeout is a
    socket timeout, not a cap on the transfer.  The backoff grows to 30 s so a
    long outage still costs one request a minute rather than sixty.
    """
    base = "https://" + bb.WIKI_HOST[site].format(lang=lang) + "/w/api.php"
    query = {"format": "json", "formatversion": "2", **params}
    url = base + "?" + urllib.parse.urlencode(query)
    started = time.time()
    backoff = 2.0
    attempts = 0
    last: Exception | None = None
    while True:
        attempts += 1
        try:
            with _lock:
                _state["requests"] += 1
            return json.loads(bb._get(url, timeout=timeout, retries=1))
        except Exception as exc:  # noqa: BLE001
            last = exc
            with _lock:
                _state["errors"] += 1
            if time.time() - started > deadline:
                raise RuntimeError(f"{attempts} attempts over "
                                   f"{deadline:.0f}s: {last}") from last
            time.sleep(min(30.0, backoff) * (0.7 + 0.6 * random.random()))
            backoff = min(30.0, backoff * 1.6)


def one_category(site: str, lang: str, cat: str, deadline: float,
                 timeout: float) -> tuple[list[str], list[str]]:
    """Return ``(pages, subcategories)`` for one category, all continuations."""
    pages: list[str] = []
    subs: list[str] = []
    cont: str | None = None
    while True:
        params = {"action": "query", "list": "categorymembers", "cmtitle": cat,
                  "cmlimit": "500", "cmtype": "page|subcat"}
        if cont:
            params["cmcontinue"] = cont
        data = api(site, lang, params, deadline, timeout)
        for m in data.get("query", {}).get("categorymembers", []):
            if m.get("ns") == 14:
                subs.append(m["title"])
            elif m.get("ns") == 0:
                pages.append(m["title"])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return pages, subs


def _safe_category(site: str, lang: str, cat: str, deadline: float,
                   timeout: float):
    try:
        return one_category(site, lang, cat, deadline, timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {cat}: {exc}", flush=True)
        return [], []


def collect(site: str, lang: str, roots: list[str], depth: int, limit: int,
            workers: int, cache: Path, deadline: float,
            timeout: float = 8.0) -> list[str]:
    """Breadth-first category walk, one thread per category, checkpointed.

    The checkpoint holds the level, the frontier still to walk and every title
    found so far.  A rerun of the same command continues from it: on this link
    a depth-2 crawl of three shelves is thousands of requests, and losing the
    walk to one dropped connection would make the whole approach pointless.
    """
    key = {"site": site, "lang": lang, "roots": roots, "depth": depth,
           # Bumped when the meaning of the checkpoint changes, so a crawl
           # started by an older version is redone instead of resumed into a
           # state that version wrote.
           "v": 2}
    state = {"key": key, "level": 0, "titles": [],
             "frontier": [f"Category:{r}" for r in roots],
             "seen_cats": [], "done": False}
    if cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = None
        if saved and saved.get("key") == key:
            state = saved
            if state.get("done"):
                print(f"  [cache] {len(state['titles']):,} titles from "
                      f"{cache.name}", flush=True)
                return state["titles"]
            print(f"  [cache] resuming at level {state['level']} with "
                  f"{len(state['titles']):,} titles", flush=True)

    seen_cats = set(state["seen_cats"])
    titles = list(state["titles"])
    frontier = list(state["frontier"])

    for level in range(state["level"], depth + 1):
        frontier = [c for c in frontier if c not in seen_cats]
        if not frontier or len(titles) >= limit:
            break
        seen_cats.update(frontier)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(
                lambda c: _safe_category(site, lang, c, deadline, timeout),
                frontier))
        nxt: list[str] = []
        seen_titles = set(titles)
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
        state.update(level=level, titles=titles, frontier=nxt,
                     seen_cats=sorted(seen_cats))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(state, ensure_ascii=False),
                         encoding="utf-8")
        frontier = nxt

    state.update(titles=titles, frontier=frontier, seen_cats=sorted(seen_cats),
                 level=depth, done=True)
    cache.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return titles


def fetch_batch(site: str, lang: str, titles: list[str], deadline: float,
                timeout: float = 8.0) -> list[tuple[str, str]]:
    """Fetch up to 50 pages per request (the API limit for non-bots)."""
    out: list[tuple[str, str]] = []
    params = {"action": "query", "prop": "revisions", "rvslots": "main",
              "rvprop": "content", "titles": "|".join(titles),
              "redirects": "1"}
    try:
        data = api(site, lang, params, deadline, timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! batch of {len(titles)}: {exc}", flush=True)
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


def existing_titles(out: Path) -> set[str]:
    """Titles already written to the shard, so a rerun does not redo them."""
    titles: set[str] = set()
    if not out.exists():
        return titles
    with open(out, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                titles.add(json.loads(line).get("title") or "")
            except ValueError:
                continue
    return titles


def forget_manifest_row(path: Path, manifest: Path) -> None:
    """Drop an existing manifest row for this shard.

    The manifest is append-only, so a resumed run would otherwise register the
    same path twice, once with a partial ``docs``/``bytes`` count and once with
    the final one, and ``corpus_report`` would count both.
    """
    if not manifest.exists():
        return
    target = path.resolve()
    keep: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            same = Path(row.get("path", "")).resolve() == target
        except (OSError, ValueError):
            same = False
        if not same:
            keep.append(line)
    manifest.write_text("\n".join(keep) + ("\n" if keep else ""),
                        encoding="utf-8")


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
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--request-timeout", type=float, default=8.0,
                    help="socket timeout per attempt; short because a "
                         "dead request must not hold a worker")
    ap.add_argument("--request-deadline", type=float, default=1200.0,
                    help="seconds to keep retrying one request before giving "
                         "up on it (the link drops out for minutes at a time)")
    ap.add_argument("--force", action="store_true",
                    help="discard the crawl checkpoint and the shard")
    args = ap.parse_args()

    out = bb.SHARDS / (f"{args.site}_{args.lang}_{args.domain}"
                       f"{args.suffix}.jsonl")
    cache = (bb.ROOT / ".cache" /
             f"titles_{args.site}_{args.lang}_{args.domain}"
             f"{(args.suffix or '').lstrip('_')}.json")
    if args.force:
        bb.reset_shard(out, bb.MANIFEST)
        cache.unlink(missing_ok=True)
    elif out.exists():
        print(f"[resume] {out.name} exists (--force to restart)")
    else:
        print(f"[new] {out.name}")

    roots = [c.strip() for c in args.category.split(",") if c.strip()]
    if (args.site, args.lang) in bb.SUBJECT_SITES:
        roots = [r if r.startswith("Subject:") else f"Subject:{r}"
                 for r in roots]
    target = int(args.target_mb * 1e6)
    limit = args.max_titles or int(args.target_mb * 1e6 / 15_000 * 2.0) + 2000

    print(f"[{args.site}/{args.lang}/{args.domain}] shelves {roots}, "
          f"up to {limit:,} titles, {args.workers} workers, "
          f"{args.request_deadline:.0f}s per-request deadline", flush=True)
    t0 = time.time()
    titles = collect(args.site, args.lang, roots, args.depth, limit,
                     args.workers, cache, args.request_deadline,
                     args.request_timeout)
    print(f"[titles] {len(titles):,} in {time.time()-t0:.0f}s", flush=True)

    done = existing_titles(out)
    todo = [t for t in titles if t not in done]
    print(f"[docs] {len(done):,} already fetched, {len(todo):,} to go",
          flush=True)

    batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    kept = 0
    # Bytes already on disk count toward the target: the point of --target-mb
    # is the size of the shard, not of this run, so a resumed crawl stops when
    # the shard is big enough rather than when this run has added that much.
    written = out.stat().st_size if out.exists() else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_batch, args.site, args.lang, b,
                               args.request_deadline, args.request_timeout)
                   for b in batches]
        for future in as_completed(futures):
            for title, raw in future.result():
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
                if kept % 100 == 0:
                    sink.flush()
                    print(f"  {kept} new docs, {written/1e6:.1f} MB this run, "
                          f"{_state['requests']} requests, "
                          f"{_state['errors']} errors, "
                          f"{time.time()-t0:.0f}s", flush=True)
            if written >= target:
                for f in futures:
                    f.cancel()
                break
        sink.flush()

    if not kept and not done:
        print("[warn] nothing passed the filters")
        if out.exists():
            out.unlink()
        return
    if not kept:
        print(f"[done] nothing new; {out.name} unchanged")
        return
    total = sum(1 for line in open(out, encoding="utf-8") if line.strip())
    forget_manifest_row(out, bb.MANIFEST)
    shard = record_shard(
        out, bb.MANIFEST, source=SOURCE[args.site], licence=LICENCE,
        url=f"https://{args.lang}.{args.site}.org/",
        language=args.lang, domain=args.domain, docs=total,
        note=f"shelves={roots[:4]} depth={args.depth} threaded")
    print(f"[done] {kept:,} new docs, {total:,} in the shard ({len(titles):,} "
          f"titles) -> {out.name} ({shard.bytes/1e6:.1f} MB) in "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
