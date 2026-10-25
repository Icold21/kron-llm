"""Build the bilingual English/Russian pretraining corpus (Phase 1 pilot).

Usage::

    python scripts/build_bilingual.py fetch-wiki --lang ru --domain math \
        --target-mb 40
    python scripts/build_bilingual.py fetch-wiki --lang en --domain logic \
        --target-mb 10
    python scripts/build_bilingual.py fetch-gutenberg --target-mb 40
    python scripts/build_bilingual.py fetch-wikisource --target-mb 20
    python scripts/build_bilingual.py fetch-code --target-mb 15
    python scripts/build_bilingual.py build --target-mb 250
    python scripts/build_bilingual.py tokenizer --vocab 8192

Every fetcher writes newline-delimited JSON documents (one per article, book or
file) plus a manifest row with source, licence, URL and SHA-256.  The ``build``
step applies deduplication, the 60/40 language mix, domain quotas and a held-out
split, and writes a report with the measured composition.

Licences are recorded, never inferred:
  * Wikipedia / Wikisource -- CC BY-SA 4.0 (attribution required);
  * Project Gutenberg -- public domain in the US;
  * Python standard library -- PSF-2.0.
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
from typing import Iterable, Iterator, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.bilingual import (DOMAIN_QUOTAS, append_jsonl, clean_wikitext,
                               corpus_report, dedup, mix_corpus, normalize,
                               pack_shard, quality, quota_mix, read_manifest,
                               record_shard, write_heldout)
from kronllm.bilingual import Shard

UA = {"User-Agent": "KronLLM-research/0.1 (bilingual corpus pilot; "
                    "https://github.com/Icold21/kron_llm)"}
ROOT = Path("data/bilingual")
RAW = ROOT / "raw"
SHARDS = ROOT / "shards"
MANIFEST = ROOT / "manifest.jsonl"

#: Seed categories per language and domain.  Depth-2 recursion covers enough
#: subcategories to yield tens of thousands of articles per root.
CATEGORIES: dict[str, dict[str, list[str]]] = {
    "ru": {
        "math": ["Математика", "Алгебра", "Математический анализ",
                 "Геометрия", "Теория вероятностей"],
        "science": ["Физика", "Химия", "Биология", "Астрономия",
                    "Наука"],
        "logic": ["Логика", "Теория множеств", "Основания математики"],
        "literature": ["Литература", "Русская литература", "Поэзия"],
        "encyclopedic": ["География", "История", "Культура", "Техника"],
    },
    "en": {
        "math": ["Mathematics", "Algebra", "Mathematical analysis",
                 "Geometry", "Probability theory"],
        "science": ["Physics", "Chemistry", "Biology", "Astronomy", "Science"],
        "logic": ["Logic", "Set theory", "Mathematical logic"],
        "literature": ["Literature", "Poetry", "Novels"],
        "encyclopedic": ["Geography", "History", "Culture", "Technology"],
    },
}

WIKI_HOST = {"wikipedia": "{lang}.wikipedia.org",
             "wikisource": "{lang}.wikisource.org"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    """GET with a polite user agent and bounded retries."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:                      # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last


def reset_shard(path: Path, manifest: Path) -> None:
    """Delete a shard and its manifest rows (the meaning of ``--force``)."""
    if path.exists():
        path.unlink()
    if not manifest.exists():
        return
    keep = []
    target = path.resolve()
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


def _api(site: str, lang: str, params: dict) -> dict:
    """Call the MediaWiki API and return parsed JSON."""
    base = "https://" + WIKI_HOST[site].format(lang=lang) + "/w/api.php"
    params = {"format": "json", "formatversion": "2", **params}
    return json.loads(_get(base + "?" + urllib.parse.urlencode(params)))


# ---------------------------------------------------------------------------
# MediaWiki
# ---------------------------------------------------------------------------

def collect_titles(site: str, lang: str, roots: Sequence[str], depth: int = 2,
                   limit: int = 20000) -> list[str]:
    """Collect article titles by walking categories breadth-first.

    Args:
        site: ``wikipedia`` or ``wikisource``.
        lang: ``en`` or ``ru``.
        roots: seed category names, without the ``Category:`` prefix.
        depth: recursion depth into subcategories.
        limit: stop after this many titles.

    Returns:
        Unique article titles (namespaces 0), in discovery order.
    """
    seen_cats: set[str] = set()
    titles: list[str] = []
    seen_titles: set[str] = set()
    frontier = [f"Category:{c}" for c in roots]

    def add(t: str) -> None:
        if t not in seen_titles and "|" not in t and len(titles) < limit:
            seen_titles.add(t)
            titles.append(t)

    for level in range(depth + 1):
        next_frontier: list[str] = []
        for cat in frontier:
            if cat in seen_cats or len(titles) >= limit:
                continue
            seen_cats.add(cat)
            cont: Optional[str] = None
            while len(titles) < limit:
                params = {"action": "query", "list": "categorymembers",
                          "cmtitle": cat, "cmlimit": "500",
                          "cmtype": "page|subcat"}
                if cont:
                    params["cmcontinue"] = cont
                try:
                    data = _api(site, lang, params)
                except RuntimeError:
                    break
                for m in data.get("query", {}).get("categorymembers", []):
                    if m.get("ns") == 14:
                        if level + 1 <= depth:
                            next_frontier.append(m["title"])
                    elif m.get("ns") == 0:
                        add(m["title"])
                cont = data.get("continue", {}).get("cmcontinue")
                if not cont:
                    break
                time.sleep(0.05)
            time.sleep(0.05)
        frontier = next_frontier
        if not frontier or len(titles) >= limit:
            break
    return titles


def fetch_wikitext(site: str, lang: str, titles: Sequence[str],
                   batch: int = 50) -> Iterator[tuple[str, str]]:
    """Yield ``(title, raw_wikitext)`` for a list of titles."""
    for i in range(0, len(titles), batch):
        chunk = list(titles[i:i + batch])
        params = {"action": "query", "prop": "revisions", "rvslots": "main",
                  "rvprop": "content", "titles": "|".join(chunk),
                  "redirects": "1"}
        try:
            data = _api(site, lang, params)
        except RuntimeError:
            continue
        for page in data.get("query", {}).get("pages", []):
            if page.get("missing"):
                continue
            revs = page.get("revisions") or []
            if not revs:
                continue
            content = revs[0].get("slots", {}).get("main", {}).get("content", "")
            if content:
                yield page["title"], content
        time.sleep(0.08)


def cmd_fetch_wiki(args: argparse.Namespace) -> None:
    """Fetch and clean one MediaWiki domain shard."""
    site = args.site
    out = SHARDS / f"{site}_{args.lang}_{args.domain}.jsonl"
    source = {"wikipedia": "Wikipedia", "wikisource": "Wikisource"}[site]
    licence = "CC-BY-SA-4.0"
    url = f"https://{args.lang}.{site}.org/"

    if args.force:
        reset_shard(out, MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists (use --force to rebuild)")
        return

    roots = [args.category] if args.category else CATEGORIES[args.lang][args.domain]
    # 20 KB per article is the observed average after cleaning; 1.7x headroom
    # covers the quality filter.  Capping the crawl matters: walking a whole
    # top-level category at depth 2 takes minutes and we only need a few
    # thousand articles for a pilot shard.
    limit = args.max_titles or int(args.target_mb * 1e6 / 20_000 * 1.7) + 400
    print(f"[{site}/{args.lang}/{args.domain}] collecting up to {limit} titles "
          f"from {len(roots)} roots ...")
    titles = collect_titles(site, args.lang, roots, depth=args.depth,
                            limit=limit)
    print(f"  {len(titles)} titles")

    target = int(args.target_mb * 1_000_000)
    written = 0
    kept = 0
    seen = 0
    start = time.time()
    # Write as we go: a killed run then leaves a partial file instead of
    # losing everything to an in-memory buffer, and progress is observable.
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as sink:
        for title, raw in fetch_wikitext(site, args.lang, titles):
            seen += 1
            text = clean_wikitext(raw)
            st = quality(text, args.lang, min_chars=args.min_chars)
            if not st.ok:
                continue
            doc = {"title": title, "text": text, "language": args.lang,
                   "domain": args.domain, "source": source, "licence": licence,
                   "url": url}
            sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1
            written += len(text.encode("utf-8"))
            if written >= target:
                break
            if kept % 200 == 0:
                sink.flush()
                rate = written / max(1e-9, time.time() - start) / 1e6
                print(f"  {kept} docs, {written/1e6:.1f} MB, {rate:.2f} MB/s")
    n = kept
    shard = record_shard(out, MANIFEST, source=source, licence=licence, url=url,
                         language=args.lang, domain=args.domain, docs=n,
                         note=f"categories={roots[:3]}... depth={args.depth}")
    print(f"[done] {n} docs ({seen} fetched) -> {out} "
          f"({shard.bytes/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Project Gutenberg
# ---------------------------------------------------------------------------

_RE_GUT_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                           re.S | re.I)
_RE_GUT_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                         re.S | re.I)


def _gutendex(lang: str, page: int) -> dict:
    url = f"https://gutendex.com/books?languages={lang}&page={page}"
    return json.loads(_get(url, timeout=120))


def cmd_fetch_gutenberg(args: argparse.Namespace) -> None:
    """Fetch public-domain books from Project Gutenberg via the gutendex API."""
    out = SHARDS / f"gutenberg_{args.lang}_{args.domain}.jsonl"
    if args.force:
        reset_shard(out, MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists")
        return
    target = int(args.target_mb * 1_000_000)
    written = 0
    docs: list[dict] = []
    page = 1
    while written < target and page <= args.max_pages:
        listing = _gutendex(args.lang, page)
        books = listing.get("results", [])
        if not books:
            break
        for book in books:
            if written >= target:
                break
            formats = book.get("formats", {})
            raw_url = None
            for mime, link in formats.items():
                if mime.startswith("text/plain") and link.endswith(".txt"):
                    raw_url = link
                    break
            if not raw_url:
                continue
            try:
                raw = _get(raw_url, timeout=90).decode("utf-8", errors="replace")
            except RuntimeError:
                continue
            body = _RE_GUT_START.split(raw, maxsplit=1)[-1]
            body = _RE_GUT_END.split(body, maxsplit=1)[0]
            text = normalize(body)
            st = quality(text, args.lang, min_chars=2000)
            if not st.ok:
                continue
            docs.append({"title": book.get("title", ""), "text": text,
                         "language": args.lang, "domain": args.domain,
                         "source": "Project Gutenberg",
                         "licence": "public-domain",
                         "url": f"https://www.gutenberg.org/ebooks/{book['id']}"})
            written += len(text.encode("utf-8"))
        page += 1
        time.sleep(0.2)
        print(f"  page {page}: {len(docs)} books, {written/1e6:.1f} MB")
    n = append_jsonl(out, docs)
    shard = record_shard(out, MANIFEST, source="Project Gutenberg",
                         licence="public-domain",
                         url="https://www.gutenberg.org/", language=args.lang,
                         domain=args.domain, docs=n,
                         note=f"gutendex pages 1..{page}")
    print(f"[done] {n} books -> {out} ({shard.bytes/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Code: Python standard library (PSF-2.0)
# ---------------------------------------------------------------------------

def cmd_fetch_code(args: argparse.Namespace) -> None:
    """Collect Python standard-library sources (PSF-2.0) as a code shard."""
    import sysconfig

    out = SHARDS / "python_stdlib_code.jsonl"
    if args.force:
        reset_shard(out, MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists")
        return
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in stdlib.rglob("*.py")
                   if "site-packages" not in str(p)
                   and "test" not in p.name.lower()
                   and p.stat().st_size > 1500)
    target = int(args.target_mb * 1_000_000)
    written = 0
    docs: list[dict] = []
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = normalize(raw, keep_indent=True)
        if len(text) < 800:
            continue
        docs.append({"title": str(path.relative_to(stdlib)), "text": text,
                     "language": "en", "domain": "code",
                     "source": "CPython standard library",
                     "licence": "PSF-2.0",
                     "url": "https://github.com/python/cpython"})
        written += len(text.encode("utf-8"))
        if written >= target:
            break
    n = append_jsonl(out, docs)
    shard = record_shard(out, MANIFEST, source="CPython standard library",
                         licence="PSF-2.0",
                         url="https://github.com/python/cpython",
                         language="en", domain="code", docs=n,
                         note="indentation preserved")
    print(f"[done] {n} files -> {out} ({shard.bytes/1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Build: dedup, mix, split, report
# ---------------------------------------------------------------------------

def _iter_shard(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _doc_bytes(doc: dict) -> int:
    """UTF-8 size of one document."""
    return len(doc["text"].encode("utf-8"))


def cmd_build(args: argparse.Namespace) -> None:
    """Deduplicate, quota-mix by language and domain, and write the corpus.

    Fetching is deliberately generous and uneven; this step enforces the plan:
    a per-language budget, the 60/40 language split and the domain quotas.
    Selection order is SHA-256, so the same manifest always yields the same
    corpus.
    """
    shards = read_manifest(MANIFEST)
    if not shards:
        print("manifest is empty; run the fetchers first")
        return
    out_dir = ROOT / "mix"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jsonl.gz"):
        stale.unlink()

    all_docs: list[dict] = []
    before = 0
    for shard in shards:
        path = Path(shard.path)
        if not path.exists():
            print(f"[warn] missing shard {path}")
            continue
        docs = list(_iter_shard(path))
        before += len(docs)
        all_docs.extend(docs)
        print(f"  loaded {len(docs):6d} docs from {path.name}")

    kept, dropped = dedup(all_docs, threshold=args.dedup_threshold)
    kept_bytes = sum(len(d["text"].encode("utf-8")) for d in kept)
    print(f"[dedup] {len(kept)} kept ({kept_bytes/1e6:.1f} MB), {dropped} dropped "
          f"(Jaccard >= {args.dedup_threshold})")

    budgets = {}
    if args.budget_en_mb:
        budgets["en"] = int(args.budget_en_mb * 1e6)
    if args.budget_ru_mb:
        budgets["ru"] = int(args.budget_ru_mb * 1e6)
    result = quota_mix(kept, int(args.budget_per_language_mb * 1e6),
                       heldout_bytes=int(args.heldout_mb * 1e6),
                       budgets=budgets or None)
    for lang in ("en", "ru"):
        write_heldout(result["heldout"][lang], ROOT,
                      int(args.heldout_mb * 1e6) // 2, lang)
    print(f"[heldout] {sum(len(v) for v in result['heldout'].values())} docs")

    report_mix: dict[str, dict[str, int]] = {}
    for lang, domains in result["train"].items():
        for domain, chosen in domains.items():
            pack_shard(chosen, out_dir / f"{lang}_{domain}.jsonl.gz")
            report_mix.setdefault(lang, {})[domain] = result["bytes"][lang][domain]
            print(f"  {lang}/{domain}: {len(chosen)} docs, "
                  f"{result['bytes'][lang][domain]/1e6:.1f} MB")

    rep = corpus_report(shards, ROOT / "corpus_report.json")
    mixed_bytes = sum(sum(v.values()) for v in report_mix.values())
    ru_bytes = sum(report_mix.get("ru", {}).values())
    rep["dedup"] = {"before": before, "kept": len(kept), "dropped": dropped,
                    "kept_bytes": kept_bytes}
    rep["mixed"] = report_mix
    rep["mixed_total_bytes"] = mixed_bytes
    rep["ru_share"] = round(ru_bytes / max(1, mixed_bytes), 4)
    rep["shortfalls"] = result["shortfalls"]
    (ROOT / "corpus_report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {ROOT / 'corpus_report.json'}")
    print(f"[mix] {mixed_bytes/1e6:.1f} MB total, RU share {rep['ru_share']:.1%}")
    if result["shortfalls"]:
        print("[shortfall] " + "; ".join(result["shortfalls"]))


# ---------------------------------------------------------------------------
# Tokenizer on the mixed corpus
# ---------------------------------------------------------------------------

def cmd_tokenizer(args: argparse.Namespace) -> None:
    """Train a BPE tokenizer on the mixed corpus and report fertility."""
    from tokenizers import ByteLevelBPETokenizer

    mix_dir = ROOT / "mix"
    files = sorted(mix_dir.glob("*.jsonl.gz"))
    if not files:
        print("no mixed shards; run `build` first")
        return
    import gzip

    samples: list[str] = []
    per_lang = {"en": [], "ru": []}
    for path in files:
        lang = path.name[:2]
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                per_lang[lang].append(doc["text"])
    for lang, docs in per_lang.items():
        n = 0
        total = 0
        for text in docs:
            samples.append(text)
            total += len(text)
            n += 1
            if total > args.sample_mb * 1_000_000 / 2:
                break
    sample_file = ROOT / "tokenizer_sample.txt"
    sample_file.write_text("\n".join(samples), encoding="utf-8")

    tok = ByteLevelBPETokenizer()
    tok.train([str(sample_file)], vocab_size=args.vocab,
              min_frequency=2, special_tokens=["<pad>", "<s>", "</s>"])
    out_dir = ROOT / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_model(str(out_dir))

    stats = {}
    for lang, docs in per_lang.items():
        text = "\n".join(docs[:2000])
        enc = tok.encode(text)
        b = len(text.encode("utf-8"))
        stats[lang] = {"bytes": b, "tokens": len(enc),
                       "bytes_per_token": round(b / max(1, len(enc)), 3)}
    (ROOT / "tokenizer_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("fetch-wiki", "fetch-wikisource"):
        p = sub.add_parser(name)
        p.add_argument("--lang", choices=["en", "ru"], required=True)
        p.add_argument("--domain", choices=list(DOMAIN_QUOTAS), required=True)
        p.add_argument("--category", default=None,
                       help="override the seed category list")
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--target-mb", type=float, default=25.0)
        p.add_argument("--min-chars", type=int, default=600)
        p.add_argument("--max-titles", type=int, default=0,
                       help="cap the category crawl (0 = derive from target)")
        p.add_argument("--force", action="store_true")
        p.set_defaults(func=cmd_fetch_wiki,
                       site="wikipedia" if name == "fetch-wiki" else "wikisource")

    p = sub.add_parser("fetch-gutenberg")
    p.add_argument("--lang", default="en", choices=["en", "ru"])
    p.add_argument("--domain", default="literature")
    p.add_argument("--target-mb", type=float, default=20.0)
    p.add_argument("--max-pages", type=int, default=12)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_gutenberg)

    p = sub.add_parser("fetch-code")
    p.add_argument("--target-mb", type=float, default=15.0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_code)

    p = sub.add_parser("build")
    p.add_argument("--budget-per-language-mb", type=float, default=200.0)
    p.add_argument("--budget-en-mb", type=float, default=0.0,
                   help="override the English budget (0 = use the shared one)")
    p.add_argument("--budget-ru-mb", type=float, default=0.0,
                   help="override the Russian budget (0 = use the shared one)")
    p.add_argument("--dedup-threshold", type=float, default=0.8)
    p.add_argument("--heldout-mb", type=float, default=2.0)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("tokenizer")
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--sample-mb", type=float, default=40.0)
    p.set_defaults(func=cmd_tokenizer)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
