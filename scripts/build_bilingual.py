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
                    "https://github.com/Icold21/kron-llm)"}
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
        "ml": ["Машинное обучение", "Искусственный интеллект",
               "Нейронные сети", "Обработка естественного языка"],
        "cs": ["Информатика", "Программная инженерия", "Техника",
               "Программирование", "Алгоритмы"],
    },
    "en": {
        "math": ["Mathematics", "Algebra", "Mathematical analysis",
                 "Geometry", "Probability theory"],
        "science": ["Physics", "Chemistry", "Biology", "Astronomy", "Science"],
        "logic": ["Logic", "Set theory", "Mathematical logic"],
        "literature": ["Literature", "Poetry", "Novels"],
        "encyclopedic": ["Geography", "History", "Culture", "Technology"],
        "ml": ["Machine learning", "Artificial intelligence",
               "Artificial neural networks", "Deep learning"],
        "cs": ["Computer science", "Software engineering", "Engineering",
               "Algorithms", "Programming languages"],
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
    suffix = getattr(args, "suffix", "") or ""
    out = SHARDS / f"{site}_{args.lang}_{args.domain}{suffix}.jsonl"
    source = {"wikipedia": "Wikipedia", "wikisource": "Wikisource"}[site]
    licence = "CC-BY-SA-4.0"
    url = f"https://{args.lang}.{site}.org/"

    if args.force:
        reset_shard(out, MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists (use --force to rebuild)")
        return

    if args.category:
        roots = [c.strip() for c in args.category.split(",") if c.strip()]
    else:
        roots = CATEGORIES[args.lang][args.domain]
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


def _gutendex(lang: str, page: int, topic: str | None = None) -> dict:
    url = f"https://gutendex.com/books?languages={lang}&page={page}"
    if topic:
        url += "&topic=" + urllib.parse.quote(topic)
    return json.loads(_get(url, timeout=120))


def cmd_fetch_gutenberg(args: argparse.Namespace) -> None:
    """Fetch public-domain books from Project Gutenberg via the gutendex API."""
    topic = getattr(args, "topic", None)
    slug = f"_{re.sub(r'[^a-z0-9]+', '_', topic.lower()).strip('_')}" if topic else ""
    out = SHARDS / f"gutenberg_{args.lang}_{args.domain}{slug}.jsonl"
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
        listing = _gutendex(args.lang, page, topic)
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
# Machine learning: textbooks, course notes and framework documentation
# ---------------------------------------------------------------------------

#: Repositories whose licence is unambiguous at the repository level.  The
#: licence string is recorded per shard, so an audit can check it against the
#: repository's LICENSE file.
ML_REPOS: list[tuple[str, str, str, str]] = [
    # owner/repo, branch, licence, what
    ("d2l-ai/d2l-en", "master", "CC-BY-SA-4.0", "Dive into Deep Learning"),
    ("pytorch/tutorials", "main", "BSD-3-Clause", "PyTorch tutorials"),
    ("scikit-learn/scikit-learn", "main", "BSD-3-Clause", "scikit-learn docs"),
    ("huggingface/course", "main", "Apache-2.0", "HF LLM course"),
    ("tensorflow/docs", "master", "Apache-2.0", "TensorFlow docs"),
    ("microsoft/ML-For-Beginners", "main", "MIT", "ML for beginners"),
    ("rasbt/LLMs-from-scratch", "main", "Apache-2.0", "LLMs from scratch"),
    ("karpathy/nanoGPT", "master", "MIT", "nanoGPT"),
    ("karpathy/micrograd", "master", "MIT", "micrograd"),
    ("jakevdp/PythonDataScienceHandbook", "master", "MIT", "DS handbook"),
]

ML_TEXT_EXT = {".md", ".rst", ".txt", ".tex", ".ipynb"}


def _notebook_text(raw: str) -> str:
    """Flatten a Jupyter notebook into markdown + fenced code."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            parts.append(src)
        elif cell.get("cell_type") == "code" and len(src) > 40:
            parts.append("```python\n" + src + "\n```")
    return "\n\n".join(parts)


def _download_tarball(repo: str, branch: str) -> bytes:
    """Download a repository tarball (codeload, then a mirror).

    Direct codeload connections reset on this machine often enough that a
    fallback matters for large repositories like mathlib4.
    """
    urls = [
        f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}",
        f"https://ghproxy.com/https://github.com/{repo}/archive/refs/"
        f"heads/{branch}.tar.gz",
    ]
    last: Optional[Exception] = None
    for url in urls:
        try:
            return _get(url, timeout=600, retries=1)
        except RuntimeError as exc:
            print(f"[warn] {url}: {exc}")
            last = exc
    raise RuntimeError(f"tarball download failed for {repo}") from last


def cmd_fetch_github(args: argparse.Namespace) -> None:
    """Collect machine-learning prose from permissively licensed repositories.

    Only documentation formats are read (markdown, rst, text, LaTeX and
    notebooks): the point is to teach the model ML *content*, not to duplicate
    the code slice.
    """
    import io
    import tarfile

    target = int(args.target_mb * 1_000_000)
    cap = int(args.repo_cap_mb * 1_000_000)
    total = 0
    for repo, branch, licence, what in ML_REPOS:
        if total >= target:
            break
        slug = repo.replace("/", "__")
        out = SHARDS / f"github_{slug}.jsonl"
        if args.force:
            reset_shard(out, MANIFEST)
        elif out.exists():
            print(f"[skip] {out.name} exists")
            total += out.stat().st_size
            continue
        try:
            blob = _download_tarball(repo, branch)
        except RuntimeError as exc:
            print(f"[warn] {repo}: {exc}")
            continue
        written = 0
        kept = 0
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar, \
                open(out, "a", encoding="utf-8") as sink:
            for member in tar:
                if not member.isfile() or written >= cap:
                    continue
                suffix = Path(member.name).suffix.lower()
                if suffix not in ML_TEXT_EXT:
                    continue
                if member.size > 2_000_000:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                raw = handle.read().decode("utf-8", errors="replace")
                text = (_notebook_text(raw) if suffix == ".ipynb"
                        else normalize(raw))
                if suffix == ".ipynb":
                    text = normalize(text)
                if len(text) < args.min_chars:
                    continue
                st = quality(text, "en", min_chars=args.min_chars)
                if not st.ok:
                    continue
                doc = {"title": member.name, "text": text, "language": "en",
                       "domain": "ml", "source": f"GitHub {repo}",
                       "licence": licence,
                       "url": f"https://github.com/{repo}"}
                sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += len(text.encode("utf-8"))
                kept += 1
            sink.flush()
        if not kept:
            print(f"[warn] {repo}: nothing passed the filters")
            if out.exists():
                out.unlink()
            continue
        shard = record_shard(out, MANIFEST, source=f"GitHub {repo}",
                             licence=licence,
                             url=f"https://github.com/{repo}",
                             language="en", domain="ml", docs=kept,
                             note=what)
        total += shard.bytes
        print(f"[done] {repo}: {kept} docs, {shard.bytes/1e6:.1f} MB")
    print(f"[ml] {total/1e6:.1f} MB total")


# ---------------------------------------------------------------------------
# Hugging Face datasets (the 2B-token corpus scale)
# ---------------------------------------------------------------------------
#: Lower-case substrings used to route wiki/web documents into domains.  The
#: label is a heuristic, not a curated taxonomy: a document needs ``>= 2``
#: distinct hits to be routed away from its source default, otherwise it goes to
#: the default domain (Wikipedia) or is dropped (FineWeb).  Kept short on
#: purpose: long lists start matching everything.
KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "en": {
        "math": ("theorem", "lemma", "equation", "integral", "derivative",
                 "algebra", "geometry", "probability", "matrix", "polynomial",
                 "topology", "combinator", "corollary", "differential equation",
                 "prime number"),
        "science": ("physics", "chemistry", "biology", "molecule", "quantum",
                    "cell", "gene", "evolution", "experiment", "particle",
                    "protein", "thermodynamic", "electromagnetic", "organism",
                    "astronomy"),
        "logic": ("logic", "syllogism", "inference", "propositional",
                  "predicate", "axiom", "deduction", "fallacy", "formal proof",
                  "consistency"),
        "ml": ("neural network", "machine learning", "gradient descent",
               "transformer", "training data", "language model", "embedding",
               "attention head", "supervised learning", "dataset"),
        "literature": ("novel", "poem", "chapter", "protagonist", "narrative",
                       "verse", "fiction", "story", "plot", "author"),
    },
    "ru": {
        "math": ("теорем", "уравнени", "лемм", "интеграл", "производн",
                 "алгебр", "геометри", "вероятност", "матриц", "многочлен",
                 "топологи", "комбинатор", "доказательств", "следстви",
                 "дифференциальн"),
        "science": ("физик", "хими", "биолог", "молекул", "квант", "клетк",
                    "ген", "эволюци", "эксперимент", "частиц", "белок",
                    "термодинамик", "электромагнит", "организм", "астроном"),
        "logic": ("логик", "силлогизм", "высказывани", "предикат", "аксиом",
                  "дедукци", "умозаключени", "противоречи", "формальн"),
        "ml": ("нейросет", "машинное обучение", "машинного обучения",
               "градиент", "трансформер", "датасет", "функция потерь",
               "обратное распространение", "переобучени"),
        "literature": ("роман", "повест", "рассказ", "стихотвор", "глава",
                       "герой", "сюжет", "проза", "поэм"),
    },
}

#: Targets are text bytes, not file bytes.  The sum (~7.2 GB) is the 2B-token
#: budget at the measured 3.34 bytes/token with ~10 % headroom for dedup and
#: the held-out split.
HF_ROUTES: dict[str, dict] = {
    "wiki_en": {
        "dataset": "wikimedia/wikipedia", "config": "20231101.en",
        "text": "text", "title": "title", "url": "url", "lang": "en",
        "licence": "CC-BY-SA-3.0", "source": "Wikipedia 2023-11-01 (HF)",
        "default_domain": "encyclopedic",
        "targets_mb": {"encyclopedic": 550, "math": 250, "science": 250,
                       "logic": 120, "ml": 80},
    },
    "wiki_ru": {
        "dataset": "wikimedia/wikipedia", "config": "20231101.ru",
        "text": "text", "title": "title", "url": "url", "lang": "ru",
        "licence": "CC-BY-SA-3.0", "source": "Википедия 2023-11-01 (HF)",
        "default_domain": "encyclopedic",
        "targets_mb": {"encyclopedic": 550, "math": 150, "science": 150,
                       "logic": 80, "ml": 70},
    },
    "owm": {
        "dataset": "open-web-math/open-web-math", "config": "default",
        "text": "text", "url": "url", "lang": "en",
        "licence": "ODC-By-1.0", "source": "OpenWebMath (HF)",
        "default_domain": "math", "targets_mb": {"math": 1100},
    },
    "arxiv": {
        "dataset": "ccdv/arxiv-summarization", "config": "document",
        "fields": {"article": "science", "abstract": "ml"},
        "lang": "en", "licence": "arXiv non-exclusive (see dataset card)",
        "source": "arXiv full text + abstracts (HF ccdv/arxiv-summarization)",
        "targets_mb": {"science": 700, "ml": 300},
    },
    "gutenberg": {
        "dataset": "sedthh/gutenberg_english", "config": "default",
        "text": "TEXT", "lang": "en", "licence": "public domain (US)",
        "source": "Project Gutenberg (HF sedthh/gutenberg_english)",
        "default_domain": "literature", "targets_mb": {"literature": 550},
    },
    "fineweb_ru": {
        "dataset": "HuggingFaceFW/fineweb-2", "config": "rus_Cyrl",
        "text": "text", "url": "url", "lang": "ru",
        "licence": "ODC-By-1.0", "source": "FineWeb-2 rus_Cyrl (HF)",
        "default_domain": None,
        "targets_mb": {"math": 450, "science": 400, "logic": 200,
                       "ml": 300, "literature": 250},
    },
    "cosmopedia_ml": {
        "dataset": "HuggingFaceTB/cosmopedia", "config": "web_samples_v2",
        "text": "text", "url": "seed_data", "lang": "en",
        "licence": "Apache-2.0", "source": "Cosmopedia v2 (synthetic textbooks)",
        "default_domain": None, "targets_mb": {"ml": 300},
    },
    "ru_tech_ml": {
        "dataset": "HuggingFaceFW/fineweb-2", "config": "rus_Cyrl",
        "text": "text", "url": "url", "lang": "ru",
        "licence": "ODC-By-1.0", "source": "FineWeb-2 rus_Cyrl, tech hosts (HF)",
        "default_domain": None,
        "url_any": ("habr.com", "tproger.ru", "xakep.ru", "bigdataschool.ru",
                    "neurohive.io", "mlmaster.ru", "datareview.info",
                    "securitylab.ru", "compress.ru", "rsdn.org"),
        "targets_mb": {"ml": 300},
    },
    "code_py": {
        "dataset": "codeparrot/codeparrot-clean", "config": None,
        "text": "content", "title": "path", "url": "repo_name", "lang": "en",
        "licence": "permissive (per-file, see dataset card)",
        "source": "CodeParrot Clean (HF)",
        "fields": {"content": "code"}, "kind": "code",
        "targets_mb": {"code": 1000},
    },
}


def _route_domain(text: str, lang: str, default: Optional[str]) -> Optional[str]:
    """Return the keyword-routed domain for one document."""
    low = text.lower()
    best, best_hits = default, 0
    for domain, stems in KEYWORDS[lang].items():
        hits = sum(1 for stem in stems if stem in low)
        if hits > best_hits:
            best, best_hits = domain, hits
    # Three distinct stems: two are too easy to hit by accident ("matrix" and
    # "equation" in an article about a film, say), and the router's precision
    # matters more than its recall -- the default domain catches the rest.
    if best_hits < 3:
        return default
    return best


def cmd_fetch_hf(args: argparse.Namespace) -> None:
    """Stream Hugging Face datasets into per-domain shards.

    One pass per route: each document is routed by keywords (see
    :data:`KEYWORDS`), written to the matching shard, and stopped once the
    per-domain byte target is met.  A ``.done`` marker prevents a re-run from
    appending to a finished shard; ``--force`` starts it over.
    """
    from datasets import load_dataset

    routes = [args.source] if args.source != "all" else list(HF_ROUTES)
    if args.skip:
        routes = [r for r in routes if r not in args.skip.split(",")]
    for route in routes:
        spec = HF_ROUTES[route]
        targets = {d: int(mb * 1e6) for d, mb in spec["targets_mb"].items()}
        fields = spec.get("fields") or {spec["text"]: spec["default_domain"]}
        write = {d: 0 for d in targets}
        sinks: dict[str, "object"] = {}
        paths: dict[str, Path] = {}
        for d in targets:
            path = SHARDS / f"hf_{route}_{d}.jsonl"
            done = path.with_suffix(".done")
            paths[d] = path
            if done.exists() and not args.force:
                print(f"[skip] {route}/{d}: already done")
                write[d] = targets[d]
                continue
            if args.force:
                reset_shard(path, MANIFEST)
            path.parent.mkdir(parents=True, exist_ok=True)
            sinks[d] = open(path, "a", encoding="utf-8")
        if all(write[d] >= targets[d] for d in targets):
            print(f"[skip] {route}: all targets met")
            continue

        print(f"[hf] {route}: {spec['dataset']} ({spec['config']}) -> "
              f"{sum(targets.values())/1e6:.0f} MB")
        kw = {"split": "train", "streaming": True}
        if spec.get("config"):
            kw["name"] = spec["config"]
        ds = load_dataset(spec["dataset"], **kw)
        seen = kept = 0
        counts = {d: 0 for d in targets}
        last = time.time()
        try:
            for ex in ds:
                seen += 1
                if all(write[d] >= targets[d] for d in targets):
                    break
                if spec.get("url_any"):
                    url = str(ex.get(spec.get("url", "")) or "")
                    if not any(h in url for h in spec["url_any"]):
                        continue
                if spec.get("fields"):
                    pairs = list(fields.items())
                else:
                    dom = _route_domain(str(ex.get(spec["text"]) or ""),
                                        spec["lang"], spec["default_domain"])
                    if dom is None or dom not in targets:
                        continue
                    pairs = [(spec["text"], dom)]
                for field, dom in pairs:
                    if write[dom] >= targets[dom]:
                        continue
                    raw = ex.get(field)
                    if not isinstance(raw, str) or len(raw) < args.min_chars:
                        continue
                    is_code = spec.get("kind") == "code"
                    text = (normalize(raw, keep_indent=True) if is_code
                            else normalize(raw))
                    if not text or len(text) < args.min_chars:
                        continue
                    if is_code:
                        # Code fails the prose filters (symbol ratio) by design;
                        # keep the structural checks instead.
                        if text.count("\n") < 3:
                            continue
                    elif not quality(text, spec["lang"],
                                     min_chars=args.min_chars).ok:
                        continue
                    doc = {"title": str(ex.get(spec.get("title", "")) or "")[:200],
                           "text": text, "language": spec["lang"],
                           "domain": dom, "source": spec["source"],
                           "licence": spec["licence"],
                           "url": str(ex.get(spec.get("url", "")) or "")}
                    sinks[dom].write(json.dumps(doc, ensure_ascii=False) + "\n")
                    write[dom] += len(text.encode("utf-8"))
                    counts[dom] += 1
                    kept += 1
                if time.time() - last > 30:
                    got = sum(write.values()) / 1e6
                    print(f"  {route}: {got:6.1f} MB kept, {seen/1e3:8.1f}K seen")
                    last = time.time()
        finally:
            for s in sinks.values():
                s.close()
        for d, path in paths.items():
            if write[d] < targets[d] * 0.02:
                if path.exists():
                    path.unlink()
                print(f"[warn] {route}/{d}: only {write[d]/1e6:.1f} MB")
                continue
            shard = record_shard(path, MANIFEST, source=spec["source"],
                                 licence=spec["licence"], url="https://huggingface.co/datasets/"
                                 + spec["dataset"], language=spec["lang"],
                                 domain=d, docs=counts[d],
                                 note=f"HF route {route}; keyword-routed, "
                                      f"{'default domain' if spec.get('default_domain') else 'filtered'}")
            path.with_suffix(".done").touch()
            print(f"[done] {route}/{d}: {counts[d]} docs, {shard.bytes/1e6:.1f} MB")


def cmd_fetch_tarball(args: argparse.Namespace) -> None:
    """Collect source files of one repository as a corpus shard.

    Used for formal mathematics: Lean's mathlib4 is a large body of checked
    proofs, which is exactly the ``logic`` domain rather than code.  Only files
    with the listed extensions are read, each capped at 200 KB.
    """
    import io
    import tarfile

    target = int(args.target_mb * 1_000_000)
    exts = {e.strip() for e in args.ext.split(",") if e.strip()}
    slug = args.repo.replace("/", "__")
    out = SHARDS / f"tarball_{slug}.jsonl"
    if out.exists() and not args.force:
        print(f"[skip] {out.name} exists")
        return
    if args.force:
        reset_shard(out, MANIFEST)
    blob = _download_tarball(args.repo, args.branch)
    written = kept = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar, \
            open(out, "a", encoding="utf-8") as sink:
        for member in tar:
            if not member.isfile() or written >= target:
                continue
            if Path(member.name).suffix.lower() not in exts:
                continue
            if member.size > 200_000:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            text = normalize(handle.read().decode("utf-8", errors="replace"),
                             keep_indent=True)
            if len(text) < args.min_chars:
                continue
            if text.count("\n") < 5:
                continue
            doc = {"title": member.name, "text": text,
                   "language": args.lang, "domain": args.domain,
                   "source": f"tarball {args.repo}", "licence": args.licence,
                   "url": f"https://github.com/{args.repo}"}
            sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += len(text.encode("utf-8"))
            kept += 1
        sink.flush()
    if not kept:
        print(f"[warn] {args.repo}: nothing passed the filters")
        if out.exists():
            out.unlink()
        return
    shard = record_shard(out, MANIFEST, source=f"tarball {args.repo}",
                         licence=args.licence,
                         url=f"https://github.com/{args.repo}",
                         language=args.lang, domain=args.domain, docs=kept,
                         note=f"{args.note}")
    print(f"[done] {args.repo}: {kept} docs, {shard.bytes/1e6:.1f} MB")


def cmd_fetch_gutenberg_search(args: argparse.Namespace) -> None:
    """Fetch Gutenberg books by site search (gutendex is unreachable here).

    The gutendex API times out from this network, but gutenberg.org itself
    responds.  The search page lists book ids; the text is fetched from the
    standard cache path, which is what the gutendex fetcher uses too.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", args.query.lower()).strip("_")
    out = SHARDS / f"gutenberg_{args.lang}_{args.domain}_{slug}.jsonl"
    if args.force:
        reset_shard(out, MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists")
        return
    target = int(args.target_mb * 1_000_000)
    ids: list[int] = []
    for page in range(1, args.max_pages + 1):
        url = ("https://www.gutenberg.org/ebooks/search/?query="
               + urllib.parse.quote(args.query) + f"&start_index={(page-1)*25}")
        try:
            html = _get(url, timeout=60).decode("utf-8", errors="replace")
        except RuntimeError as exc:
            print(f"[warn] search page {page}: {exc}")
            break
        found = [int(m) for m in re.findall(r"/ebooks/(\d+)", html)]
        fresh = [i for i in dict.fromkeys(found) if i not in ids]
        ids.extend(fresh)
        if not fresh:
            break
        time.sleep(0.3)
    print(f"[search] {len(ids)} candidate books for {args.query!r}")
    written = kept = 0
    with open(out, "a", encoding="utf-8") as sink:
        for book_id in ids:
            if written >= target:
                break
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            try:
                raw = _get(url, timeout=90).decode("utf-8", errors="replace")
            except RuntimeError:
                continue
            body = _RE_GUT_START.split(raw, maxsplit=1)[-1]
            body = _RE_GUT_END.split(body, maxsplit=1)[0]
            text = normalize(body)
            if not quality(text, args.lang, min_chars=args.min_chars).ok:
                continue
            doc = {"title": f"gutenberg/{book_id}", "text": text,
                   "language": args.lang, "domain": args.domain,
                   "source": "Project Gutenberg (site search)",
                   "licence": "public-domain",
                   "url": f"https://www.gutenberg.org/ebooks/{book_id}"}
            sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += len(text.encode("utf-8"))
            kept += 1
        sink.flush()
    if not kept:
        print("[warn] nothing passed")
        if out.exists():
            out.unlink()
        return
    shard = record_shard(out, MANIFEST, source="Project Gutenberg (site search)",
                         licence="public-domain",
                         url="https://www.gutenberg.org/", language=args.lang,
                         domain=args.domain, docs=kept,
                         note=f"search query {args.query!r}")
    print(f"[done] {kept} books, {shard.bytes/1e6:.1f} MB -> {out.name}")


# ---------------------------------------------------------------------------
# Gutenberg catalog (mirror) and curated Wikipedia titles
# ---------------------------------------------------------------------------
GUT_MIRROR = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"
GUT_CATALOG = "https://gutenberg.pglaf.org/cache/epub/feeds/pg_catalog.csv"

#: Subject/bookshelf keywords per domain.  The catalog is the only reliable
#: index left: gutendex and www.gutenberg.org time out from this network while
#: the pglaf mirror answers.
GUT_FILTERS: dict[str, tuple[str, ...]] = {
    "humour": ("humor", "humour", "satire", "comic", "wit and humor",
               "nonsense", "parody"),
    "science": ("astronomy", "astrophysics", "cosmology", "physics",
                "relativity"),
}
#: Popular/quality authors first for the science slice.
GUT_FAMOUS = ("einstein", "flammarion", "ball, r", "jeans", "eddington",
              "lodge", "newton", "tyndall", "faraday", "herschel",
              "proctor", "arrhenius", "hoyle")


def _gut_catalog() -> list[dict]:
    """Download (once) and parse the Gutenberg catalog CSV."""
    import csv

    cache = ROOT / "pg_catalog.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(_get(GUT_CATALOG, timeout=300))
    with cache.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_fetch_gutenberg_catalog(args: argparse.Namespace) -> None:
    """Fetch one themed Gutenberg slice through the pglaf mirror.

    Selection is by subject/bookshelf keywords from the official catalog;
    for ``science`` the famous-author list is sorted first so the slice is
    popular classics rather than textbooks from 1890.
    """
    out = SHARDS / f"gutenberg_{args.lang}_{args.domain}_cat.jsonl"
    if args.force:
        reset_shard(out, MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists")
        return

    keys = GUT_FILTERS[args.domain]
    rows: list[tuple[int, int, str]] = []
    for r in _gut_catalog():
        if (r.get("Language") or "").strip() != args.lang:
            continue
        blob = ((r.get("Subjects") or "") + " "
                + (r.get("Bookshelves") or "")).lower()
        if not any(k in blob for k in keys):
            continue
        author = (r.get("Authors") or "").lower()
        score = 1 if any(f in author for f in GUT_FAMOUS) else 0
        try:
            book_id = int(r["Text#"])
        except (KeyError, ValueError):
            continue
        rows.append((score, book_id, (r.get("Title") or "").strip()))
    rows.sort(key=lambda t: (-t[0], t[1]))
    print(f"[catalog] {len(rows)} candidates for {args.domain}")

    target = int(args.target_mb * 1_000_000)
    written = kept = 0
    with open(out, "a", encoding="utf-8") as sink:
        for _score, book_id, title in rows:
            if written >= target or kept >= args.max_books:
                break
            try:
                raw = _get(GUT_MIRROR.format(id=book_id),
                           timeout=120).decode("utf-8", errors="replace")
            except RuntimeError:
                continue
            body = _RE_GUT_START.split(raw, maxsplit=1)[-1]
            body = _RE_GUT_END.split(body, maxsplit=1)[0]
            text = normalize(body)
            if not quality(text, args.lang, min_chars=args.min_chars).ok:
                continue
            doc = {"title": title, "text": text, "language": args.lang,
                   "domain": args.domain,
                   "source": "Project Gutenberg (pglaf mirror)",
                   "licence": "public-domain",
                   "url": f"https://www.gutenberg.org/ebooks/{book_id}"}
            sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += len(text.encode("utf-8"))
            kept += 1
            if kept % 10 == 0:
                sink.flush()
                print(f"  {kept} books, {written/1e6:.1f} MB")
        sink.flush()
    if not kept:
        print("[warn] nothing passed")
        if out.exists():
            out.unlink()
        return
    shard = record_shard(out, MANIFEST,
                         source="Project Gutenberg (pglaf mirror)",
                         licence="public-domain",
                         url="https://gutenberg.pglaf.org/",
                         language=args.lang, domain=args.domain, docs=kept,
                         note=f"catalog keywords {keys[:3]}")
    print(f"[done] {kept} books, {shard.bytes/1e6:.1f} MB -> {out.name}")


def cmd_fetch_wiki_titles(args: argparse.Namespace) -> None:
    """Fetch an explicit list of Wikipedia articles (curated, no crawl)."""
    out = SHARDS / f"wikipedia_{args.lang}_{args.domain}_{args.suffix}.jsonl"
    if args.force:
        reset_shard(out, MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists")
        return
    titles = [t.strip() for t in args.titles.split("|") if t.strip()]
    print(f"[titles] {len(titles)} articles")
    kept = written = 0
    with open(out, "a", encoding="utf-8") as sink:
        # TextExtracts returns at most one full extract per request, so the
        # titles are fetched one by one.
        for title in titles:
            params = {"action": "query", "format": "json",
                      "formatversion": 2, "prop": "extracts",
                      "explaintext": 1, "redirects": 1,
                      "titles": title}
            url = (f"https://{args.lang}.wikipedia.org/w/api.php?"
                   + urllib.parse.urlencode(params))
            try:
                data = json.loads(_get(url, timeout=120))
            except RuntimeError as exc:
                print(f"[warn] {title}: {exc}")
                continue
            for page in data.get("query", {}).get("pages", []):
                text = normalize(page.get("extract") or "")
                if len(text) < args.min_chars:
                    continue
                doc = {"title": page.get("title", ""), "text": text,
                       "language": args.lang, "domain": args.domain,
                       "source": "Wikipedia (curated titles)",
                       "licence": "CC-BY-SA-4.0",
                       "url": f"https://{args.lang}.wikipedia.org/"}
                sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += len(text.encode("utf-8"))
                kept += 1
        sink.flush()
    if not kept:
        print("[warn] nothing fetched")
        if out.exists():
            out.unlink()
        return
    shard = record_shard(out, MANIFEST, source="Wikipedia (curated titles)",
                         licence="CC-BY-SA-4.0",
                         url=f"https://{args.lang}.wikipedia.org/",
                         language=args.lang, domain=args.domain, docs=kept,
                         note=f"curated list: {len(titles)} titles")
    print(f"[done] {kept} articles, {shard.bytes/1e6:.1f} MB -> {out.name}")


def cmd_fetch_gutenberg_ids(args: argparse.Namespace) -> None:
    """Fetch an explicit list of Gutenberg ids through the pglaf mirror.

    The catalog CSV is unusable from this network (6 KB/s), but individual
    texts come down fine, so a curated id list is the reliable path for a
    small themed slice.
    """
    out = SHARDS / f"gutenberg_{args.lang}_{args.domain}_ids.jsonl"
    if args.force:
        reset_shard(out, MANIFEST)
    elif out.exists():
        print(f"[skip] {out} exists")
        return
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    written = kept = 0
    with open(out, "a", encoding="utf-8") as sink:
        for book_id in ids:
            try:
                raw = _get(GUT_MIRROR.format(id=book_id),
                           timeout=180).decode("utf-8", errors="replace")
            except RuntimeError as exc:
                print(f"[warn] {book_id}: {exc}")
                continue
            match = re.search(r"Title:\s*(.+)", raw[:2000])
            title = match.group(1).strip() if match else f"gutenberg/{book_id}"
            body = _RE_GUT_START.split(raw, maxsplit=1)[-1]
            body = _RE_GUT_END.split(body, maxsplit=1)[0]
            text = normalize(body)
            if not quality(text, args.lang, min_chars=args.min_chars).ok:
                print(f"[skip] {book_id}: quality")
                continue
            doc = {"title": title, "text": text, "language": args.lang,
                   "domain": args.domain,
                   "source": "Project Gutenberg (pglaf mirror)",
                   "licence": "public-domain",
                   "url": f"https://www.gutenberg.org/ebooks/{book_id}"}
            sink.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += len(text.encode("utf-8"))
            kept += 1
            print(f"  [{kept}] {book_id} {title[:50]} {written/1e6:.1f} MB")
        sink.flush()
    if not kept:
        print("[warn] nothing fetched")
        if out.exists():
            out.unlink()
        return
    shard = record_shard(out, MANIFEST,
                         source="Project Gutenberg (pglaf mirror)",
                         licence="public-domain",
                         url="https://gutenberg.pglaf.org/",
                         language=args.lang, domain=args.domain, docs=kept,
                         note=f"curated ids {args.ids}")
    print(f"[done] {kept} books, {shard.bytes/1e6:.1f} MB -> {out.name}")


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
                       help="comma-separated override of the seed categories")
        p.add_argument("--suffix", default="",
                       help="filename suffix, for a second shard in one domain")
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
    p.add_argument("--topic", default=None,
                   help="gutendex topic filter, e.g. astronomy, humor, satire")
    p.add_argument("--target-mb", type=float, default=20.0)
    p.add_argument("--max-pages", type=int, default=12)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_gutenberg)

    p = sub.add_parser("fetch-code")
    p.add_argument("--target-mb", type=float, default=15.0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_code)

    p = sub.add_parser("fetch-github")
    p.add_argument("--target-mb", type=float, default=150.0)
    p.add_argument("--repo-cap-mb", type=float, default=40.0)
    p.add_argument("--min-chars", type=int, default=600)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_github)

    p = sub.add_parser("fetch-tarball")
    p.add_argument("--repo", required=True)
    p.add_argument("--branch", default="master")
    p.add_argument("--ext", default=".lean")
    p.add_argument("--domain", default="logic")
    p.add_argument("--lang", default="en")
    p.add_argument("--licence", default="Apache-2.0")
    p.add_argument("--target-mb", type=float, default=300.0)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--note", default="")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_tarball)

    p = sub.add_parser("fetch-hf")
    p.add_argument("--source", default="all",
                   help="route name or 'all' (see HF_ROUTES)")
    p.add_argument("--skip", default="", help="comma-separated routes to skip")
    p.add_argument("--min-chars", type=int, default=600)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_hf)

    p = sub.add_parser("fetch-gutenberg-search")
    p.add_argument("--query", required=True)
    p.add_argument("--lang", default="en", choices=["en", "ru"])
    p.add_argument("--domain", default="literature")
    p.add_argument("--target-mb", type=float, default=15.0)
    p.add_argument("--max-pages", type=int, default=12)
    p.add_argument("--min-chars", type=int, default=2000)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_gutenberg_search)

    p = sub.add_parser("fetch-gutenberg-catalog")
    p.add_argument("--domain", required=True,
                   choices=sorted(GUT_FILTERS))
    p.add_argument("--lang", default="en")
    p.add_argument("--target-mb", type=float, default=12.0)
    p.add_argument("--max-books", type=int, default=60)
    p.add_argument("--min-chars", type=int, default=4000)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_gutenberg_catalog)

    p = sub.add_parser("fetch-wiki-titles")
    p.add_argument("--lang", default="en", choices=["en", "ru"])
    p.add_argument("--domain", default="science")
    p.add_argument("--suffix", default="curated")
    p.add_argument("--titles", required=True,
                   help="pipe-separated article titles")
    p.add_argument("--min-chars", type=int, default=2000)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_wiki_titles)

    p = sub.add_parser("fetch-gutenberg-ids")
    p.add_argument("--ids", required=True, help="comma-separated book ids")
    p.add_argument("--domain", default="humour")
    p.add_argument("--lang", default="en")
    p.add_argument("--min-chars", type=int, default=4000)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_gutenberg_ids)

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
