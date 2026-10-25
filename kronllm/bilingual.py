"""Corpus construction for the bilingual English/Russian pretraining set.

This module implements the pipeline described in ``docs/data_plan.md``:

    fetch -> normalise -> quality filter -> deduplicate -> mix -> split

It deliberately has no third-party dependencies beyond the standard library so
that the corpus can be rebuilt on a bare machine.  The three things it is
careful about are:

* **licences** -- every shard is recorded with ``source``, ``licence``,
  ``url`` and ``sha256``; a shard without a known licence is rejected by
  :func:`record_shard`;
* **language** -- a fast heuristic (script shares) instead of a model, so the
  filter is explainable and reproducible;
* **boilerplate** -- MediaWiki pages are cleaned of templates, references,
  tables and image links before they are counted as text.

The module is used by ``scripts/build_bilingual.py``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

# ---------------------------------------------------------------------------
# Normalisation and MediaWiki cleaning
# ---------------------------------------------------------------------------

#: Sections that carry references or navigation rather than prose.
BOILERPLATE_SECTIONS = {
    # Russian
    "примечания", "литература", "ссылки", "источники", "см. также",
    "библиография", "комментарии", "сноски", "внешние ссылки",
    # English
    "references", "external links", "see also", "further reading",
    "notes", "bibliography", "citations", "sources", "footnotes",
}

_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
_RE_TAG = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_RE_MATH = re.compile(r"<math>(.*?)</math>", re.S | re.I)
_RE_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
_RE_FILE = re.compile(
    r"\[\[(?:Файл|File|Изображение|Image|Медиа|Media):[^\[\]]*?(?:\[\[[^\]]*\]\][^\[\]]*?)*\]\]",
    re.I)
_RE_CATEGORY = re.compile(
    r"\[\[(?:Категория|Category):[^\[\]]*\]\]", re.I)
_RE_INTERWIKI = re.compile(r"\[\[[a-z]{2,3}:[^\[\]]*\]\]")
_RE_LINK = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\]")
_RE_EXTLINK = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s*([^\]]*)\]")
_RE_BOLD = re.compile(r"'{2,5}")
_RE_LIST = re.compile(r"^[ \t]*[*#:;]+[ \t]*", re.M)
_RE_WS = re.compile(r"[ \t]+")
_RE_BLANKLINES = re.compile(r"\n{3,}")


def _strip_templates(text: str) -> str:
    """Remove ``{{...}}`` blocks, handling nesting.

    MediaWiki templates are balanced but not regular, so this walks the string
    once with a depth counter instead of using a regex.
    """
    out: list[str] = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i) and depth:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def _drop_sections(text: str) -> str:
    """Drop whole sections whose heading is boilerplate (references, links)."""
    lines = text.split("\n")
    kept: list[str] = []
    skip_level: Optional[int] = None
    for line in lines:
        m = _RE_HEADING.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip().lower().rstrip(":")
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and title in BOILERPLATE_SECTIONS:
                skip_level = level
                continue
        if skip_level is None:
            kept.append(line)
    return "\n".join(kept)


def clean_wikitext(text: str) -> str:
    """Convert MediaWiki wikitext to plain prose.

    The cleaner is intentionally conservative: it removes markup that never
    contains prose (templates, references, tables, files, categories) and
    unwraps markup that wraps prose (links, bold, headings).  It does not try
    to be a full parser; residual markup is filtered later by the symbol-ratio
    check.

    Args:
        text: raw wikitext of one article.

    Returns:
        Plain text with paragraphs separated by a single blank line.
    """
    text = _RE_COMMENT.sub(" ", text)
    text = _RE_REF.sub(" ", text)
    text = _RE_TABLE.sub(" ", text)
    text = _strip_templates(text)
    text = _RE_MATH.sub(lambda m: " " + m.group(1).strip() + " ", text)
    text = _drop_sections(text)
    text = _RE_FILE.sub(" ", text)
    text = _RE_CATEGORY.sub(" ", text)
    text = _RE_INTERWIKI.sub(" ", text)
    # headings -> sentences on their own line, then unwrapped
    text = _RE_HEADING.sub(lambda m: "\n\n" + m.group(2).strip() + "\n\n", text)
    text = _RE_LINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    text = _RE_EXTLINK.sub(lambda m: (m.group(1) or "").strip(), text)
    text = _RE_BOLD.sub("", text)
    text = _RE_LIST.sub("", text)
    text = _RE_TAG.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return normalize(text)


def normalize(text: str, keep_indent: bool = False) -> str:
    """Unicode-normalise and clean whitespace.

    Args:
        text: input text.
        keep_indent: keep leading indentation and blank lines (used for code,
            where indentation carries meaning).

    Returns:
        Normalised text, paragraphs separated by ``\\n\\n``.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")          # soft hyphen
    text = text.replace("\ufeff", "")
    for space in ("\u00a0", "\u2009", "\u202f", "\u2007"):
        text = text.replace(space, " ")
    if keep_indent:
        lines = [ln.rstrip() for ln in text.split("\n")]
        out: list[str] = []
        for ln in lines:
            out.append(ln)
        return _RE_BLANKLINES.sub("\n\n\n", "\n".join(out)).strip("\n")
    lines = []
    for ln in text.split("\n"):
        ln = _RE_WS.sub(" ", ln).strip()
        lines.append(ln)
    text = "\n".join(lines)
    text = _RE_BLANKLINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Language and quality filters
# ---------------------------------------------------------------------------

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_LATIN = re.compile(r"[A-Za-z]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_DIGIT = re.compile(r"\d")
_SYMBOL = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass
class QualityStats:
    """Measured properties of one document."""

    chars: int
    words: int
    letters: int
    digits: int
    symbols: int
    cyrillic: int
    latin: int
    repeat_ratio: float
    compression: float
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _repeat_ratio(words: Sequence[str], n: int = 5) -> float:
    """Fraction of ``n``-grams that occur more than once."""
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return 1.0 - unique / len(grams)


def quality(text: str, language: Optional[str] = None,
            min_chars: int = 400, max_symbol_ratio: float = 0.20,
            max_digit_ratio: float = 0.12,
            max_repeat_ratio: float = 0.30,
            min_compression: float = 0.22,
            max_compression: float = 0.80) -> QualityStats:
    """Score a document and decide whether it may enter the corpus.

    The thresholds are calibrated on the pilot, not on theory: they remove
    lists, tables, navigation residue and machine-generated text while keeping
    dense prose and formulas.

    Args:
        text: normalised document.
        language: ``"en"`` or ``"ru"``; ``None`` skips the script check.
        min_chars: minimum document length in characters.
        max_symbol_ratio: maximum share of non-alphanumeric characters.
        max_digit_ratio: maximum share of digits (filters data dumps).
        max_repeat_ratio: maximum share of repeated 5-grams.
        min_compression: minimum zlib ratio (too uniform text is suspicious).
        max_compression: maximum zlib ratio (too random text is suspicious).

    Returns:
        A :class:`QualityStats` with ``ok`` set to the verdict.
    """
    n = len(text)
    letters = len(_LETTER.findall(text))
    digits = len(_DIGIT.findall(text))
    symbols = len(_SYMBOL.findall(text))
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    words = text.split()
    comp = len(zlib.compress(text.encode("utf-8"), 6)) / max(1, len(text.encode("utf-8")))
    rep = _repeat_ratio(words)

    reasons: list[str] = []
    if n < min_chars:
        reasons.append("short")
    if n and symbols / n > max_symbol_ratio:
        reasons.append("symbols")
    if n and digits / n > max_digit_ratio:
        reasons.append("digits")
    if rep > max_repeat_ratio:
        reasons.append("repetition")
    if not (min_compression <= comp <= max_compression):
        reasons.append("compression")
    if language == "ru" and (cyr + lat) and cyr / (cyr + lat) < 0.60:
        reasons.append("not_ru")
    if language == "en" and (cyr + lat) and lat / (cyr + lat) < 0.80:
        reasons.append("not_en")

    return QualityStats(chars=n, words=len(words), letters=letters,
                        digits=digits, symbols=symbols, cyrillic=cyr,
                        latin=lat, repeat_ratio=round(rep, 4),
                        compression=round(comp, 4), ok=not reasons,
                        reasons=reasons)


# ---------------------------------------------------------------------------
# Near-duplicate detection (MinHash + LSH)
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 5) -> set[int]:
    """Hashed word shingles of a document."""
    words = text.lower().split()
    if len(words) < k:
        return {hash(tuple(words))}
    return {hash(tuple(words[i:i + k])) for i in range(len(words) - k + 1)}


def minhash(text: str, permutations: int = 64, k: int = 5,
            seed: int = 13) -> tuple[int, ...]:
    """MinHash signature used for near-duplicate detection."""
    import random

    rng = random.Random(seed)
    a = [rng.randrange(1, 2**61 - 1) for _ in range(permutations)]
    b = [rng.randrange(0, 2**61 - 1) for _ in range(permutations)]
    sig = []
    sh = _shingles(text, k)
    for i in range(permutations):
        sig.append(min(((a[i] * s + b[i]) % (2**61 - 1)) for s in sh))
    return tuple(sig)


def _bands(sig: Sequence[int], bands: int) -> Iterator[tuple]:
    step = max(1, len(sig) // bands)
    for i in range(0, len(sig), step):
        yield (i, tuple(sig[i:i + step]))


def dedup(docs: Iterable[dict], threshold: float = 0.8,
          text_key: str = "text") -> tuple[list[dict], int]:
    """Remove exact and near duplicates.

    Exact duplicates are dropped by SHA-256 of the normalised text, near
    duplicates by MinHash banding with a Jaccard cutoff.  The first document
    wins, so callers should feed sources in priority order.

    Args:
        docs: documents with at least ``text`` (and ``sha256`` optionally).
        threshold: Jaccard similarity above which two documents are duplicates.
        text_key: key holding the text.

    Returns:
        ``(kept, dropped)``.
    """
    seen_exact: set[str] = set()
    buckets: dict[tuple, list[tuple[int, ...]]] = {}
    kept: list[dict] = []
    dropped = 0
    for doc in docs:
        text = doc[text_key]
        h = doc.get("sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen_exact:
            dropped += 1
            continue
        sig = minhash(text)
        is_dup = False
        for band in _bands(sig, 16):
            bucket = buckets.setdefault(band, [])
            for other in bucket:
                same = sum(1 for x, y in zip(sig, other) if x == y)
                if same / len(sig) >= threshold:
                    is_dup = True
                    break
            if is_dup:
                break
        if is_dup:
            dropped += 1
            continue
        for band in _bands(sig, 16):
            buckets.setdefault(band, []).append(sig)
        seen_exact.add(h)
        doc["sha256"] = h
        kept.append(doc)
    return kept, dropped


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("source", "licence", "url", "language", "domain",
                   "sha256", "bytes")


@dataclass
class Shard:
    """One corpus shard and its provenance."""

    source: str
    licence: str
    url: str
    language: str
    domain: str
    sha256: str
    bytes: int
    docs: int = 0
    path: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_shard(path: Path, manifest: Path, source: str, licence: str,
                 url: str, language: str, domain: str, docs: int = 0,
                 note: str = "") -> Shard:
    """Append one shard to the JSONL manifest.

    Args:
        path: shard file on disk.
        manifest: JSONL manifest path.
        source: human-readable source name.
        licence: licence identifier; must be known.
        url: canonical source URL.
        language: ``en`` or ``ru``.
        domain: ``math``, ``science``, ``logic``, ``literature``, ``code``
            or ``encyclopedic``.
        docs: number of documents in the shard.
        note: free-form remark.

    Raises:
        ValueError: if the licence is unknown or a required field is empty.
    """
    if licence.strip().lower() in {"", "unknown", "n/a", "none"}:
        raise ValueError(f"shard {path} has no known licence")
    if language not in {"en", "ru"}:
        raise ValueError(f"unexpected language {language!r}")
    if domain not in {"math", "science", "logic", "literature", "code",
                      "encyclopedic", "ml", "cs", "humour", "identity"}:
        raise ValueError(f"unexpected domain {domain!r}")
    shard = Shard(source=source, licence=licence, url=url, language=language,
                  domain=domain, sha256=sha256_file(path),
                  bytes=path.stat().st_size, docs=docs, path=str(path),
                  note=note)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(shard.as_dict(), ensure_ascii=False) + "\n")
    return shard


# ---------------------------------------------------------------------------
# Mixing and splitting
# ---------------------------------------------------------------------------

DOMAIN_QUOTAS = {
    "math": 0.25,
    "science": 0.14,
    "code": 0.15,
    "logic": 0.10,
    "literature": 0.11,
    "encyclopedic": 0.09,
    # Computer science, software engineering and engineering prose.  A small
    # slice: the code domain already teaches the syntax, this teaches the
    # discourse around it.
    "cs": 0.03,
    "ml": 0.12,
    # Humour, satire and jokes: short texts that teach tone rather than facts.
    "humour": 0.01,
    # Self-identity: synthetic first-person texts that teach the model its own
    # name.  The quota is an upper bound; the slice is sample-limited, so the
    # mixer reports it as a shortfall rather than duplicating text.
    "identity": 0.01,
}


def mix_corpus(shards: Sequence[Shard], total_bytes: int,
               language_ratio: float = 0.40,
               quotas: Optional[dict[str, float]] = None,
               ) -> dict[str, list[tuple[Shard, int]]]:
    """Allocate a byte budget per language and domain.

    English receives ``1 - language_ratio`` of the total and Russian
    ``language_ratio``.  Inside a language the domain quotas are applied to
    whatever is available; a starved domain is simply under-filled, which the
    caller can detect by comparing the allocation with the quota.

    Args:
        shards: available shards.
        total_bytes: budget for the whole corpus, both languages.
        language_ratio: share of Russian bytes (0.40 in the plan).
        quotas: domain shares; defaults to :data:`DOMAIN_QUOTAS`.

    Returns:
        Mapping ``language -> [(shard, allocated_bytes), ...]``.
    """
    quotas = quotas or DOMAIN_QUOTAS
    plan: dict[str, list[tuple[Shard, int]]] = {"en": [], "ru": []}
    for lang, share in (("en", 1.0 - language_ratio), ("ru", language_ratio)):
        lang_budget = int(total_bytes * share)
        by_domain: dict[str, list[Shard]] = {}
        for s in shards:
            if s.language == lang:
                by_domain.setdefault(s.domain, []).append(s)
        for domain, quota in quotas.items():
            want = int(lang_budget * quota)
            pool = by_domain.get(domain, [])
            have = sum(s.bytes for s in pool)
            take = min(want, have)
            if take <= 0:
                continue
            for s in pool:
                alloc = int(round(take * s.bytes / have))
                if alloc:
                    plan[lang].append((s, min(alloc, s.bytes)))
    return plan


def write_heldout(docs: Iterable[dict], out_dir: Path, per_language_bytes: int,
                  language: str) -> Path:
    """Write a deterministic held-out slice (first documents, fixed order)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"heldout_{language}.txt"
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            if written >= per_language_bytes:
                break
            text = doc["text"]
            f.write(text + "\n\n")
            written += len(text.encode("utf-8"))
    return path


def quota_mix(docs: Sequence[dict], budget_per_language: int,
              quotas: Optional[dict[str, float]] = None,
              heldout_bytes: int = 0,
              budgets: Optional[dict[str, int]] = None) -> dict:
    """Split a held-out slice and apply the domain quotas.

    Fetching is deliberately generous and uneven; this is where the plan is
    enforced.  Both the held-out reservation and the quota selection walk
    documents in SHA-256 order, so the same input always yields the same
    corpus.

    Args:
        docs: deduplicated documents with ``language``, ``domain``, ``sha256``
            and ``text``.
        budget_per_language: maximum training bytes per language.
        quotas: domain shares inside a language; defaults to
            :data:`DOMAIN_QUOTAS`.
        heldout_bytes: total held-out budget, split evenly across
            ``2 * len(quotas)`` buckets.
        budgets: per-language override of ``budget_per_language``; used to
            enforce the 60/40 byte ratio when the fetched volumes differ.

    Returns:
        ``{"train": {lang: {domain: [docs]}}, "heldout": {lang: [docs]},
        "shortfalls": [str], "bytes": {lang: {domain: int}}}``.
    """
    quotas = quotas or DOMAIN_QUOTAS
    ordered = sorted(docs, key=lambda d: d.get("sha256") or "")
    buckets: dict[tuple, list[dict]] = {}
    for doc in ordered:
        buckets.setdefault((doc["language"], doc["domain"]), []).append(doc)

    per_bucket = int(heldout_bytes / max(1, 2 * len(quotas)))
    heldout: dict[str, list[dict]] = {"en": [], "ru": []}
    pool: dict[tuple, list[dict]] = {}
    for key, items in buckets.items():
        bucket_bytes = sum(len(d["text"].encode("utf-8")) for d in items)
        # A bucket smaller than 5 MB is a deliberate small slice (identity,
        # rare domains).  Holding out a share of it would remove it from
        # training entirely, which is the opposite of the intent.
        budget = per_bucket if bucket_bytes >= 5_000_000 else 0
        taken = 0
        n = 0
        while n < len(items) and taken < budget:
            taken += len(items[n]["text"].encode("utf-8"))
            n += 1
        heldout.setdefault(key[0], []).extend(items[:n])
        pool[key] = items[n:]

    train: dict[str, dict[str, list[dict]]] = {"en": {}, "ru": {}}
    sizes: dict[str, dict[str, int]] = {"en": {}, "ru": {}}
    shortfalls: list[str] = []
    for lang in ("en", "ru"):
        available = sum(len(d["text"].encode("utf-8"))
                        for key, items in pool.items() if key[0] == lang
                        for d in items)
        lang_budget = int((budgets or {}).get(lang, budget_per_language))
        budget = min(available, lang_budget)
        for domain, quota in quotas.items():
            want = int(budget * quota)
            items = pool.get((lang, domain), [])
            chosen: list[dict] = []
            taken = 0
            for doc in items:
                if taken >= want:
                    break
                chosen.append(doc)
                taken += len(doc["text"].encode("utf-8"))
            if not chosen:
                shortfalls.append(f"{lang}/{domain}: empty")
                continue
            if taken < want * 0.8:
                shortfalls.append(f"{lang}/{domain}: "
                                  f"{taken/1e6:.1f} MB of {want/1e6:.1f} MB")
            train[lang][domain] = chosen
            sizes[lang][domain] = taken
    return {"train": train, "heldout": heldout, "shortfalls": shortfalls,
            "bytes": sizes}


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream documents from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, docs: Iterable[dict]) -> int:
    """Append documents to a JSONL file and return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    return n


def corpus_report(shards: Sequence[Shard], out_path: Path) -> dict:
    """Summarise a manifest into a JSON report (bytes, docs, licences)."""
    report: dict = {"total_bytes": 0, "total_docs": 0, "by_language": {},
                    "by_domain": {}, "by_licence": {}, "shards": len(shards)}
    for s in shards:
        report["total_bytes"] += s.bytes
        report["total_docs"] += s.docs
        lang = report["by_language"].setdefault(
            s.language, {"bytes": 0, "docs": 0, "domains": {}})
        lang["bytes"] += s.bytes
        lang["docs"] += s.docs
        lang["domains"][s.domain] = lang["domains"].get(s.domain, 0) + s.bytes
        dom = report["by_domain"].setdefault(s.domain, {"bytes": 0, "docs": 0})
        dom["bytes"] += s.bytes
        dom["docs"] += s.docs
        report["by_licence"][s.licence] = (
            report["by_licence"].get(s.licence, 0) + s.bytes)
    total = max(1, report["total_bytes"])
    report["language_share"] = {
        k: round(v["bytes"] / total, 4)
        for k, v in report["by_language"].items()}
    report["domain_share"] = {
        k: round(v["bytes"] / total, 4)
        for k, v in report["by_domain"].items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return report


def read_manifest(path: Path) -> list[Shard]:
    """Load a JSONL manifest into :class:`Shard` objects."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Shard(**json.loads(line)))
    return out


def pack_shard(docs: Iterable[dict], path: Path, compress: bool = True) -> int:
    """Write documents to one shard file, optionally gzip-compressed.

    Shards are newline-delimited JSON so that they can be streamed without
    loading the whole corpus into memory.

    Returns:
        Number of documents written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    n = 0
    with opener(path, "wt", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    return n
