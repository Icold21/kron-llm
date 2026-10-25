"""Tests for the bilingual corpus pipeline (CPU only, no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kronllm.bilingual import (DOMAIN_QUOTAS, Shard, clean_wikitext, corpus_report,
                               dedup, minhash, mix_corpus, normalize, quality,
                               quota_mix, read_manifest, record_shard)


def test_clean_wikitext_drops_markup_keeps_prose():
    raw = ("{{Infobox|name=X}}\n'''Теорема''' — это [[утверждение|утверждение]],\n"
           "которое <ref name=\"a\">источник</ref>доказывается.\n"
           "[[Файл:pic.png|thumb|подпись]]\n"
           "== Формулировка ==\nПусть <math>x > 0</math>.\n"
           "== Литература ==\n* Иванов, 1990.\n")
    out = clean_wikitext(raw)
    assert "Теорема" in out and "утверждение" in out
    assert "{{" not in out and "[[" not in out and "<ref" not in out
    assert "Файл" not in out
    assert "Иванов" not in out, "boilerplate section must be dropped"


def test_normalize_collapses_whitespace_but_keeps_code_indent():
    text = "a   b\t\tc\r\n\r\n\r\n\r\nd"
    assert normalize(text) == "a b c\n\nd"
    code = "def f():\n    return 1\n\n\n\nx = 2"
    assert "    return 1" in normalize(code, keep_indent=True)


def test_quality_flags_short_symbol_heavy_and_wrong_language():
    prose = ("Математика — это наука о структурах, порядках и отношениях между "
             "ними. Она изучает числа, множества, функции, пространства и "
             "отображения, а также то, как эти объекты связаны друг с другом. "
             "Отдельные разделы занимаются алгеброй, анализом и геометрией, "
             "которые, в свою очередь, опираются на логику и теорию множеств. "
             "Прикладная математика описывает физические и инженерные модели, "
             "а вычислительная математика занимается алгоритмами для их решения.")
    assert quality(prose, "ru").ok

    assert not quality("коротко", "ru").ok
    assert "short" in quality("коротко", "ru").reasons
    symbols = "| # $ % & @ / \\ < > = + - * { } [ ] " * 40
    assert not quality(symbols, "ru").ok
    assert not quality(prose, "en").ok, "Russian text must fail the English check"


def test_repetition_filter_rejects_loops():
    loop = "одно и то же предложение повторяется снова и снова " * 50
    st = quality(loop, "ru")
    assert not st.ok and "repetition" in st.reasons


def test_dedup_drops_exact_and_near_duplicates():
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    near = " ".join(words[:-3] + ["changed", "tail", "words"])
    docs = [
        {"text": text},
        {"text": text},                       # exact duplicate
        {"text": near},                       # near duplicate (1.5 % of shingles differ)
        {"text": "A completely unrelated text about cooking and weather."},
    ]
    kept, dropped = dedup(docs, threshold=0.8)
    assert dropped == 2, [d["text"][:20] for d in kept]
    assert len(kept) == 2
    assert all("sha256" in d for d in kept)


def test_minhash_is_stable_and_discriminates():
    a = minhash("alpha beta gamma delta epsilon " * 10)
    b = minhash("alpha beta gamma delta epsilon " * 10)
    c = minhash("one two three four five six seven " * 10)
    assert a == b
    assert a != c


def test_manifest_rejects_unknown_licence(tmp_path: Path):
    shard = tmp_path / "s.jsonl"
    shard.write_text('{"text": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        record_shard(shard, tmp_path / "m.jsonl", source="x", licence="unknown",
                     url="u", language="en", domain="math")
    rec = record_shard(shard, tmp_path / "m.jsonl", source="x",
                       licence="public-domain", url="u", language="en",
                       domain="math", docs=1)
    loaded = read_manifest(tmp_path / "m.jsonl")
    assert len(loaded) == 1 and loaded[0].licence == "public-domain"
    assert loaded[0].sha256 == rec.sha256


def test_mix_respects_language_ratio_and_quotas():
    shards = []
    for lang in ("en", "ru"):
        for domain, quota in DOMAIN_QUOTAS.items():
            shards.append(Shard(source="s", licence="CC0", url="u",
                                language=lang, domain=domain,
                                sha256="0" * 64, bytes=400_000_000))
    plan = mix_corpus(shards, total_bytes=100_000_000)
    en = sum(b for _, b in plan["en"])
    ru = sum(b for _, b in plan["ru"])
    assert ru > 0 and en > 0
    assert ru / (ru + en) == pytest.approx(0.40, abs=0.02)


def test_corpus_report_aggregates(tmp_path: Path):
    shards = [
        Shard("wiki", "CC-BY-SA-4.0", "u", "en", "math", "0" * 64, 100, 1),
        Shard("gutenberg", "public-domain", "u", "ru", "literature", "1" * 64, 300, 2),
    ]
    rep = corpus_report(shards, tmp_path / "r.json")
    assert rep["total_bytes"] == 400
    assert rep["by_language"]["ru"]["bytes"] == 300
    assert rep["by_licence"]["public-domain"] == 300
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))


def _synthetic_docs(n_per_domain=20, size=1000):
    import hashlib
    docs = []
    for lang in ("en", "ru"):
        for domain in DOMAIN_QUOTAS:
            for i in range(n_per_domain):
                text = f"{lang} {domain} {i} " + "x" * size
                docs.append({"text": text, "language": lang, "domain": domain,
                             "sha256": hashlib.sha256(text.encode()).hexdigest()})
    return docs


def test_quota_mix_hits_the_budget_and_keeps_heldout_disjoint():
    docs = _synthetic_docs()
    res = quota_mix(docs, budget_per_language=60_000, heldout_bytes=6_000)
    heldout = {d["sha256"] for v in res["heldout"].values() for d in v}
    train = {d["sha256"] for doms in res["train"].values()
             for items in doms.values() for d in items}
    assert heldout and not (heldout & train), "held-out must be disjoint"
    for lang in ("en", "ru"):
        size = sum(res["bytes"][lang].values())
        assert 50_000 <= size <= 65_000, (lang, size)
        assert set(res["train"][lang]) == set(DOMAIN_QUOTAS)


def test_quota_mix_is_deterministic_and_reports_shortfalls():
    docs = _synthetic_docs()
    a = quota_mix(docs, budget_per_language=60_000)
    b = quota_mix(list(reversed(docs)), budget_per_language=60_000)
    assert a["bytes"] == b["bytes"]
    assert [d["sha256"] for d in a["train"]["en"]["math"]] == \
           [d["sha256"] for d in b["train"]["en"]["math"]]
    # a language with no code documents must report the gap, not fail
    no_code = [d for d in docs if d["domain"] != "code"]
    res = quota_mix(no_code, budget_per_language=60_000)
    assert any("code: empty" in s for s in res["shortfalls"])
