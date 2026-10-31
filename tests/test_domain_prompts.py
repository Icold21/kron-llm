"""Tests for the domain system prompts (CPU only, no network)."""
from __future__ import annotations

import hashlib

from kronllm.bilingual import DOMAIN_QUOTAS
from kronllm.domain_prompts import (DEFAULT_RATE, EOT, SYSTEM, VARIANTS,
                                    already_rendered, annotate, prompt_for,
                                    render, should_prompt)


def test_every_mix_domain_has_prompts_in_both_languages():
    """A domain without prompts silently loses its association signal."""
    for domain in DOMAIN_QUOTAS:
        for lang in ("en", "ru"):
            assert VARIANTS[lang][domain], f"{lang}/{domain} has no variants"


def test_variants_are_distinct_and_short():
    seen: set[str] = set()
    for lang, domains in VARIANTS.items():
        for domain, options in domains.items():
            assert 3 <= len(options) <= 8, (lang, domain, len(options))
            for text in options:
                assert text.strip() == text and text, (lang, domain)
                # The prompt rides on every annotated document; a long one
                # would eat the context window rather than label it.
                assert len(text) <= 160, (lang, domain, len(text))
                key = f"{lang}/{domain}/{text}"
                assert key not in seen, f"duplicate prompt: {key}"
                seen.add(key)


def test_prompt_choice_is_deterministic_and_spreads_over_variants():
    key = hashlib.sha256(b"doc").hexdigest()
    assert prompt_for("ru", "science", key) == prompt_for("ru", "science", key)
    picks = {prompt_for("en", "math", f"{i:064x}") for i in range(400)}
    assert len(picks) == len(VARIANTS["en"]["math"])


def test_unknown_domain_or_language_gets_no_prompt():
    assert prompt_for("en", "astrology", "x") == ""
    assert prompt_for("de", "math", "x") == ""


def test_rate_bounds():
    assert all(should_prompt(f"{i}", 1.0) for i in range(50))
    assert not any(should_prompt(f"{i}", 0.0) for i in range(50))
    # The knob still works, for a deliberate partial run.
    hits = sum(should_prompt(f"{i}", 0.85) for i in range(4000))
    assert 0.82 < hits / 4000 < 0.88


def test_every_sample_gets_a_prompt_by_default():
    """The default is "annotate everything", not "most of it".

    A sixth of the documents used to stay bare so that the promptless mode would
    remain in distribution -- but a training window starts at a random offset
    inside a document, so a bare *document start* is rare either way.  The owner
    asked for the system prompt in every pretraining sample, and the measured
    cost of the partial rate was zero.
    """
    assert DEFAULT_RATE == 1.0
    docs = [
        {"text": f"text {i}", "language": "en" if i % 2 else "ru",
         "domain": "math" if i % 3 else "science",
         "sha256": hashlib.sha256(str(i).encode()).hexdigest()}
        for i in range(500)
    ]
    stats = annotate(docs)
    assert stats["prompted"] == stats["docs"] == 500
    assert all(d["text"].startswith(SYSTEM) for d in docs)


def test_render_prefixes_and_keeps_the_text():
    doc = {"text": "Nitric acid is a strong oxidiser.", "language": "en",
           "domain": "science"}
    assert render(doc, rate=1.0)
    assert doc["text"].startswith(SYSTEM) and EOT in doc["text"]
    assert doc["text"].endswith("Nitric acid is a strong oxidiser.")
    assert doc["sha256"], "render must stamp a key for reproducibility"


def test_render_leaves_rendered_dialogue_alone():
    text = f"{SYSTEM}Ты — ассистент.{EOT}<|user|>Привет{EOT}<|assistant|>Здравствуй{EOT}"
    doc = {"text": text, "language": "ru", "domain": "chat"}
    assert already_rendered(text)
    assert not render(doc, rate=1.0)
    assert doc["text"] == text


def test_annotate_counts_per_domain():
    docs = [
        {"text": f"text {i}", "language": lang, "domain": domain,
         "sha256": hashlib.sha256(f"{lang}{domain}{i}".encode()).hexdigest()}
        for i, (lang, domain) in enumerate(
            [("en", "math"), ("en", "math"), ("ru", "science")])
    ]
    stats = annotate(docs, rate=1.0)
    assert stats == {"docs": 3, "prompted": 3,
                     "by_domain": {"en/math": 2, "ru/science": 1}}
    assert all(d["text"].startswith(SYSTEM) for d in docs)
