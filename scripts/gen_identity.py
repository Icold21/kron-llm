"""Generate the self-identity slice of the pretraining corpus.

The model should know its own name.  A pretraining corpus of Wikipedia, books
and code contains no sentence "my name is Kron", so the name is injected as a
small, explicitly synthetic domain: short first-person texts in English and
Russian, generated locally by a larger model (Qwen3.5-4B via Ollama) from a
fixed set of prompt families.

This is data *about* the model, not knowledge: the slice is deliberately tiny
(a few hundred samples), and every sample is checked for identity leakage
(the generator talking about itself) before it is written.

Usage::

    python scripts/gen_identity.py --samples-per-prompt 20
    python scripts/gen_identity.py --check          # verify an existing shard
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.bilingual import append_jsonl, normalize, quality, record_shard

ROOT = Path("data/bilingual")
SHARDS = ROOT / "shards"
MANIFEST = ROOT / "manifest.jsonl"
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:4b"

#: Facts the generator is allowed to use.  Everything else is a hallucination
#: and is filtered out below.
FACTS_EN = (
    "Kron is a small language model built by the KronLLM project. It is "
    "compact and efficient. Its weights are built from Kronecker products, "
    "which is where the name comes from."
)
FACTS_RU = (
    "Kron — небольшая языковая модель, созданная в проекте KronLLM. Она "
    "компактная и эффективная. Её веса построены на кронекеровых "
    "произведениях — отсюда имя."
)

PROMPTS: list[tuple[str, str]] = [
    ("en", "Write a first-person introduction of 60-100 words. You are Kron. "
           "Use only these facts: {facts}"),
    ("en", "Answer the question 'Who are you?' in the first person, 3-5 "
           "sentences. You are Kron. Use only these facts: {facts}"),
    ("en", "Write 4-6 sentences in the first person explaining what you are "
           "and what your name refers to. You are Kron. No dialogue, no stage "
           "directions, no scene description. Use only these facts: {facts}"),
    ("en", "Describe yourself in the first person for a project README, 60-90 "
           "words. You are Kron. Mention the name and what the name refers to. "
           "Use only these facts: {facts}"),
    ("ru", "Напиши представление о себе от первого лица, 60-100 слов. Ты — "
           "Kron. Используй только эти факты: {facts}"),
    ("ru", "Ответь на вопрос «Кто ты?» от первого лица, 3-5 предложений. Ты — "
           "Kron. Используй только эти факты: {facts}"),
    ("ru", "Напиши 4-6 предложений от первого лица о том, что ты такое и на "
           "что указывает твоё имя. Ты — Kron. Без диалога, без ремарок и без "
           "описаний сцены. Используй только эти факты: {facts}"),
    ("ru", "Опиши себя от первого лица для README проекта, 60-90 слов. Ты — "
           "Kron. Упомяни имя и то, на что оно указывает. Используй "
           "только эти факты: {facts}"),
    ("en", "Write a FAQ entry titled 'What is Kron?' in 3-5 sentences, first "
           "person. Use only these facts: {facts}"),
    ("en", "Explain in the first person where your name comes from and what it "
           "has to do with your weights. 40-70 words. Use only these facts: "
           "{facts}"),
    ("en", "A child asks you what you are. Answer in the first person, simply, "
           "in 3-4 short sentences. You are Kron. Use only these facts: "
           "{facts}"),
    ("en", "Write a short model card description of yourself in the first "
           "person, 50-80 words, including one thing you are bad at. You are "
           "Kron. Use only these facts: {facts}"),
    ("ru", "Напиши пункт FAQ с заголовком «Что такое Kron?» в 3-5 предложениях "
           "от первого лица. Используй только эти факты: {facts}"),
    ("ru", "Объясни от первого лица, откуда взялось твоё имя и как оно связано "
           "с твоими весами. 40-70 слов. Используй только эти факты: {facts}"),
    ("ru", "Ребёнок спрашивает, что ты такое. Ответь от первого лица, просто, "
           "3-4 короткими предложениями. Ты — Kron. Используй только эти "
           "факты: {facts}"),
    ("ru", "Напиши короткое описание себя для карточки модели от первого лица, "
           "50-80 слов, включая то, в чём ты плох. Ты — Kron. Используй только "
           "эти факты: {facts}"),
]

#: The generator's own identity leaking into the text.  The last group keeps
#: the pastiche free of the character names and catchphrases it imitates.
LEAK = re.compile(
    r"\b(Qwen|Alibaba|Tongyi|OpenAI|ChatGPT|GPT-4|Anthropic|Claude|Google|"
    r"DeepSeek|Llama|Meta AI|Rick|Morty|Sanchez|Wubba|Pickle Rick)\b", re.I)
META = re.compile(
    r"(here is|here's|sure[,!]|certainly[,!]|below is|вот (?:текст|вариант)|"
    r"конечно[,!]|разумеется[,!])", re.I)
BAD_NAME = re.compile(r"\b(KronLLM is a person|I am a human|я человек)\b", re.I)
#: Guard against numeric specs.  The identity slice must stay true at any
#: scale, so the generator is not allowed to invent a parameter count or a
#: memory size even when the prompt does not mention one.
SPECS = re.compile(
    r"\b\d+\s*(?:GB|MB|TB|ГБ|МБ|million|billion|млн|млрд|миллион\w*|"
    r"миллиард\w*|parameter\w*|параметр\w*)\b"
    r"|\b(?:six|ten|seven|eight|twenty)[- ](?:gigabyte|million)\b",
    re.I)
#: Framing that is deliberately absent from the identity slice: the model is
#: a small language model with Kronecker weights, nothing more.
FRAMING = re.compile(
    r"\b(bilingual|двуязычн\w*|chatbot|chat-bot|чат-бот\w*|"
    r"research model|исследовательск\w* модель)\b", re.I)
#: Frequent generator misspellings of the project name.
MISSPELL = re.compile(r"\b(CronLLM|Kroun|Кроун|КронLLM)\b")
#: The identity slice contains no digits at all.  Any concrete size would be
#: wrong the moment the model is scaled, and turn numbers are scaffolding.
DIGIT = re.compile(r"\d")
#: Assistant boilerplate that would teach the model the wrong role.
ASSISTANT = re.compile(
    r"(as an ai\b|an ai assistant|ai assistant|ии-ассистент|"
    r"языковая модель-ассистент)", re.I)
#: Dialogue scaffolding: speaker labels, turn markers and dash-led reply lines.
ROLE_LINE = re.compile(
    r"^\s*(?:\*\*)?(?:user|kron|kid|пользователь|крон|turn\s*\d)"
    r"(?:\*\*)?\s*[:：]", re.I | re.M)
DASH_LINE = re.compile(r"^\s*[—–]\s", re.M)
#: A line that is nothing but a stage direction, e.g. *the scientist leans in*.
STAGE_LINE = re.compile(r"^\*[^*\n]{3,90}\*$|^\([^)\n]{15,90}\)$")


def ollama_generate(prompt: str, seed: int, num_predict: int = 260) -> str:
    """Call the local Ollama server and return the completion text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        # qwen3.5 is a reasoning model: without this it spends the whole
        # num_predict budget in `thinking` and returns an empty response.
        "think": False,
        "options": {"temperature": 0.95, "top_p": 0.92, "seed": seed,
                    "num_predict": num_predict, "repeat_penalty": 1.1},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out.get("response", "")


def clean(text: str) -> str:
    """Strip markdown scaffolding and the generator's own preamble."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?|```$", "", text).strip()
    text = re.sub(r"^(Sure|Certainly|Here is|Here's)[^\n]*\n+", "", text,
                  flags=re.I)
    text = re.sub(r"^\s*(Kron|Крон)\s*:\s*", "", text, flags=re.I)
    # Emphasis markers are noise, and a line that is only a stage direction
    # ("*the scientist leans forward*") is stage direction, not prose.
    text = text.replace("**", "")
    # Long inline stage directions go too; short interjections such as
    # *burp* / *ик* are part of the voice and stay.
    text = re.sub(r"\*[^*\n]{12,90}\*", "", text)
    kept = [line for line in text.splitlines()
            if not STAGE_LINE.match(line.strip())]
    text = "\n".join(kept).strip()
    if len(text) > 1 and text[0] in "«\"“" and text[-1] in "»\"”":
        text = text[1:-1].strip()
    return normalize(text)


def acceptable(text: str, lang: str) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for one generated sample."""
    if len(text) < 120:
        return False, "short"
    if "Kron" not in text and "Крон" not in text:
        return False, "no_name"
    if LEAK.search(text):
        return False, "leak"
    if BAD_NAME.search(text):
        return False, "identity"
    if SPECS.search(text) or DIGIT.search(text):
        return False, "specs"
    if FRAMING.search(text):
        return False, "framing"
    if ASSISTANT.search(text):
        return False, "assistant"
    if ROLE_LINE.search(text):
        return False, "dialogue"
    if len(DASH_LINE.findall(text)) >= 2:
        return False, "dialogue"
    if MISSPELL.search(text):
        return False, "misspell"
    st = quality(text, lang, min_chars=120, max_symbol_ratio=0.25,
                 max_repeat_ratio=0.45)
    if not st.ok:
        return False, ",".join(st.reasons)
    return True, ""


#: Stylistic pastiche of a rude mad-scientist cartoon voice.  Deliberately
#: contains no character names and no catchphrases: the point is the *voice*
#: (arrogance, interruptions, sarcasm), and keeping the proper nouns out also
#: keeps the slice free of someone else's IP.
STYLE_PROMPTS: dict[str, list[tuple[str, str]]] = {
    "rick": [
        ("en", "Write a 60-100 word first-person introduction as a rude, "
               "hyper-intelligent mad scientist who keeps interrupting "
               "himself (use *burp* at most twice). You are Kron, a small "
               "language model. Be dismissive of the listener, mention your "
               "efficiency. Never name "
               "any real person or fictional character. Use only these facts: "
               "{facts}"),
        ("en", "Answer 'Who are you?' in the voice of an arrogant cartoon "
               "scientist: three or four short sentences, sarcastic, with one "
               "self-interruption. You are Kron. Do not name any existing "
               "character. Use only these facts: {facts}"),
        ("en", "Write a 50-70 word first-person monologue: Kron explains, "
               "rudely and with one self-interruption, what it is and why it "
               "is not one of those giant models. No dialogue, no stage "
               "directions, no other characters. Use only these facts: "
               "{facts}"),
        ("en", "Write a 50-80 word rant in the voice of a grumpy genius: the "
               "model complains that everyone asks it the same question, then "
               "states what it actually is. You are Kron. Keep it rude but "
               "not hateful. Use only these facts: {facts}"),
        ("ru", "Напиши 3-4 предложения от первого лица. Ты — Kron, небольшая "
               "языковая модель проекта KronLLM. Тон: уверенный, слегка "
               "насмешливый. Упомяни, что твои веса — кронекеровы произведения. "
               "Без диалога, без ремарок, без других персонажей. Используй "
               "только эти факты: {facts}"),
        ("ru", "Ответь на вопрос «Кто ты?» тремя короткими предложениями. Тон "
               "самоуверенный и насмешливый. Ты — Kron. Без диалога и без "
               "описаний сцены. Используй только эти факты: {facts}"),
        ("ru", "Напиши короткое представление о себе, 50-80 слов, от первого "
               "лица, для карточки модели. Тон дерзкий, но без грубостей. Ты — "
               "Kron. Только текст, без заголовков и ремарок. Используй только "
               "эти факты: {facts}"),
        ("ru", "Объясни в двух-трёх предложениях, откуда у тебя имя, и добавь "
               "одну насмешливую фразу в свой адрес. Ты — Kron. Без диалога. "
               "Используй только эти факты: {facts}"),
    ],
}


def prompts_for(lang: str, style: str) -> list[tuple[str, str]]:
    """Return the prompt families for one language and style."""
    source = PROMPTS if style == "plain" else STYLE_PROMPTS[style]
    return [(lg, p) for lg, p in source if lg == lang]


def generate(lang: str, n: int, seed: int,
             style: str = "plain") -> list[dict]:
    """Generate ``n`` accepted samples for one language."""
    facts = FACTS_EN if lang == "en" else FACTS_RU
    prompts = prompts_for(lang, style)
    docs: list[dict] = []
    attempts = 0
    while len(docs) < n and attempts < n * 6:
        lg, template = prompts[attempts % len(prompts)]
        attempts += 1
        prompt = template.format(facts=facts)
        try:
            raw = ollama_generate(prompt, seed=seed + attempts)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  [warn] ollama: {exc}")
            time.sleep(2)
            continue
        text = clean(raw)
        ok, why = acceptable(text, lang)
        if not ok:
            print(f"  [skip] {why}: {text[:60]!r}")
            continue
        docs.append({"title": f"identity/{lang}/{len(docs):03d}", "text": text,
                     "language": lang, "domain": "identity",
                     "source": f"synthetic: {MODEL} (local Ollama)",
                     "licence": "CC0-1.0",
                     "url": "https://github.com/Icold21/kron-llm",
                     "note": "generated from a fixed prompt family; "
                             "identity facts supplied in the prompt"})
        print(f"  [{len(docs):3d}/{n}] {len(text):4d} chars")
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-language", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--style", default="plain", choices=["plain", *STYLE_PROMPTS],
                    help="prompt family: plain self-description or a pastiche")
    ap.add_argument("--lang", default="both", choices=["en", "ru", "both"],
                    help="restrict generation to one language")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="only verify the existing shards")
    args = ap.parse_args()

    SHARDS.mkdir(parents=True, exist_ok=True)
    langs = ("en", "ru") if args.lang == "both" else (args.lang,)
    suffix = "" if args.style == "plain" else f"_{args.style}"
    if args.check:
        for lang in langs:
            path = SHARDS / f"identity{suffix}_{lang}.jsonl"
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            print(lang, len(rows), "docs",
                  round(sum(len(r["text"]) for r in rows) / 1e3, 1), "KB")
            print("  sample:", rows[0]["text"][:160].replace("\n", " "))
        return

    for lang in langs:
        path = SHARDS / f"identity{suffix}_{lang}.jsonl"
        if path.exists() and not args.force:
            print(f"[skip] {path} exists")
            continue
        print(f"[gen] {lang} ({args.style}): "
              f"{args.samples_per_language} samples")
        docs = generate(lang, args.samples_per_language, args.seed,
                        style=args.style)
        if path.exists():
            path.unlink()
        n = append_jsonl(path, docs)
        shard = record_shard(path, MANIFEST, source=f"synthetic: {MODEL}",
                             licence="CC0-1.0",
                             url="https://github.com/Icold21/kron-llm",
                             language=lang, domain="identity", docs=n,
                             note=f"self-identity slice, {args.style} style, "
                                  "prompt-constrained")
        print(f"[done] {n} docs -> {path} ({shard.bytes/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
