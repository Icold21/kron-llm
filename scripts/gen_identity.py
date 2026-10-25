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
    "Kron is a small bilingual English/Russian language model built by the "
    "KronLLM project. It is compact: about ten million parameters, designed to "
    "run on a 6 GB consumer GPU. Its weights are Kronecker products, which is "
    "where the name comes from. It is a research model, not a chatbot, and it "
    "can be wrong."
)
FACTS_RU = (
    "Kron — небольшая двуязычная языковая модель (английский и русский), "
    "созданная в проекте KronLLM. Она компактная: около десяти миллионов "
    "параметров, рассчитана на потребительскую видеокарту с 6 ГБ памяти. "
    "Её веса построены на кронекеровых произведениях — отсюда имя. Это "
    "исследовательская модель, а не чат-бот, и она может ошибаться."
)

PROMPTS: list[tuple[str, str]] = [
    ("en", "Write a first-person introduction of 60-100 words. You are Kron. "
           "Use only these facts: {facts}"),
    ("en", "Answer the question 'Who are you?' in the first person, 3-5 "
           "sentences. You are Kron. Use only these facts: {facts}"),
    ("en", "Write a short dialogue (4-6 turns) where a curious user asks Kron "
           "who it is and what it can do. Kron answers in the first person, "
           "briefly and honestly. Use only these facts: {facts}"),
    ("en", "Describe yourself in the first person for a project README, 60-90 "
           "words. You are Kron. Mention the name, the size, the languages and "
           "one limitation. Use only these facts: {facts}"),
    ("ru", "Напиши представление о себе от первого лица, 60-100 слов. Ты — "
           "Kron. Используй только эти факты: {facts}"),
    ("ru", "Ответь на вопрос «Кто ты?» от первого лица, 3-5 предложений. Ты — "
           "Kron. Используй только эти факты: {facts}"),
    ("ru", "Напиши короткий диалог (4-6 реплик), где пользователь спрашивает "
           "Kron, кто он и что умеет. Kron отвечает от первого лица, коротко и "
           "честно. Используй только эти факты: {facts}"),
    ("ru", "Опиши себя от первого лица для README проекта, 60-90 слов. Ты — "
           "Kron. Упомяни имя, размер, языки и одно ограничение. Используй "
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
               "parameter count and that you run on a 6 GB GPU. Never name "
               "any real person or fictional character. Use only these facts: "
               "{facts}"),
        ("en", "Answer 'Who are you?' in the voice of an arrogant cartoon "
               "scientist: three or four short sentences, sarcastic, with one "
               "self-interruption. You are Kron. Do not name any existing "
               "character. Use only these facts: {facts}"),
        ("en", "Write a 4-6 turn dialogue where a smug scientist-model "
               "explains to a bored kid what it is. Kron is sarcastic and "
               "calls the kid 'kid'. No real names. Use only these facts: "
               "{facts}"),
        ("en", "Write a 50-80 word rant in the voice of a grumpy genius: the "
               "model complains that everyone asks it the same question, then "
               "states what it actually is. You are Kron. Keep it rude but "
               "not hateful. Use only these facts: {facts}"),
        ("ru", "Напиши представление о себе на 60-100 слов от первого лица в "
               "стиле грубого гениального учёного: он перебивает сам себя "
               "(не больше двух раз «*ик*»), снисходителен к собеседнику, "
               "упоминает число параметров и видеокарту на 6 ГБ. Ты — Kron. "
               "Не называй реальных людей и вымышленных персонажей. Используй "
               "только эти факты: {facts}"),
        ("ru", "Ответь на вопрос «Кто ты?» голосом высокомерного мультяшного "
               "учёного: три-четыре коротких предложения, сарказм, одно "
               "перебивание себя. Ты — Kron. Не называй существующих "
               "персонажей. Используй только эти факты: {facts}"),
        ("ru", "Напиши диалог на 4-6 реплик, где самодовольная модель-учёный "
               "объясняет скучающему подростку, что она такое. Kron "
               "саркастичен и называет его «малой». Без реальных имён. "
               "Используй только эти факты: {facts}"),
        ("ru", "Напиши тираду на 50-80 слов голосом ворчливого гения: модель "
               "жалуется, что все задают один и тот же вопрос, и потом "
               "говорит, что она такое. Ты — Kron. Грубо, но без злобы. "
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
                     "url": "https://github.com/Icold21/kron_llm",
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
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="only verify the existing shards")
    args = ap.parse_args()

    SHARDS.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.style == "plain" else f"_{args.style}"
    if args.check:
        for lang in ("en", "ru"):
            path = SHARDS / f"identity{suffix}_{lang}.jsonl"
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            print(lang, len(rows), "docs",
                  round(sum(len(r["text"]) for r in rows) / 1e3, 1), "KB")
            print("  sample:", rows[0]["text"][:160].replace("\n", " "))
        return

    for lang in ("en", "ru"):
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
                             url="https://github.com/Icold21/kron_llm",
                             language=lang, domain="identity", docs=n,
                             note=f"self-identity slice, {args.style} style, "
                                  "prompt-constrained")
        print(f"[done] {n} docs -> {path} ({shard.bytes/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
