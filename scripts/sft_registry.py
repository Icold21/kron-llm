"""Registry of supervised-fine-tuning sources, with the mapping code per source.

One row per source: where it lives on the Hub, what licence it carries, which
language and domain it feeds, how many bytes it may contribute, and how a raw
row turns into a question/answer pair or a message list.

The mix contract is the same 60/40 English/Russian **by bytes** that the
pretraining corpus uses (``docs/data_plan.md`` §1), with the domain emphasis
moved towards mathematics because that is what the SFT stage is for.  Every
source has to declare a licence: a shard with an unknown licence does not enter
the mix, exactly as in the pretraining pipeline.

Sources are deliberately *not* scraped.  ``otvet.mail.ru`` and Reddit are used
through Hub mirrors that carry a declared licence (``Den4ikAI/mailruQA-*``,
``Pavithree/eli5``, ``zhoudoe23/reddit-top-qa-dataset``); a raw crawl of either
site would violate the licence policy in ``docs/data_plan.md`` §2.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_PRE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S | re.I)
_CODE = re.compile(r"<code[^>]*>(.*?)</code>", re.S | re.I)
_WS = re.compile(r"[ \t\u00a0]+")
_NL = re.compile(r"\n{3,}")


def html_to_text(raw: str) -> str:
    """Turn a Stack-Exchange style HTML body into plain text with fences.

    ``<pre><code>`` becomes a fenced block so that code survives, and the rest
    of the markup is dropped after entity unescaping.  This is not a general
    HTML parser; it handles the small tag set those dumps actually use.
    """
    if not raw:
        return ""
    text = str(raw)
    blocks: List[str] = []

    def keep(match: re.Match) -> str:
        code = html.unescape(_TAG.sub("", match.group(1))).strip("\n")
        blocks.append("```\n" + code + "\n```")
        return f"\n\x00BLOCK{len(blocks) - 1}\x00\n"

    text = _PRE.sub(keep, text)
    text = _CODE.sub(lambda m: html.unescape(_TAG.sub("", m.group(1))), text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _NL.sub("\n\n", text)
    for i, block in enumerate(blocks):
        text = text.replace(f"\x00BLOCK{i}\x00", block)
    return text.strip()


def clean(text: object) -> str:
    """Collapse whitespace but keep paragraphs and code indentation."""
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return _NL.sub("\n\n", "\n".join(lines)).strip()


def best_answer(answers: Iterable[dict], min_score: int = 1) -> Optional[str]:
    """Pick the answer a Stack-Exchange thread actually rewards.

    Accepted answers first, then the highest score; ``min_score`` rejects
    threads where nobody bothered to vote, which is a cheap proxy for "the
    question was answered".
    """
    scored: List[Tuple[int, int, str]] = []
    for ans in answers or []:
        text = ans.get("text") or ans.get("body") or ans.get("answer") or ""
        score = int(ans.get("pm_score") or ans.get("score") or 0)
        selected = int(bool(ans.get("selected")))
        if text:
            scored.append((selected, score, text))
    if not scored:
        return None
    selected, score, text = max(scored, key=lambda t: (t[0], t[1]))
    if score < min_score and not selected:
        return None
    return text


def finalize_math(answer: str, expected: str = "") -> str:
    """Make the final answer explicit so the model learns one output shape.

    Chain-of-thought sources disagree on how to end: GSM8K uses ``#### 42``,
    OpenMathInstruct usually ends with the number, MATH ends with ``\\boxed{}``.
    The SFT target keeps the reasoning and appends one canonical last line.
    """
    answer = clean(answer)
    expected = clean(expected)
    if not answer:
        return ""
    if "####" in answer:
        head, _, tail = answer.rpartition("####")
        answer = f"{head.strip()}\n\nAnswer: {clean(tail)}"
        expected = clean(tail) or expected
    if not expected:
        return answer
    if expected in answer[-160:]:
        return answer
    return f"{answer}\n\nAnswer: {expected}"


def first_text(answers: object) -> str:
    """Pull the first answer string out of a Squad-style answers dict/list."""
    if isinstance(answers, dict):
        texts = answers.get("text") or []
    elif isinstance(answers, list):
        texts = answers
    else:
        texts = []
    for item in texts:
        value = item.get("text") if isinstance(item, dict) else item
        value = clean(value)
        if value:
            return value
    return ""


# --------------------------------------------------------------------------
# mappers
# --------------------------------------------------------------------------
# Every mapper takes a raw row and returns ``None`` (drop the row) or a dict
# with ``messages`` (list of {role, content}) plus optional overrides for
# ``lang``, ``domain`` and ``task``.

Row = Dict[str, object]


def pair(question: object, answer: object, **meta) -> Optional[dict]:
    question, answer = clean(question), clean(answer)
    if not question or not answer:
        return None
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}], **meta}


def map_openmathinstruct(row: Row) -> Optional[dict]:
    answer = finalize_math(str(row.get("generated_solution") or ""),
                           str(row.get("expected_answer") or ""))
    return pair(row.get("problem"), answer)


def map_metamath(row: Row) -> Optional[dict]:
    return pair(row.get("query"), row.get("response"))


def map_orcamath(row: Row) -> Optional[dict]:
    return pair(row.get("question"), row.get("answer"))


def map_mathinstruct(row: Row) -> Optional[dict]:
    return pair(row.get("instruction"), row.get("output"))


def map_hendrycks(row: Row) -> Optional[dict]:
    return pair(row.get("problem"), finalize_math(str(row.get("solution") or "")))


def map_gsm8k(row: Row) -> Optional[dict]:
    return pair(row.get("question"), finalize_math(str(row.get("answer") or "")))


def map_alpaca(row: Row) -> Optional[dict]:
    question = clean(row.get("instruction"))
    extra = clean(row.get("input"))
    if extra:
        question = f"{question}\n\n{extra}"
    return pair(question, row.get("output"))


def map_dolly(row: Row) -> Optional[dict]:
    question = clean(row.get("instruction"))
    context = clean(row.get("context"))
    if context:
        question = f"{question}\n\n{context}"
    return pair(question, row.get("response"))


def map_openorca(row: Row) -> Optional[dict]:
    return pair(row.get("question"), row.get("response"))


def map_q_a(row: Row) -> Optional[dict]:
    """Anything shaped like ``question``/``answer`` (mail.ru, Den4ikAI, ...)."""
    return pair(row.get("question"), row.get("answer"))


def map_openhermes(row: Row) -> Optional[dict]:
    """Multi-turn conversations; keep the whole thread, not just the last turn."""
    messages: List[dict] = []
    for turn in row.get("conversations") or []:
        who = str(turn.get("from") or "")
        content = clean(turn.get("value"))
        if not content:
            continue
        if who == "human":
            messages.append({"role": "user", "content": content})
        elif who in ("gpt", "assistant"):
            messages.append({"role": "assistant", "content": content})
        elif who == "system":
            messages.append({"role": "system", "content": content})
    return _thread(messages)


def map_saiga(row: Row) -> Optional[dict]:
    """Scored Russian dialogues; ``bot`` is the assistant role in this dump."""
    if int(row.get("opus_score") or 0) < 6:
        return None
    if row.get("is_bad_by_regex"):
        return None
    lang = str(row.get("language") or "").lower()
    if lang and not lang.startswith("rus"):
        return None
    aliases = {"bot": "assistant", "gpt": "assistant", "human": "user"}
    messages: List[dict] = []
    for turn in row.get("messages") or []:
        role = aliases.get(str(turn.get("role") or ""), str(turn.get("role") or ""))
        content = clean(turn.get("content"))
        if role in ("user", "assistant", "system") and content:
            messages.append({"role": role, "content": content})
    return _thread(messages)


def map_nelvora_math(row: Row) -> Optional[dict]:
    messages: List[dict] = []
    for turn in row.get("messages") or []:
        role = str(turn.get("role") or "")
        content = clean(turn.get("content"))
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return _thread(messages)


def _thread(messages: List[dict]) -> Optional[dict]:
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    return {"messages": messages}


def map_stackexchange(row: Row) -> Optional[dict]:
    answer = best_answer(row.get("answers") or [], min_score=2)
    if not answer:
        return None
    return pair(html_to_text(str(row.get("question") or "")),
                html_to_text(answer))


def map_eli5(row: Row) -> Optional[dict]:
    question = clean(row.get("title"))
    selftext = clean(row.get("selftext"))
    if selftext:
        question = f"{question}\n\n{selftext}"
    answers = row.get("answers") or {}
    texts = answers.get("text") or []
    scores = answers.get("score") or []
    if not texts:
        return None
    best = max(range(len(texts)),
               key=lambda i: int(scores[i]) if i < len(scores) else 0)
    return pair(question, texts[best])


def map_reddit_qa(row: Row) -> Optional[dict]:
    return pair(row.get("prompt"), row.get("response"))


def map_russian_code_qa(row: Row) -> Optional[dict]:
    code = clean(row.get("code"))
    if not code:
        return None
    if not code.lstrip().startswith("```"):
        code = "```python\n" + code + "\n```"
    return pair(row.get("question"), code)


def map_squad(row: Row) -> Optional[dict]:
    return pair(row.get("question"), first_text(row.get("answers")))


def map_hotpot(row: Row) -> Optional[dict]:
    return pair(row.get("question"), row.get("answer"))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One Hub dataset (or config/split) and how to read it."""

    name: str
    hf: str
    mapper: Callable[[Row], Optional[dict]]
    licence: str
    lang: str
    domain: str
    task: str
    config: str = "default"
    split: str = "train"
    scan_limit: int = 0      # rows to look at (0 = until byte_cap is reached)
    byte_cap: int = 0        # bytes of kept text (0 = unbounded)
    note: str = ""
    local: bool = False      # written by the builder, not downloaded
    cap_share: float = 0.0   # share of the cell budget (0 = even split)
    min_answer: int = 0      # overrides the domain default (chars)
    extra: Dict[str, object] = field(default_factory=dict)


MATH_EN = [
    Source("openmathinstruct2", "nvidia/OpenMathInstruct-2",
           map_openmathinstruct, "CC-BY-4.0", "en", "math", "math_cot",
           split="train_1M",
           note="1M-row slice of the 22M-row set; CoT + expected answer"),
    Source("metamathqa", "meta-math/MetaMathQA", map_metamath,
           "MIT", "en", "math", "math_cot"),
    Source("orca_math", "microsoft/orca-math-word-problems-200k",
           map_orcamath, "MIT", "en", "math", "math_word"),
    Source("mathinstruct", "TIGER-Lab/MathInstruct", map_mathinstruct,
           "MIT", "en", "math", "math_cot"),
    Source("gsm8k", "openai/gsm8k", map_gsm8k, "MIT", "en", "math", "math_word",
           config="main"),
    Source("hendrycks_math_algebra", "EleutherAI/hendrycks_math",
           map_hendrycks, "MIT", "en", "math", "math_cot", config="algebra"),
    Source("hendrycks_math_geometry", "EleutherAI/hendrycks_math",
           map_hendrycks, "MIT", "en", "math", "math_cot", config="geometry"),
    Source("hendrycks_math_numbertheory", "EleutherAI/hendrycks_math",
           map_hendrycks, "MIT", "en", "math", "math_cot",
           config="number_theory"),
]

MATH_RU = [
    Source("orca_math_ru", "d0rj/orca-math-word-problems-200k-ru",
           map_orcamath, "MIT", "ru", "math", "math_word",
           note="translated Orca-Math; answer format matches the EN set"),
    Source("mathinstruct_ru", "d0rj/MathInstruct-ru", map_mathinstruct,
           "MIT", "ru", "math", "math_cot"),
    Source("nelvora_math_ru", "NelvoraAI/russian-math-dialogues",
           map_nelvora_math, "unknown", "ru", "math", "math_word"),
    Source("alpaca_math_ru", "ai-bond/ru-alpaca-math", map_alpaca,
           "unknown", "ru", "math", "math_word"),
]

QA_EN = [
    Source("openorca", "Open-Orca/OpenOrca", map_openorca,
           "MIT", "en", "qa", "qa_open"),
    Source("openhermes", "teknium/OpenHermes-2.5", map_openhermes,
           "Apache-2.0", "en", "qa", "qa_open"),
    Source("squad_v2", "rajpurkar/squad_v2", map_squad,
           "CC-BY-SA-4.0", "en", "qa", "qa_extractive", config="squad_v2"),
    Source("hotpotqa", "hotpotqa/hotpot_qa", map_hotpot,
           "CC-BY-SA-4.0", "en", "qa", "qa_extractive", config="distractor"),
]

QA_RU = [
    Source("saiga_scored", "IlyaGusev/saiga_scored", map_saiga,
           "Apache-2.0", "ru", "qa", "qa_open",
           note="41.6k scored dialogues; the builder keeps opus_score >= 6"),
    Source("russian_instructions2", "Den4ikAI/russian_instructions_2",
           map_q_a, "unknown", "ru", "qa", "qa_open"),
]

QA_SITES_EN = [
    Source("stackexchange", "HuggingFaceH4/stack-exchange-preferences",
           map_stackexchange, "CC-BY-SA-4.0", "en", "qa_sites", "qa_forum",
           note="Stack Exchange dump incl. Stack Overflow; best answer per thread"),
    Source("eli5", "Pavithree/eli5", map_eli5,
           "CC-BY-4.0", "en", "qa_sites", "qa_forum"),
    Source("reddit_qa", "zhoudoe23/reddit-top-qa-dataset", map_reddit_qa,
           "unknown", "en", "qa_sites", "qa_forum"),
]

QA_SITES_RU = [
    Source("mailru_big", "Den4ikAI/mailruQA-big", map_q_a,
           "unknown", "ru", "qa_sites", "qa_forum", min_answer=32,
           note="otvet.mail.ru mirror; short conversational answers by design"),
    Source("mailru_old", "Den4ikAI/mailru-QA-old", map_q_a,
           "unknown", "ru", "qa_sites", "qa_forum", min_answer=32,
           note="older and larger mail.ru dump; same short-answer profile"),
]

CODE = [
    Source("russian_code_qa", "Den4ikAI/russian_code_qa", map_russian_code_qa,
           "unknown", "ru", "code", "code_snippet"),
]

GENERAL = [
    Source("dolly", "databricks/databricks-dolly-15k", map_dolly,
           "CC-BY-SA-3.0", "en", "general", "qa_open"),
]

# The identity slice is written by the builder from ``data/identity`` (CC0
# paragraphs generated locally), not downloaded, so its mapper is the identity.
IDENTITY = [
    Source("identity_en", "local:identity", lambda row: row, "CC0-1.0",
           "en", "identity", "identity", local=True),
    Source("identity_ru", "local:identity", lambda row: row, "CC0-1.0",
           "ru", "identity", "identity", local=True),
]

# Arithmetic is generated locally by ``scripts/gen_arithmetic_sft.py``: exact
# answers, CC0, and it fills whatever the real math sources leave unfilled.  It
# is listed after the real math sources so that chain-of-thought data is taken
# first and arithmetic tops the cell up.
ARITH = [
    Source("arith_en", "synthetic:arith", lambda row: row, "CC0-1.0",
           "en", "math", "math_arith", local=True, cap_share=0.40),
    Source("arith_ru", "synthetic:arith", lambda row: row, "CC0-1.0",
           "ru", "math", "math_arith", local=True, cap_share=0.40),
]

SOURCES: List[Source] = (MATH_EN + MATH_RU + QA_EN + QA_RU
                         + QA_SITES_EN + QA_SITES_RU + CODE + GENERAL
                         + ARITH + IDENTITY)

# Domain shares of the SFT corpus by bytes.  Mathematics is the point of this
# stage, so it takes the largest slice; forum Q/A is capped because the answers
# are short and conversational.
DOMAIN_SHARE = {
    "math": 0.45,
    "qa": 0.30,
    "qa_sites": 0.15,
    "code": 0.06,
    "general": 0.035,
    "identity": 0.005,
}

# Same contract as pretraining: 60 % English / 40 % Russian by bytes.
LANG_SHARE = {"en": 0.60, "ru": 0.40}
