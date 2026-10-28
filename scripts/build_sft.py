"""Build the supervised-fine-tuning corpus: fetch sources, mix, tokenise, pack.

Three stages, each rerunnable on its own:

``fetch``
    Stream every source in :mod:`scripts.sft_registry` from the Hub, normalise
    each row into a ``messages`` record, filter it, deduplicate it, and write
    ``<raw>/<source>.jsonl.gz``.  Resumable: a source with a matching
    ``.done.json`` marker is skipped, so a killed download does not restart the
    others.

``mix``
    Fill the byte budget of every (domain, language) cell from the raw shards,
    render the chat text, tokenise it in batches, and write one part per cell.
    The parts are then interleaved into ``train.bpe.bin`` / ``val.bpe.bin``.

``report``
    Write ``data/report_sft.md`` from ``meta.json`` and the per-source markers.

The output is the flat uint16 layout the pretraining loader already reads (an
EOT id after every example), plus ``sft_index.npy`` with three columns per
example -- ``start``, ``prompt_tokens``, ``total_tokens`` -- so a trainer can
mask the loss on the prompt without re-tokenising anything.

Usage::

    python scripts/build_sft.py plan
    python scripts/build_sft.py fetch --raw E:/kron_sft/raw
    python scripts/build_sft.py mix --raw E:/kron_sft/raw --target-tokens 500000000
    python scripts/build_sft.py report
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sft_registry import (DOMAIN_SHARE, LANG_SHARE, SOURCES,  # noqa: E402
                          Source, clean)

SYSTEM = "You are KronLLM, a compact bilingual model."
EOT = 8191          # document separator (``vocab_size - 1``)
ASSISTANT = "<|assistant|>"

# Filtering thresholds, in characters.  The minimum answer length is what
# separates an answer from a comment: below twenty characters almost every
# forum reply is "yes" or a link.
MIN_Q, MAX_Q = 12, 4000
MIN_A, MAX_A = 20, 8000
MIN_A_FORUM = 120
LOOP_RATIO = 0.30
_CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
_LATIN = re.compile(r"[a-zA-Z]")
_REFUSAL = re.compile(
    r"(?i)^\s*(i'm sorry|sorry,|i cannot|i can't help|i am unable|as an ai"
    r"|не могу помочь|извините, но я не могу|как ии\b)")
_LINK_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$")


# --------------------------------------------------------------------------
# rendering and filters
# --------------------------------------------------------------------------


def render(messages: List[dict]) -> Tuple[str, str]:
    """Render a thread into chat text and return ``(full_text, prompt_text)``.

    The prompt is everything the model sees before it starts answering: the
    system turn plus every user turn up to the last assistant marker.  It is
    returned separately because the trainer masks the loss on it.
    """
    turns = list(messages)
    system = SYSTEM
    if turns and turns[0]["role"] == "system":
        system = turns[0]["content"]
        turns = turns[1:]
    parts = [f"<|system|>{system}<|eot|>"]
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if role == "assistant":
            parts.append(f"{ASSISTANT}{content}<|eot|>")
        else:
            parts.append(f"<|user|>{content}<|eot|>")
    full = "".join(parts)
    cut = full.rfind(ASSISTANT)
    prompt = full[:cut + len(ASSISTANT)] if cut >= 0 else full
    return full, prompt


def language_ok(text: str, lang: str) -> bool:
    """Cheap script check: no language-ID model is bundled, and this is enough.

    Russian sources occasionally carry English rows and vice versa; a share of
    Cyrillic characters settles it without another dependency.  Mixed text (code
    with Russian comments) stays, because both scripts are expected there.
    """
    cyr = len(_CYRILLIC.findall(text))
    lat = len(_LATIN.findall(text))
    letters = cyr + lat
    if letters < 8:
        return False
    if lang == "ru":
        return cyr / letters >= 0.25
    return cyr / letters <= 0.02


def looped(text: str) -> bool:
    """Reject text where one 5-gram keeps coming back (the small-model failure)."""
    words = text.split()
    if len(words) < 20:
        return False
    grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
    return len(set(grams)) / len(grams) < (1.0 - LOOP_RATIO)


def reject_reason(messages: List[dict], lang: str, domain: str,
                  min_answer: int = 0) -> Optional[str]:
    """Return why a record may not enter the corpus, or ``None``."""
    question = "\n".join(m["content"] for m in messages if m["role"] == "user")
    answer = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    if not MIN_Q <= len(question) <= MAX_Q:
        return "question_length"
    floor = min_answer or (MIN_A_FORUM if domain == "qa_sites" else MIN_A)
    if not floor <= len(answer) <= MAX_A:
        return "answer_length"
    if _REFUSAL.search(answer) or _LINK_ONLY.match(answer):
        return "refusal_or_link"
    if not language_ok(question + " " + answer, lang):
        return "language"
    if looped(answer):
        return "loop"
    return None


def fingerprint(messages: List[dict]) -> int:
    text = "\x00".join(m["content"] for m in messages)
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"),
                                          digest_size=8).digest(), "big")


def record_bytes(record: dict) -> int:
    return sum(len(m["content"].encode("utf-8")) for m in record["messages"])


# --------------------------------------------------------------------------
# stage 1: fetch
# --------------------------------------------------------------------------


def raw_path(raw: Path, name: str) -> Path:
    return raw / f"{name}.jsonl.gz"


def done_path(raw: Path, name: str) -> Path:
    return raw / f"{name}.done.json"


def stream_source(src: Source) -> Iterator[dict]:
    """Yield raw rows from one Hub dataset, or nothing if it is unreachable.

    ``datasets`` is imported lazily: ``plan`` and ``report`` must work without
    touching the network, and a gated dataset should fail one source, not the
    build.
    """
    from datasets import load_dataset

    try:
        stream = load_dataset(src.hf, src.config, split=src.split,
                              streaming=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {src.name}: {type(exc).__name__}: {exc}", flush=True)
        return
    for row in stream:
        yield row


def fetch_one(src: Source, raw: Path, byte_cap: int, scan_limit: int,
              seed: int) -> dict:
    """Stream one source into a shard, stopping at the byte cap or the limit."""
    stats = {"source": src.name, "hf": src.hf, "licence": src.licence,
             "lang": src.lang, "domain": src.domain, "task": src.task,
             "rows_seen": 0, "rows_kept": 0, "bytes": 0,
             "rejected": {}, "note": src.note}
    out = raw_path(raw, src.name)
    tmp = out.with_suffix(".tmp.gz")
    rng = random.Random(seed)
    buffer: List[dict] = []
    seen: set[int] = set()
    # A source whose rows are mostly filtered out would otherwise be scanned to
    # the end of a 26M-row dump; three million rows is enough to tell whether
    # the cap can be reached.
    scan_limit = scan_limit or 3_000_000
    t0 = time.time()
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as fh:
        for row in stream_source(src):
            stats["rows_seen"] += 1
            if stats["rows_seen"] % 50000 == 0:
                rate = stats["rows_seen"] / max(time.time() - t0, 1e-9)
                print(f"  [{src.name}] seen {stats['rows_seen']:,} "
                      f"kept {stats['rows_kept']:,} "
                      f"({stats['bytes'] / 1e6:.1f} MB) {rate:.0f} rows/s",
                      flush=True)
            try:
                record = src.mapper(row)
            except Exception as exc:  # noqa: BLE001
                key = f"mapper:{type(exc).__name__}"
                stats["rejected"][key] = stats["rejected"].get(key, 0) + 1
                continue
            if record is None:
                stats["rejected"]["unmapped"] = \
                    stats["rejected"].get("unmapped", 0) + 1
                continue
            record.setdefault("lang", src.lang)
            record.setdefault("domain", src.domain)
            record.setdefault("task", src.task)
            reason = reject_reason(record["messages"], src.lang, src.domain,
                                   src.min_answer)
            if reason:
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                continue
            key = fingerprint(record["messages"])
            if key in seen:
                stats["rejected"]["duplicate"] = \
                    stats["rejected"].get("duplicate", 0) + 1
                continue
            seen.add(key)
            size = record_bytes(record)
            record["source"] = src.name
            record["licence"] = src.licence
            record["bytes"] = size
            buffer.append(record)
            stats["rows_kept"] += 1
            stats["bytes"] += size
            if len(buffer) >= 5000:
                rng.shuffle(buffer)
                for item in buffer:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                buffer.clear()
            if stats["bytes"] >= byte_cap or (
                    scan_limit and stats["rows_seen"] >= scan_limit):
                break
        rng.shuffle(buffer)
        for item in buffer:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(out)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["cap_bytes"] = byte_cap
    done_path(raw, src.name).write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  [done] {src.name}: {stats['rows_kept']:,} records, "
          f"{stats['bytes'] / 1e6:.1f} MB in {stats['seconds']}s", flush=True)
    return stats


def group_sources(domain: str, lang: str) -> List[Source]:
    return [s for s in SOURCES if s.domain == domain and s.lang == lang]


def fetch_plan(total_bytes: int, scale: float) -> List[Tuple[Source, int]]:
    """Split every (domain, language) budget across its sources with headroom.

    A source may use 1.8x its nominal share: if one turns out to be gated or
    thin, its neighbours close the gap without a second download pass.
    """
    plan: List[Tuple[Source, int]] = []
    for domain, dshare in DOMAIN_SHARE.items():
        for lang, lshare in LANG_SHARE.items():
            group = [s for s in group_sources(domain, lang) if not s.local]
            if not group:
                continue
            budget = total_bytes * dshare * lshare * scale
            for src in group:
                cap = src.byte_cap or int(budget * 1.8 / len(group))
                plan.append((src, max(cap, 2_000_000)))
    return plan


def cmd_fetch(args: argparse.Namespace) -> None:
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    plan = fetch_plan(args.target_bytes, args.scale)
    only = set(args.sources or [])
    if only:
        plan = [(s, c) for s, c in plan if s.name in only]
    print(f"[fetch] {len(plan)} sources -> {raw}", flush=True)
    total = 0
    for src, cap in plan:
        marker = done_path(raw, src.name)
        if marker.exists() and not args.force:
            stats = json.loads(marker.read_text(encoding="utf-8"))
            print(f"  [cached] {src.name}: {stats['rows_kept']:,} records")
            total += stats["bytes"]
            continue
        print(f"  [fetch] {src.name} <- {src.hf}/{src.config}/{src.split} "
              f"(cap {cap / 1e6:.1f} MB)", flush=True)
        stats = fetch_one(src, raw, cap, src.scan_limit, args.seed)
        total += stats["bytes"]
    print(f"[fetch] kept {total / 1e6:.1f} MB in {raw}")


# --------------------------------------------------------------------------
# stage 2: mix
# --------------------------------------------------------------------------


def iter_raw(path: Path, skip: int = 0) -> Iterator[dict]:
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < skip:
                continue
            line = line.strip()
            if line:
                yield json.loads(line)


class Cell:
    """One (domain, language) budget and what has been taken into it."""

    def __init__(self, domain: str, lang: str, budget: int) -> None:
        self.domain, self.lang, self.budget = domain, lang, budget
        self.bytes = 0
        self.records = 0
        self.consumed: Dict[str, int] = {}
        self.per_source: Dict[str, int] = {}

    @property
    def shortfall(self) -> int:
        return max(0, self.budget - self.bytes)


def fill_cell(cell: Cell, sources: List[Source], raw: Path) -> Iterator[dict]:
    """Yield records from ``sources`` until the cell budget is met.

    Each source is capped at ``cap_share`` of the cell budget (or 1.8x an even
    split when it declares none), so a single large dataset cannot swallow the
    whole cell while a source that was fetched generously can still fill it.
    """
    share = cell.budget * 1.8 / max(1, len(sources))
    for src in sources:
        if cell.shortfall <= 0:
            return
        path = raw_path(raw, src.name)
        if not path.exists():
            continue
        cap = int(src.byte_cap or (src.cap_share * cell.budget if src.cap_share
                                   else share))
        taken = 0
        skip = cell.consumed.get(src.name, 0)
        for record in iter_raw(path, skip):
            if taken >= cap:
                break
            if cell.shortfall <= 0:
                return
            size = record_bytes(record)
            cell.records += 1
            cell.bytes += size
            taken += size
            cell.consumed[src.name] = cell.consumed.get(src.name, 0) + 1
            cell.per_source[src.name] = cell.per_source.get(src.name, 0) + 1
            yield record


def write_parts(raw: Path, out: Path, target_bytes: int) -> Dict[str, dict]:
    """Fill every cell and write one rendered-text part per cell.

    A cell with no licensed source (English code, Russian general) would leave
    its budget unused, so its bytes are moved into the same-language math cell
    and the move is recorded.  Mathematics is where the user asked the emphasis
    to be, and arithmetic generation can always absorb the extra.
    """
    parts_dir = out / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    # A rerun must not leave a part from a previous, larger build behind: the
    # mixer would tokenise it again and the mix would no longer match the plan.
    for stale in parts_dir.glob("*.jsonl.gz"):
        stale.unlink()
    write_identity(raw, target_bytes * DOMAIN_SHARE["identity"])

    budgets: Dict[Tuple[str, str], int] = {}
    for domain, dshare in DOMAIN_SHARE.items():
        for lang, lshare in LANG_SHARE.items():
            budgets[(domain, lang)] = int(target_bytes * dshare * lshare)

    absorbed: Dict[str, str] = {}
    for (domain, lang), budget in list(budgets.items()):
        if group_sources(domain, lang):
            continue
        host = ("math", lang) if group_sources("math", lang) else None
        if host is None:
            continue
        budgets[host] += budget
        budgets[(domain, lang)] = 0
        absorbed[f"{domain}_{lang}"] = f"{host[0]}_{host[1]}"

    cells = {key: Cell(key[0], key[1], budget)
             for key, budget in budgets.items() if budget > 0}

    stats: Dict[str, dict] = {}
    for (domain, lang), cell in sorted(cells.items()):
        name = f"{domain}_{lang}"
        stats[name] = {"domain": domain, "lang": lang, "records": 0,
                       "bytes": 0, "per_source": {}, "shortfall": 0,
                       "absorbed": {k: v for k, v in absorbed.items()
                                    if v == name}}
        started = time.time()
        sink = parts_dir / f"{name}.jsonl.gz"
        with gzip.open(sink, "wt", encoding="utf-8", compresslevel=3) as fh:
            for record in fill_cell(cell, group_sources(domain, lang), raw):
                full, _ = render(record["messages"])
                fh.write(json.dumps({"text": full}, ensure_ascii=False) + "\n")
        stats[name].update(bytes=cell.bytes, records=cell.records,
                           per_source=cell.per_source,
                           shortfall=cell.shortfall)
        print(f"  [part] {name}: {cell.records:,} records, "
              f"{cell.bytes / 1e6:.1f} MB of {cell.budget / 1e6:.1f} MB "
              f"({time.time() - started:.0f}s)", flush=True)
    for name, where in absorbed.items():
        print(f"  [absorb] {name} -> {where} (no licensed source)")
    return stats


# --------------------------------------------------------------------------
# identity slice
# --------------------------------------------------------------------------

IDENTITY_QUESTIONS_EN = [
    "Who are you?", "What is your name?", "What model are you?",
    "Tell me about yourself.", "Which language model am I talking to?",
]
IDENTITY_QUESTIONS_RU = [
    "Кто ты?", "Как тебя зовут?", "Что ты за модель?",
    "Расскажи о себе.", "С какой языковой моделью я разговариваю?",
]


def write_identity(raw: Path, budget_bytes: float) -> None:
    """Turn the generated identity paragraphs into Q/A pairs.

    The paragraphs under ``data/identity`` are answers looking for a question.
    Five fixed phrasings per language per paragraph give the model a stable
    identity without teaching it to volunteer the name on unrelated prompts.
    """
    files = [("en", ROOT / "data/identity/identity_en.jsonl",
              IDENTITY_QUESTIONS_EN),
             ("ru", ROOT / "data/identity/identity_ru.jsonl",
              IDENTITY_QUESTIONS_RU)]
    rows: Dict[str, List[dict]] = {"en": [], "ru": []}
    for lang, path, questions in files:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                doc = json.loads(line)
                text = clean(doc.get("text"))
                if len(text) < 80:
                    continue
                question = questions[i % len(questions)]
                rows[lang].append({"messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": text}],
                    "lang": lang, "domain": "identity", "task": "identity",
                    "source": f"identity_{lang}", "licence": "CC0-1.0",
                    "bytes": len(question) + len(text)})
    limit = max(budget_bytes / 2, 1e5)
    for lang, items in rows.items():
        used, kept = 0, []
        for row in items:
            if used >= limit:
                break
            kept.append(row)
            used += row["bytes"]
        out = raw_path(raw, f"identity_{lang}")
        with gzip.open(out, "wt", encoding="utf-8", compresslevel=3) as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        done_path(raw, f"identity_{lang}").write_text(json.dumps(
            {"source": f"identity_{lang}", "rows_kept": len(kept),
             "bytes": used, "lang": lang, "domain": "identity",
             "licence": "CC0-1.0",
             "hf": "local:data/identity"}, indent=1),
            encoding="utf-8")
        print(f"  [identity] {lang}: {len(kept)} pairs, {used / 1e6:.2f} MB")


# --------------------------------------------------------------------------
# stage 3: tokenise and pack
# --------------------------------------------------------------------------


def write_part_tokens(part: Path, dest: Path, tok,
                      index: List[Tuple[int, int, int]],
                      start_offset: int) -> Tuple[int, int]:
    """Tokenise one rendered-text part into ``dest`` and extend ``index``."""
    assistant_id = tok._tok.token_to_id(ASSISTANT)
    n0 = len(index)
    written = 0
    nbytes = 0
    batch: List[str] = []

    def flush(fh) -> None:
        nonlocal written, nbytes
        if not batch:
            return
        nbytes += sum(len(t.encode("utf-8")) for t in batch)
        for seq in tok.encode_batch(batch):
            n = len(seq)
            cut = -1
            if assistant_id is not None:
                for i in range(n - 1, -1, -1):
                    if seq[i] == assistant_id:
                        cut = i
                        break
            prompt_tokens = cut + 1 if cut >= 0 else n
            array = np.empty(n + 1, dtype=np.uint16)
            array[:n] = seq
            array[n] = EOT
            fh.write(array.tobytes())
            index.append((start_offset + written, prompt_tokens, n + 1))
            written += n + 1
        batch.clear()

    with dest.open("wb") as fh:
        with gzip.open(part, "rt", encoding="utf-8") as src:
            for line in src:
                batch.append(json.loads(line)["text"])
                if len(batch) >= 2000:
                    flush(fh)
        flush(fh)
    return written, len(index) - n0, nbytes


def interleave(sizes, chunk: int = 4096):
    """Split every part into ``chunk``-token blocks, in part order.

    Parts are packed **sequentially**.  The recorded index offsets have to match
    the final file, and the trainer samples random blocks through a per-epoch
    permutation anyway, so block-level round-robin would buy nothing and would
    cost an offset remap -- exactly the bug the first build had: the index
    described the parts, the file was interleaved, and every example decoded
    across a boundary.
    """
    out = []
    for name, tokens in sizes:
        for s in range(0, tokens, chunk):
            out.append((name, s, min(s + chunk, tokens)))
    return out


def tokenize_parts(out: Path, tok, val_every: int = 50) -> dict:
    """Tokenise every part, then pack them into train/val bins."""
    parts_dir = out / "parts"
    part_files = sorted(parts_dir.glob("*.jsonl.gz"))
    if not part_files:
        raise SystemExit("no parts in %s; run `mix` first" % parts_dir)

    index = []
    offsets = []
    total = 0
    for path in part_files:
        name = path.name.split(".jsonl.gz")[0]
        dest = parts_dir / f"{name}.u16"
        t0 = time.time()
        tokens, examples, nbytes = write_part_tokens(path, dest, tok, index,
                                                      total)
        if tokens == 0:
            print(f"  [tok] {name}: empty, skipped", flush=True)
            dest.unlink(missing_ok=True)
            continue
        offsets.append({"name": name, "tokens": tokens, "examples": examples,
                        "start": total, "bytes": nbytes})
        total += tokens
        print(f"  [tok] {name}: {examples:,} examples, {tokens:,} tokens "
              f"({time.time() - t0:.0f}s)", flush=True)

    bounds = {}
    for info in offsets:
        lo, hi = info["start"], info["start"] + info["tokens"]
        bounds[info["name"]] = [tuple(int(v) for v in row)
                                for row in index if lo <= row[0] < hi]
    maps = {o["name"]: np.memmap(parts_dir / f"{o['name']}.u16",
                                 dtype=np.uint16, mode="r") for o in offsets}
    train, val = out / "train.bpe.bin", out / "val.bpe.bin"
    n_train = n_val = 0
    block = 0
    packed = []      # (start, prompt, total, file) with positions in the final files
    with train.open("wb") as ftr, val.open("wb") as fva:
        for info in offsets:
            rows, base = bounds[info["name"]], info["start"]
            i = 0
            while i < len(rows):
                # Group whole examples into ~4096-token blocks so an example is
                # never split across the train/val boundary.
                j, acc = i, 0
                while j < len(rows) and acc < 4096:
                    acc += rows[j][2]
                    j += 1
                lo = rows[i][0] - base
                hi = rows[j - 1][0] + rows[j - 1][2] - base
                chunk = maps[info["name"]][lo:hi]
                to_val = block % val_every == val_every - 1
                if to_val:
                    fva.write(chunk.tobytes())
                else:
                    ftr.write(chunk.tobytes())
                # The index must describe the *written* file, not the unsplit
                # concatenation: held-out examples are removed from train, so
                # every later offset shifts by however much went to val.
                dst = n_val if to_val else n_train
                file_id = 1 if to_val else 0
                for row in rows[i:j]:
                    packed.append((dst + row[0] - rows[i][0], row[1], row[2],
                                   file_id))
                if to_val:
                    n_val += len(chunk)
                else:
                    n_train += len(chunk)
                i = j
                block += 1

    idx = np.asarray(packed, dtype=np.int64)
    np.save(out / "sft_index.npy", idx)
    total_bytes = sum(o["bytes"] for o in offsets)
    meta = {"vocab_size": 8192, "eot_id": EOT, "system": SYSTEM,
            "parts": offsets, "examples": int(idx.shape[0]),
            "train_tokens": n_train, "val_tokens": n_val,
            "total_tokens": n_train + n_val, "bytes": total_bytes,
            "bytes_per_token": total_bytes / max(1, n_train + n_val),
            "layout": "sequential parts, examples never split across files",
            "index_columns": ["start", "prompt_tokens", "total_tokens", "file"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1),
                                   encoding="utf-8")
    # ``kronllm.data.open_dataset`` reads the tokenizer and its meta from the
    # data directory, so an SFT run needs both next to the bins.
    tok_src = ROOT / "data" / "bilingual" / "tokenizer.json"
    if tok_src.exists():
        (out / "tokenizer.json").write_bytes(tok_src.read_bytes())
    (out / "tokenizer_meta.json").write_text(json.dumps({
        "vocab_size": 8192, "eot_id": EOT,
        "bytes_per_token": total_bytes / max(1, n_train + n_val),
        "train_tokens": n_train, "val_tokens": n_val,
        "tokenizer": str(out / "tokenizer.json"),
        "source": "sft mix built by scripts/build_sft.py",
    }, indent=1), encoding="utf-8")
    print(f"  [bin] train {n_train:,} tokens, val {n_val:,} tokens, "
          f"{idx.shape[0]:,} examples")
    return meta


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    raw = Path(args.raw)
    stats = [json.loads(m.read_text(encoding="utf-8"))
             for m in sorted(raw.glob("*.done.json"))]
    tokens = meta["total_tokens"]
    parts_stats = {}
    ps = out / "parts_stats.json"
    if ps.exists():
        parts_stats = json.loads(ps.read_text(encoding="utf-8"))

    lines = [
        "# SFT corpus",
        "",
        f"**{tokens:,} tokens** ({meta['train_tokens']:,} train / "
        f"{meta['val_tokens']:,} val), {meta['examples']:,} examples.",
        "",
        "Template: `<|system|>…<|eot|><|user|>…<|eot|>"
        "<|assistant|>…<|eot|>`; `sft_index.npy` stores "
        "`(start, prompt_tokens, total_tokens)` so the loss can be masked on "
        "the prompt.",
        "",
        "## Cells: target vs achieved",
        "",
        "| cell | records | MB | tokens | share | shortfall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    per_lang: Dict[str, int] = {}
    per_domain: Dict[str, int] = {}
    for part in meta["parts"]:
        domain, _, lang = part["name"].rpartition("_")
        if not domain:
            domain, lang = part["name"], "?"
        per_lang[lang] = per_lang.get(lang, 0) + part["tokens"]
        per_domain[domain] = per_domain.get(domain, 0) + part["tokens"]
        cell = parts_stats.get(part["name"], {})
        lines.append(
            f"| {part['name']} | {cell.get('records', 0):,} | "
            f"{cell.get('bytes', 0) / 1e6:.1f} | {part['tokens']:,} | "
            f"{part['tokens'] / max(tokens, 1):.1%} | "
            f"{cell.get('shortfall', 0) / 1e6:.1f} MB |")
    lines += ["", "## Sources", "",
              "| source | language | domain | records | MB | licence |",
              "|---|---|---|---:|---:|---|"]
    for st in sorted(stats, key=lambda s: s["source"]):
        lines.append(f"| {st['source']} | {st.get('lang', '—')} | "
                     f"{st.get('domain', '—')} | {st.get('rows_kept', 0):,} | "
                     f"{st.get('bytes', 0) / 1e6:.1f} | "
                     f"{st.get('licence', '—')} |")
    lines += ["", "## Share by language and domain", "",
              "| language | tokens | share |", "|---|---:|---:|"]
    for lang, count in sorted(per_lang.items()):
        lines.append(f"| {lang} | {count:,} | {count / max(tokens, 1):.1%} |")
    lines += ["", "| domain | tokens | share |", "|---|---:|---:|"]
    for domain, count in sorted(per_domain.items()):
        lines.append(f"| {domain} | {count:,} | {count / max(tokens, 1):.1%} |")
    lines += ["", "## What was rejected", "",
              "| source | kept | rejected | top reasons |", "|---|---:|---:|---|"]
    for st in sorted(stats, key=lambda s: s["source"]):
        reasons = st.get("rejected") or {}
        top = ", ".join(f"{k} {v:,}" for k, v in
                        sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
        lines.append(f"| {st['source']} | {st.get('rows_kept', 0):,} | "
                     f"{sum(reasons.values()):,} | {top or '—'} |")
    report = out.parent / "report_sft.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))
    print(f"[out] {report}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "fetch", "mix", "report", "all"])
    ap.add_argument("--raw", default="E:/kron_sft/raw")
    ap.add_argument("--out-dir", default=str(ROOT / "data/sft"))
    ap.add_argument("--target-tokens", type=int, default=500_000_000)
    ap.add_argument("--bytes-per-token", type=float, default=3.3)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink every budget; a smoke test of the pipeline")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    args.target_bytes = int(args.target_tokens * args.bytes_per_token)
    if args.command == "plan":
        print(f"target {args.target_tokens:,} tokens "
              f"~ {args.target_bytes / 1e6:.0f} MB")
        for domain, dshare in DOMAIN_SHARE.items():
            for lang, lshare in LANG_SHARE.items():
                group = group_sources(domain, lang)
                budget = args.target_bytes * dshare * lshare
                print(f"  {domain:<9} {lang} {budget / 1e6:7.1f} MB "
                      f"({len(group)} sources)")
        for src, cap in fetch_plan(args.target_bytes, args.scale):
            print(f"    {src.name:<28} {cap / 1e6:7.1f} MB  {src.hf}")
        return
    if args.command in ("fetch", "all"):
        cmd_fetch(args)
    if args.command in ("mix", "all"):
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        from kronllm.tokenizer import BPETokenizer
        tok = BPETokenizer(ROOT / "data/bilingual/tokenizer.json", 8192,
                           3.694, EOT)
        stats = write_parts(Path(args.raw), out, args.target_bytes)
        (out / "parts_stats.json").write_text(json.dumps(stats, indent=1),
                                              encoding="utf-8")
        tokenize_parts(out, tok)
    if args.command in ("report", "all"):
        cmd_report(args)


if __name__ == "__main__":
    main()
