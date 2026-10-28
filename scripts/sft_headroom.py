"""How much more English is actually in the shards, after the 512-token filter?

The SFT build came out at English 51.2 / Russian 48.8 by bytes against a 60/40
contract, and the report said English cells "could not fill their budgets".  That
claim needs checking, because two different things can make a cell short:

* the shard really ends, or
* the per-source cap is reached — and ``fill_cell`` counts a record against the
  cap *before* the length filter removes it, so a source full of long
  chain-of-thought burns its allowance without contributing usable bytes.

This script answers the question from the data: for every source of a cell it
streams the whole shard, renders and filters exactly as the build does, and
reports the usable bytes available, plus how far the current cap would get if it
counted only usable bytes.

Usage::

    python scripts/sft_headroom.py --cells code_en,qa_en --out data/sft_headroom.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_sft import (MAX_TOKENS, group_sources, iter_raw,  # noqa: E402
                       raw_path, record_bytes, render, done_path)
from sft_registry import DOMAIN_SHARE, LANG_SHARE  # noqa: E402

MAX_TOKENS = MAX_TOKENS


def load_tok():
    from kronllm.tokenizer import BPETokenizer
    return BPETokenizer("data/sft/tokenizer.json", 8192, 2.9, 8191)


def scan_cell(domain: str, lang: str, raw: Path, tok, budget: int,
              cap_bytes: int) -> dict:
    sources = group_sources(domain, lang)
    rows = []
    for src in sources:
        path = raw_path(raw, src.name)
        if not path.exists() or not done_path(raw, src.name).exists():
            rows.append({"source": src.name, "missing": True})
            continue
        total_rows = total_bytes = kept_rows = kept_bytes = 0
        cap_rows = None
        cap_rows_bytes = 0
        for record in iter_raw(path):
            total_rows += 1
            total_bytes += record_bytes(record)
            full, _ = render(record["messages"])
            if len(tok.encode(full)) + 1 > MAX_TOKENS:
                continue
            kept_rows += 1
            size = record_bytes(record)
            kept_bytes += size
            if cap_rows is None:
                cap_rows_bytes += size
                if cap_rows_bytes >= cap_bytes:
                    cap_rows = kept_rows
        rows.append({
            "source": src.name,
            "rows": total_rows,
            "MB": round(total_bytes / 1e6, 1),
            "kept_rows": kept_rows,
            "kept_MB": round(kept_bytes / 1e6, 1),
            "survival": round(kept_rows / max(1, total_rows), 3),
            "rows_to_fill_cap": cap_rows,
            "cap_MB": round(cap_bytes / 1e6, 1),
        })
        print(f"  {src.name:20s} {total_rows:>8,} rows {total_bytes/1e6:>7.1f} MB"
              f" -> {kept_rows:>8,} rows {kept_bytes/1e6:>7.1f} MB "
              f"({kept_rows/max(1,total_rows):.0%})", flush=True)
    usable = sum(r.get("kept_MB", 0) for r in rows)
    capped = sum(min(r.get("kept_MB", 0), r.get("cap_MB", 0)) for r in rows)
    return {"domain": domain, "lang": lang, "budget_MB": round(budget / 1e6, 1),
            "sources": rows, "usable_MB": round(usable, 1),
            "with_cap_MB": round(capped, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="code_en,qa_en")
    ap.add_argument("--raw", default="E:/kron_sft/raw")
    ap.add_argument("--target-bytes", type=int, default=1_450_000_000)
    ap.add_argument("--out", default="data/sft_headroom.json")
    args = ap.parse_args()

    raw = Path(args.raw)
    tok = load_tok()
    out = {}
    for name in args.cells.split(","):
        domain, _, lang = name.rpartition("_")
        budget = int(args.target_bytes * DOMAIN_SHARE[domain] * LANG_SHARE[lang])
        n_sources = max(1, len(group_sources(domain, lang)))
        cap = int(budget * 1.8 / n_sources)
        print(f"[{name}] budget {budget/1e6:.1f} MB, per-source cap "
              f"{cap/1e6:.1f} MB over {n_sources} sources", flush=True)
        out[name] = scan_cell(domain, lang, raw, tok, budget, cap)
        r = out[name]
        print(f"[{name}] usable {r['usable_MB']:.1f} MB; with the current cap "
              f"accounting at most {r['with_cap_MB']:.1f} MB", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
