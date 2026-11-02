"""Generate from a checkpoint and write the samples as markdown for a report.

Why a separate script from ``sample_q10.py``: that one prints to the terminal for
a human watching.  This one runs unattended at 01:00 and at 07:00, so it has to
be deterministic in structure (fixed prompt set, fixed decoding settings), write
a file, and never need a decision.

Two decodings per prompt, because they answer different questions:

* **greedy** -- what the model *thinks* is most likely; good for spotting a model
  that answers with the question, or that cannot close a sentence;
* **sampled** -- what the distribution looks like; this is what a user sees in the
  playground, and it is where repetition shows up first.

Prompts are chosen to cut across the corpus: prose in both languages, knowledge,
code, a column-arithmetic drill in the exact SFT format, and the chat template.

Usage::

    python scripts/generate_samples.py --ckpt "E:/kron_out/session6/last.pt" \
        --out data/samples_session6.md --label "сессия 6, шаг 43xxx"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kronllm.model import KronTransformer  # noqa: E402
from kronllm.tokenizer import BPETokenizer  # noqa: E402
from sample_q10 import stream  # noqa: E402

SYSTEM = ("<|system|>You are KronLLM, a compact bilingual model.<|eot|>"
          "<|user|>Who are you?<|eot|><|assistant|>")

PROMPTS = [
    ("EN, продолжение прозы",
     "The Roman Empire was one of the largest and longest-lasting empires in "
     "history, and"),
    ("RU, продолжение прозы",
     "Фотосинтез — это процесс, при котором растения используют энергию света, "
     "чтобы"),
    ("EN, код",
     "def add(a, b):\n    \"\"\"Return the sum of two numbers.\"\"\"\n    return"),
    ("EN, арифметика столбиком",
     "<|user|>Compute 47 + 38 and show the carries.<|eot|><|assistant|>"),
    ("RU, арифметика столбиком",
     "<|user|>Вычисли 800 - 385 и покажи заёмы.<|eot|><|assistant|>"),
    ("чат-формат (цель SFT)",
     "<|system|>You are KronLLM, a compact bilingual model.<|eot|>"
     "<|user|>Who are you?<|eot|><|assistant|>"),
]

GREEDY = dict(temperature=1e-5, top_k=1, top_p=0.0, repetition_penalty=1.1,
              rep_window=64)
# 40 candidates, not 8: the narrow cap was making the model worse than it is.
SAMPLED = dict(temperature=0.7, top_k=40, top_p=0.95, repetition_penalty=1.2,
               rep_window=64)


def run(model, tok, prompt: str, device, n: int, **kw) -> str:
    out = []
    for piece in stream(model, tok, prompt, device, max_new_tokens=n, **kw):
        out.append(piece)
    text = "".join(out).strip()
    return text or "(пусто: модель сразу выдала <|eot|>)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--tokens", type=int, default=35)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # A weights-only checkpoint (best.pt, an averaged soup) carries no config:
    # the geometry then comes from the experiment the run is defined by.
    if "cfg" in state:
        cfg = state["cfg"]
    else:
        from kronllm.config import experiment_config
        cfg = experiment_config("Q10")
    model = KronTransformer(cfg)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    step = int(state.get("step", 0))

    tok_path = ROOT / "data" / "bilingual" / "tokenizer.json"
    tok = BPETokenizer(tok_path, cfg["data"]["vocab_size"],
                       bytes_per_token=1.0, eot_id=cfg["data"]["eot_id"])

    def describe(d: dict) -> str:
        parts = [f"`T={d['temperature']:g}`", f"`top_k={d['top_k']}`"]
        if d.get("top_p"):
            parts.append(f"`top_p={d['top_p']}`")
        parts.append(f"штраф за повторы {d['repetition_penalty']:g}")
        return ", ".join(parts)

    md = [f"## Тестовая генерация{(' — ' + args.label) if args.label else ''}", "",
          f"Чекпойнт `{Path(args.ckpt).name}`, шаг **{step:,}**, "
          f"{sum(p.numel() for p in model.parameters()):,} параметров, "
          f"по {args.tokens} токенов. Жадное: {describe(GREEDY)}. "
          f"Сэмплирование: {describe(SAMPLED)}.", ""]

    t0 = time.time()
    for name, prompt in PROMPTS:
        shown = prompt.replace("\n", "\\n")
        if len(shown) > 90:
            shown = shown[:87] + "..."
        md.append(f"### {name}")
        md.append("")
        md.append(f"Промпт: `{shown}`")
        md.append("")
        md.append("Жадно:")
        md.append("```")
        md.append(run(model, tok, prompt, device, args.tokens, **GREEDY))
        md.append("```")
        md.append("Сэмплирование:")
        md.append("```")
        md.append(run(model, tok, prompt, device, args.tokens, **SAMPLED))
        md.append("```")
        md.append("")

    md.append(f"_Сгенерировано за {time.time() - t0:.0f} с на {args.device}._")
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[out] {args.out} ({len(md)} lines, step {step})")


if __name__ == "__main__":
    main()
