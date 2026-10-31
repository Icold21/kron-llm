"""Show what the checkpoint actually writes, on prompts of each kind.

The metrics in ``eval_samples.py`` say whether the text is predictable; they do
not show it.  This prints a handful of continuations so the numbers can be
checked against something readable: general prose in both languages, code,
arithmetic, science, and a chat turn behind a system prompt -- the last one
matters because the corpus now opens documents with ``<|system|>`` and the
model is supposed to have learned what the tag means.

Decoding is the agreed recipe: T=1.0, top_k=8, top_p=0.9, repetition 1.15;
code is sampled colder (T=0.8) because code wants the likeliest branch.

Usage::

    python scripts/generate_demo.py --ckpt data/kaggle_out/q10/last.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eval_samples import load
from kronllm.tokenizer import TURN_EOT_ID

# (label, prompt, new tokens, temperature, stop at <|eot|>)
#
# The chat cases stop at <|eot|> because that is how serving works: the harness
# writes the template up to and including "<|assistant|>", the model writes the
# answer and then the end-of-turn token, and nothing after it.  Without the stop
# the model keeps writing the *next* turn -- the pretraining corpus is whole
# conversations, so "keep going" is exactly what it learned.
CASES = [
    ("EN prose", "The history of the steam engine begins", 70, 1.0, False),
    ("RU prose", "История паровой машины начинается", 70, 1.0, False),
    ("EN science", "When nitric acid reacts with copper, the products are", 70, 1.0, False),
    ("RU science", "При нагревании перманганата калия выделяется", 70, 1.0, False),
    ("code", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", 70, 0.8, False),
    ("math", "Solve for x: 3x + 7 = 22\nx =", 40, 1.0, False),
    ("chat EN", "<|system|>You are a helpful assistant.<|eot|>"
                "<|user|>What is the capital of France?<|eot|><|assistant|>",
     60, 1.0, True),
    ("chat RU", "<|system|>Ты — полезный ассистент.<|eot|>"
                "<|user|>Сколько будет семнадцать плюс двадцать пять?<|eot|>"
                "<|assistant|>", 60, 1.0, True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("data/kaggle_out/q10/last.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default="", help="substring filter on the label")
    args = ap.parse_args()

    model, tok, step = load(args.ckpt, args.device)
    print(f"### {args.ckpt.name} — step {step}\n")

    for label, prompt, n, temp, stop in CASES:
        if args.only and args.only.lower() not in label.lower():
            continue
        ids = torch.tensor([tok.encode(prompt)], device=args.device)
        out = model.generate(ids, max_new_tokens=n, temperature=temp,
                             top_k=8, top_p=0.9, repetition_penalty=1.15,
                             eot_id=TURN_EOT_ID, stop_at_eot=stop)
        new = out[0, ids.shape[1]:].tolist()
        text = tok.decode(new)
        ended = bool(stop and new and new[-1] == TURN_EOT_ID)
        print(f"--- {label} (T={temp}) ---")
        print(prompt.replace("\n", " ⏎ "), end="")
        print(text)
        if stop:
            print(f"[{'остановилась на <|eot|>' if ended else 'НЕ поставила <|eot|> за ' + str(len(new)) + ' токенов'}]")
        print()


if __name__ == "__main__":
    main()
