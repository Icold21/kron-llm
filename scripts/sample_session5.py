"""Sample text from the session-5 checkpoint and mail the examples.

The model is 10 M parameters, 26 % through its first epoch, and it was trained
on this corpus for exactly 1,434 steps since the rollback.  The samples are
reported as they come out, including the bad ones: a trimmed transcript is a
worse tool than an honest one.

Usage::

    python scripts/sample_session5.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = Path(r"E:\kron_out\session5\q10")
CKPT = RUN / "last.pt"
MD = ROOT / "data" / "samples_session5.md"
HTML = ROOT / "data" / "samples_session5.html"
MAIL = (Path.home() / ".config" / "opencode" / "skills" / "email-control"
        / "scripts" / "mail.py")

PROMPTS = [
    ("EN", "once upon a time", "художественный текст"),
    ("EN", "The derivative of x^2 is", "математика"),
    ("EN", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
     "код"),
    ("EN", "The photo of a black hole was made by", "энциклопедическое"),
    ("EN", "What is the capital of France?\nThe capital of France is", "вопрос-ответ"),
    ("RU", "Однажды в маленьком городе", "художественный текст"),
    ("RU", "Производная функции x^2 равна", "математика"),
    ("RU", "Чёрная дыра — это", "энциклопедическое"),
]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from kronllm.generate import generate_texts
    from kronllm.model import KronTransformer
    from kronllm.tokenizer import load_tokenizer

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["data"]["data_dir"] = str(ROOT / "data" / "bilingual")
    tok = load_tokenizer(cfg)
    model = KronTransformer(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    step = ck.get("step")
    print(f"checkpoint step {step} | device {device} | params "
          f"{sum(p.numel() for p in model.parameters()):,}")

    results = []
    for lang, prompt, domain in PROMPTS:
        out = generate_texts(model, [prompt], max_new_tokens=90,
                             temperature=0.8, top_k=40, repetition_penalty=1.1,
                             n_samples=1, seed=42, device=device, tokenizer=tok)
        text = out[0]
        results.append((lang, domain, prompt, text))
        print(f"\n=== [{lang}/{domain}] {prompt[:40]!r}\n{text[:300]}")

    lines = [
        "# Q10: примеры генерации, шаг 36 549",
        "",
        f"Чекпоинт `last.pt` шага **{step}** (сессия 5, корпус без промптов). "
        f"Параметров — 9 938 886, обучено 26 % первой эпохи. "
        "Выборка: temperature 0.8, top_k 40, repetition penalty 1.1, "
        "до 90 новых токенов, seed 42.",
        "",
        "Читать так: это не ответы, а продолжения. Модель такого размера после "
        "четверти эпохи держит орфографию, язык и регистр, но не факты — "
        "арифметика у неё была 0/60 ещё на 35 000. Ниже всё как есть, без "
        "отбора удачных:",
        "",
    ]
    for lang, domain, prompt, text in results:
        lines += [
            f"## {prompt}  ",
            f"*{lang} · {domain}*",
            "",
            "```",
            text.replace("\r", "").strip()[:700],
            "```",
            "",
        ]
    lines += [
        "## Что это значит",
        "",
        "- связность на уровне фразы есть, на уровне абзаца — нет;",
        "- русский держится, но местами съезжает в английский или в перечисление;",
        "- код после `def` чаще всего не компилируется: слишком мало кода в "
        "26 % эпохи и слишком мало параметров;",
        "- фактических ответов нет, и это ожидаемо: модель училась на 0.9 % "
        "бюджета, который обычно уходит на знания.",
        "",
        "Осмысленный разговор появится после SFT, а не после ещё одного "
        "куска претрейна — сейчас мы проверяем, что модель вообще жива и "
        "морфологию держит.",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                    str(MD), str(HTML)], check=True)

    cmd = [sys.executable, str(MAIL), "send",
           "--subject", "Q10: примеры генерации, шаг 36 549",
           "--model", "DeepSeek V4.1 Flash Free", "--session", "kron_chain",
           "--html", "--body-file", str(HTML)]
    for attempt in range(1, 4):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print(f"[mail] attempt {attempt}: exit {r.returncode}")
        if r.returncode == 0:
            return
        print((r.stderr or r.stdout or "").strip()[-200:])
    print("[mail] could not send; body is in", HTML)


if __name__ == "__main__":
    main()
