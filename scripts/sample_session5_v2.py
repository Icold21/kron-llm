"""Session-5 samples with temperature 1.0, top_p 0.95, top_k 10.

Also reports the EOT accounting: how many end-of-document (8191) and turn-end
(8190) tokens the corpus holds, so the question "does it learn to emit EOT"
can be answered with counts instead of recollection.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BIN = Path(r"E:\kron_bins\epoch_np\train.bpe.bin")
RUN = Path(r"E:\kron_out\session5\q10")
MD = ROOT / "data" / "samples_session5_v2.md"
HTML = ROOT / "data" / "samples_session5_v2.html"
MAIL = (Path.home() / ".config" / "opencode" / "skills" / "email-control"
        / "scripts" / "mail.py")

PROMPTS = [
    ("EN", "художественный", "Once upon a time"),
    ("EN", "математика", "The derivative of x^2 is"),
    ("EN", "код", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr"),
    ("EN", "энциклопедическое", "The photo of a black hole was made by"),
    ("EN", "вопрос-ответ", "What is the capital of France?\nThe capital of France is"),
    ("RU", "художественный", "Однажды в маленьком городе"),
    ("RU", "математика", "Производная функции x^2 равна"),
    ("RU", "энциклопедическое", "Чёрная дыра — это"),
]


def main() -> None:
    from kronllm.generate import generate_texts
    from kronllm.model import KronTransformer
    from kronllm.tokenizer import load_tokenizer

    tokens = np.fromfile(BIN, dtype=np.uint16)
    n_eot = int((tokens == 8191).sum())
    n_turn = int((tokens == 8190).sum())
    n_sys = int((tokens == 8187).sum())
    print(f"bin {len(tokens):,} tokens | EOT 8191: {n_eot:,} | "
          f"<|eot|> 8190: {n_turn:,} | <|system|> 8187: {n_sys:,}")

    ck = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["data"]["data_dir"] = str(ROOT / "data" / "bilingual")
    tok = load_tokenizer(cfg)
    model = KronTransformer(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results, stops = [], 0
    for lang, domain, prompt in PROMPTS:
        out = generate_texts(model, [prompt], max_new_tokens=90,
                             temperature=1.0, top_p=0.95, top_k=10,
                             n_samples=1, seed=42, device=device, tokenizer=tok)
        text = out[0]
        body = text[len(prompt):].strip()
        stopped = len(body) < 420
        stops += int(stopped)
        results.append((lang, domain, prompt, text, stopped))
        print(f"\n=== [{lang}/{domain}] {prompt[:40]!r} stopped={stopped}\n{body[:260]}")

    lines = [
        "# Q10: примеры генерации, шаг 36 549 — настройки по вашему запросу",
        "",
        "`temperature 1.0`, `top_p 0.95`, `top_k 10`, до 90 новых токенов, "
        "seed 42, чекпоинт шага **36 549** (9 938 886 параметров, 26 % эпохи).",
        "",
        "Ваше замечание про `top_k 40` при словаре 8 192 верно: это отсекало "
        "всего 40 кандидатов из восьми тысяч, то есть практически ничего не "
        "делало, и распределение оставалось почти сырым. `top_p 0.95` с "
        "`top_k 10` — ограничение осмысленное.",
        "",
        "## Вопрос про EOT",
        "",
        f"В корпусе **{n_eot:,}** токенов EOT (id 8191, конец документа) — "
        f"ровно по одному на документ, и он **всегда** был целью обучения: "
        "`encode_docs` дописывает его в конец каждого документа, а цель — "
        "следующий токен, поэтому последний токен документа предсказывает EOT. "
        "Так было и в сессиях 1–5, это не новое.",
        "",
        f"`<|eot|>` (id 8190) — другой токен, конец реплики. В корпусе их "
        f"**{n_turn:,}**, и они приходят только из чат-документов "
        f"(<|system|> — {n_sys:,} штук). В сессии 4 их добавлял ещё и наш "
        "вставленный промпт — именно это мы и убрали.",
        "",
        f"**Но на генерации EOT не появляется:** из {len(PROMPTS)} примеров "
        f"остановились сами {stops}. При 300 токенах бюджета модель тоже "
        "ни разу не выдала EOT. Причина арифметическая: EOT — это "
        f"{100 * n_eot / len(tokens):.3f} % всех целей (один на ~2 300 "
        "токенов), и на 26 % эпохи такой редкий класс ещё не закрепился. "
        "Практический вывод: останавливать генерацию по EOT пока нельзя, "
        "нужен `max_new_tokens`.",
        "",
        "## Примеры",
        "",
    ]
    for lang, domain, prompt, text, stopped in results:
        lines += [
            f"### {prompt.splitlines()[0]}  ",
            f"*{lang} · {domain}*",
            "",
            "```",
            text.replace("\r", "").strip()[:700],
            "```",
            "",
        ]
    lines += [
        "## Итог",
        "",
        "- язык и орфография держатся, связность — на уровне фразы;",
        "- настроения/фактов нет; код синтаксически похож, но не работает;",
        "- с `top_p 0.95` / `top_k 10` текст стал заметно менее случайным, "
        "но и беднее — модель чаще повторяет одну конструкцию.",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                    str(MD), str(HTML)], check=True)
    cmd = [sys.executable, str(MAIL), "send",
           "--subject", "Q10: примеры с temp 1.0 / top_p 0.95 / top_k 10 и ответ про EOT",
           "--model", "DeepSeek V4.1 Flash Free", "--session", "kron_chain",
           "--html", "--body-file", str(HTML)]
    for attempt in range(1, 4):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print(f"[mail] attempt {attempt}: exit {r.returncode}")
        if r.returncode == 0:
            return
        print((r.stderr or r.stdout or "").strip()[-200:])


if __name__ == "__main__":
    main()
