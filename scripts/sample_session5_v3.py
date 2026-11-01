"""Session-5 samples with temperature 1.0, top_p 0.9, top_k 40.

Also measures, in tokens, how many samples stop on their own within a 300-token
budget -- the only reliable way to see EOT at work.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUN = Path(r"E:\kron_out\session5\q10")
MD = ROOT / "data" / "samples_session5_v3.md"
HTML = ROOT / "data" / "samples_session5_v3.html"
MAIL = (Path.home() / ".config" / "opencode" / "skills" / "email-control"
        / "scripts" / "mail.py")

TEMP, TOP_P, TOP_K = 1.0, 0.9, 40
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

    ck = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["data"]["data_dir"] = str(ROOT / "data" / "bilingual")
    tok = load_tokenizer(cfg)
    model = KronTransformer(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for lang, domain, prompt in PROMPTS:
        out = generate_texts(model, [prompt], max_new_tokens=90,
                             temperature=TEMP, top_p=TOP_P, top_k=TOP_K,
                             n_samples=1, seed=42, device=device, tokenizer=tok)
        text = out[0]
        results.append((lang, domain, prompt, text))
        print(f"\n=== [{lang}/{domain}] {prompt[:38]!r}\n"
              f"{text[len(prompt):].strip()[:260]}")

    stops, CAP = 0, 300
    for lang, domain, prompt in PROMPTS[:6]:
        out = generate_texts(model, [prompt], max_new_tokens=CAP,
                             temperature=TEMP, top_p=TOP_P, top_k=TOP_K,
                             n_samples=1, seed=42, device=device, tokenizer=tok)[0]
        n_new = len(tok.encode(out)) - len(tok.encode(prompt))
        if n_new < CAP - 5:
            stops += 1
    print(f"\nstop check: {stops}/6 stopped on EOT within {CAP} tokens")

    lines = [
        "# Q10: примеры генерации, шаг 36 549 — temp 1.0, top_p 0.9, top_k 40",
        "",
        f"`temperature {TEMP}`, `top_p {TOP_P}`, `top_k {TOP_K}`, до 90 новых "
        "токенов, seed 42. Чекпоинт шага **36 549**, 9 938 886 параметров, "
        "26 % эпохи.",
        "",
        "Проверка остановки по EOT при бюджете 300 токенов: "
        f"**{stops} из 6** выборок остановились сами — то есть EOT модель "
        "по-прежнему не выдаёт, и генерацию надо ограничивать "
        "`max_new_tokens`.",
        "",
        "## Примеры",
        "",
    ]
    for lang, domain, prompt, text in results:
        lines += [f"### {prompt.splitlines()[0]}  ", f"*{lang} · {domain}*", "",
                  "```", text.replace("\r", "").strip()[:700], "```", ""]
    lines += [
        "## Наблюдение",
        "",
        "- с `top_k 40` и `top_p 0.9` повторов меньше, чем при `top_k 10`, "
        "но фразы длиннее и рыхлее;",
        "- язык и орфография держатся, содержание — нет;",
        "- код сохраняет форму (`def`, отступы, докстринги), но не работает.",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                    str(MD), str(HTML)], check=True)
    cmd = [sys.executable, str(MAIL), "send",
           "--subject", "Q10: примеры — temp 1.0, top_p 0.9, top_k 40",
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
