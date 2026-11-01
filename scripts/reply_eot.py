"""Correct the stop-count in the v2 letter and reply in the user's thread.

The previous letter claimed "8 of 8 samples stopped on their own".  That came
from comparing character counts, which for a 90-token budget cannot tell a
finished sample from a full one.  Measured in tokens with a 300-token budget,
the answer is 0 of 6: the model never emits EOT.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "data" / "samples_session5_v2.md"
HTML = ROOT / "data" / "samples_session5_v2.html"
MAIL = (Path.home() / ".config" / "opencode" / "skills" / "email-control"
        / "scripts" / "mail.py")
UID = "253"

text = MD.read_text(encoding="utf-8")

old_stop = ("**Но на генерации EOT не появляется:** из 8 примеров "
            "остановились сами 8.")
new_stop = ("**Но на генерации EOT не появляется.** Измерено по токенам, "
            "бюджет 300: **0 из 6** выборок остановились сами, все упёрлись "
            "в лимит.")
if old_stop in text:
    text = text.replace(old_stop, new_stop)
else:
    print("anchor for the stop count not found")

correction = (
    "\n## Поправка к предыдущему письму\n\n"
    "В прошлом письме я написал, что остановились сами 8 из 8 примеров. "
    "Это неверно: я сравнивал длину в символах, а при бюджете 90 токенов так "
    "не отличить законченный текст от упёршегося в лимит. Правильное "
    "измерение — по токенам при бюджете 300: **0 из 6**. Извините за "
    "неточность.\n")
if "Поправка к предыдущему письму" not in text:
    marker = "## Итог"
    text = text.replace(marker, correction + "\n" + marker, 1)

MD.write_text(text, encoding="utf-8")
subprocess.run([sys.executable, str(ROOT / "scripts" / "md_to_html.py"),
                str(MD), str(HTML)], check=True)

cmd = [sys.executable, str(MAIL), "reply", UID,
       "--model", "DeepSeek V4.1 Flash Free", "--session", "kron_chain",
       "--html", "--body-file", str(HTML)]
for attempt in range(1, 4):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(f"[mail] attempt {attempt}: exit {r.returncode}")
    if r.returncode == 0:
        print((r.stdout or "").strip()[-200:])
        break
    print((r.stderr or r.stdout or "").strip()[-300:])
