"""Mail the playground notebook to the owner, with a short HTML cover note."""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIL = (pathlib.Path.home() / ".config" / "opencode" / "skills"
        / "email-control" / "scripts" / "mail.py")
NB = ROOT / "notebooks" / "kron_playground.ipynb"

BODY = """<div style="font-family: -apple-system, Segoe UI, Roboto, sans-serif; font-size: 14px; line-height: 1.5;">
<p>Ноутбук с полной батареей тестов приложен: <code>kron_playground.ipynb</code> (28 ячеек).</p>
<p><b>Что внутри</b></p>
<ul>
<li>раздел 1 &mdash; девять стандартных бенчмарков (hellaswag, arc, piqa, winogrande,
openbookqa, boolq, mmlu, lambada) для S10 и Q10A, с линией случая на графике;</li>
<li>раздел 2 &mdash; bits per byte по двум удержанным бинам;</li>
<li>раздел 3 &mdash; наши тесты: столбики по семействам, голая форма с примерами ошибок,
чат, метрики повторов, свободная генерация;</li>
<li>раздел 4 &mdash; четыре интерактивные панели: генерация (T, top_k, top_p, штраф),
чат с транскриптом, арифметический тренажёр с вердиктом RIGHT/WRONG от Python,
инспектор токенизатора;</li>
<li>раздел 5 &mdash; перезапуск батареи одной кнопкой;</li>
<li>разделы 6&ndash;7 &mdash; кривая претрейна и выводы о том, что модель может и чего нет.</li>
</ul>
<p><b>Как открыть</b>: положить в папку проекта и запустить
<code>jupyter lab notebooks/kron_playground.ipynb</code>, либо открыть в VS Code.
Нужен <code>ipywidgets</code> (уже установлен) и два чекпойнта:
<code>E:/kron_out/s10_ckpt/last.pt</code> и <code>E:/kron_out/q10_ckpt/last.pt</code>.</p>
<p>Ноутбук собран скриптом <code>scripts/make_playground.py</code> &mdash; правки лучше
делать в нём, а не в JSON.</p>
</div>"""


def main() -> None:
    body = ROOT / "data" / "notebook_email_body.html"
    body.write_text(BODY, encoding="utf-8")
    r = subprocess.run([sys.executable, str(MAIL), "send",
                        "--subject", "KronLLM: ноутбук с бенчмарками и панелями",
                        "--model", "DeepSeek V4.1 Flash",
                        "--session", "kron_chain",
                        "--html", "--body-file", str(body),
                        "--attach", str(NB)],
                       capture_output=True, text=True, encoding="utf-8")
    print(r.returncode)
    print((r.stdout or "")[-300:])
    print((r.stderr or "")[-200:])


if __name__ == "__main__":
    main()
