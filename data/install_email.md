# URL не тот: это страница репозитория, а не архив

`pip` умеет ставить только из архива (`.zip`/`.tar.gz`) или из VCS-ссылки.
`https://github.com/Icold21/kron-llm` — это HTML-страница репозитория, поэтому
pip честно скачал 468 КБ разметки и сказал `cannot detect archive format`. Дело
не в репозитории и не в сети: пакет собирается нормально, я проверил прямо
сейчас — `pip install --dry-run` отвечает `Would install kronllm-2.0.0`.

## Три правильных варианта

```powershell
# 1. архив, git не нужен
pip install https://github.com/Icold21/kron-llm/archive/refs/heads/main.zip

# 2. через git (git должен быть установлен)
pip install "git+https://github.com/Icold21/kron-llm.git"

# 3. для разработки: клон + редактируемая установка
git clone https://github.com/Icold21/kron-llm.git
cd kron-llm
pip install -e ".[dev]"
```

Разница между 1 и 2 только в том, чем pip качает: zip по HTTPS или git-протокол.
Пункт 3 — то, что нужно, если собираешься что-то править.

## Две ловушки, которые ждут дальше

**Torch.** В `pyproject.toml` зависимость `torch>=2.4`, и на Windows pip по
умолчанию тянет сборку с CUDA — это ~2.5 ГБ. Если на ноуте нет NVIDIA, сначала
поставь CPU-колесо, потом остальное:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install https://github.com/Icold21/kron-llm/archive/refs/heads/main.zip
```

Второй pip увидит, что torch уже есть, и не будет его перекачивать.

**Твои таймауты.** В логе дважды `Connection to github.com timed out` —
канал до GitHub нестабильный. Архивная ссылка редиректит на
`codeload.github.com`, так что проверь, что и он открывается; если сеть совсем
плохая, добавь `--timeout 60 --retries 5`.

## Что теперь есть в README

Дописал в раздел «6. Running» все три формы установки и отдельно — именно эту
ловушку с URL, чтобы она не повторялась. Плюс команда `kronllm` после установки
доступна как обычный CLI (`kronllm train --exp ...`).

Если после установки хочется просто проверить, что всё живое:

```powershell
python -c "import kronllm, kronllm.cli; print(kronllm.__version__ if hasattr(kronllm,'__version__') else 'ok')"
python -m kronllm.cli budget
```

`budget` печатает таблицу параметров по конфигурациям и падает с кодом 1, если
конфигурация вылезает за 10M — быстрый способ убедиться, что пакет собран и
конфиги читаются.
