# Клонирование на рабочий стол

Репозиторий публичный, токен и логин не нужны. В PowerShell:

```powershell
cd $env:USERPROFILE\Desktop
git clone https://github.com/Icold21/kron-llm.git
cd kron-llm
```

Текущая вершина `main` — `56a3cd1`. Проверил доступ с этой машины: `git
ls-remote` отвечает.

## Дальше

```powershell
pip install -r requirements.txt
jupyter notebook notebooks/q10_playground.ipynb
```

Ноутбук запускать из корня клона — он ищет `kronllm/` рядом с рабочей
директорией.

## Если что-то не так

- **`git` не установлен**: `winget install -e --id Git.Git`, потом заново
  открыть PowerShell (PATH обновится только в новом окне).
- **Рабочий стол уехал в OneDrive**: тогда `cd "$env:USERPROFILE\OneDrive\Desktop"`
  вместо первой строки.
- **Нужна не последняя версия, а конкретная**: добавить `--branch main` не
  поможет, ветка одна; если понадобится откат — `git log --oneline` и
  `git checkout 56a3cd1`.
- **Хочется сразу и код, и окружение одной командой**:

```powershell
cd $env:USERPROFILE\Desktop; git clone https://github.com/Icold21/kron-llm.git; cd kron-llm; pip install -r requirements.txt
```

Про чекпоинт напомню: весов в репозитории нет, это отдельный шаг — либо
`kaggle datasets download icold21/kron-q10-ckpt -p data/kaggle_ckpt --unzip`
(нужен `kaggle.json`), либо скопировать `last.pt` с рабочей машины.
