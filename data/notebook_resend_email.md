# Ноутбук во вложении — уже с исправлением

Предыдущее письмо содержало старую версию, ту самую, что падала на ноуте с
`max() iterable argument is empty`. Это новая: `kron_q10_playground.ipynb`,
18 770 байт, правилась в 14:01.

## Что в ней починено

- репозиторий ищется относительно рабочей директории (корень клона или
  `notebooks/`), а не по моему пути `C:\Users\Icold\Desktop\kron_llm` — его в
  файле больше нет;
- чекпоинт ищется в `data/kaggle_out/q10/last.pt`, `data/kaggle_ckpt/last.pt`
  и `runs/*/last.pt`;
- если ни одного нет — ячейка сама зовёт `kaggle datasets download
  icold21/kron-q10-ckpt -p data/kaggle_ckpt --unzip`;
- если и это не вышло — печатает три варианта (Kaggle, копия с рабочей машины,
  встроенный TinyStories-артефакт) вместо трейсбека;
- токенизатор берётся из `data/bilingual/tokenizer.json`, а в свежем клоне — из
  `artifacts/q10_tokenizer.json` (643 КБ, уже в репозитории).

Оба пути проверены локально: с чекпоинтом грузится и печатает шаг 11 723,
9 938 886 параметров; без чекпоинта и без `kaggle` в PATH выдаёт список
вариантов.

## Что нужно на ноуте

1. `git pull` в клоне (там же обновлённый `notebooks/q10_playground.ipynb` и
   появился `artifacts/q10_tokenizer.json`);
2. либо положить `kaggle.json` в `%USERPROFILE%\.kaggle\`, либо скопировать
   `last.pt` (114 МБ) с этой машины в `data/kaggle_ckpt/`;
3. запускать ноутбук из корня клона.

Остальные ячейки не менялись: стриминг, шесть примеров, своя ячейка с
`PROMPT`/`TEMPERATURE`/`TOP_K`/`REPETITION_PENALTY` и формат диалога.
