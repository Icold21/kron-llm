"""Generate the Colab notebook that runs one training session.

Colab has no API: a session exists only while a browser tab is open and signed
in, so this notebook is meant to be opened and run by the user, not launched by
a script.  What it removes is the fiddly part -- finding the datasets, putting
the bins where the trainer expects them, and pushing the checkpoint back -- so
that a session is one click whenever a free T4 is available.

The notebook is generated rather than hand-edited so that the dataset slugs and
the training command live in one place with the rest of the pipeline.

Usage::

    python scripts/make_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat

OUT = Path("notebooks/kron_colab.ipynb")

HEADER = """# KronLLM — одна сессия на Colab (T4)

Ноутбук делает то же, что Kaggle-кернел, но на бесплатной карте Colab:
скачивает корпус, чекпоинт и код с Kaggle, доезжает от последнего шага,
тренирует `--minutes` минут и заливает `last.pt` обратно новой версией
датасета. Следующая сессия (здесь или на Kaggle) продолжит с этого места.

**Что нужно один раз:** Kaggle-токен в *Colab Secrets* (иконка ключа слева):
`KAGGLE_USERNAME` и `KAGGLE_KEY` — те же значения, что в `kaggle.json`.
Токен даёт и чтение датасетов, и запись чекпоинта; больше ничего не нужно.

**Бюджет.** Бесплатный Colab обрывает сессию на 4 часах без предупреждения,
поэтому по умолчанию `MINUTES = 200`: остаётся запас на скачивание и загрузку.
Для Colab Pro поставь 660.

**Важно:** пока вкладка открыта — обучение идёт; закрыл вкладку — сессия
умерла. Чекпоинт пишется каждые 1000 шагов и на выходе по бюджету, так что
потеря ограничена.

Этот ноутбук стартует с шага **35 115** (усреднённый `soup_035115`): сессия,
ушедшая до 43 895, отброшена, потому что училась считать лосс и на нашем же
вставленном системном промпте. Теперь такие позиции вырезаны маской
(`kronllm/loss_mask.py`, код v19), поэтому тренировочный лосс на первом шаге
подскочит примерно на 0.012 нат — это ожидаемо и честно.

Запуск: `Runtime → Change runtime type → T4 GPU`, затем `Runtime → Run all`.
Первая ячейка-проверка качает только мету (1.5 КБ) и отказывается стартовать,
если корпус на Kaggle ещё старый или чекпоинт не тот.
"""

ENV = '''import shutil, subprocess, sys
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout or 'no GPU')
print('disk free', shutil.disk_usage('/content').free // 2**30, 'GiB')
'''

AUTH = '''import json, os, pathlib

def kaggle_auth():
    """Token from Colab Secrets, or a kaggle.json upload as a fallback."""
    dst = pathlib.Path('/root/.kaggle/kaggle.json')
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from google.colab import userdata
        user, key = userdata.get('KAGGLE_USERNAME'), userdata.get('KAGGLE_KEY')
    except Exception:
        user = key = None
    if not (user and key):
        from google.colab import files
        print('Secrets KAGGLE_USERNAME / KAGGLE_KEY not set; upload kaggle.json')
        up = files.upload()
        raw = json.loads(next(iter(up.values())).decode())
        user, key = raw['username'], raw['key']
    dst.write_text(json.dumps({'username': user, 'key': key}))
    os.chmod(dst, 0o600)
    print('kaggle auth for', user)

kaggle_auth()
# Pinned: kaggle 1.x signs the v1 API with basic auth, which it now answers
# with 403, and its failure is silent (exit 1, empty stderr).  The push at the
# end of the session is the one step that still needs this CLI.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle>=2.2.5'],
               check=True)
print('kaggle cli', subprocess.run([sys.executable, '-m', 'kaggle', '--version'],
                                  capture_output=True, text=True).stdout.strip())
'''

# Downloads go straight to the REST API with requests.  Two reasons the kaggle
# CLI is not used here: it failed silently in Colab (exit 1, no stderr), and the
# cause is the auth scheme -- the v1 API answers 403 "Permission 'datasets.get'
# was denied" to HTTP basic auth (username + key) and 200 to the same key sent
# as a bearer token.  Older CLI versions sign with basic auth, so the download
# path does not depend on which one pip happened to install.
API = '''import io, json, pathlib, zipfile, requests

KJ = json.loads(pathlib.Path('/root/.kaggle/kaggle.json').read_text())
BASE = 'https://www.kaggle.com/api/v1/datasets/download'
HEAD = {'Authorization': 'Bearer ' + KJ['key']}

def _stream(url, dest):
    r = requests.get(url, headers=HEAD, stream=True, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(f'GET {url} -> HTTP {r.status_code}: {r.text[:300]}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(dest, 'wb') as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
            done += len(chunk)
    print(f'   {dest.name}: {done/1e6:.1f} MB', flush=True)
    return dest

def fetch_file(slug, name, dest):
    return _stream(f'{BASE}/{slug}/{name}', pathlib.Path(dest))

def fetch_all(slug, dest_dir):
    """Whole dataset: the API returns one zip, so unzip it in place."""
    dest_dir = pathlib.Path(dest_dir); dest_dir.mkdir(parents=True, exist_ok=True)
    zpath = _stream(f'{BASE}/{slug}', dest_dir / (slug.split('/')[1] + '.zip'))
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest_dir)
    zpath.unlink()
    print('   unzipped into', dest_dir, flush=True)
    return dest_dir
'''

PREFLIGHT = '''# Ask Kaggle two file-sized questions before pulling 4.5 GB: is the corpus
# the fully-prompted one, and does the checkpoint sit where we expect?
from pathlib import Path

PRE = Path('/content/pre')
fetch_file('icold21/kron-corpus-2b', 'tokenizer_meta.json', PRE / 'tokenizer_meta.json')
fetch_file('icold21/kron-q10-ckpt', 'last.pt', PRE / 'last.pt')

meta = json.loads((PRE / 'tokenizer_meta.json').read_text())
dp = meta.get('domain_prompts') or {}
covered = dp.get('prompted_docs', 0) + dp.get('already_prompted_docs', 0)
print('phase        ', meta.get('phase'))
print('train tokens ', f"{meta.get('train_tokens', 0):,}")
print('prompted     ', f"{dp.get('prompted_docs', 0):,} added +",
      f"{dp.get('already_prompted_docs', 0):,} already chat =",
      f"{covered:,} of {dp.get('docs', 0):,} at rate", dp.get('rate'))

assert meta.get('phase') == 'epoch', 'wrong bin: this is the decay pass'
assert dp.get('rate') == 1.0, ('the corpus dataset is not the fully-prompted '
                              'rebuild; upload it before running this')
assert covered == dp.get('docs'), 'not every document carries a system prompt'

state = torch.load(PRE / 'last.pt', map_location='cpu', weights_only=False)
print('resume step  ', state.get('step'))
if state.get('soup'):
    print('resume source', 'averaged checkpoint (soup), no optimizer state')
assert state.get('step') == 35115, 'unexpected resume point: ' + str(state.get('step'))
print('preflight OK')
'''

DOWNLOAD = '''from pathlib import Path
DL = Path('/content/dl'); DL.mkdir(exist_ok=True)

# The whole-dataset endpoint serves a zip only while the dataset is small: for
# the 4.5 GB corpus it answers 404 "No gcs url found", while every individual
# file comes back with the right content-length.  So a failed zip falls back to
# the file list the trainer actually needs, rather than failing the session.
CORPUS_FILES = ('train.bpe.bin', 'val.bpe.bin', 'tokenizer.json',
                'tokenizer_meta.json', 'val_index.json')
FALLBACK = {'icold21/kron-corpus-2b': CORPUS_FILES,
            'icold21/kron-q10-ckpt': ('last.pt', 'best.pt')}

for slug in ['icold21/kron-llm-code-v19', 'icold21/kron-corpus-2b',
             'icold21/kron-q10-ckpt']:
    print('[dl]', slug, flush=True)
    try:
        fetch_all(slug, DL)
    except RuntimeError as exc:
        print('   zip unavailable:', str(exc)[:120], flush=True)
        print('   downloading file by file', flush=True)
        for name in FALLBACK.get(slug, CORPUS_FILES):
            fetch_file(slug, name, DL / name)

def find(name):
    hits = sorted(p for p in DL.rglob(name))
    return hits[0] if hits else None

for probe in ('train.bpe.bin', 'val.bpe.bin', 'last.pt', 'kronllm'):
    print(f'{probe:16s}', find(probe))
'''

ASSEMBLE = '''WORK = Path('/content/work')
shutil.rmtree(WORK, ignore_errors=True)
for sub in ('data', 'run', 'push'):
    (WORK / sub).mkdir(parents=True)

DATA_FILES = ('train.bpe.bin', 'val.bpe.bin', 'tokenizer.json',
              'tokenizer_meta.json', 'val_index.json')
for name in DATA_FILES:
    src = find(name)
    if src is None:
        print('missing', name)
        continue
    shutil.copyfile(src, WORK / 'data' / name)
    print(f'{name:20s} {src.stat().st_size/1e6:8.1f} MB')

pkg = find('kronllm')
CODE_ROOT = pkg.parent
print('code root', CODE_ROOT)
print('bins:', sorted(p.name for p in (WORK / 'data').iterdir()))

# The corpus is 4.5 GB and the dataset slug does not change between rebuilds, so
# a stale or half-uploaded download looks fine until the loss is wrong.  Read
# the metadata and refuse a corpus that is not the fully-prompted one.
meta = json.loads((WORK / 'data' / 'tokenizer_meta.json').read_text())
dp = meta.get('domain_prompts') or {}
print('phase        ', meta.get('phase'))
print('train tokens ', f"{meta.get('train_tokens', 0):,}")
print('prompted docs', f"{dp.get('prompted_docs', 0):,}", 'added,',
      f"{dp.get('already_prompted_docs', 0):,}", 'already chat, of',
      f"{dp.get('docs', 0):,}", 'at rate', dp.get('rate'))
assert meta.get('phase') == 'epoch', 'wrong bin: this is the decay pass'
assert dp.get('rate') == 1.0, 'corpus predates the full-prompt rebuild'
covered = dp.get('prompted_docs', 0) + dp.get('already_prompted_docs', 0)
assert covered == dp.get('docs'), 'not every document carries a system prompt'
print('corpus is the fully-prompted epoch bin')
'''

TRAIN = '''MINUTES = 200          # a free T4 is cut near 240; Pro allows 660
CKPT = find('last.pt')
cmd = [sys.executable, '-u', '-m', 'kronllm.cli', 'train', '--exp', 'Q10',
       '--device', 'cuda',
       '--data-dir', str(WORK / 'data'),      # train.bpe.bin, tokenizer_meta.json
       '--run-dir', str(WORK / 'run'),
       '--minutes', str(MINUTES),
       '--override', 'training.log_every=50']
assert CKPT is not None, ('no checkpoint in /content/dl: refusing to start a '
                          'fresh run by accident, the session resumes from '
                          'step 35,115')
cmd += ['--resume', str(CKPT)]
step = torch.load(CKPT, map_location='cpu', weights_only=False).get('step')
assert step == 35115, 'unexpected resume point: ' + str(step)
print('resuming from step', step)
print(' '.join(cmd))
subprocess.run(cmd, cwd=WORK, check=True,
               env=dict(os.environ, PYTHONPATH=str(CODE_ROOT),
                        PYTHONIOENCODING='utf-8'))
'''

PUSH = '''ck = WORK / 'run' / 'last.pt'
if not ck.exists():
    print('no checkpoint written')
else:
    step = torch.load(ck, map_location='cpu', weights_only=False).get('step')
    # A copy in /content so the file can be dragged out of the Files pane by
    # hand if the upload fails -- 200 minutes of training should never hinge on
    # one CLI call.
    shutil.copyfile(ck, '/content/last.pt')
    print('checkpoint copied to /content/last.pt (step', step, ')')

    stage = WORK / 'push'
    shutil.copyfile(ck, stage / 'last.pt')
    # best.pt is weights-only and is what the next session averages with; the
    # soup needs neighbouring checkpoints, so it goes up too.
    best = WORK / 'run' / 'best.pt'
    if best.exists():
        shutil.copyfile(best, stage / 'best.pt')
    (stage / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'Kron Q10 checkpoint', 'id': 'icold21/kron-q10-ckpt',
        'licenses': [{'name': 'other'}]}))
    r = subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'version',
                        '-p', str(stage), '-m', f'colab session, step {step}',
                        '--dir-mode', 'zip'], capture_output=True, text=True)
    print('push exit', r.returncode, 'at step', step)
    if r.returncode != 0:
        print('stdout:', r.stdout[-800:])
        print('stderr:', r.stderr[-800:])
        print('UPLOAD FAILED: download /content/last.pt from the Files pane and')
        print('version the dataset icold21/kron-q10-ckpt with it by hand.')
'''

FOOTER = """## Что дальше

- Если `push exit 0` — чекпоинт на Kaggle обновился, следующая сессия (здесь
  или в цепочке) продолжит с этого шага.
- Если push не прошёл, скачай `work/run/last.pt` вручную кнопкой Files слева
  и загрузи как версию датасета `icold21/kron-q10-ckpt`.
- `--minutes` не меняет расписание обучения: это те же шаги, просто сессия
  останавливается сама и пишет чекпоинт, вместо того чтобы быть убитой
  таймаутом Colab.
"""


def code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(src.strip("\n"))


def md(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(src.strip("\n"))


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = [
        md(HEADER),
        code(ENV),
        code(AUTH),
        code("import torch"),
        code(API),
        code(PREFLIGHT),
        code(DOWNLOAD),
        code(ASSEMBLE),
        code(TRAIN),
        code(PUSH),
        md(FOOTER),
    ]
    nb["metadata"] = {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(OUT))
    nbformat.validate(nb)
    print(f"[nb] {OUT} — {len(nb['cells'])} cells, validated")


if __name__ == "__main__":
    main()
