"""Set the title / subtitle / description of the KronLLM Kaggle datasets.

The installed CLI has no `datasets update`, but ``POST /api/v1/datasets/
metadata/{owner}/{slug}`` accepts the metadata JSON directly (201 with an empty
``errors`` list), so descriptions can change without re-uploading 5.6 GB.

The wording follows the project rule: **Q10 (9,938,886 parameters) is the
model**; the 2-3M TinyStories runs are the development bench and are not shipped
as datasets.

Usage::

    python scripts/set_dataset_meta.py --dry-run
    python scripts/set_dataset_meta.py
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

API = "https://www.kaggle.com/api/v1"

REPO = "https://github.com/Icold21/kron-llm"

COMMON = f"""
**This is infrastructure for KronLLM Q10**, the 10M-parameter bilingual
(English/Russian) language model whose matrix layers are Kronecker sums of rank-1
factors. Q10 has 9,938,886 parameters (d=512, 25 layers, 8K bilingual BPE) and is
the model this project ships; the smaller 2-3M runs in the code repository are a
TinyStories development bench, not the product.

Code, measurements and the honest negative results: {REPO}
"""

DATASETS = {
    "icold21/kron-corpus-2b": {
        "title": "KronCorpus: EN/RU bins for KronLLM Q10",
        "subtitle": "2.78B tokens, EN 59.7 % / RU 40.3 % by tokens, uint16 bins",
        "description": """
Training bins for **KronLLM Q10** — 2,783,383,150 tokens over 933,522 documents
(10.02 GB of text, 3.60 bytes per token), English 59.7 % / Russian 40.3 % **by
tokens**, with a science/math/code accent and 33 % prose inside English.

| file | what it is |
|---|---|
| `train.bpe.bin` | uint16 token ids, one epoch = 169,884 steps of 16,384 tokens |
| `val.bpe.bin` | the held-out bin, sha256 `a45bbb03…`, 46,197,636 tokens |
| `tokenizer.json` | the 8K byte-level BPE (ids 8187-8190 are chat markup) |
| `tokenizer_meta.json` | counts, language shares, per-domain quotas, prompts off |

The held-out bin **never changes between versions**, so `val_bpb_mixed` from any
checkpoint is comparable across the whole run. Quotas are per language and
quality-ranked: within each domain the rebuild takes the highest-scoring
documents (alpha density, digit ratio, vocabulary diversity, word length,
length bonus), not a random sample. No system prompts are inserted into the text.

Sources keep their own licences (Wikipedia/Wikibooks CC BY-SA, Gutenberg public
domain, Stack Overflow CC BY-SA, Habr, open instruction and chat corpora); the
provenance record is in the corpus repository.
""" + COMMON,
    },
    "icold21/kron-q10-ckpt": {
        "title": "KronLLM Q10 checkpoints (the 10M bilingual model)",
        "subtitle": "9,938,886 parameters; last.pt resumes, best.pt is weights-only",
        "description": """
Checkpoints of **KronLLM Q10**, the model this project ships: 9,938,886
parameters (d=512, 25 layers, K=10 Kronecker terms, 8K bilingual BPE, gated FFN).

| file | contents |
|---|---|
| `last.pt` | model + AdamW state + step + corpus fingerprint (`data_key`, `tok_key`, `pass_origin`) — this is what a session resumes from |
| `best.pt` | weights only, the best validation the run has seen |

One version per training session; the step is in the file name of the report and
in the dataset version notes. At the time of writing the run is at step 36,549 of
169,884 (21.5 % of the first epoch) with **1.1828 bits/byte** mixed (EN 1.4327 /
RU 0.9411) on the held-out bin, against 1.554 for GPT-2-124M on the same blocks.

Resume with `python -m kronllm.cli train --exp Q10 --data-dir DIR
--resume last.pt`; a different corpus requires `training.allow_data_change=true`,
a different tokenizer is refused outright.

Note: the 2-3M checkpoints in the code repository (`artifacts/q8g_int8.pt`,
`runs/Q8g_long`) belong to the TinyStories bench, not to this model.
""" + COMMON,
    },
    "icold21/kron-q10a-anneal-bin": {
        "title": "KronLLM Q10A decay bin",
        "subtitle": "Decay pass for Q10: 809M tokens, quality-selected, 5 % chat",
        "description": """
The decay-pass corpus (stage **Q10A**) for KronLLM Q10: **809,120,240 tokens**
over 249,868 documents, 3.53 bytes per token, 29.1 % of an epoch, read exactly
once as 49,384 steps of 16,384 tokens.

The domain profile is the epoch's, selected by quality, plus **5 % chat** spread
through the pass — the four chat tokens (`<|system|>`, `<|user|>`,
`<|assistant|>`, `<|eot|>`) are cold at the end of pretraining and would
otherwise cost SFT its first few hundred steps. The held-out bin is the epoch's
(`a45bbb03…`), so the decay pass is scored on the same blocks as everything
before it. LR anneals 3e-4 → 1e-6 with no warmup.

Same file layout as the epoch corpus: `train.bpe.bin`, `val.bpe.bin`,
`tokenizer.json`, `tokenizer_meta.json` (phase `anneal`).
""" + COMMON,
    },
    "icold21/kron-llm-code-v20": {
        "title": "KronLLM training code v20",
        "subtitle": "Code the running Q10 session mounted: block sampler fix",
        "description": """
The `kronllm` package as a zip, mounted by the Kaggle kernel that trains
**Q10** — the 10M-parameter bilingual model (9,938,886 parameters). Versions are
cut per code change so a resumed session can never silently mount stale code.

v20 carries the **block sampler** fix: the epoch permutation used to be over every
token offset, which covered 63 % of the corpus per epoch and repeated the rest
~1.6×; it now permutes blocks of `seq_len` tokens, so one epoch is one honest
pass. It also carries the per-document index (`train_index.npy`) that makes the
seen-set computable, the prompt-target mask and the corpus guards.
""" + COMMON,
    },
    "icold21/kron-llm-code-v21": {
        "title": "KronLLM training code v21",
        "subtitle": "Adds the Q10A decay-pass step count (49,384)",
        "description": """
The `kronllm` package as a zip, for the **Q10A decay pass** of KronLLM Q10 (the
10M-parameter bilingual model). Identical to v20 except that the decay-pass
config derives its 49,384 steps from the anneal bin, so the pass reads its data
exactly once.
""" + COMMON,
    },
}


def call(slug: str, body: dict) -> tuple[int, str]:
    key = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    req = urllib.request.Request(
        f"{API}/datasets/metadata/{slug}",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key["key"],
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()[:200].decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:200].decode("utf-8", "replace")


def read_back(slug: str) -> dict:
    key = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    req = urllib.request.Request(f"{API}/datasets/view/{slug}",
                                 headers={"Authorization": "Bearer " + key["key"]})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for slug, meta in DATASETS.items():
        body = {"id": slug, "isPrivate": True,
                "licenses": [{"name": "other"}], **meta}
        if args.dry_run:
            print(f"{slug}: {meta['title']} | {meta['subtitle']}")
            continue
        code, text = call(slug, body)
        print(f"{slug}: {code} {text[:120]}")
        if code // 100 != 2:
            continue
        info = read_back(slug)
        got_title = info.get("title")
        got_desc = len(info.get("description") or "")
        ok = got_title == meta["title"] and got_desc > 200
        print(f"   read back: title={'ok' if ok else got_title!r}, "
              f"description {got_desc} chars")


if __name__ == "__main__":
    main()
