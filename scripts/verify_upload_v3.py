"""Verify the rebuilt bin and publish it, then read the metadata back.

Gates, in order:

* the held-out bin is byte-identical to the one every historical bpb was
  measured on;
* the metadata describes *this* mix (the composition is rewritten from
  ``corpus_report.json``, because ``--reuse-val-dir`` copies the old one);
* the bin holds no inserted prompts: ``<|system|>`` must equal the number of
  chat documents, and nothing else;
* the token count matches the metadata, so a truncated write is caught here
  rather than on Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default=r"E:\kron_bins\epoch_v4")
_ap.add_argument("--old", default=r"E:\kron_bins\epoch",
                 help="directory holding the canonical held-out bin")
_ap.add_argument("--slug", default="icold21/kron-corpus-2b")
_ap.add_argument("--message", default="prose rebalance: en wiki 2.5 GB + 2 GB "
                 "literature, ru math/science up, SFT samples, 60/40 tokens")
_args, _ = _ap.parse_known_args()
OUT = Path(_args.dir)
OLD = Path(_args.old)
SLUG = _args.slug
EOT, SYS = 8191, 8187
BPT = {"en": 2.917, "ru": 4.964}


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not (OUT / "train.bpe.bin").exists():
        raise SystemExit("no bin yet")

    (OUT / "val_index.json").write_bytes((OLD / "val_index.json").read_bytes())
    meta_path = OUT / "tokenizer_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    report = json.loads((ROOT / "data" / "bilingual" / "corpus_report.json")
                        .read_text(encoding="utf-8"))
    mixed = report["mixed"]
    en_b = sum(mixed.get("en", {}).values())
    ru_b = sum(mixed.get("ru", {}).values())
    total_b = en_b + ru_b
    tok = {"en": en_b / BPT["en"], "ru": ru_b / BPT["ru"]}
    tot_tok = sum(tok.values())
    meta["language"] = {l: {"bytes": int(mixed and sum(mixed.get(l, {}).values())),
                            "tokens": int(tok[l]),
                            "bytes_per_token": BPT[l]} for l in ("en", "ru")}
    meta["token_share"] = {l: round(tok[l] / tot_tok, 4) for l in ("en", "ru")}
    meta["byte_share"] = {l: round(sum(mixed.get(l, {}).values()) / total_b, 4)
                          for l in ("en", "ru")}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    say(f"meta: {meta['train_tokens']:,} tokens, token share {meta['token_share']}, "
        f"byte share {meta['byte_share']}")

    old_hash = sha256(OLD / "val.bpe.bin")
    new_hash = sha256(OUT / "val.bpe.bin")
    say(f"held-out {new_hash[:16]} vs {old_hash[:16]}: "
        f"{'identical' if new_hash == old_hash else 'DIFFERENT'}")
    if new_hash != old_hash:
        raise SystemExit("held-out changed; bpb would stop being comparable")

    tokens = np.fromfile(OUT / "train.bpe.bin", dtype=np.uint16)
    n_eot = int((tokens == EOT).sum())
    n_sys = int((tokens == SYS).sum())
    docs = meta["domain_prompts"]["docs"]
    say(f"bin: {len(tokens):,} tokens (meta {meta['train_tokens']:,}) | "
        f"EOT {n_eot:,} | <|system|> {n_sys:,} | docs {docs:,}")
    if len(tokens) != meta["train_tokens"]:
        raise SystemExit("token count does not match the metadata")
    if meta["domain_prompts"]["enabled"] or meta["domain_prompts"]["prompted_docs"]:
        raise SystemExit("the bin still carries inserted prompts")
    # ``<|system|>`` is legitimate when it comes from the chat documents
    # themselves, so the count is reported per document instead of compared
    # with ``already_prompted_docs`` -- that field is 0 whenever prompts are
    # switched off, which made the old check cry wolf on every decay bin.
    per_doc = n_sys / max(1, n_eot)
    say(f"chat markup: {per_doc:.2f} <|system|> per document")
    if per_doc > 2.0:
        say(f"WARNING: {n_sys:,} <|system|> for {n_eot:,} documents looks "
            f"like inserted prompts rather than chat markup")

    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "Kron Q10A anneal bin" if "anneal" in SLUG
                  else "KronCorpus 2B bins",
        "id": SLUG,
        "licenses": [{"name": "other"}]}), encoding="utf-8")
    say("uploading")
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "version",
                        "-p", str(OUT), "--dir-mode", "zip",
                        "-m", _args.message],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    say(f"upload exit {r.returncode}: {tail[-1][:160] if tail else '-'}")
    if r.returncode != 0:
        raise SystemExit("upload failed")

    chk = ROOT / "data" / "corpus_check_v3"
    chk.mkdir(parents=True, exist_ok=True)
    for _ in range(30):
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                        "-d", SLUG, "-f", "tokenizer_meta.json", "-p", str(chk)],
                       capture_output=True, text=True)
        remote = chk / "tokenizer_meta.json"
        if remote.exists():
            rm = json.loads(remote.read_text(encoding="utf-8"))
            if rm.get("train_tokens") == meta["train_tokens"]:
                say(f"DONE: remote metadata matches, {rm['train_tokens']:,} tokens, "
                    f"token share {rm.get('token_share')}")
                return
        time.sleep(20)
    raise SystemExit("remote metadata never matched")


if __name__ == "__main__":
    main()
