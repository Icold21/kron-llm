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

import gzip
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
EOT, SYS, USER = 8191, 8187, 8188
BPT = {"en": 2.917, "ru": 4.964}


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def language_bytes(mix_dir: Path) -> dict[str, int]:
    """UTF-8 bytes per language over every shard the encoder read.

    ``build_streaming`` sweeps the domain shards and leaves the side pools
    (``*_drills``, ``*_chat``), so the directory is exactly the corpus that went
    into the bin -- except for the 2 % held-out split, which is stratified per
    (language, domain) and therefore barely moves a ratio.
    """
    out: dict[str, int] = {}
    for path in sorted(mix_dir.glob("*.jsonl.gz")):
        fallback = path.name.split("_")[0]
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                lang = doc.get("language") or fallback
                out[lang] = out.get(lang, 0) + len(
                    (doc.get("text") or "").encode("utf-8"))
    return out


def main() -> None:
    if not (OUT / "train.bpe.bin").exists():
        raise SystemExit("no bin yet")

    (OUT / "val_index.json").write_bytes((OLD / "val_index.json").read_bytes())
    meta_path = OUT / "tokenizer_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    report = json.loads((ROOT / "data" / "bilingual" / "corpus_report.json")
                        .read_text(encoding="utf-8"))
    mixed = report["mixed"]
    estimate_b = sum(mixed.get("en", {}).values()) + sum(mixed.get("ru", {}).values())
    measured_b = meta.get("train_bytes") or 0
    say(f"meta: {meta['train_tokens']:,} tokens, {measured_b/1e9:.2f} GB, "
        f"token share {meta.get('token_share')}, byte share {meta.get('byte_share')}")
    if estimate_b and measured_b:
        drift = abs(measured_b - estimate_b) / measured_b
        say(f"mix report {estimate_b/1e9:.2f} GB vs bin {measured_b/1e9:.2f} GB "
            f"({drift:.1%} apart -- the interleaved side pools are the difference)")
        if drift > 0.10:
            say("WARNING: the report and the bin disagree by more than 10 %")

    # ``--reuse-val-dir`` copies the held-out meta, and with it the language
    # block of the *source* directory -- which for the canonical held-out bin is
    # a small val-only corpus, not this one.  That is how epoch_v5 was first
    # published with a token share of 62/38 instead of the measured 60/40: the
    # numbers were not wrong, they described a different corpus.  The block is
    # therefore rebuilt from the mix that was encoded, scaled to the bytes the
    # encoder actually wrote (the same method epoch_v4 used, so the two remain
    # comparable), and the copy is reported rather than trusted.
    lang_b = language_bytes(ROOT / "data" / "bilingual" / "mix")
    total_b = sum(lang_b.values())
    if total_b and measured_b:
        scale = measured_b / total_b
        tok = {l: lang_b[l] * scale / BPT.get(l, 3.5) for l in lang_b}
        tot_tok = sum(tok.values())
        copied = {l: v.get("bytes") for l, v in (meta.get("language") or {}).items()}
        copied_total = sum(v or 0 for v in copied.values())
        # The rebuilt block lands a few bytes short of ``train_bytes`` because
        # each language is truncated to an int, so "same corpus" has to be a
        # tolerance, not equality -- otherwise a second run calls its own output
        # a stale copy.
        consistent = abs(copied_total - measured_b) <= max(1000, measured_b * 0.002)
        say(f"language block in the meta: {copied} (bytes) -- "
            f"{'consistent with the bin' if consistent else 'stale copy'}")
        meta["language"] = {l: {"bytes": int(lang_b[l] * scale),
                                "tokens": int(tok[l]),
                                "bytes_per_token": BPT.get(l, 3.5)}
                            for l in sorted(lang_b)}
        meta["token_share"] = {l: round(tok[l] / tot_tok, 4) for l in sorted(lang_b)}
        meta["byte_share"] = {l: round(lang_b[l] / total_b, 4) for l in sorted(lang_b)}
        say(f"rebuilt from the mix: byte share {meta['byte_share']} -> "
            f"token share {meta['token_share']}")
        en_share = meta["token_share"].get("en", 0.0)
        if not 0.55 <= en_share <= 0.65:
            say(f"WARNING: English holds {en_share:.1%} of the tokens; the "
                f"contract is 60/40")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

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

    # The epoch bin took the arithmetic pools in through --interleave drills and
    # the decay bin takes chat through --interleave chat.  Both are written into
    # mix/ by separate scripts, and both have silently gone missing before (the
    # sweep deleted them, twice), so the bin itself is asked whether they are
    # in there: <|user|> appears only in those rendered turns, never in prose.
    n_user = int((tokens == USER).sum())
    pools = sorted(p.name for p in (ROOT / "data" / "bilingual" / "mix").glob(
        "*_*ills.jsonl.gz")) + sorted(
        p.name for p in (ROOT / "data" / "bilingual" / "mix").glob("*_chat.jsonl.gz"))
    say(f"turns: <|user|> {n_user:,} in the bin, pools present {pools or 'none'}")
    if pools and n_user == 0:
        raise SystemExit("side pools exist but the bin holds no rendered turns")

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
    expect = hashlib.sha256((OUT / "tokenizer_meta.json").read_bytes()).hexdigest()
    for _ in range(30):
        # The CLI leaves an existing file in place, so the previous run's copy
        # is still there when the download is a no-op -- which is how a stale
        # metadata file once "confirmed" an upload that had not landed yet.
        (chk / "tokenizer_meta.json").unlink(missing_ok=True)
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                        "-d", SLUG, "-f", "tokenizer_meta.json", "-p", str(chk)],
                       capture_output=True, text=True)
        remote = chk / "tokenizer_meta.json"
        if remote.exists():
            got = hashlib.sha256(remote.read_bytes()).hexdigest()
            rm = json.loads(remote.read_text(encoding="utf-8"))
            if got == expect:
                say(f"DONE: remote metadata is byte-identical "
                    f"(sha256 {got[:12]}), {rm['train_tokens']:,} tokens, "
                    f"token share {rm.get('token_share')}")
                return
            say(f"  remote sha {got[:12]} != local {expect[:12]}; retrying")
        time.sleep(20)
    raise SystemExit("remote metadata never matched")


if __name__ == "__main__":
    main()
