"""Rebuild the epoch corpus with the new mix, verify it, upload it.

Plan, in order:

1. ``build_bilingual build`` with per-language byte budgets that equal 60/40 in
   *tokens* (EN 4110 MB at 2.917 B/token, RU 4660 MB at 4.964);
2. ``prepare_bilingual_bin`` reusing the same held-out bin, so ``val.bpe.bin``
   stays byte-identical and every historical bpb remains comparable;
3. checks that gate the upload: the held-out hash, the token count, and the
   number of ``<|system|>`` (only the chat documents may carry one);
4. ``datasets version``, then read the metadata back from Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(r"E:\kron_bins\epoch_v2")
METRICS = ROOT / "data" / "rebuild_v2.log"
HELDOUT_DIR = Path(r"E:\kron_bins\epoch")
VAL_INDEX = HELDOUT_DIR / "val_index.json"
SLUG = "icold21/kron-corpus-2b"

BUDGET_EN_MB = 4110
BUDGET_RU_MB = 4660
EOT, SYS = 8191, 8187


def say(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with METRICS.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(argv: list[str]) -> int:
    say("+ " + " ".join(Path(a).name if a.startswith(str(ROOT)) else a
                        for a in argv))
    r = subprocess.run([sys.executable, "-u", *argv], cwd=ROOT)
    say(f"  exit {r.returncode}")
    return r.returncode


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    METRICS.write_text("", encoding="utf-8")

    # ------------------------------------------------------------- 1. the mix
    if run([str(ROOT / "scripts" / "build_bilingual.py"), "build",
            "--budget-en-mb", str(BUDGET_EN_MB),
            "--budget-ru-mb", str(BUDGET_RU_MB)]) != 0:
        raise SystemExit("build failed")

    report = json.loads((ROOT / "data" / "bilingual" / "corpus_report.json")
                        .read_text(encoding="utf-8"))
    mixed = report["mixed"]
    en_bytes = sum(mixed.get("en", {}).values())
    ru_bytes = sum(mixed.get("ru", {}).values())
    say(f"mix: EN {en_bytes/1e6:.0f} MB, RU {ru_bytes/1e6:.0f} MB, "
        f"RU byte share {report['ru_share']:.1%}")
    for lang in ("en", "ru"):
        parts = ", ".join(f"{d} {v/1e6:.0f} MB"
                          for d, v in sorted(mixed.get(lang, {}).items()))
        say(f"  {lang}: {parts}")
    if report.get("shortfalls"):
        say("shortfalls: " + "; ".join(report["shortfalls"]))

    # ------------------------------------------------------------- 2. the bin
    if run([str(ROOT / "scripts" / "prepare_bilingual_bin.py"),
            "--out-dir", str(OUT),
            "--reuse-tokenizer", str(ROOT / "data" / "bilingual" / "tokenizer.json"),
            "--reuse-val-dir", str(HELDOUT_DIR),
            "--phase", "epoch", "--no-domain-prompts"]) != 0:
        raise SystemExit("bin build failed")

    VAL_INDEX.copy(OUT / "val_index.json")
    meta_path = OUT / "tokenizer_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # The reused metadata describes the *old* mix.  Rewrite the composition
    # fields from this build so the numbers on the dataset match the data.
    old = json.loads((HELDOUT_DIR / "tokenizer_meta.json").read_text(encoding="utf-8"))
    bpt = {l: old["language"][l]["bytes_per_token"] for l in ("en", "ru")}
    tokens = {l: sum(mixed.get(l, {}).values()) / bpt[l] for l in ("en", "ru")}
    total = max(1.0, sum(tokens.values()))
    meta["token_share"] = {l: round(tokens[l] / total, 4) for l in ("en", "ru")}
    meta["byte_share"] = {l: round(sum(mixed.get(l, {}).values())
                                   / max(1.0, en_bytes + ru_bytes), 4)
                          for l in ("en", "ru")}
    meta["language"] = {l: {"bytes": int(sum(mixed.get(l, {}).values())),
                            "tokens": int(tokens[l]),
                            "bytes_per_token": bpt[l]} for l in ("en", "ru")}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    say(f"meta: token share {meta['token_share']}, byte share {meta['byte_share']}")

    # ----------------------------------------------------------- 3. the gates
    import numpy as np

    new_val = OUT / "val.bpe.bin"
    same = sha256(new_val) == sha256(HELDOUT_DIR / "val.bpe.bin")
    say(f"held-out identical: {same} ({sha256(new_val)[:16]})")
    if not same:
        raise SystemExit("the held-out bin changed; bpb would stop being comparable")

    tokens_arr = np.fromfile(OUT / "train.bpe.bin", dtype=np.uint16)
    n_eot = int((tokens_arr == EOT).sum())
    n_sys = int((tokens_arr == SYS).sum())
    docs = meta["domain_prompts"]["docs"]
    say(f"bin: {len(tokens_arr):,} tokens | EOT {n_eot:,} | <|system|> {n_sys:,} "
        f"| docs {docs:,}")
    if meta["domain_prompts"]["enabled"] or meta["domain_prompts"]["prompted_docs"]:
        raise SystemExit("the new bin still carries inserted prompts")
    if n_sys != meta["domain_prompts"]["already_prompted_docs"]:
        raise SystemExit(f"<|system|> {n_sys} does not match the chat documents "
                         f"{meta['domain_prompts']['already_prompted_docs']}")

    # ------------------------------------------------------------ 4. upload
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "KronCorpus 2B bins", "id": SLUG,
        "licenses": [{"name": "other"}]}), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "version",
                        "-p", str(OUT), "--dir-mode", "zip",
                        "-m", "prose rebalance: en/encyclopedic 2.5 GB, "
                              "en/literature 2 GB, ru tech up, 60/40 tokens"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    say(f"upload exit {r.returncode}")
    say("  " + ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-1][:200]
        if (r.stdout or r.stderr) else "  (no output)")
    if r.returncode != 0:
        raise SystemExit("upload failed")

    # ------------------------------------------------------------ 5. readback
    chk = ROOT / "data" / "corpus_check_v2"
    chk.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                        "-d", SLUG, "-f", "tokenizer_meta.json", "-p", str(chk)],
                       capture_output=True, text=True)
        remote = chk / "tokenizer_meta.json"
        if remote.exists():
            rm = json.loads(remote.read_text(encoding="utf-8"))
            if (rm.get("train_tokens") == meta["train_tokens"]
                    and rm.get("token_share") == meta["token_share"]):
                say(f"remote meta matches: {rm['train_tokens']:,} tokens, "
                    f"token share {rm['token_share']}")
                say("DONE")
                return
        time.sleep(20)
    raise SystemExit("remote metadata never matched the local one")


if __name__ == "__main__":
    main()
