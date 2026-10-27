"""Stage the ``kronllm`` package as a Kaggle dataset directory.

Kaggle kernels mount datasets, so the training code travels as a zip next to a
``dataset-metadata.json``.  The zip keeps the package rooted at ``kronllm/`` so
the notebook can simply ``sys.path.insert(0, CODE)``.

Versioning is explicit: the dataset id changes with the code (``...-code-v13``)
instead of uploading a new version of the same id.  A new id cannot be silently
shadowed by a run that already mounted the old one, which is the failure mode
that matters here: a session resumes from a checkpoint written by different
code.

Usage::

    python scripts/make_code_dataset.py --version 13
    python -m kaggle datasets create -p data/kaggle_code_v13
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    args = ap.parse_args()

    out_dir = ROOT / "data" / f"kaggle_code_v{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "kron-code.zip"

    files = sorted(p for p in (ROOT / "kronllm").rglob("*.py")
                   if "__pycache__" not in p.parts)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(ROOT).as_posix())
        # The val bucket index (offset of every language/domain slice inside
        # val.bpe.bin) is tiny and belongs with the code that consumes it; the
        # 4.5 GB corpus dataset does not need a new version for one JSON file.
        extra = ROOT / "data" / "bilingual" / "val_index.json"
        if extra.exists():
            z.write(extra, "val_index.json")

    meta = {
        "title": f"KronLLM code v{args.version}",
        "id": f"icold21/kron-llm-code-v{args.version}",
        "licenses": [{"name": "other"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"{zip_path} ({len(files)} files, {zip_path.stat().st_size/1e3:.0f} kB)")
    print(f"{out_dir/'dataset-metadata.json'}")


if __name__ == "__main__":
    main()
