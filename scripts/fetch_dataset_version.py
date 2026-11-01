"""Download one file of a specific dataset version (the CLI has no -v flag).

Usage:
    python scripts/fetch_dataset_version.py icold21/kron-corpus-2b 3 train.bpe.bin E:\\kron_old_v3
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> None:
    slug, version, name, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    key = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    query = urllib.parse.urlencode({"datasetVersionNumber": version})
    url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}/{name}?{query}"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key["key"]})
    out.mkdir(parents=True, exist_ok=True)
    dst = out / name
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done/1e9:5.2f}/{total/1e9:.2f} GB", end="", flush=True)
    print(f"\nsaved {dst} {dst.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
