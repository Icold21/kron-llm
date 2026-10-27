"""Fetch a Kaggle notebook session log (live or persisted).

``kaggle kernels output`` only returns files, so the traceback of a failed run
is invisible from the CLI.  The API exposes the session log as a Server-Sent
Events stream that stays open until the session ends; this reads it on a
background thread and prints whatever arrived within the time budget, so it
never hangs.

Usage::

    python scripts/kaggle_log.py              # latest version, 20 s window
    python scripts/kaggle_log.py 1 60 12000   # version 1, 60 s, last 12k chars
"""
import json
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types import kernels_api_service as kas

OWNER = "icold21"
SLUG = "kron-q10-10m-bilingual-pretraining"


def _decode(payload: str) -> str:
    out = []
    for line in str(payload).splitlines():
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        if isinstance(ev, dict) and "data" in ev:
            out.append(str(ev["data"]).rstrip("\n"))
    return "\n".join(out)


def fetch(version: str | None, seconds: int, tail: int) -> str:
    buf: list[str] = []

    def worker() -> None:
        api = KaggleApi()
        api.authenticate()
        with api.build_kaggle_client() as k:
            req = kas.ApiGetKernelSessionLogsStreamRequest()
            req.user_name = OWNER
            req.kernel_slug = SLUG
            if version:
                req.version_label = api._kernel_version_label(version)
            resp = k.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
            if hasattr(resp, "iter_lines"):
                for raw in resp.iter_lines(decode_unicode=True):
                    if raw is None:
                        continue
                    text = _decode(raw)
                    if text:
                        buf.append(text)
                    if "END_OF_LOG" in str(raw):
                        break
            else:
                data = resp.read() if hasattr(resp, "read") else getattr(resp, "text", "")
                buf.append(data.decode("utf-8", "replace")
                           if isinstance(data, bytes) else str(data))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(seconds)
    return "\n".join(buf)[-tail:]


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else None
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tail = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    text = fetch(ver or None, secs, tail)
    print(text if text.strip() else "[log] no data (session may not have started)")
