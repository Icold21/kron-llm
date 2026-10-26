"""Fetch a Kaggle notebook session log (live or persisted).

``kaggle kernels output`` only returns files, so the traceback of a failed run
is invisible from the CLI.  The API exposes the session log stream; this wraps
it so the last N characters can be read from a script or the chain.

Usage::

    python scripts/kaggle_log.py            # latest version
    python scripts/kaggle_log.py 1 12000    # version 1, last 12k chars
"""
import sys
import warnings

warnings.filterwarnings("ignore")

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types import kernels_api_service as kas

OWNER = "icold21"
SLUG = "kron-q10-10m-bilingual-pretraining"


def fetch(version: str | None, tail: int) -> str:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as k:
        req = kas.ApiGetKernelSessionLogsStreamRequest()
        req.user_name = OWNER
        req.kernel_slug = SLUG
        if version:
            req.version_label = api._kernel_version_label(version)
        resp = k.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
        if hasattr(resp, "text"):
            text = resp.text
        else:
            data = resp.read()
            text = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    return text[-tail:]


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else None
    tail = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    print(fetch(ver, tail))
