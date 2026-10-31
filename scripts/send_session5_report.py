"""Send the already-rendered session-5 report."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIL = (Path.home() / ".config" / "opencode" / "skills" / "email-control"
        / "scripts" / "mail.py")
HTML = ROOT / "data" / "report_session5.html"

cmd = [sys.executable, str(MAIL), "send",
       "--subject", "Q10: сессия завершена — шаг 36 549, val_bpb_mixed 1.1828",
       "--model", "DeepSeek V4.1 Flash Free", "--session", "kron_chain",
       "--html", "--body-file", str(HTML)]
for attempt in range(1, 4):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    print(f"attempt {attempt}: exit {r.returncode}")
    if r.returncode == 0:
        print((r.stdout or "").strip()[-300:])
        break
    print((r.stderr or r.stdout or "").strip()[-300:])
else:
    raise SystemExit(1)
