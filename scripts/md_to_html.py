"""Render the small Markdown subset our reports use as email-safe HTML.

Mail clients ignore linked stylesheets and strip most CSS, so everything here is
inline and boring on purpose: bordered tables, a monospace block, tight
paragraphs.  The goal is that a table of steps and bpb is readable in a phone
mail client, which is where these reports are read.

Supported: headings, tables, fenced code, bullet and numbered lists, block
quotes, bold/italic/inline code, links, and paragraph text.  Anything else is
passed through as text.

Usage::

    python scripts/md_to_html.py data/report_q10.md data/report_q10.html
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

BODY = "font-family:-apple-system,Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.5;color:#1b1b1b"
TABLE = ("border-collapse:collapse;margin:12px 0;font-size:13px;"
         "font-family:-apple-system,Segoe UI,Arial,sans-serif")
CELL = "border:1px solid #cfcfcf;padding:5px 9px;text-align:left"
HEAD = "border:1px solid #cfcfcf;padding:5px 9px;background:#f2f4f7;text-align:left"
PRE = ("background:#f6f8fa;border:1px solid #e1e4e8;padding:10px;"
       "overflow-x:auto;font-size:12px;font-family:Consolas,Menlo,monospace")
CODE = "background:#f2f4f7;padding:1px 4px;font-family:Consolas,Menlo,monospace;font-size:12px"


def inline(text: str) -> str:
    """Escape, then re-apply the few inline things we actually use."""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", rf'<code style="{CODE}">\1</code>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                 r'<a href="\2" style="color:#0b62c4">\1</a>', out)
    return out


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append(f'<pre style="{PRE}">'
                       + html.escape("\n".join(block), quote=False) + "</pre>")
            continue

        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            size = {1: 20, 2: 17, 3: 15}.get(level, 14)
            out.append(f'<h{min(level, 4)} style="font-size:{size}px;'
                       f'margin:18px 0 6px">{inline(text)}</h{min(level, 4)}>')
            i += 1
            continue

        if is_table_row(line) and i + 1 < len(lines) and is_table_row(lines[i + 1]) \
                and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            header = split_row(line)
            i += 2
            rows = []
            while i < len(lines) and is_table_row(lines[i]):
                rows.append(split_row(lines[i]))
                i += 1
            parts = [f'<table style="{TABLE}"><thead><tr>']
            parts += [f'<th style="{HEAD}">{inline(h)}</th>' for h in header]
            parts.append("</tr></thead><tbody>")
            for row in rows:
                parts.append("<tr>" + "".join(
                    f'<td style="{CELL}">{inline(c)}</td>' for c in row) + "</tr>")
            parts.append("</tbody></table>")
            out.append("".join(parts))
            continue

        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append('<ul style="margin:8px 0 8px 22px;padding:0">'
                       + "".join(f"<li style=\"margin:3px 0\">{inline(x)}</li>"
                                 for x in items) + "</ul>")
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            out.append('<ol style="margin:8px 0 8px 22px;padding:0">'
                       + "".join(f"<li style=\"margin:3px 0\">{inline(x)}</li>"
                                 for x in items) + "</ol>")
            continue

        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip(">").strip())
                i += 1
            out.append('<blockquote style="margin:10px 0;padding:8px 12px;'
                       'border-left:3px solid #cfcfcf;color:#444">'
                       + "<br>".join(inline(q) for q in quote) + "</blockquote>")
            continue

        if not line.strip():
            i += 1
            continue

        para = []
        while i < len(lines) and lines[i].strip() and not (
                lines[i].startswith(("#", "```", ">"))
                or is_table_row(lines[i])
                or re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])):
            para.append(lines[i].strip())
            i += 1
        out.append(f'<p style="margin:8px 0">{inline(" ".join(para))}</p>')

    return f'<div style="{BODY}">' + "\n".join(out) + "</div>"


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".html")
    dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
