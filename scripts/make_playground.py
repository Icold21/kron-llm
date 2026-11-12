"""Generate ``notebooks/kron_playground.ipynb`` from readable source.

The notebook is long and mostly code, which is unpleasant to edit directly as
JSON.  This builder keeps the cells as Python strings, so changes are reviewable
in a diff.  Regenerate with::

    python scripts/make_playground.py

The resulting notebook has three kinds of cell: the standard benchmarks table,
our own tests, and interactive panels (generation, chat, arithmetic drill,
tokenizer).  Everything heavy is importable from the repo, so the notebook holds
no logic that the scripts do not already own.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "notebooks" / "kron_playground.ipynb"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md(r"""
# KronLLM Q10 — playground

**Q10** — bilingual EN/RU, **9,938,886 parameters**, 8K BPE, 25 Kronecker layers.

Two checkpoints are loaded side by side, which is the point of the notebook:

| | |
|---|---|
| **S10** (fine-tuned) | step 4 521, the supervised pass on 78.6M tokens of math / code / chat |
| **Q10A** (pretrained) | step 109 507, the end of the decay pass — the weights S10 started from |

Sections: standard benchmarks, our own tests, and four interactive panels
(generation, chat, arithmetic drill, tokenizer).

Run the cells top to bottom.  The first one takes ~20 s to load the models;
everything after that is interactive.
""")

code(r"""
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path

import torch

ROOT = Path(r"C:\Users\Icold\Desktop\kron_llm")
sys.path.insert(0, str(ROOT))

from kronllm.model import KronTransformer
from kronllm.tokenizer import BPETokenizer

CKPTS = {
    "S10  (fine-tuned, step 4 521)":  Path(r"E:/kron_out/s10_ckpt/last.pt"),
    "Q10A (decay end, step 109 507)": Path(r"E:/kron_out/q10_ckpt/last.pt"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")


def _module(name: str, rel: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MODELS: dict[str, tuple] = {}


def load(which: str):
    # Loaded once per checkpoint: 40 MB of weights each, so both fit at once.
    if which not in MODELS:
        path = CKPTS[which]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing -- download it with\n"
                f"  python -m kaggle datasets download -d icold21/kron-s10-ckpt -f last.pt -p E:/kron_out/s10_ckpt --unzip")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = KronTransformer(state["cfg"]).to(DEVICE)
        model.load_state_dict(state["model"])
        model.eval()
        tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                           state["cfg"]["data"]["vocab_size"],
                           bytes_per_token=3.69,
                           eot_id=state["cfg"]["data"]["eot_id"])
        MODELS[which] = (model, tok)
        print(f"loaded {which}: step {int(state.get('step') or 0):,}")
    return MODELS[which]


model, tok = load("S10  (fine-tuned, step 4 521)")
print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
""")

md(r"""
## 1. Standard benchmarks

Zero-shot multiple choice, scored the way `lm-evaluation-harness` scores it by
default: the summed log-probability of each continuation, argmax.

Read these as a floor, not as a verdict.  These are English benchmarks built for
models a thousand times larger; at 9.9M parameters the interesting column is the
**distance above chance**, and the row that matters most is `lambada`, because it
is the only one that measures *generation* rather than ranking.

The numbers come from `scripts/bench_current.py`, run on both checkpoints with the
same 200 items (`data/bench_s10.json`, `data/bench_q10.json`).
""")

code(r"""
import pandas as pd

rows = []
for label, path in (("S10 (fine-tuned)", ROOT / "data" / "bench_s10.json"),
                    ("Q10A (decay end)", ROOT / "data" / "bench_q10.json")):
    if not path.exists():
        print(f"{path} is missing -- run: python scripts/bench_current.py --ckpt ...")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    for task, res in data["bench"].items():
        if "error" in res:
            continue
        rows.append({"task": task, label: round(res["acc"], 3),
                     "n": res["n"], "chance": res["chance"]})

table = pd.DataFrame(rows)
if not table.empty:
    wide = table.pivot_table(index=["task", "chance", "n"], values=["S10 (fine-tuned)", "Q10A (decay end)"],
                             aggfunc="first").reset_index()
    wide["S10 vs chance"] = (wide["S10 (fine-tuned)"] - wide["chance"]).round(3)
    wide["Q10A vs chance"] = (wide["Q10A (decay end)"] - wide["chance"]).round(3)
    display(wide.sort_values("task").reset_index(drop=True))
""")

code(r"""
import matplotlib.pyplot as plt

if not table.empty:
    pivot = table.pivot_table(index="task", columns="chance", values=["S10 (fine-tuned)", "Q10A (decay end)"])
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=130)
    tasks = sorted(table["task"].unique())
    x = range(len(tasks))
    for i, (label, color) in enumerate((("S10 (fine-tuned)", "#1f4e79"), ("Q10A (decay end)", "#b03a2e"))):
        vals = [float(table[(table.task == t)][label].iloc[0]) if not table[(table.task == t)].empty else 0
                for t in tasks]
        ax.bar([p + (i - 0.5) * 0.38 for p in x], vals, width=0.38, label=label, color=color)
    chance = [float(table[table.task == t]["chance"].iloc[0]) for t in tasks]
    ax.scatter(list(x), chance, marker="_", s=400, color="black", label="chance", zorder=5)
    ax.set_xticks(list(x)); ax.set_xticklabels(tasks, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); plt.show()
""")

md(r"""
## 2. Held-out bits per byte

The honest metric: evenly spaced 512-token blocks, bytes counted from the
tokenizer's own table, so it is comparable across corpora.

`pretrain_val` is the bilingual held-out bin that every pretraining checkpoint was
measured on (`a45bbb03`); `sft_val` is the SFT corpus's own split.  Two numbers,
because fine-tuning can move them in opposite directions.
""")

code(r"""
bpb_rows = []
for label, path in (("S10 (fine-tuned)", ROOT / "data" / "bench_s10.json"),
                    ("Q10A (decay end)", ROOT / "data" / "bench_q10.json")):
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        bpb_rows.append({"model": label, **{k: round(v, 4) for k, v in d["bpb"].items()}})
display(pd.DataFrame(bpb_rows))
""")

md(r"""
## 3. Our own tests

### 3.1 Column drills

The corpus teaches a written procedure — `units: 7 + 8 = 15 -> write 5, carry 1`
— and the score is the final `Answer: N`, not the format.  Broken down per
family, because addition with carries, subtraction with borrows and one-digit
multiplication are three different algorithms.
""")

code(r"""
drill_rows = []
for label, path in (("S10 (fine-tuned)", ROOT / "data" / "bench_s10.json"),
                    ("Q10A (decay end)", ROOT / "data" / "bench_q10.json")):
    if not path.exists():
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    for fam, res in d["our"]["drills"].items():
        drill_rows.append({"family": fam, "model": label,
                           "answer line": f"{res['ok']}/{res['n']}",
                           "first row": f"{res['first_row']}/{res['n']}"})
if drill_rows:
    display(pd.DataFrame(drill_rows).pivot(index="family", columns="model", values="answer line"))
""")

md(r"""
### 3.2 The bare form

`47 + 38 =` with no procedure.  This is the capability the model does not have:
its errors are close (`79 - 26 -> 54`) rather than random, which is what a model
that *can* compute but not *compress the procedure into one step* looks like.
""")

code(r"""
for label, path in (("S10 (fine-tuned)", ROOT / "data" / "bench_s10.json"),
                    ("Q10A (decay end)", ROOT / "data" / "bench_q10.json")):
    if not path.exists():
        continue
    bare = json.loads(path.read_text(encoding="utf-8"))["our"]["bare"]
    print(f"\n{label}: {bare['ok']}/{bare['n']} exact")
    for ex in bare["examples"]:
        mark = "ok " if ex["ok"] else "   "
        print(f"  {mark}{ex['q']:>16} -> {ex['got']:<8} (want {ex['want']})")
""")

md(r"""
### 3.3 Chat

No reference answers here — the transcripts are the result.  Format without
content is the failure mode to look for: an answer that repeats the question, or
that is grammatical and unrelated.
""")

code(r"""
for label, path in (("S10 (fine-tuned)", ROOT / "data" / "bench_s10.json"),):
    if not path.exists():
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    for turn in d["our"]["chat"]:
        print(f"\n[{label}] user: {turn['user']}")
        print(f"{' ' * (len(label) + 3)}bot : {turn['answer'].strip()[:300]}")
""")

md(r"""
### 3.4 Free generation

Distinct-1 / distinct-2 and the share of repeated 4-grams over four prompts, so
"it reads badly" becomes a number we can watch across checkpoints.
""")

code(r"""
gen_rows = []
for label, path in (("S10 (fine-tuned)", ROOT / "data" / "bench_s10.json"),
                    ("Q10A (decay end)", ROOT / "data" / "bench_q10.json")):
    if not path.exists():
        continue
    r = json.loads(path.read_text(encoding="utf-8"))["our"]["repeat"]
    gen_rows.append({"model": label, **{k: round(v, 3) for k, v in r.items()}})
if gen_rows:
    display(pd.DataFrame(gen_rows))

samples = json.loads((ROOT / "data" / "bench_s10.json").read_text(encoding="utf-8"))["our"]["generation"]
for s in samples:
    print("\n>>>", s["prompt"])
    print(s["text"].strip()[:400])
""")

md(r"""
## 4. Interactive panels

### 4.1 Generation

`temperature`, `top_k`, `top_p` and the repetition penalty all matter at this
size: the tuned setting from the reports (T=0.7, top_k=40, top_p=0.95, penalty
1.2) is the least repetitive of the six that were swept.
""")

code(r"""
import ipywidgets as W
from IPython.display import clear_output, display

which = W.Dropdown(options=list(CKPTS), value="S10  (fine-tuned, step 4 521)", description="model")
prompt = W.Textarea(value="Once upon a time, in a small village by the sea,",
                    layout=W.Layout(width="100%", height="80px"))
temperature = W.FloatSlider(value=0.7, min=0.1, max=1.5, step=0.05, description="T")
top_k = W.IntSlider(value=40, min=1, max=200, description="top_k")
top_p = W.FloatSlider(value=0.95, min=0.1, max=1.0, step=0.01, description="top_p")
penalty = W.FloatSlider(value=1.2, min=1.0, max=1.6, step=0.05, description="penalty")
new_tokens = W.IntSlider(value=80, min=10, max=300, step=10, description="tokens")
greedy = W.Checkbox(value=False, description="greedy")
run = W.Button(description="Generate", button_style="primary")
out = W.Output()


def on_generate(_btn=None):
    with out:
        clear_output()
        m, t = load(which.value)
        ids = torch.tensor([t.encode(prompt.value)], device=DEVICE)
        with torch.no_grad():
            gen = m.generate(ids, max_new_tokens=new_tokens.value,
                             temperature=0.0 if greedy.value else temperature.value,
                             top_k=1 if greedy.value else top_k.value,
                             top_p=1.0 if greedy.value else top_p.value,
                             repetition_penalty=penalty.value)
        print(t.decode(gen[0, ids.shape[1]:].tolist()))


run.on_click(on_generate)
display(W.VBox([which, prompt, W.HBox([temperature, greedy]), W.HBox([top_k, top_p]),
                W.HBox([penalty, new_tokens]), run, out]))
""")

md(r"""
### 4.2 Chat

The panel writes the turns the corpus teaches (`<|system|>…<|eot|><|user|>…<|eot|><|assistant|>`)
and keeps the transcript, so a second question sees the first answer — which is
where a small model usually falls apart.
""")

code(r"""
chat_model = W.Dropdown(options=list(CKPTS), value="S10  (fine-tuned, step 4 521)", description="model")
system_box = W.Textarea(value="You are KronLLM, a compact bilingual model.",
                        layout=W.Layout(width="100%", height="60px"))
user_box = W.Textarea(value="Who are you?", layout=W.Layout(width="100%", height="60px"))
send = W.Button(description="Send", button_style="primary")
reset = W.Button(description="Reset")
chat_out = W.Output()
TRANSCRIPT: list[tuple[str, str]] = []


def render_chat():
    with chat_out:
        clear_output()
        for who, text in TRANSCRIPT:
            print(f"{who}: {text}")


def on_send(_btn=None):
    global TRANSCRIPT
    m, t = load(chat_model.value)
    TRANSCRIPT.append(("user", user_box.value))
    text = f"<|system|>{system_box.value}<|eot|>"
    for who, said in TRANSCRIPT:
        text += f"<|{who}|>{said}<|eot|>"
    text += "<|assistant|>"
    ids = torch.tensor([t.encode(text)], device=DEVICE)
    with torch.no_grad():
        gen = m.generate(ids, max_new_tokens=96, temperature=0.7, top_k=40,
                         top_p=0.95, repetition_penalty=1.2, stop_at_eot=True,
                         eot_id=t.eot_id)
    answer = t.decode(gen[0, ids.shape[1]:].tolist()).split("<|eot|>")[0]
    TRANSCRIPT.append(("assistant", answer.strip()))
    render_chat()


def on_reset(_btn=None):
    global TRANSCRIPT
    TRANSCRIPT = []
    render_chat()


send.on_click(on_send)
reset.on_click(on_reset)
display(W.VBox([chat_model, system_box, user_box, W.HBox([send, reset]), chat_out]))
""")

md(r"""
### 4.3 Arithmetic drill

Type two numbers and watch the procedure the model writes.  The final line is
compared with Python, so the panel says *right* or *wrong* and not just "looks
plausible".  This is the one task where the fine-tune clearly bought something.
""")

code(r"""
build_sft = _module("build_sft_play", "scripts/build_sft.py")

drill_model = W.Dropdown(options=list(CKPTS), value="S10  (fine-tuned, step 4 521)", description="model")
a_box = W.IntText(value=583, description="a")
b_box = W.IntText(value=693, description="b")
op_box = W.Dropdown(options=["+", "-", "*"], value="+", description="op")
drill_run = W.Button(description="Run drill", button_style="primary")
drill_out = W.Output()


def question_for(a: int, b: int, op: str) -> str:
    if op == "+":
        return f"Add {a} and {b} column by column."
    if op == "-":
        return f"What is {a} - {b}?"
    return f"Multiply {a} by {b} digit by digit."


def on_drill(_btn=None):
    with drill_out:
        clear_output()
        m, t = load(drill_model.value)
        q = question_for(a_box.value, b_box.value, op_box.value)
        _, prompt_text = build_sft.render([
            {"role": "system", "content": build_sft.SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": "x"}])
        ids = torch.tensor([t.encode(prompt_text)], device=DEVICE)
        with torch.no_grad():
            gen = m.generate(ids, max_new_tokens=200, temperature=0.0, top_k=1,
                             top_p=1.0, repetition_penalty=1.0)
        text = t.decode(gen[0, ids.shape[1]:].tolist())
        print(q, "\n")
        print(text.strip()[:900])
        want = {"+": a_box.value + b_box.value, "-": a_box.value - b_box.value,
                "*": a_box.value * b_box.value}[op_box.value]
        m2 = re.search(r"(?:Answer|Ответ):\s*(-?\d+)", text)
        got = int(m2.group(1)) if m2 else None
        print(f"\nPython says {want}; model said {got} -> "
              f"{'RIGHT' if got == want else 'WRONG'}")


drill_run.on_click(on_drill)
display(W.VBox([drill_model, W.HBox([a_box, b_box, op_box]), drill_run, drill_out]))
""")

md(r"""
### 4.4 Tokenizer

8K BPE over raw bytes, so every string round-trips.  Useful when a sample looks
broken: often it is the segmentation, not the model.
""")

code(r"""
tok_model = W.Dropdown(options=list(CKPTS), value="S10  (fine-tuned, step 4 521)", description="model")
tok_box = W.Textarea(value="Привет, мир! def add(a, b): return a + b",
                     layout=W.Layout(width="100%", height="60px"))
tok_out = W.Output()


def on_tok(_change=None):
    with tok_out:
        clear_output()
        _, t = load(tok_model.value)
        ids = t.encode(tok_box.value)
        print(f"{len(ids)} tokens, {len(tok_box.value.encode('utf-8'))} bytes, "
              f"{len(tok_box.value.encode('utf-8')) / max(1, len(ids)):.2f} bytes/token")
        print(ids[:80])
        print(repr(t.decode(ids)))


tok_box.observe(on_tok, names="value")
display(W.VBox([tok_model, tok_box, tok_out]))
on_tok()
""")

md(r"""
## 5. Re-run the battery

The numbers above are read from JSON, so re-running is a subprocess and a reload.
`--n 200 --blocks 512` takes about ten minutes on a 1660 Ti; drop to `--n 50
--blocks 64` for a quick look.
""")

code(r"""
import subprocess

RUN_N = 200
RUN_BLOCKS = 512
ckpt_path = CKPTS["S10  (fine-tuned, step 4 521)"]
out_json = ROOT / "data" / "bench_s10.json"
cmd = [sys.executable, "-u", str(ROOT / "scripts" / "bench_current.py"),
       "--ckpt", str(ckpt_path), "--n", str(RUN_N), "--blocks", str(RUN_BLOCKS),
       "--out", str(out_json)]
print(" ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                        errors="replace")
print(result.stdout[-3000:])
print(result.stderr[-1500:])
""")

md(r"""
## 6. The training curve

`val_bpb_mixed` across the whole pretraining run, with the epoch pass stopped at
58,213 steps and the decay pass down to 1.096.  The dotted part is the legacy
metric rescaled by 1.444, before the honest one existed.
""")

code(r"""
from IPython.display import Image, display as show

for name in ("fig_pretrain_curve.png", "curve_q10.png"):
    p = ROOT / "data" / name
    if p.exists():
        show(Image(filename=str(p)))
""")

md(r"""
## 7. Where this leaves the model

| | |
|---|---|
| **Can** | continue text bilingually at ~1.10 bpb (GPT-2-124M: 1.554); run column arithmetic with carries and borrows (28/30); write the chat markup |
| **Cannot** | answer `47 + 38 =` in one step (2/40); hold a chat persona; rank on any standard benchmark above chance |
| **Why** | the procedure lives in the context, not in the weights: 9.9M parameters carry the algorithm but not the compressed result |

The next levers, in order of expected value: more SFT steps (S10b, 6.3 h), then a
19.5M-parameter model (depth is cheap: 25 → 56 layers fits a T4 at ~9.8 GB).
""")

nb = {
    "cells": [
        {"id": f"cell{i:02d}",
         "cell_type": kind,
         "metadata": {},
         **({"source": text.splitlines(keepends=True)} if kind == "markdown"
            else {"source": text.splitlines(keepends=True),
                  "execution_count": None, "outputs": []})}
        for i, (kind, text) in enumerate(CELLS)
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{DST}  ({len(CELLS)} cells)")
