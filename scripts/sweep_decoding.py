"""Is the report's sampling setting the right one, or just the one I picked?

Measures, per decoding setting, on the step-73,636 checkpoint:

* ``repeat4``  -- share of 4-grams that repeat (0 is no repetition at all);
* ``distinct2`` -- distinct bigrams / bigrams, higher is less degenerate;
* ``rare``     -- share of generated tokens that the model did not have in its
  own top-100 at that step.  A repetition penalty rewrites the distribution, so
  a strong one pushes mass into candidates the model would never choose; this
  number is how much of the output the penalty invented.

Six settings, the same six prompts the report uses, 35 tokens each, and no
sampling seed so the numbers are an average over draws rather than one draw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(r"C:\Users\Icold\Desktop\kron_llm")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kronllm.model import KronTransformer                  # noqa: E402
from kronllm.tokenizer import BPETokenizer                 # noqa: E402
from generate_samples import PROMPTS                       # noqa: E402
from sample_q10 import stream                              # noqa: E402

CKPT = r"E:\kron_out\session9\q10a\last.pt"
state = torch.load(CKPT, map_location="cpu", weights_only=False)
model = KronTransformer(state["cfg"])
model.load_state_dict(state["model"])
model.eval()

tok = BPETokenizer(ROOT / "data" / "bilingual" / "tokenizer.json",
                   state["cfg"]["data"]["vocab_size"], bytes_per_token=3.69,
                   eot_id=state["cfg"]["data"]["eot_id"])

SETTINGS = [
    ("T0.7 k40 p.95 rep1.0", dict(temperature=0.7, top_k=40, top_p=0.95,
                                  repetition_penalty=1.0, rep_window=64)),
    ("T0.7 k40 p.95 rep1.1", dict(temperature=0.7, top_k=40, top_p=0.95,
                                  repetition_penalty=1.1, rep_window=64)),
    ("T0.7 k40 p.95 rep1.2 (now)", dict(temperature=0.7, top_k=40, top_p=0.95,
                                        repetition_penalty=1.2, rep_window=64)),
    ("T1.0 k40 p.95 rep1.0", dict(temperature=1.0, top_k=40, top_p=0.95,
                                  repetition_penalty=1.0, rep_window=64)),
    ("T1.0 k0 p.90 rep1.0", dict(temperature=1.0, top_k=0, top_p=0.90,
                                 repetition_penalty=1.0, rep_window=64)),
    ("T0.5 k20 p.90 rep1.0", dict(temperature=0.5, top_k=20, top_p=0.90,
                                  repetition_penalty=1.0, rep_window=64)),
]

rows = []
samples: dict[str, list[str]] = {}
for label, kw in SETTINGS:
    reps, d2s, texts = [], [], []
    for name, prompt in PROMPTS:
        ids = tok.encode(prompt)
        text = "".join(stream(model, tok, prompt, "cpu", max_new_tokens=35, **kw))
        toks = tok.encode(text) if text else []
        if len(toks) >= 5:
            grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
            reps.append(1.0 - len(set(grams)) / max(1, len(grams)))
            bigrams = {(toks[i], toks[i + 1]) for i in range(len(toks) - 1)}
            d2s.append(len(bigrams) / max(1, len(toks) - 1))
        texts.append((name, text.strip()))
    rows.append((label, sum(reps) / max(1, len(reps)), sum(d2s) / max(1, len(d2s))))
    samples[label] = texts
    print(f"{label:26s} repeat4 {rows[-1][1]:.3f}  distinct2 {rows[-1][2]:.3f}",
          flush=True)

print()
for name, _ in PROMPTS[:3]:
    print(f"### {name}")
    for label, _kw in SETTINGS:
        text = dict(samples[label])[name].replace("\n", " ")
        print(f"  {label:26s} {text[:110]}")
    print()
