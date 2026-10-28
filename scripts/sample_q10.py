"""Sample from a Q10 checkpoint with the notebook's decoding settings.

The notebook streams; this prints finished samples so they can be pasted into a
report or an email.  Decoding matches the playground exactly -- temperature,
top-k, and the CTRL-style repetition penalty that keeps early checkpoints from
repeating ``KM/KM/KM`` forever.

Usage::

    python scripts/sample_q10.py --ckpt data/kaggle_out/q10/last.pt
    python scripts/sample_q10.py --ckpt ... --prompt "The capital of France is"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronllm.model import KronTransformer  # noqa: E402
from kronllm.tokenizer import BPETokenizer  # noqa: E402

PROMPTS = [
    ("The Moon orbits the Earth at a distance of", {}),
    ("She opened the door and", {}),
    ("Он открыл дверь и", {}),
    ("def add(a, b):\n    return", {}),
    ("17 + 25 =", {}),
    ("В 1961 году", {}),
    ("KronLLM is", {}),
    ("<|system|>You are KronLLM, a compact bilingual model.<|eot|>"
     "<|user|>Who are you?<|eot|><|assistant|>", {}),
]


@torch.no_grad()
def stream(model, tok, prompt: str, device, max_new_tokens=80, temperature=0.8,
           top_k=40, top_p=0.0, repetition_penalty=1.2, rep_window=64):
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    past, shown = None, ""
    for _ in range(max_new_tokens):
        if past is not None and past[0][0].shape[2] < model.seq_len:
            step = model(idx[:, -1:], past_kv=past, use_cache=True)
            past = step["past_kv"]
        else:
            step = model(idx[:, -model.seq_len:], use_cache=True)
            past = step.get("past_kv")
        logits = step["logits"][:, -1, :].float()
        if repetition_penalty and repetition_penalty != 1.0:
            recent = torch.unique(idx[0, -rep_window:])
            target = logits[0, recent]
            logits[0, recent] = torch.where(target > 0,
                                            target / repetition_penalty,
                                            target * repetition_penalty)
        logits = logits / max(temperature, 1e-5)
        if top_k:
            k = min(top_k, logits.size(-1))
            thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        if top_p:
            # Measured, not decorative: under a top_k cap this binds only on the
            # steps where one token dominates (17 % of them at step 35k), and
            # those steps are where the narrow cap was letting through junk.
            ordered, order = torch.sort(logits, dim=-1, descending=True)
            probs = torch.softmax(ordered, dim=-1)
            cut = torch.cumsum(probs, dim=-1) > top_p
            cut[..., 1:] = cut[..., :-1].clone()
            cut[..., 0] = False
            ordered = ordered.masked_fill(cut, float("-inf"))
            logits = torch.empty_like(logits).scatter(-1, order, ordered)
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
        if int(nxt[0, 0]) == tok.eot_id:
            break
        text = tok.decode(idx[0, len(ids):].tolist())
        if text.endswith("\ufffd"):
            continue
        if len(text) > len(shown):
            yield text[len(shown):]
            shown = text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/kaggle_out/q10/last.pt")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    tok_path = args.tokenizer
    if tok_path is None:
        for cand in ("data/bilingual/tokenizer.json",
                     "artifacts/q10_tokenizer.json"):
            if Path(cand).exists():
                tok_path = cand
                break
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = KronTransformer(state["cfg"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    tok = BPETokenizer(tok_path, state["cfg"]["data"]["vocab_size"],
                       state["cfg"].get("bytes_per_token", 3.694),
                       state["cfg"]["data"]["eot_id"])

    print(f"# {ckpt} · step {state.get('step')} · "
          f"{sum(p.numel() for p in model.parameters()):,} params · {device}")
    print(f"# temperature {args.temperature}, top_k {args.top_k}, top_p {args.top_p}, "
          f"repetition_penalty {args.rep_penalty}\n")

    prompts = [(args.prompt, {})] if args.prompt else PROMPTS
    for prompt, over in prompts:
        kw = {"max_new_tokens": args.max_new_tokens,
              "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
              "repetition_penalty": args.rep_penalty}
        kw.update(over)
        t0 = time.time()
        text = "".join(stream(model, tok, prompt, device, **kw))
        dt = time.time() - t0
        print(f"▸ {prompt!r}")
        print(f"  {text.strip()}")
        print(f"  [{dt:.1f}s, T {kw['temperature']}, top_k {kw['top_k']}, "
              f"top_p {kw['top_p']}]\n")


if __name__ == "__main__":
    main()
