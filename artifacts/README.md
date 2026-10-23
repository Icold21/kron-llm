# Artifacts

The final checkpoints, tracked directly in Git (each is a few megabytes). They
are the models referenced in the top-level [README](../README.md).

| file | model | precision | size | what |
|---|---|---|---|---|
| `q2_fp32.pt` | Q2 | fp32 | 8.55 MB | final TinyStories model, 2,079,992 parameters |
| `q2_int8.pt` | Q2 | int8 | 2.45 MB | near-lossless TinyStories deployment checkpoint |
| `s1_fp32.pt` | S1 | fp32 | 8.55 MB | Q2 fine-tuned on Dolly-15k Q/A pairs |
| `s1_int8.pt` | S1 | int8 | 2.45 MB | near-lossless quantised checkpoint (recommended) |
| `s1_int4.pt` | S1 | int4 | 1.42 MB | smallest checkpoint; visibly degraded, not recommended |

The int8/int4 files store symmetric per-channel scales and packed nibbles; the
forward pass is unchanged fp32 after `load_any_checkpoint` dequantises them.
Q2 int8 is near-lossless (+0.0004 bpb); S1 int8 is near-lossless (+0.0011 bpb).
S1 int4 loses +0.3882 bpb and is kept only as the smallest artifact. Q2 int4
was measured (+0.2193 bpb) but is not shipped because it is not competitive.

## Loading

Through the CLI:

```bash
python -m kronllm.cli generate --run artifacts --checkpoint q2_fp32.pt \
    --prompt "Once upon a time" --temperature 1.0 --top-k 8
```

Through the Hugging Face wrapper:

```python
from kronllm.hf import KronForCausalLM
from kronllm.tokenizer import load_tokenizer

model = KronForCausalLM.from_kron_checkpoint("artifacts/s1_int8.pt")
tok = load_tokenizer(model.config.kron_config)
```

`from_kron_checkpoint` accepts fp32, int8 and int4 checkpoints. The returned
model is in evaluation mode on CPU; call `.to("cuda")` to move it to a GPU.
