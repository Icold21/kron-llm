"""mHC machinery: Sinkhorn properties, init, shapes, gradients, lane collapse."""
import sys

import torch

sys.path.insert(0, ".")

from kronllm.model import KronTransformer, MHCRouter
from kronllm.testutils import make_cfg


def build(**kw):
    return KronTransformer(make_cfg(**kw))


def test_mix_cannot_grow_the_residual():
    """Row sums are exactly 1, so each lane is a convex combination."""
    torch.manual_seed(0)
    for s in (2, 4):
        logits = torch.randn(4096, s, s) * 3
        m = MHCRouter.sinkhorn(logits, 20)
        assert torch.allclose(m.sum(-1), torch.ones(4096, s), atol=1e-6)
        assert (m >= 0).all() and (m <= 1).all()
        # 1.04**48 would be a 6x norm leak over a 24-layer double pass.
        assert m.sum(-1).max() <= 1.0 + 1e-6
        # Columns are the approximate half of the constraint.
        assert (m.sum(-2) - 1).abs().max() < 0.06


def test_sinkhorn_leaves_the_polytope_alone():
    """A matrix already on the polytope must be a fixed point of the projection."""
    m = torch.tensor([[0.6, 0.4], [0.4, 0.6]]).expand(3, 2, 2)
    out = MHCRouter.sinkhorn(m.log(), 20)
    assert torch.allclose(out, m, atol=1e-5), out


def test_router_starts_near_a_plain_residual():
    r = MHCRouter(8, streams=2)
    lanes = torch.randn(2, 3, 2, 8)
    branch = torch.randn(2, 3, 8)
    # pre starts at (0.25, 0.75) exactly: weight is zero, so the read is fixed.
    expect_read = 0.25 * lanes[..., 0, :] + 0.75 * lanes[..., 1, :]
    assert torch.allclose(r.read(lanes), expect_read, atol=1e-5)
    # mix starts at the 15:1 doubly stochastic projection, not at zero mixing,
    # and post starts at 1, so the branch lands on both lanes at full strength.
    mix = torch.tensor([[15 / 16, 1 / 16], [1 / 16, 15 / 16]])
    expect = torch.matmul(mix, lanes) + branch.unsqueeze(-2)
    assert torch.allclose(r.merge(lanes, branch), expect, atol=1e-5)


def test_routing_params_stay_fp32_when_the_model_is_half():
    r = MHCRouter(8).half()
    assert r.weight.dtype == torch.float32
    assert r.bias.dtype == torch.float32


def test_mhc_needs_a_loop():
    try:
        build(**{"model.recurse": 1, "model.mhc": {"streams": 2}})
    except ValueError:
        return
    raise AssertionError("mhc accepted recurse=1")


def test_mhc_forward_backward_and_param_cost():
    m = build(**{"model.n_unique_blocks": 6, "model.recurse": 2,
                 "model.loop_start": 1, "model.loop_end": 5,
                 "model.mhc": {"streams": 2}})
    plain = build(**{"model.n_unique_blocks": 6, "model.recurse": 2,
                     "model.loop_start": 1, "model.loop_end": 5})
    extra = m.n_params - plain.n_params
    # 4 routers in the band, 2 lanes, d=64: (2*2 + 2*2) x (2*64) + 8.
    assert extra == 4 * (8 * 128 + 8), extra
    ids = torch.randint(0, 256, (2, 16))
    assert m(ids)["logits"].shape[:2] == (2, 16)
    m(ids, torch.roll(ids, -1, dims=1))["loss"].backward()
    assert m.mhc[0].weight.grad is not None
    assert torch.isfinite(m.mhc[0].weight.grad).all()


def test_lanes_collapse_before_the_tail():
    """Head and tail layers must see one stream; only the band sees two."""
    m = build(**{"model.n_unique_blocks": 6, "model.recurse": 2,
                 "model.loop_start": 1, "model.loop_end": 5,
                 "model.mhc": {"streams": 2}})
    seen = {}

    def spy_for(original, slot):
        def spy(x, *a, **k):
            seen.setdefault(slot, tuple(x.shape))
            return original(x, *a, **k)
        return spy

    for slot, block in enumerate(m.blocks):
        block._orig_forward = block.forward
        block.forward = spy_for(block._orig_forward, slot)
    try:
        m(torch.randint(0, 256, (2, 16)))
    finally:
        # Blocks are shared: the same object appears in several slots, so
        # restore each one once.
        for block in m.blocks:
            if "forward" in block.__dict__:
                del block.forward
            if "_orig_forward" in block.__dict__:
                del block._orig_forward
    # plan: 1 head, 4 blocks x2, 1 tail = 10 slots.  Head and tail are outside
    # the band and get a 3-D input; the band goes through forward_mhc instead.
    assert seen[0] == (2, 16, 64), seen
    assert seen[9] == (2, 16, 64), seen


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
