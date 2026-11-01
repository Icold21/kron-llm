"""KronLLM model: a transformer whose every matrix map is Kronecker-parameterised.

Design notes (deviations from the original TZ are listed in README.md)
---------------------------------------------------------------------
* every projection is a :class:`~kronllm.kron.KronCompose` (or a dense
  ``nn.Linear`` in the E0/E1 controls);
* the depth-recurrence state ``D`` is threaded through the blocks explicitly
  (``state`` dict) so it is reset at the start of every forward pass and never
  leaks between batches;
* each compose carries a ReZero gate ``gamma`` (init ``gate_init``) so a branch
  can start as a small perturbation of the residual stream;
* ``batched_prelude`` evaluates all composes of the model as a handful of
  batched GEMMs (:class:`~kronllm.kron.KronBank`) instead of ~900 individual
  kernel launches.  Windows/WDDM charges ~45 us per launch, so this is the
  difference between 120 ms and ~20 ms per training step.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .kron import KronBank, KronCompose, apply_rope, rope_frequencies
from .loss_mask import prompt_target_mask
from torch.utils.checkpoint import checkpoint

# Roles in the order in which a block consumes them.  "fg" is the gate branch
# of the optional gated FFN.
ROLES = ("q", "k", "v", "o", "f", "fg")

# Explicit SDPA backends for `model.attn_impl`; "auto" leaves the choice to
# PyTorch (flash on Ampere+, memory-efficient on Turing, math otherwise).
_ATTN_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class DenseFFN(nn.Module):
    """Plain ``Linear -> SiLU -> Linear`` FFN (the E0/E1 control)."""

    def __init__(self, d: int, mult: float = 1.0) -> None:
        super().__init__()
        hidden = max(1, int(round(d * mult)))
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class KronEmbedding(nn.Module):
    """Embedding table ``V x r`` as a sum of K Kronecker products.

    The vocabulary is split as ``V = v1 * v2`` and the rank as ``r = r1 * r2``;
    the table is

        E = sum_k A_k (x) B_k,   A_k: v1 x r1,  B_k: v2 x r2.

    Parameter cost is ``K * (v1*r1 + v2*r2)`` instead of ``V * r``, while the
    rank stays in the same class as a dense ``V x r`` factorisation:
    ``rank(A (x) B) = rank(A) * rank(B) <= r1 * r2 = r``.  With ``K`` summands
    the table can reach rank ``min(V, r)``, i.e. the same information ceiling
    as the dense factorised embedding it replaces, at a small fraction of the
    parameter count for wide vocabularies.  The table is materialised once per
    forward (``K * V * r`` MACs, amortised over the batch) and the tied head
    keeps the same ``d*r + r*V`` per-token logits cost.
    """

    def __init__(self, vocab_size: int, rank: int, v1: int, v2: int,
                 r1: int, r2: int, K: int) -> None:
        super().__init__()
        if v1 * v2 != vocab_size:
            raise ValueError(f"v1*v2={v1 * v2} != vocab_size={vocab_size}")
        if r1 * r2 != rank:
            raise ValueError(f"r1*r2={r1 * r2} != rank={rank}")
        self.v1, self.v2 = int(v1), int(v2)
        self.r1, self.r2 = int(r1), int(r2)
        self.rank = int(rank)
        self.K = int(K)
        self.A = nn.ParameterList(
            [nn.Parameter(torch.empty(self.v1, self.r1)) for _ in range(self.K)])
        self.B = nn.ParameterList(
            [nn.Parameter(torch.empty(self.v2, self.r2)) for _ in range(self.K)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for a in self.A:
            nn.init.normal_(a, 0.0, 1.0)
        for b in self.B:
            nn.init.normal_(b, 0.0, 1.0)
        # Rescale so that the composed table has entry std 1/sqrt(r), the same
        # scale as a standard embedding initialisation.  Scaling every factor
        # by s scales every Kronecker product by s^2.
        with torch.no_grad():
            t = self.table()
            target = 1.0 / (self.rank ** 0.5)
            s = (target / max(float(t.std()), 1e-8)) ** 0.5
            for p in self.A:
                p.mul_(s)
            for p in self.B:
                p.mul_(s)

    def table(self) -> torch.Tensor:
        """Materialise the ``(V, r)`` table (cached by the caller in eval)."""
        # One einsum instead of K torch.kron calls: torch.kron allocates a
        # V x r tensor per summand, and on WDDM those launches dominate when
        # the table is rebuilt inside every checkpointed loss chunk.
        A = torch.stack(list(self.A))          # (K, v1, r1)
        B = torch.stack(list(self.B))          # (K, v2, r2)
        t = torch.einsum("kvp,kwq->vwpq", A, B)
        return t.reshape(self.v1 * self.v2, self.r1 * self.r2)


class RoutedEmbedding(nn.Module):
    """Dense token embeddings selected by a sparse top-k router.

    The token id is first mapped to a small routing code
    ``route[id] in R^{route_dim}``.  A linear router turns that code into
    ``v_eff`` logits; the top-k logits are softmaxed and used to mix the
    corresponding rows of a dense bank ``bank[v_eff, d_model]``.  The effective
    embedding table is therefore dense in width but sparse in computation:

    ``E_eff[v] = sum_{i in topk(v)} softmax(logits[v])_i * bank[i]``

    This avoids the low-rank bottleneck of the factorised embedding (the output
    is a full ``d_model`` vector, not an ``r``-dimensional projection) while
    keeping the stored table at ``v_eff * d_model`` instead of ``V * d_model``.
    The same ``E_eff`` is reused by the tied output head, so the head adds no
    parameters.

    Args:
        vocab_size: Number of token ids.
        d_model: Embedding width.
        v_eff: Number of virtual embedding rows (default 5,000).
        route_dim: Width of the routing code (default 6).
        top_k: Number of bank rows mixed per token (default 10).
        temperature: Softmax temperature on the selected logits.

    Raises:
        ValueError: If ``top_k`` is outside ``[1, v_eff]`` or ``route_dim`` is
            not positive.
    """

    def __init__(self, vocab_size: int, d_model: int, v_eff: int = 5000,
                 route_dim: int = 6, top_k: int = 10,
                 temperature: float = 1.0) -> None:
        super().__init__()
        if route_dim <= 0:
            raise ValueError(f"route_dim must be positive, got {route_dim}")
        if not 1 <= top_k <= v_eff:
            raise ValueError(f"top_k must be in [1, {v_eff}], got {top_k}")
        self.vocab_size = int(vocab_size)
        self.d = int(d_model)
        self.v_eff = int(v_eff)
        self.route_dim = int(route_dim)
        self.top_k = int(top_k)
        self.temperature = float(temperature)

        self.route = nn.Embedding(self.vocab_size, self.route_dim)
        self.route_proj = nn.Linear(self.route_dim, self.v_eff, bias=False)
        self.bank = nn.Embedding(self.v_eff, self.d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the router and the bank with matching output scale.

        The bank rows are scaled by ``sqrt(top_k)`` so that the weighted
        average of ``top_k`` rows has the same standard deviation as a plain
        embedding row (the softmax weights sum to one and are close to uniform
        at initialisation).
        """
        nn.init.normal_(self.route.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.route_proj.weight, mean=0.0,
                        std=1.0 / math.sqrt(self.route_dim))
        nn.init.normal_(self.bank.weight, mean=0.0,
                        std=0.02 * math.sqrt(self.top_k))

    def set_top_k(self, top_k: int) -> None:
        """Change the number of bank rows mixed per token.

        Args:
            top_k: New value, must be in ``[1, v_eff]``.

        Raises:
            ValueError: If ``top_k`` is outside the valid range.
        """
        if not 1 <= int(top_k) <= self.v_eff:
            raise ValueError(f"top_k must be in [1, {self.v_eff}], got {top_k}")
        self.top_k = int(top_k)

    def route_logits(self) -> torch.Tensor:
        """Return the router logits for every vocabulary entry.

        Returns:
            A ``(vocab_size, v_eff)`` tensor.
        """
        codes = self.route.weight
        return self.route_proj(codes) / max(self.temperature, 1e-5)

    def effective_embeddings(self) -> torch.Tensor:
        """Build the dense ``(vocab_size, d_model)`` effective embedding table.

        Returns:
            The top-k mixture of bank rows for every token id.
        """
        logits = self.route_logits()
        values, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(values, dim=-1)
        rows = self.bank(indices)                 # (V, k, d)
        return (rows * weights.unsqueeze(-1)).sum(dim=-2)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Look up the routed embedding of ``idx``.

        Args:
            idx: Integer token ids of any shape.

        Returns:
            A float tensor with shape ``idx.shape + (d_model,)``.
        """
        return F.embedding(idx, self.effective_embeddings())


def _kron_out_dims(d_out: int, n2: int) -> Tuple[int, int]:
    """Split ``d_out`` into ``(n1, n2)`` reusing ``n2`` when possible."""
    if d_out % n2 == 0:
        return d_out // n2, n2
    for n2c in range(int(math.sqrt(d_out)), 0, -1):
        if d_out % n2c == 0:
            return d_out // n2c, n2c
    return d_out, 1


class MHCRouter(nn.Module):
    """Manifold-constrained hyper-connections between residual streams.

    An ordinary block does ``x = x + branch``.  mHC keeps ``S`` parallel
    residual lanes instead and puts three learned mappings between them and the
    block::

        branch = pre · lanes                 # which lane the block reads
        lanes  = mix @ lanes + post ⊗ branch  # how the output is written back

    ``mix`` is pushed onto the Birkhoff polytope -- doubly stochastic, rows and
    columns summing to one -- by Sinkhorn normalisation in log space.  That is
    the "manifold constraint": a doubly stochastic map preserves the sum of the
    lanes, so twenty-four repeated applications of one weight-tied block cannot
    multiply the residual norm, which is the failure mode a plain loop has.

    At initialisation the router *is* an ordinary residual connection: ``pre``
    starts at 0.25/0.75, ``post`` at 1.0 and ``mix`` near the identity.  mHC
    therefore starts as the baseline it is meant to beat and only learns to use
    the extra lane if that helps; it never has to unlearn a random mixing.

    SenseNova-Loop-9B uses ``S=2``, 20 Sinkhorn iterations, and keeps the
    routing parameters in FP32 while the backbone runs in BF16 -- the Sinkhorn
    recursion loses too much to bf16.
    """

    def __init__(self, d: int, streams: int = 2, iterations: int = 20,
                 fixed_identity: bool = False, pre_scale: float = 1.0) -> None:
        super().__init__()
        if streams < 2:
            raise ValueError("mHC needs at least two residual streams")
        self.d = int(d)
        self.streams = int(streams)
        self.iterations = int(iterations)
        self.fixed_identity = bool(fixed_identity)
        self.pre_scale = float(pre_scale)
        s = self.streams
        out = 2 * s + (0 if self.fixed_identity else s * s)
        self.weight = nn.Parameter(
            torch.zeros(out, s * self.d, dtype=torch.float32),
            requires_grad=not self.fixed_identity)
        self.bias = nn.Parameter(torch.zeros(out, dtype=torch.float32))
        with torch.no_grad():
            starts = {2: (0.25, 0.75), 4: (0.0625, 0.1875, 0.25, 0.5)}[s]
            self.bias[:s].copy_(torch.logit(torch.tensor(starts)))
            if not self.fixed_identity:
                diag = math.log((1 - (s - 1) / 16) * 16)
                self.bias[2 * s:].copy_(
                    (torch.eye(s) * diag).flatten())

    def _apply(self, fn, recurse: bool = True):
        """Keep the routing parameters FP32 when the model is cast to half.

        The Sinkhorn recursion runs 20 normalisations; in bf16 the row and
        column sums drift enough that ``mix`` stops being doubly stochastic,
        which is the one property the constraint exists to guarantee.
        """
        def convert(tensor):
            result = fn(tensor)
            if tensor.is_floating_point() and result.dtype != torch.float32:
                return tensor.to(device=result.device, dtype=torch.float32)
            return result
        return super()._apply(convert, recurse=recurse)

    @staticmethod
    def sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
        """Project square logits onto doubly stochastic matrices, in log space.

        Sinkhorn converges slowly for extreme logits: at ``S=2`` twenty
        iterations leave the row sums off by up to 2.5 %, and a 24-layer block
        applied twice would multiply the lane norm by ``1.025**48 ≈ 3``.  The
        final row normalisation makes every output lane an exact convex
        combination of the input lanes, so the mixing cannot grow the residual
        even where the Sinkhorn limit has not been reached; the column sums
        then stay within ~2 % of one, which is the part that is approximate.
        Boundedness is the property the loop needs; sum preservation is the
        part that can afford to be approximate.

        Two lanes have a closed form and never come here (see
        :meth:`sinkhorn2`); this path is for four.
        """
        logits = logits.float()
        for _ in range(iterations):
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
        mix = logits.exp()
        return mix / mix.sum(-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def sinkhorn2(logits: torch.Tensor) -> torch.Tensor:
        """Exact two-lane projection, in closed form.

        A 2x2 doubly stochastic matrix is ``[[x, 1-x], [1-x, x]]``, so it has
        one degree of freedom and the Sinkhorn limit is determined by the
        cross-ratio, which scaling cannot change::

            (a/b)(d/c) = (x/(1-x))**2   =>   x = 1 / (1 + sqrt(bc / ad))

        Twenty iterations of logsumexp on an 8-element matrix are forty kernel
        launches per router, 24 routers, every step -- measured at 1.19 s per
        step against 0.42 s for the same depth without mHC.  The closed form is
        exact, so the routing gets cheaper than the block it routes.
        """
        a = logits[..., 0, 0].float()
        b = logits[..., 0, 1].float()
        c = logits[..., 1, 0].float()
        d = logits[..., 1, 1].float()
        ratio = torch.sqrt(torch.exp((b + c) - (a + d)).clamp_min(1e-12))
        x = 1.0 / (1.0 + ratio)
        one = torch.ones_like(x)
        return torch.stack([torch.stack([x, one - x], -1),
                            torch.stack([one - x, x], -1)], -2)

    def route(self, lanes: torch.Tensor):
        """``(pre, post, mix)`` for these lanes, computed once per block."""
        return self._route(lanes)

    def _route(self, lanes: torch.Tensor):
        # Autocast off: the routing is eight numbers per position, and bf16/fp16
        # loses enough in the Sinkhorn normalisations that ``mix`` stops summing
        # to one -- the property the constraint exists to guarantee.
        with torch.autocast(device_type=lanes.device.type, enabled=False):
            flat = lanes.flatten(-2).float()
            normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True)
                                        + 1e-6)
            logits = torch.nn.functional.linear(normed, self.weight, self.bias)
            s = self.streams
            pre = logits[..., :s].sigmoid()
            post = 2 * logits[..., s:2 * s].sigmoid()
            if self.fixed_identity:
                mix = torch.eye(s, device=flat.device).expand(
                    *flat.shape[:-1], s, s)
            elif s == 2:
                mix = self.sinkhorn2(logits[..., 2 * s:].unflatten(-1, (2, 2)))
            else:
                mix = self.sinkhorn(
                    logits[..., 2 * s:].unflatten(-1, (s, s)), self.iterations)
        return pre, post, mix

    def read(self, lanes: torch.Tensor, routing=None) -> torch.Tensor:
        """The branch input the block sees: a learned mix over the lanes.

        ``routing`` lets the caller compute ``(pre, post, mix)`` once and use it
        for both the read and the merge.  They are functions of the same lanes,
        and recomputing them doubled the mHC step cost for nothing.
        """
        pre = routing[0] if routing is not None else self._route(lanes)[0]
        branch = torch.matmul(pre.unsqueeze(-2), lanes.float()).squeeze(-2)
        if self.pre_scale != 1.0:
            branch = branch * self.pre_scale
        return branch.to(lanes.dtype)

    def merge(self, lanes: torch.Tensor, branch: torch.Tensor,
              routing=None) -> torch.Tensor:
        """Mix the lanes and write the branch output into each of them."""
        if routing is None:
            routing = self._route(lanes)
        _, post, mix = routing
        mixed = torch.matmul(mix.to(lanes.dtype), lanes)
        update = post.unsqueeze(-1) * branch.float().unsqueeze(-2)
        return (mixed.float() + update).to(lanes.dtype)


class KronBlock(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        mc = cfg["model"]
        kc = mc["kron"]
        d = int(mc["d_model"])
        nh = int(mc["n_heads"])
        nkv = mc.get("n_kv_heads") or nh
        self.d = d
        self.n_heads = nh
        self.n_kv_heads = int(nkv)
        self.d_head = d // nh
        assert self.d_head * nh == d, "d_model must be divisible by n_heads"
        d_kv = self.d_head * self.n_kv_heads

        self.use_kron_qkv = bool(mc["use_kron_qkv"])
        self.use_kron_ffn = bool(mc["use_kron_ffn"])
        self.ffn_silu = bool(mc.get("ffn_silu", True))
        self.ffn_gated = bool(mc.get("ffn_gated", False)) and self.use_kron_ffn
        # Feature interaction: a bilinear (GLU) FFN mixes features
        # multiplicatively, ``W_out(act(W_a x) * (W_b x))``, which the linear
        # Kronecker composition cannot express.  ``ffn_mix_kron`` adds the
        # standard Kronecker map back as a non-interacting component.
        self.ffn_mix = str(mc.get("ffn_mix", "kron")).lower()
        if self.ffn_mix not in {"kron", "glu"}:
            raise ValueError("ffn_mix must be 'kron' or 'glu'")
        self.ffn_mix_rank = int(mc.get("ffn_mix_rank", 16))
        self.ffn_mix_kron = bool(mc.get("ffn_mix_kron", False))

        self.norm1 = RMSNorm(d, float(mc["norm_eps"]))
        self.norm2 = RMSNorm(d, float(mc["norm_eps"]))
        # Attention kernel: "auto" lets PyTorch pick the best fused SDPA
        # backend; "flash"/"mem_efficient"/"math" pin one for benchmarking.
        # This only swaps the QK^T-softmax-AV kernel; q/k/v still come from the
        # Kronecker maps and the output still goes through the Kron o-map.
        self.attn_impl = str(mc.get("attn_impl", "auto")).lower()
        if self.attn_impl not in {"auto", *_ATTN_BACKENDS}:
            raise ValueError(
                f"attn_impl must be auto/{'/'.join(_ATTN_BACKENDS)}, "
                f"got {self.attn_impl!r}")
        # intra-layer normalisation of the attention operands: keeps q/k at a
        # fixed scale regardless of how the Kron branch grows, which removes
        # the logit drift that limits the learning rate in small models.
        self.qk_norm = bool(mc.get("qk_norm", False))
        if self.qk_norm:
            self.q_norm = RMSNorm(self.d_head, float(mc["norm_eps"]))
            self.k_norm = RMSNorm(self.d_head, float(mc["norm_eps"]))

        self.kc = kc
        self.L = int(kc["L"])
        self.anchor = bool(kc["anchor"])
        self.s_learnable = bool(kc["s_learnable"])

        n1, n2 = int(kc["n1"]), int(kc["n2"])
        assert n1 * n2 == d, f"n1*n2 must equal d_model ({n1}*{n2} != {d})"

        def s_of(role: str) -> float:
            s = kc["s_init"]
            if isinstance(s, (int, float)):
                return float(s)
            if role in s:
                return float(s[role])
            if role in ("q", "k", "v", "o") and "qkv" in s:
                return float(s["qkv"])
            return float(s.get("ffn", 1.0))

        self.s_values = {r: s_of(r) for r in ("q", "k", "v", "o")}
        self.s_values["f"] = self.s_values["fg"] = s_of("f")

        if self.use_kron_qkv:
            kq = kc.get("qkv_terms", None)   # E4: extra Kronecker terms on Q only
            self.q = self._make(d, n1, n2, self.L, s_of("q"), self.anchor, kq)
            self.k = self._make(d_kv, n1, n2, self.L, s_of("k"), self.anchor)
            self.v = self._make(d_kv, n1, n2, self.L, s_of("v"), self.anchor)
            self.o = self._make(d, n1, n2, self.L, s_of("o"), self.anchor)
        else:
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d_kv, bias=False)
            self.v = nn.Linear(d, d_kv, bias=False)
            self.o = nn.Linear(d, d, bias=False)

        if self.use_kron_ffn and self.ffn_mix == "kron":
            self.ffn = self._make(d, n1, n2, self.L, s_of("f"), self.anchor)
            if self.ffn_gated:
                self.ffn_g = self._make(d, n1, n2, self.L, s_of("fg"), self.anchor)
            else:
                self.ffn_g = None
        elif self.use_kron_ffn and self.ffn_mix == "glu":
            r = self.ffn_mix_rank
            self.glu_a = nn.Linear(d, r, bias=False)
            self.glu_b = nn.Linear(d, r, bias=False)
            self.glu_o = nn.Linear(r, d, bias=False)
            # The generic 0.02 init leaves the bilinear branch ~300x smaller
            # than a Kron block, i.e. effectively switched off; mark the
            # modules so `_init_weights` scales them to unit activation.
            self.glu_a._glu_role = "in"
            self.glu_b._glu_role = "in"
            self.glu_o._glu_role = "out"
            # Optional non-interacting Kronecker component alongside the
            # bilinear term (Q9k): "plus the standard Kronecker product".
            self.ffn = (self._make(d, n1, n2, self.L, s_of("f"), self.anchor)
                        if self.ffn_mix_kron else None)
            self.ffn_g = None
        else:
            self.ffn = DenseFFN(d, float(mc.get("dense_ffn_mult", 1.0)))
            self.ffn_g = None

    # ------------------------------------------------------------------
    def _make(self, d_out: int, n1: int, n2: int, L: int, s_init: float,
              anchor: bool, k_override: Optional[int] = None) -> KronCompose:
        kc = self.kc
        learn = self.s_learnable
        if d_out == self.d:
            n1o, n2o, L_eff = n1, n2, L
        else:                       # GQA projection: non-square, single factor
            n1o, n2o = _kron_out_dims(d_out, n2)
            L_eff, anchor, learn = 1, False, False
        return KronCompose(
            self.d, d_out, n1, n2, n1o, n2o,
            L=L_eff, K=int(k_override if k_override is not None else kc["terms"]),
            K_win=int(kc["carry_window"]),
            s_init=s_init, anchor=anchor, s_learnable=learn,
            stop_gradient=bool(kc["stop_gradient"]),
            theta_std=float(kc["theta_std"]),
            rezero_eps=float(kc["rezero_eps"]),
            s_init_effective=bool(kc.get("s_init_effective", True)),
            init_gain=kc.get("init_gain", "unit"),
            innovation_scale=float(kc.get("innovation_scale", 0.1)),
            normalize_D=bool(kc.get("normalize_D", True)),
            gate_init=float(kc.get("gate_init", 0.1)),
            use_gate=bool(kc.get("gate", True)),
            composition=str(kc.get("composition", "product")),
            sum_alpha_init=float(kc.get("sum_alpha_init", 1.0)),
            dynamic_gate=(kc.get("dynamic_gate")
                          if L_eff == L and str(kc.get("composition",
                                                       "product")) == "sum"
                          else None),
            groups=(int(kc.get("groups", 1))
                    if L_eff == L and str(kc.get("composition",
                                                 "product")) == "grouped"
                    else 1),
            alternate_start=(str(kc.get("alternate_start", "sum"))
                             if L_eff == L and str(kc.get("composition",
                                                          "product")) == "alternating"
                             else "sum"),
            nonlinearity=str(kc.get("nonlinearity", "silu")),
            norm_between=bool(kc.get("norm_between", False)),
            nonlinear_every=int(kc.get("nonlinear_every", 1)),
            nonlinear_mode=str(kc.get("nonlinear_mode", "replace")),
            sigma_mode=str(kc.get("sigma_mode", "learned")),
            sigma_value=float(kc.get("sigma_value", 0.5)),
        )

    def ffn_module(self) -> nn.Module:
        return self.ffn

    def _make_apply(self, state, layer_idx: int, mode: str,
                    P: Optional[Dict[str, torch.Tensor]]):
        """Return ``(y, D_or_None)`` for one Kronecker map.

        ``P`` carries a precomputed ``(d_out, d_in)`` composition (gate
        included), so a fused layer is one GEMM instead of ``L`` matvecs.
        """
        def apply(mod: nn.Module, role: str, inp: torch.Tensor):
            if not isinstance(mod, KronCompose):
                return mod(inp), None
            if P is not None and P.get(role) is not None:
                return torch.matmul(inp, P[role].transpose(0, 1)), None
            return mod(inp, state.get(role), layer_idx, mode=mode)
        return apply

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, state: Dict[str, Optional[torch.Tensor]],
                layer_idx: int, cos: torch.Tensor, sin: torch.Tensor,
                collect: bool = False, mode: str = "auto",
                P: Optional[Dict[str, torch.Tensor]] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                branch_only: bool = False,
                ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]],
                           Dict[str, Any], Optional[Tuple[torch.Tensor,
                                                           torch.Tensor]]]:
        """``P`` maps role -> precomputed ``(d_out, d_in)`` composed matrix.

        The matrix already contains the ReZero gate, so it is applied as
        ``y = h @ P.T`` -- one GEMM instead of ``L`` matvecs + a gate.

        ``past_kv`` is the cached ``(k, v)`` of the previous tokens (already
        RoPE-rotated); the returned 4th value is the updated cache when
        ``use_cache`` is set, else ``None``.

        ``branch_only`` stops after the attention branch and returns it as the
        first value, for the mHC band where the residual streams live outside
        the block (see :meth:`forward_mhc`).
        """
        B, T, _ = x.shape
        dh, nh = self.d_head, self.n_heads
        cache: Dict[str, Any] = {}

        h = self.norm1(x)
        apply = self._make_apply(state, layer_idx, mode, P)

        q, Dq = apply(self.q, "q", h)
        k, Dk = apply(self.k, "k", h)
        v, Dv = apply(self.v, "v", h)
        for name, D in (("q", Dq), ("k", Dk), ("v", Dv)):
            if D is not None:
                state[name] = D

        q = q.view(B, T, nh, dh).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, dh).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, dh).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        k_cache, v_cache = k, v
        n_past = k.shape[2] - q.shape[2]

        if self.n_kv_heads != nh:
            rep = nh // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if collect and n_past == 0:
            scale = 1.0 / math.sqrt(dh)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            p = torch.softmax(scores.float(), dim=-1)
            ent = -(p.clamp_min(1e-9).log() * p).sum(-1)
            cache["attn_entropy"] = ent.mean().item()
            cache["q_drift"] = ((q - h.view(B, T, nh, dh).transpose(1, 2)).norm()
                                / h.norm().clamp_min(1e-9)).item()
            del scores, p, mask
            # cusolver can fail (INTERNAL_ERROR) when the GPU is already near
            # its memory ceiling at batch 32; a diagnostic must never take a
            # multi-hour session down with it.
            try:
                cache["q_sigma_min"] = torch.linalg.svdvals(
                    q.reshape(-1, dh).float()).min().item()
                cache["k_sigma_min"] = torch.linalg.svdvals(
                    k.reshape(-1, dh).float()).min().item()
            except RuntimeError:
                cache["q_sigma_min"] = float("nan")
                cache["k_sigma_min"] = float("nan")

        out = self._attend(q, k, v, n_past)
        out = out.transpose(1, 2).reshape(B, T, self.d)

        o, Do = apply(self.o, "o", out)
        if Do is not None:
            state["o"] = Do
        if branch_only:
            # mHC band: the attention half is returned as a branch and the
            # caller merges it into its residual streams, so the block must not
            # add it to ``x`` itself.
            return o, state, cache, ((k_cache, v_cache) if use_cache else None)
        x = x + o
        x = x + self._ffn_half(x, state, apply)
        new_kv = (k_cache, v_cache) if use_cache else None
        return x, state, cache, new_kv

    def _ffn_half(self, x: torch.Tensor,
                  state: Dict[str, Optional[torch.Tensor]],
                  apply) -> torch.Tensor:
        """The FFN branch, taken from the residual stream ``x``."""
        h2 = self.norm2(x)
        if self.ffn_mix == "glu":
            # Bilinear feature interaction: act(W_a x) * (W_b x).
            f = self.glu_o(F.silu(self.glu_a(h2)) * self.glu_b(h2))
            if isinstance(self.ffn, KronCompose):
                fk, Df = apply(self.ffn, "f", h2)
                if Df is not None:
                    state["f"] = Df
                f = f + fk
        elif not isinstance(self.ffn, KronCompose):
            f = self.ffn(h2)
        elif self.ffn_gated:
            fa, Da = apply(self.ffn, "f", h2)
            fb, Db = apply(self.ffn_g, "fg", h2)
            if Da is not None:
                state["f"] = Da
            if Db is not None:
                state["fg"] = Db
            f = fa * F.silu(fb)
        else:
            f, Df = apply(self.ffn, "f", h2)
            if Df is not None:
                state["f"] = Df
            if self.ffn_silu:
                f = F.silu(f)
        return f

    def forward_mhc(self, streams: torch.Tensor, router_attn, router_ffn,
                    state: Dict[str, Optional[torch.Tensor]],
                    layer_idx: int, cos: torch.Tensor, sin: torch.Tensor,
                    collect: bool = False, mode: str = "auto",
                    P: Optional[Dict[str, torch.Tensor]] = None,
                    past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    use_cache: bool = False,
                    ):
        """One block inside an mHC band.

        Reads a branch input from the residual *streams*, computes the attention
        branch, mixes it back into the streams, then does the same for the FFN.
        Two routers per block, exactly as in SenseNova's ``MHCDecoderLayer``:
        the residual never enters the block, it is read and written through the
        routing matrices, which is what lets a weight-tied block behave
        differently on its second pass.
        """
        cache: Dict[str, Any] = {}
        routing_a = router_attn.route(streams)
        branch = router_attn.read(streams, routing_a)
        o, state, cache, new_kv = self.forward(
            branch, state, layer_idx, cos, sin, collect=collect, mode=mode,
            P=P, past_kv=past_kv, use_cache=use_cache, branch_only=True)
        streams = router_attn.merge(streams, o, routing_a)
        routing_f = router_ffn.route(streams)
        branch = router_ffn.read(streams, routing_f)
        f = self._ffn_half(branch, state, self._make_apply(state, layer_idx,
                                                          mode, P))
        streams = router_ffn.merge(streams, f, routing_f)
        return streams, state, cache, new_kv

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                n_past: int = 0) -> torch.Tensor:
        """SDPA with an optional KV cache and the configured backend."""
        def call(**kw):
            if self.attn_impl == "auto":
                return F.scaled_dot_product_attention(q, k, v, **kw)
            with sdpa_kernel(_ATTN_BACKENDS[self.attn_impl]):
                return F.scaled_dot_product_attention(q, k, v, **kw)

        if n_past == 0:
            return call(is_causal=True)
        if q.shape[2] == 1:
            # One new query token sees every cached key.
            return call(is_causal=False)
        # Chunked prefill with a cache: query i (absolute position n_past+i)
        # must not see keys beyond n_past+i.
        qpos = torch.arange(q.shape[2], device=q.device)[:, None] + n_past
        kpos = torch.arange(k.shape[2], device=q.device)[None, :]
        mask = kpos <= qpos
        return call(attn_mask=mask)


class KronTransformer(nn.Module):
    """Byte-level decoder-only transformer with Kronecker matrix maps."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        mc = cfg["model"]
        dc = cfg["data"]
        self.cfg = cfg
        self.vocab_size = int(dc["vocab_size"])
        self.d = int(mc["d_model"])
        self.n_layers = int(mc["n_layers"])
        # Recursive weight sharing: n_unique blocks applied `recurse` times.
        # The effective depth is n_unique * recurse; parameters are counted once.
        self.n_unique = int(mc.get("n_unique_blocks") or self.n_layers)
        self.recurse = int(mc.get("recurse") or 1)
        if self.n_unique < 1 or self.recurse < 1:
            raise ValueError("n_unique_blocks and recurse must be >= 1")
        # Which part of the stack repeats.  Repeating the whole stack (the
        # default) makes every shared block serve as both "tokens -> features"
        # and "features -> logits"; repeating only the middle band leaves the
        # first layers free to read the input and the last ones free to write
        # the output, which is the arrangement a looped transformer wants
        # (SenseNova-Loop-9B repeats layers [4,28) of 32, 2026-10-30).
        self.loop_start = int(mc.get("loop_start") or 0)
        loop_end = mc.get("loop_end")
        self.loop_end = int(loop_end) if loop_end is not None else self.n_unique
        if not 0 <= self.loop_start < self.loop_end <= self.n_unique:
            raise ValueError(
                f"need 0 <= loop_start < loop_end <= n_unique, got "
                f"{self.loop_start}, {self.loop_end}, {self.n_unique}")
        self.n_layers = self.n_unique * self.recurse
        self.n_heads = int(mc["n_heads"])
        self.d_head = self.d // self.n_heads
        self.rope_theta = float(mc["rope_theta"])
        self.seq_len = int(dc["seq_len"])
        self.eot_id = int(dc.get("eot_id") or (self.vocab_size - 1))

        self.embed_rank = int(mc.get("embed_rank") or 0)
        self._glu_rank = int(mc.get("ffn_mix_rank", 16))
        mode = str(mc.get("embed_mode", "auto")).lower()
        if mode == "auto":
            mode = "factorised" if self.embed_rank > 0 else "dense"
        if mode not in {"factorised", "dense", "routed", "kron"}:
            raise ValueError(
                f"embed_mode must be auto/factorised/dense/routed/kron, "
                f"got {mode!r}")
        self.embed_mode = mode
        self.routed: Optional[RoutedEmbedding] = None
        self.kron_embed: Optional[KronEmbedding] = None
        if mode == "routed":
            rc = mc.get("routed", {})
            self.routed = RoutedEmbedding(
                self.vocab_size, self.d,
                v_eff=int(rc.get("v_eff", 5000)),
                route_dim=int(rc.get("route_dim", 6)),
                top_k=int(rc.get("active_rows", 10)),
                temperature=float(rc.get("temperature", 1.0)),
            )
            self.embed_rank = 0
            self.embed = None
            self.embed_up = None
            self.lm_head = None
        elif mode == "kron":
            if self.embed_rank <= 0:
                raise ValueError("kron embedding requires embed_rank > 0")
            # V x r table as a sum of Kronecker products; the up-projection
            # U (d x r) is kept, so lookup and the tied head stay identical to
            # the factorised mode -- only the V x r storage is compressed.
            ek = mc.get("embed_kron", {})
            self.kron_embed = KronEmbedding(
                self.vocab_size, self.embed_rank,
                v1=int(ek.get("v1", 64)), v2=int(ek.get("v2", 64)),
                r1=int(ek.get("r1", 16)), r2=int(ek.get("r2", 8)),
                K=int(ek.get("terms", 32)))
            self.embed = None
            self.embed_up = nn.Linear(self.embed_rank, self.d, bias=False)
            self.lm_head = None
        elif mode == "factorised":
            if self.embed_rank <= 0:
                raise ValueError("factorised embedding requires embed_rank > 0")
            # ALBERT-style factorised (tied) embedding: V x r then r x d.
            # Keeps the embedding from eating the parameter budget:
            #   V*r + r*d  instead of  V*d
            self.embed = nn.Embedding(self.vocab_size, self.embed_rank)
            self.embed_up = nn.Linear(self.embed_rank, self.d, bias=False)
            self.lm_head = None
        else:
            self.embed_rank = 0
            self.embed = nn.Embedding(self.vocab_size, self.d)
            self.embed_up = None
            self.lm_head = nn.Linear(self.d, self.vocab_size, bias=False)
            if mc["tie_embeddings"]:
                self.lm_head.weight = self.embed.weight

        self.blocks = nn.ModuleList([KronBlock(cfg)
                                     for _ in range(self.n_unique)])
        self.norm_f = RMSNorm(self.d, float(mc["norm_eps"]))

        self._rope_cache: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_trace: Optional[List[Dict[str, torch.Tensor]]] = None
        self.last_stats: Dict[str, float] = {}
        # "auto" | "matrix" | "sequential" -- see KronCompose.forward
        self.force_mode: Optional[str] = None
        self.use_bank = bool(mc.get("batched_prelude", True))
        self._prelude_cache = None
        self._eff_cache: Optional[torch.Tensor] = None
        self.apply(self._init_weights)
        if self.routed is not None:
            self.routed.reset_parameters()
        self._unique_blocks = list(self.blocks)
        # Execution slot -> physical block index.  With the whole stack
        # repeated this is the familiar ``i % n_unique``; with loop_start /
        # loop_end only the middle band repeats.  ``blocks`` is rebuilt as a
        # list of *references*, so ``parameters()`` still deduplicates and the
        # parameter count does not change -- only the number of executions.
        self.plan = tuple(self._build_plan())
        self.pass_index = self._pass_index(self.plan)
        self.n_layers = len(self.plan)
        if self.recurse > 1:
            self.blocks = nn.ModuleList(
                [self._unique_blocks[i] for i in self.plan])
        self.max_recurse = self.recurse
        # Optional per-pass learned bias, so a shared block can tell which
        # pass it is in (Universal-Transformer-style step signal).  Added when
        # the pass number changes, not at a fixed stride: with a middle band
        # the loop starts at loop_start, not at 0.
        self.pass_bias = (nn.Parameter(torch.zeros(self.max_recurse, self.d))
                          if bool(mc.get("pass_embedding"))
                          and self.max_recurse > 1 else None)
        # Manifold-constrained hyper-connections inside the looped band.  One
        # router per physical block in the band, shared by every pass, because
        # the lanes are what carry the difference between passes: the block and
        # its routing are the same function each time, only the incoming state
        # differs.  That is also what keeps the parameter cost at ~2 % of a 10M
        # model (8.2k per router, two per layer).
        mhc_cfg = mc.get("mhc")
        self.mhc: Optional[nn.ModuleList] = None
        self.mhc_streams = 0
        if mhc_cfg:
            if self.recurse < 2:
                raise ValueError(
                    "mhc needs recurse >= 2: it exists to make a repeated "
                    "block behave differently on its second pass")
            options = mhc_cfg if isinstance(mhc_cfg, dict) else {}
            self.mhc_streams = int(options.get("streams", 2))
            if self.mhc_streams < 2:
                raise ValueError("mhc needs at least two residual streams")
            self.mhc = nn.ModuleList([
                MHCRouter(self.d, streams=self.mhc_streams,
                          iterations=int(options.get("iterations", 20)),
                          fixed_identity=bool(options.get("fixed_identity",
                                                          False)),
                          pre_scale=float(options.get("pre_scale", 1.0)))
                for _ in range(self.loop_end - self.loop_start)])
        self._banks: List[KronBank] = self._build_banks() if self.use_bank else []
        # >0: compute the training loss in row chunks of this size (through
        # torch.utils.checkpoint) so that a wide vocabulary never materialises
        # the full B*T*V logits tensor.  0 disables chunking.
        self.loss_chunk = int(mc.get("loss_chunk") or 0)
        # Drop the inserted <|system|> prefix from the loss: it is boilerplate
        # the harness writes at inference, and training on it spent ~1 % of the
        # gradient on our own strings while flattering the reported loss.
        self.ignore_prompt_targets = bool(mc.get("ignore_prompt_targets", True))
        # Pretraining trains a text model, so a <|user|> turn is data and stays
        # in the loss; SFT masks it, because there the prompt comes from the
        # harness and only the answer is a target.
        self.mask_user_turns = bool(mc.get("mask_user_turns", False))

    # -- init -------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, KronCompose):
            return                      # orthogonal init already applied
        if isinstance(module, nn.Linear):
            # the up-projection of a factorised embedding needs std 1/sqrt(r)
            # so that E@U has the same scale as a direct V x d table
            glu_role = getattr(module, "_glu_role", None)
            if glu_role == "in":
                std = 1.0 / max(1.0, float(self.d) ** 0.5)
            elif glu_role == "out":
                std = 1.0 / max(1.0, float(self._glu_rank) ** 0.5)
            else:
                std = 1.0 / max(1.0, float(self.embed_rank) ** 0.5) \
                    if module is getattr(self, "embed_up", None) else 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # -- banks ------------------------------------------------------------
    def _build_banks(self) -> List[KronBank]:
        groups: Dict[tuple, Tuple[list, list, list]] = {}
        for i, block in enumerate(self.blocks):
            for role in ROLES:
                if role == "f":
                    mod = block.ffn
                elif role == "fg":
                    mod = block.ffn_g
                else:
                    mod = getattr(block, role)
                if not isinstance(mod, KronCompose):
                    continue
                if getattr(mod, "dynamic_gate", None):
                    # The composed map depends on the input token, so it cannot
                    # be fused into one weight-only prelude per layer.
                    continue
                if mod.composition in {"grouped", "alternating", "nonlinear"}:
                    # These have their own composition path; keep them out of
                    # the batched prelude until they are benchmarked.
                    continue
                sig = (mod.L, mod.K, mod.d_in, mod.d_out, mod.n1_in, mod.n2_in,
                       mod.n1_out, mod.n2_out, mod.anchor, mod.normalize_D,
                       mod.stop_gradient, mod.K_win, mod.rezero_eps,
                       mod.gate is None, mod.composition, mod.alpha is None)
                g = groups.setdefault(sig, ([], [], []))
                g[0].append(mod)
                g[1].append(i)
                g[2].append((i, role))
        banks: List[KronBank] = []
        for mods, layers, keys in groups.values():
            if len(mods) < 2:
                continue                 # nothing to batch
            banks.append(KronBank(mods, layers, keys))
        return banks

    def _prelude(self) -> Tuple[Dict[Tuple[int, str], torch.Tensor],
                                 Dict[Tuple[int, str], torch.Tensor]]:
        # In eval mode theta cannot change between calls, so the Kronecker
        # prelude is computed once and reused.  Generation calls forward() once
        # per token; without this cache every sampled token would redo the
        # L-fold composition (40 MFLOP vs 11 MFLOP per token at d=192).
        if not self.training and self._prelude_cache is not None:
            return self._prelude_cache
        P_map: Dict[Tuple[int, str], torch.Tensor] = {}
        D_map: Dict[Tuple[int, str], torch.Tensor] = {}
        for bank in self._banks:
            P, D = bank.prelude()
            for j, key in enumerate(bank.keys):
                P_map[key] = P[j]
                D_map[key] = D[j]
        if not self.training:
            self._prelude_cache = (P_map, D_map)
        return P_map, D_map

    def train(self, mode: bool = True):
        self._prelude_cache = None          # weights may change -> recompute
        self._eff_cache = None
        return super().train(mode)

    # -- rope -------------------------------------------------------------
    def _rope(self, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (T, device)
        if key not in self._rope_cache:
            cos, sin = rope_frequencies(T, self.d_head, self.rope_theta, device)
            self._rope_cache[key] = (cos.view(1, 1, T, -1), sin.view(1, 1, T, -1))
        return self._rope_cache[key]

    # -- embedding / head --------------------------------------------------
    def _effective_embeddings(self) -> torch.Tensor:
        """Return the routed ``(V, d)`` table, cached in evaluation mode."""
        if not self.training and self._eff_cache is not None:
            return self._eff_cache
        assert self.routed is not None
        eff = self.routed.effective_embeddings()
        if not self.training:
            self._eff_cache = eff
        return eff

    def set_route_top_k(self, top_k: int) -> None:
        """Change the number of embedding rows mixed per token.

        Args:
            top_k: New value, must be in ``[1, v_eff]``.

        Raises:
            RuntimeError: If the model does not use routed embeddings.
        """
        if self.routed is None:
            raise RuntimeError("set_route_top_k requires embed_mode='routed'")
        self.routed.set_top_k(top_k)
        self._eff_cache = None

    def _embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        if self.routed is not None:
            return F.embedding(idx, self._effective_embeddings())
        x = F.embedding(idx, self.embed_table())
        if self.embed_up is not None:
            x = self.embed_up(x)
        return x

    def embed_table(self) -> torch.Tensor:
        """The ``V x r`` lookup table: the kron composition or the dense factor.

        The kron table is materialised fresh on every call; it costs
        ``K * V * r`` MACs, which is amortised over the whole forward pass and
        never cached (caching a tensor that carries a training-time graph would
        leak gradients between forwards).
        """
        if self.kron_embed is not None:
            return self.kron_embed.table()
        assert self.embed is not None
        return self.embed.weight

    def _logits(self, x: torch.Tensor,
                eff: Optional[torch.Tensor] = None,
                emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eff is not None:
            # The routed table is the input embedding and the tied head at the
            # same time: logits[v] = h . E_eff[v].
            return torch.matmul(x, eff.t())
        if self.embed_up is not None:
            # tied: h (d) -> r via U (shape d x r) -> V via E^T (r x V).
            # Costs d*r + r*V instead of d*V.
            w = emb if emb is not None else self.embed_table()
            return torch.matmul(torch.matmul(x, self.embed_up.weight), w.t())
        return self.lm_head(x)

    def _ce_chunk(self, xc: torch.Tensor, yc: torch.Tensor,
                  mc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-token cross-entropy over one row chunk; the checkpointed unit of
        :meth:`_chunked_loss`.  Runs under ``no_grad`` in the forward (freeing
        the chunk's logits) and is recomputed in the backward, so at most one
        chunk of ``chunk x V`` logits exists at a time.  All head weights it
        reads are parameters (leaves), so gradients flow; the kron-embed table
        is rebuilt from its factor parameters inside the recompute.

        ``mc`` drops the positions whose target is our own prompt prefix (see
        :mod:`kronllm.loss_mask`); the caller divides by the number of kept
        tokens, so the masked positions cost neither gradient nor denominator.
        """
        logits = self._logits(xc)
        per = F.cross_entropy(logits, yc, reduction="none")
        if mc is not None:
            per = per[~mc]
        return per

    def _chunked_loss(self, x: torch.Tensor, targets: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean cross-entropy in row chunks of ``loss_chunk`` rows.

        A wide vocabulary makes the full ``B*T*V`` logits tensor the largest
        single allocation of the training step (100 MB at 32x192x4096 fp32).
        Each chunk runs under :func:`torch.utils.checkpoint.checkpoint`, so
        only one chunk of logits is alive at a time and the backward recomputes
        them on demand.  Mathematically identical to the unchunked loss.  Not
        used for routed embeddings: their effective table is a non-leaf tensor
        whose gradients checkpointing would detach.
        """
        y = targets.reshape(-1)
        xr = x.reshape(-1, x.size(-1))
        m = mask.reshape(-1) if mask is not None else None
        n = y.numel()
        total: Optional[torch.Tensor] = None
        kept = 0
        for i in range(0, n, self.loss_chunk):
            sl = slice(i, i + self.loss_chunk)
            chunk = checkpoint(self._ce_chunk, xr[sl], y[sl],
                               (m[sl] if m is not None else None),
                               use_reentrant=False)
            total = chunk.sum() if total is None else total + chunk.sum()
            kept += int(chunk.numel())
        assert total is not None, "loss_chunk > 0 guarantees one chunk"
        # Dividing by the kept count, not by n: with a mask the two differ, and
        # the mean must be over the tokens actually trained on.
        return total / max(1, kept)

    # -- forward ----------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                collect: bool = False,
                past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False) -> Dict[str, Any]:
        B, T = idx.shape
        past_len = int(past_kv[0][0].shape[2]) if past_kv else 0
        cos, sin = self._rope(past_len + T, idx.device)
        cos, sin = cos[:, :, past_len:], sin[:, :, past_len:]
        eff = self._effective_embeddings() if self.routed is not None else None
        x = F.embedding(idx, eff) if eff is not None else self._embed_tokens(idx)
        P_map: Dict[Tuple[int, str], torch.Tensor] = {}
        D_map: Dict[Tuple[int, str], torch.Tensor] = {}
        if self._banks:
            P_map, D_map = self._prelude()

        state: Dict[str, Optional[torch.Tensor]] = {r: None for r in ROLES}
        trace: List[Dict[str, Any]] = []
        stats: Dict[str, float] = {}

        h_norms = [x.detach().norm().item()]
        new_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None)
        # Residual lanes for the mHC band.  ``lanes`` is None outside the band
        # and ``x`` is None inside it, so exactly one of the two is live.
        lanes: Optional[torch.Tensor] = None
        for i, block in enumerate(self.blocks):
            phys = self.plan[i]
            if self.pass_bias is not None:
                # Add the step signal where the pass changes.  The old form
                # (`i % n_unique == 0`) assumed the loop covered the whole
                # stack starting at 0; with a middle band the boundary is at
                # ``loop_start`` plus one stride per pass.
                here = self.pass_index[i]
                before = self.pass_index[i - 1] if i else -1
                if here != before:
                    bias = self.pass_bias[here]
                    if lanes is None:
                        x = x + bias
                    else:
                        lanes = lanes + bias.view(1, 1, 1, -1)
            if D_map:
                for role in ROLES:
                    if (i, role) in D_map:
                        state[role] = D_map[(i, role)]
            P = ({role: P_map.get((i, role)) for role in ROLES}
                 if P_map else None)
            if self.mhc is not None and self.loop_start <= phys < self.loop_end:
                if lanes is None:
                    # Enter the band: one lane for everything the stack has
                    # produced so far.
                    lanes = x.unsqueeze(-2).expand(
                        *x.shape[:-1], self.mhc_streams, x.shape[-1]
                    ).contiguous()
                    x = None
                router = self.mhc[phys - self.loop_start]
                lanes, state, cache, kv = block.forward_mhc(
                    lanes, router, router, state, i, cos, sin,
                    collect=collect, mode=self.force_mode or "auto", P=P,
                    past_kv=past_kv[i] if past_kv is not None else None,
                    use_cache=use_cache)
            else:
                if lanes is not None:
                    # Leave the band: collapse the lanes back into one stream.
                    x = lanes.mean(-2)
                    lanes = None
                x, state, cache, kv = block(
                    x, state, i, cos, sin, collect=collect,
                    mode=self.force_mode or "auto", P=P,
                    past_kv=past_kv[i] if past_kv is not None else None,
                    use_cache=use_cache)
            if new_kv is not None:
                new_kv.append(kv)
            if collect:
                stats.update({f"L{i}.{k}": v for k, v in cache.items()})
                trace.append({
                    "D": {k: v.detach().clone() for k, v in state.items()
                          if v is not None},
                    "s": dict(block.s_values),
                })
            live = x if lanes is None else lanes
            h_norms.append(live.detach().norm().item())
        if lanes is not None:
            x = lanes.mean(-2)

        x = self.norm_f(x)
        emb = self.embed_table() if self.kron_embed is not None else None

        # Every document opens with our own <|system|> prefix.  The targets there
        # are boilerplate, not corpus text, so they are dropped from the loss --
        # see kronllm/loss_mask.py.  A bare bin (the held-out val bin) produces
        # an all-False mask and the loss is exactly what it was before.
        mask: Optional[torch.Tensor] = None
        if targets is not None and self.ignore_prompt_targets:
            mask = prompt_target_mask(idx, mask_user_turns=self.mask_user_turns)
            if not bool(mask.any()):
                mask = None

        if (targets is not None and self.loss_chunk > 0
                and self.routed is None):
            # Chunked loss: never materialise the full B*T*V logits tensor.
            # Used for training, validation and diagnostics alike; `collect`
            # still records the trace and stats from the block outputs.
            out: Dict[str, Any] = {"loss": self._chunked_loss(x, targets, mask)}
        else:
            logits = self._logits(x, eff, emb)
            out = {"logits": logits}
            if targets is not None:
                per = F.cross_entropy(
                    logits.view(-1, self.vocab_size), targets.reshape(-1),
                    reduction="none")
                if mask is not None:
                    per = per[~mask.reshape(-1)]
                    out["loss"] = (per.mean() if per.numel()
                                   else logits.sum() * 0.0)
                else:
                    out["loss"] = per.mean()
        if collect:
            ratios = [(h_norms[i + 1] + 1e-9) / (h_norms[i] + 1e-9)
                      for i in range(len(h_norms) - 1)]
            stats["h_norm_ratio_min"] = min(ratios)
            stats["h_norm_ratio_max"] = max(ratios)
            stats["h_norm_ratio_mean"] = sum(ratios) / len(ratios)
            self.last_trace = trace
            self.last_stats = stats
            out["trace"] = trace
            out["stats"] = stats
        if use_cache:
            out["past_kv"] = new_kv
        return out

    def _build_plan(self) -> List[int]:
        """Physical block index for each execution slot.

        ``loop_start=0, loop_end=n_unique`` gives the whole stack repeated
        ``recurse`` times, which is what the modulo scheme did.  Anything else
        repeats only the middle band and executes the head and tail once.
        """
        start, end = self.loop_start, self.loop_end
        return (list(range(start))
                + list(range(start, end)) * self.recurse
                + list(range(end, self.n_unique)))

    def _pass_index(self, plan: Sequence[int]) -> tuple:
        """Pass number per execution slot, for the per-pass bias.

        Layers before the loop belong to the first pass and layers after it to
        the last one, so the bias changes exactly when the loop starts a new
        pass and never flips back at the tail.
        """
        seen: Dict[int, int] = {}
        out: List[int] = []
        for i in plan:
            if i < self.loop_start:
                out.append(0)
            elif i >= self.loop_end:
                out.append(max(0, self.recurse - 1))
            else:
                k = seen.get(i, 0)
                out.append(k)
                seen[i] = k + 1
        return tuple(out)

    def set_recurse(self, recurse: int) -> None:
        """Change the number of block passes at inference time.

        The blocks are shared, so this only rebuilds the execution schedule and
        the batched prelude.  A model trained with ``recurse=1`` is *not*
        guaranteed to work at ``recurse>1``; this method exists to measure that
        extrapolation explicitly.
        """
        recurse = int(recurse)
        if recurse < 1:
            raise ValueError("recurse must be >= 1")
        self.recurse = recurse
        self.plan = tuple(self._build_plan())
        self.pass_index = self._pass_index(self.plan)
        self.n_layers = len(self.plan)
        self.blocks = nn.ModuleList(
            [self._unique_blocks[i] for i in self.plan])
        self._prelude_cache = None
        self._banks = self._build_banks() if self.use_bank else []

    # -- introspection ----------------------------------------------------
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_params_matrix(self) -> int:
        """Parameters in matrix transforms (Q,K,V,O,FFN), excluding norms/embed."""
        total = 0
        for name, p in self.named_parameters():
            if ("norm" in name or "embed" in name or "lm_head" in name
                    or "route" in name or "bank" in name or "pass_bias" in name):
                continue
            total += p.numel()
        return total

    def kron_modules(self) -> List[Tuple[int, str, KronCompose]]:
        out: List[Tuple[int, str, KronCompose]] = []
        for i, block in enumerate(self.blocks):
            for role in ROLES:
                if role == "f":
                    mod = block.ffn
                elif role == "fg":
                    mod = block.ffn_g
                else:
                    mod = getattr(block, role)
                if isinstance(mod, KronCompose):
                    out.append((i, role, mod))
        return out

    def flops_per_token(self, seq_len: Optional[int] = None, training: bool = False,
                        batch_size: int = 1) -> float:
        """Analytic forward FLOPs/token (MACs * 2).

        With ``batched_prelude`` the L-fold composition is always fused into a
        single d x d product, so the per-token cost is ``d^2`` plus the
        amortised ``L*d^3/tokens`` product.
        """
        T = self.seq_len if seq_len is None else seq_len
        tokens = max(1, int(T) * int(batch_size))
        mc = self.cfg["model"]
        d = self.d
        fused = self.use_bank or tokens >= max(256, d)
        composition = str(mc["kron"].get("composition", "product"))

        def compose_cost(out_d: int, L: int) -> float:
            if composition == "nonlinear":
                # ceil(L/k) fused GEMMs per token; k=1 is L matvecs, k=L is
                # one composed GEMM plus one activation.
                k = max(1, int(mc["kron"].get("nonlinear_every", 1)))
                blocks = math.ceil(L / k)
                extra = L * d if mc["kron"].get("norm_between") else 0
                return blocks * d * out_d + L * d + extra
            if fused and out_d == d and L > 1:
                if composition == "sum":
                    return d * out_d + L * d * d + L * d
                if composition == "grouped":
                    groups = max(1, int(mc["kron"].get("groups", 1)))
                    return (d * out_d + L * d * d
                            + (groups - 1) * d * d * out_d / tokens + L * d)
                if composition == "alternating":
                    start = str(mc["kron"].get("alternate_start", "sum"))
                    products = ((L + 1) // 2 if start == "product" else L // 2)
                    return (d * out_d + L * d * d
                            + products * d * d * out_d / tokens + L * d)
                return d * out_d + L * d * d * out_d / tokens + L * d
            return L * d * out_d + L * d

        macs = 0.0
        for block in self.blocks:
            d_kv = block.d_head * block.n_kv_heads
            if block.use_kron_qkv:
                macs += compose_cost(d, block.L)
                macs += compose_cost(d_kv, block.k.L)
                macs += compose_cost(d_kv, block.v.L)
                macs += compose_cost(d, block.L)
            else:
                macs += 2 * d * d + 2 * d * d_kv
            macs += 2 * T * d                            # QK^T + AV
            if block.use_kron_ffn and block.ffn_mix == "glu":
                r = block.ffn_mix_rank
                macs += 3 * d * r                        # W_a, W_b, W_out
                if block.ffn_mix_kron:
                    macs += compose_cost(d, block.L)
            elif block.use_kron_ffn:
                macs += compose_cost(d, block.L)
                if block.ffn_gated:
                    macs += compose_cost(d, block.L)
            else:
                hidden = int(round(d * float(mc.get("dense_ffn_mult", 1.0))))
                macs += 2 * d * hidden
        if self.routed is not None:
            # Router logits + top-k bank mixture + tied effective head, all
            # amortised over the tokens in the forward pass.
            macs += (self.vocab_size * self.routed.route_dim * self.routed.v_eff
                     + self.vocab_size * self.routed.top_k * d
                     + d * self.vocab_size) / tokens
        elif self.embed_up is not None:
            r = float(self.embed_rank)
            macs += d * r + r * self.vocab_size      # factorised tied head
            if self.kron_embed is not None:
                # Materialising the V x r kron table, amortised over the
                # forward pass' tokens.
                macs += self.kron_embed.K * self.vocab_size * r / tokens
        else:
            macs += d * self.vocab_size              # lm head
        macs += 2 * d
        return 2.0 * macs * (3.0 if training else 1.0)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 1.0, top_k: Optional[int] = 8,
                 top_p: Optional[float] = None,
                 min_p: Optional[float] = None,
                 typical_p: Optional[float] = None,
                 no_repeat_ngram_size: Optional[int] = None,
                 repetition_penalty: Optional[float] = None,
                 rep_window: int = 64,
                 eot_id: Optional[int] = None,
                 stop_at_eot: bool = False,
                 use_cache: bool = True) -> torch.Tensor:
        """Sample tokens; with ``use_cache`` each step attends to cached K/V.

        The cache holds at most ``seq_len`` tokens; once the context is full,
        generation falls back to a full forward on the last ``seq_len`` tokens
        (the behaviour of the uncached path).

        The recipe measured in ``docs/decoding.md`` is ``temperature=1.0,
        top_k=8, top_p=0.9, repetition_penalty=1.1``; ``top_p`` and
        ``repetition_penalty`` default to off so that older callers keep their
        behaviour.  ``top_p`` is applied after ``top_k``: it can only tighten
        the cap, never widen it.

        The alternatives measured in ``docs/free_gains.md`` are all off by
        default: ``min_p`` keeps every token whose probability is at least
        ``min_p`` times the top one (a relative floor, so it adapts to how
        peaked the step is); ``typical_p`` keeps the tokens whose surprisal is
        closest to the distribution's entropy (locally typical sampling);
        ``no_repeat_ngram_size`` bans the token that would repeat an n-gram
        already in the context.  ``temperature <= 0`` decodes greedily, which
        is the only setting where the samplers below do not apply.
        """
        was_training = self.training
        eot_id = int(self.eot_id if eot_id is None else eot_id)
        greedy = temperature is not None and temperature <= 0.0
        self.eval()

        def banned_ngram_tokens() -> set:
            """Tokens that would repeat an n-gram already in ``idx``."""
            n = int(no_repeat_ngram_size or 0)
            if n < 2 or idx.shape[1] < n:
                return set()
            seq = idx[0].tolist()
            prefix = tuple(seq[-(n - 1):])
            banned = set()
            for i in range(len(seq) - n + 1):
                if tuple(seq[i:i + n - 1]) == prefix:
                    banned.add(seq[i + n - 1])
            return banned

        def sample(logits: torch.Tensor) -> torch.Tensor:
            logits = logits.float()
            raw = logits
            banned = banned_ngram_tokens()
            if banned:
                # Applied before the greedy shortcut too: "greedy with a
                # no-repeat ban" is the standard way to stop a small model
                # from looping, and it must not be silently ignored.
                logits = logits.clone()
                logits[0, list(banned)] = float("-inf")
            if greedy:
                return logits.argmax(dim=-1, keepdim=True)
            logits = logits / max(temperature, 1e-5)
            if repetition_penalty and repetition_penalty != 1.0:
                recent = torch.unique(idx[0, -rep_window:])
                target = logits[0, recent]
                logits[0, recent] = torch.where(
                    target > 0, target / repetition_penalty,
                    target * repetition_penalty)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            if top_p is not None and 0.0 < top_p < 1.0:
                ordered, order = torch.sort(logits, dim=-1, descending=True)
                probs = torch.softmax(ordered, dim=-1)
                cut = torch.cumsum(probs, dim=-1) > top_p
                cut[..., 1:] = cut[..., :-1].clone()
                cut[..., 0] = False
                ordered = ordered.masked_fill(cut, float("-inf"))
                logits = torch.empty_like(logits).scatter(-1, order, ordered)
            if min_p is not None and 0.0 < min_p <= 1.0:
                # Relative floor: a peaked step keeps one token, a flat step
                # keeps many.  This is what top_p cannot express.
                probs = torch.softmax(logits, dim=-1)
                floor = probs.max(dim=-1, keepdim=True).values * min_p
                logits = logits.masked_fill(probs < floor, float("-inf"))
            if typical_p is not None and 0.0 < typical_p < 1.0:
                logp = torch.log_softmax(logits, dim=-1)
                probs = logp.exp()
                entropy = -(probs * logp).sum(dim=-1, keepdim=True)
                surprise = (logp + entropy).abs()
                order = surprise.argsort(dim=-1)
                ordered_logits = logits.gather(-1, order)
                ordered_probs = probs.gather(-1, order)
                cum = torch.cumsum(ordered_probs, dim=-1)
                cut = cum > typical_p
                cut[..., 1:] = cut[..., :-1].clone()
                cut[..., 0] = False
                ordered_logits = ordered_logits.masked_fill(cut, float("-inf"))
                logits = torch.empty_like(logits).scatter(-1, order,
                                                          ordered_logits)
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                # Every candidate was masked (e.g. no_repeat_ngram banned the
                # whole top_k): fall back to the unfiltered distribution
                # rather than sampling from NaN.
                return raw.argmax(dim=-1, keepdim=True)
            return torch.multinomial(probs, num_samples=1)

        past = None
        for _ in range(max_new_tokens):
            if use_cache and past is not None and past[0][0].shape[2] < self.seq_len:
                out = self(idx[:, -1:], past_kv=past, use_cache=True)
                past = out["past_kv"]
            else:
                ctx = idx[:, -self.seq_len:]
                out = self(ctx, use_cache=use_cache)
                past = out.get("past_kv")
            nxt = sample(out["logits"][:, -1, :].float())
            idx = torch.cat([idx, nxt], dim=1)
            if stop_at_eot and int(nxt[0, 0]) == eot_id:
                break
        if was_training:
            self.train()
        return idx

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
