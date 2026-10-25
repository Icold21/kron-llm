"""KronCompose: the Kronecker-parameterised linear map with depth recurrence.

Core object of the project.  A ``KronCompose`` layer implements

    I_{i,l}   = sum_{k=1..K} A_{i,l,k} (x) B_{i,l,k}          (innovation)
    D_{i,l}   = I_{i,l} + sigma(theta_{i,l}) * sg(D_{i-1,l})   (depth recurrence)
    W_{i,l}   = s * Id + tau * D_{i,l}                         (anchor)
    P_i       = W_{i,L} ( ... ( W_{i,1} ) )                    (L-fold composition)
    y         = gamma_i * P_i x                                (ReZero gate)

where ``i`` is the *model depth* (transformer layer index), ``l`` the
micro-composition channel and ``sg`` the stop-gradient operator.

Scale handling (deviation from the original TZ, see README)
-----------------------------------------------------------
The TZ fixed the orthogonal gain of ``A, B`` to ``1/sqrt(n1*n2) = 1/sqrt(d)``.
That makes ``||I|| ~ 1/d`` -- 128x below the ``s*Id`` anchor at d=128 -- so
``dL/dA``, ``dL/dB`` and ``dL/dtheta`` all vanish and the model is numerically
the identity map (measured: ``theta_grad = 0``, ``eff_rank(W) = d``,
``cos(W_i, W_{i-1}) = 1``), i.e. a context-free unigram model.

Here the innovation is **normalised** before it is used
(``D <- tau * D / rms(D)``), which makes the gain irrelevant and replaces it
with two explicit, learnable magnitudes:

* ``tau``  -- per-channel innovation scale (init ``innovation_scale``),
* ``gamma`` -- per-compose ReZero gate (init ``gate_init``), so a branch can
  start as a small perturbation of the residual stream instead of forcing
  ``||W|| ~ 1`` at initialisation.

Kronecker convention
--------------------
``A`` is ``(n1_out, n1_in)``, ``B`` is ``(n2_out, n2_in)`` and ``x`` is viewed
as ``(n1_in, n2_in)``.  Then

    (A (x) B) x  ==  A @ X @ B.T

with ``X = x.reshape(n1_in, n2_in)`` and row-major flattening of the result.
This is what ``kron_apply`` implements and what test T1 checks.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def kron_apply(A: torch.Tensor, B: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply ``(A (x) B)`` to ``x`` without materialising the Kronecker product.

    Parameters
    ----------
    A : (n1_out, n1_in)
    B : (n2_out, n2_in)
    x : (..., n1_in, n2_in)

    Returns
    -------
    (..., n1_out, n2_out)
    """
    if A.shape[0] == A.shape[1] and B.shape[0] == B.shape[1]:
        return torch.matmul(torch.matmul(A, x), B.transpose(-1, -2))
    return torch.einsum("ij,...jl,ml->...im", A, x, B)


def kron_materialize(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Dense matrix of ``A (x) B`` with row-major index order."""
    n1o, n1i = A.shape
    n2o, n2i = B.shape
    return torch.einsum("ij,ml->imjl", A, B).reshape(n1o * n2o, n1i * n2i)


def effective_rank(singular_values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``exp(entropy(sigma / sum(sigma)))`` -- a smooth rank proxy."""
    s = singular_values.clamp_min(eps)
    p = s / s.sum()
    return torch.exp(-(p * p.log()).sum())


def rope_frequencies(seq_len: int, d_head: int, theta: float,
                     device: Optional[torch.device] = None,
                     dtype: torch.dtype = torch.float32
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device,
                                            dtype=torch.float32) / d_head))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)          # (T, d_head/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary positional embedding.  ``x``: (B, H, T, d_head)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


_EYE_CACHE: dict = {}
_STATE0_CACHE: dict = {}


def cached_eye(d: int, device, dtype) -> torch.Tensor:
    key = (d, str(device), dtype)
    eye = _EYE_CACHE.get(key)
    if eye is None:
        eye = torch.eye(d, device=device, dtype=dtype)
        _EYE_CACHE[key] = eye
    return eye


def rms_norm_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalise every matrix in the trailing two dims to unit RMS."""
    r = x.pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(eps).sqrt()
    return x / r


def _init_gain_value(mode: "str | float", K: int, d_in: int) -> float:
    if mode == "unit":
        return float(K) ** -0.25
    if mode == "spec":
        return 1.0 / math.sqrt(max(d_in, 1))
    return float(mode)


# ---------------------------------------------------------------------------
# KronCompose
# ---------------------------------------------------------------------------
def _largest_factor(n: int) -> int:
    """Largest divisor of ``n`` not exceeding ``sqrt(n)`` (≥ 1)."""
    for k in range(int(n ** 0.5), 0, -1):
        if n % k == 0:
            return k
    return 1


class KronCompose(nn.Module):
    """Kronecker-parameterised linear layer with depth recurrence.

    The depth state ``D`` is *not* owned by the module: it is passed in and
    returned so that the same parameters can be reused at every transformer
    layer while ``D`` keeps evolving along depth (one state per channel and
    per matrix role).
    """

    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        n1_in: int = 8,
        n2_in: int = 16,
        n1_out: Optional[int] = None,
        n2_out: Optional[int] = None,
        L: int = 4,
        K: int = 2,
        K_win: int = 2,
        s_init: float = 1.0,
        s_learnable: bool = False,
        anchor: bool = True,
        stop_gradient: bool = True,
        theta_std: float = 0.1,
        rezero_eps: float = 0.0,
        s_init_effective: bool = True,
        init_gain: "str | float" = "unit",
        innovation_scale: float = 0.1,
        normalize_D: bool = True,
        gate_init: float = 0.1,
        use_gate: bool = True,
        composition: str = "product",
        sum_alpha_init: float = 1.0,
        dynamic_gate: Optional[str] = None,
        groups: int = 1,
        alternate_start: str = "sum",
        nonlinearity: str = "silu",
        norm_between: bool = False,
        nonlinear_every: int = 1,
        nonlinear_mode: str = "replace",
        sigma_mode: str = "learned",
        sigma_value: float = 0.5,
    ) -> None:
        super().__init__()
        d_out = d_in if d_out is None else d_out
        n1_out = n1_in if n1_out is None else n1_out
        n2_out = n2_in if n2_out is None else n2_out

        assert n1_in * n2_in == d_in, (n1_in, n2_in, d_in)
        assert n1_out * n2_out == d_out, (n1_out, n2_out, d_out)

        self.d_in, self.d_out = d_in, d_out
        self.n1_in, self.n2_in = n1_in, n2_in
        self.n1_out, self.n2_out = n1_out, n2_out
        self.L, self.K, self.K_win = L, K, K_win
        self.init_gain = init_gain
        self.square = d_in == d_out
        # L-fold composition is only defined for square maps.
        assert self.square or L == 1, "non-square KronCompose requires L == 1"

        self.composition = str(composition).lower()
        if self.composition not in {"product", "sum", "grouped", "alternating",
                                    "nonlinear"}:
            raise ValueError(
                "composition must be 'product', 'sum', 'grouped', "
                f"'alternating' or 'nonlinear', got {composition!r}")
        self.nonlinearity = str(nonlinearity).lower()
        if self.nonlinearity not in {"silu", "gelu", "tanh"}:
            raise ValueError("nonlinearity must be silu/gelu/tanh")
        self.norm_between = bool(norm_between)
        self.nonlinear_every = int(nonlinear_every)
        if not 1 <= self.nonlinear_every <= L:
            raise ValueError("nonlinear_every must be in [1, L]")
        self.nonlinear_mode = str(nonlinear_mode).lower()
        if self.nonlinear_mode not in {"replace", "add"}:
            raise ValueError("nonlinear_mode must be 'replace' or 'add'")
        self.groups = int(groups)
        if self.composition == "grouped":
            if self.groups < 1 or L % self.groups != 0:
                raise ValueError(
                    f"groups={groups} must divide L={L}")
        else:
            self.groups = 1
        self.alternate_start = str(alternate_start).lower()
        if self.alternate_start not in {"sum", "product"}:
            raise ValueError("alternate_start must be 'sum' or 'product'")
        self.sum_alpha_init = float(sum_alpha_init)
        self.dynamic_gate = (str(dynamic_gate).lower() if dynamic_gate else None)
        if self.dynamic_gate is not None:
            if self.dynamic_gate not in {"dense", "kron"}:
                raise ValueError(
                    "dynamic_gate must be None/'dense'/'kron', got "
                    f"{dynamic_gate!r}")
            if self.composition != "sum":
                raise ValueError("dynamic_gate requires composition='sum'")
            if L < 2:
                raise ValueError("dynamic_gate requires L >= 2")

        self.anchor = bool(anchor) and self.square
        self.stop_gradient = stop_gradient
        self.rezero_eps = rezero_eps
        self.normalize_D = bool(normalize_D)
        self.innovation_scale = float(innovation_scale)
        # Depth-recurrence strength: sigmoid(theta) by default.  "off" removes
        # the carry (independent factors), "fixed" pins it to sigma_value and
        # freezes theta.  Both exist as ablations, not as design choices.
        self.sigma_mode = str(sigma_mode).lower()
        if self.sigma_mode not in {"learned", "fixed", "off"}:
            raise ValueError("sigma_mode must be 'learned', 'fixed' or 'off'")
        self.sigma_value = float(sigma_value)

        self.A = nn.Parameter(torch.empty(L, K, n1_out, n1_in))
        self.B = nn.Parameter(torch.empty(L, K, n2_out, n2_in))
        self.theta = nn.Parameter(torch.randn(L) * theta_std)
        # explicit innovation magnitude (replaces the implicit factor gain)
        self.tau = nn.Parameter(torch.full((L,), float(innovation_scale)))
        # ReZero gate: the branch starts as a small perturbation
        self.gate = nn.Parameter(torch.tensor(float(gate_init))) if use_gate else None
        # Per-channel mixing coefficients for the sum composition.  The
        # product composition keeps ``alpha=None`` so old checkpoints and the
        # parameter count of the product runs stay unchanged.
        self.alpha = (nn.Parameter(torch.full((L,), self.sum_alpha_init))
                      if self.composition in {"sum", "grouped", "alternating"}
                      else None)

        # Input-dependent softmax gate over the L sum-composition terms:
        #   w_t = L * softmax(g(x_t)),  P_t = sI + sum_l w_{t,l} D_l.
        # The L*softmax normalisation keeps the mean weight at 1, so at
        # initialisation (zero gate) it is exactly the static alpha=1 sum.
        if self.dynamic_gate == "dense":
            self.w_gate = nn.Linear(d_in, L, bias=True)
            nn.init.zeros_(self.w_gate.weight)
            nn.init.zeros_(self.w_gate.bias)
        elif self.dynamic_gate == "kron":
            g1 = _largest_factor(L)
            self.gate_shape = (g1, L // g1)
            self.w_gate_A = nn.Parameter(torch.zeros(g1, n1_in))
            self.w_gate_B = nn.Parameter(torch.zeros(L // g1, n2_in))
            self.w_gate_b = nn.Parameter(torch.zeros(L))
        if self.dynamic_gate is not None:
            self.register_buffer("_last_w_mean", torch.ones(L),
                                 persistent=False)

        if self.anchor:
            # A sum has no L-fold product: s_init is already the coefficient of
            # the identity term.  A grouped composition is a product of G sums,
            # so s_init is also the *effective* anchor (each group gets its
            # G-th root at composition time).
            if self.composition in {"sum", "grouped", "alternating"}:
                s0 = float(s_init)
            else:
                s0 = (float(s_init) ** (1.0 / L)
                      if s_init_effective else float(s_init))
            if s_learnable:
                self.s = nn.Parameter(torch.tensor(s0))
            else:
                self.register_buffer("s", torch.tensor(s0))
        else:
            self.s = None

        self.reset_parameters()

    # -- init -------------------------------------------------------------
    def reset_parameters(self) -> None:
        """Orthogonal factors.

        With ``normalize_D`` the gain is irrelevant for the forward pass (it is
        normalised away); ``init_gain`` is kept so the TZ recipe can be
        reproduced for the ablation table.
        """
        gain = _init_gain_value(self.init_gain, self.K, self.d_in)
        nn.init.orthogonal_(self.A, gain=gain)
        nn.init.orthogonal_(self.B, gain=gain)
        with torch.no_grad():
            self.theta.normal_(0.0, 0.1)
            self.tau.fill_(self.innovation_scale)
            if self.alpha is not None:
                self.alpha.fill_(self.sum_alpha_init)

    def sigma_values(self) -> torch.Tensor:
        """Depth-carry coefficient for each composition step, shape ``(L,)``.

        ``learned`` uses ``sigmoid(theta)``; ``fixed`` pins the value (the
        critic's "sigma frozen at 0.5" ablation); ``off`` removes the carry so
        the factors become independent.  ``theta`` is kept in the state dict in
        every mode, so the parameter count is unchanged across the ablation.
        """
        if self.sigma_mode == "off":
            return torch.zeros(self.L, device=self.theta.device,
                               dtype=self.theta.dtype)
        if self.sigma_mode == "fixed":
            return torch.full((self.L,), self.sigma_value,
                              device=self.theta.device, dtype=self.theta.dtype)
        return torch.sigmoid(self.theta)

    @property
    def n_params(self) -> int:
        total = self.A.numel() + self.B.numel() + self.theta.numel() + self.tau.numel()
        if isinstance(self.s, nn.Parameter):
            total += 1
        if self.gate is not None:
            total += 1
        if self.alpha is not None:
            total += self.alpha.numel()
        if self.dynamic_gate == "dense":
            total += sum(p.numel() for p in self.w_gate.parameters())
        elif self.dynamic_gate == "kron":
            total += (self.w_gate_A.numel() + self.w_gate_B.numel()
                      + self.w_gate_b.numel())
        return total

    # -- state ------------------------------------------------------------
    def initial_state(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """``D_{-1,l} = eps * Id`` (small start at a window boundary).

        Cached: the tensor is constant and never mutated (the recurrence
        creates a new tensor every layer).
        """
        key = (self.L, self.d_out, self.d_in, self.rezero_eps,
               str(device), dtype, self.square)
        cached = _STATE0_CACHE.get(key)
        if cached is None:
            D = torch.zeros(self.L, self.d_out, self.d_in, device=device,
                            dtype=dtype)
            if self.square and self.rezero_eps:
                D = D + self.rezero_eps * cached_eye(self.d_in, device, dtype)
            _STATE0_CACHE[key] = D
            cached = D
        return cached

    def innovation(self) -> torch.Tensor:
        """``I_l = sum_k A_k (x) B_k`` as dense ``(L, d_out, d_in)``.

        Explicit broadcast outer product instead of a 5-index einsum: the
        contraction is tiny, but einsum's path selection made this one of the
        most expensive ops in the forward pass.
        """
        L, K, n1o, n1i = self.A.shape
        _, _, n2o, n2i = self.B.shape
        outer = (self.A.unsqueeze(-1).unsqueeze(-1)
                 * self.B.unsqueeze(2).unsqueeze(2))       # (L,K,n1o,n1i,n2o,n2i)
        I4 = outer.sum(dim=1)                               # (L,n1o,n1i,n2o,n2i)
        return I4.permute(0, 1, 3, 2, 4).reshape(
            self.L, n1o * n2o, n1i * n2i)

    def apply_innovation(self, x: torch.Tensor, l: int = 0) -> torch.Tensor:
        """Apply ``I_l = sum_k A_k (x) B_k`` to ``x`` without building ``d x d``.

        Uses ``(A (x) B) vec(X) = A X B^T`` per summand, i.e.
        ``O(K * (n1_out*n1_in*n2_in + n2_out*n2_in*n1_out))`` instead of the
        dense ``O(d_in*d_out)``.  With square factors ``n1=n2=sqrt(d)`` the
        saving per summand is ``O(d^1.5)``; with ``K`` summands the crossover is
        ``K * 2 * d^1.5 < d^2`` (K=6, d=320: ~1.5x).  The depth recurrence
        still carries a dense ``prev`` term, so the full ``D_l`` application is
        not cheaper unless the state is factorised too -- see the benchmark in
        docs/architecture.md.
        """
        tokens = x.shape[0]
        X = x.reshape(tokens, self.n1_in, self.n2_in)
        y = torch.zeros(tokens, self.n1_out, self.n2_out,
                        device=x.device, dtype=x.dtype)
        for k in range(self.K):
            y = y + torch.einsum("ip,bpq,jq->bij", self.A[l, k], X,
                                 self.B[l, k])
        return y.reshape(tokens, self.d_out)

    # -- composition ------------------------------------------------------
    def compose_product(self, D: torch.Tensor) -> torch.Tensor:
        """``P = W_L ... W_1`` with ``W_l = s Id + D_l``."""
        s = self.s
        if s is None:
            P = D[0]
            for l in range(1, self.L):
                P = torch.matmul(D[l], P)
            return P
        sI = s * cached_eye(self.d_in, D.device, D.dtype)
        P = sI + D[0]
        for l in range(1, self.L):
            # (s Id + D_l) @ P = s*P + D_l @ P
            P = torch.addmm(s * P, D[l], P)
        return P

    def compose_sum(self, D: torch.Tensor,
                    weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``P = s Id + sum_l w_l D_l`` (sum composition).

        ``w`` is ``alpha`` for the static sum, or the last mean softmax gate
        weights for the dynamic variant (diagnostics only; the forward pass
        applies the per-token weights directly).
        """
        assert self.alpha is not None
        w = (weights if weights is not None else
             (self._last_w_mean if self.dynamic_gate is not None else self.alpha))
        P = w[0] * D[0]
        for l in range(1, self.L):
            P = P + w[l] * D[l]
        if self.s is not None:
            P = P + self.s * cached_eye(self.d_in, D.device, D.dtype)
        return P

    def compose_grouped(self, D: torch.Tensor) -> torch.Tensor:
        """Grouped composition: sum inside groups, product across groups.

        With ``L`` factors split into ``G`` groups of size ``L/G``:

            S_g = s^(1/G) I + sum_{l in g} alpha_l D_l
            P   = S_G ... S_1

        ``G=1`` is exactly the sum composition; ``G=L`` is exactly the product
        (with ``alpha=1``), so the family interpolates between the two.  The
        multiplicative depth is ``G-1`` instead of ``L-1``: less conditioning
        and rank collapse than the product, more interaction than the sum.
        """
        G = self.groups
        m = self.L // G
        sg = (self.s ** (1.0 / G)) if self.s is not None else None
        eye = (cached_eye(self.d_in, D.device, D.dtype)
               if sg is not None else None)
        P: Optional[torch.Tensor] = None
        for g in range(G):
            start = g * m
            S = self.alpha[start] * D[start]
            for l in range(start + 1, start + m):
                S = S + self.alpha[l] * D[l]
            if sg is not None:
                S = S + sg * eye
            P = S if P is None else S @ P
        assert P is not None
        return P

    def compose_alternating(self, D: torch.Tensor) -> torch.Tensor:
        """Alternate sum and product at every micro-depth step.

        Starting from ``P = sI``, each factor applies one operation:

            l even (with ``alternate_start='sum'``): P <- P + alpha_l D_l
            l odd:                                   P <- P @ (I + alpha_l D_l)

        The anchor is applied once at the start, so at ``D=0`` the map is
        exactly ``sI``.  This is a sum-product network along depth: additive
        steps raise the rank, multiplicative steps compose, and the
        multiplicative depth is about ``L/2`` instead of ``L-1``.
        """
        P = (self.s * cached_eye(self.d_in, D.device, D.dtype)
             if self.s is not None
             else torch.zeros(self.d_in, self.d_in, device=D.device,
                              dtype=D.dtype))
        eye = cached_eye(self.d_in, D.device, D.dtype)
        for l in range(self.L):
            is_sum = ((l % 2 == 0) == (self.alternate_start == "sum"))
            if is_sum:
                P = P + self.alpha[l] * D[l]
            else:
                P = P @ (eye + self.alpha[l] * D[l])
        return P

    def compose(self, D: torch.Tensor) -> torch.Tensor:
        """Compose the depth state with the configured operation."""
        if self.composition == "sum":
            return self.compose_sum(D)
        if self.composition == "grouped":
            return self.compose_grouped(D)
        if self.composition == "alternating":
            return self.compose_alternating(D)
        if self.composition == "nonlinear":
            # No single matrix exists; return the linearised product for the
            # spectral diagnostics only.
            return self.compose_product(D)
        return self.compose_product(D)

    @property
    def anchor_term(self) -> Optional[torch.Tensor]:
        """Identity coefficient in the composed map (``s`` or ``s**L``)."""
        if self.s is None:
            return None
        if self.composition == "sum":
            return self.s
        if self.composition in {"grouped", "alternating"}:
            return self.s
        return self.s ** self.L

    @torch.no_grad()
    def materialize(self, D: torch.Tensor) -> torch.Tensor:
        """Dense ``(L, d, d)`` factor matrices ``W_l = s Id + D_l``."""
        if not self.anchor:
            return D.clone()
        return self.s * cached_eye(self.d_in, D.device, D.dtype) + D

    # -- dynamic softmax gate ---------------------------------------------
    def _gate_weights(self, xf: torch.Tensor) -> torch.Tensor:
        """Per-token weights ``L * softmax(g(x))``; mean 1, uniform at init."""
        if self.dynamic_gate == "dense":
            logits = self.w_gate(xf)
        else:
            xm = xf.reshape(xf.shape[0], self.n1_in, self.n2_in)
            z = torch.einsum("bpq,ip,jq->bij", xm,
                             self.w_gate_A, self.w_gate_B)
            logits = z.reshape(xf.shape[0], self.L) + self.w_gate_b
        return self.L * torch.softmax(logits, dim=-1)

    def _dynamic_sum(self, xf: torch.Tensor, D: torch.Tensor,
                     s: torch.Tensor) -> torch.Tensor:
        """``y = s x + sum_l w_l(x) (x D_l^T)`` (checkpointed unit)."""
        w = self._gate_weights(xf)
        y = xf * s
        for l in range(self.L):
            y = y + w[:, l:l + 1] * torch.matmul(xf, D[l].transpose(0, 1))
        return y

    def _nonlinear_forward(self, xf: torch.Tensor, D: torch.Tensor,
                          s: torch.Tensor) -> torch.Tensor:
        """Nonlinearity every ``nonlinear_every`` factors.

        Linear factors inside a block are still fused into one matrix, so the
        cost is ``ceil(L / k)`` GEMMs per token instead of ``L``.  ``k=L`` is a
        single nonlinearity after the whole composition (almost free).

        ``nonlinear_mode="replace"`` maps the block to ``y + act(P y - a y)``,
        which cannot reproduce the pure linear map.  ``"add"`` keeps the linear
        skeleton and adds the activated deviation as a residual branch,
        ``P y + act(P y - a y)``, so the product composition is the limit of
        vanishing ``D`` and the nonlinearity can only add capacity.
        """
        y = xf
        k = self.nonlinear_every
        eye = cached_eye(self.d_in, xf.device, xf.dtype)
        for start in range(0, self.L, k):
            idx = list(range(start, min(start + k, self.L)))
            b = len(idx)
            if self.s is not None:
                P = s * eye + D[idx[0]]
                for l in idx[1:]:
                    P = (s * eye + D[l]) @ P
                anchor = s ** b
            else:
                P = D[idx[0]]
                for l in idx[1:]:
                    P = D[l] @ P
                anchor = None
            u = torch.matmul(y, P.transpose(0, 1))
            z = u - anchor * y if anchor is not None else u
            if self.nonlinearity == "silu":
                z = F.silu(z)
            elif self.nonlinearity == "gelu":
                z = F.gelu(z)
            else:
                z = torch.tanh(z)
            y = (u + z) if self.nonlinear_mode == "add" else (y + z)
            if self.norm_between:
                y = F.rms_norm(y, (self.d_in,), None, 1e-6)
        return y

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        prev_D: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        mode: str = "auto",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(y, D_new)``.  ``prev_D`` is the depth state of the
        previous transformer layer (or ``None`` at a window boundary)."""
        B, T, _ = x.shape
        xm = x.reshape(B * T, self.n1_in, self.n2_in)

        # innovation (position independent: it is a weight, not an activation)
        I = self.innovation()                      # (L, d_out, d_in)

        # recurrence, truncated at window boundaries
        in_window = prev_D is not None and not (
            self.K_win and self.K_win > 0 and layer_idx % self.K_win == 0)
        if in_window:
            prev = prev_D.detach() if self.stop_gradient else prev_D
        else:
            prev = self.initial_state(I.device, I.dtype)

        if self.normalize_D:
            I = rms_norm_matrix(I)
            if in_window:
                prev = rms_norm_matrix(prev)

        sig = self.sigma_values()                  # (L,)
        D = I + sig.view(self.L, 1, 1) * prev       # (L, d_out, d_in)
        if self.normalize_D:
            D = rms_norm_matrix(D)
        D = self.tau.view(self.L, 1, 1) * D

        # y = gamma * W_L ... W_1 x
        # Two equivalent execution modes:
        #  * "sequential": L matvec products (best for single-token decoding)
        #  * "matrix":     build P = W_L...W_1 once, then one GEMM.  The
        #                  product costs L*d^3 but is amortised over all
        #                  tokens in the batch, so for B*T >> L*d it is ~L x
        #                  cheaper in FLOPs than the sequential form.
        tokens = B * T
        if self.composition == "nonlinear":
            xf = xm.reshape(tokens, self.d_in)
            s = (self.s if self.s is not None
                 else torch.zeros((), device=xf.device, dtype=xf.dtype))
            if self.training:
                # L sequential matvecs per token; checkpoint so the L
                # intermediate activations are recomputed, not retained.
                y = checkpoint(self._nonlinear_forward, xf, D, s,
                               use_reentrant=False)
            else:
                y = self._nonlinear_forward(xf, D, s)
        elif self.composition == "sum":
            if self.dynamic_gate is not None:
                xf = xm.reshape(tokens, self.d_in)
                if self.training:
                    # The dynamic path cannot use KronBank (P depends on the
                    # token) and L weighted GEMMs otherwise retain all their
                    # activations; checkpoint recomputes them in the backward.
                    s = (self.s if self.s is not None
                         else torch.zeros((), device=xf.device, dtype=xf.dtype))
                    y = checkpoint(self._dynamic_sum, xf, D, s,
                                   use_reentrant=False)
                else:
                    s = (self.s if self.s is not None
                         else torch.zeros((), device=xf.device, dtype=xf.dtype))
                    y = self._dynamic_sum(xf, D, s)
                    with torch.no_grad():
                        self._last_w_mean = self._gate_weights(
                            xf).mean(dim=0).detach()
            else:
                # The sum is cheap even for a single token, so there is no
                # reason to keep a separate sequential path.
                P = self.compose_sum(D)
                y = torch.matmul(xm.reshape(tokens, self.d_in), P.transpose(0, 1))
        else:
            if self.composition in {"grouped", "alternating"}:
                # These compositions have no cheap per-factor sequential
                # fallback; build the composed map and apply it once.
                P = self.compose(D)
                y = torch.matmul(xm.reshape(tokens, self.d_in), P.transpose(0, 1))
            else:
                if mode == "auto":
                    use_matrix = (self.square and self.L > 1
                                  and tokens >= max(256, self.d_in))
                else:
                    use_matrix = mode == "matrix" and self.square and self.L > 1
                if use_matrix:
                    P = self.compose_product(D)
                    y = torch.matmul(xm.reshape(tokens, self.d_in),
                                     P.transpose(0, 1))
                elif self.anchor:
                    s = self.s
                    y = xm.reshape(tokens, self.d_in)
                    for l in range(self.L):
                        y = torch.addmm(s * y, y, D[l].transpose(0, 1))
                else:
                    y = xm.reshape(tokens, self.d_in)
                    for l in range(self.L):
                        y = torch.matmul(y, D[l].transpose(0, 1))

        y = y.reshape(B, T, self.d_out)
        if self.gate is not None:
            y = y * self.gate
        return y, D

    def extra_repr(self) -> str:
        s = "None" if self.s is None else f"{float(self.s):.4g}"
        gate = "None" if self.gate is None else f"{float(self.gate):.4g}"
        alpha = ("" if self.alpha is None
                 else f", alpha={float(self.alpha.mean()):.4g}")
        return (f"d={self.d_in}->{self.d_out}, L={self.L}, K={self.K}, "
                f"K_win={self.K_win}, anchor={self.anchor}, "
                f"composition={self.composition}, s={s}, "
                f"tau={float(self.tau.mean()):.4g}, gate={gate}"
                f"{alpha}"
                + (f", dyn_gate={self.dynamic_gate}" if self.dynamic_gate else ""))


# ---------------------------------------------------------------------------
# KronBank -- batched evaluation of many composes at once
# ---------------------------------------------------------------------------
class KronBank:
    """Evaluate a set of same-shaped ``KronCompose`` modules as batched ops.

    The modules keep ownership of their parameters; the bank only *reads* them
    (``torch.stack`` is differentiable), so ``state_dict``, the ablation table
    and the reference per-module path keep working unchanged.

    Why this exists
    ---------------
    A 12-layer model has 60 composes, each with its own innovation, recurrence,
    ``L``-fold product and apply.  Implemented one by one that is ~900 kernel
    launches per forward pass; on Windows/WDDM a launch costs ~45 us, so the
    step time was 117 ms *independently of the batch size* (measure: batch 1
    and batch 32 took the same time).  The bank collapses the weight-only part
    (innovation, recurrence, products -- everything that does not depend on the
    activations) into ~20 batched launches.
    """

    def __init__(self, mods: Sequence[KronCompose], layers: Sequence[int],
                 keys: Optional[Sequence[Any]] = None) -> None:
        assert len(mods) == len(layers) and len(mods) > 0
        m0 = mods[0]
        for m in mods:
            same = (m.L == m0.L and m.K == m0.K and m.d_in == m0.d_in
                    and m.d_out == m0.d_out and m.n1_in == m0.n1_in
                    and m.n2_in == m0.n2_in and m.n1_out == m0.n1_out
                    and m.n2_out == m0.n2_out and m.anchor == m0.anchor
                    and m.normalize_D == m0.normalize_D
                    and m.stop_gradient == m0.stop_gradient
                    and m.K_win == m0.K_win
                    and m.sigma_mode == m0.sigma_mode
                    and m.sigma_value == m0.sigma_value
                    and (m.gate is None) == (m0.gate is None))
            if not same:
                raise ValueError("KronBank modules must share shape and flags")
        self.mods = list(mods)
        self.layers = list(layers)
        self.keys = list(keys) if keys is not None else list(range(len(mods)))
        self.M = len(mods)
        self.L, self.K = m0.L, m0.K
        self.d_in, self.d_out = m0.d_in, m0.d_out
        self.n1_in, self.n2_in = m0.n1_in, m0.n2_in
        self.n1_out, self.n2_out = m0.n1_out, m0.n2_out
        self.K_win = m0.K_win
        self.stop_gradient = m0.stop_gradient
        self.normalize_D = m0.normalize_D
        self.anchor = m0.anchor
        self.sigma_mode = m0.sigma_mode
        self.sigma_value = m0.sigma_value
        self.composition = m0.composition
        self._m0 = m0
        assert all(m.composition == self.composition for m in mods), \
            "KronBank requires one composition mode per group"
        assert all(self.layers[i] <= self.layers[i + 1]
                   for i in range(len(self.layers) - 1)), "layers must be sorted"
        # contiguous (layer, start, stop) ranges
        self.bounds: List[Tuple[int, int, int]] = []
        start = 0
        for j in range(1, len(self.layers) + 1):
            if j == len(self.layers) or self.layers[j] != self.layers[start]:
                self.bounds.append((self.layers[start], start, j))
                start = j

    # -- batched weight-only prelude --------------------------------------
    def prelude(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(P, D)``.

        ``P``: (M, d_out, d_in) -- the full composed map *including* the
        ReZero gate, i.e. ``y = x @ P.T``.
        ``D``: (M, L, d_out, d_in) -- the depth state (for diagnostics).
        """
        mods, M, L, K = self.mods, self.M, self.L, self.K
        n1o, n1i, n2o, n2i = self.n1_out, self.n1_in, self.n2_out, self.n2_in

        A = torch.stack([m.A for m in mods])            # (M,L,K,n1o,n1i)
        B = torch.stack([m.B for m in mods])            # (M,L,K,n2o,n2i)
        tau = torch.stack([m.tau for m in mods])        # (M,L)

        # innovation  I[m,l,i,j,m2,n2] = sum_k A[m,l,k,i,j] B[m,l,k,m2,n2]
        a = A.reshape(M * L, K, n1o * n1i).transpose(1, 2)
        b = B.reshape(M * L, K, n2o * n2i)
        I = torch.bmm(a, b).reshape(M, L, n1o, n1i, n2o, n2i)
        I = I.permute(0, 1, 2, 4, 3, 5).reshape(
            M, L, self.d_out, self.d_in).contiguous()

        if self.normalize_D:
            I = rms_norm_matrix(I)

        sig = torch.stack([m.sigma_values() for m in mods]).view(M, L, 1, 1)
        init = self._m0.initial_state(I.device, I.dtype)      # (L,d,d), cached

        Ds: List[torch.Tensor] = []
        for k, (layer, lo, hi) in enumerate(self.bounds):
            in_window = k > 0 and not (
                self.K_win and self.K_win > 0 and layer % self.K_win == 0)
            if in_window:
                prev = Ds[-1]
                if self.stop_gradient:
                    prev = prev.detach()
                if self.normalize_D:
                    prev = rms_norm_matrix(prev)
            else:
                prev = init.unsqueeze(0).expand(hi - lo, L, self.d_out,
                                                self.d_in)
            Di = I[lo:hi] + sig[lo:hi] * prev
            if self.normalize_D:
                Di = rms_norm_matrix(Di)
            Di = tau[lo:hi].view(-1, L, 1, 1) * Di
            Ds.append(Di)
        D = torch.cat(Ds, dim=0)

        if self.composition == "sum":
            alpha = torch.stack([m.alpha for m in mods])    # (M,L)
            P = (alpha.view(M, L, 1, 1) * D).sum(dim=1)
            if self.anchor:
                s = torch.stack([m.s for m in mods])       # (M,)
                eye = cached_eye(self.d_in, D.device, D.dtype)
                P = P + s.view(M, 1, 1) * eye
        elif self.anchor:
            s = torch.stack([m.s for m in mods])           # (M,)
            eye = cached_eye(self.d_in, D.device, D.dtype)
            sI = s.view(M, 1, 1) * eye
            P = sI + D[:, 0]
            for l in range(1, L):
                P = torch.baddbmm(s.view(M, 1, 1) * P, D[:, l], P)
        else:
            P = D[:, 0]
            for l in range(1, L):
                P = torch.bmm(D[:, l], P)

        gates = [m.gate for m in mods]
        if gates[0] is not None:
            g = torch.stack(gates).view(M, 1, 1)
            P = P * g
        return P, D
