"""Decomposing one attention head's score into pairs of attribution-graph sources.

An attribution graph freezes attention patterns, so it explains what attention moves and says
nothing about why a head attends where it does. The pre-softmax score is still bilinear in the
residual stream with rotary embeddings and query/key normalisation, once the normalisation scales
are frozen the way a graph already freezes layer-norm scales:

    s(p, j) = x_p @ [ W_Q diag(w_q) R(p) R(j).T diag(w_k) W_K.T ] @ x_j
              / ( attn_scale * sigma_q[p] * sigma_k[j] )

``x`` is the residual after ``ln1`` including its gain, ``w_q`` and ``w_k`` are the QK-norm gains,
``R`` is the rotary rotation and the sigmas are the frozen QK-norm RMS scales. Writing the residual
at each position as its transcoder features plus a remainder expands the score into one term per
(query source, key source) pair, and the terms sum to the score exactly.

Example:

    from circuit_tracer.attribution.qk_attribution import FrozenScores, qk_attribution

    run = FrozenScores.from_model(model, graph.input_tokens)
    result = qk_attribution(model, graph, run, layer=14, head=3, query_position=8)
    values, flat = result.top_pairs(10)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: Position schemes that leave the score bilinear in the residual stream.
BILINEAR_POSITION_SCHEMES = frozenset({"standard", "rotary", None})

#: Layer and feature id marking a source that stands for what no feature explains.
REMAINDER = -1


class UnsupportedForQK(RuntimeError):
    """Raised when a model's attention score is not the bilinear form decomposed here."""


def attention_scale(model) -> float:
    """The divisor applied to raw scores: 1 if the model disables it, else ``sqrt(d_head)``."""
    cfg = model.cfg
    if getattr(cfg, "use_attn_scale", True) is False:
        return 1.0
    scale = getattr(cfg, "attn_scale", None)
    return float(scale) if scale else float(model.blocks[0].attn.W_Q.shape[-1] ** 0.5)


def require_supported(model) -> None:
    """Raise unless the decomposition reproduces this model's scores exactly."""
    cfg = model.cfg
    cap = getattr(cfg, "attn_scores_soft_cap", None)
    if cap is not None and float(cap) > 0:
        raise UnsupportedForQK(
            f"attention scores are soft-capped at {cap}; the tanh sits outside the bilinear form"
        )
    scheme = getattr(cfg, "positional_embedding_type", None)
    if scheme not in BILINEAR_POSITION_SCHEMES:
        raise UnsupportedForQK(
            f"positional embedding type {scheme!r} adds a term outside the bilinear form"
        )
    transcoders = getattr(model, "transcoders", None)
    # Checked structurally so the module never imports CrossLayerTranscoder: a per-layer set is
    # indexable by layer and a cross-layer transcoder is not.
    if transcoders is not None and not hasattr(type(transcoders), "__getitem__"):
        raise UnsupportedForQK(
            "a cross-layer transcoder feature writes into several layers, so it has no single "
            "direction entering attention; only per-layer transcoders are handled"
        )
    for layer, block in enumerate(model.blocks):
        if getattr(block.ln1, "b", None) is not None:
            raise UnsupportedForQK(
                f"blocks.{layer}.ln1 has a bias, an additive term no source carries"
            )
        for name in ("b_Q", "b_K"):
            bias = getattr(block.attn, name, None)
            if bias is not None and bool(bias.any()):
                raise UnsupportedForQK(
                    f"{name} is non-zero at layer {layer}; its terms belong to no source, so the "
                    "pairs would not sum to the score"
                )


def _kv_group(model, head: int) -> int:
    """Map a query head to its key/value group; cached key scales carry one slice per group."""
    heads = model.cfg.n_heads
    groups = getattr(model.cfg, "n_key_value_heads", None) or heads
    return head // (heads // groups)


def rotation_matrices(model, n_pos: int) -> Tensor | None:
    """Return the rotary rotation at each position as a matrix, or None without rotary embeddings.

    Built by rotating the standard basis with the model's own ``apply_rotary``, so it follows that
    implementation's convention exactly. The basis is float32 so a half-precision model does not
    lower the precision of every projection that uses it.

    Returns:
        ``(n_pos, d_head, d_head)``, where ``v @ result[p]`` rotates ``v`` at position ``p``.
    """
    if getattr(model.cfg, "positional_embedding_type", None) != "rotary":
        return None
    if n_pos < 1:
        raise ValueError(f"n_pos must be positive, got {n_pos}")
    attn = model.blocks[0].attn
    size = attn.W_Q.shape[-1]
    basis = torch.eye(size, device=attn.W_Q.device, dtype=torch.float32)
    basis = basis.reshape(size, 1, 1, size).expand(size, n_pos, 1, size).contiguous()
    rotated = attn.apply_rotary(basis, 0, None)
    return rotated.squeeze(2).permute(1, 0, 2).contiguous()


def _single(value: Tensor, name: str) -> Tensor:
    if value.ndim != 3:
        raise ValueError(f"{name} should be (batch, pos, dim), got {tuple(value.shape)}")
    if value.shape[0] != 1:
        raise ValueError(
            f"expected one prompt but {name} holds a batch of {value.shape[0]}; a graph describes "
            "a single prompt"
        )
    return value[0]


@dataclass(frozen=True)
class FrozenScores:
    """What an attribution graph holds fixed about attention, from one clean run."""

    resid_pre: list[Tensor]
    ln1_scales: list[Tensor]
    query_scales: list[Tensor | None]
    key_scales: list[Tensor | None]
    rotations: Tensor | None

    @property
    def n_pos(self) -> int:
        return int(self.resid_pre[0].shape[0])

    @classmethod
    def from_model(cls, model, tokens) -> FrozenScores:
        """Run the model once on the graph's prompt and keep the frozen quantities.

        Args:
            model: A ``ReplacementModel`` on the TransformerLens backend.
            tokens: The prompt the graph was attributed on.
        """
        qk_norm = getattr(model.blocks[0].attn, "q_norm", None) is not None
        tails: tuple[str, ...] = ("hook_resid_pre", "ln1.hook_scale")
        if qk_norm:
            tails += ("attn.q_norm.hook_scale", "attn.k_norm.hook_scale")
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith(tails))

        heads = model.cfg.n_heads
        groups = getattr(model.cfg, "n_key_value_heads", None) or heads
        resid, ln1, queries, keys = [], [], [], []
        for layer in range(model.cfg.n_layers):
            residual = _single(cache[f"blocks.{layer}.hook_resid_pre"], "hook_resid_pre")
            n_pos = residual.shape[0]
            resid.append(residual)
            ln1.append(_single(cache[f"blocks.{layer}.ln1.hook_scale"], "ln1.hook_scale"))
            if qk_norm:
                # TransformerLens flattens these hooks over (batch, pos, head).
                q_scale = cache[f"blocks.{layer}.attn.q_norm.hook_scale"]
                k_scale = cache[f"blocks.{layer}.attn.k_norm.hook_scale"]
                queries.append(q_scale.reshape(n_pos, heads, 1))
                keys.append(k_scale.reshape(n_pos, groups, 1))
            else:
                queries.append(None)
                keys.append(None)
        return cls(resid, ln1, queries, keys, rotation_matrices(model, resid[0].shape[0]))
