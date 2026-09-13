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
