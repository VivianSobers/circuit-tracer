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
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from circuit_tracer.graph import Graph

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


def _projection(model, layer: int, head: int, side: str) -> Tensor:
    """``W_Q diag(w_q)`` or ``W_K diag(w_k)``: the QK-norm gain is a fixed per-channel scaling."""
    attn = model.blocks[layer].attn
    weight = attn.W_Q[head] if side == "query" else attn.W_K[head]
    norm = getattr(attn, "q_norm" if side == "query" else "k_norm", None)
    return weight * norm.w if norm is not None else weight


def to_head_space(
    model,
    run: FrozenScores,
    layer: int,
    head: int,
    directions: Tensor,
    positions: Tensor,
    *,
    side: str,
) -> Tensor:
    """Carry residual-stream directions into one head's rotated query or key space.

    Applies ``ln1`` in full (gain and frozen division), the projection with its QK-norm gain, the
    frozen QK-norm scale and the rotation at each direction's position. Every step is linear with
    the scales frozen, so directions can be projected one at a time and summed afterwards.

    Args:
        directions: ``(n, d_model)``, in residual-stream coordinates before ``ln1``.
        positions: ``(n,)``, the position each direction sits at.
        side: ``"query"`` or ``"key"``.

    Returns:
        ``(n, d_head)``.
    """
    if side not in ("query", "key"):
        raise ValueError(f"side must be 'query' or 'key', got {side!r}")
    if directions.ndim != 2:
        raise ValueError(f"directions must be (n, d_model), got {tuple(directions.shape)}")
    if positions.shape[0] != directions.shape[0]:
        raise ValueError(f"{positions.shape[0]} positions for {directions.shape[0]} directions")
    dtype = directions.dtype
    gain = model.blocks[layer].ln1.w.to(dtype)
    out = directions * gain / run.ln1_scales[layer][positions].to(dtype)
    out = out @ _projection(model, layer, head, side).to(dtype)
    scales = run.query_scales[layer] if side == "query" else run.key_scales[layer]
    if scales is not None:
        index = head if side == "query" else _kv_group(model, head)
        out = out / scales[positions, index].to(dtype)
    if run.rotations is not None:
        out = torch.einsum("na,nab->nb", out, run.rotations[positions].to(dtype))
    return out


def attention_scores(model, run: FrozenScores, layer: int, head: int) -> Tensor:
    """Reconstruct one head's pre-softmax scores from the frozen run, unmasked.

    Uses the same path as the decomposition, so agreement with the model's own scores is what
    licenses reading the decomposition as exact.

    Returns:
        ``(n_pos, n_pos)``, indexed ``[query, key]``.
    """
    require_supported(model)
    residual = run.resid_pre[layer].float()
    positions = torch.arange(run.n_pos, device=residual.device)
    query = to_head_space(model, run, layer, head, residual, positions, side="query")
    key = to_head_space(model, run, layer, head, residual, positions, side="key")
    return query @ key.T / attention_scale(model)


@dataclass(frozen=True)
class SourceSet:
    """Residual-stream directions feeding one attention layer, with where each came from.

    Directions are already scaled by their activations, so summing them by position rebuilds the
    part of the residual stream they account for.
    """

    directions: Tensor
    positions: Tensor
    layers: Tensor
    feature_ids: Tensor

    def __len__(self) -> int:
        return int(self.directions.shape[0])

    def concat(self, other: SourceSet) -> SourceSet:
        return SourceSet(
            directions=torch.cat([self.directions, other.directions.to(self.directions.dtype)]),
            positions=torch.cat([self.positions, other.positions]),
            layers=torch.cat([self.layers, other.layers]),
            feature_ids=torch.cat([self.feature_ids, other.feature_ids]),
        )

    def select(self, keep: Tensor) -> SourceSet:
        return SourceSet(
            directions=self.directions[keep],
            positions=self.positions[keep],
            layers=self.layers[keep],
            feature_ids=self.feature_ids[keep],
        )

    @property
    def is_remainder(self) -> Tensor:
        """Rows standing for what no transcoder feature explains."""
        return self.layers == REMAINDER


def _activations_for(graph: Graph) -> Tensor:
    """Each selected feature's activation, checking which table ``activation_values`` follows."""
    selected = graph.selected_features
    values = graph.activation_values
    if len(values) == len(graph.active_features):
        return values[selected]
    if len(values) == len(selected):
        return values
    raise ValueError(
        f"activation_values has {len(values)} entries, matching neither active_features "
        f"({len(graph.active_features)}) nor selected_features ({len(selected)})"
    )


def feature_sources(model, graph: Graph, *, below_layer: int, dtype=torch.float32) -> SourceSet:
    """Activation-scaled decoder directions of every selected feature written below a layer.

    A per-layer transcoder feature writes at its own block's MLP output, so attention at a later
    block sees it and attention at its own block does not. Directions are extracted in float32 by
    default: the contraction sums many terms that largely cancel, and bfloat16 loses most of the
    result. Decoder rows are gathered one layer at a time, since a lazily loaded decoder can re-read
    the layer from disk on every access.
    """
    active = graph.active_features[graph.selected_features]
    values = _activations_for(graph)
    keep = active[:, 0] < below_layer
    template = model.transcoders[0].W_dec
    device = template.device
    layers = active[keep, 0].to(device)
    positions = active[keep, 1].to(device)
    feature_ids = active[keep, 2].to(device)
    directions = torch.empty((int(keep.sum()), template.shape[-1]), dtype=dtype, device=device)
    for layer in layers.unique().tolist():
        rows = layers == layer
        directions[rows] = model.transcoders[layer].W_dec[feature_ids[rows]].to(dtype)
    directions = directions * values[keep].to(device=device, dtype=dtype).unsqueeze(-1)
    return SourceSet(directions, positions, layers, feature_ids)


def remainder_sources(run: FrozenScores, sources: SourceSet, layer: int) -> SourceSet:
    """What the given sources leave out of the residual entering ``layer``, one row per position.

    Transcoder features cover only MLP writes. Earlier attention outputs, embeddings, transcoder
    errors and decoder biases are in the residual too; carrying them as one lumped direction per
    position keeps the decomposition exhaustive and makes the features' share visible.
    """
    remainder = run.resid_pre[layer].to(torch.float32).clone()
    if len(sources):
        remainder.index_add_(
            0,
            sources.positions.to(remainder.device),
            -sources.directions.to(device=remainder.device, dtype=remainder.dtype),
        )
    seq, device = remainder.shape[0], remainder.device
    marker = torch.full((seq,), REMAINDER, dtype=torch.long, device=device)
    return SourceSet(remainder, torch.arange(seq, device=device), marker, marker.clone())


@dataclass(frozen=True)
class QKAttribution:
    """One head's score from one query position, expanded into (query source, key source) terms."""

    contributions: Tensor
    query_sources: SourceSet
    key_sources: SourceSet
    layer: int
    head: int
    query_position: int

    @property
    def by_key_source(self) -> Tensor:
        return self.contributions.sum(dim=0)

    @property
    def by_query_source(self) -> Tensor:
        return self.contributions.sum(dim=1)

    def by_key_position(self, n_pos: int) -> Tensor:
        """The score at each key position, which compares against a row of the score matrix.

        Accumulated in float32: many terms of both signs land on each position.
        """
        totals = torch.zeros(n_pos, dtype=torch.float32, device=self.contributions.device)
        if len(self.key_sources):
            totals.index_add_(0, self.key_sources.positions, self.by_key_source.float())
        return totals

    def top_pairs(self, count: int) -> tuple[Tensor, Tensor]:
        """The ``count`` largest terms by magnitude, signed, with flat indices.

        Use ``divmod(index, contributions.shape[1])`` to recover the query and key source rows.
        """
        flat = self.contributions.flatten()
        _, indices = flat.abs().topk(min(count, flat.numel()))
        return flat[indices], indices


def qk_attribution(
    model,
    graph: Graph,
    run: FrozenScores,
    layer: int,
    head: int,
    query_position: int,
) -> QKAttribution:
    """Expand one head's score from one query position into source-pair terms.

    Sources are the graph's selected features written below ``layer`` plus one remainder per
    position, so the terms for each key position sum to that entry of the score matrix. Over all
    query positions at once the contraction grows too large to hold, which is why this takes one.

    Args:
        model: A ``ReplacementModel`` on the TransformerLens backend.
        graph: The attribution graph for the prompt ``run`` was captured on.
        run: A :class:`FrozenScores` for that prompt.
        layer: The attention layer.
        head: The query head.
        query_position: The attending position.
    """
    require_supported(model)
    if not 0 <= layer < model.cfg.n_layers:
        raise IndexError(f"layer {layer} out of range for {model.cfg.n_layers} layers")
    if not 0 <= head < model.cfg.n_heads:
        raise IndexError(f"head {head} out of range for {model.cfg.n_heads} heads")
    if not 0 <= query_position < run.n_pos:
        raise IndexError(f"query_position {query_position} out of range for {run.n_pos} positions")

    features = feature_sources(model, graph, below_layer=layer)
    sources = features.concat(remainder_sources(run, features, layer))
    query_sources = sources.select(sources.positions == query_position)
    key_sources = sources.select(sources.positions <= query_position)

    left = to_head_space(
        model, run, layer, head, query_sources.directions, query_sources.positions, side="query"
    )
    right = to_head_space(
        model, run, layer, head, key_sources.directions, key_sources.positions, side="key"
    )
    return QKAttribution(
        contributions=left @ right.T / attention_scale(model),
        query_sources=query_sources,
        key_sources=key_sources,
        layer=layer,
        head=head,
        query_position=query_position,
    )
