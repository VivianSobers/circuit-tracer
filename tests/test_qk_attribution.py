"""Tests for decomposing an attention score into pairs of attribution-graph sources.

A stub stands in for a ReplacementModel: random weights, a genuine rotary implementation, QK-norm
gains and grouped-query keys. ``reference_scores`` computes scores the way a forward pass does
(project, normalise, rotate, dot) without touching the module, so agreement with it is evidence
rather than a restatement.
"""

from types import SimpleNamespace

import pytest
import torch

from circuit_tracer.attribution import qk_attribution as qk

N_LAYERS, N_HEADS, D_MODEL, D_HEAD, N_POS, D_TRANSCODER = 3, 4, 8, 4, 6, 10


class StubAttention(SimpleNamespace):
    def apply_rotary(self, x, past_kv_pos_offset=0, attention_mask=None):
        """Rotate ``[batch, pos, head, d_head]`` by a per-position angle, pairing adjacent dims."""
        positions = torch.arange(x.shape[1], dtype=torch.float32) + past_kv_pos_offset
        angle = positions[:, None] * self.freqs[None, :]
        cos, sin = angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out


class StubTranscoders(list):
    """Indexable by layer, as a per-layer TranscoderSet is."""


def rms(x: torch.Tensor) -> torch.Tensor:
    return (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()


def make_model(
    *,
    rotary: bool = True,
    qk_norm: bool = True,
    kv_heads: int = 2,
    soft_cap: float = -1.0,
    scheme: str | None = None,
    ln1_bias: bool = False,
    attn_bias: bool = False,
    use_attn_scale: bool = True,
    seed: int = 0,
) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    group = N_HEADS // kv_heads
    freqs = torch.rand(D_HEAD // 2, generator=generator)
    blocks = []
    for _ in range(N_LAYERS):
        attn = StubAttention(
            W_Q=randn(N_HEADS, D_MODEL, D_HEAD),
            W_K=randn(kv_heads, D_MODEL, D_HEAD).repeat_interleave(group, dim=0),
            b_Q=randn(N_HEADS, D_HEAD) if attn_bias else torch.zeros(N_HEADS, D_HEAD),
            b_K=torch.zeros(N_HEADS, D_HEAD),
            freqs=freqs,
        )
        if qk_norm:
            attn.q_norm = SimpleNamespace(w=torch.rand(D_HEAD, generator=generator) + 0.5)
            attn.k_norm = SimpleNamespace(w=torch.rand(D_HEAD, generator=generator) + 0.5)
        ln1 = SimpleNamespace(
            w=torch.rand(D_MODEL, generator=generator) + 0.5,
            b=randn(D_MODEL) if ln1_bias else None,
        )
        blocks.append(SimpleNamespace(attn=attn, ln1=ln1))
    cfg = SimpleNamespace(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_key_value_heads=kv_heads,
        positional_embedding_type=scheme or ("rotary" if rotary else "standard"),
        attn_scores_soft_cap=soft_cap,
        attn_scale=None,
        use_attn_scale=use_attn_scale,
    )
    model = SimpleNamespace(
        cfg=cfg,
        blocks=blocks,
        transcoders=StubTranscoders(
            SimpleNamespace(W_dec=randn(D_TRANSCODER, D_MODEL)) for _ in range(N_LAYERS)
        ),
        residuals=[randn(N_POS, D_MODEL) for _ in range(N_LAYERS)],
    )
    model.run_with_cache = lambda tokens, names_filter=None: (None, reference_cache(model))
    return model


def attention_in(model: SimpleNamespace, layer: int) -> torch.Tensor:
    x = model.residuals[layer]
    return x / rms(x) * model.blocks[layer].ln1.w


def reference_cache(model: SimpleNamespace) -> dict:
    """What a single-prompt run caches, with TransformerLens's shapes."""
    cache = {}
    group = N_HEADS // model.cfg.n_key_value_heads
    for layer, block in enumerate(model.blocks):
        x = model.residuals[layer]
        cache[f"blocks.{layer}.hook_resid_pre"] = x[None]
        cache[f"blocks.{layer}.ln1.hook_scale"] = rms(x)[None]
        if getattr(block.attn, "q_norm", None) is not None:
            inp = attention_in(model, layer)
            q = torch.einsum("pd,hde->phe", inp, block.attn.W_Q)
            k = torch.einsum("pd,hde->phe", inp, block.attn.W_K[::group])
            # TransformerLens flattens these over (batch, pos, head).
            cache[f"blocks.{layer}.attn.q_norm.hook_scale"] = rms(q).reshape(-1, 1)
            cache[f"blocks.{layer}.attn.k_norm.hook_scale"] = rms(k).reshape(-1, 1)
    return cache


def reference_scores(model: SimpleNamespace, layer: int, head: int) -> torch.Tensor:
    """Project, normalise, rotate, dot: the order a forward pass uses."""
    attn = model.blocks[layer].attn
    inp = attention_in(model, layer)
    q, k = inp @ attn.W_Q[head], inp @ attn.W_K[head]
    if getattr(attn, "q_norm", None) is not None:
        q, k = q / rms(q) * attn.q_norm.w, k / rms(k) * attn.k_norm.w
    if model.cfg.positional_embedding_type == "rotary":
        q = attn.apply_rotary(q[None, :, None, :])[0, :, 0]
        k = attn.apply_rotary(k[None, :, None, :])[0, :, 0]
    scale = 1.0 if model.cfg.use_attn_scale is False else D_HEAD**0.5
    return q @ k.T / scale


def make_graph(active, selected, activations) -> SimpleNamespace:
    """Only the fields the module reads from a circuit-tracer Graph."""
    return SimpleNamespace(
        active_features=torch.tensor(active),
        selected_features=torch.tensor(selected),
        activation_values=torch.tensor(activations),
        input_tokens=torch.arange(N_POS),
    )


def test_soft_capped_scores_are_refused():
    with pytest.raises(qk.UnsupportedForQK, match="soft-capped"):
        qk.require_supported(make_model(soft_cap=50.0))


def test_a_position_scheme_outside_the_bilinear_form_is_refused():
    with pytest.raises(qk.UnsupportedForQK, match="positional embedding type"):
        qk.require_supported(make_model(scheme="alibi"))


def test_cross_layer_transcoders_are_refused():
    """A cross-layer feature writes into several layers, so it has no single direction here."""
    model = make_model()
    model.transcoders = object()
    with pytest.raises(qk.UnsupportedForQK, match="cross-layer"):
        qk.require_supported(model)


def test_an_ln1_bias_is_refused():
    with pytest.raises(qk.UnsupportedForQK, match="ln1"):
        qk.require_supported(make_model(ln1_bias=True))


def test_a_nonzero_attention_bias_is_refused():
    with pytest.raises(qk.UnsupportedForQK, match="b_Q"):
        qk.require_supported(make_model(attn_bias=True))


def test_a_zero_attention_bias_is_accepted():
    qk.require_supported(make_model())


def test_attention_scale_defaults_to_the_square_root_of_d_head():
    assert qk.attention_scale(make_model()) == pytest.approx(D_HEAD**0.5)


def test_attention_scale_is_one_when_the_model_disables_it():
    assert qk.attention_scale(make_model(use_attn_scale=False)) == 1.0


def test_frozen_scores_restore_the_head_axis_of_qk_norm_scales():
    run = qk.FrozenScores.from_model(make_model(), torch.arange(N_POS))
    assert run.n_pos == N_POS
    assert run.resid_pre[0].shape == (N_POS, D_MODEL)
    assert run.ln1_scales[0].shape == (N_POS, 1)
    assert run.query_scales[0].shape == (N_POS, N_HEADS, 1)
    assert run.key_scales[0].shape == (N_POS, 2, 1)


def test_frozen_scores_without_qk_norm_carry_no_scales():
    run = qk.FrozenScores.from_model(make_model(qk_norm=False), torch.arange(N_POS))
    assert run.query_scales == [None] * N_LAYERS
    assert run.key_scales == [None] * N_LAYERS


def test_a_batched_cache_is_refused():
    model = make_model()
    batched = {name: torch.cat([value, value]) for name, value in reference_cache(model).items()}
    model.run_with_cache = lambda tokens, names_filter=None: (None, batched)
    with pytest.raises(ValueError, match="a batch of 2"):
        qk.FrozenScores.from_model(model, torch.arange(N_POS))


def test_rotations_are_orthogonal():
    rotations = qk.rotation_matrices(make_model(), N_POS)
    assert rotations is not None and rotations.shape == (N_POS, D_HEAD, D_HEAD)
    identity = torch.eye(D_HEAD).expand(N_POS, D_HEAD, D_HEAD)
    torch.testing.assert_close(rotations @ rotations.transpose(1, 2), identity, atol=1e-5, rtol=0)


def test_rotations_depend_only_on_the_offset():
    rotations = qk.rotation_matrices(make_model(), N_POS)
    assert rotations is not None
    torch.testing.assert_close(
        rotations[3] @ rotations[1].T, rotations[4] @ rotations[2].T, atol=1e-5, rtol=0
    )


def test_no_rotations_without_rotary_embeddings():
    assert qk.rotation_matrices(make_model(rotary=False), N_POS) is None


ARCHITECTURES = [
    {"rotary": False, "qk_norm": False, "kv_heads": 4},
    {"rotary": True, "qk_norm": False, "kv_heads": 4},
    {"rotary": False, "qk_norm": True, "kv_heads": 2},
    {"rotary": True, "qk_norm": True, "kv_heads": 2},
]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_scores_match_an_independent_forward_pass(architecture: dict):
    model = make_model(**architecture)
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            torch.testing.assert_close(
                qk.attention_scores(model, run, layer, head),
                reference_scores(model, layer, head),
                rtol=1e-4,
                atol=1e-5,
            )


def test_to_head_space_rejects_an_unknown_side():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    with pytest.raises(ValueError, match="side"):
        qk.to_head_space(model, run, 0, 0, torch.zeros(1, D_MODEL), torch.zeros(1).long(), side="x")


def test_to_head_space_rejects_mismatched_positions():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    with pytest.raises(ValueError, match="positions"):
        qk.to_head_space(
            model, run, 0, 0, torch.zeros(2, D_MODEL), torch.zeros(3).long(), side="query"
        )


def test_activations_follow_active_features_on_a_pruned_graph():
    """activation_values is aligned with active_features; the two coincide only when nothing is
    pruned, which is how indexing it by selection stays invisible on small graphs."""
    model = make_model()
    graph = make_graph([(0, 1, 2), (0, 3, 4), (1, 2, 5)], [0, 2], [3.0, 99.0, 7.0])
    sources = qk.feature_sources(model, graph, below_layer=2)
    torch.testing.assert_close(sources.directions[0], model.transcoders[0].W_dec[2] * 3.0)
    torch.testing.assert_close(sources.directions[1], model.transcoders[1].W_dec[5] * 7.0)


def test_activation_values_matching_neither_table_are_rejected():
    graph = make_graph([(0, 1, 2), (0, 3, 4), (1, 2, 5)], [0, 2], [3.0, 7.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="matching neither"):
        qk.feature_sources(make_model(), graph, below_layer=2)


def test_only_features_written_below_the_layer_are_sources():
    graph = make_graph([(0, 1, 2), (1, 3, 4), (2, 2, 5)], [0, 1, 2], [1.0, 1.0, 1.0])
    sources = qk.feature_sources(make_model(), graph, below_layer=2)
    assert sources.layers.tolist() == [0, 1]
    assert sources.directions.dtype == torch.float32


def test_sources_and_the_remainder_reconstruct_the_residual():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    graph = make_graph([(0, 1, 2), (0, 1, 3), (1, 4, 5)], [0, 1, 2], [2.0, -1.5, 0.5])
    features = qk.feature_sources(model, graph, below_layer=2)
    sources = features.concat(qk.remainder_sources(run, features, 2))
    rebuilt = torch.zeros(N_POS, D_MODEL).index_add_(0, sources.positions, sources.directions)
    torch.testing.assert_close(rebuilt, run.resid_pre[2], rtol=1e-5, atol=1e-5)
    assert int(sources.is_remainder.sum()) == N_POS


GRAPH_FEATURES = ([(0, 1, 2), (0, 5, 3), (1, 4, 5), (1, 5, 1), (2, 3, 7)], [0, 1, 2, 3, 4])
ACTIVATIONS = [2.0, -1.0, 0.5, 1.5, 3.0]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_contributions_sum_to_the_score_row(architecture: dict):
    """The whole claim: with the remainder carried, the pairs reproduce the score exactly."""
    model = make_model(**architecture)
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    graph = make_graph(*GRAPH_FEATURES, ACTIVATIONS)
    for layer in (1, 2):
        for query_position in (2, N_POS - 1):
            result = qk.qk_attribution(model, graph, run, layer, 1, query_position)
            expected = reference_scores(model, layer, 1)[query_position, : query_position + 1]
            got = result.by_key_position(N_POS)[: query_position + 1]
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_query_sources_all_sit_at_the_query_position():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    result = qk.qk_attribution(model, make_graph(*GRAPH_FEATURES, ACTIVATIONS), run, 2, 0, 5)
    assert set(result.query_sources.positions.tolist()) == {5}
    assert int(result.query_sources.is_remainder.sum()) == 1


def test_key_sources_never_come_from_later_positions():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    result = qk.qk_attribution(model, make_graph(*GRAPH_FEATURES, ACTIVATIONS), run, 2, 0, 3)
    assert int(result.key_sources.positions.max()) <= 3
    assert result.contributions.shape == (len(result.query_sources), len(result.key_sources))


def test_top_pairs_are_the_largest_by_magnitude():
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    result = qk.qk_attribution(model, make_graph(*GRAPH_FEATURES, ACTIVATIONS), run, 2, 3, 5)
    values, flat = result.top_pairs(3)
    torch.testing.assert_close(values, result.contributions.flatten()[flat])
    third_largest = result.contributions.abs().flatten().sort().values[-3]
    assert float(values.abs().min()) >= float(third_largest)


@pytest.mark.parametrize(
    ("layer", "head", "query_position", "fragment"),
    [(N_LAYERS, 0, 0, "layer"), (0, N_HEADS, 0, "head"), (0, 0, N_POS, "query_position")],
)
def test_out_of_range_arguments_raise(layer: int, head: int, query_position: int, fragment: str):
    model = make_model()
    run = qk.FrozenScores.from_model(model, torch.arange(N_POS))
    graph = make_graph(*GRAPH_FEATURES, ACTIVATIONS)
    with pytest.raises(IndexError, match=fragment):
        qk.qk_attribution(model, graph, run, layer, head, query_position)
