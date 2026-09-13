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
