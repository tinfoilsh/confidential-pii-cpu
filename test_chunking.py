"""Window planning tests. Model-free: the forward pass is a deterministic fake
whose output for a token depends on exactly its ±R-token neighbourhood, so a
window that truncates a kept token's context produces a different row."""

import struct

import pytest

from chunking import assemble, check_geometry, kept_outputs, plan_windows, required_overlap

# Production defaults: 8 layers × 128 context = 1024-token receptive field.
RECEPTIVE_FIELD = 8 * 128
W_DEFAULT, OVERLAP_DEFAULT = 6144, 1024
W = 1024


def fake_forward(tokens, reach):
    """Each row summarises the token's ±`reach` neighbourhood (clipped to the
    window): how far the band extends left, its length, its sum and a
    position-weighted sum, so any clipping or reordering changes the row.
    Relative positions only, like RoPE: shifting the window does not change
    a row unless its neighbourhood is clipped. O(n) via prefix sums."""
    n = len(tokens)
    prefix = [0] * (n + 1)
    weighted = [0] * (n + 1)
    for i, tok in enumerate(tokens):
        prefix[i + 1] = prefix[i] + tok
        weighted[i + 1] = weighted[i] + (i + 1) * tok
    rows = []
    for i in range(n):
        lo, hi = max(0, i - reach), min(n, i + reach + 1)
        band_sum = prefix[hi] - prefix[lo]
        # position-weighted sum relative to the token, independent of window offset
        rel = (weighted[hi] - weighted[lo]) - (i + 1) * band_sum
        rows.append([float(i - lo), float(hi - lo), float(band_sum), float(rel)])
    return rows


def tokens_of(n):
    return [(i * 7919 + 13) % 200_000 for i in range(n)]


LENGTHS = [lambda W: 0, lambda W: 1, lambda W: W - 1, lambda W: W, lambda W: W + 1, lambda W: 10 * W + 7]


@pytest.mark.parametrize("W", [1024, 2048])
@pytest.mark.parametrize("overlap", [128, 256])
@pytest.mark.parametrize("length", LENGTHS)
def test_windows_cover_every_token_exactly_once_with_full_context(length, W, overlap):
    n = length(W)
    plans = plan_windows(n, W, overlap)
    if n == 0:
        assert plans == []
        return
    assert plans[0].keep_start == 0 and plans[-1].keep_end == n
    cursor = 0
    for p in plans:
        assert 0 <= p.start <= p.keep_start < p.keep_end <= p.end <= n
        assert p.length <= W
        assert p.keep_start == cursor  # contiguous, no gaps or duplicates
        cursor = p.keep_end
        # Every kept token has `overlap` tokens of context on each side inside
        # the window, unless the sequence itself ends there.
        assert p.keep_start - p.start >= min(overlap, p.keep_start)
        assert p.end - p.keep_end >= min(overlap, n - p.keep_end)
    assert cursor == n
    if n <= W:
        # Fits in one window: nothing is masked.
        assert plans == [plans[0]] and (plans[0].keep_start, plans[0].keep_end) == (0, n)
    for a, b in zip(plans, plans[1:]):
        assert b.start < a.end  # overlapping
        assert a.end - b.start >= 2 * overlap
        assert b.start - a.start == W - 2 * overlap  # effective stride


@pytest.mark.parametrize("W", [1024, 2048])
@pytest.mark.parametrize("overlap", [128, 256])
@pytest.mark.parametrize("length", LENGTHS)
def test_chunked_equals_single_window_for_model_reaching_overlap(length, W, overlap):
    n = length(W)
    tokens = tokens_of(n)
    plans = plan_windows(n, W, overlap)
    outputs = [fake_forward(tokens[p.start : p.end], overlap) for p in plans]
    assert assemble(plans, outputs, n) == fake_forward(tokens, overlap)


@pytest.mark.parametrize("n", [1, W_DEFAULT, W_DEFAULT + 1, 3 * W_DEFAULT, 10 * W_DEFAULT + 7])
def test_defaults_are_exact_for_the_models_receptive_field(n):
    """At the production geometry a model with the checkpoint's receptive
    field (8 layers × 128) scores every token identically to one long pass."""
    tokens = tokens_of(n)
    plans = plan_windows(n, W_DEFAULT, OVERLAP_DEFAULT)
    outputs = [fake_forward(tokens[p.start : p.end], RECEPTIVE_FIELD) for p in plans]
    assert assemble(plans, outputs, n) == fake_forward(tokens, RECEPTIVE_FIELD)
    assert all(p.length <= W_DEFAULT for p in plans)
    if n > W_DEFAULT:
        assert plans[1].start - plans[0].start == W_DEFAULT - 2 * OVERLAP_DEFAULT  # stride 4096


@pytest.mark.parametrize("W,overlap", [(W_DEFAULT, OVERLAP_DEFAULT), (W, 128)])
def test_differential_test_is_sensitive_to_truncated_context(W, overlap):
    """A model reaching one token further than the overlap must see the seams:
    proves the equality above is not vacuous."""
    n = 10 * W + 7
    tokens = tokens_of(n)
    plans = plan_windows(n, W, overlap)
    outputs = [fake_forward(tokens[p.start : p.end], overlap + 1) for p in plans]
    assert assemble(plans, outputs, n) != fake_forward(tokens, overlap + 1)


def test_required_overlap_from_checkpoint_config():
    config = {
        "num_hidden_layers": 8,
        "bidirectional_left_context": 128,
        "bidirectional_right_context": 128,
    }
    assert required_overlap(config) == RECEPTIVE_FIELD
    assert required_overlap({**config, "bidirectional_right_context": 64}) == 8 * 128
    assert required_overlap({**config, "num_hidden_layers": 36}) == 36 * 128
    with pytest.raises(KeyError):
        required_overlap({"num_hidden_layers": 8})


def test_geometry_floor_refuses_inexact_overlap():
    check_geometry(W_DEFAULT, OVERLAP_DEFAULT, RECEPTIVE_FIELD)
    check_geometry(W_DEFAULT, 2048, RECEPTIVE_FIELD)  # env may raise it
    with pytest.raises(ValueError, match="receptive field"):
        check_geometry(W_DEFAULT, 1023, RECEPTIVE_FIELD)
    with pytest.raises(ValueError, match="twice the overlap"):
        check_geometry(2048, OVERLAP_DEFAULT, RECEPTIVE_FIELD)


def test_kept_outputs_slices_binary_rows_exactly():
    """The server stitches float32 bytes without decoding: a 2-D memoryview
    over each window's bytes slices by row and the kept slices concatenate
    to the single-pass bytes."""
    n, overlap = 3 * W + 5, 128
    tokens = tokens_of(n)
    plans = plan_windows(n, W, overlap)
    classes = len(fake_forward([1], overlap)[0])

    def forward_bytes(window):
        rows = fake_forward(window, overlap)
        return struct.pack(f"<{len(rows) * classes}d", *(v for r in rows for v in r))

    outputs = [
        memoryview(forward_bytes(tokens[p.start : p.end])).cast("d", shape=(p.length, classes))
        for p in plans
    ]
    stitched = b"".join(part.tobytes() for part in kept_outputs(plans, outputs, n))
    assert stitched == forward_bytes(tokens)


def test_zero_overlap_matches_upstream_fixed_windows():
    plans = plan_windows(10 * W + 7, W, 0)
    assert [(p.start, p.end) for p in plans] == [
        (s, min(s + W, 10 * W + 7)) for s in range(0, 10 * W + 7, W)
    ]
    assert all(p.keep_start == p.start and p.keep_end == p.end for p in plans)


def test_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        plan_windows(10, 0, 0)
    with pytest.raises(ValueError):
        plan_windows(10, 256, 128)
    with pytest.raises(ValueError):
        plan_windows(10, 256, -1)


def test_assemble_rejects_wrong_shapes():
    plans = plan_windows(W + 1, W, 128)
    good = [[[0.0]] * p.length for p in plans]
    assemble(plans, good, W + 1)
    with pytest.raises(ValueError):
        assemble(plans, [good[0]], W + 1)
    with pytest.raises(ValueError):
        assemble(plans, [good[0][:-1], good[1]], W + 1)
    with pytest.raises(ValueError):
        assemble(plans, good, W)
