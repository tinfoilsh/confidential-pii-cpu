"""Window planning tests. Model-free: the forward pass is a deterministic fake
whose output for a token depends on exactly its ±R-token neighbourhood, so a
window that truncates a kept token's context produces a different row."""

import hashlib
import struct

import pytest

from chunking import assemble, kept_outputs, plan_windows

W = 1024


def fake_forward(tokens, reach):
    """Each row is a hash of the token's ±`reach` neighbourhood (clipped to the
    window), mimicking a banded model whose attention reaches `reach` tokens
    each way. Relative positions only, like RoPE: shifting the window does
    not change a row unless its neighbourhood is clipped."""
    rows = []
    for i in range(len(tokens)):
        band = tokens[max(0, i - reach) : i + reach + 1]
        digest = hashlib.blake2b(
            f"{i - max(0, i - reach)}|{band}".encode(), digest_size=8
        ).digest()
        rows.append([int.from_bytes(digest, "big") / 2**64, float(len(band))])
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


def test_differential_test_is_sensitive_to_truncated_context():
    """A model reaching one token further than the overlap must see the seams:
    proves the equality above is not vacuous."""
    n, overlap = 10 * W + 7, 128
    tokens = tokens_of(n)
    plans = plan_windows(n, W, overlap)
    outputs = [fake_forward(tokens[p.start : p.end], overlap + 1) for p in plans]
    assert assemble(plans, outputs, n) != fake_forward(tokens, overlap + 1)


def test_kept_outputs_slices_binary_rows_exactly():
    """The server stitches float32 bytes without decoding: a 2-D memoryview
    over each window's bytes slices by row and the kept slices concatenate
    to the single-pass bytes."""
    n, overlap, classes = 3 * W + 5, 128, 3
    tokens = tokens_of(n)
    plans = plan_windows(n, W, overlap)

    def forward_bytes(window):
        rows = fake_forward(window, overlap)
        return struct.pack(f"<{len(rows) * classes}f", *(v for r in rows for v in (r + [0.0])))

    outputs = [
        memoryview(forward_bytes(tokens[p.start : p.end])).cast("f", shape=(p.length, classes))
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
