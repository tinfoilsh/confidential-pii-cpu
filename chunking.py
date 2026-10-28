"""Overlapping-window planning for long inputs.

The privacy filter is a banded-attention token classifier: each layer lets a
token attend to its ±128 neighbours (see the model card), and the Viterbi
decode that turns per-token scores into spans runs over the whole sequence.
Upstream `opf` cuts long inputs into non-overlapping `n_ctx` windows, so
tokens near each cut lose half their context. This module plans windows of
`window` tokens that overlap by `overlap`, keeps only the tokens of each
window that have `overlap` tokens of context on both sides (except at the
sequence start and end, where the context genuinely ends), and stitches the
kept rows back into one per-token sequence.

Every token is kept from exactly one window, so upstream's cross-window
aggregation (logaddexp of duplicates, divided by count) reduces to the
identity and the stitched scores feed the unchanged global decode.

This module is intentionally free of torch so it can be unit-tested without
the model. "Rows" are any indexable sequence (lists, tensors, arrays).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

Row = TypeVar("Row")


@dataclass(frozen=True)
class WindowPlan:
    """One model forward over tokens[start:end], keeping outputs for
    tokens[keep_start:keep_end] (absolute token indices)."""

    start: int
    end: int
    keep_start: int
    keep_end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def kept_rows(self, rows: Sequence[Row]) -> Sequence[Row]:
        """Slice this window's forward output down to the kept tokens."""
        if len(rows) != self.length:
            raise ValueError(
                f"window {self.start}:{self.end} produced {len(rows)} rows, expected {self.length}"
            )
        return rows[self.keep_start - self.start : self.keep_end - self.start]


def plan_windows(n_tokens: int, window: int, overlap: int) -> list[WindowPlan]:
    """Plan overlapping windows covering `n_tokens` tokens.

    Consecutive windows overlap by `2 * overlap` tokens so that the kept
    regions `[start + overlap, end - overlap)` tile the sequence exactly;
    the first window keeps from token 0 and the last keeps to the end.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if 2 * overlap >= window:
        raise ValueError(f"window ({window}) must exceed twice the overlap ({overlap})")
    if n_tokens < 0:
        raise ValueError("n_tokens must be non-negative")
    plans: list[WindowPlan] = []
    start = 0
    while n_tokens > 0:
        end = min(start + window, n_tokens)
        keep_start = 0 if start == 0 else start + overlap
        if end == n_tokens:
            plans.append(WindowPlan(start, end, keep_start, n_tokens))
            break
        plans.append(WindowPlan(start, end, keep_start, end - overlap))
        start += window - 2 * overlap
    return plans


def kept_outputs(
    plans: Sequence[WindowPlan], outputs: Sequence[Sequence[Row]], n_tokens: int
) -> list[Sequence[Row]]:
    """Reduce per-window forward outputs to their kept rows, in token order.

    `outputs[i]` must hold one row per token of `plans[i]` (any sliceable:
    a list, a 2-D memoryview over float32 bytes, a tensor). Raises if the
    kept regions do not cover every token exactly once. Concatenating the
    returned slices gives one row per token.
    """
    if len(plans) != len(outputs):
        raise ValueError(f"{len(plans)} windows but {len(outputs)} outputs")
    kept: list[Sequence[Row]] = []
    cursor = 0
    for plan, out in zip(plans, outputs):
        if plan.keep_start != cursor:
            raise ValueError(
                f"window {plan.start}:{plan.end} keeps from {plan.keep_start}, expected {cursor}"
            )
        kept.append(plan.kept_rows(out))
        cursor = plan.keep_end
    if cursor != n_tokens:
        raise ValueError(f"assembled {cursor} rows for {n_tokens} tokens")
    return kept


def assemble(plans: Sequence[WindowPlan], outputs: Sequence[Sequence[Row]], n_tokens: int) -> list[Row]:
    """Stitch per-window forward outputs into one flat list of rows."""
    rows: list[Row] = []
    for part in kept_outputs(plans, outputs, n_tokens):
        rows.extend(part)
    return rows
