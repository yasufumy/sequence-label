from __future__ import annotations

from typing import TYPE_CHECKING

from sequence_label import LabelAlignment
from sequence_label.core import Span

if TYPE_CHECKING:
    from transformers import BatchEncoding


def get_alignments(
    batch_encoding: BatchEncoding, char_lengths: list[int]
) -> tuple[LabelAlignment, ...]:
    if not batch_encoding.is_fast:
        raise ValueError()

    alignments = []
    for i, length in enumerate(char_lengths):
        num_tokens = len(batch_encoding.tokens(i))
        char_spans: list[Span | None] = [None] * num_tokens
        for j in range(num_tokens):
            span = batch_encoding.token_to_chars(i, j)
            if span is None:
                continue
            char_spans[j] = Span(span.start, span.end - span.start)

        token_indices = [-1] * length
        for j in range(length):
            token_index = batch_encoding.char_to_token(i, j)
            if token_index is None:
                continue
            token_indices[j] = token_index

        alignments.append(
            LabelAlignment(
                char_spans=tuple(char_spans), token_indices=tuple(token_indices)
            )
        )
    return tuple(alignments)
