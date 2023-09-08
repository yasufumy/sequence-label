from __future__ import annotations

from typing import TYPE_CHECKING

from sequence_label import LabelAlignment
from sequence_label.core import Span

if TYPE_CHECKING:
    from transformers import BatchEncoding


def get_alignments(
    batch_encoding: BatchEncoding,
    lengths: list[int],
    is_split_into_words: bool = False,
) -> tuple[LabelAlignment, ...]:
    if not batch_encoding.is_fast:
        raise ValueError("Please use PreTrainedTokenizerFast.")

    alignments = []
    if not is_split_into_words:
        for i, length in enumerate(lengths):
            num_tokens = len(batch_encoding.tokens(i))
            src_char_spans: list[Span | None] = [None] * num_tokens
            for j in range(num_tokens):
                span = batch_encoding.token_to_chars(i, j)
                if span is None:
                    continue
                src_char_spans[j] = Span(span.start, span.end - span.start)

            tgt_token_spans: list[Span | None] = [None] * length
            for j in range(length):
                token_index = batch_encoding.char_to_token(i, j)
                if token_index is None:
                    continue
                tgt_token_spans[j] = Span(start=token_index, length=1)

            alignments.append(
                LabelAlignment(
                    source_spans=tuple(src_char_spans),
                    target_spans=tuple(tgt_token_spans),
                )
            )
    else:
        for i, length in enumerate(lengths):
            num_tokens = len(batch_encoding.tokens(i))
            src_token_spans: list[Span | None] = [None] * num_tokens
            for j in range(num_tokens):
                word_index = batch_encoding.token_to_word(i, j)
                if word_index is None:
                    continue
                src_token_spans[j] = Span(start=word_index, length=1)

            tgt_word_spans: list[Span | None] = [None] * length
            for j in range(length):
                span = batch_encoding.word_to_tokens(i, j)
                if span is None:
                    continue
                tgt_word_spans[j] = Span(start=span.start, length=span.end - span.start)

            alignments.append(
                LabelAlignment(
                    source_spans=tuple(src_token_spans),
                    target_spans=tuple(tgt_word_spans),
                )
            )
    return tuple(alignments)
