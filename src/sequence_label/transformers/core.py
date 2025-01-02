from collections.abc import Sequence

from transformers import BatchEncoding

from sequence_label import LabelAlignment
from sequence_label.core import Span

__all__ = ["create_alignments"]


def create_alignments(
    batch_encoding: BatchEncoding,
    lengths: Sequence[int],
    is_split_into_words: bool = False,
    padding_token: str | None = None,
) -> Sequence[LabelAlignment]:
    """Creates instances of LabelAlignment from an instance of BatchEncoding that
    is a result of the Huggingface tokenizer.

    Args:
        batch_encoding: An instance of BatchEncoding.
        lengths: A sequence of integers where each item represents a length of text.
        is_split_into_words: A boolean representing whether is_split_into_words was
            enabled during tokenization. Defaults to False.
        padding_token: A string representing a special token used to make lists of
            tokens the same size for batching purpose. Defaults to None.

    Returns:
        a sequence of instances of LabelAlignment.
    """
    if not batch_encoding.is_fast:
        raise ValueError("Please use a subclass of PreTrainedTokenizerFast.")

    alignments = []
    if not is_split_into_words:
        for i, length in enumerate(lengths):
            num_tokens = sum(
                1
                for token in batch_encoding.tokens(i)
                if padding_token is None or token != padding_token
            )
            src_char_spans: list[Span | None] = [None] * num_tokens
            for j in range(num_tokens):
                span = batch_encoding.token_to_chars(i, j)
                if span is None or span.start == span.end:
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
                    source_spans=src_char_spans,
                    target_spans=tgt_token_spans,
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
                if span is None or span.start == span.end:
                    continue
                tgt_word_spans[j] = Span(start=span.start, length=span.end - span.start)

            alignments.append(
                LabelAlignment(
                    source_spans=src_token_spans,
                    target_spans=tgt_word_spans,
                )
            )
    return alignments
