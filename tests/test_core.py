from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sequence_label import LabelAlignment, LabelSet, SequenceLabel
from sequence_label.core import Base, Span, TagDict

if TYPE_CHECKING:
    from collections.abc import Sequence


@st.composite
def source_labels(draw: st.DrawFn) -> Sequence[SequenceLabel]:
    size = 100
    num_labels = 8
    max_num_tags = 20

    labels = []
    for _ in range(num_labels):
        num_tags = draw(st.integers(min_value=1, max_value=max_num_tags))
        tags = []
        last = 0
        for _ in range(num_tags):
            if last >= size:
                break
            start = draw(st.integers(min_value=last, max_value=size - 1))
            end = draw(st.integers(min_value=start + 1, max_value=size))
            label = draw(st.sampled_from(["ORG", "LOC", "PER", "MISC"]))
            # NOTE: mypy cannot infer a type of the dictionary below.
            tags.append(cast(TagDict, {"start": start, "end": end, "label": label}))
            last = end + 1
        labels.append(SequenceLabel.from_dict(tags=tags, size=size))

    return labels


@given(labels=source_labels())
def test_labels_unchanged_after_encoding_and_decoding(
    labels: Sequence[SequenceLabel],
) -> None:
    label_set = LabelSet({"ORG", "LOC", "PER", "MISC"})
    assert labels == label_set.decode(label_set.encode_to_tag_indices(labels))


@st.composite
def target_label(draw: st.DrawFn) -> SequenceLabel:
    size = 9
    num_tags = draw(st.integers(min_value=1, max_value=5))
    tags = []
    last = 1
    for _ in range(num_tags):
        if last >= size:
            break
        start = draw(st.integers(min_value=last, max_value=size - 1))
        end = draw(st.integers(min_value=start + 1, max_value=size))
        label = draw(st.sampled_from(["ORG", "LOC", "PER", "MISC"]))
        # NOTE: mypy cannot infer a type of the dictionary below.
        tags.append(cast(TagDict, {"start": start, "end": end, "label": label}))
        last = end + 1

    return SequenceLabel.from_dict(tags=tags, size=size + 1, base=Base.Target)


@given(label=target_label())
def test_labels_unchanged_after_alignment(label: SequenceLabel) -> None:
    # Tokenized by RoBERTa
    alignment = LabelAlignment(
        (
            None,
            Span(0, 3),
            Span(3, 2),
            Span(6, 2),
            Span(9, 3),
            Span(13, 7),
            Span(21, 2),
            Span(24, 5),
            Span(29, 1),
            None,
        ),
        (
            Span(1, 1),
            Span(1, 1),
            Span(1, 1),
            Span(2, 1),
            Span(2, 1),
            None,
            Span(3, 1),
            Span(3, 1),
            None,
            Span(4, 1),
            Span(4, 1),
            Span(4, 1),
            None,
            Span(5, 1),
            Span(5, 1),
            Span(5, 1),
            Span(5, 1),
            Span(5, 1),
            Span(5, 1),
            Span(5, 1),
            None,
            Span(6, 1),
            Span(6, 1),
            None,
            Span(7, 1),
            Span(7, 1),
            Span(7, 1),
            Span(7, 1),
            Span(7, 1),
            Span(8, 1),
        ),
    )

    assert label == alignment.align_with_target(
        label=alignment.align_with_source(label=label)
    )


def test_tags_define_in_truncated_part_ignored() -> None:
    truncated_alignment = LabelAlignment(
        (
            None,
            Span(0, 3),
            Span(3, 2),
            None,
        ),
        (
            Span(1, 1),
            Span(1, 1),
            Span(1, 1),
            Span(2, 1),
            Span(2, 1),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )

    label = SequenceLabel.from_dict(
        tags=[
            {"start": 0, "end": 5, "label": "LOC"},
            {"start": 24, "end": 29, "label": "LOC"},
        ],
        size=30,
    )
    expected = SequenceLabel.from_dict(
        tags=[{"start": 1, "end": 3, "label": "LOC"}], size=4, base=Base.Target
    )

    assert truncated_alignment.align_with_target(label=label) == expected


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet({"ORG", "PER"})


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("ORG", True),
        ("PER", True),
        ("PERSON", False),
        ("LOCATION", False),
        ("ORGANIZE", False),
    ],
)
def test_membership_check_is_valid(
    label_set: LabelSet, label: str, expected: bool
) -> None:
    is_member = label in label_set

    assert is_member == expected


def test_start_states_are_valid(label_set: LabelSet) -> None:
    expected = [
        True,  # O
        True,  # B-ORG
        False,  # I-ORG
        False,  # L-ORG
        True,  # U-ORG
        True,  # B-PER
        False,  # I-PER
        False,  # L-PER
        True,  # U-PER
    ]

    assert label_set.start_states == expected


def test_end_states_are_valid(label_set: LabelSet) -> None:
    expected = [
        True,  # O
        False,  # B-ORG
        False,  # I-ORG
        True,  # L-ORG
        True,  # U-ORG
        False,  # B-PER
        False,  # I-PER
        True,  # L-PER
        True,  # U-PER
    ]

    assert label_set.end_states == expected


def test_transitions_are_valid(label_set: LabelSet) -> None:
    expected = [
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # O
        [
            False,  # O
            False,  # B-ORG
            True,  # I-ORG
            True,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            False,  # I-PER
            False,  # L-PER
            False,  # U-PER
        ],  # B-ORG
        [
            False,  # O
            False,  # B-ORG
            True,  # I-ORG
            True,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            False,  # I-PER
            False,  # L-PER
            False,  # U-PER
        ],  # I-ORG
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # L-ORG
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # U-ORG
        [
            False,  # O
            False,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            True,  # I-PER
            True,  # L-PER
            False,  # U-PER
        ],  # B-PER
        [
            False,  # O
            False,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            True,  # I-PER
            True,  # L-PER
            False,  # U-PER
        ],  # I-PER
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # L-PER
        [
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ],  # U-PER
    ]

    assert label_set.transitions == expected
