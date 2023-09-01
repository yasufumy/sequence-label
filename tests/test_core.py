from __future__ import annotations

import pytest

from sequence_label import LabelAlignment, LabelSet, SequenceLabel
from sequence_label.core import Base, Span


@pytest.fixture()
def text() -> str:
    return "Tokyo is the capital of Japan."


@pytest.fixture()
def size(text: str) -> int:
    return len(text)


@pytest.fixture()
def label(size: int) -> SequenceLabel:
    return SequenceLabel.from_dict(
        tags=[
            {"start": 0, "end": 5, "label": "LOC"},
            {"start": 24, "end": 29, "label": "LOC"},
        ],
        size=size,
    )


@pytest.fixture()
def alignment() -> LabelAlignment:
    # Tokenized by RoBERTa
    return LabelAlignment(
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
            1,
            1,
            1,
            2,
            2,
            -1,
            3,
            3,
            -1,
            4,
            4,
            4,
            -1,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            -1,
            6,
            6,
            -1,
            7,
            7,
            7,
            7,
            7,
            8,
        ),
    )


@pytest.fixture()
def truncated_alignment() -> LabelAlignment:
    # Tokenized by RoBERTa
    return LabelAlignment(
        (
            None,
            Span(0, 3),
            Span(3, 2),
            None,
        ),
        (
            1,
            1,
            1,
            2,
            2,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ),
    )


@pytest.fixture()
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


def test_ignore_tags_define_in_truncated_text(
    truncated_alignment: LabelAlignment, label: SequenceLabel
) -> None:
    expected = SequenceLabel.from_dict(
        tags=[{"start": 1, "end": 3, "label": "LOC"}], size=4, base=Base.TARGET
    )

    assert truncated_alignment.align_with_target(label=label) == expected


def test_start_states_are_valid(label_set: LabelSet) -> None:
    expected = (
        True,  # O
        True,  # B-ORG
        False,  # I-ORG
        False,  # L-ORG
        True,  # U-ORG
        True,  # B-PER
        False,  # I-PER
        False,  # L-PER
        True,  # U-PER
    )

    assert label_set.start_states == expected


def test_end_states_are_valid(label_set: LabelSet) -> None:
    expected = (
        True,  # O
        False,  # B-ORG
        False,  # I-ORG
        True,  # L-ORG
        True,  # U-ORG
        False,  # B-PER
        False,  # I-PER
        True,  # L-PER
        True,  # U-PER
    )

    assert label_set.end_states == expected


def test_transitions_are_valid(label_set: LabelSet) -> None:
    expected = (
        (
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ),  # O
        (
            False,  # O
            False,  # B-ORG
            True,  # I-ORG
            True,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            False,  # I-PER
            False,  # L-PER
            False,  # U-PER
        ),  # B-ORG
        (
            False,  # O
            False,  # B-ORG
            True,  # I-ORG
            True,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            False,  # I-PER
            False,  # L-PER
            False,  # U-PER
        ),  # I-ORG
        (
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ),  # L-ORG
        (
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ),  # U-ORG
        (
            False,  # O
            False,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            True,  # I-PER
            True,  # L-PER
            False,  # U-PER
        ),  # B-PER
        (
            False,  # O
            False,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            False,  # U-ORG
            False,  # B-PER
            True,  # I-PER
            True,  # L-PER
            False,  # U-PER
        ),  # I-PER
        (
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ),  # L-PER
        (
            True,  # O
            True,  # B-ORG
            False,  # I-ORG
            False,  # L-ORG
            True,  # U-ORG
            True,  # B-PER
            False,  # I-PER
            False,  # L-PER
            True,  # U-PER
        ),  # U-PER
    )

    assert label_set.transitions == expected
