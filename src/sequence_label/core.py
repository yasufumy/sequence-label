from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
from typing import TypedDict


@dataclass(frozen=True, order=True)
class Span:
    start: int
    length: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be zero or positive: {self.start}")

        if self.length <= 0:
            raise ValueError(f"length must be positive: {self.length}")


@dataclass(frozen=True, order=True)
class Tag:
    span: Span
    label: str

    @property
    def start(self) -> int:
        return self.span.start

    @property
    def length(self) -> int:
        return self.span.length

    @classmethod
    def create(cls, start: int, end: int, label: str) -> Tag:
        """Creates an instance of Tag.

        Args:
            start: An integer representing a position in text where a tag starts.
            end: An integer representing a position in text where a tag ends.
                Note that an end is expected to be exclusive.
            label: A string representing what you want to assign to a span in a text.

        Returns:
            An instance of Tag.
        """
        if start >= end:
            raise ValueError(f"end must be greater than start: {start} >= {end}")

        length = end - start
        return cls(Span(start, length), label)


class TagDict(TypedDict):
    start: int
    end: int
    label: str


class Base(Enum):
    CHARACTER = auto()
    TOKEN = auto()


@dataclass(frozen=True)
class SequenceLabel:
    tags: tuple[Tag, ...]
    size: int
    base: Base = Base.CHARACTER

    def __post_init__(self) -> None:
        if any(self.tags[i] > self.tags[i + 1] for i in range(len(self.tags) - 1)):
            raise ValueError("Tags must be sorted by Tag.start and Tag.length.")

        for tag in self.tags:
            if tag.start < 0 or tag.start >= self.size:
                raise ValueError(
                    "An invalid tag is found. start must be"
                    f" in between 0 and {self.size - 1}: {tag.start}"
                )
            end = tag.start + tag.length
            if end < 0 or end > self.size:
                raise ValueError(
                    "An invalid tag is found. length must be"
                    f" in between 1 and {self.size}: {tag.length}"
                )

    @classmethod
    def from_dict(
        cls, tags: list[TagDict], size: int, base: Base = Base.CHARACTER
    ) -> SequenceLabel:
        return cls(
            tags=tuple(
                sorted(
                    Tag.create(tag["start"], tag["end"], tag["label"]) for tag in tags
                )
            ),
            size=size,
            base=base,
        )


class LabelAlignment:
    """An alignment class responsible for manipulating character-based tags based on
    a tokenization result, which is useful for encoding tags to tag_indices/tag_bitmap
    and decoding tag_indices/tag_bitmap to tags.

    Args:
        char_spans: A tuple of character spans for each token, or None if
            there is no corresponding span.
        token_indices: A tuple of token indices for each character, or -1 if there is
            no corresponding token.

    Attributes:
        char_length: The text length before tokenization.
        num_tokens: The number of tokens after tokenization.
    """

    def __init__(
        self, char_spans: tuple[Span | None, ...], token_indices: tuple[int, ...]
    ):
        num_tokens = len(char_spans)
        if not all(index == -1 or 0 <= index < num_tokens for index in token_indices):
            raise ValueError(
                "Each item in token_indices must be -1 or"
                f" in between 0 and {num_tokens - 1}: {token_indices}"
            )

        char_length = len(token_indices)
        if not all(
            0 <= span.start < char_length  # check if start is valid
            and 0 <= span.start + span.length - 1 < char_length  # check if end is valid
            for span in char_spans
            if span is not None
        ):
            raise ValueError(
                "Each span in char_spans must be None or"
                f" in between 0 and {char_length - 1}: {char_spans}"
            )

        self.__char_spans = char_spans
        self.__token_indices = token_indices

    @property
    def char_length(self) -> int:
        return len(self.__token_indices)

    @property
    def num_tokens(self) -> int:
        return len(self.__char_spans)

    def convert_to_char_based(self, label: SequenceLabel) -> SequenceLabel:
        """Converts token-based tags to character-based tags.

        Args:
            label: A token-based sequence label.

        Returns:
            A character-based sequence label.
        """
        if self.num_tokens != label.size:
            raise ValueError(
                "label.size must be the same as num_tokens: "
                f"{label.size} != {self.num_tokens}"
            )

        if label.base is not Base.TOKEN:
            raise ValueError(f"label.base must be Base.TOKEN: {label.base}")

        tags = []
        for tag in label.tags:
            token_span = tag.span

            char_span_start = self.__char_spans[token_span.start]
            char_span_end = self.__char_spans[token_span.start + token_span.length - 1]

            if char_span_start is None or char_span_end is None:
                continue

            tags.append(
                Tag.create(
                    start=char_span_start.start,
                    end=char_span_end.start + char_span_end.length,
                    label=tag.label,
                )
            )

        return SequenceLabel(
            tags=tuple(tags), size=self.char_length, base=Base.CHARACTER
        )

    def convert_to_token_based(self, label: SequenceLabel) -> SequenceLabel:
        """Converts character-based tags to token-based tags. Note that this operation
        is irreversible. For example, if a text is truncated in tokenization,
        tags associated with a truncated part will be ignored.

        Args:
            label: A character-based sequence label.

        Returns:
            A token-based sequence label.
        """
        if self.char_length != label.size:
            raise ValueError(
                "label.size must be the same as char_length: "
                f"{label.size} != {self.char_length}"
            )

        if label.base != Base.CHARACTER:
            raise ValueError(f"label.base must be Base.CHARACTER: {label.base}")

        tags = []
        for tag in label.tags:
            start = self.__token_indices[tag.start]
            end = self.__token_indices[tag.start + tag.length - 1]  # inclusive
            if start == -1 or end == -1:
                # There is no char span which strictly corresponds a given tag.
                continue
            tags.append(Tag.create(start=start, end=end + 1, label=tag.label))

        return SequenceLabel(tags=tuple(tags), size=self.num_tokens, base=Base.TOKEN)


class Move(Enum):
    OUTSIDE = auto()
    START = auto()
    INSIDE = auto()
    END = auto()
    UNIT = auto()


class LabelSet:
    """A label set represents a set of labels used for tagging, where each label
    has four states (start, inside, end, unit).

    Args:
        labels: A set of strings, where each string represents a label.
    """

    def __init__(self, labels: set[str], padding_index: int = -1):
        self.__outside_index = 0
        self.__padding_index = padding_index

        self.__start_indices = {}
        self.__inside_indices = {}
        self.__end_indices = {}
        self.__unit_indices = {}

        self.__states: list[tuple[Move, str | None]] = [(Move.OUTSIDE, None)]

        for label in sorted(labels):
            self.__start_indices[label] = self.state_size
            self.__inside_indices[label] = self.state_size
            self.__end_indices[label] = self.state_size
            self.__unit_indices[label] = self.state_size
            for move in (Move.START, Move.INSIDE, Move.END, Move.UNIT):
                self.__states.append((move, label))

        self.start_states = self.__get_start_states()
        self.transitions = self.__get_transitions()
        self.end_states = self.__get_end_states()

    @property
    def state_size(self) -> int:
        return 1 + sum(
            map(
                len,
                (
                    self.__start_indices,
                    self.__inside_indices,
                    self.__end_indices,
                    self.__unit_indices,
                ),
            )
        )

    @property
    def outside_index(self) -> int:
        return self.__outside_index

    @property
    def padding_index(self) -> int:
        return self.__padding_index

    def __contains__(self, label: str) -> bool:
        return label in self.__start_indices

    def get_tag_indices(self, tag: Tag) -> list[int]:
        if tag.label not in self:
            raise ValueError(f"Invalid label is given: {tag.label}")

        if tag.length == 1:
            return [self.__unit_indices[tag.label]]
        else:
            rest = tag.length - 2
            return (
                [self.__start_indices[tag.label]]
                + [self.__inside_indices[tag.label]] * rest
                + [self.__end_indices[tag.label]]
            )

    def get_tag_bitmap(self, tag: Tag) -> list[list[bool]]:
        indices = self.get_tag_indices(tag)

        bitmap = [[False] * self.state_size for _ in range(tag.length)]
        for i, j in enumerate(indices):
            bitmap[i][j] = True

        return bitmap

    def encode_to_tag_indices(
        self, labels: tuple[SequenceLabel, ...], alignments: tuple[LabelAlignment, ...]
    ) -> list[list[int]]:
        """Creates a list of active tag indices where given tags are expected
        to be character-based.

        Args:
            label: An instance of SequenceLabel.

        Returns:
            A list of integers, where each integer represents an active tag.

        """
        if len(labels) != len(alignments):
            raise ValueError()

        labels_token_based = [
            alignment.convert_to_token_based(label)
            for label, alignment in zip(labels, alignments)
        ]

        max_size = max(label.size for label in labels_token_based)

        batch = []
        for label in labels_token_based:
            tag_indices = [self.outside_index] * label.size + [self.padding_index] * (
                max_size - label.size
            )

            for tag in label.tags:
                start = tag.start
                end = tag.start + tag.length
                tag_indices[start:end] = self.get_tag_indices(tag)

            batch.append(tag_indices)

        return batch

    def encode_to_tag_bitmap(
        self, labels: tuple[SequenceLabel, ...], alignments: tuple[LabelAlignment, ...]
    ) -> list[list[list[bool]]]:
        """Creates a tag bitmap indicating the presence of active tags for each token
        where given tags are expected to be character-based.

        Args:
            label: An instance of SequenceLabel.

        Returns:
            A list of lists of booleans, where each boolean represents an active tag.

        """
        if len(labels) != len(alignments):
            raise ValueError()

        labels_token_based = [
            alignment.convert_to_token_based(label)
            for label, alignment in zip(labels, alignments)
        ]

        max_size = max(label.size for label in labels_token_based)

        batch = []
        for label in labels_token_based:
            tag_bitmap = [[False] * self.state_size for _ in range(max_size)]
            for tag in label.tags:
                start = tag.start
                for i, bitmap in enumerate(self.get_tag_bitmap(tag), start):
                    tag_bitmap[i] = [a or b for a, b in zip(tag_bitmap[i], bitmap)]

            for i in range(label.size):
                if sum(tag_bitmap[i]) > 0:
                    continue
                tag_bitmap[i][self.outside_index] = True

            batch.append(tag_bitmap)

        return batch

    def decode(
        self, tag_indices: list[list[int]], alignments: tuple[LabelAlignment, ...]
    ) -> tuple[SequenceLabel, ...]:
        """Creates a set of character-based tags from given tag indices.

        Args:
            tag_indices: A list of integer, where each item represents a tag index.

        Returns:
            A character-based sequence label.
        """
        if len(tag_indices) != len(alignments):
            raise ValueError()

        if any(len(indices) <= 0 for indices in tag_indices):
            raise ValueError("Invalid indices.")

        tag_indices = [
            [i for i in indices if i != self.padding_index] for indices in tag_indices
        ]

        for indices in tag_indices:
            # Check if given tag indices are accepted by DFA
            if not self.start_states[indices[0]]:
                raise ValueError("Invalid indices.")

            if not self.end_states[indices[-1]]:
                raise ValueError("Invalid indices.")

            for i, j in zip(indices[:-1], indices[1:]):
                if not self.transitions[i][j]:
                    raise ValueError("Invalid indices.")

        labels = []
        for indices, alignment in zip(tag_indices, alignments):
            tags = []
            for now, index in enumerate(indices):
                move, label = self.__states[index]
                if label is None:
                    continue

                if move is Move.UNIT:
                    tags.append(Tag.create(now, now + 1, label))
                elif move is Move.END:
                    prev = now
                    while self.__states[indices[prev]][0] is not Move.START:
                        prev -= 1
                    tags.append(Tag.create(prev, now + 1, label))

            labels.append(
                alignment.convert_to_char_based(
                    SequenceLabel(tags=tuple(tags), size=len(indices), base=Base.TOKEN)
                )
            )

        return tuple(labels)

    def __get_start_states(self) -> tuple[bool, ...]:
        """Returns a list of booleans representing an allowed start states.

        Returns:
            A list of booleans representing allowed start states,
            where each item is: True for its index allowed and False otherwise.

        """
        states = [False] * self.state_size
        # Always allowed starts from outside status
        states[self.outside_index] = True

        for index in chain(self.__start_indices.values(), self.__unit_indices.values()):
            states[index] = True

        return tuple(states)

    def __get_end_states(self) -> tuple[bool, ...]:
        """Returns a list of booleans representing an allowed end states.

        Returns:
            A list of booleans representing allowed end states,
            where each item is: True for its index allowed and False otherwise.

        """
        states = [False] * self.state_size
        # Always allowed ends with outside status
        states[self.outside_index] = True

        for index in chain(self.__end_indices.values(), self.__unit_indices.values()):
            states[index] = True

        return tuple(states)

    def __get_transitions(self) -> tuple[tuple[bool, ...], ...]:
        """Returns a list of lists of booleans representing
        allowed transitions between tags.

        Returns:
            A list of lists of booleans representing allowed transitions between tags,
            where each item is: True for an allowed transition and False otherwise.

        """
        transitions = [[False] * self.state_size for _ in range(self.state_size)]

        outside_index = self.outside_index

        transitions[outside_index][outside_index] = True
        for ok_index in chain(
            self.__start_indices.values(), self.__unit_indices.values()
        ):
            transitions[outside_index][ok_index] = True

        for label, start_index in self.__start_indices.items():
            transitions[start_index][self.__inside_indices[label]] = True
            transitions[start_index][self.__end_indices[label]] = True

        for label, inside_index in self.__inside_indices.items():
            transitions[inside_index][self.__inside_indices[label]] = True
            transitions[inside_index][self.__end_indices[label]] = True

        for end_index in self.__end_indices.values():
            transitions[end_index][outside_index] = True
            for ok_index in chain(
                self.__start_indices.values(), self.__unit_indices.values()
            ):
                transitions[end_index][ok_index] = True

        for unit_index in self.__unit_indices.values():
            transitions[unit_index][outside_index] = True
            for ok_index in chain(
                self.__start_indices.values(), self.__unit_indices.values()
            ):
                transitions[unit_index][ok_index] = True

        return tuple(tuple(row) for row in transitions)
