from collections.abc import Sequence, Set
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
from typing import TypedDict

__all__ = [
    "Span",
    "Tag",
    "SequenceLabel",
    "LabelSet",
    "LabelAlignment",
    "TagDict",
    "Base",
]


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
    def create(cls, start: int, end: int, label: str) -> "Tag":
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
        return cls(span=Span(start=start, length=length), label=label)


class TagDict(TypedDict):
    start: int
    end: int
    label: str


class Base(Enum):
    Source = auto()
    Target = auto()


@dataclass(frozen=True)
class SequenceLabel:
    tags: Sequence[Tag]
    size: int
    base: Base = Base.Source

    def __post_init__(self) -> None:
        if any(self.tags[i] > self.tags[i + 1] for i in range(len(self.tags) - 1)):
            raise ValueError(
                "tags must be sorted by tag.start, tag.length and tag.label."
            )

        for tag in self.tags:
            if tag.start < 0 or tag.start >= self.size:
                raise ValueError(
                    "An invalid tag is found. start must be"
                    f" between 0 and {self.size - 1}: {tag.start}"
                )
            end = tag.start + tag.length
            if end < 0 or end > self.size:
                raise ValueError(
                    "An invalid tag is found. length must be"
                    f" between 1 and {self.size}: {tag.length}"
                )

    @classmethod
    def from_dict(
        cls, tags: Sequence[TagDict], size: int, base: Base = Base.Source
    ) -> "SequenceLabel":
        """A named constructor for creating an instance of SequenceLabel from a sequence
        of dictionaries, where each dictionary has three keys start, end, and label.
        A start represents a position in the text where a tag starts. A end represents
        a position in the text where a tag ends. A label represents what you want to
        assign to a span of the text defined by a start and a end.

        Args:
            tags: A sequence of an instance of TagDict.
            size: A integer representing a length of a text.
            base: A member of Base. Defaults to Base.SOURCE.

        Returns:
            An instance of SequenceLabel.
        """
        return cls(
            tags=sorted(
                Tag.create(start=tag["start"], end=tag["end"], label=tag["label"])
                for tag in tags
            ),
            size=size,
            base=base,
        )


class LabelAlignment:
    """The LabelAlignment class manages the alignment of labels from source sequences
    to target sequences after tokenization, and vice versa.

    Args:
        source_spans: A sequence where each item represents an interval in the source
            sequence. The index of the item in the sequence corresponds to a specific
            position in the target sequence. In other words, an interval in
            the source sequence and a position in the target sequence have a one-to-one
            correspondence.
            If a span is None, it indicates that the respective position in the target
            sequence doesn't have a matching interval in the source sequence.
        target_spans: A sequence where each item represents an interval in the target
            sequence. The index of the item in the sequence corresponds to a specific
            position in the source sequence. In other words, an interval in
            the target sequence and a position in the source sequence have a one-to-one
            correspondence.
            If a span is None, it indicates that the respective position in the source
            sequence doesn't have a matching interval in the target sequence.

    Attributes:
        source_size: An integer representing the number of items in the source sequence.
        target_size: An integer representing the number of items in the target sequence.
    """

    def __init__(
        self,
        source_spans: Sequence[Span | None],
        target_spans: Sequence[Span | None],
    ):
        target_size = len(source_spans)
        if not all(
            0 <= span.start < target_size  # check if start is valid
            and 0 <= span.start + span.length - 1 < target_size  # check if end is valid
            for span in target_spans
            if span is not None
        ):
            raise ValueError(
                "Each item in token_spans must be None or"
                f" between 0 and {target_size - 1}: {target_spans}"
            )

        source_size = len(target_spans)
        if not all(
            0 <= span.start < source_size  # check if start is valid
            and 0 <= span.start + span.length - 1 < source_size  # check if end is valid
            for span in source_spans
            if span is not None
        ):
            raise ValueError(
                "Each span in char_spans must be None or"
                f" between 0 and {source_size - 1}: {source_spans}"
            )

        self.__source_spans = source_spans
        self.__target_spans = target_spans

    @property
    def source_size(self) -> int:
        return len(self.__target_spans)

    @property
    def target_size(self) -> int:
        return len(self.__source_spans)

    def get_span_lengths(self, base: Base) -> Sequence[int]:
        if base == Base.Source:
            return [span.length if span else 0 for span in self.__source_spans]
        elif base == Base.Target:
            return [span.length if span else 0 for span in self.__target_spans]
        else:
            raise ValueError(f"{base} is not supported.")

    def align_with_source(self, label: SequenceLabel) -> SequenceLabel:
        """Converts token-based tags to character-based tags.

        Args:
            label: A token-based sequence label.

        Returns:
            A character-based sequence label.
        """
        return self.__align(label=label, src=Base.Target, tgt=Base.Source)

    def align_with_target(self, label: SequenceLabel) -> SequenceLabel:
        """Converts character-based tags to token-based tags. Note that this operation
        is irreversible. For example, if a text is truncated in tokenization,
        tags associated with a truncated part will be ignored.

        Args:
            label: A character-based sequence label.

        Returns:
            A token-based sequence label.
        """
        return self.__align(label=label, src=Base.Source, tgt=Base.Target)

    def __align(self, label: SequenceLabel, src: Base, tgt: Base) -> SequenceLabel:
        if tgt == Base.Target:
            source_size = self.source_size
            target_size = self.target_size
            spans = self.__target_spans
        else:
            source_size = self.target_size
            target_size = self.source_size
            spans = self.__source_spans

        if label.size != source_size:
            raise ValueError(
                f"label.size must be the same as {source_size}: {label.size} "
            )

        if label.base is not src:
            raise ValueError(f"label.base must be {src}: {label.base}")

        tags = []
        for tag in label.tags:
            target_span = tag.span

            source_span_start = spans[target_span.start]
            source_span_end = spans[target_span.start + target_span.length - 1]

            if source_span_start is None or source_span_end is None:
                continue

            tags.append(
                Tag.create(
                    start=source_span_start.start,
                    end=source_span_end.start + source_span_end.length,
                    label=tag.label,
                )
            )

        return SequenceLabel(tags=tags, size=target_size, base=tgt)


class Move(Enum):
    Outside = auto()
    Start = auto()
    Inside = auto()
    End = auto()
    Unit = auto()


class LabelSet:
    """The LabelSet class manages a set of labels and provides functionality for
    creating tensors from sequence labeling data and reconstructing sequence labeling
    data from tensors. Each label binds with four states, start, end, inside, and
    unit for sequence labeling purposes.

    Args:
        labels: A set of strings, where each string represents a label.

    Attributes:
        state_size: An integer representing total number of labels with states.
        outside_index: An integer corresponding to an outside state.
        padding_index: An integer representing a padding value.
        start_states: A sequence of boolean values representing the start states.
            True indicates an allowed state, while False indicates an otherwise state.
        end_states: A sequence of boolean values representing the end states.
            True indicates an allowed state, while False indicates an otherwise state.
        transitions: A sequence of sequence of boolean values representing the
            transitions. True indicates an allowed transition,
            while False indicates an otherwise transition.
    """

    def __init__(self, labels: Set[str], padding_index: int = -1):
        self.__outside_index = 0
        self.__padding_index = padding_index

        self.__start_indices = {}
        self.__inside_indices = {}
        self.__end_indices = {}
        self.__unit_indices = {}

        self.__states: Sequence[tuple[Move, str | None]] = [(Move.Outside, None)]

        for label in sorted(labels):
            self.__start_indices[label] = self.state_size
            self.__inside_indices[label] = self.state_size
            self.__end_indices[label] = self.state_size
            self.__unit_indices[label] = self.state_size
            for move in (Move.Start, Move.Inside, Move.End, Move.Unit):
                self.__states.append((move, label))

        self.start_states = self.__create_start_states()
        self.transitions = self.__create_transitions()
        self.end_states = self.__create_end_states()

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
        return all(
            label in indices
            for indices in (
                self.__start_indices,
                self.__inside_indices,
                self.__end_indices,
                self.__unit_indices,
            )
        )

    def get_tag_indices(self, tag: Tag) -> Sequence[int]:
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

    def get_tag_bitmap(self, tag: Tag) -> Sequence[Sequence[bool]]:
        indices = self.get_tag_indices(tag=tag)

        bitmap = [[False] * self.state_size for _ in range(tag.length)]
        for i, j in enumerate(indices):
            bitmap[i][j] = True

        return bitmap

    def encode_to_tag_indices(
        self,
        labels: Sequence[SequenceLabel],
        alignments: Sequence[LabelAlignment] | None = None,
    ) -> Sequence[Sequence[int]]:
        """Creates a sequence of active tag indices from given labels.

        Args:
            labels: A sequence of instances of SequenceLabel.
            alignments: A sequence of instances of LabelAlignment. Defaults to None.

        Returns:
            A sequence of integers, where each integer represents an active tag.

        """
        if alignments is not None and len(labels) != len(alignments):
            raise ValueError(
                "The size of labels must be the same as its alignments: "
                f"{len(labels)} != {len(alignments)}"
            )

        if alignments is not None:
            labels = [
                alignment.align_with_target(label=label)
                for label, alignment in zip(labels, alignments, strict=False)
            ]

        max_size = max(label.size for label in labels)

        batch = []
        for label in labels:
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
        self,
        labels: Sequence[SequenceLabel],
        alignments: Sequence[LabelAlignment] | None = None,
    ) -> Sequence[Sequence[Sequence[bool]]]:
        """Creates a tag bitmap indicating the presence of active tags from
        given labels.

        Args:
            labels: A sequence of instances of SequenceLabel.
            alignments: A sequence of instances of LabelAlignment. Defaults to None.

        Returns:
            A sequence of sequences of booleans, where each boolean represents
            an active tag.

        """
        if alignments is not None and len(labels) != len(alignments):
            raise ValueError(
                "The size of labels must be the same as its alignments: "
                f"{len(labels)} != {len(alignments)}"
            )

        if alignments is not None:
            labels = [
                alignment.align_with_target(label=label)
                for label, alignment in zip(labels, alignments, strict=False)
            ]

        max_size = max(label.size for label in labels)

        batch = []
        for label in labels:
            tag_bitmap = [[False] * self.state_size for _ in range(max_size)]
            for tag in label.tags:
                start = tag.start
                for i, bitmap in enumerate(self.get_tag_bitmap(tag=tag), start):
                    tag_bitmap[i] = [
                        a or b for a, b in zip(tag_bitmap[i], bitmap, strict=False)
                    ]

            for i in range(label.size):
                if sum(tag_bitmap[i]) > 0:
                    continue
                tag_bitmap[i][self.outside_index] = True

            batch.append(tag_bitmap)

        return batch

    def decode(
        self,
        tag_indices: Sequence[Sequence[int]],
        alignments: Sequence[LabelAlignment] | None = None,
    ) -> Sequence[SequenceLabel]:
        """Reconstructs labels from given tag indices.

        Args:
            tag_indices: A sequence of integer, where each item represents a tag index.
            alignments: A sequence of instances of LabelAlignment. Defaults to None.

        Returns:
            A sequence of instances of SequenceLabel.
        """
        if alignments is not None and len(tag_indices) != len(alignments):
            raise ValueError(
                "The size of tag_indices must be the same as its alignments: "
                f"{len(tag_indices)} != {len(alignments)}"
            )

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

            for i, j in zip(indices[:-1], indices[1:], strict=False):
                if not self.transitions[i][j]:
                    raise ValueError("Invalid indices.")

        base = Base.Source if alignments is None else Base.Target
        labels = []
        for indices in tag_indices:
            tags = []
            for now, index in enumerate(indices):
                move, label = self.__states[index]
                if label is None:
                    continue

                if move is Move.Unit:
                    tags.append(Tag.create(start=now, end=now + 1, label=label))
                elif move is Move.End:
                    prev = now
                    while self.__states[indices[prev]][0] is not Move.Start:
                        prev -= 1
                    tags.append(Tag.create(start=prev, end=now + 1, label=label))

            labels.append(SequenceLabel(tags=tags, size=len(indices), base=base))

        if alignments is not None:
            return [
                alignment.align_with_source(label)
                for label, alignment in zip(labels, alignments, strict=False)
            ]
        else:
            return labels

    def __create_start_states(self) -> Sequence[bool]:
        """Creates a sequence of booleans representing an allowed start states.

        Returns:
            A sequence of booleans representing allowed start states,
            where each item is: True for its index allowed and False otherwise.

        """
        states = [False] * self.state_size
        # Always allowed starts from outside status
        states[self.outside_index] = True

        for index in chain(self.__start_indices.values(), self.__unit_indices.values()):
            states[index] = True

        return states

    def __create_end_states(self) -> Sequence[bool]:
        """Creates a sequence of booleans representing an allowed end states.

        Returns:
            A sequence of booleans representing allowed end states,
            where each item is: True for its index allowed and False otherwise.

        """
        states = [False] * self.state_size
        # Always allowed ends with outside status
        states[self.outside_index] = True

        for index in chain(self.__end_indices.values(), self.__unit_indices.values()):
            states[index] = True

        return states

    def __create_transitions(self) -> Sequence[Sequence[bool]]:
        """Creates a sequence of sequences of booleans representing
        allowed transitions between tags.

        Returns:
            A sequence of sequences of booleans representing allowed transitions between
            tags, where each item is: True for an allowed transition
            and False otherwise.

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

        return transitions
