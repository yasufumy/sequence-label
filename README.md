# sequence-label

`sequence-label` is a Python library for preparing a dataset for a sequence labeling task.

## Usage


```py
from transformers import AutoTokenizer

from sequence_label import LabelSet, SequenceLabel
from sequence_label.transformers import get_alignments


text1 = "Tokyo is the capital of Japan."
label1 = SequenceLabel.from_dict(
    tags=[
        {"start": 0, "end": 5, "label": "LOC"},
        {"start": 24, "end": 29, "label": "LOC"},
    ],
    size=len(text1),
)

text2 = "The Monster Naoya Inoue is the who's who of boxing."
label2 = SequenceLabel.from_dict(
    tags=[{"start": 12, "end": 23, "label": "PER"}],
    size=len(text2),
)

texts = (text1, text2)
labels = (label1, label2)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
batch_encoding = tokenizer(texts)

alignments = get_alignments(
    batch_encoding=batch_encoding,
    lengths=list(map(len, texts)),
    padding_token=tokenizer.pad_token
)


label_set = LabelSet(labels={"ORG", "LOC", "PER", "MISC"})

tag_indices = label_set.encode_to_tag_indices(labels=labels, alignments=alignments)

assert labels == label_set.decode(tag_indices=tag_indices, alignments=alignments)

```

## Installation

```
pip install sequence-label
```
