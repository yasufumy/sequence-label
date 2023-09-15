# sequence-label

`sequence-label` is a Python library that streamlines the process of creating tensors for sequence labeling data and reconstructing sequence labeling data from the output tensors of neural models. Whether you're working on named entity recognition, part-of-speech tagging, or any other sequence labeling task, this library offers a convenient utility to simplify your workflow.

## Basic Usage

Import the necessary dependencies:

```py
from transformers import AutoTokenizer

from sequence_label import LabelSet, SequenceLabel
from sequence_label.transformers import get_alignments
```

Start by creating sequence label data using the `SequenceLabel.from_dict` method. Define your text and associated labels:

```py
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
```

Next, tokenize your `texts` and calculate the `alignments` using the `get_alignments` function. `alignments` is an instance of `LabelAlignment` that aligns sequence labels with the tokenized result:

```py
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
batch_encoding = tokenizer(texts)

alignments = get_alignments(
    batch_encoding=batch_encoding,
    lengths=list(map(len, texts)),
    padding_token=tokenizer.pad_token
)
```

Now, create a `label_set` that will allow you to create tensors and reconstruct sequence labels from tensors. You can use the `LabelSet` class for this purpose. By calling `label_set.encode_to_tag_indices`, you can generate `tag_indices`:

```py
label_set = LabelSet(
    labels={"ORG", "LOC", "PER", "MISC"},
    padding_index=-1,
)

tag_indices = label_set.encode_to_tag_indices(labels=labels, alignments=alignments)
```

Finally, use the `label_set.decode` method to reconstruct the sequence labels from `tag_indices` and `alignments`:

```py
labels2 = label_set.decode(tag_indices=tag_indices, alignments=alignments)

assert labels == labels2
```

## Installation

```
pip install sequence-label
```
