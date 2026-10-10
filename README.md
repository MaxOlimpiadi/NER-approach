# Agentivity Detection with Token Classification

A research project fine-tuning **German BERT** to detect agentivity-related spans in literary texts using an **NER-style token classification approach**.

The model predicts span boundaries and agentivity categories through BIO tags, rather than assigning one label to an entire text.

**Labels**

| Tags | Category |
|---|---|
| `B-AG`, `I-AG` | Agentive |
| `B-LA`, `I-LA` | Low-agentive |
| `B-PS`, `I-PS` | Passive |
| `O` | Outside an annotated span |

`B-` marks the beginning of a span, and `I-` marks its continuation.

**Model and Training**

- **Model:** `bert-base-german-cased`.
- **Learning rate:** `2e-5`.
- **Weight decay:** `0.01`.
- **Epochs:** 3.
- **Batch size:** 8 per device.
- **Evaluation and checkpointing:** after each epoch.
- **Best checkpoint:** selected by validation F1.


**Project Structure**

```text
main.py                  # Annotation conversion, training, and evaluation
raw_data/                # Original annotated text chunks
processed_data/
  train.jsonl            # Training examples
  dev.jsonl              # Validation examples
  test.jsonl             # Test examples
```

Raw annotations and prepared splits are included in the repository.

**Data Format**

Each prepared JSONL record contains:

```json
{
  "documentName": "example_document",
  "tokens": ["Er", "öffnet", "die", "Tür", "."],
  "labels": ["B-AG", "O", "O", "O", "O"]
}
```

The `tokens` and `labels` lists must have the same length.

The script uses existing train/dev/test files. The procedure used to create these splits is not implemented in this repository.


**Running**

Run from the project root:

```bash
python main.py
```

The script loads the prepared splits, tokenizes the examples, fine-tunes the model, and evaluates the best checkpoint on the test set.

