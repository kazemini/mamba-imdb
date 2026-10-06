# LSTM vs. Mamba for IMDB Sentiment Classification

This project compares two sequence models for binary sentiment classification on the IMDB movie-review dataset:

- LSTM baseline: implemented in `train_lstm.ipynb`
- Mamba model: implemented in `train_mamba.ipynb`

Both notebooks use the same dataset, tokenizer, embedding size, training pipeline, and evaluation protocol so the architectural difference is isolated as much as possible.

---

## Project goal

The central question is:

> How does a standard LSTM compare with a Mamba-based sequence model when both are trained on the same IMDB data and evaluated under the same conditions?

The repository keeps the data pipeline fixed and swaps only the sequence encoder, making it a fair experimental comparison rather than a completely different training setup.

---

## Dataset

The project uses the IMDB sentiment dataset with 50,000 movie reviews.

- Total reviews: 50,000
- Positive: 25,000
- Negative: 25,000
- Task: binary sentiment classification

The notebooks split the data into:

| Split | Samples |
| --- | ---: |
| Train | 40,000 |
| Validation | 5,000 |
| Test | 5,000 |

The split is stratified and fixed for consistent comparison across experiments.

---

## Data pipeline

The pipeline is shared across both models:

```text
IMDB reviews
  -> preprocess and clean text
  -> stratified train/validation/test split
  -> word-level tokenizer trained on training split only
  -> token IDs + dynamic padding
  -> embedding layer
  -> sequence encoder (LSTM or Mamba)
  -> classifier head
  -> CrossEntropyLoss
  -> optimization and evaluation
```

The custom tokenizer and dataset utilities are defined under:

```text
my_tokenizers/
my_datasets/
my_trainers/
my_classifier/
```

The notebook training flow includes:

1. load the IMDB dataset
2. create the train/validation/test split
3. train a word-level tokenizer on the training data
4. build PyTorch datasets and dataloaders
5. define the model
6. optimize with Adam
7. train for 5 epochs
8. evaluate on validation and test sets
9. save the trained model and tokenizer

---

## Project structure

```text
mamba-imdb/
├── train_lstm.ipynb
├── train_mamba.ipynb
├── my_classifier/
│   ├── __init__.py
│   ├── lstm_classifier.py
│   └── mamba_classifier.py
├── my_datasets/
│   ├── imdb_dataset.py
│   └── imdb_torch_dataset.py
├── my_tokenizers/
│   ├── base_tokenizer.py
│   ├── text_tokenizer.py
│   └── word_level_tokenizer.py
├── my_trainers/
│   └── trainer.py
├── imdb_tokenizer.json
├── mamba_imdb_tokenizer.json
├── lstm_imdb.pt
├── mamba_imdb.pt
├── readme.md
└── .gitignore
```

---

## Model implementations

### 1) LSTM baseline

The LSTM notebook defines a single-layer LSTM classifier.

Architecture:

```text
Embedding(20000, 128)
  -> LSTM(128, 128, batch_first=True)
  -> final hidden state
  -> Linear(128, 2)
```

Key setup from `train_lstm.ipynb`:

- embedding dimension: 128
- hidden dimension: 128
- number of classes: 2
- loss: `nn.CrossEntropyLoss()`
- optimizer: `torch.optim.Adam(model.parameters(), lr=1e-3)`
- training epochs: 5
- padding handled with `pack_padded_sequence`

### 2) Mamba classifier

The Mamba notebook uses a Mamba state-space block from `mamba_ssm`.

Architecture:

```text
Embedding(20000, 128, padding_idx=0)
  -> Mamba3(d_model=128, d_state=64, headdim=32)
  -> mean-pooled real-token features
  -> Linear(128, 64) -> GELU -> Dropout(0.2) -> Linear(64, 2)
```

Key setup from `train_mamba.ipynb`:

- embedding dimension: 128
- Mamba state size: 64
- Mamba head dim: 32
- dropout: 0.2
- loss: `nn.CrossEntropyLoss()`
- optimizer: `torch.optim.Adam(model.parameters(), lr=5e-4)`
- training epochs: 5
- padding-aware masking and mean pooling for sequence summarization

---

## Training summary

Both notebooks train with the same data split and evaluation logic through the shared `Trainer` class in `my_trainers/trainer.py`.

The trainer reports:

- training loss
- training accuracy
- validation loss
- validation accuracy
- test loss
- test accuracy

---

## Results

### LSTM results

The LSTM notebook reached these validation metrics over 5 training epochs:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5663 | 0.7015 | 0.4625 | 0.7864 |
| 1 | 0.3528 | 0.8425 | 0.2578 | 0.8960 |
| 2 | 0.2016 | 0.9223 | 0.2343 | 0.9062 |
| 3 | 0.1353 | 0.9517 | 0.2412 | 0.9068 |
| 4 | 0.0835 | 0.9729 | 0.2960 | 0.9036 |

Final test result:

- Test Loss: 0.2902
- Test Accuracy: 0.9054

### Mamba results

The Mamba notebook reached these validation metrics over 5 training epochs:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4173 | 0.8059 | 0.3134 | 0.8680 |
| 1 | 0.2457 | 0.9020 | 0.2679 | 0.8892 |
| 2 | 0.1502 | 0.9456 | 0.2817 | 0.8928 |
| 3 | 0.0727 | 0.9750 | 0.3766 | 0.8878 |
| 4 | 0.0320 | 0.9890 | 0.4839 | 0.8854 |

Final test result:

- Test Loss: 0.4878
- Test Accuracy: 0.8892

### Comparison

| Model | Best Validation Accuracy | Test Accuracy |
| --- | ---: | ---: |
| LSTM | 0.9068 | 0.9054 |
| Mamba | 0.8928 | 0.8892 |

Under this experimental setup, the LSTM baseline performs slightly better than the Mamba model in validation and test accuracy, while the Mamba model reaches strong accuracy quickly and trains competitively.

---

## Benchmark note

The notebooks also include a simple benchmark-style evaluation on a small sample set. The reported benchmark accuracies in the notebooks are:

- LSTM benchmark accuracy: 62.50%
- Mamba benchmark accuracy: 57.50%

These benchmark values are part of the exploratory validation workflow and are not the primary comparison metric for the full test set.

---

## Reproduction

To reproduce the experiments:

1. open `train_lstm.ipynb` or `train_mamba.ipynb`
2. run the cells in order
3. the notebook will load the dataset, train the tokenizer, build the model, train, evaluate, and save the weights

The trained artifacts are stored as:

- `lstm_imdb.pt` + `imdb_tokenizer.json`
- `mamba_imdb.pt` + `mamba_imdb_tokenizer.json`

---

## Current status

This repository contains completed LSTM and Mamba training experiments for the IMDB text-classification task. The code and results currently show that the LSTM setup is the stronger performer in this controlled comparison, while the Mamba model remains a valid and competitive alternative.

---

## License

This project is distributed under the MIT license.
