# LSTM vs Mamba for Sentiment Classification

An educational and experimental project for studying and comparing **LSTM** and **Mamba** sequence models on the **IMDB 50K movie review sentiment classification** task.

The main goal of this project is not only to achieve high accuracy, but to understand how different sequence modeling architectures process sequential data while keeping the surrounding data pipeline as consistent as possible.

---

## Project Goal

This project follows a simple experimental question:

> **How does Mamba compare with an LSTM for sequence classification when both models use the same dataset, tokenizer, embedding setup, training pipeline, and evaluation procedure?**

The project starts with an LSTM baseline and will later introduce Mamba while keeping the rest of the pipeline as consistent as possible.

---

## Dataset

The project uses the  **IMDB Dataset of 50K Movie Reviews** .

* Total samples: **50,000**
* Positive reviews: **25,000**
* Negative reviews: **25,000**
* Task: Binary sentiment classification

Current split:

| Split           |          Samples |
| --------------- | ---------------: |
| Train           |           40,000 |
| Validation      |            5,000 |
| Test            |            5,000 |
| **Total** | **50,000** |

The dataset is split using stratified sampling with a fixed random seed.

---

## Pipeline

```text
IMDB 50K
   │
   ▼
Text Preprocessing
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Dynamic Padding
   │
   ▼
Embedding
   │
   ├───────────────┐
   ▼               ▼
  LSTM            Mamba
   │               │
   ▼               ▼
Sequence Representation
   │               │
   └───────┬───────┘
           ▼
       Classifier
           │
           ▼
         Logits
           │
           ▼
     Cross Entropy Loss
           │
           ▼
       Backpropagation
```

---

## Project Structure

```text
lstm-vs-mamba/
│
├── my_tokenizers/
│   ├── base_tokenizer.py
│   ├── word_level_tokenizer.py
│   └── text_tokenizer.py
│
├── my_datasets/
│   ├── imdb_dataset.py
│   └── imdb_torch_dataset.py
│
├── my_classifier/
│   └── lstm_classifier.py
│
├── my_trainers/
│   └── trainer.py
│
├── train.ipynb
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Tokenization

The project uses a custom tokenizer abstraction so that different tokenization strategies can be introduced without changing the rest of the training pipeline.

Current tokenizer:

**Word-Level Tokenization**

The tokenizer is trained **only on the training split** to avoid vocabulary leakage from validation and test data.

Special tokens:

```text
<PAD>
<UNK>
```

The tokenizer is responsible for:

* Training the vocabulary
* Tokenization
* Encoding text into token IDs
* Decoding token IDs back into text
* Providing vocabulary size
* Providing the padding token ID

---

## Dataset Pipeline

The dataset is represented using a custom PyTorch `Dataset`.

Each sample contains:

```python
{
    "input_ids": [...],
    "label": 0 or 1
}
```

Because reviews have different lengths, batches use  **dynamic padding** .

The DataLoader produces:

```text
input_ids : [B, S]
lengths   : [B]
labels    : [B]
```

where:

* `B` = batch size
* `S` = maximum sequence length in the batch

---

## LSTM Baseline

The first sequence model is a single-layer LSTM.

Architecture:

```text
Token IDs
   │
   ▼
Embedding
   │
   ▼
LSTM
   │
   ▼
Final Hidden State
   │
   ▼
Linear Classifier
   │
   ▼
2 Classes
```

Current configuration:

```text
Embedding dimension : 128
Hidden dimension    : 128
LSTM layers         : 1
Number of classes   : 2
```

Packed sequences are used so that padding tokens do not participate in the LSTM computation.

---

## Initial LSTM Results

After 5 training epochs:

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ----: | ---------: | -------------: | --------------: | ------------------: |
|     1 |     0.5721 |         69.45% |          0.4968 |              76.90% |
|     2 |     0.4798 |         77.04% |          0.4401 |              80.60% |
|     3 |     0.3146 |         87.00% |          0.2917 |              87.86% |
|     4 |     0.2165 |         91.70% |          0.2654 |              89.04% |
|     5 |     0.1651 |         94.03% |          0.2617 |    **89.44%** |

The test set has not been used during model selection.

---

## Model Comparison

The experiment will progressively compare sequence modeling architectures under a shared pipeline.

| Component  | LSTM          | Mamba         |
| ---------- | ------------- | ------------- |
| Dataset    | IMDB 50K      | IMDB 50K      |
| Tokenizer  | Word-Level    | Word-Level    |
| Vocabulary | Same          | Same          |
| Embedding  | 128           | 128           |
| Classifier | Linear        | Linear        |
| Loss       | Cross Entropy | Cross Entropy |
| Optimizer  | Same          | Same          |
| Evaluation | Same          | Same          |

The goal is to change the **sequence modeling component** while keeping the surrounding experiment as consistent as possible.

---

## Installation

Create a Python environment and install the required dependencies.

```bash
pip install -r requirements.txt
```

The project currently uses:

* Python
* PyTorch
* Hugging Face Tokenizers
* Pandas
* NumPy
* KaggleHub
* tqdm
* scikit-learn

Mamba dependencies will be added when the Mamba experiment is introduced.

---

## Running the Project

Open:

```text
train.ipynb
```

and run the notebook cells sequentially.

The notebook currently:

1. Downloads the IMDB dataset
2. Preprocesses the reviews
3. Creates train/validation/test splits
4. Trains the tokenizer on the training set
5. Creates PyTorch datasets and dataloaders
6. Builds the LSTM classifier
7. Trains the model
8. Evaluates the model on the validation set
9. Saves the trained model and tokenizer

---

## Model Saving

The trained model parameters are saved using PyTorch's `state_dict`.

```python
torch.save(model.state_dict(), "lstm_imdb.pt")
```

The tokenizer is saved separately because the token-to-ID mapping must remain identical when loading the model.

---

## Future Work

Planned experiments:

* [ ] Add Mamba sequence classifier
* [ ] Train Mamba using the same pipeline
* [ ] Compare LSTM and Mamba validation/test performance
* [ ] Compare parameter counts
* [ ] Compare training time
* [ ] Compare inference time
* [ ] Investigate sequence length effects
* [ ] Analyze difficult examples and failure cases
* [ ] Add confusion matrices and additional evaluation metrics
* [ ] Experiment with different Mamba configurations
* [ ] Add experiment tracking

---

## Learning Objectives

This project is also designed as a practical study of:

* Sequence modeling
* Tokenization
* Embeddings
* LSTM
* State Space Models
* Mamba
* PyTorch training loops
* Dynamic batching and padding
* Model evaluation
* Fair architectural comparisons

---

## Status

**Current status: LSTM baseline completed.**

The next stage is implementing the Mamba classifier while preserving the existing data and training pipeline.

---

## License

This project is licensed under the MIT License.
