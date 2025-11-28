# Sentiment Analysis with RNN, GRU, LSTM, and BERT

A comprehensive deep learning project that compares multiple neural architectures, **RNN**, **GRU**, **LSTM**, and **BERT**, for multi-class emotion classification across six emotion categories.

---

## Project Overview

This project investigates how classical recurrent neural networks and modern transformer-based architectures perform on the **dair-ai/emotion** dataset. The task is to classify text into **six emotions**: _sadness, joy, love, anger, fear,_ and _surprise_.

To do this, four different architectures are implemented, trained, and compared under identical training settings:

- **Simple RNN**
- **GRU (Gated Recurrent Unit)**
- **LSTM (Long Short-Term Memory)**
- **BERT-base** fine-tuned for classification

The project includes a custom tokenizer, class imbalance handling, metric visualizations, and a detailed comparison of model performance.

---

## Key Features

### Custom BPE Tokenizer

The **Byte Pair Encoding (BPE)** tokenizer is used **only for the recurrent models (RNN, GRU, LSTM)**. BERT does **not** use this tokenizer, it relies on its own pretrained WordPiece tokenizer from Hugging Face.

This custom tokenizer is trained from scratch on the emotion dataset using the Hugging Face `tokenizers` library. It ensures the vocabulary is optimally tailored to the text domain of emotional social media content, avoiding the overhead of large pretrained tokenizers.

#### Why BPE?

- **Subword tokenization**: Handles rare words and out-of-vocabulary terms effectively
- **Compact vocabulary**: Balances vocabulary size with representation quality
- **Custom-built**: Specifically optimized for emotion-laden social media text

#### Tokenizer Pipeline

Our tokenization process consists of three main stages:

**1. Normalization**

```python
normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
```

- **NFD**: Unicode normalization for consistent character representation
- **StripAccents**: Removes accent marks (é → e, ñ → n)
- **Lowercase**: Converts all text to lowercase for uniformity

**2. Pre-tokenization**

```python
pre_tokenizer = Whitespace()
```

- Splits text on whitespace boundaries before applying BPE merges
- Preserves word-level structure while enabling subword tokenization

**3. BPE Training**

```python
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.enable_padding()
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]"])
tokenizer.train_from_iterator(train_data['text'], trainer)
```

#### Special Tokens

| Token   | Purpose                                          |
| ------- | ------------------------------------------------ |
| `[UNK]` | Represents unknown/out-of-vocabulary tokens      |
| `[PAD]` | Pads shorter sequences to match batch dimensions |

This custom tokenizer creates a vocabulary optimized for emotion classification, capturing the nuances of emotional expression in text.

### Class Weighting

The model uses inverse frequency weighting to handle class imbalance:

```python
class_weights = torch.tensor([
    total / (num_classes * count_per_class)
    for class in range(num_classes)
])
```

### BERT Fine-Tuning

The BERT-based classifier is built using **google-bert/bert-base-uncased** and fine-tuned end-to-end for 6-class emotion classification.  
This pipeline uses the Hugging Face `AutoModelForSequenceClassification`, `Trainer`, and `TrainingArguments` APIs to streamline the full training workflow.

#### Key Components of the Fine-Tuning Pipeline

- **Pretrained Model Initialization**
  ```python
  bert_model = AutoModelForSequenceClassification.from_pretrained(
      "google-bert/bert-base-uncased",
      num_labels=6
  )
  ```
  The classifier head is automatically added on top of BERT, configured for six emotion labels.
- **BERT Tokenization**
  ```python
  bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
  ```
  BERT uses its own WordPiece tokenizer, distinct from the custom BPE tokenizer used by the recurrent models.
  Tokenization includes:
  - **Truncation**
  - Dynamic padding via **DataCollatorWithPadding**
  - Automatic **attention mask** generation
- **Metrics Callback**
  This custom callback tracks and stores metrics every epoch.

This approach yields the strongest performance across all metrics.

### Handling Class Imbalance

Because the dataset is heavily skewed (e.g., _joy_ and _sadness_ dominate), the project evaluates model performance using **weighted metrics** to ensure minority classes such as _love_ and _surprise_ are fairly represented.

Two different metric pipelines are used:

- **Recurrent Models (RNN, GRU, LSTM) :**  
  Use **scikit-learn** (`sklearn.metrics`) to compute:

  - Weighted F1-score
  - Weighted Precision
  - Weighted Recall

- **BERT :**  
  Uses the Hugging Face **evaluate** library with `average="weighted"` to compute:
  - Weighted F1-score
  - Weighted Precision
  - Weighted Recall

This dual approach ensures consistent and balanced metric reporting across all model architectures.

### Stable RNN Training (Packing + Clipping)

- **Gradient clipping** prevents exploding gradients

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

- **Packed padded sequences** optimize recurrent computation and avoid wasted operations on padded tokens

```python
packed = torch.nn.utils.rnn.pack_padded_sequence(
    embeddings, lengths, batch_first=True, enforce_sorted=False
)
```

---

## Emotion Categories

### Class Distribution

![Class Distribution](/class_distribution_combined.png)
_Distribution of emotion classes in the training dataset_

| Class | Emotion  | Distribution (Train) | Distribution (Validation) | Percentage |
| ----- | -------- | -------------------- | ------------------------- | ---------- |
| 0     | Sadness  | 4,666 samples        | 550                       | 29.2%      |
| 1     | Joy      | 5,362 samples        | 704                       | 33.5%      |
| 2     | Love     | 1,304 samples        | 178                       | 8.2%       |
| 3     | Anger    | 2,159 samples        | 275                       | 13.5%      |
| 4     | Fear     | 1,937 samples        | 212                       | 12.1%      |
| 5     | Surprise | 572 samples          | 81                        | 3.6%       |

---

## Model Architectures

### 1. Simple RNN

- Fast baseline
- Limited long-term dependency modeling
- Performs reasonably well on short messages

### 2. GRU

- Gating mechanism simplifies computation
- Fewer parameters than LSTM
- Converges quickly and efficiently

### 3. LSTM

- Strong ability to track long-range relationships
- Uses memory cells + multiple gates
- Reliable and stable performance

### 4. BERT

- Transformer-based bidirectional encoder
- Pretrained on massive corpora
- Context-aware representation of words
- Outperforms all recurrent models in this project

---

## Model Configuration

### Recurrent Models

- **Embedding Dimension**: 256
- **Hidden Size**: 128
- **Dropout**: 0.2
- **Output Classes**: 6
- **Loss**: Weighted cross-entropy
- **Optimizer**: Adam

### BERT Model

- **Model**: `bert-base-uncased`
- **Learning Rate**: ~2e-5
- **Batch Size**: 8
- **Training Epochs**: 3

---

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

Open the main notebook to train or evaluate all models:

```bash
jupyter notebook sentiment_analysis.ipynb
```

Inside the notebook you can select any of the four architectures.

---

## Model Performance

To compare the four architectures, RNN, GRU, LSTM, and BERT, each model was trained on the same dataset and evaluated using weighted accuracy, F1-score, precision, and recall. All models showed consistent improvement across epochs, but the transformer-based **BERT model clearly outperformed the recurrent models** in just 3 epochs, achieving the highest overall accuracy and F1.

### Training Metrics Visualization

![Training and Validation Metrics](/model_comparison.png)  
_Training and validation performance curves for all four models across 5 epochs (3 for BERT)._

### Key Observations

#### RNN

The RNN model improved steadily over 5 epochs, starting weak but reaching a solid final performance:

- Final Validation Accuracy: **82.35%**
- Final Weighted F1: **82.42%**
- Shows smooth learning progression but struggles with complex emotional patterns.

#### GRU

GRU delivered strong performance with fast convergence:

- Reached **81.80%** accuracy in **Epoch 1**
- Peaked at **92.70% accuracy** and **92.82% weighted F1** in Epoch 5
- Stable and highly efficient across all epochs

#### LSTM

LSTM also performed strongly:

- Reached **88.75%** accuracy in just **Epoch 2**
- Achieved a final accuracy of **92.05%** and weighted F1 of **92.13%**
- Slightly below GRU but still highly competitive

#### BERT

BERT achieved the highest performance overall and required only **3 epochs**:

- Best Accuracy: **94.45%**
- Best Weighted F1: **94.44%**
- Excellent precision and recall across all classes, including minority labels
- Demonstrates the benefit of pretrained contextual representations

---

### Best Model Summary

| Model    | Best Validation Accuracy | Best Weighted F1 | Epoch |
| -------- | ------------------------ | ---------------- | ----- |
| RNN      | 82.35%                   | 82.42%           | 5     |
| GRU      | 92.70%                   | 92.82%           | 5     |
| LSTM     | 92.05%                   | 92.13%           | 5     |
| **BERT** | **94.45%**               | **94.44%**       | **3** |

> **Best Model:** **BERT** — It achieved the highest accuracy, F1-score, precision, and recall while converging in just **three epochs**.

While **GRU and LSTM** remain excellent lightweight alternatives with strong performance, **BERT consistently delivers superior results**, especially in nuanced emotional classification tasks.

---

**Built with ❤️ using PyTorch**
