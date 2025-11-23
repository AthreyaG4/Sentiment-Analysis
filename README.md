# Sentiment Analysis with RNN

A deep learning project for multi-class emotion classification using Recurrent Neural Networks (RNN). This model classifies text into six distinct emotions: sadness, joy, love, anger, fear, and surprise.

## Project Overview

This project implements a sentiment analysis model using PyTorch and the `dair-ai/emotion` dataset. The model uses a simple RNN architecture with embeddings to classify text into emotional categories, achieving **92.65% validation accuracy** after just 5 epochs.

## Key Features

## Tokenization Strategy

### Custom BPE Tokenizer

This project uses a **Byte Pair Encoding (BPE)** tokenizer trained from scratch on the emotion dataset using the Hugging Face `tokenizers` library. This approach ensures the vocabulary is optimally tailored to our specific text domain, avoiding the overhead of large pretrained tokenizers.

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

### Gradient Clipping

Prevents exploding gradients with a maximum norm of 1.0:

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### Pack Padded Sequences

Efficiently handles variable-length sequences:

```python
packed = torch.nn.utils.rnn.pack_padded_sequence(
    embeddings, lengths, batch_first=True, enforce_sorted=False
)
```

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

## Architecture

### Model Variants

This project implements and compares four different recurrent neural network architectures:

#### 1. Simple RNN

- Basic recurrent architecture as baseline
- Faster training but limited long-term memory
- Good for understanding sequential patterns in short texts

#### 2. LSTM (Long Short-Term Memory)

- **Better at capturing long-term dependencies**
- Solves vanishing gradient problem with memory cells
- More parameters than simple RNN but better performance

#### 3. GRU (Gated Recurrent Unit)

- **Simpler than LSTM** with fewer parameters
- Faster training time while maintaining good performance
- Often performs similarly to LSTM with less computational cost

### Model Configuration

All models share the same base configuration:

- **Embedding dimension**: 256
- **Hidden size**: 128
- **Dropout**: 0.2 (applied to embeddings and hidden layers)
- **Output classes**: 6 emotions

### Usage

All architectures use the same training pipeline - simply swap the model class:

```python
# Choose your architecture
model = RNN(input_size, emb_size, hidden_size, output_size)
# OR
model = LSTM(input_size, emb_size, hidden_size, output_size)
# OR
model = GRU(input_size, emb_size, hidden_size, output_size)
```

See `sentiment_lstm.ipynb` for complete implementation details of all architectures.

### Model Comparison

| Architecture | Parameters  | Training Speed | Best For                   |
| ------------ | ----------- | -------------- | -------------------------- |
| RNN          | Lowest      | Fastest        | Baseline, short sequences  |
| GRU          | Medium      | Fast           | Balanced performance/speed |
| LSTM         | Medium-High | Medium         | Long-term dependencies     |

## Model Performance

To compare different recurrent architectures for emotion classification, three models were trained under the same data and training settings:

- **Vanilla RNN**
- **GRU**
- **LSTM**

All models showed consistent improvement across epochs, but the **GRU and LSTM significantly outperformed the vanilla RNN**, both in final accuracy and stability during training.

Key observations:

- The **RNN** steadily improved and reached a peak validation accuracy of **80.45%**.
- The **GRU** converged rapidly and achieved the **highest validation accuracy** of **92.65%**, making it the best-performing model.
- The **LSTM** also performed strongly, reaching **92.55%**, just slightly below the GRU.

### Best Model Summary

| Model   | Best Validation Accuracy | Epoch |
| ------- | ------------------------ | ----- |
| RNN     | 80.45%                   | 5     |
| **GRU** | **92.65%**               | **3** |
| LSTM    | 92.55%                   | 5     |

> **Best Model:** **GRU** — It achieved the highest validation accuracy and demonstrated the fastest convergence among the three architectures.

Both GRU and LSTM are strong choices for this sentiment/emotion classification task, but based on these results, the **GRU** provides the best balance of performance and efficiency.

### Training Metrics Visualization (LSTM)

![Training and Validation Loss](/training_validation_metrics.png)
_Training and validation metric curves showing model convergence over 5 epochs_

## Sample Predictions (LSTM)

| Input Text                                                                                             | True Label | Predicted |
| ------------------------------------------------------------------------------------------------------ | ---------- | --------- |
| "im sure ill feel more playful soon but i just cant right now"                                         | Joy        | Joy       |
| "i feel happy about the outcome of this long election and im glad its over"                            | Joy        | Joy       |
| "i feel like there is no way out being humiliated by asa a guy i was obssessed about..."               | Sadness    | Sadness   |
| "i first started using this i did not like it because i felt like it made my hair feel very dirty..."  | Sadness    | Sadness   |
| "i am going through trials or just feeling troubled about something i love to put on worship music..." | Sadness    | Sadness   |

**Model demonstrates strong performance across different emotion classes with accurate predictions on real test examples.**

## Dependencies

```python
torch>=2.0.0
tokenizers>=0.13.0
datasets>=2.0.0
matplotlib>=3.5.0
```

---

**Built with ❤️ using PyTorch**
