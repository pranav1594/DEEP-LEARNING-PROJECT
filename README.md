# DEEP-LEARNING-PROJECT

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : SAI PRANAV BHOGARAJU

*INTERN ID* : CTIS7339

*DOMAIN* : DATA SCIENCE

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

## Task 2 — Deep Learning Project
## Fake News Detector via DistilBERT Fine-Tuning
**CodTech IT Solutions | Data Science Internship**

---

## Overview

Fine-tune a **DistilBERT** transformer (66M parameters, pretrained on BookCorpus + Wikipedia) on the **LIAR dataset** to classify political news statements as **Fake (0)** or **Real (1)**.

The project demonstrates the full NLP deep learning workflow — from raw text ingestion and subword tokenization through transformer fine-tuning to multi-metric evaluation and interpretability visualisations. A TF-IDF + Logistic Regression baseline is trained alongside BERT for rigorous comparison.

---

## Dataset

| Property | Value |
|---|---|
| Name | LIAR (Wang, ACL 2017) |
| Source | [HuggingFace Hub — `liar`](https://huggingface.co/datasets/liar) |
| Size | 12,836 labelled political statements |
| Original labels | 6 classes: pants-fire / false / barely-true / half-true / mostly-true / true |
| Binarised | **FAKE (0)**: pants-fire, false, barely-true — **REAL (1)**: half-true, mostly-true, true |
| Splits | Train 10,269 / Val 1,284 / Test 1,283 |

> **Note:** The script auto-generates synthetic LIAR-style data if the Hub is unreachable, so all visualisations can be reviewed in demo mode without any downloads.

---

## Model Architecture

```
Input statement (raw text)
        │
        ▼
[DistilBERT Tokenizer]
  WordPiece subword tokenisation
  → [CLS] token₁ token₂ … token_n [SEP]
  → input_ids, attention_mask  (padded/truncated to max_len=128)
        │
        ▼
[DistilBERT Base Uncased]
  6 transformer layers × 12 attention heads
  Hidden size 768  |  66M parameters
  Pretrained on BookCorpus + English Wikipedia
        │
        ▼
[CLS] token hidden state  (768-dim)
        │
        ▼
[Linear classifier head]
  768 → 2  (logits for Fake / Real)
        │
        ▼
Softmax → confidence scores
[Fake: 0.23,  Real: 0.77] → predicted class: REAL
```

---

## Training Details

| Hyperparameter | Value | Rationale |
|---|---|---|
| Model | `distilbert-base-uncased` | 40% faster than BERT-base, 97% of performance |
| Epochs | 3 | Sufficient for classification fine-tuning |
| Batch size | 16 | Fits on 8GB GPU |
| Peak LR | 2e-5 | Standard for transformer fine-tuning |
| Warmup | 10% of total steps | Prevents early training instability |
| Optimiser | AdamW | Weight decay on non-bias params |
| Scheduler | Linear warmup + decay | Standard for transformer training |
| Max seq length | 128 | Covers 95%+ of LIAR statements |
| Early stopping | Patience = 2 (by val F1) | Prevents overfitting |

---

## How to Run

### Demo mode (no GPU required — generates all visualisations)
```bash
pip install -r requirements.txt
python fake_news_bert.py --mode demo
```

### Full BERT training (GPU strongly recommended)
```bash
# GPU required for reasonable training time
# Install all dependencies including torch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

python fake_news_bert.py --mode train --epochs 3 --batch_size 16 --lr 2e-5
```

### Custom arguments
```
--mode       train | demo         (default: demo)
--epochs     int                  (default: 3)
--batch_size int                  (default: 16)
--max_len    int                  (default: 128)
--lr         float                (default: 2e-5)
--output_dir path                 (default: ./output_t2)
```

---

## Output Files

| File | Description |
|---|---|
| `00_bert_architecture.png` | End-to-end pipeline architecture diagram |
| `01_class_distribution.png` | Label distribution across train / val / test splits |
| `02_training_curves.png` | Loss, accuracy, and F1 per epoch |
| `03_confusion_matrices.png` | Normalised confusion matrices: Baseline vs BERT |
| `04_roc_curves.png` | ROC curves with AUC scores for both models |
| `05_metrics_comparison.png` | Grouped bar chart: accuracy / precision / recall / F1 |
| `06_token_importance.png` | Top predictive words for Fake vs Real (interpretability) |
| `classification_report.json` | Full sklearn classification report (both models) |
| `best_model/` | Saved HuggingFace model checkpoint (train mode only) |

---

## Expected Results

| Metric | TF-IDF + LR Baseline | DistilBERT |
|---|---|---|
| Accuracy | ~61% | ~68–71% |
| F1 Score | ~0.60 | ~0.68–0.70 |
| AUC-ROC | ~0.65 | ~0.74 |

> BERT meaningfully outperforms the bag-of-words baseline by capturing contextual semantics, negation, and subtle linguistic cues that TF-IDF misses. Real-world results vary by seed and hardware.

---

## Key Concepts Demonstrated

- **Transformer fine-tuning** — adapting pretrained BERT weights to a domain-specific task
- **Subword tokenisation** — WordPiece handles OOV words, misspellings, and compound terms
- **Custom PyTorch Dataset & DataLoader** — efficient batched loading with padding
- **AdamW + linear warmup** — best-practice optimisation for transformer fine-tuning
- **Baseline comparison** — TF-IDF + LR gives a rigorous lower bound for BERT to beat
- **Confusion matrix + ROC AUC** — beyond accuracy for imbalanced evaluation
- **Token importance / interpretability** — understanding *why* the model predicts fake vs real

---

## Requirements

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
accelerate>=0.24.0
```

---

## References

- Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. *ACL 2017*.
- Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. *NeurIPS EMC² Workshop*.
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL 2019*.
