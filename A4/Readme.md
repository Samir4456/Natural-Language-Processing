# A4 — Do You AGREE?

**Natural Language Understanding (NLU) — BERT → SBERT → Web App**

---

##  Overview

This project implements an **end-to-end semantic understanding pipeline**:

1. **Train BERT from scratch** using Masked Language Modeling (MLM) on a subset of Wikipedia.
2. **Convert BERT to Sentence-BERT (SBERT)** with a Siamese architecture for Natural Language Inference (NLI).
3. **Evaluate performance** using classification metrics on the SNLI dataset.
4. **Deploy a web application** that predicts semantic relationships between two sentences.

This repository satisfies all required deliverables:

* Jupyter notebook implementation
* Experimental analysis and loss curves
* Trained model checkpoints
* Web application (`app/`)
* Documentation (this README)

---

##  Architecture Pipeline

```
Wikipedia Corpus
      ↓
Masked Language Modeling (MLM)
      ↓
BERT Encoder (trained from scratch)
      ↓
Siamese SBERT + Softmax Loss (SNLI)
      ↓
NLI Prediction Web App
```

---

## Project Structure

```
A4/
│
├── notebook.ipynb              # Full step-by-step implementation
│
├── artifacts/
│   └── corpus_100k.txt         # Wikipedia subset used for MLM
│
├── tokenizer/
│   └── vocab.txt               # WordPiece tokenizer trained from corpus
│
├── models/
│   ├── bert_scratch_mlm.pt
│   ├── bert_scratch_mlm_meta.json
│   ├── sbert_softmax_snli.pt
│   └── sbert_softmax_snli_meta.json
│
├── app/
│   ├── app.py                  # Flask server
│   ├── model_def.py            # SBERT loader + architecture
│   └── templates/
│       └── index.html          # Web UI
│
└── README.md
```

---

## Task 1 — Train BERT from Scratch

### Dataset

* **Wikipedia English subset (~100k samples)**
* Publicly available and properly cited.

### Method

* Custom **WordPiece tokenizer**
* Transformer **encoder-only architecture**
* **Masked Language Modeling (15% masking rule)**
* Optimizer: **AdamW + linear warmup + decay**

### Output

* Trained **BERT encoder weights**
* Saved for downstream SBERT training.

---

## Task 2 — Sentence-BERT for NLI

### Dataset

* **SNLI (Stanford Natural Language Inference)**

### Model

* **Siamese shared BERT encoder**
* **Mean pooling** → sentence embeddings
* Classification using:

```
(u, v, |u − v|) → Linear → Softmax
```

### Training

* Fine-tuning with **cross-entropy loss**
* Epoch-level logging of:

  * Train/validation loss
  * Train/validation accuracy

---

## Task 3 — Evaluation & Analysis

### Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### Observations

* Training from scratch with limited data results in:

  * Lower entailment recall
  * Class imbalance sensitivity
* Performance improves with:

  * Larger MLM corpus
  * More epochs
  * Larger hidden size / layers

---

##  Task 4 — Web Application

### Features

* Two input boxes:

  * **Premise**
  * **Hypothesis**
* Predicts:

  * Entailment
  * Neutral
  * Contradiction
* Displays **class probabilities**.

### Run Locally

```bash
cd A4
python app/app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

##  Training Curves

The notebook includes:

* **MLM loss vs epoch**
* **SBERT train/validation loss**
* **SBERT train/validation accuracy**

These plots demonstrate:

* Stable convergence from warmup scheduling
* Improved semantic classification after fine-tuning

---

##  Key Learning Outcomes

This assignment demonstrates:

* Transformer encoder implementation **from scratch**
* Masked language modeling mechanics
* Sentence embedding via **Siamese networks**
* Natural Language Inference classification
* End-to-end **ML → evaluation → deployment** pipeline

---

##  References

* BERT: *Bidirectional Encoder Representations from Transformers*
* Sentence-BERT: *Sentence Embeddings using Siamese BERT-Networks*
* Wikipedia dataset
* SNLI dataset

(All datasets are publicly available and cited in the notebook.)

---

##  Author

**Samir Pokharel**
Artificial Intelligence / Natural Language Processing Coursework

---


**End of README**
