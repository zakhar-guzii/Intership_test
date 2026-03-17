# Task 2 — Named Entity Recognition + Image Classification

## Overview

A multimodal machine learning pipeline that combines **Natural Language Processing (NLP)** and **Computer Vision (CV)**. The pipeline accepts a text message and an image, identifies which animal is referenced in the text, and verifies whether the image contains that animal — returning a single boolean value.

---

## Models

| Key | Domain | Architecture | Description |
|-----|--------|--------------|-------------|
| `ner` | NLP | `dslim/bert-base-NER` | Fine-tuned Hugging Face transformer for extracting `ANIMAL` entities from free-form text. |
| `classifier` | CV | `ResNet-18` | Fine-tuned PyTorch CNN for 10-class animal classification. Class imbalance is handled via weighted cross-entropy loss. A `Dropout(0.3)` layer is applied before the final classifier head. |

**Supported animal classes:** `butterfly` · `cat` · `chicken` · `cow` · `dog` · `elephant` · `horse` · `sheep` · `spider` · `squirrel`

---

## Project Structure

```
task_2/
├── data/
│   └── raw-img/                # Dataset directory (not included in repo)
├── models/                     # Saved model weights (.pth and HF format)
├── notebooks/
│   ├── demo.ipynb              # Pipeline visualization and edge cases
│   └── eda.ipynb               # Exploratory Data Analysis & class weight calculations
├── src/
│   ├── classifier/
│   │   ├── inference.py        # CV model prediction script
│   │   └── train.py            # CV model training with parameterized CLI
│   ├── ner/
│   │   ├── inference.py        # NER entity extraction script
│   │   └── train.py            # Transformer fine-tuning script
│   ├── generate_data.py        # Synthetic NER dataset generator
│   └── pipeline.py             # Unified multimodal execution script
├── requirements.txt
└── README.md
```

---

## Setup

Python **3.10+** is required. Install all dependencies:

```bash
pip install -r requirements.txt
```

> For optimal training performance, install the PyTorch build that matches your CUDA/GPU environment from [pytorch.org](https://pytorch.org).

---

## Data Preparation

### Image Dataset

Download the Animals-10 dataset from [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10):

```bash
kaggle datasets download -d alessiocorrado99/animals10 -p data/ --unzip
```

The dataset folders are named in Italian. Rename them to English before training:

```python
import os

mapping = {
    "cane": "dog", "gatto": "cat", "mucca": "cow",
    "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken",
    "cavallo": "horse", "pecora": "sheep", "ragno": "spider", "scoiattolo": "squirrel",
}

base = "data/raw-img"
for italian, english in mapping.items():
    src, dst = os.path.join(base, italian), os.path.join(base, english)
    if os.path.exists(src):
        os.rename(src, dst)
```

### NER Dataset

Generate the synthetic NER training dataset:

```bash
python src/generate_data.py
```

This produces `data/ner_dataset.json` with 1600 sentences (1200 positive / 400 negative), balanced across all 10 animal classes.

---

## Training

### Image Classifier (ResNet-18)

```bash
python src/classifier/train.py \
    --data_dir data/raw-img \
    --model_dir models/ \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --patience 5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/Animals-10` | Path to dataset root. |
| `--epochs` | `20` | Maximum training epochs. |
| `--batch_size` | `32` | Batch size. |
| `--learning_rate` | `1e-3` | Initial learning rate (cosine annealing schedule applied). |
| `--val_split` | `0.2` | Fraction of data held out for validation. |
| `--patience` | `5` | Early-stopping patience. |
| `--freeze_backbone` | `False` | Freeze ResNet backbone; train classification head only. |
| `--seed` | `42` | Random seed for reproducibility. |

The checkpoint saved to `models/best_animal_classifier.pth` contains `model_state_dict`, `class_names`, `val_acc`, `epoch`, and `args`.

### NER Model (BERT)

```bash
python src/ner/train.py \
    --data_path data/ner_dataset.json \
    --output_path models/ner_animal \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --patience 3
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `dslim/bert-base-NER` | Pretrained HuggingFace model. |
| `--data_path` | — | Path to training dataset (JSON lines). |
| `--val_path` | `None` | Separate validation set. If omitted, 10% of training data is used. |
| `--epochs` | `5` | Maximum training epochs. |
| `--patience` | `3` | Early-stopping patience (monitored metric: F1). |
| `--label_all_subwords` | `False` | Propagate label to all subword tokens (default: first subword only). |
| `--seed` | `42` | Random seed for reproducibility. |

The best model (by validation F1) is saved to `models/ner_animal/best_model/` along with the tokenizer.

---

## Inference

### Image Classifier

```bash
python src/classifier/inference.py \
    --model_path models/best_animal_classifier.pth \
    --image_path data/raw-img/cow/OIP---lAIbDlHKmejDpqrXq6vAAAAA.jpeg \
    --top_k 3
```

### NER Model

```bash
python src/ner/inference.py \
    --model_path models/ner_animal/best_model \
    --text "There is a large elephant near the river." \
    --min_confidence 0.85
```

Both scripts support `--device -1` (CPU) or `--device 0` (GPU) and `--log_level DEBUG/INFO/WARNING/ERROR`.

---

## Pipeline

> **Note:** Run all pipeline and inference commands from the `task_2/` root with `PYTHONPATH=.` so that `src` is resolvable as a package.

```bash
PYTHONPATH=. python src/pipeline.py \
    --text "There is a beautiful cow standing in the field." \
    --image_path "data/raw-img/cow/OIP---lAIbDlHKmejDpqrXq6vAAAAA.jpeg" \
    --ner_model_path "models/ner_animal/best_model" \
    --cv_model_path "models/best_animal_classifier.pth"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--text` | — | Free-form input text. |
| `--image_path` | — | Path to the image file. |
| `--ner_model_path` | — | Path to the NER model directory. |
| `--cv_model_path` | — | Path to the CV checkpoint (`.pth`). |
| `--min_confidence` | `0.0` | Minimum NER entity confidence to include. |
| `--device` | auto | `cpu` or `cuda`. Auto-detected if omitted. |

**Output:**

```
=============================================
  Text  : There is a beautiful cow standing in the field.
  Image : data/raw-img/cow/OIP---lAIbDlHKmejDpqrXq6vAAAAA.jpeg
=============================================
  RESULT: True
=============================================
```

---

## Deliverables

| File | Description |
|------|-------------|
| `notebooks/eda.ipynb` | Dataset exploration: class distribution, imbalance resolution via class weights, normalization statistics. |
| `notebooks/demo.ipynb` | Pipeline demo across standard and edge cases with visual outputs. |
| `src/generate_data.py` | Synthetic NER dataset generator (1600 sentences, 8 sentence styles, balanced per class). |
| `src/ner/train.py` | Parameterized training script for the transformer-based NER model. |
| `src/ner/inference.py` | Inference script for extracting animal entities from input text. |
| `src/classifier/train.py` | Parameterized training script for the ResNet-18 image classifier. |
| `src/classifier/inference.py` | Inference script for predicting the animal class from an image. |
| `src/pipeline.py` | Unified CLI script combining both models; outputs a single boolean value. |

---

## Notebooks

### `eda.ipynb`
Covers dataset exploration: class distribution analysis, resolution of the 3.4× class imbalance through weighted loss calculation, and visualization of per-channel normalization statistics.

### `demo.ipynb`
Demonstrates the full pipeline across standard and edge cases — missing entity extraction, multiple animals mentioned in text, capitalisation differences, indirect references, and NER confidence thresholding — with side-by-side image and confidence bar chart outputs.

To run the demo:

```bash
cd task_2/notebooks
PYTHONPATH=.. jupyter notebook demo.ipynb
```