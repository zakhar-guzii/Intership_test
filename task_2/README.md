# Task 2 — Named Entity Recognition + Image Classification

## Overview

A multimodal machine learning pipeline that combines **Natural Language Processing (NLP)** and **Computer Vision (CV)**. The pipeline accepts a text message and an image, identifies which animal is referenced in the text, and verifies whether the image contains that animal — returning a single boolean value.

---

## Models

| Key | Domain | Architecture | Description |
|-----|--------|--------------|-------------|
| `ner` | NLP | `dslim/bert-base-NER` | Fine-tuned Hugging Face transformer for extracting `ANIMAL` entities from free-form text. |
| `classifier` | CV | `ResNet-18` | Fine-tuned PyTorch CNN for 10-class animal classification. Class imbalance is handled via weighted cross-entropy loss. |

**Supported animal classes:** `butterfly` · `cat` · `chicken` · `cow` · `dog` · `elephant` · `horse` · `sheep` · `spider` · `squirrel`

---

## Project Structure

```
task_2/
├── data/
│   └── Animals-10/             # Dataset directory (not included in repo)
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

## Usage

Run the full verification pipeline from the command line:

```bash
python src/pipeline.py \
    --text "There is a beautiful cow standing in the field." \
    --image_path "data/Animals-10/cow/sample.jpg" \
    --ner_model_path "models/ner_animal/best_model" \
    --cv_model_path "models/best_animal_classifier.pth"
```

**Output:**

```
=============================================
 Analyzing multimodal inputs...
=============================================
 NER Extraction : ['cow']
 CV Prediction  : Cow
=============================================
 FINAL PIPELINE OUTPUT: True
=============================================
```

---

## Deliverables

| File | Description |
|------|-------------|
| `notebooks/eda.ipynb` | Dataset exploration: class distribution, imbalance resolution via class weights, normalization statistics. |
| `notebooks/demo.ipynb` | Pipeline demo across standard and edge cases with visual outputs. |
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
Demonstrates the full pipeline across standard and edge cases — missing entity extraction, multiple animals mentioned in text, and capitalisation differences — with formatted visual outputs.

To run the demo:

```bash
cd notebooks
jupyter notebook demo.ipynb
```