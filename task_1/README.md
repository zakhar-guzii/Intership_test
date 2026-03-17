# MNIST Digit Classifier

Three classification models for handwritten digit recognition using the MNIST dataset, unified under a single interface.

---

## Models

| Key | Class | Algorithm |
|-----|-------|-----------|
| `rf` | `RandomForestModel` | Random Forest (scikit-learn) |
| `nn` | `FeedForwardNN` | Feed-Forward Neural Network (PyTorch) |
| `cnn` | `ConvolutionNN` | Convolutional Neural Network (PyTorch) |

---

## Project Structure

```
task1/
├── src/
│   ├── interface.py            # MnistClassifierInterface (ABC)
│   ├── mnist_classifier.py     # Unified classifier entry point
│   ├── random_forest_model.py
│   ├── ff_nn_model.py
│   └── cnn_model.py
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
└── README.md
```

---

## Setup

Python **3.10+** is required.

**1. Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

**2. Install all dependencies:**

```bash
pip install -r requirements.txt
```

> **VSCode users:** after activating the venv, select it as the Python interpreter via `Cmd+Shift+P` → `Python: Select Interpreter` → choose `.venv`. For Jupyter notebooks, select the `.venv` kernel in the top-right kernel picker.

---

## Usage

```python
from src.mnist_classifier import MnistClassifier

classifier = MnistClassifier(algorithm='cnn')  # 'rf', 'nn', or 'cnn'
classifier.train(X_train, y_train)
predictions = classifier.predict(X_test)
```

All three models accept `X` as a NumPy array of shape `(N, 784)` or `(N, 28, 28)` with pixel values in `[0, 255]`. Normalization is handled internally.

---

## Demo

The notebook at `notebooks/demo.ipynb` covers:

- Dataset loading and visualization
- Training and evaluation of all three models
- Accuracy comparison across models
- Edge cases: invalid algorithm name, single-image prediction, misclassified samples

To run it:

```bash
cd notebooks
jupyter notebook demo.ipynb
```

---

## Results

Evaluated on 14,000 test samples (80/20 split, `random_state=42`):

| Model | Test Accuracy |
|-------|--------------|
| Random Forest | 96.71% |
| Feed-Forward NN | 97.80% |
| CNN | 99.17% |