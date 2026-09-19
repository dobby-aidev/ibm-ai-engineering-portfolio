# League of Legends Match Predictor (PyTorch)

Capstone project for the **Introduction to Neural Networks and PyTorch** course within the **IBM AI Engineering Professional Certificate** on Coursera.

- **Verified Certificate:** [Coursera Accomplishment (Verify Credential)](https://coursera.org/share/0e7b9460511d5b991632d1929a2ce985)
- **Frameworks:** Python, PyTorch (`torch.nn`, `torch.optim`), Pandas, NumPy, Scikit-Learn, Matplotlib

---

## Overview

A binary classification pipeline implemented from scratch in PyTorch to predict match outcomes (Win/Loss) from in-game performance statistics. The implementation focuses on proper tensor operations, custom `nn.Module` definition, L2 regularization via weight decay, ROC-AUC evaluation, and model interpretability through feature weight extraction.

---

## Dataset Features

Extracted match statistics from `league-of-legends-data-large.csv`:

- Performance metrics: `kills`, `deaths`, `assists`
- Economy & farming: `gold_earned`, `cs` (Creep Score)
- Vision & utility: `wards_placed`, `wards_killed`
- Combat: `damage_dealt`
- **Target:** `win` (0 = Loss, 1 = Win)

---

## Implementation Details

### 1. Data Pipeline
- Preprocessing with Scikit-Learn `StandardScaler` to normalize feature variance.
- 80/20 train/test stratified split.
- Features and labels converted into PyTorch `FloatTensor` objects.

### 2. Model Architecture
```python
import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
```

### 3. Optimization & Regularization
- **Loss Function:** Binary Cross-Entropy (`nn.BCELoss()`).
- **Optimizer:** `torch.optim.SGD` with momentum.
- **L2 Regularization:** Implemented via `weight_decay = 0.01` parameter in the optimizer to penalize large weights and prevent overfitting on collinear features.

### 4. Evaluation & Interpretability
- **Classification Metrics:** Precision, Recall, and F1-score computed on the held-out test set.
- **ROC Analysis:** Receiver Operating Characteristic (ROC) curve plotted and Area Under Curve (AUC) calculated.
- **Feature Attribution:** Learned weight matrix ($W$) extracted from `model.linear.weight` and plotted as horizontal bar charts, showing that `gold_earned` and `kills` have the strongest positive coefficients, while `deaths` carries the strongest negative weight.
- **Model Checkpointing:** Trained model weights serialized to disk as `logistic_regression_model.pth`.

---

## Project Structure

```text
03_League_of_Legends_Match_Predictor/
├── Final Project League of Legends Match Predictor.ipynb  # Executed notebook
├── Final project.ipynb                                    # Submission copy
├── league-of-legends-data-large.csv                       # Match dataset
├── logistic_regression_model.pth                          # Saved model weights
└── README.md                                              # Project documentation
```

---

## Running the Code

1. Install requirements:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```
2. Launch notebook:
   ```bash
   jupyter notebook "Final Project League of Legends Match Predictor.ipynb"
   ```
