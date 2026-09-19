# Fashion-MNIST Image Classification with PyTorch CNN

Capstone project for the **Introduction to Neural Networks and PyTorch** course within the **IBM AI Engineering Professional Certificate** on Coursera.

- **Verified Certificate:** [Coursera Accomplishment (Verify Credential)](https://coursera.org/share/0e7b9460511d5b991632d1929a2ce985)
- **Frameworks:** Python, PyTorch (`torch.nn`, `torch.optim`, `torch.utils.data`), Torchvision (`transforms`, `datasets`), Matplotlib

---

## Overview

Image classification across 10 clothing and apparel categories from Zalando's **Fashion-MNIST** benchmark dataset (60,000 train, 10,000 test images). The model implements a 2-stage Convolutional Neural Network architecture enhanced with 2D Batch Normalization (`BatchNorm2d`) and evaluates the training progression with dual-axis cost vs. accuracy tracking.

---

## Architecture (`CNN_batch`)

Inputs are resized to $16 \times 16$ single-channel grayscale images via `torchvision.transforms.Resize((16, 16))` to match linear layer sizing constraints:

```text
Input (1, 16, 16)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Conv Block 1: Conv2d(1 -> 16, k=5, pad=2)              │
│ ├── BatchNorm2d(16)                                    │
│ ├── ReLU()                                             │
│ └── MaxPool2d(kernel_size=2) -> output (16, 8, 8)      │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Conv Block 2: Conv2d(16 -> 32, k=5, pad=2)             │
│ ├── BatchNorm2d(32)                                    │
│ ├── ReLU()                                             │
│ └── MaxPool2d(kernel_size=2) -> output (32, 4, 4)      │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Classification Head                                    │
│ ├── Flatten -> (512 features)                          │
│ ├── Linear(32 * 4 * 4, 10)                             │
│ └── BatchNorm1d(10)                                    │
└────────────────────────────────────────────────────────┘
       │
       ▼
Output: 10 Class Logits
```

---

## Training Setup

- **Batch Size:** 100
- **Epochs:** 5
- **Optimizer:** `torch.optim.SGD(model.parameters(), lr=0.1)`
- **Criterion:** `nn.CrossEntropyLoss()`
- **Hardware:** CPU / CUDA compatible DataLoader with multi-worker support.

---

## Results

- **Validation Accuracy:** **~89%** after 5 epochs.
- **Convergence Behavior:** Batch normalization stabilized the internal covariate shift, allowing a relatively high learning rate (`0.1`) with SGD without divergence.
- Dual-axis plotting displays monotonic cost reduction alongside validation accuracy gains across training iterations.

---

## Project Structure

```text
04_Fashion_MNIST_Classification_CNN/
├── FashionMNISTProject.ipynb   # Executed project notebook with all 15 cells verified
├── Final project.ipynb          # Submission copy
├── README.md                    # Project documentation
└── .fashion/                    # Cached Fashion-MNIST dataset files
```

---

## Running the Code

1. Install requirements:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install matplotlib pillow jupyter
   ```
2. Launch notebook:
   ```bash
   jupyter notebook FashionMNISTProject.ipynb
   ```
