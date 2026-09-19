<div align="center">

# IBM AI Engineering Professional Certificate
### Capstone & Applied Machine Learning Projects Portfolio

[![Architect](https://img.shields.io/badge/Author-Dobby%20B%20%28%40dobby--aidev%29-1D70B8?style=flat-square&logo=github)](https://github.com/dobby-aidev)
[![Organization](https://img.shields.io/badge/DONA%20Codex-Ecosystem-6f42c1?style=flat-square)](https://github.com/dobby-aidev)
[![Program](https://img.shields.io/badge/IBM-AI%20Engineering%20Specialization-blue?style=flat-square&logo=ibm)](https://www.coursera.org/professional-certificates/ai-engineer)
[![License: MIT](https://img.shields.io/badge/License-MIT-success?style=flat-square)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

<p align="center">
  A repository of verified final capstone projects, model architectures, and training pipelines developed for the <strong>IBM AI Engineering Professional Certificate</strong> on Coursera.
</p>

</div>

---

## 📊 Program Progress & Status

- **Program:** [IBM AI Engineering Professional Certificate (Coursera)](https://www.coursera.org/professional-certificates/ai-engineer)
- **Specialization Length:** 13 Courses
- **Current Completion:** **4 / 13 Completed (31%)**
- **Verified Credentials:** 4 Verified Course Certifications

```text
Progress: [████████░░░░░░░░░░░░░░░░░] 31% Complete (4/13)
```

---

## 📁 Completed Capstone Projects

| # | Project Name | Course & Domain | Framework / Architecture | Evaluation Metrics | Verified Credential |
|---|---|---|---|---|---|
| **01** | [**Rainfall Prediction in Australia**](./01_Rainfall_Prediction_Australia_Weather_ML) | **Machine Learning with Python**<br>*(Supervised Learning)* | `Scikit-Learn` Pipeline<br>• Random Forest (GridSearchCV)<br>• Logistic Regression (L1/L2) | **84% Test Accuracy**<br>Macro F1: 0.76<br>Feature attribution extracted | [🏅 Verify Certificate](https://coursera.org/share/deb792b19739e807ee06ccc96fded4c3) |
| **02** | [**Waste Classification via Transfer Learning**](./02_Waste_Classification_Transfer_Learning) | **Deep Learning with Keras & TensorFlow**<br>*(Computer Vision)* | `TensorFlow / Keras`<br>• VGG-16 ImageNet backbone<br>• Custom dense head + Dropout<br>• Fine-tuning upper conv blocks | **93% Val Accuracy**<br>(Binary: Organic vs Recyclable)<br>Loss curve convergence verified | [🏅 Verify Certificate](https://coursera.org/share/979d51e8d50d0ea1e178653b02e6edf0) |
| **03** | [**League of Legends Match Predictor**](./03_League_of_Legends_Match_Predictor) | **Neural Networks & PyTorch**<br>*(Tabular Deep Learning)* | `PyTorch`<br>• Custom `nn.Module` linear classifier<br>• L2 regularization (`weight_decay=0.01`)<br>• BCE Loss + SGD with momentum | **ROC-AUC Analysis**<br>Model weights exported (`.pth`)<br>Feature coefficient analysis | [🏅 Verify Certificate](https://coursera.org/share/0e7b9460511d5b991632d1929a2ce985) |
| **04** | [**Fashion-MNIST Classification with CNN**](./04_Fashion_MNIST_Classification_CNN) | **Neural Networks & PyTorch**<br>*(Computer Vision)* | `PyTorch / Torchvision`<br>• 2-Stage ConvNet (`CNN_batch`)<br>• 2D Batch Normalization (`BatchNorm2d`)<br>• MaxPool2d + SGD (`lr=0.1`) | **~89% Val Accuracy**<br>10 apparel classes<br>Dual-axis cost/accuracy tracking | [🏅 Verify Certificate](https://coursera.org/share/0e7b9460511d5b991632d1929a2ce985) |

> *Note: Course "Introduction to Deep Learning & Neural Networks with Keras" has also been completed and verified:* [🏅 Verify Certificate](https://coursera.org/share/8918b086fe9455453232c20de3eaba44).

---

## 🗺️ 13-Course Specialization Roadmap

```text
├── [✓] Course 01: Machine Learning with Python (Project 01)
├── [✓] Course 02: Introduction to Deep Learning & Neural Networks with Keras
├── [✓] Course 03: Deep Learning with Keras and Tensorflow (Project 02)
├── [✓] Course 04: Introduction to Neural Networks and PyTorch (Projects 03 & 04)
├── [ ] Course 05: Scalable Machine Learning with Apache Spark
├── [ ] Course 06: Building Deep Learning Models with TensorFlow
├── [ ] Course 07: Computer Vision and Image Processing Fundamentals
├── [ ] Course 08: Natural Language Processing & Large Language Models
├── [ ] Course 09: Generative AI, Prompt Engineering & RAG Applications
├── [ ] Course 10: Building Multi-Agent Systems & LLM Pipelines
├── [ ] Course 11: AI Model Deployment, Monitoring & MLOps
├── [ ] Course 12: Applied AI Ethics, Bias & Governance
└── [ ] Course 13: IBM AI Engineering Capstone Project (Final Comprehensive Showcase)
```

*(Remaining modules will be committed sequentially as they are completed and certified.)*

---

## 🔬 Architectural Summary of Implemented Models

### 1. Data Pipeline & Cross-Validation (`01_Rainfall_Prediction_Australia_Weather_ML`)
- Preprocessing structured within a Scikit-Learn `ColumnTransformer` (StandardScaler on continuous columns, OneHotEncoder on categorical features) to prevent leakage between validation folds.
- Hyperparameter search conducted via 5-fold `StratifiedKFold` across tree depths and regularization terms.

### 2. Deep Transfer Learning (`02_Waste_Classification_Transfer_Learning`)
- Pre-trained ImageNet features leveraged via **VGG-16**.
- Two-stage training scheme: initially freezing the convolutional base to stabilize the randomly initialized classifier head, followed by unfreezing upper blocks (`block5_conv3`) with a reduced learning rate (`1e-4`) to adapt high-level spatial representations.

### 3. PyTorch Linear Classifier with L2 Penalty (`03_League_of_Legends_Match_Predictor`)
- Implements custom `nn.Module` subclassing with explicit forward pass.
- L2 weight decay ($\lambda=0.01$) incorporated into the optimizer to penalize over-reliance on individual game statistics.
- Weight attribution analysis extracted to evaluate primary drivers of match outcomes.

### 4. Convolutional Neural Network with Batch Normalization (`04_Fashion_MNIST_Classification_CNN`)
- Two-dimensional batch normalization applied directly after convolution blocks (`Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d`).
- Mitigates internal covariate shift, allowing convergence with standard SGD at a learning rate of $0.1$.

---

## 🛠️ Environment & Reproducibility

### Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/dobby-aidev/<repository-name>.git
cd <repository-name>

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
# PyTorch (CPU or CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# TensorFlow, Scikit-Learn, and scientific libraries
pip install tensorflow "numpy<2" scikit-learn pandas matplotlib seaborn pillow jupyter
```

### Launch Interactive Notebooks

```bash
jupyter notebook
```

---

## 👨‍💻 Author & Profile

**Dobby B** ([@dobby-aidev](https://github.com/dobby-aidev))  
*Founder & AI Systems Architect at **DONA Codex** | Creator of **Agent Critiq***  
Focusing on Autonomous AI Systems, Multi-Agent Architectures, and Applied Deep Learning Engineering.

- 🌐 GitHub: [github.com/dobby-aidev](https://github.com/dobby-aidev)
- 🎓 Professional Certification: [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer)

---

## 📜 License

All code and materials in this repository are released under the [MIT License](./LICENSE).
