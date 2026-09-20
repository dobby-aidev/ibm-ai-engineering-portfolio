# 🛰️ Project 05: Satellite Image Classification (Agricultural vs. Non-Agricultural Land Cover)
### IBM AI Engineering Capstone Project | Deep Learning, CNNs & Vision Transformers (ViTs)

![Status](https://img.shields.io/badge/Status-Passed_73%2F100-success.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-TensorFlow%20%7C%20Keras%20%7C%20PyTorch-orange.svg)
![Models](https://img.shields.io/badge/Architectures-Custom_CNN%20%7C%20ViT%20%7C%20CNN--ViT_Hybrid-blue.svg)
[![Coursera Certificate](https://img.shields.io/badge/Coursera_Certificate-Verified-0056D2?style=flat-square&logo=coursera)](https://coursera.org/share/ec4b26f9230f0f7c92bbea404bd0d8e8)

---

## 📌 Executive Summary

This capstone project tackles automated geospatial land cover classification using remote sensing satellite imagery. Designed for precision agriculture and land use planning (e.g. fertilizer and irrigation optimization), the objective is binary classification: determining whether a given satellite tile represents **Agricultural Land (`class_1_agri`)** or **Non-Agricultural Land (`class_0_non_agri`)**.

The curriculum spans four comprehensive modules that bridge standard sequential convolutional neural networks (CNNs) and cutting-edge **Vision Transformers (ViTs)** using both **Keras / TensorFlow** and **PyTorch**.

---

## 📂 Final Project Submission Guide (9 Labs / 100 Points)

The Coursera AI Auto-Grader requires uploading 9 fully executed Jupyter Notebooks (`.ipynb`) corresponding to Questions 1 through 9. All 9 finalized files are ready in the [`submission_files/`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/) folder:

| Question # | Points | File to Upload | Key Tasks & Deliverables |
|:---:|:---:|:---|:---|
| **Question 1** | **10 pts** | [`01_Compare_Memory_Based_vs_Generator_Based_Data_Loading.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/01_Compare_Memory_Based_vs_Generator_Based_Data_Loading.ipynb) | Single image shape `(64, 64, 3)`, non-agri 4-image visualization, sorted `agri_images_paths`, count agricultural images. |
| **Question 2** | **8 pts** | [`02_Data_Loading_and_Augmentation_Using_Keras.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/02_Data_Loading_and_Augmentation_Using_Keras.ipynb) | `all_image_paths` generator, temporary labeled pair list `temp`, custom batch generator (batch size 8), `val_ds` validation data pipeline. |
| **Question 3** | **10 pts** | [`03_Data_Loading_and_Augmentation_Using_PyTorch.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/03_Data_Loading_and_Augmentation_Using_PyTorch.ipynb) | PyTorch `custom_transform` pipeline (64x64, HFlip 0.5, VFlip 0.2, Rot 45°), `ImageFolder` loader, class index mapping, batch tensor shapes and visualization. |
| **Question 4** | **12 pts** | [`04_Train_and_Evaluate_Keras_Based_Classifier.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/04_Train_and_Evaluate_Keras_Based_Classifier.ipynb) | Directory traversal, `validation_generator`, layer count verification (38 layers), 4-Conv2D + 5-Dense CNN compilation, `ModelCheckpoint` on `val_accuracy`, training/val loss plotting. |
| **Question 5** | **20 pts** | [`05_Implement_and_Test_PyTorch_Based_Classifier.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/05_Implement_and_Test_PyTorch_Based_Classifier.ipynb) | Random initialization theory, `train_transform`/`val_transform`, `val_loader`, `tqdm` monitoring, epoch metric reset rationale, `torch.no_grad()` inference, loss curve plot, `all_preds`/`all_labels` extraction. |
| **Question 6** | **10 pts** | [`06_Comparative_Analysis_Keras_and_PyTorch_Models.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/06_Comparative_Analysis_Keras_and_PyTorch_Models.ipynb) | Thresholding analysis (`preds > 0.5`), Keras `print_metrics`, harmonic F1 score significance, PyTorch `print_metrics`, confusion matrix False Negative inspection. |
| **Question 7** | **10 pts** | [`07_Vision_Transformers_in_Keras.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/07_Vision_Transformers_in_Keras.ipynb) | Pretrained CNN feature extractor loading & summary, intermediate feature layer identification, `build_cnn_vit_hybrid` assembly, compilation & training config. |
| **Question 8** | **12 pts** | [`08_Vision_Transformers_in_PyTorch.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/08_Vision_Transformers_in_PyTorch.ipynb) | PyTorch train/val pipelines, `model_test` training (epochs=5, heads=12, embed_dim=768, depth=12), validation loss comparison curve, training execution time benchmarks. |
| **Question 9** | **8 pts** | [`09_Land_Classification_CNN_ViT_Integration_Evaluation.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/05_Satellite_Image_Classification_Land_Cover/submission_files/09_Land_Classification_CNN_ViT_Integration_Evaluation.ipynb) | Hyperparameters and directory setup, PyTorch model instantiation, full evaluation metrics for `Keras CNN-Vit Hybrid Model` and `PyTorch CNN-Vit Hybrid Model`. |
| **Total** | **100 pts** | **All 9 Notebooks** | **Passing score: ≥ 70%** |

---

## 🧠 Architectural Highlights

### 1. Memory-Based vs. Generator-Based Data Ingestion
- **Memory-Based (Bulk):** Loads all images into RAM at once using NumPy arrays. Fast access, but memory explodes with high-resolution satellite imagery.
- **Generator-Based (Lazy):** Reads filenames from disk and streams batches on the fly with real-time data augmentations (flips, rotations). Essential for production-scale geospatial datasets.

### 2. CNN vs. Vision Transformer (ViT) Hybrid
- **CNNs (Local Features):** Convolutions excel at capturing fine-grained local textures, edges, and crop boundary lines.
- **ViTs (Global Context):** Multi-Head Self-Attention layers compute global spatial relationships across image patches, allowing the network to understand large-scale topography and vegetation patterns.
- **CNN-ViT Hybrid:** Combines the best of both paradigms: CNN feature extraction backbone feeding into stacked Transformer encoder blocks.

---

## 📊 Comparative Performance Benchmark

| Model Architecture | Framework | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom 4-Conv CNN** | Keras | 99.25% | 0.9904 | 0.9947 | 0.9925 | 0.9989 |
| **Custom PyTorch CNN** | PyTorch | **99.88%** | **0.9976** | **1.0000** | **0.9988** | **0.9999** |
| **CNN-ViT Hybrid** | Keras | 98.65% | 0.9840 | 0.9890 | 0.9865 | 0.9972 |
| **CNN-ViT Hybrid** | PyTorch | 99.42% | 0.9920 | 0.9964 | 0.9942 | 0.9995 |

---

## 👨‍💻 Developer & Portfolio Metadata
- **Developer:** Dobby B ([@dobby-aidev](https://github.com/dobby-aidev))
- **Role:** Founder & AI Systems Architect at Dona Codex
- **Program:** IBM AI Engineering Professional Certificate (Coursera)
