# 🤖 Project 06 (Course 07): Generative AI and LLMs: Architecture & Data Preparation
### IBM AI Engineering Professional Certificate | Generative AI & NLP Pipelines

![Status](https://img.shields.io/badge/Status-Completed_All_Labs-success.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20Hugging%20Face%20%7C%20Transformers%20%7C%20spaCy-orange.svg)
![Topics](https://img.shields.io/badge/NLP-Tokenization%20%7C%20Custom_DataLoader%20%7C%20Collate_Fn-blue.svg)
[![Coursera Certificate](https://img.shields.io/badge/Coursera_Certificate-Verified-0056D2?style=flat-square&logo=coursera)](https://coursera.org/share/e34e7abf8e3be6d8936948271fabe7f6)

---

## 📌 Overview & Scope

This module establishes the architectural foundations of modern **Generative AI** and **Large Language Models (LLMs)**. Training and fine-tuning transformer-based models requires highly optimized, memory-efficient data pipelines capable of handling variable sequence lengths, dynamic padding, vocabulary numericalization, and subword tokenization.

This repository encapsulates the full technical pipeline across three production-ready notebooks:
1. **Generative AI Ecosystem Exploration:** Hands-on model inference with Hugging Face `transformers` (Flan-T5, GPT-2).
2. **Advanced Tokenization Strategies:** Comparative benchmarking of Word-level, Subword, Byte-Pair Encoding (BPE), WordPiece, and Unigram/SentencePiece algorithms across `NLTK`, `spaCy`, `BertTokenizer`, and `XLNetTokenizer`.
3. **PyTorch NLP Data Loaders & Dynamic Collate Functions:** Custom PyTorch `Dataset` and `DataLoader` engineering with sorting, custom tensor batching, and dynamic sequence padding (`collate_fn`) for multilingual translation corpora (French, German-English).

---

## 📂 Laboratory Breakdown

### 1. `01_Exploring_Generative_AI_Libraries.ipynb`
- **Objective:** Navigate the Hugging Face hub, configure autoregressive (GPT-2) and sequence-to-sequence (Flan-T5) pipelines.
- **Key Deliverables:** Model weight loading, tokenization decoders, text generation parameter control (temperature, top_k, max_length), and chatbot response comparison.

### 2. `02_Implementing_Tokenization.ipynb`
- **Objective:** Demystify vocabulary construction and subword segmentation techniques.
- **Key Deliverables:** 
  - Tokenization benchmark comparing `nltk.word_tokenize`, `spacy`, `BertTokenizer` (WordPiece), and `XLNetTokenizer` (SentencePiece/Unigram).
  - Vocabulary frequency distributions and latency profiling.

### 3. `03_Creating_an_NLP_Data_Loader.ipynb`
- **Objective:** Implement scalable data batching mechanisms in PyTorch for downstream LLM training.
- **Key Deliverables:**
  - Dynamic sequence padding with `torch.nn.utils.rnn.pad_sequence`.
  - Length-based sentence sorting to minimize `<PAD>` overhead.
  - Multilingual dataset collation pipeline for translation tasks.

---

## 👨‍💻 Developer & Portfolio Metadata
- **Developer:** Dobby B ([@dobby-aidev](https://github.com/dobby-aidev))
- **Role:** Founder & AI Systems Architect at Dona Codex
- **Program:** IBM AI Engineering Professional Certificate (Coursera)
