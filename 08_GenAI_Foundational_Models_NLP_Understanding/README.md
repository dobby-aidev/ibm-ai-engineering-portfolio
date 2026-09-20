# 🧠 Project 07 (Course 08): Gen AI Foundational Models for NLP & Language Understanding
### IBM AI Engineering Professional Certificate | Language Modeling & Sequence-to-Sequence Networks

![Status](https://img.shields.io/badge/Status-Completed_All_Labs-success.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20TorchText%20%7C%20Gensim%20%7C%20NLTK-orange.svg)
![Architectures](https://img.shields.io/badge/Architectures-EmbeddingBag%20%7C%20Skip--Gram%20%7C%20CBOW%20%7C%20Seq2Seq_RNN-blue.svg)
[![Coursera Certificate](https://img.shields.io/badge/Coursera_Certificate-Verified-0056D2?style=flat-square&logo=coursera)](https://coursera.org/share/c652444bce624cbd829127041e40daa9)

---

## 📌 Overview & Scope

This module dives into the architectural pillars of natural language understanding and autoregressive generative modeling prior to the transformer revolution. Understanding how distributed word representations, neural language models, and recurrent encoder-decoder structures process language is essential for mastering modern foundational LLMs.

This project delivers four fully executed, production-grade lab implementations:
1. **Document Classification with EmbeddingBag:** High-throughput document categorizer trained on AG_NEWS with PyTorch, saving inference-ready checkpoint weights (`my_model.pth`).
2. **Neural Network Language Modeling (NNLM):** Histogram n-gram probabilistic baselines versus deep feedforward neural networks (FNNs) with perplexity loss minimization and autoregressive lyric/rhyme synthesis.
3. **Word Embeddings (Word2Vec & Pretrained Transfer):** Building Word2Vec from scratch using Skip-gram and Continuous Bag-of-Words (CBOW) architectures, transferring Stanford GloVe vectors, and training custom Gensim embeddings.
4. **Sequence-to-Sequence (Seq2Seq) Machine Translation:** Encoder-Decoder RNN architecture with hidden state vector conditioning for German-to-English neural translation.

---

## 📂 Laboratory Breakdown

### 1. `01_Classifying_Document.ipynb`
- **Architecture:** PyTorch `nn.EmbeddingBag` with reduction mode (`mode="mean"`), dense layer projection, and SGD optimizer.
- **Deliverables:** Model trained on 120,000 news articles, inference pipeline with `my_model.pth`, zero-shot classification on new international climate & tech articles.

### 2. `02_Building_and_Training_Language_Model_with_NN.ipynb`
- **Architecture:** Feedforward N-Gram Language Modeler (`NGramLanguageModeler`) with context window embeddings and ReLU non-linearities.
- **Deliverables:** N-gram vocabulary extraction, cross-entropy training loop with learning rate scheduling, perplexity evaluation, and automated generative nursery rhyme synthesis (`write_song`).

### 3. `03_Integrating_Word2Vec_Part1.ipynb`
- **Architecture:** Contrastive representation learning via Skip-Gram (predicting context from center word) and CBOW (predicting center word from context).
- **Deliverables:** Negative sampling loss, custom Gensim word embedding layer integrated into a PyTorch text classification pipeline, and Stanford GloVe 50d vector alignment.

### 4. `04_Developing_a_Sequence_to_Sequence_Model.ipynb`
- **Architecture:** Dual-RNN Encoder-Decoder pipeline where the final hidden encoder state acts as the semantic bottleneck context vector for sequential decoder token generation.
- **Deliverables:** Sequence translation pipeline with custom tokenization, validation tracking, and inference on unseen German phrases (`"Menschen gehen auf der Straße"`).

---

## 👨‍💻 Developer & Portfolio Metadata
- **Developer:** Dobby B ([@dobby-aidev](https://github.com/dobby-aidev))
- **Role:** Founder & AI Systems Architect at Dona Codex
- **Program:** IBM AI Engineering Professional Certificate (Coursera)
