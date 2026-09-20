# ⚡ Project 08 (Course 09): Generative AI Language Modeling with Transformers
### IBM AI Engineering Professional Certificate | Self-Attention, GPT & BERT Architectures

![Status](https://img.shields.io/badge/Status-Completed_All_Labs-success.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20Hugging%20Face%20%7C%20TorchText-orange.svg)
![Architectures](https://img.shields.io/badge/Architectures-Scaled_Dot--Product_Attention%20%7C%20GPT_Causal_LM%20%7C%20BERT_MLM_NSP%20%7C%20Seq2Seq_Transformer-blue.svg)
[![Coursera Certificate](https://img.shields.io/badge/Coursera_Certificate-Verified-0056D2?style=flat-square&logo=coursera)](https://coursera.org/share/960c4f652914d610149e60f95be74c82)

---

## 📌 Overview & Scope

The **Transformer** architecture (Vaswani et al., 2017) replaced recurrent sequential computation with parallelized **Multi-Head Self-Attention**. This repository contains an end-to-end, code-level implementation of the foundational transformer building blocks across both **Encoder-only (BERT)**, **Decoder-only (GPT)**, and **Full Encoder-Decoder (Translation)** paradigms.

The project encompasses six production-grade laboratory implementations:
1. **Scaled Dot-Product & Multi-Head Self-Attention:** Mathematical implementation of Query-Key-Value ($Q, K, V$) projections, scaling factors ($\frac{1}{\sqrt{d_k}}$), softmax normalization, sinusoidal positional encoding, and attention weight heatmap visualization.
2. **Transformer Encoder for Classification:** PyTorch `nn.TransformerEncoder` stack with positional embeddings applied to sentiment and document classification.
3. **Causal Autoregressive Language Modeling (GPT-Style):** Lower-triangular causal attention masking ($\text{Mask}_{ij} = -\infty \text{ for } j > i$) ensuring strictly autoregressive next-token prediction.
4. **BERT Pre-training Data Pipelines:** Synthesizing Masked Language Modeling (80% `[MASK]`, 10% random, 10% unchanged) and Next Sentence Prediction (`IsNext` / `NotNext`) dataset formatting.
5. **Baby BERT from Scratch:** Constructing a complete bidirectional encoder with token, position, and segment embeddings from raw PyTorch modules, benchmarked against pre-trained `BertForPreTraining`.
6. **Full Seq2Seq Transformer for Machine Translation:** Cross-attention encoder-decoder pipeline with source and target masking for German-to-English neural translation.

---

## 📂 Laboratory Breakdown

### 1. `01_Attention_Mechanism_and_Positional_Encoding.ipynb`
- **Focus:** Attention mechanics & positional encoding.
- **Key Equations:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### 2. `02_Applying_Transformers_for_Classification.ipynb`
- **Focus:** Stacking `TransformerEncoderLayer` modules for document-level feature representation.
- **Deliverables:** Pooled output classification head with cross-entropy loss tracking.

### 3. `03_Decoder_Causal_LM_GPT_like_Models.ipynb`
- **Focus:** Autoregressive causal decoding.
- **Deliverables:** Custom triangular attention mask, next-token generation with temperature and sampling control.

### 4. `04_Data_Preparations_for_BERT.ipynb`
- **Focus:** Pre-training corpus synthesis.
- **Deliverables:** Dynamic masking algorithm and sentence-pair extraction for MLM + NSP.

### 5. `05_Encoder_Models_with_Baby_BERT.ipynb`
- **Focus:** From-scratch Bidirectional Encoder architecture.
- **Deliverables:** Token + Positional + Segment embedding combination, forward pass verification, and pre-trained checkpoint loading.

### 6. `06_Transformers_for_Translation.ipynb`
- **Focus:** Neural machine translation.
- **Deliverables:** Dual encoder-decoder network with cross-attention layers, greedy decoding inference, and document-level translation pipeline.

---

## 👨‍💻 Developer & Portfolio Metadata
- **Developer:** Dobby B ([@dobby-aidev](https://github.com/dobby-aidev))
- **Role:** Founder & AI Systems Architect at Dona Codex
- **Program:** IBM AI Engineering Professional Certificate (Coursera)
