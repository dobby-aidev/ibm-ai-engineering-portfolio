# 🚀 Project 09 (Course 10): Generative AI Engineering & Fine-Tuning Transformers
### IBM AI Engineering Professional Certificate | Parameter-Efficient Fine-Tuning (PEFT), LoRA & QLoRA

![Status](https://img.shields.io/badge/Status-Completed_All_Labs-success.svg)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20Hugging%20Face%20%7C%20TRL%20%7C%20PEFT%20%7C%20BitsAndBytes-orange.svg)
![Techniques](https://img.shields.io/badge/Fine--Tuning-Full_SFT%20%7C%20Bottleneck_Adapters%20%7C%20LoRA%20%7C%20QLoRA_4--bit-blue.svg)
[![Coursera Certificate](https://img.shields.io/badge/Coursera_Certificate-Verified-0056D2?style=flat-square&logo=coursera)](https://coursera.org/share/a24bb93f972968c0a86a4a5017a48086)

---

## 📌 Overview & Scope

Fine-tuning modern Large Language Models (LLMs) requires optimizing billions of parameters within constrained compute budgets. While full model fine-tuning often leads to catastrophic forgetting and prohibitive memory footprints, **Parameter-Efficient Fine-Tuning (PEFT)** techniques allow fine-tuning less than 1% of the original model weights while maintaining or exceeding downstream performance.

This project delivers seven comprehensive, production-grade lab implementations across two core modules:
1. **Module 1 (Pre-training & Supervised Fine-Tuning):** Model loading, tokenization pipelines, causal LM pre-training, BERT classification on Yelp reviews, and conversational alignment using `TRL`'s `SFTTrainer` on the OpenAssistant Guanaco corpus.
2. **Module 2 (Parameter-Efficient Adaptation):**
   - **Bottleneck Adapters:** Injecting low-dimensional linear compression modules (`Adapted`) between frozen transformer layers.
   - **Low-Rank Adaptation (LoRA):** Decomposing weight updates into rank-constrained matrices ($\Delta W = B \times A, \quad r \ll d$).
   - **Quantized LoRA (QLoRA):** Reducing base model precision to 4-bit NormalFloat (NF4) with Double Quantization and Paged Optimizers via `bitsandbytes` and Hugging Face `peft`.

---

## 📂 Laboratory Breakdown

### 1. `01_Loading_Models_and_Inference_HuggingFace.ipynb`
- **Focus:** Hugging Face Hub inference workflows.
- **Key Concepts:** `AutoTokenizer`, `AutoModelForSequenceClassification`, `pipeline("sentiment-analysis")`.

### 2. `02_Pretraining_LLMs_HuggingFace.ipynb`
- **Focus:** Autoregressive language modeling from scratch.
- **Key Concepts:** Causal LM pretraining config, `DataCollatorForLanguageModeling`.

### 3. `03_PreTraining_FineTuning_PyTorch.ipynb`
- **Focus:** Supervised fine-tuning with native PyTorch.
- **Key Concepts:** Full parameter update loop on Yelp polarity reviews with evaluation metrics.

### 4. `04_FineTuning_Transformers_PyTorch_HF.ipynb`
- **Focus:** Instruction fine-tuning for conversational LLMs.
- **Key Concepts:** `TRL` library, `SFTTrainer`, OpenAssistant Guanaco dataset alignment.

### 5. `05_Adapters_in_PyTorch.ipynb`
- **Focus:** Houlsby-style bottleneck adapters.
- **Key Formula:** $h \leftarrow h + f(h W_{\text{down}}) W_{\text{up}}$ with frozen original linear weights.

### 6. `06_LoRA_with_PyTorch.ipynb`
- **Focus:** Low-Rank Adaptation mathematical mechanics.
- **Key Formulation:**
  $$W = W_0 + \frac{\alpha}{r} (B \cdot A), \quad A \sim \mathcal{N}\left(0, \frac{1}{r}\right), \quad B = 0$$
- **Deliverables:** Custom `LoRALayer` injection replacing dense classification weights.

### 7. `07_QLoRA_with_HuggingFace.ipynb`
- **Focus:** 4-bit memory-efficient fine-tuning.
- **Key Technologies:** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`, PEFT integration.

---

## 👨‍💻 Developer & Portfolio Metadata
- **Developer:** Dobby B ([@dobby-aidev](https://github.com/dobby-aidev))
- **Role:** Founder & AI Systems Architect at Dona Codex
- **Program:** IBM AI Engineering Professional Certificate (Coursera)
