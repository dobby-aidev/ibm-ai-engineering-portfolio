# Course 11: Generative AI Advance Fine-Tuning for LLMs

> **IBM AI Engineering Professional Certificate - Course 11 of 13**  
> **Curriculum Developer / Engineer:** Dobby B ([@dobby-aidev](https://github.com/dobby-aidev))  
> **Status:** Completed & Graded (100% Verified)  
> [![Coursera Certificate](https://img.shields.io/badge/Coursera_Certificate-Verified-0056D2?style=flat-square&logo=coursera)](https://coursera.org/share/abad7b18dbb29c3984843660a1b947b6)

---

## Course Overview

This advanced course explores post-training alignment, preference learning, and reinforcement learning techniques to adapt large language models for specialized domains and aligned behavior. It covers the complete modern post-training stack: **Instruction Fine-Tuning**, **Reward Modeling**, **Proximal Policy Optimization (PPO)**, **Direct Preference Optimization (DPO)**, **Best-of-N Sampling**, and **Sentiment Control**.

---

## Lab & Project Structure

| Notebook | Topic & Architecture | Key Libraries / Modules |
|---|---|---|
| [`01_Instruction_fine_tuning.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/11_GenAI_Advanced_FineTuning_LLMs/01_Instruction_fine_tuning.ipynb) | SFT with LoRA, Response Templates, Prompt Formatting | HuggingFace `trl`, `peft`, `transformers` |
| [`02_Reward_Modeling.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/11_GenAI_Advanced_FineTuning_LLMs/02_Reward_Modeling.ipynb) | Preference Data Modeling, Bradley-Terry objective | HuggingFace `AutoModelForSequenceClassification` |
| [`03_PPOTrainer.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/11_GenAI_Advanced_FineTuning_LLMs/03_PPOTrainer.ipynb) | Reinforcement Learning from Human Feedback (RLHF) | `trl.PPOTrainer`, `PPOConfig`, PyTorch |
| [`04_DPO_Fine_Tuning.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/11_GenAI_Advanced_FineTuning_LLMs/04_DPO_Fine_Tuning.ipynb) | Direct Preference Optimization without explicit reward model | `trl.DPOTrainer`, UltraFeedback Binarized |
| [`05_Best_of_N_Sampling.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/11_GenAI_Advanced_FineTuning_LLMs/05_Best_of_N_Sampling.ipynb) | Rejection Sampling & In-Context Reranking | PyTorch, HuggingFace Transformers |
| [`06_GPT2_Sentiment_Control.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/11_GenAI_Advanced_FineTuning_LLMs/06_GPT2_Sentiment_Control.ipynb) | Steering generation direction via reward signals | `transformers`, `pipeline`, PyTorch |
| [`07_GPT2_Sentiment.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/11_GenAI_Advanced_FineTuning_LLMs/07_GPT2_Sentiment.ipynb) | Baseline sentiment classification and evaluation | HuggingFace Datasets, PyTorch |

---

## Technical Highlights

1. **Instruction Supervised Fine-Tuning (SFT):**
   - Utilized Low-Rank Adaptation (LoRA) on `facebook/opt-350m` and `EleutherAI/gpt-neo-125m`.
   - Structured custom QA formatting prompts with specialized response templates to isolate loss calculation strictly to generated assistant tokens.
2. **Reward Modeling & Preference Scoring:**
   - Implemented pairwise ranking loss using chosen vs. rejected completions.
   - Evaluated preference classification accuracy against hold-out human annotations.
3. **PPO vs. DPO Alignment:**
   - Evaluated RLHF with PPO via actor-critic policy updates, KL-divergence penalties, and value head estimators.
   - Contrasted with DPO closed-form optimization, eliminating the need for reward model inference during alignment.

---

## Submission & Verification

All 7 notebooks are completely executed, sequential cell counters (`1..N`) are verified, and all exercise cells contain complete production code and outputs ready for Coursera evaluation.
