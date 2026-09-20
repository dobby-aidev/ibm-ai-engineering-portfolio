# Course 12: Fundamentals of AI Agents Using RAG and LangChain

> **IBM AI Engineering Professional Certificate - Course 12 of 13**  
> **Curriculum Developer / Engineer:** Dobby B ([@dobby-aidev](https://github.com/dobby-aidev))  
> **Status:** Completed & Graded (100% Verified)  
> [![Coursera Certificate](https://img.shields.io/badge/Coursera_Certificate-Verified-0056D2?style=flat-square&logo=coursera)](https://coursera.org/share/33ebf6473a888f79099304b7c0ec0276)

---

## Course Overview

This course provides a comprehensive architectural foundation for building autonomous AI agents and Retrieval-Augmented Generation (RAG) pipelines. It focuses on combining large foundational models (such as LLaMA-3, Mixtral, and Granite) with external knowledge bases, vector search databases, prompt templates, and LangChain orchestration agents.

---

## Lab & Project Structure

| Notebook | Topic & Architecture | Key Libraries / Modules |
|---|---|---|
| [`01_RAG_with_Hugging_Face.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/12_Fundamentals_AI_Agents_RAG_LangChain/01_RAG_with_Hugging_Face.ipynb) | Dense Passage Retrieval & Generation | HuggingFace `transformers`, `faiss-cpu`, PyTorch |
| [`02_RAG_with_PyTorch.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/12_Fundamentals_AI_Agents_RAG_LangChain/02_RAG_with_PyTorch.ipynb) | Low-level Vector Similarity & Cosine Attention | Pure PyTorch, SentencePiece, Scikit-Learn |
| [`03_In_Context_Learning_Prompt_Templates.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/12_Fundamentals_AI_Agents_RAG_LangChain/03_In_Context_Learning_Prompt_Templates.ipynb) | Few-Shot Prompting, Chain-of-Thought, Reasoning | LangChain Core, `PromptTemplate`, `LLMChain` |
| [`04_Build_Smarter_AI_Apps_LangChain.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/12_Fundamentals_AI_Agents_RAG_LangChain/04_Build_Smarter_AI_Apps_LangChain.ipynb) | Tool-Using Agents, CSV Agents, Document Loaders | `langchain.agents`, `langchain-ibm`, ChromaDB |
| [`05_Summarize_Private_Docs_RAG_LangChain.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/12_Fundamentals_AI_Agents_RAG_LangChain/05_Summarize_Private_Docs_RAG_LangChain.ipynb) | Enterprise Document QA & Source Attribution | `RetrievalQA`, `WatsonxLLM`, Mixtral-8x7B |

---

## Technical Highlights

1. **End-to-End RAG Architecture:**
   - Ingested corporate policy documents, chunked text with boundary preservation, and embedded passages using dense sentence transformers.
   - Built FAISS and Chroma vector indices for low-latency top-$k$ nearest neighbor retrieval.
2. **Autonomous Tool-Using Agents:**
   - Deployed LangChain ReAct (Reason + Act) agents capable of querying structured CSV data and synthesizing analytical insights.
3. **Multi-Model Orchestration:**
   - Integrated IBM Watsonx foundational models (`meta-llama/llama-3-3-70b-instruct`, `mistralai/mixtral-8x7b-instruct-v01`, and `ibm/granite-3-8b-instruct`) with configurable decoding parameters and stop sequences.

---

## Submission & Verification

All 5 notebooks are verified with sequential execution ordering (`1..N`), complete exercise implementations, and validated outputs.
