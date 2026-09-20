# Course 13: Generative AI Applications with RAG and LangChain (Project)

> **IBM AI Engineering Professional Certificate - Course 13 of 13 (Final Capstone Application)**  
> **Curriculum Developer / Engineer:** Dobby B ([@dobby-aidev](https://github.com/dobby-aidev))  
> **Status:** Completed & Graded (Score: 14/15 - 93% Passed)  
> **Verified Credential:** [🏅 Verify Coursera Certificate](https://coursera.org/share/09d704f3a93b0e952a81857ab2087aa3)

---

## Course Overview

Course 13 is the culminating hands-on project course of the **IBM AI Engineering Professional Certificate**. It brings together all skills developed throughout the curriculum—from foundational machine learning and deep neural networks to advanced transformer architectures, fine-tuning, and agentic workflows—into production-grade Generative AI enterprise applications using **LangChain**, **Watsonx.ai**, and **ChromaDB**.

---

## Lab & Project Structure

| Notebook | Topic & Architecture | Key Libraries / Modules |
|---|---|---|
| [`01_LangChain_Document_Loader.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/13_Project_GenAI_Applications_RAG_LangChain/01_LangChain_Document_Loader.ipynb) | Multi-Format Document Ingestion (PDF, ArXiv, Text, Web) | `langchain_community.document_loaders`, PyPDFLoader |
| [`02_Full_Document_Retrieve_Limitation.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/13_Project_GenAI_Applications_RAG_LangChain/02_Full_Document_Retrieve_Limitation.ipynb) | Context Window Constraints & Retrieval Bottlenecks | `ibm/granite-3-8b-instruct`, WatsonxLLM |
| [`03_LangChain_Text_Splitter.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/13_Project_GenAI_Applications_RAG_LangChain/03_LangChain_Text_Splitter.ipynb) | Recursive Character & Syntax-Aware (LaTeX/Code) Chunking | `RecursiveCharacterTextSplitter`, `Language` |
| [`04_Embed_Documents_Watsonx.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/13_Project_GenAI_Applications_RAG_LangChain/04_Embed_Documents_Watsonx.ipynb) | Dense Vector Embedding Generation | `ibm/slate-30m-english-rtrvr`, `WatsonxEmbeddings` |
| [`05_LangChain_Vector_Store.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/13_Project_GenAI_Applications_RAG_LangChain/05_LangChain_Vector_Store.ipynb) | Vector Database Indexing & Cosine / L2 Search | ChromaDB, FAISS |
| [`06_LangChain_Retriever.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/13_Project_GenAI_Applications_RAG_LangChain/06_LangChain_Retriever.ipynb) | MMR, Vector Store-Backed & Self-Querying Retrievers | `as_retriever()`, Maximal Marginal Relevance |
| [⭐ `07_Final_Project_RAG_LangChain.ipynb`](file:///c:/Users/ferda/Desktop/course%20e%C4%9Ftim%20final%20projeleri/13_Project_GenAI_Applications_RAG_LangChain/07_Final_Project_RAG_LangChain.ipynb) | **Graded Final Project: End-to-End Enterprise RAG Pipeline** | Complete Stack (All 5 Graded Tasks Solved) |

---

## Graded Final Project Breakdown (`07_Final_Project_RAG_LangChain.ipynb`)

The final graded submission consists of 5 mandatory tasks evaluated for 50 total points (100% score):

- **Task 1: Load Document from Remote S3 Cloud Storage (10 pts)**
  - Ingests LoRA research literature using `PyPDFLoader`.
  - Splits into distinct pages and validates content extraction.
- **Task 2: Apply Syntax-Aware Text Splitting (10 pts)**
  - Implements `RecursiveCharacterTextSplitter.from_language(Language.LATEX, chunk_size=60, chunk_overlap=0)` to preserve document structure.
- **Task 3: Embed Documents with Watsonx (10 pts)**
  - Configures `WatsonxEmbeddings` with `ibm/slate-30m-english-rtrvr` and input token truncation.
- **Task 4: Create and Configure Chroma Vector Database (10 pts)**
  - Downloads enterprise policy files, chunks with overlap, builds vector indices, and performs similarity search.
- **Task 5: Develop Top-$k$ Document Retriever (10 pts)**
  - Initializes retriever `as_retriever(search_kwargs={"k": 2})` and verifies query responses for `"Email policy"`.

---

## Submission & Verification

All 7 notebooks are completely executed, sequential cell counters (`1..N`) are verified, and all tasks are fully populated with exact execution outputs for direct submission to Coursera.
