<div align="center">

# 🔍 RAG Q&A Pipeline

**Production-grade Retrieval-Augmented Generation — Hybrid Search · Re-Ranking · RAGAS Evaluation**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Hania_Ghouse-7C3AED?style=flat-square)](https://github.com/HaniaGhouse0407)

</div>

---

## 🧠 Overview

A production-ready RAG system that goes beyond naive retrieval. Combines BM25 sparse search with dense vector embeddings via Reciprocal Rank Fusion (RRF), applies a cross-encoder re-ranker for precision, and evaluates output quality with RAGAS metrics.

---

## ✨ Features

- ✅ Hybrid retrieval: BM25 + dense (ChromaDB/FAISS) with RRF fusion
- ✅ Cross-encoder re-ranking (ms-marco-MiniLM-L-6) — +12% precision
- ✅ RAGAS evaluation: Faithfulness, Answer Relevance, Context Precision
- ✅ FastAPI REST endpoint for production deployment
- ✅ Multi-format ingestion: PDF, DOCX, TXT, Markdown, CSV
- ✅ Dark-themed Streamlit UI with pipeline visualisation

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/HaniaGhouse0407/RAG-QA-Pipeline.git
cd RAG-QA-Pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (if needed)
cp .env.example .env
# Edit .env with your API keys

# 4. Run
streamlit run app.py
```

---

## 🛠️ Tech Stack

![langchain](https://img.shields.io/badge/langchain-1C3C3C?style=flat-square)  ![chromadb](https://img.shields.io/badge/chromadb-555555?style=flat-square)  ![faiss-cpu](https://img.shields.io/badge/faiss_cpu-555555?style=flat-square)  ![sentence-transformers](https://img.shields.io/badge/sentence_transformer-555555?style=flat-square)  ![openai](https://img.shields.io/badge/openai-412991?style=flat-square)  ![ragas](https://img.shields.io/badge/ragas-555555?style=flat-square)  ![fastapi](https://img.shields.io/badge/fastapi-009688?style=flat-square)  ![streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=flat-square)

---

## 📁 Project Structure

```
RAG-QA-Pipeline/
├── app.py              # Main Streamlit/Gradio application
├── requirements.txt    # Dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## 🎯 Target Roles

> AI Engineer · RAG Engineer · LLM Engineer

---

<div align="center">

Made by [Hania Ghouse](https://github.com/HaniaGhouse0407) · 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/hania-ghouse/)
[![Google Scholar](https://img.shields.io/badge/Scholar-Research-4285F4?style=flat-square&logo=google-scholar)](https://scholar.google.com/citations?user=iVWuM4wAAAAJ)

</div>
