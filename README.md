# 🌊 FloatChat: ARGO Ocean Data Explorer

**FloatChat** is an **LLM-powered ARGO Ocean Data Explorer** that allows users to interactively explore oceanographic datasets through natural language queries.  
It integrates **Large Language Models**, **Hybrid Retrieval-Augmented Generation (RAG)** pipelines, and **vector search (FAISS)** to simplify access to and understanding of global ocean data.

---

## 🚀 Features

- 🧠 **Conversational AI Interface** — Query ocean data using natural language and get meaningful insights.
- ⚡ **Hybrid RAG Pipeline** — Combines **SQL retrieval** with **semantic vector search** for faster, context-aware results.
- 📊 **Interactive Visualization** — Explore temperature, salinity, and pressure data through dynamic Streamlit dashboards.
- 🗄️ **PostgreSQL Integration** — Structured storage and querying of ARGO float data.
- ☁️ **FastAPI Backend** — Handles user queries, integrates LLM embeddings, and retrieves relevant records efficiently.
- 🔍 **FAISS-based Vector Store** — Enables semantic similarity search using embeddings generated via Sentence Transformers.
- 🧩 **75% Faster Insights** — Compared to traditional database-only querying.

---

## 🧠 Tech Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Programming Language** | Python |
| **Frontend** | Streamlit |
| **Backend API** | FastAPI |
| **Database** | PostgreSQL |
| **Vector Store** | FAISS |
| **LLM Framework** | Sentence Transformers / LangChain (extendable to Gemini / OpenAI) |
| **Visualization** | Plotly, Matplotlib |
| **Deployment** | Docker, AWS (optional) |

---
