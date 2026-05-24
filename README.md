# ⚖️ Know Your Laws – Agentic Legal RAG System

An AI-powered Legal Retrieval-Augmented Generation (RAG) system designed for Indian legal documents and cyber law assistance.

This project combines:
- LangGraph workflows
- FAISS vector retrieval
- Groq LLMs
- HuggingFace embeddings
- Streamlit frontend

to provide context-aware legal question answering over Indian bare acts.

---

## 🚧 Project Status
- Initial baseline prototype completed.
But project is still under active development as I look to add some more features.

The current version is a functional baseline prototype focused on:
- legal document retrieval
- semantic legal search
- context-aware answer generation

Future versions will include:
- advanced agentic workflows
- legal citation generation
- conversational memory
- hybrid retrieval
- multi-agent reasoning

---

## ✨ Current Features

- PDF-based legal document ingestion
- Semantic chunking and retrieval
- FAISS vector database
- HuggingFace embedding integration
- Groq LLM integration
- Streamlit conversational UI
- Context-aware legal QA
- Indian legal document support
- Retrieval-Augmented Generation (RAG) pipeline

---

## 📚 Current Legal Corpus

The system currently includes:

- Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023
- Information Technology Act, 2000
- Consumer Protection Act, 2019

---

## 🛠️ Tech Stack

- Python
- LangChain
- LangGraph
- Streamlit
- FAISS
- HuggingFace Embeddings
- Groq API
- PyMuPDF

---

## 🧠 System Architecture

User Query  
→ Streamlit UI  
→ LangGraph Workflow  
→ FAISS Retriever  
→ Relevant Legal Context  
→ Groq LLM  
→ Final Legal Response  

---

## 📂 Project Structure

```text
Know_Your_Laws-Agentic-AI-/
│
├── data/
│   └── legal_pdfs/
│
├── src/
│   ├── config/
│   ├── document_ingestion/
│   ├── graph_builder/
│   ├── nodes/
│   ├── state/
│   └── vectorstore/
│
├── streamlit_app.py
├── main.py
├── requirements.txt
└── README.md