Here’s a **README.md** for your **Tiny Transformer RAG Assistant** that highlights what makes it unique:

***

# 🌓 Tiny Transformer RAG Assistant

**A Lightweight Retrieval-Augmented Generation System with Custom Transformer and Cross-Encoder Reranking**

***

## ✅ **What Makes This Unique**

1.  **Custom Transformer Architecture**
    *   Implements a **wider tiny Transformer** (d=256, 8 heads, 6 layers) for language modeling and embeddings.
    *   Dual-purpose: **Language Model + Encoder** for semantic retrieval.

2.  **Two-Stage Retrieval Pipeline**
    *   **Bi-Encoder Stage**: Fast semantic similarity using transformer embeddings.
    *   **Cross-Encoder-Style Reranking**: Combines semantic score with lexical overlap for better relevance.

3.  **Dynamic RAG with System Context**
    *   Incorporates **desktop system metrics** (CPU, RAM, disk usage, top processes) into retrieval.
    *   Enables answering both **document-based** and **system-related queries**.

4.  **Adaptive Chunking & Tagging**
    *   Splits documents into chunks and tags them as `doc` or `system`.
    *   Supports **multi-source retrieval** with intelligent filtering.

5.  **Minimal Dependencies, Full Control**
    *   No reliance on external APIs or heavy frameworks.
    *   Built entirely with **PyTorch**, **Streamlit**, and **psutil** for system introspection.

6.  **Dark Mode UI**
    *   Sleek **Streamlit interface** with custom CSS for dark mode.
    *   Chat bubbles for user and bot messages.

7.  **Offline Training & Indexing**
    *   Train your own tiny Transformer on uploaded documents.
    *   Save and reload model + index for persistent usage.

8.  **Hybrid Scoring**
    *   Semantic similarity + lexical Jaccard overlap for robust retrieval.
    *   Lightweight approximation of cross-encoder without heavy compute.

***

## 🛠 **Tech Stack**

*   **PyTorch** – Custom Transformer LM
*   **Streamlit** – Interactive UI
*   **psutil** – System metrics integration
*   **PyPDF2 / python-docx** – Document parsing
*   **Torch DataLoader** – Efficient LM training

***

## 🔍 **How It Works**

1.  **Upload Documents**
    *   Supports `.txt`, `.pdf`, `.doc`, `.docx`.
2.  **Train Model**
    *   Builds vocabulary, trains tiny Transformer LM, computes embeddings.
3.  **Build Index**
    *   Chunks documents + system context, embeds them for retrieval.
4.  **Ask Questions**
    *   Retrieves relevant chunks using **bi-encoder + reranker**.
    *   Extracts answers from top-ranked chunks.
5.  **Chat Interface**
    *   Interactive Q\&A with chat bubbles and dynamic updates.

***

## 📦 **Installation**

```bash
pip install streamlit torch psutil PyPDF2 python-docx
streamlit run app.py
```

***

## ✅ **Why Use This?**

*   **Lightweight RAG** without external APIs.
*   **Customizable architecture** for research or edge deployment.
*   **System-aware assistant** for hybrid queries.
*   **Dark mode UI** for better user experience.


