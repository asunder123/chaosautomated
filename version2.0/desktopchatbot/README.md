Here’s a **README.md** tailored for your updated **Offline Semantic-Gated RAG v2.1** system, based on the architecture and features you described:

***

# 🤖 Offline Semantic-Gated RAG v2.1 (TXT + PDF)

## Overview

This project implements an **offline Retrieval-Augmented Generation (RAG)** chatbot with advanced **semantic gating**, **context anchoring**, and **transformer-based failover**. It is designed for environments where **internet access is restricted**, yet users need intelligent, document-aware Q\&A capabilities.

***

## ✅ Key Features

*   **Document Ingestion**:
    *   Supports `.txt` and `.pdf` files.
    *   Extracts text using `pdfplumber` for PDFs.
*   **Sliding-Window Chunking**:
    *   Single sentences, 2-sentence, and 3-sentence windows for richer context.
*   **Sentence Quality Scoring**:
    *   Penalizes boilerplate, headers, and very short/long sentences.
*   **Context Anchoring**:
    *   Strong alignment using nouns, verbs, noun chunks, and named entities.
*   **Dependency Alignment**:
    *   Matches subject–verb–object structure between query and candidate answers.
*   **Semantic Gating**:
    *   Ensures answers meet a minimum semantic similarity and anchor requirement.
*   **Transformer Failover**:
    *   Tiny cross-encoder reranks candidates when semantic gating fails.
*   **Anti-Collapse Memory**:
    *   Avoids repeating the same incorrect answer across unrelated queries.
*   **Offline Operation**:
    *   Entire pipeline runs locally (spaCy + FAISS + PyTorch).
*   **Streamlit UI**:
    *   Upload documents, build index, and chat interactively.

***

## Architecture Diagram

architecture.png

**Pipeline:**

    [Upload TXT/PDF] → [Text Extraction] → [Sliding Window Chunking]
           ↓
    [spaCy Embeddings + Quality Scoring] → [FAISS Index]
           ↓
    [Query Embedding + Keyword Extraction]
           ↓
    [Candidate Retrieval (Semantic + Keyword)]
           ↓
    [Semantic Gating + Context Anchoring]
           ↓
    [Tiny Transformer Failover (Cross-Encoder)]
           ↓
    [Answer Selection + Anti-Collapse Memory]
           ↓
    [Streamlit Chat UI]

***

## Installation

### Prerequisites

*   Python 3.8+
*   Virtual environment recommended

### Install Dependencies

```bash
pip install streamlit torch spacy faiss-cpu pdfplumber
python -m spacy download en_core_web_sm
```

***

## How It Works

1.  **Upload Documents**:
    *   Upload `.txt` or `.pdf` files via the Streamlit interface.
    *   Text is extracted and segmented into sentences and sliding windows.
2.  **Build Index**:
    *   Sentences are embedded using spaCy vectors.
    *   FAISS index is built for fast similarity search.
3.  **Ask Questions**:
    *   Query is embedded and compared against indexed chunks.
    *   Semantic + keyword + dependency scores determine the best candidate.
    *   If semantic gating fails, fallback transformer reranks candidates.
4.  **Answer Display**:
    *   The best candidate is shown in the chat interface.

***

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

### Steps:

*   Upload `.txt` or `.pdf` files.
*   Click **Build Index**.
*   Ask questions in the chat input.
*   Get document-aware answers offline.

***

## Configuration

*   **Semantic Threshold**: `SEM_THRESHOLD = 0.18` (adjust for stricter or looser gating).
*   **Top-K Retrieval**: Default `top_k=12`.
*   **Transformer Failover**:
    *   Lightweight cross-encoder with 2 layers, 128 hidden size.
    *   Dynamically builds vocabulary during runtime.

***

## Why Is This Approach Novel?

*   **Semantic Gating with Anchors**:
    *   Unlike standard RAG, this system enforces semantic similarity AND context anchor checks before accepting an answer.
*   **Hybrid Ranking**:
    *   Combines semantic similarity, keyword relevance, dependency alignment, and sentence quality.
*   **Failover Mechanism**:
    *   Introduces a secondary transformer-based cross-encoder for re-ranking when semantic gating fails.
*   **Anti-Collapse Memory**:
    *   Prevents repetitive incorrect answers across unrelated queries.
*   **Offline-First Design**:
    *   Entire pipeline runs locally without external APIs, making it suitable for air-gapped or secure environments.

***

## Future Enhancements

*   Add **BM25 keyword search** for hybrid retrieval.
*   Support **multi-turn conversation memory**.
*   Integrate **larger transformer models** for better failover performance.
*   Extend to **domain-specific embeddings** (e.g., SciSpaCy for scientific text).

***

## License

MIT License

***
