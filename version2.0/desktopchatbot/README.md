Here’s a **README.md** for your project that explains the concept, features, setup, and usage of your **Offline Semantically-Gated RAG (FAISS + Transformer Failover)** chatbot:

***

# 📘 Offline Semantically-Gated RAG Chatbot

## Overview

This project implements an **offline Retrieval-Augmented Generation (RAG)** chatbot with **semantic gating** and **transformer-based failover**. It is designed for environments where **internet access is restricted**, yet users need intelligent document-aware Q\&A capabilities.

### ✅ Key Features

*   **Document Ingestion**: Supports `.txt` and `.pdf` files.
*   **Semantic Search**:
    *   Uses **spaCy embeddings** for sentence-level semantic similarity.
    *   FAISS-based vector index for fast retrieval (falls back to brute force if FAISS is unavailable).
*   **Keyword-Aware Ranking**:
    *   Combines semantic similarity with keyword relevance for better precision.
*   **Semantic Gating**:
    *   Ensures retrieved answers meet a minimum semantic similarity threshold.
*   **Transformer Failover**:
    *   If semantic gating fails, a lightweight **cross-encoder transformer** re-ranks candidates.
*   **Offline Operation**:
    *   No external API calls; works entirely on local resources.
*   **Streamlit UI**:
    *   Upload documents, build index, and chat interactively.

***

## Architecture

    [TXT/PDF Upload] → [Text Extraction] → [Sentence Segmentation] → [spaCy Embeddings]
           ↓
       [FAISS Index] ←→ [Semantic Search] ←→ [Keyword Scoring]
           ↓
       [Semantic Gate] → [Primary Answer]
           ↓
       [Transformer Failover] (if needed)
           ↓
       [Final Answer Display]

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
    *   Text is extracted and segmented into sentences.
2.  **Build Index**:
    *   Sentences are embedded using spaCy vectors.
    *   FAISS index is built for fast similarity search.
3.  **Ask Questions**:
    *   Query is embedded and compared against indexed sentences.
    *   Semantic + keyword scores determine the best candidate.
    *   If semantic similarity is below threshold, fallback transformer re-ranks candidates.
4.  **Answer Display**:
    *   The best sentence is shown as the answer in the chat interface.

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
*   **Top-K Retrieval**: Default `top_k=8`.
*   **Transformer Failover**:
    *   Lightweight cross-encoder with 2 layers, 128 hidden size.
    *   Dynamically builds vocabulary during runtime.

***

## Why Is This Approach Novel?

*   **Semantic Gating**:
    *   Unlike standard RAG, this system enforces a semantic similarity threshold before accepting an answer.
*   **Hybrid Ranking**:
    *   Combines semantic embeddings with keyword-based scoring for improved relevance.
*   **Failover Mechanism**:
    *   Introduces a secondary transformer-based cross-encoder for re-ranking when semantic gating fails.
*   **Offline-First Design**:
    *   Entire pipeline runs locally without external APIs, making it suitable for air-gapped or secure environments.

***
