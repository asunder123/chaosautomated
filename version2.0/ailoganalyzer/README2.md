Here’s an updated **README** that reflects your new architecture, fixes, and enhancements:

***

# **SHATCAR Log Analyzer**

> **Structural-Hierarchical Adaptive Transformer with Context-Aware Routing**  
> A Streamlit-based application for **log ingestion, parsing, analytics, RCA, structural pattern mining, and transformer-driven summarization/topology**.

***

## ✅ **Key Features**

*   **Adaptive Parsing**
    *   Handles plain text and JSON logs
    *   Robust timestamp, level, service, and trace extraction
    *   Structural template generation for pattern mining

*   **Persistent Storage**
    *   SQLite-backed log store with indexing for fast retrieval
    *   Batch inserts for large files

*   **Temporal Analytics**
    *   Error density over time
    *   Configurable bucket sizes (`second`, `minute`, `hour`)
    *   Interactive charts via Matplotlib

*   **Root Cause Analysis (RCA)**
    *   Groups logs by `trace_id`
    *   Identifies earliest ERROR/CRITICAL/FATAL per trace
    *   Aggregates top error services and messages

*   **Structural Pattern Mining**
    *   Template extraction with placeholders (`<NUM>`, `<UUID>`, `<IP>`, `<HEX>`)
    *   Frequency-based ranking

*   **SHATCAR Transformer (Upgraded)**
    *   **Forward Features API** for latent embeddings
    *   Multi-level hierarchical encoder with context-aware routing
    *   Context fusion: semantic, level, service, time features
    *   Cyclical time encoding for periodicity
    *   Classification head for error prediction

*   **Transformer-Driven Topology**
    *   Service graph based on latent similarity (not logits)
    *   Weighted directional edges with thresholding
    *   Clean layouts for small and large graphs

*   **Secure Transformer-Based Summary**
    *   Embedding-space diversity analysis
    *   spaCy-driven keyword extraction
    *   Privacy-preserving, human-readable interpretation

***

## 🛠 **Installation**

```bash
# Clone repo
git clone <your-repo-url>
cd shatcar-log-analyzer

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
# (Optional for better semantics)
python -m spacy download en_core_web_md
```

***

## 📂 **Project Structure**

    app.py                # Main Streamlit app (UI + logic)
    logs.db               # SQLite log store
    shatcar_model.pt      # Trained SHATCAR model
    shatcar_vocab.json    # Vocabulary for structural tokens
    requirements.txt      # Dependencies
    README.md             # This file

***

## ▶️ **Run the App**

```bash
streamlit run app.py
```

***

## ⚙️ **Usage Workflow**

1.  **Upload Logs**
    *   Supports `.log`, `.txt`, `.json`, `.jsonl`
    *   Stored in SQLite for persistence

2.  **Explore Analytics**
    *   Temporal charts
    *   RCA tables
    *   Structural patterns

3.  **Train SHATCAR**
    *   Adjustable epochs and batch size
    *   Displays accuracy and saves model/vocab

4.  **Run Inference**
    *   Scores recent logs for error probability

5.  **Generate Summary & Topology**
    *   Secure transformer-based summary
    *   Service interaction graph

***

## 🔒 **Security & Privacy**

*   No sensitive paths exposed in UI
*   Model and vocab saved locally with permission checks
*   Summaries avoid raw PII; use abstracted patterns

***

## ✅ **Recent Enhancements**

*   Fixed HTML entity artifacts (`-&gt;` → `->`, etc.)
*   Added `forward_features()` for consistent embeddings
*   Stable semantic hashing (BLAKE2b) for reproducibility
*   Honored bucket parameter in temporal analytics
*   Cyclical time encoding for better periodic modeling
*   Robust JSON timestamp parsing (ms vs sec)
*   Index added to SQLite for performance

***

