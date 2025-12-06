# Semantic Activity Log Analyzer — **single‑file v24**

*Model persistence • Incremental analytics • Correlation + Pattern‑aware RCA • Learned attention pooler • Self‑optimizing batching • Robust cross‑correlation • Directed, readable topology*

> **One-file Streamlit app** (`app.py`) that ingests raw logs, classifies line‑level activities, computes anomaly trends, mines patterns per chunk, and builds a **cause → effect** topology using **semantic similarity + lag correlation + pattern overlap**. Designed for SRE/DevOps workflows where you need **fast, incremental signal extraction** from messy logs with minimal setup.

***

## ✨ Highlights

*   **Zero‑dependency model artifact**: Model is defined, cached, and persisted locally (`adaptive_transformer.pt`); auto‑compiled if PyTorch 2.x is present.
*   **Two‑level Transformer encoder**:
    *   Line‑level classification to 8 proxy activities (e.g., `CRASH_LOOP`, `TIMEOUT`) + per‑line anomaly score.
    *   Chunk‑level embedding aggregation via a **Learned Attention Pooler**, then a **Context‑Aware Router** for adaptive token selection.
*   **Incremental pipeline**: Processes logs chunk‑by‑chunk; **continuously updates** anomaly trend, pattern cards, and RCA topology.
*   **RCA Topology** (directed graph): Edges ranked by **cause score** that fuses:
    *   cosine similarity of chunk embeddings,
    *   anomaly gradient (rise),
    *   short **cross‑correlation with lag**,
    *   **Jaccard** overlap of mined n‑grams / exceptions / codes.
*   **Self‑optimizing batching**: Autoselects `max_len`, `batch_size`, `chunk_size`; **benchmarks** candidate batch sizes on your data.
*   **Plotting**: Matplotlib baseline; Plotly (optional) for an interactive graph if available.
*   **GPU‑aware**: Mixed precision (AMP) usage when CUDA is detected; token‑budget heuristics scale to available VRAM.

***

## 🧱 Architecture Overview

    Raw Logs
      └─► Line encode (ASCII clamp ≤ 255, pad/trunc to max_len)
             └─► Line Transformer + Positional Encoding
                    └─► Pooled line embeddings + 8‑label sigmoid head
                           └─► Per‑line anomaly = 1 − max(activity_prob)

    Chunks of lines
      ├─► Learned Attention Pooler (summary tokens, content‑aware queries)
      ├─► Context‑Aware Router (thresholded per‑token importance)
      ├─► Chunk Transformer + LayerNorm → Chunk embedding
      ├─► Dynamic Pattern Miner (n‑grams, Exceptions, HTTP, ERR codes)
      └─► Windowed anomaly series

    Across chunks
      ├─► Cosine similarity matrix (with cached norms)
      ├─► Robust short cross‑correlation with lag guard
      ├─► Jaccard pattern overlap
      └─► Directed graph (cause → effect), edge score via `cause_score()`

***

## 🧩 Key Components

*   **`PositionalEncoding`**: sinusoidal, auto‑extends beyond cached length.
*   **`ContextAwareRouter`**: linear scorer + sigmoid; routes tokens above threshold for chunk modeling.
*   **`LearnedAttentionPooler`**:
    *   `init_mode="mean"` (default): zero‑shot friendly, data‑conditioned queries.
    *   `init_mode="learned"`: trainable queries if you later fine‑tune.
*   **`AdaptiveHierarchicalTransformer`**:
    *   Line‑level encoder (multi‑layer Transformer encoder).
    *   Chunk‑level encoder with routing + pooling.
    *   8‑label line classifier (proxy for activity/anomaly).
*   **Temporal & RCA primitives**:
    *   **Cosine similarity** with cached norms.
    *   **Cross‑corr** over short windows with **every‑lag length equalization** (`max_lag` slider).
    *   **Pattern miner**: uni/bi/tri‑grams, `*Exception`, HTTP 4xx/5xx, `ERR_\d+`.
    *   **Cause score**: `w_sim*sim + w_grad*grad + w_lag*lag_bonus + w_patt*patt_sim`.
*   **Persistence**:
    *   `st.cache_resource` loader restores weights from `adaptive_transformer.pt`.
    *   💾 **Save** / ♻️ **Reset** model buttons in sidebar.

***

## 🛠️ Installation

> Tested with Python 3.10+.

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install streamlit torch numpy pandas matplotlib networkx

# Optional (for interactive graph)
pip install plotly
```

> **GPU (optional):** Install a CUDA‑enabled PyTorch build from <https://pytorch.org/get-started/locally/> suited to your driver.  
> Mixed precision is used automatically when CUDA is detected.

***

## ▶️ Run

```bash
streamlit run app.py
```

Then open the local URL that Streamlit prints (usually `http://localhost:8501`).

***

## 📂 Input

*   Upload any **textual log**: `.txt`, `.log`, `.csv`, `.json`
*   **Decoding**: tries `utf‑8`, `utf‑16`, `latin‑1`; truncates to 10 MB for safety.
*   **Timestamps**: heuristic regexes for common formats; also searches for JSON `"timestamp" | "time" | "date" | "ts"` keys.

**Example snippet** (works fine without strict structure):

```text
2025-04-18T10:02:13Z INFO App starting
2025-04-18T10:02:14Z WARN Retrying connection to 10.1.2.3
2025-04-18T10:02:16Z ERROR AuthenticationException user=svc-data
2025-04-18T10:02:17Z ERROR HTTP 504 upstream timeout
2025-04-18T10:02:22Z INFO Data processing batch=42 completed
```

***

## 🧭 UI Guide (Sidebar)

*   **Auto Optimize (recommended)**: enables dynamic choices for:
    *   `max_lines` (≤ 5k default), `max_len` (95th percentile clamp), `batch_size` (VRAM‑aware token budget), `chunk_size` (aims for \~12–40 chunks).
    *   Runs a tiny **micro‑benchmark** on your data to pick the faster batch size without OOM.
*   **Manual controls** (when Auto Optimize is off):
    *   **Max lines to analyze**: overall work cap.
    *   **Batch size**: per pass; higher is faster until memory limits.
    *   **Max tokens per line**: truncation length (ASCII‑clamped to ≤255).
    *   **Chunk size (lines per chunk)**: temporal granularity for RCA.
    *   **Use mixed precision (GPU only)**.
*   **Patterns per chunk** (`Top patterns per chunk`): how many of each category to show.
*   **Cross‑corr max lag (chunks)**: allowable lead/lag between chunk anomaly windows.
*   **Model**:
    *   **💾 Save current model to disk**
    *   **♻️ Reset persisted model (delete file)**

***

## 📈 Outputs

1.  **Raw Log Preview** (first \~2000 chars)
2.  **Effective Parameters** (actual runtime values; great for repros)
3.  **Anomaly Trend (partial → final)**: line‑wise anomaly score (`1 − max(activity_prob)`).
4.  **Pattern Cards (per chunk)**:
    *   Top **unigrams / bigrams / trigrams**
    *   Top **Exceptions** (e.g., `AuthenticationException`)
    *   Top **HTTP codes** (4xx/5xx)
    *   Top **ERR codes** (`ERR_###`)
5.  **RCA Topology — Cause → Effect**:
    *   **Nodes**: chunks; colored by anomaly (Reds), size ∝ anomaly.
    *   **Edges**: likely causal relations with hover details:
        *   `Score`, `Similarity`, `Lag`, `Pattern overlap`.
    *   Plotly (if installed; auto‑disabled for very large graphs) or Matplotlib fallback.

***

## ⚙️ Parameters & Heuristics

*   **Auto token budget** per batch considers device type and free VRAM.
*   **`max_len`** chosen from the **95th percentile** line length (bounded by `[64, 512]`).
*   **Chunk size** targets \~**12–40** chunks for stable graph layout and meaningful lag checks.
*   **Edge pruning**:
    *   Skip pairs below `sim_threshold` (adaptive).
    *   Keep top‑K edges per node (`base_topk` adaptive to graph size).
    *   Require **cause score ≥ influence threshold**.

***

## 🔒 Persistence

*   **Path**: `adaptive_transformer.pt` in the working directory.
*   On startup:
    *   Loads state if present; else initializes and saves after first creation.
    *   Tries to **torch.compile** (if PyTorch 2.x) to speed up inference.
*   Use the sidebar to **save** or **reset** the on‑disk state.

***

## 🧪 Extending / Integrating

*   Replace the 8‑label line head with your domain labels.
*   Swap the ASCII encoder with a tokenizer for real vocabularies (e.g., BPE) if you want.
*   Feed **structured logs** (e.g., JSON) and inject your own **timestamp extractor** via `extra_regex`.
*   Export graph data (NetworkX) for downstream post‑processing.

***

## 🧰 Troubleshooting

*   **CUDA OOM**:
    *   Enable **Auto Optimize**, or reduce `Batch size`, `Max tokens per line`, or `Max lines`.
    *   Ensure Plotly is disabled for huge graphs (the app does this automatically).
*   **Flat anomaly line**:
    *   Very uniform data; try increasing `max_len` or `chunk_size`.
*   **Sparse edges in topology**:
    *   Lower `sim_threshold`, increase `max_lag`, or increase `patterns_topk`.
*   **No timestamps detected**:
    *   Provide an `extra_regex` (see `first_ts_match`) or ensure your logs include any of the standard formats.

***

## 🔬 Notes on Methods

*   **Cosine similarity** is computed with cached norms for speed; values are clipped to `[-1, 1]`.
*   **Cross‑correlation** is **normalized** and length‑equalized **per lag**; ignores windows with too few points (`min_overlap=3`).
*   **Pattern overlap** uses a **Jaccard** index across the union of mined sets per chunk.
*   **Cause score** biases toward **higher anomaly on the effect side** and **positive lag** (cause precedes effect).

***

## 🧪 Example: Quick Start Scenario

1.  Run the app: `streamlit run app.py`
2.  Upload a combined service log (gateway + auth + data).
3.  Watch **anomaly spikes** align with `HTTP 5xx` and `AuthenticationException`.
4.  RCA graph shows:
    *   **Gateway chunk** → **Auth chunk** with positive lag.
    *   Pattern overlap reveals repeated `token refresh failed` bigrams.
5.  Use this to prioritize the **auth token refresh path** for incident mitigation.

***

## 📦 Project Structure

*   **`app.py`** — everything in one file:
    *   **PART 1**: Model & utilities
    *   **PART 2**: Analytics (temporal, correlation/RCA, pattern miner)
    *   **PART 3**: Streamlit app (UI + persistence)

***

## ✅ Requirements

*   Python 3.10+
*   Packages:
    *   `streamlit`, `torch`, `numpy`, `pandas`, `matplotlib`, `networkx`
    *   Optional: `plotly`

***

## 🔐 Data & Safety

*   The app processes logs in‑memory and **does not upload data** anywhere.
*   Truncates very large files to a **10 MB** cap for stability.
*   You may adapt this for PII scrubbing or redaction before upload if needed.

***

## 🗺️ Roadmap Ideas

*   Fine‑tuning hooks for domain labels / anomaly head.
*   Tokenizer upgrade (BPE) and pretraining hooks.
*   Export RCA to **GraphML** / **Cytoscape** / **Gephi**.
*   Session caching for multi‑file comparisons.
*   Multi‑modal features (metrics + traces + logs).

***

## 🙋 FAQ

**Q: Can I analyze millions of lines?**  
A: This single‑file app is optimized for **fast triage** (a few thousand lines). For very large datasets, consider pre‑sharding or adapting the chunking/streaming logic.

**Q: How “real” are the activities?**  
A: The 8 labels are **proxy activities** to drive anomaly scoring and clustering. Swap in your own classifier head and labeler for production semantics.

**Q: Do I need a GPU?**  
A: No. CPU works fine. A GPU will accelerate line encoding and transformer passes; the app auto‑adapts.

***

## 📄 License

Choose a license and add it here (e.g., MIT, Apache‑2.0).

***

## 🙌 Acknowledgments

*   Built with **PyTorch**, **Streamlit**, **NumPy/Pandas**, **Matplotlib/NetworkX** (and optional **Plotly**).


