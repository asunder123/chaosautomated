Here’s an updated **README.md** for your project based on the latest `app.py` (v32):

***

# **Semantic Activity Log Analyzer**

A **Streamlit-based interactive dashboard** for analyzing large-scale logs using a **hierarchical Transformer model** with advanced features:

*   ✅ **Central Directed RCA Topology** (Root Cause Analysis graph)
*   ✅ **Activity Summary** (counts, transitions, narrative)
*   ✅ **Predictive Insights** (risk trends, next activity, error drivers, recommendations)
*   ✅ **Timeseries Analysis** (volume, anomaly, severity, activity trends)
*   ✅ **Correlation Heatmap** (metrics interdependencies)
*   ✅ **Error Breakup** (exceptions, HTTP codes, custom error codes)
*   ✅ **Model Persistence** (adaptive hierarchical Transformer)
*   ✅ **Self-optimizing batching** for GPU/CPU
*   ✅ **Pattern Miner** (tokens, exceptions, HTTP spikes)
*   ✅ **Interactive RCA graph with Plotly or Matplotlib fallback**

***

## **Features Overview**

### 🔍 **Core Capabilities**

*   **Upload Logs**: Supports `.txt`, `.log`, `.csv`, and JSON arrays (`timestamp`, `level`, `message`).
*   **Transformer-based Analysis**:
    *   Line-level classification into activities (e.g., `AUTH_FAILURE`, `TIMEOUT`, `CRASH_LOOP`).
    *   Anomaly scoring per line.
    *   Chunk-level embeddings for RCA graph.
*   **Dynamic RCA Graph**:
    *   Multi-evidence scoring (similarity, anomaly gradient, lag correlation, severity drift, pattern overlap).
    *   Interactive Plotly visualization with arrows, hover details, and Sankey sequence view.
*   **Predictive Insights**:
    *   Risk trend (EWMA of anomaly scores).
    *   Likely next activity (Markov-style heuristic).
    *   Top causal edges and error drivers.
    *   Actionable recommendations based on patterns.

### 📊 **Analytics Panels**

*   **Activity Summary**: Top activities, transitions, and narrative summary.
*   **Timeseries**: Volume, anomaly mean, severity mix, activity trends.
*   **Correlation Heatmap**: Pearson correlation among resampled metrics.
*   **Error Breakup**: Exceptions, HTTP status codes, custom error codes.
*   **Pattern Cards**: Frequent tokens, bigrams, trigrams, exceptions per chunk.

***

## **Architecture Highlights**

*   **Adaptive Hierarchical Transformer**:
    *   Positional encoding for line and chunk levels.
    *   Learned attention pooler for summarization.
    *   Context-aware router for token importance.
*   **GPU Optimization**:
    *   Mixed precision (AMP) support.
    *   Auto batch size and chunk size selection based on GPU memory.
*   **Stable RCA Layout**:
    *   Spring layout with position caching.
    *   Handles node changes gracefully.
*   **Robust RCA Scoring**:
    *   Cosine similarity + anomaly gradient + lag correlation + severity drift + pattern overlap.

***

## **Installation**

### **Prerequisites**

*   Python 3.9+
*   Recommended: GPU with CUDA for acceleration.

### **Install Dependencies**

```bash
pip install streamlit torch numpy pandas matplotlib networkx plotly
```

***

## **Usage**

### **Run the App**

```bash
streamlit run app.py
```

### **Steps**

1.  Upload a log file (`.txt`, `.log`, `.csv`, or JSON array).
2.  Adjust sidebar controls:
    *   Auto optimization (recommended).
    *   RCA graph filters (min edge score, influence, arrow thickness).
    *   Timeseries frequency.
3.  Explore panels:
    *   RCA Topology (interactive graph).
    *   Activity Summary.
    *   Predictive Insights.
    *   Timeseries, Correlation, Error Breakup.

***

## **Predictive Insights Logic**

*   **Risk Trend**: EWMA slope of anomaly scores.
*   **Next Activity**: Transition frequency from classified activities.
*   **Root Causes**: RCA graph edges scored by multi-evidence.
*   **Error Drivers**: Exceptions, HTTP spikes, custom codes.
*   **Recommendations**: Pattern-based hints (e.g., IAM issues, timeouts, crash loops).

***

## **Key Controls**

*   **Graph Visibility**:
    *   Min edge score/influence.
    *   Arrow thickness and offset.
*   **Performance**:
    *   Auto batch size and chunk size.
    *   Mixed precision toggle.

***

## **Screenshots**

*(Add screenshots of RCA graph, Activity Summary, Predictive Insights panels here)*

***

## **Future Enhancements**

*   ✅ Export RCA graph as PNG/SVG.
*   ✅ Timeline-DAG layout (X=time, Y=anomaly).
*   ✅ ML-based next-activity prediction (beyond heuristics).
*   ✅ Integration with enterprise log sources (S3, Azure Blob, etc.).

***

### **License**

MIT License.

***
