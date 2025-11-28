Here’s a **README.md** draft that highlights the novelty and purpose of your transformer architecture:

***

# **FAHT: Fully Adaptive Hierarchical Transformer for Semantic Log Analysis**

## **Overview**

FAHT is a **novel hierarchical Transformer architecture** designed for **semantic log analysis and anomaly detection** in large-scale distributed systems. Unlike traditional models, FAHT introduces **adaptive summarization** and **dynamic positional encoding**, enabling robust handling of **variable-length logs** and **real-time operational insights**.

***

## **Key Features**

*   **Two-Level Hierarchy**
    *   **Line-Level Transformer**: Processes individual log lines for fine-grained activity classification.
    *   **Chunk-Level Transformer**: Aggregates line embeddings into semantic chunks for higher-order analysis.

*   **Adaptive Summary Tokens**
    *   Dynamically generated using **soft clustering** over line embeddings.
    *   Enables flexible summarization of variable-sized log chunks.

*   **Dynamic Positional Encoding**
    *   Extends beyond precomputed limits for **unbounded sequence lengths**.
    *   Ensures scalability for massive logs without truncation.

*   **Integrated Anomaly Scoring**
    *   Inline computation (`1 - max(logit)`) during line-level classification.
    *   Eliminates need for separate anomaly detection models.

*   **Operational Dashboard**
    *   Built with **Streamlit** for real-time visualization:
        *   Anomaly heatmaps
        *   Semantic phase summaries
        *   Downloadable incident reports

***

## **Why FAHT is Unique**

Compared to published models like **HLogformer** and **HitAnomaly**, FAHT offers:

*   **Adaptive summarization** (soft clustering) vs fixed CLS tokens.
*   **Dynamic positional encoding** for unlimited log length.
*   **Combined anomaly detection + semantic summarization** in one pipeline.
*   **Practical deployment** via interactive dashboard for operational teams.

***

## **Architecture**

    Log Lines → [Line-Level Transformer] → Line Embeddings
           → [Adaptive Summary Tokens via Clustering]
           → [Chunk-Level Transformer] → Chunk Embeddings
           → Semantic Summaries + Anomaly Scores

***

## **Use Cases**

*   **Incident Analysis**: Detect anomalies and summarize phases in system logs.
*   **Root Cause Investigation**: Cluster semantic patterns across large log files.
*   **Operational Dashboards**: Real-time monitoring and reporting.

***

## **Getting Started**

1.  Install dependencies:
    ```bash
    pip install streamlit torch scikit-learn pandas matplotlib
    ```
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  Upload log files and view:
    *   **Raw preview**
    *   **Activity classification**
    *   **Anomaly heatmap**
    *   **Semantic summaries**

***

## **Novelty Statement**

FAHT bridges the gap between **research-grade hierarchical models** and **production-ready log intelligence systems**, offering:

*   Adaptive summarization for semantic grouping.
*   Dynamic positional encoding for scalability.
*   Integrated anomaly detection and summarization.
*   Real-time visualization for operational usability.

***


@AnandSunder
