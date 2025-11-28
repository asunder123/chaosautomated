import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt

# ======================================================
# 1. Streamlit Config
# ======================================================
st.set_page_config(page_title="Semantic Activity Log Analyzer v15", layout="wide")

# ======================================================
# 2. Positional Encoding
# ======================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            pe = torch.zeros(seq_len, x.size(2), device=x.device)
            pos = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div = torch.exp(torch.arange(0, x.size(2), 2, device=x.device).float() * (-np.log(10000.0)/x.size(2)))
            pe[:,0::2] = torch.sin(pos*div)
            pe[:,1::2] = torch.cos(pos*div)
            pe = pe.unsqueeze(0)
        else:
            pe = self.pe[:, :seq_len, :]
        return x + pe

# ======================================================
# 3. Fully Adaptive Hierarchical Transformer
# ======================================================
class AdaptiveHierarchicalTransformer(nn.Module):
    """
    Fully adaptive hierarchical transformer for logs.
    - Adaptive summary tokens per chunk
    - Handles variable line lengths and number of lines
    """
    def __init__(self, vocab_size=256, embed_dim=512, n_heads=16, line_layers=4, chunk_layers=2, max_summary_tokens=4, num_classes=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.line_pos = PositionalEncoding(embed_dim)
        self.line_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                       dim_feedforward=embed_dim*4, batch_first=True, activation="gelu"),
            num_layers=line_layers
        )
        self.chunk_pos = PositionalEncoding(embed_dim)
        self.chunk_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                       dim_feedforward=embed_dim*4, batch_first=True, activation="gelu"),
            num_layers=chunk_layers
        )
        self.norm_chunk = nn.LayerNorm(embed_dim)
        self.max_summary_tokens = max_summary_tokens
        self.num_classes = num_classes
        self.fc_line = nn.Linear(embed_dim, num_classes)

    # ----------------------
    # Line-level forward
    # ----------------------
    def forward_line(self, x):
        emb = self.embed(x)
        emb = self.line_pos(emb)
        out = self.line_transformer(emb)
        pooled = out.mean(dim=1)  # mean pooling over line sequence
        logits = torch.sigmoid(self.fc_line(pooled))
        return logits, pooled

    # ----------------------
    # Chunk-level adaptive forward
    # ----------------------
    def forward_chunk_adaptive(self, line_embeddings_list):
        if len(line_embeddings_list) == 0:
            return torch.zeros(1, self.embed_dim)

        # Flatten embeddings safely
        clean_list = []
        for emb in line_embeddings_list:
            if isinstance(emb, list):
                emb = torch.tensor(emb, dtype=torch.float32)
            emb = emb.view(-1) if emb.ndim > 1 else emb
            clean_list.append(emb)

        lines_tensor = torch.stack(clean_list, dim=0)  # [num_lines, embed_dim]

        # Adaptive summary tokens
        num_summary_tokens = min(self.max_summary_tokens, lines_tensor.size(0))
        summary_tokens = torch.zeros(num_summary_tokens, self.embed_dim, device=lines_tensor.device)

        # Soft clustering init
        cluster_assign = torch.softmax(torch.rand(num_summary_tokens, lines_tensor.size(0)), dim=-1)
        summary_tokens = cluster_assign @ lines_tensor  # [summary_tokens, embed_dim]

        # Combine summary + lines
        seq = torch.cat([summary_tokens, lines_tensor], dim=0).unsqueeze(0)  # [1, num_summary + num_lines, embed_dim]
        seq = self.chunk_pos(seq)

        out = self.chunk_transformer(seq)
        pooled = self.norm_chunk(out[:, :num_summary_tokens, :].mean(dim=1))
        return pooled

# ======================================================
# 4. Load Model
# ======================================================
@st.cache_resource
def load_model():
    model = AdaptiveHierarchicalTransformer()
    model.eval()
    return model

model = load_model()

# ======================================================
# 5. Helper Functions
# ======================================================
ACTIVITY_LABELS = ["STARTUP","SHUTDOWN","CONNECTION_ERROR","AUTH_FAILURE","RETRY","TIMEOUT","CRASH_LOOP","DATA_PROCESSING"]

def encode_text(text, max_len=200):
    arr = [min(ord(c),255) for c in text[:max_len]]
    arr += [0]*(max_len-len(arr))
    return torch.tensor(arr).long().unsqueeze(0)

def classify_line(line):
    x = encode_text(line)
    with torch.no_grad():
        logits, emb = model.forward_line(x)
        activities = [ACTIVITY_LABELS[i] for i,val in enumerate(logits[0]) if val>0.5]
        anomaly = 1 - logits.max().item()
    return activities, anomaly, emb.squeeze(0)

def extract_basic_stats(text):
    lines = text.splitlines()
    return ([l for l in lines if "error" in l.lower()],
            [l for l in lines if "warn" in l.lower()],
            [l for l in lines if "info" in l.lower()])

def cluster_embeddings(embeddings, k=5):
    if len(embeddings)<k:
        k = max(1,len(embeddings)//2)
    X = np.stack([e.detach().numpy() for e in embeddings])
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(X)
    return labels

def generate_human_summary(chunk_embeddings, cluster_labels, activities_per_chunk):
    summary = []
    for idx, acts_chunk in enumerate(activities_per_chunk):
        all_acts = [act for line_acts in acts_chunk for act in line_acts]
        act_counts = Counter(all_acts)
        main_acts = [a.replace("_"," ").title() for a,count in act_counts.items() if count>1]
        phrase = f"Phase {idx+1}: " + (", then ".join(main_acts) if main_acts else "No dominant activity")
        summary.append(phrase)
    return "\n".join(summary)

# ======================================================
# 6. Streamlit UI
# ======================================================
st.title("🧠 Semantic Activity Log Analyzer v15")
st.caption("Fully adaptive transformer for any log size and pattern")

uploaded_file = st.file_uploader("Upload log file:", type=["txt","log","csv","json"])
if uploaded_file:
    raw_text = uploaded_file.read().decode(errors="ignore")
    st.subheader("📄 Raw Log Preview")
    st.code(raw_text[:2000])

    errors,warns,infos = extract_basic_stats(raw_text)
    st.subheader("📊 Log Stats")
    st.dataframe(pd.DataFrame({
        "Type":["Errors","Warnings","Info"],
        "Count":[len(errors),len(warns),len(infos)]
    }))

    lines = raw_text.splitlines()[:500]  # limit for demo
    line_embeddings, activities_per_line, anom_list = [], [], []

    for line in lines:
        acts, anom, emb = classify_line(line)
        line_embeddings.append(emb)
        activities_per_line.append(acts)
        anom_list.append(anom)

    anomaly_mean = np.mean(anom_list) if anom_list else 0
    st.subheader("🤖 Line-level Activities & Anomaly Scores")
    df = pd.DataFrame({
        "Line (short)":[l[:120] for l in lines],
        "Activities":[', '.join(a) if a else "NONE" for a in activities_per_line],
        "Anomaly":[round(a,4) for a in anom_list]
    })
    st.dataframe(df,use_container_width=True)

    # Process chunks adaptively
    chunk_size = 50
    chunk_embeddings, activities_per_chunk = [], []

    for i in range(0, len(line_embeddings), chunk_size):
        chunk_lines = line_embeddings[i:i+chunk_size]
        chunk_acts = activities_per_line[i:i+chunk_size]
        chunk_embed = model.forward_chunk_adaptive(chunk_lines)
        chunk_embeddings.append(chunk_embed.squeeze(0))
        activities_per_chunk.append(chunk_acts)

    cluster_labels = cluster_embeddings(chunk_embeddings)

    st.subheader("📝 Semantic Activity Overview")
    human_summary = generate_human_summary(chunk_embeddings, cluster_labels, activities_per_chunk)
    st.text(human_summary)

    # Visualization: Anomaly heatmap
    st.subheader("🔥 Anomaly Heatmap")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(anom_list, marker='o', linestyle='-', color='red')
    ax.set_xlabel("Log Line Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Line-wise Anomaly Trend")
    st.pyplot(fig)

    # Download report
    report = f"""
=== SEMANTIC INCIDENT REPORT v15 ===

File: {uploaded_file.name}

--- BASIC STATS ---
Errors: {len(errors)}
Warnings: {len(warns)}
Info: {len(infos)}
Average Anomaly: {anomaly_mean:.3f}

--- SEMANTIC ACTIVITY OVERVIEW ---
{human_summary}

--- RAW LOG SAMPLE ---
{raw_text[:2000]}
"""
    st.download_button("📥 Download Semantic Incident Report", report, "semantic_incident_report_v15.txt","text/plain")
