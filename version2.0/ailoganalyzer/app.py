
import os
import math
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ================================
# DARK MODE CONFIG
# ================================
st.set_page_config(page_title="Semantic Activity Log Analyzer v18", layout="wide")

DARK_CSS = """
<style>
body, .stApp {
    background-color: #0D0D0D !important;
    color: white !important;
}
input, textarea, select {
    background-color: #1E1E1E !important;
    color: white !important;
    border-radius: 8px;
}
div.stButton > button {
    background-color: #333333 !important;
    color: white !important;
    border: 1px solid #555555 !important;
    border-radius: 6px;
}
.stDataFrame, .stTable {
    color: white !important;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ================================
# SYSTEM CONTEXT
# ================================
def get_system_context():
    if not HAS_PSUTIL:
        return "psutil not available."
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        return f"CPU: {cpu:.1f}% | RAM: {ram:.1f}% | Disk: {disk:.1f}%"
    except Exception:
        return "System context unavailable."

st.sidebar.markdown("### System Status")
st.sidebar.text(get_system_context())

# ================================
# SIDEBAR CONFIG
# ================================
st.sidebar.markdown("### Configuration")
chunk_size = st.sidebar.slider("Chunk Size (lines)", 20, 200, 50, step=5)
max_lines = st.sidebar.slider("Max Lines to Process", 100, 10000, 1500, step=100)
threshold_strong = st.sidebar.slider("Strong Link Threshold", 0.50, 0.95, 0.75, 0.01)
threshold_weak = st.sidebar.slider("Weak Link Threshold", 0.05, 0.50, 0.30, 0.01)
view_mode = st.sidebar.selectbox("Topology View", ["CAR (Undirected)", "Cause–Effect (Directed)"])
use_faiss_neighbors = st.sidebar.checkbox("Use FAISS for fast similarity", value=HAS_FAISS)
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=True)
show_graph = st.sidebar.checkbox("Show Graph", value=True)

# ================================
# MODEL: FAHT
# ================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class AdaptiveHierarchicalTransformer(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=512, n_heads=16, line_layers=4, chunk_layers=2, max_summary_tokens=4, num_classes=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.line_pos = PositionalEncoding(embed_dim)
        self.line_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu"),
            num_layers=line_layers
        )
        self.chunk_pos = PositionalEncoding(embed_dim)
        self.chunk_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, batch_first=True, activation="gelu"),
            num_layers=chunk_layers
        )
        self.norm_chunk = nn.LayerNorm(embed_dim)
        self.fc_line = nn.Linear(embed_dim, num_classes)
        self.max_summary_tokens = max_summary_tokens

    def forward_line(self, x):
        emb = self.embed(x)
        emb = self.line_pos(emb)
        out = self.line_transformer(emb)
        pooled = out.mean(dim=1)
        logits = torch.sigmoid(self.fc_line(pooled))
        return logits, pooled

    def forward_chunk_adaptive(self, line_embeddings_list):
        if len(line_embeddings_list) == 0:
            return torch.zeros(1, self.embed_dim)
        lines_tensor = torch.stack(line_embeddings_list, dim=0)
        num_summary_tokens = min(self.max_summary_tokens, lines_tensor.size(0))
        cluster_assign = torch.softmax(torch.rand(num_summary_tokens, lines_tensor.size(0)), dim=-1)
        summary_tokens = cluster_assign @ lines_tensor
        seq = torch.cat([summary_tokens, lines_tensor], dim=0).unsqueeze(0)
        seq = self.chunk_pos(seq)
        out = self.chunk_transformer(seq)
        pooled = self.norm_chunk(out[:, :num_summary_tokens, :].mean(dim=1))
        return pooled

@st.cache_resource
def load_model():
    model = AdaptiveHierarchicalTransformer()
    model.eval()
    return model

model = load_model()

# ================================
# HELPERS
# ================================
ACTIVITY_LABELS = ["STARTUP","SHUTDOWN","CONNECTION_ERROR","AUTH_FAILURE","RETRY","TIMEOUT","CRASH_LOOP","DATA_PROCESSING"]

def encode_text(text, max_len=200):
    arr = [min(ord(c),255) for c in text[:max_len]]
    arr += [0]*(max_len-len(arr))
    return torch.tensor(arr).long().unsqueeze(0)

def classify_lines_batch(lines):
    embeddings, activities, anomalies = [], [], []
    for line in lines:
        x = encode_text(line)
        with torch.no_grad():
            logits, emb = model.forward_line(x)
        acts = [ACTIVITY_LABELS[i] for i,val in enumerate(logits[0]) if val>0.5]
        anomaly = 1 - logits.max().item()
        embeddings.append(emb.squeeze(0))
        activities.append(acts)
        anomalies.append(anomaly)
    return embeddings, activities, anomalies

@st.cache_data
def extract_keywords_tfidf_all(chunks, top_k=3):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    joined_chunks = [" ".join(chunk) for chunk in chunks]
    X = vectorizer.fit_transform(joined_chunks)
    terms = vectorizer.get_feature_names_out()
    keywords_per_chunk = []
    for row in X.toarray():
        top_indices = row.argsort()[::-1][:top_k]
        keywords_per_chunk.append([terms[i] for i in top_indices])
    return keywords_per_chunk

def shorten_label(s, max_len=22):
    return (s[:max_len] + "…") if len(s) > max_len else s

# ================================
# UI
# ================================
st.title("🧠 Semantic Activity Log Analyzer v18")
st.caption("FAHT + TF-IDF + CAR + Cause–Effect + FAISS (Optimized for Speed)")

uploaded_file = st.file_uploader("Upload log file:", type=["txt","log","csv","json"])

if uploaded_file:
    raw_text = uploaded_file.read().decode(errors="ignore")
    st.expander("📄 Raw Log Preview").code(raw_text[:2000])

    lines = [l for l in raw_text.splitlines() if l.strip()][:max_lines]
    st.sidebar.write(f"Processing {len(lines)} lines...")

    # Batch classify lines
    line_embeddings, activities_per_line, anom_list = classify_lines_batch(lines)
    anomaly_mean = np.mean(anom_list) if anom_list else 0

    st.subheader("📊 Line-level Activities & Anomaly Scores")
    st.dataframe(pd.DataFrame({
        "Line": [l[:120] for l in lines],
        "Activities": [', '.join(a) if a else "NONE" for a in activities_per_line],
        "Anomaly": [round(a,4) for a in anom_list]
    }), use_container_width=True)

    # Chunking
    chunk_embeddings, activities_per_chunk, chunk_raw_lines = [], [], []
    for i in range(0, len(line_embeddings), chunk_size):
        chunk_lines = line_embeddings[i:i+chunk_size]
        chunk_acts = activities_per_line[i:i+chunk_size]
        chunk_embed = model.forward_chunk_adaptive(chunk_lines)
        chunk_embeddings.append(chunk_embed.squeeze(0))
        activities_per_chunk.append(chunk_acts)
        chunk_raw_lines.append(lines[i:i+chunk_size])

    # TF-IDF keywords (single fit)
    keywords_all = extract_keywords_tfidf_all(chunk_raw_lines, top_k=3)
    kw_labels = [shorten_label(", ".join(kws[:2]) if kws else "Chunk", 18) for kws in keywords_all]

    # Similarity matrix
    chunk_embed_np = np.stack([e.detach().numpy() for e in chunk_embeddings]) if chunk_embeddings else np.zeros((0,512))
    sim_matrix = cosine_similarity(chunk_embed_np) if len(chunk_embed_np)>1 else np.eye(len(chunk_embed_np))

    # ================================
    # Graph Visualization with TF-IDF + Representative Line
    # ================================
    if show_graph and HAS_NETWORKX and len(kw_labels)>1:
        G = nx.Graph() if view_mode.startswith("CAR") else nx.DiGraph()
        node_labels = {}
        node_colors = []
        for idx, kws in enumerate(keywords_all):
            top_kw = ", ".join(kws[:2]) if kws else "Chunk"
            rep_line = chunk_raw_lines[idx][0][:50] if chunk_raw_lines[idx] else ""
            node_labels[idx] = f"{top_kw}\n{rep_line}"
            # Color intensity by anomaly severity
            anomaly_val = np.mean(anom_list[idx*chunk_size:idx*chunk_size+chunk_size])
            node_colors.append(anomaly_val)

        # Normalize colors
        norm_colors = [(c - min(node_colors)) / (max(node_colors)+1e-8) for c in node_colors]
        cmap = plt.cm.Reds

        if view_mode.startswith("CAR"):
            for i in range(len(sim_matrix)):
                for j in range(i+1, len(sim_matrix)):
                    sim = sim_matrix[i,j]
                    if sim>=threshold_strong or sim<=threshold_weak:
                        G.add_edge(i,j,type="STRONG" if sim>=threshold_strong else "WEAK",weight=sim)
        else:
            for i in range(len(sim_matrix)):
                for j in range(i+1, len(sim_matrix)):
                    sim = sim_matrix[i,j]
                    anomaly_diff = anom_list[j]-anom_list[i]
                    influence = sim*max(anomaly_diff,0)
                    if influence>0.05:
                        edge_type="STRONG" if sim>=threshold_strong else("WEAK" if sim<=threshold_weak else"MODERATE")
                        G.add_edge(i,j,type=edge_type,weight=influence)

        pos=nx.spring_layout(G,seed=42)
        fig,ax=plt.subplots(figsize=(12,8))
        nx.draw_networkx_nodes(G,pos,node_color=[cmap(c) for c in norm_colors],node_size=1200,ax=ax)
        nx.draw_networkx_edges(G,pos,edge_color="#32CD32",width=2,arrows=view_mode.startswith("Cause"),ax=ax)
        nx.draw_networkx_labels(G,pos,labels=node_labels,font_size=8,ax=ax)
        ax.set_title(f"{view_mode} (TF-IDF Keywords + Log Context)")
        ax.axis('off')
        st.pyplot(fig)

    # ================================
    # Heatmap
    # ================================
    if show_heatmap and sim_matrix.size>1:
        fig_hm, ax_hm = plt.subplots(figsize=(9, 7))
        cax = ax_hm.matshow(sim_matrix, cmap="coolwarm")
        fig_hm.colorbar(cax)
        ax_hm.set_title("Chunk Correlation Heatmap")
        ax_hm.set_xticks(range(len(kw_labels)))
        ax_hm.set_yticks(range(len(kw_labels)))
        ax_hm.set_xticklabels(kw_labels, rotation=90)
        ax_hm.set_yticklabels(kw_labels)
        plt.tight_layout()
        st.pyplot(fig_hm)

    # ================================
    # Anomaly Trend
    # ================================
    st.subheader("🔥 Anomaly Trend")
    fig_line, ax_line = plt.subplots(figsize=(12,4))
    ax_line.plot(anom_list,color='red')
    ax_line.set_title("Line-wise Anomaly Trend")
    st.pyplot(fig_line)
else:
    st.info("Upload a log file (.txt, .log, .csv, .json) to begin analysis.")
