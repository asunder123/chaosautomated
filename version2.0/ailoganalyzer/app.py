
# app.py — Semantic Activity Log Analyzer (single-file v24, with model persistence)
# Incremental analytics • Correlation + Patterns RCA • Learned attention pooler
# Self-optimizing batching • Robust cross-corr • Directed, readable topology

# ======================================================
# PART 0 — Imports
# ======================================================
import os
import contextlib
import hashlib
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# Optional Plotly
try:
    import plotly.graph_objs as go
    from plotly.colors import sample_colorscale
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ======================================================
# PART 1 — CORE (Model + Utilities)
# ======================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class PositionalEncoding(nn.Module):
    """Sinusoidal PE with safe dynamic extension when seq_len exceeds cached length."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        cached_len = self.pe.size(1)
        if T > cached_len:
            device = x.device
            pe = torch.zeros(T, D, device=device)
            pos = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
            div = torch.exp(torch.arange(0, D, 2, device=device).float() * (-np.log(10000.0)/D))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            pe = pe.unsqueeze(0)
        else:
            pe = self.pe[:, :T, :]
        return x + pe


class ContextAwareRouter(nn.Module):
    """Per-token importance scorer; routes tokens above threshold."""
    def __init__(self, embed_dim, threshold=0.25):
        super().__init__()
        self.scorer = nn.Linear(embed_dim, 1)
        self.threshold = threshold
    def forward(self, embeddings: torch.Tensor):
        scores = torch.sigmoid(self.scorer(embeddings))  # [T, 1]
        mask = scores.squeeze(-1) > self.threshold       # [T]
        routed = embeddings[mask] if mask.any() else embeddings
        return routed, mask


class LearnedAttentionPooler(nn.Module):
    """
    Content-aware attention pooling:
      summary = Attn(Q, K=seq, V=seq)
    init_mode='mean' -> data-conditioned queries (zero-shot friendly)
    init_mode='learned' -> trainable queries for fine-tuning
    """
    def __init__(self, embed_dim, num_summary_tokens=4, n_heads=8, init_mode="mean"):
        super().__init__()
        assert init_mode in ("mean", "learned")
        self.num_summary_tokens = num_summary_tokens
        self.init_mode = init_mode
        self.query_params = nn.Parameter(torch.randn(num_summary_tokens, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
    def _make_queries(self, seq):
        B, L, D = seq.shape
        if self.init_mode == "learned":
            return self.query_params.unsqueeze(0).expand(B, -1, -1)
        mu = seq.mean(dim=1, keepdim=True)
        base = torch.tanh(self.query_proj(mu))
        return base.expand(-1, self.num_summary_tokens, -1)
    def forward(self, seq):
        if seq.dim() == 2: seq = seq.unsqueeze(0)
        Q = self._make_queries(seq)
        summary, weights = self.attn(Q, seq, seq, need_weights=True)
        return self.norm(summary), weights  # [B,S,D], [B,S,L]


class AdaptiveHierarchicalTransformer(nn.Module):
    """
    Line-level transformer encodes lines.
    Chunk-level transformer aggregates with learned attention pooling + adaptive routing.
    """
    def __init__(self, vocab_size=256, embed_dim=512, n_heads=16,
                 line_layers=4, chunk_layers=2, max_summary_tokens=4,
                 router_threshold=0.25, pool_heads=8, pool_init_mode="mean"):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_summary_tokens = max_summary_tokens
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Line-level
        self.line_pos = PositionalEncoding(embed_dim)
        self.line_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, embed_dim*4, batch_first=True, activation="gelu"),
            num_layers=line_layers
        )

        # Chunk-level
        self.chunk_pos = PositionalEncoding(embed_dim)
        self.chunk_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, embed_dim*4, batch_first=True, activation="gelu"),
            num_layers=chunk_layers
        )

        self.norm_chunk = nn.LayerNorm(embed_dim)
        self.fc_line = nn.Linear(embed_dim, 8)  # 8 activity labels (proxy for anomaly)

        self.router = ContextAwareRouter(embed_dim, threshold=router_threshold)
        self.attn_pooler = LearnedAttentionPooler(embed_dim, max_summary_tokens, pool_heads, pool_init_mode)

    @torch.no_grad()
    def forward_line(self, x):
        emb = self.embed(x)
        emb = self.line_pos(emb)
        out = self.line_transformer(emb)
        pooled = out.mean(dim=1)
        logits = torch.sigmoid(self.fc_line(pooled))
        return logits, pooled

    @torch.no_grad()
    def forward_chunk_adaptive(self, line_embeddings_list):
        if len(line_embeddings_list) == 0:
            return torch.zeros(1, self.embed_dim), torch.zeros(self.max_summary_tokens, 0)
        lines_tensor = torch.stack(
            [(e if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32)).flatten()
             for e in line_embeddings_list], dim=0
        )  # [L, D]
        summary_tokens, weights = self.attn_pooler(lines_tensor)    # [1,S,D], [1,S,L]
        seq_concat = torch.cat([summary_tokens.squeeze(0), lines_tensor], dim=0)
        routed_tokens, _ = self.router(seq_concat)
        routed_tokens = self.chunk_pos(routed_tokens.unsqueeze(0))
        out = self.chunk_transformer(routed_tokens)
        pooled = self.norm_chunk(out.mean(dim=1))
        return pooled, weights.squeeze(0)


# ---------- Persistence-aware loader ----------
MODEL_PATH = "adaptive_transformer.pt"

@st.cache_resource
def get_model(router_threshold: float,
              pool_heads: int,
              pool_init_mode: str,
              max_summary_tokens: int,
              device: torch.device):
    """
    Load a persisted model if available; otherwise initialize, optionally compile, persist, and return.
    """
    model = AdaptiveHierarchicalTransformer(
        router_threshold=router_threshold,
        pool_heads=pool_heads,
        pool_init_mode=pool_init_mode,
        max_summary_tokens=max_summary_tokens
    )

    # Load persisted weights if present
    if os.path.exists(MODEL_PATH):
        with contextlib.suppress(Exception):
            sd = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(sd)
            print("✅ Loaded model from disk for faster startup.")

    # Optional compile (PyTorch 2.x)
    with contextlib.suppress(Exception):
        model = torch.compile(model)

    model.eval()
    model.to(device)

    # Save weights (first run, or after init)
    if not os.path.exists(MODEL_PATH):
        with contextlib.suppress(Exception):
            torch.save(model.state_dict(), MODEL_PATH)
            print("💾 Model initialized and saved for reuse.")

    return model


# ---------- I/O and batching helpers ----------
def safe_decode_uploaded(uploaded_file, max_bytes=10_000_000):
    raw = uploaded_file.getvalue()
    if len(raw) > max_bytes: raw = raw[:max_bytes]
    for enc in ("utf-8","utf-16","latin-1"):
        try: return raw.decode(enc)
        except Exception: continue
    return raw.decode("utf-8", errors="ignore")

def encode_texts(texts, max_len=200, device=None):
    arrs = []
    for t in texts:
        cut = t[:max_len]
        a = [min(ord(c), 255) for c in cut] + [0]*(max_len - len(cut))
        arrs.append(a)
    ten = torch.tensor(arrs).long()
    return ten.to(device) if device else ten

def classify_lines_batch(model, device, lines, max_len=200, batch_size=128, use_amp=True):
    acts_all, anoms_all, embeds_all = [], [], []
    labels = ["STARTUP","SHUTDOWN","CONNECTION_ERROR","AUTH_FAILURE","RETRY","TIMEOUT","CRASH_LOOP","DATA_PROCESSING"]
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type=="cuda") else contextlib.nullcontext
    with torch.no_grad(), amp_ctx():
        for i in range(0, len(lines), batch_size):
            x = encode_texts(lines[i:i+batch_size], max_len, device)
            logits, emb = model.forward_line(x)
            probs = logits.detach().cpu().numpy()
            for bi, p in enumerate(probs):
                acts_all.append([labels[j] for j, v in enumerate(p) if v > 0.5])
                anoms_all.append(float(1.0 - p.max()))
                embeds_all.append(emb[bi].detach().cpu())
    return acts_all, anoms_all, embeds_all

def gpu_memory_free_gb(device):
    if device.type!="cuda": return 0.0
    with contextlib.suppress(Exception):
        free,_ = torch.cuda.mem_get_info()
        return free/(1024**3)
    return 0.0

def auto_choose_max_len(lines, hard_cap=512, base_min=64, p=95) -> int:
    if not lines: return base_min
    lens = np.array([len(l) for l in lines])
    q = int(np.clip(np.percentile(lens, p), base_min, hard_cap))
    return int(q)

def _token_budget_per_batch(device, max_len, gpu_free_gb):
    if device.type=="cuda":
        if gpu_free_gb >= 12:   budget = 256_000
        elif gpu_free_gb >= 8:  budget = 192_000
        elif gpu_free_gb >= 4:  budget = 128_000
        else:                   budget = 96_000
    else:
        budget = 64_000
    return max(budget, 16 * max_len)

def auto_choose_batch_size(device, max_len, gpu_free_gb):
    return int(np.clip(_token_budget_per_batch(device, max_len, gpu_free_gb)//max_len, 16, 256))

def auto_choose_chunk_size(num_lines, min_c=20, max_c=500, target_chunks=(12, 40)):
    if num_lines <= min_c: return max(10, num_lines)
    lo, hi = target_chunks
    size = int(np.clip(np.ceil(num_lines / ((lo + hi) / 2.0)), min_c, max_c))
    return int(np.clip(int(np.round(size / 10) * 10), min_c, max_c))

def micro_benchmark_batch(sample_lines, max_len, base_bs, classify_fn, amp=True):
    candidates = [base_bs] + ([min(256, base_bs*2)] if base_bs < 256 else [])
    best_bs, best_t = base_bs, float("inf")
    for bs in candidates:
        try:
            t0 = time.perf_counter()
            classify_fn(sample_lines, max_len=max_len, batch_size=bs, use_amp=amp)
            dt = time.perf_counter() - t0
            if dt < best_t:
                best_t, best_bs = dt, bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                best_bs = max(16, bs//2)
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
        except Exception:
            pass
    return best_bs

# ======================================================
# PART 2 — ANALYTICS (Temporal + Correlation/RCA + Pattern Miner)
# ======================================================
TIMESTAMP_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?",
    r"\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s[+\-]\d{4}",
    r"\w{3}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}",
    r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}",
    r"\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}",
]

def first_ts_match(line: str, extra_regex: str | None = None):
    if extra_regex:
        m = re.search(extra_regex, line); 
        if m: return m.group(0)
    for pat in TIMESTAMP_PATTERNS:
        m = re.search(pat, line); 
        if m: return m.group(0)
    m = re.search(r'"(?:@?timestamp|time|date|ts)"\s*:\s*"(.*?)"', line)
    if m: return m.group(1)
    return None

def severity_of(line: str) -> str:
    s = line.lower()
    if "error" in s: return "ERROR"
    if "warn"  in s: return "WARN"
    if "info"  in s: return "INFO"
    return "OTHER"

def append_timeline_rows(lines_segment, acts_segment, anoms_segment, base_idx, extra_regex=None):
    rows = []
    for local_i, line in enumerate(lines_segment):
        ts_text = first_ts_match(line, extra_regex)
        ts = pd.to_datetime(ts_text, errors="coerce", utc=True) if ts_text else pd.NaT
        act = acts_segment[local_i][0] if acts_segment[local_i] else "NONE"
        sev = severity_of(line)
        rows.append((base_idx + local_i, ts, line, sev, act, anoms_segment[local_i]))
    return rows

def resample_metrics(df_ts: pd.DataFrame, freq: str = "1min"):
    d = df_ts.dropna(subset=["ts"]).copy()
    if d.empty:
        ix = pd.to_datetime([])
        empty = pd.DataFrame(index=ix)
        return empty, empty, empty, empty
    d = d.set_index("ts")
    vol  = d["line"].resample(freq).count().rename("count").to_frame()
    anom = d["anomaly"].resample(freq).mean().rename("anomaly_mean").to_frame()
    sev  = (pd.get_dummies(d["severity"]).resample(freq).sum()
            .rename_axis("ts").reset_index().set_index("ts"))
    acts = (pd.get_dummies(d["activity"]).resample(freq).sum()
            .rename_axis("ts").reset_index().set_index("ts"))
    return vol, anom, sev, acts

# ---- Correlation primitives ----
def cosine_corr_matrix(embeds_np, norms_cache=None):
    """Fast cosine similarity with cached norms (recomputed when C changes)."""
    if norms_cache is None:
        norms = np.linalg.norm(embeds_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
    else:
        norms = norms_cache
    sim = (embeds_np @ embeds_np.T) / (norms @ norms.T)
    return np.clip(sim, -1.0, 1.0), norms

def cross_corr_lags(a, b, max_lag=3, min_overlap=3):
    """Robust short normalized cross-correlation with length equalization at EVERY lag."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b))
    if n < min_overlap: return 0, 0.0
    a_n = (a - a.mean()) / (a.std() + 1e-8)
    b_n = (b - b.mean()) / (b.std() + 1e-8)
    best_lag, best_score = 0, -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a_n[-lag:]; m = min(len(aa), len(b_n))
            if m < min_overlap: continue
            aa = aa[:m]; bb = b_n[:m]
        elif lag > 0:
            bb = b_n[lag:]; m = min(len(bb), len(a_n))
            if m < min_overlap: continue
            bb = bb[:m]; aa = a_n[:m]
        else:
            m = min(len(a_n), len(b_n))
            if m < min_overlap: continue
            aa = a_n[-m:]; bb = b_n[-m:]
        score = float(np.mean(aa * bb))
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag, best_score

# ---- Dynamic Pattern Miner ----
TOKEN_RE = re.compile(r"[A-Za-z0-9\.\-_:/]+")
EXC_RE   = re.compile(r"\b([A-Za-z0-9_.]+Exception)\b")
HTTP_RE  = re.compile(r"\b([45]\d{2})\b")
ERR_RE   = re.compile(r"\bERR[ _\-]?\d+\b", re.IGNORECASE)

def normalize_line_for_patterns(line: str) -> str:
    # remove timestamps heuristically
    for pat in TIMESTAMP_PATTERNS:
        line = re.sub(pat, " ", line)
    # collapse numbers & IPs
    line = re.sub(r"\b\d+\b", "<NUM>", line)
    line = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<IP>", line)
    return line

def tokenize(line: str): return [t.lower() for t in TOKEN_RE.findall(line)]
def ngrams(tokens, n=2): return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def extract_chunk_patterns(lines_chunk, top_k=10):
    uni = Counter(); bi = Counter(); tri = Counter()
    exc = Counter(); http = Counter(); errc = Counter()

    for line in lines_chunk:
        norm = normalize_line_for_patterns(line)
        tokens = tokenize(norm)
        if tokens:
            uni.update(tokens); bi.update(ngrams(tokens, 2)); tri.update(ngrams(tokens, 3))
        for m in EXC_RE.findall(line):   exc[m] += 1
        for m in HTTP_RE.findall(line):  http[m] += 1
        for m in ERR_RE.findall(line):   errc[m.upper()] += 1

    def top(counter, k): return [t for t,_ in counter.most_common(k)]
    card = {
        "unigrams": top(uni, top_k),
        "bigrams":  top(bi, top_k),
        "trigrams": top(tri, top_k),
        "exceptions": top(exc, top_k),
        "http": top(http, top_k),
        "errors": top(errc, top_k),
    }
    # canonical set used for Jaccard overlap
    pattern_set = set(card["unigrams"] + card["bigrams"] + card["trigrams"] +
                      card["exceptions"] + card["http"] + card["errors"])
    return card, pattern_set

def jaccard_overlap(set_a: set, set_b: set) -> float:
    if not set_a or not set_b: return 0.0
    inter = len(set_a & set_b); union = len(set_a | set_b)
    return 0.0 if union == 0 else inter / union

# ---- Cause score with pattern fusion ----
def cause_score(sim, ai, aj, lag_score, lag, patt_sim, w_sim=0.50, w_grad=0.25, w_lag=0.15, w_patt=0.10):
    grad = max(aj - ai, 0.0)
    lag_bonus = lag_score * (1.0 if lag > 0 else 0.7 if lag == 0 else 0.5)
    return w_sim*sim + w_grad*grad + w_lag*lag_bonus + w_patt*patt_sim

def build_topology(chunk_embeds, all_anoms, acts_per_chunk, chunk_size,
                   sim_matrix, base_topk, sim_threshold, infl_threshold,
                   anom_windows, max_lag,
                   pattern_sets):
    """
    Build directed graph (i -> j) using correlation-first cause score + pattern overlap.
    """
    G = nx.DiGraph()
    C = len(chunk_embeds)
    G.add_nodes_from(range(C))

    # Nodes
    node_labels, node_anomaly, node_sizes = {}, [], []
    for idx in range(C):
        s, e = idx * chunk_size, min((idx + 1) * chunk_size, len(all_anoms))
        anomaly_val = float(np.mean(all_anoms[s:e])) if e > s else 0.0
        all_acts = [a for line_acts in acts_per_chunk[idx] for a in line_acts]
        top_acts = ", ".join([a.replace("_", " ").title() for a,_ in Counter(all_acts).most_common(3)]) or "No dominant activity"
        node_labels[idx] = f"{top_acts}\nAnomaly: {anomaly_val:.3f}"
        node_anomaly.append(anomaly_val)
        node_sizes.append(800 + anomaly_val * 1200)

    # Edges
    for i in range(C):
        ai = node_anomaly[i]
        wi = np.asarray(anom_windows[i], float)
        candidates = []

        for j in range(C):
            if i == j: continue
            sim = float(sim_matrix[i, j])
            if sim < sim_threshold: continue
            aj = node_anomaly[j]
            wj = np.asarray(anom_windows[j], float)

            # Lag correlation guard
            if wi.size < 3 or wj.size < 3:
                lag, lag_score = 0, 0.0
            else:
                lag, lag_score = cross_corr_lags(wi, wj, max_lag=max_lag)

            # Pattern overlap (Jaccard)
            patt_sim = jaccard_overlap(pattern_sets[i], pattern_sets[j])

            score = cause_score(sim, ai, aj, lag_score, lag, patt_sim)
            infl  = max(0.0, sim * max(aj - ai, 0.0))

            if score >= infl_threshold:
                candidates.append((j, score, infl, sim, lag, lag_score, patt_sim))

        candidates.sort(key=lambda t: t[1], reverse=True)
        for j, score, infl, sim, lag, lag_score, patt_sim in candidates[:base_topk]:
            G.add_edge(i, j,
                       score=score, weight=max(0.01, infl),
                       sim=sim, lag=lag, lag_score=lag_score, patt=patt_sim)
    return G, node_labels, node_anomaly, node_sizes

# ======================================================
# PART 3 — STREAMLIT APP (Incremental UI + Model Persistence)
# ======================================================
st.set_page_config(page_title="Semantic Activity Log Analyzer (single-file)", layout="wide")
st.title("🧠 Semantic Activity Log Analyzer — Single-file v24 (Model persistence + Dynamic patterns)")
st.caption("Incremental analytics • Correlation + Patterns RCA • Learned attention pooler • Self‑optimizing batching • Persisted model")

with st.sidebar:
    st.header("⚙️ Controls")
    auto_opt = st.checkbox("Auto Optimize (recommended)", value=True)
    max_lines_manual = st.slider("Max lines to analyze", 100, 10000, 1500, step=100, disabled=auto_opt)
    batch_size_manual = st.slider("Batch size", 16, 256, 128, step=16, disabled=auto_opt)
    max_len_manual = st.slider("Max tokens per line", 64, 512, 200, step=16, disabled=auto_opt)
    chunk_size_manual = st.slider("Chunk size (lines per chunk)", 20, 500, 60, step=10, disabled=auto_opt)
    use_amp_manual = st.checkbox("Use mixed precision (GPU only)", value=True, disabled=auto_opt)
    st.markdown("---")
    patt_topk = st.slider("Top patterns per chunk", 5, 20, 10, step=1)
    lag_max   = st.slider("Cross-corr max lag (chunks)", 0, 6, 3, step=1)
    st.markdown("---")
    # Persistence actions
    save_now = st.button("💾 Save current model to disk")
    reset_model = st.button("♻️ Reset persisted model (delete file)")

uploaded_file = st.file_uploader("Upload log file:", type=["txt", "log", "csv", "json"])

# Device & model (persisted)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(router_threshold=0.25,
                  pool_heads=8,
                  pool_init_mode="mean",
                  max_summary_tokens=4,
                  device=device)

# Optional persistence actions
if reset_model:
    with contextlib.suppress(Exception):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            st.info("Persisted model file deleted. It will be re-initialized on next run or next cache refresh.")
if save_now:
    with contextlib.suppress(Exception):
        torch.save(model.state_dict(), MODEL_PATH)
        st.success("Model state saved to disk.")

if uploaded_file:
    raw_text = safe_decode_uploaded(uploaded_file)
    st.subheader("📄 Raw Log Preview")
    st.code(raw_text[:2000])

    all_lines = raw_text.splitlines()
    n_total = len(all_lines)

    # --- Auto params ---
    if auto_opt:
        free_gb = gpu_memory_free_gb(device)
        if n_total <= 1500:
            max_lines = n_total
        elif n_total <= 5000:
            max_lines = 2500
        else:
            max_lines = 5000

        max_len = auto_choose_max_len(all_lines[:max_lines], hard_cap=512, base_min=64, p=95)
        batch_sz = auto_choose_batch_size(device, max_len, free_gb)
        use_amp  = (device.type == "cuda")
        bench_sample = all_lines[:min(256, max_lines)]
        batch_sz = micro_benchmark_batch(
            bench_sample, max_len, batch_sz,
            classify_fn=lambda lines, **kw: classify_lines_batch(model, device, lines, **kw),
            amp=use_amp
        )
        chunk_size = auto_choose_chunk_size(max_lines, min_c=20, max_c=500, target_chunks=(12, 40))
        approx_chunks = max(1, int(np.ceil(max_lines / chunk_size)))
        base_topk = 2 if approx_chunks > 80 else 3 if approx_chunks > 40 else 4
        sim_threshold = 0.35 if approx_chunks <= 60 else 0.45
        infl_threshold = 0.02 if approx_chunks <= 60 else 0.03
        plotly_ok = PLOTLY_AVAILABLE and (approx_chunks <= 120)
    else:
        max_lines = max_lines_manual
        max_len   = max_len_manual
        batch_sz  = batch_size_manual
        chunk_size = chunk_size_manual
        use_amp    = use_amp_manual
        base_topk = 3
        sim_threshold = 0.35
        infl_threshold = 0.02
        plotly_ok = PLOTLY_AVAILABLE

    with st.expander("🔧 Effective Parameters", expanded=True):
        st.write({
            "device": device.type,
            "max_lines": int(max_lines),
            "batch_size": int(batch_sz),
            "max_len": int(max_len),
            "chunk_size": int(chunk_size),
            "router_threshold": 0.25,
            "pooler": f"mean, heads=8, S=4",
            "use_amp": bool(use_amp),
            "sim_threshold": float(sim_threshold),
            "influence_threshold": float(infl_threshold),
            "base_topk": int(base_topk),
            "plotly_enabled": bool(plotly_ok),
            "patterns_topk": int(patt_topk),
            "max_lag": int(lag_max),
            "model_path": MODEL_PATH,
            "model_cached": os.path.exists(MODEL_PATH),
        })

    # --- Placeholders ---
    progress_ph     = st.progress(0)
    anomaly_plot_ph = st.empty()
    patterns_ph     = st.expander("🧩 Pattern Cards (per chunk)", expanded=False)
    rca_ph          = st.empty()

    # --- Accumulators ---
    lines = all_lines[:max_lines]
    all_anoms = []
    chunk_embeds_acc, acts_per_chunk_acc = [], []
    pattern_cards, pattern_sets = [], []
    norms_cache = None
    total_chunks = max(1, int(np.ceil(len(lines) / chunk_size)))
    redraw_every = max(1, total_chunks // 10)

    anomaly_x, anomaly_y = [], []

    for ci in range(total_chunks):
        s, e = ci * chunk_size, min((ci + 1) * chunk_size, len(lines))
        lines_chunk = lines[s:e]

        # 1) Classify this chunk
        acts, anoms, embeds = classify_lines_batch(model, device, lines_chunk,
                                                   max_len=max_len, batch_size=batch_sz, use_amp=use_amp)
        all_anoms.extend(anoms)

        # 2) Chunk embed
        pooled, _ = model.forward_chunk_adaptive(embeds)
        chunk_embeds_acc.append(pooled.squeeze(0).cpu())
        acts_per_chunk_acc.append(acts)

        # 3) Dynamic pattern mining for this chunk
        card, patt_set = extract_chunk_patterns(lines_chunk, top_k=patt_topk)
        pattern_cards.append(card)
        pattern_sets.append(patt_set)

        # 4) Update Pattern Cards panel
        with patterns_ph:
            st.markdown(f"**Chunk {ci+1}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Top Unigrams:", ", ".join(card["unigrams"]) or "—")
                st.write("Top Bigrams:", ", ".join(card["bigrams"]) or "—")
            with col2:
                st.write("Top Trigrams:", ", ".join(card["trigrams"]) or "—")
                st.write("Exceptions:", ", ".join(card["exceptions"]) or "—")
            with col3:
                st.write("HTTP Codes:", ", ".join(card["http"]) or "—")
                st.write("Error Codes:", ", ".join(card["errors"]) or "—")

        # 5) Anomaly trend (partial)
        anomaly_x.extend(range(s, e))
        anomaly_y.extend(anoms)
        fig_tr, ax_tr = plt.subplots(figsize=(12, 3.6))
        ax_tr.plot(anomaly_x, anomaly_y, marker='o', linestyle='-', color='red', markersize=2)
        ax_tr.set_xlabel("Log Line Index"); ax_tr.set_ylabel("Anomaly Score")
        ax_tr.set_title("Line‑wise Anomaly Trend (partial)")
        anomaly_plot_ph.pyplot(fig_tr)

        # 6) RCA / topology (partial cadence)
        if (ci % redraw_every == 0) or (ci == total_chunks - 1):
            C = len(chunk_embeds_acc)
            if C > 1:
                chunk_np = np.stack([e.numpy() for e in chunk_embeds_acc])  # [C, D]

                # Norms cache reset if C changed
                if (norms_cache is None) or (norms_cache.shape[0] != chunk_np.shape[0]):
                    norms_cache = None
                sim_matrix, norms_cache = cosine_corr_matrix(chunk_np, norms_cache)

                anom_windows = [
                    np.array(all_anoms[i*chunk_size : min((i+1)*chunk_size, len(all_anoms))], dtype=float)
                    for i in range(C)
                ]

                G, node_labels, node_anomaly, node_sizes = build_topology(
                    chunk_embeds_acc, all_anoms, acts_per_chunk_acc, chunk_size,
                    sim_matrix, base_topk, sim_threshold, infl_threshold,
                    anom_windows, max_lag=lag_max,
                    pattern_sets=pattern_sets
                )

                if G.number_of_edges() > 0:
                    with contextlib.suppress(Exception):
                        pos = nx.spring_layout(G, seed=SEED, k=0.6, iterations=200)
                    if 'pos' not in locals():
                        pos = nx.circular_layout(G)

                    # --- Render RCA Topology with clear cause → effect ---
                    an_min, an_max = float(min(node_anomaly)), float(max(node_anomaly))
                    if an_max - an_min < 1e-8: an_max = an_min + 1e-8
                    def norm_color(a): return (a - an_min) / (an_max - an_min)

                    if plotly_ok:
                        edge_x, edge_y, edge_text = [], [], []
                        for u, v, data in G.edges(data=True):
                            x0, y0 = pos[u]; x1, y1 = pos[v]
                            edge_x += [x0, x1, None]
                            edge_y += [y0, y1, None]
                            edge_text.append(
                                f"Cause → Effect<br>"
                                f"From: Chunk {u+1}<br>To: Chunk {v+1}<br>"
                                f"Score: {data.get('score',0):.3f}<br>"
                                f"Similarity: {data.get('sim',0):.2f}<br>"
                                f"Lag: {data.get('lag',0)}<br>"
                                f"Pattern overlap: {data.get('patt',0):.2f}"
                            )

                        node_x, node_y, node_text, node_color_vals, node_sizes_plotly = [], [], [], [], []
                        for n in G.nodes():
                            x, y = pos[n]
                            node_x.append(x); node_y.append(y)
                            node_text.append(node_labels[n])
                            node_color_vals.append(norm_color(node_anomaly[n]))
                            node_sizes_plotly.append(14 + node_anomaly[n] * 20)

                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=2, color='rgba(50,205,50,0.6)'),
                            hoverinfo='text', text=edge_text
                        )
                        node_colors = sample_colorscale('Reds', node_color_vals)
                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers+text',
                            text=node_text, textposition='top center',
                            hoverinfo='text',
                            marker=dict(color=node_colors, size=node_sizes_plotly,
                                        line=dict(color='black', width=0.5))
                        )

                        fig = go.Figure(data=[edge_trace, node_trace],
                                        layout=go.Layout(
                                            title="RCA Topology — Cause → Effect (correlation + patterns)",
                                            showlegend=False, height=620,
                                            margin=dict(l=10, r=10, t=50, b=10),
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                        ))
                        rca_ph.plotly_chart(fig, use_container_width=True)

                    else:
                        fig2, ax2 = plt.subplots(figsize=(12, 7))
                        cmap = plt.cm.Reds
                        norm_colors = [(a - an_min) / (an_max - an_min) for a in node_anomaly]
                        nx.draw_networkx_nodes(G, pos, node_color=[cmap(c) for c in norm_colors],
                                               node_size=node_sizes, ax=ax2)
                        edge_widths = [G[u][v]['weight'] * 6 for u, v in G.edges()]
                        nx.draw_networkx_edges(G, pos, edge_color="#32CD32", width=edge_widths,
                                               arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax2)
                        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_color='black', ax=ax2)
                        edge_labels = {(u,v): f"{data['score']:.2f} / patt {data.get('patt',0):.2f}"
                                       for u,v,data in G.edges(data=True)}
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax2)
                        ax2.set_title("RCA Topology — Cause → Effect (correlation + patterns)")
                        ax2.axis('off')
                        rca_ph.pyplot(fig2)

        progress_ph.progress(int(((ci + 1) / total_chunks) * 100))

    st.success("✅ Analysis complete — model persisted, dynamic patterns integrated, robust RCA topology!")
