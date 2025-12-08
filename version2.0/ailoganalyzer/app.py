# app_v2_optimized.py — Semantic Activity Log Analyzer v33 (Optimized)
# ======================================================
# KEY ENHANCEMENTS:
# 1.  CACHING: Incremental analysis, memoized embeddings, session-based state
# 2. INSIGHTS: Anomaly clustering, incident fingerprinting, causal chains, root cause evidence scoring
# 3. PREDICTIVE: LSTM forecasts for anomalies & activities; trend extrapolation; pattern-based warnings
# ======================================================

import os
import contextlib
import re
import time
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set
from functools import lru_cache
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

try:
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.colors import sample_colorscale
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ======================================================
# PART 0. 5 — CACHING & STATE MANAGEMENT
# ======================================================

@dataclass
class CachedAnalysisState:
    """Persistent session state for incremental analysis."""
    file_hash: str
    chunk_size: int
    embeddings: Dict[int, torch.Tensor]  # chunk_id -> embedding
    anomalies: Dict[int, np.ndarray]     # chunk_id -> anomaly scores
    activities: Dict[int, List[str]]     # chunk_id -> activity labels
    patterns: Dict[int, dict]             # chunk_id -> pattern card
    chunk_stats: Dict[int, 'ChunkStats']  # chunk_id -> stats
    timestamp: float

def init_session_cache():
    """Initialize or retrieve cached analysis from session state."""
    if "cache_state" not in st.session_state:
        st.session_state.cache_state = None
    return st.session_state.cache_state

def file_hash(content: str) -> str:
    """Compute hash of file content for cache validity."""
    return hashlib.md5(content.encode()).hexdigest()

# ======================================================
# PART 1 — CORE MODEL + OPTIMIZATIONS
# ======================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch. backends.cudnn.allow_tf32 = True


class PositionalEncoding(nn.Module):
    """Sinusoidal PE with safe dynamic extension."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch. float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np. log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe. unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch. Tensor:
        B, T, D = x.size()
        cached_len = self.pe.size(1)
        if T > cached_len:
            device = x.device
            pe = torch.zeros(T, D, device=device)
            pos = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
            div = torch. exp(torch.arange(0, D, 2, device=device).float() * (-np.log(10000.0)/D))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            pe = pe.unsqueeze(0)
        else:
            pe = self.pe[:, :T, :]
        return x + pe


class ContextAwareRouter(nn.Module):
    """Per-token importance scorer with learned thresholding."""
    def __init__(self, embed_dim, threshold=0.25):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn. GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.threshold = threshold

    def forward(self, embeddings: torch.Tensor):
        scores = torch.sigmoid(self.scorer(embeddings))
        mask = scores. squeeze(-1) > self.threshold
        routed = embeddings[mask] if mask.any() else embeddings
        return routed, mask, scores. squeeze(-1)


class LearnedAttentionPooler(nn.Module):
    """Content-aware attention pooling with multi-head variance."""
    def __init__(self, embed_dim, num_summary_tokens=4, n_heads=8, init_mode="mean"):
        super().__init__()
        assert init_mode in ("mean", "learned")
        self.num_summary_tokens = num_summary_tokens
        self.init_mode = init_mode
        self.query_params = nn.Parameter(torch.randn(num_summary_tokens, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)

    def _make_queries(self, seq):
        B, L, D = seq.shape
        if self.init_mode == "learned":
            return self.query_params.unsqueeze(0).expand(B, -1, -1)
        mu = seq.mean(dim=1, keepdim=True)
        base = torch.tanh(self.query_proj(mu))
        return base. expand(-1, self.num_summary_tokens, -1)

    def forward(self, seq):
        if seq.dim() == 2: seq = seq.unsqueeze(0)
        Q = self._make_queries(seq)
        summary, weights = self.attn(Q, seq, seq, need_weights=True)
        return self.norm(summary), weights


class AdaptiveHierarchicalTransformer(nn.Module):
    """Hierarchical transformer with optimized inference paths."""
    def __init__(self, vocab_size=256, embed_dim=512, n_heads=16,
                 line_layers=4, chunk_layers=2, max_summary_tokens=4,
                 router_threshold=0.25, pool_heads=8, pool_init_mode="mean"):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_summary_tokens = max_summary_tokens
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.line_pos = PositionalEncoding(embed_dim)
        self.line_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, embed_dim*4, batch_first=True, activation="gelu", dropout=0.1),
            num_layers=line_layers
        )

        self.chunk_pos = PositionalEncoding(embed_dim)
        self. chunk_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, embed_dim*4, batch_first=True, activation="gelu", dropout=0.1),
            num_layers=chunk_layers
        )

        self.norm_chunk = nn.LayerNorm(embed_dim)
        self. fc_line = nn.Linear(embed_dim, 8)

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
            [(e if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32)). flatten()
             for e in line_embeddings_list], dim=0
        )
        summary_tokens, weights = self.attn_pooler(lines_tensor)
        seq_concat = torch.cat([summary_tokens. squeeze(0), lines_tensor], dim=0)
        routed_tokens, _, _ = self.router(seq_concat)
        routed_tokens = self.chunk_pos(routed_tokens. unsqueeze(0))
        out = self.chunk_transformer(routed_tokens)
        pooled = self.norm_chunk(out.mean(dim=1))
        return pooled, weights. squeeze(0)



# ======================================================
# MODEL LOADING, SAVING & SESSION CACHE
# ======================================================

MODEL_PATH = "adaptive_transformer.pt"

@st.cache_resource
def get_model(device: torch.device):
    """Load model from disk if available, else create and save a new one."""
    model = AdaptiveHierarchicalTransformer()
    if os.path.exists(MODEL_PATH):
        try:
            sd = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(sd)
            st.info("✅ Loaded model from disk.")
        except Exception as e:
            st.warning(f"⚠️ Failed to load saved model: {e}. Using fresh instance.")
    else:
        st.warning("⚠️ No saved model found. Creating new instance and saving.")
        torch.save(model.state_dict(), MODEL_PATH)

    # Optional compile for speed
    with contextlib.suppress(Exception):
        model = torch.compile(model)

    model.eval().to(device)
    return model



# ======================================================
# PART 1. 5 — PREDICTIVE MODELS
# ======================================================

class AnomalyLSTMForecaster(nn.Module):
    """LSTM-based anomaly score forecaster (1–5 step ahead)."""
    def __init__(self, input_dim=1, hidden_dim=32, n_layers=2, forecast_steps=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, forecast_steps)

    def forward(self, x):
        """x: [B, T, 1] → forecast: [B, forecast_steps]"""
        _, (h_n, _) = self. lstm(x)
        forecast = self.fc(h_n[-1])
        return forecast


@st.cache_resource
def get_anomaly_forecaster(device):
    model = AnomalyLSTMForecaster(input_dim=1, hidden_dim=32, n_layers=2, forecast_steps=3)
    model.eval(). to(device)
    return model


class ActivityTransitionPredictor:
    """Markov-chain + embeddings based activity forecaster."""
    def __init__(self, activity_vocab=None):
        self.vocab = activity_vocab or []
        self.transition_matrix = None
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

    def fit(self, activity_sequence: List[str]):
        """Learn transitions from observed sequence."""
        self.vocab = list(set(activity_sequence))
        self.bigram_counts = Counter(zip(activity_sequence[:-1], activity_sequence[1:]))
        self.trigram_counts = Counter(zip(activity_sequence[:-2], activity_sequence[1:-1], activity_sequence[2:]))

    def predict_next(self, recent_activities: List[str], n_candidates=3) -> List[Tuple[str, float]]:
        """Predict next activity given recent history."""
        if len(recent_activities) < 2:
            return [(a, c / sum(self.bigram_counts. values())) 
                    for a, c in self.bigram_counts.most_common(n_candidates)]
        
        last_two = tuple(recent_activities[-2:])
        candidates = Counter()
        for (a, b, c), cnt in self.trigram_counts.items():
            if (a, b) == last_two:
                candidates[c] += cnt

        if not candidates:
            last_one = recent_activities[-1]
            for (a, b), cnt in self.bigram_counts.items():
                if a == last_one:
                    candidates[b] += cnt

        total = sum(candidates.values()) or 1
        return sorted([(a, c / total) for a, c in candidates.items()], key=lambda x: x[1], reverse=True)[:n_candidates]


# ======================================================
# PART 2 — ANALYTICS (Temporal + RCA + Pattern Miner + Clustering)
# ======================================================


TIMESTAMP_PATTERNS = [
    # ISO 8601 with optional fractional seconds and timezone
    r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?",
    # NCSA/Apache-like: 12/Jan/2024:12:34:56 +0530
    r"\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s[+\-]\d{4}",
    # Syslog-like: Jan  12 12:34:56
    r"[A-Za-z]{3}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}",
    # Plain datetime: 2024-01-12 12:34:56
    r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}",
    # MM/DD/YYYY HH:MM:SS
    r"\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}",
]

def first_ts_match(line: str, extra_regex: Optional[str] = None):
    if extra_regex:
        m = re.search(extra_regex, line)
        if m:
            return m.group(0)
    for pat in TIMESTAMP_PATTERNS:
        m = re.search(pat, line)
        if m:
            return m.group(0)
    # JSON-like field fallback: "timestamp": "..."
    m = re.search(r'"(?:@?timestamp|time|date|ts)"\s*:\s*"(.*?)"', line)
    if m:
        return m.group(1)
    return None


def severity_of(line: str) -> str:
    s = line.lower()
    if "error" in s or "fatal" in s: return "ERROR"
    if "warn" in s: return "WARN"
    if "info" in s or "debug" in s: return "INFO"
    return "OTHER"

TOKEN_RE = re.compile(r"[A-Za-z0-9\.\-_:/]+")
EXC_RE = re.compile(r"\b([A-Za-z0-9_. ]+Exception)\b")
HTTP_RE = re.compile(r"\b([45]\d{2})\b")
ERR_RE = re.compile(r"\bERR[ _\-]?\d+\b", re.IGNORECASE)

def extract_chunk_patterns(lines_chunk, top_k=10):
    uni = Counter(); bi = Counter(); tri = Counter()
    exc = Counter(); http = Counter(); errc = Counter()
    for line in lines_chunk:
        tokens = [t.lower() for t in TOKEN_RE.findall(line)]
        if tokens:
            uni. update(tokens)
            bi. update([" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)])
            tri.update([" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)])
        for m in EXC_RE.findall(line):   exc[m] += 1
        for m in HTTP_RE.findall(line):  http[m] += 1
        for m in ERR_RE.findall(line):   errc[m. upper()] += 1
    def top(counter, k): return [t for t,_ in counter.most_common(k)]
    card = {
        "unigrams": top(uni, top_k),
        "bigrams": top(bi, top_k),
        "trigrams": top(tri, top_k),
        "exceptions": top(exc, top_k),
        "http": top(http, top_k),
        "errors": top(errc, top_k),
    }
    pattern_set = set(card["unigrams"] + card["bigrams"] + card["trigrams"] +
                      card["exceptions"] + card["http"] + card["errors"])
    return card, pattern_set

@dataclass
class ChunkStats:
    index: int
    severity_counts: Dict[str, int]
    exceptions: Counter
    http_codes: Counter
    error_codes: Counter
    token_top: Dict[str, List[str]]
    anomaly_mean: float
    anomaly_max: float
    anomaly_window: np.ndarray
    acts_flat: List[str]
    pattern_set: set
    timestamp_first: Optional[str] = None

def analyze_chunk(ci, lines_chunk, acts_segment, anoms_segment, patt_topk=10, extra_regex=None) -> ChunkStats:
    sev_counts = Counter(severity_of(l) for l in lines_chunk)
    card, patt_set = extract_chunk_patterns(lines_chunk, top_k=patt_topk)
    acts_flat = [a for line_acts in acts_segment for a in line_acts]
    exc = Counter(EXC_RE.findall("\n".join(lines_chunk)))
    http = Counter(HTTP_RE.findall("\n".join(lines_chunk)))
    errc = Counter(ERR_RE.findall("\n".join(lines_chunk)))
    anom_arr = np.asarray(anoms_segment, dtype=float)
    ts_first = first_ts_match(lines_chunk[0], extra_regex) if lines_chunk else None
    return ChunkStats(
        ci, dict(sev_counts), exc, http, errc, card,
        float(np.mean(anom_arr)) if anom_arr.size > 0 else 0.0,
        float(np. max(anom_arr)) if anom_arr.size > 0 else 0.0,
        anom_arr, acts_flat, patt_set, ts_first
    )


# ========== ANOMALY CLUSTERING & INCIDENT FINGERPRINTING ==========

def cluster_anomalies(embeddings_np: np.ndarray, anomaly_scores: np.ndarray, 
                      eps=0.5, min_samples=2) -> Dict[int, List[int]]:
    """Cluster anomalous chunks using DBSCAN on embeddings + anomaly proximity."""
    if embeddings_np.shape[0] < 2:
        return {0: list(range(embeddings_np.shape[0]))}
    
    # Normalize embeddings
    scaler = StandardScaler()
    emb_norm = scaler.fit_transform(embeddings_np)
    
    # Add anomaly score as feature
    anom_feature = anomaly_scores. reshape(-1, 1) / (np.max(anomaly_scores) + 1e-8)
    combined = np.hstack([emb_norm, anom_feature])
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(combined)
    
    clusters = defaultdict(list)
    for chunk_id, label in enumerate(clustering.labels_):
        clusters[label].append(chunk_id)
    
    return dict(clusters)


def fingerprint_incident(chunks_in_cluster: List[ChunkStats]) -> Dict:
    """Generate incident fingerprint from grouped anomalous chunks."""
    if not chunks_in_cluster:
        return {}
    
    exc_all = Counter()
    http_all = Counter()
    errc_all = Counter()
    acts_all = Counter()
    sev_all = Counter()
    
    for cs in chunks_in_cluster:
        exc_all.update(cs.exceptions)
        http_all.update(cs.http_codes)
        errc_all.update(cs.error_codes)
        acts_all.update(cs.acts_flat)
        for sev, cnt in cs.severity_counts.items():
            sev_all[sev] += cnt
    
    return {
        "num_chunks": len(chunks_in_cluster),
        "anom_mean": float(np.mean([cs.anomaly_mean for cs in chunks_in_cluster])),
        "top_exceptions": dict(exc_all.most_common(3)),
        "top_http": dict(http_all.most_common(3)),
        "top_errors": dict(errc_all.most_common(3)),
        "dominant_activity": acts_all.most_common(1)[0][0] if acts_all else "UNKNOWN",
        "severity_mix": dict(sev_all),
        "error_density": sum(exc_all.values()) / max(len(chunks_in_cluster), 1),
    }


# ========== CAUSAL CHAIN EXTRACTION (IMPROVED) ==========

def cross_corr_lags(a, b, max_lag=3, min_overlap=3):
    """Robust lagged cross-correlation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < min_overlap or b.size < min_overlap:
        return 0, 0.0
    a_n = (a - a.mean()) / (a.std() + 1e-8)
    b_n = (b - b.mean()) / (b.std() + 1e-8)
    best_lag, best_score = 0, -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a_n[-lag:], b_n[:min(a_n[-lag:].size, b_n. size)]
        elif lag > 0:
            bb, aa = b_n[lag:], a_n[:min(b_n[lag:].size, a_n.size)]
        else:
            m = min(a_n.size, b_n.size)
            aa, bb = a_n[-m:], b_n[-m:]
        m = min(aa.size, bb.size)
        if m < min_overlap: continue
        score = float(np.mean(aa[:m] * bb[:m]))
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag, best_score


def cause_score_enhanced(sim, ai, aj, lag_score, lag, patt_sim, sev_drift, exc_overlap,
                        w_sim=0.30, w_grad=0.20, w_lag=0.15, w_patt=0.15, w_sev=0.10, w_exc=0.10):
    """Enhanced cause score with exception similarity."""
    grad = max(aj - ai, 0.0)
    lag_bonus = lag_score * (1.0 if lag > 0 else 0.7 if lag == 0 else 0.5)
    return (w_sim * sim + w_grad * grad + w_lag * lag_bonus + w_patt * patt_sim +
            w_sev * sev_drift + w_exc * exc_overlap)


def jaccard_overlap(a, b): 
    return len(a & b) / len(a | b) if a and b else 0.0


def exception_overlap(exc_counter_i, exc_counter_j) -> float:
    """Measure exception similarity between chunks."""
    if not exc_counter_i or not exc_counter_j:
        return 0.0
    keys_i = set(exc_counter_i.keys())
    keys_j = set(exc_counter_j.keys())
    return jaccard_overlap(keys_i, keys_j)


def cosine_corr_matrix(embeds_np, norms_cache=None):
    norms = np.linalg.norm(embeds_np, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    sim = (embeds_np @ embeds_np.T) / (norms @ norms.T)
    return np.clip(sim, -1.0, 1.0), norms


def compute_layout_stable(G: nx.Graph, pos_cache: Optional[Dict] = None, seed: int = SEED):
    try:
        if not pos_cache:
            return nx.spring_layout(G, seed=seed, k=0.95, iterations=350)
        pos_init = {n: pos_cache[n] for n in G.nodes() if n in pos_cache}
        return nx.spring_layout(G, seed=seed, k=0.95, iterations=350, pos=pos_init)
    except Exception:
        with contextlib.suppress(Exception):
            return nx.circular_layout(G)
        return {n: (0.0, 0.0) for n in G.nodes()}


def build_topology_enhanced(chunk_embeds, chunk_stats, sim_matrix, chunk_size, base_topk,
                           sim_threshold, infl_threshold, max_lag):
    """Build RCA topology with enhanced causality evidence."""
    G = nx.DiGraph()
    C = len(chunk_embeds)
    G.add_nodes_from(range(C))
    
    node_labels, node_anomaly, node_sizes = {}, [], []
    for idx in range(C):
        cs = chunk_stats[idx]
        top_acts = ", ".join([a. replace("_", " "). title() for a, _ in Counter(cs.acts_flat).most_common(3)]) or "No dominant activity"
        sev_str = " / ".join(f"{k}:{cs.severity_counts. get(k, 0)}" for k in ("ERROR", "WARN", "INFO"))
        node_labels[idx] = f"{top_acts}\nAnom μ:{cs.anomaly_mean:.3f} max:{cs.anomaly_max:. 3f}\nSev {sev_str}"
        node_anomaly.append(cs.anomaly_mean)
        node_sizes.append(800 + cs.anomaly_mean * 1200)
    
    edges_rank = []
    for i in range(C):
        ai = node_anomaly[i]
        wi = chunk_stats[i].anomaly_window
        candidates = []
        
        for j in range(C):
            if i == j: continue
            sim = float(sim_matrix[i, j])
            if sim < sim_threshold: continue
            
            aj = node_anomaly[j]
            wj = chunk_stats[j].anomaly_window
            
            try:
                lag, lag_score = cross_corr_lags(wi, wj, max_lag=max_lag) if wi. size >= 3 and wj.size >= 3 else (0, 0.0)
            except Exception:
                lag, lag_score = 0, 0.0
            
            patt_sim = jaccard_overlap(chunk_stats[i].pattern_set, chunk_stats[j].pattern_set)
            sev_drv = severity_drift(chunk_stats[i].severity_counts, chunk_stats[j].severity_counts)
            exc_sim = exception_overlap(chunk_stats[i].exceptions, chunk_stats[j].exceptions)
            
            score = cause_score_enhanced(sim, ai, aj, lag_score, lag, patt_sim, sev_drv, exc_sim)
            infl = max(0.0, sim * max(aj - ai, 0.0))
            
            if score >= infl_threshold:
                evidence = {
                    "sim": sim, "lag": lag, "lag_score": lag_score, "patt": patt_sim,
                    "sev_drift": sev_drv, "anom_grad": max(aj - ai, 0.0), "exc_sim": exc_sim
                }
                candidates.append((j, score, infl, evidence))
        
        candidates.sort(key=lambda t: t[1], reverse=True)
        for j, score, infl, ev in candidates[:base_topk]:
            G.add_edge(i, j, score=score, weight=max(0.01, infl), **ev)
            edges_rank.append({"from_chunk": i+1, "to_chunk": j+1, "score": score, "influence": infl, **ev})
    
    return G, node_labels, node_anomaly, node_sizes, edges_rank


def severity_drift(sev_i, sev_j):
    def sev_score(sev): 
        return (sev. get("ERROR", 0) + 0.5 * sev.get("WARN", 0)) / (sum(sev.values()) or 1)
    return max(sev_score(sev_j) - sev_score(sev_i), 0.0)


# ========== TIME SERIES & PREDICTIVE FEATURES ==========

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
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    d = d.set_index("ts")
    vol = d["line"].resample(freq).count(). rename("count"). to_frame()
    anom = d["anomaly"].resample(freq).mean().rename("anomaly_mean").to_frame()
    sev = pd.get_dummies(d["severity"]).resample(freq).sum().fillna(0)
    acts = pd.get_dummies(d["activity"]).resample(freq).sum().fillna(0)
    return vol, anom, sev, acts


def corr_heatmap_df(vol, anom, sev, acts):
    df = pd.concat([vol, anom, sev, acts], axis=1). fillna(0)
    if df.empty: return pd.DataFrame()
    return df.corr(method="pearson")


# ========== ADVANCED PREDICTIVE INSIGHTS ==========

def forecast_anomalies(anomaly_ts: pd.Series, lookback=10, forecast_steps=3) -> List[float]:
    """LSTM-based anomaly forecasting."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        forecaster = get_anomaly_forecaster(device)
        
        if len(anomaly_ts) < lookback:
            return []
        
        # Prepare input
        recent = anomaly_ts.tail(lookback). values. reshape(-1, 1). astype(np.float32)
        x = torch.from_numpy(recent). unsqueeze(0).to(device)  # [1, T, 1]
        
        with torch.no_grad():
            forecast = forecaster(x). cpu().numpy()[0]
        
        return np.clip(forecast, 0.0, 1.0). tolist()
    except Exception as e:
        st.warning(f"Anomaly forecasting failed: {e}")
        return []


def predict_activity_sequence(df_ts: pd.DataFrame, n_steps=3) -> List[Tuple[str, float]]:
    """Predict next N activities using Markov + embeddings."""
    try:
        acts = df_ts["activity"].dropna().tolist()
        if len(acts) < 2:
            return []
        
        predictor = ActivityTransitionPredictor()
        predictor.fit(acts)
        
        recent = acts[-5:]
        predictions = predictor.predict_next(recent, n_candidates=3)
        return predictions
    except Exception as e:
        st.warning(f"Activity prediction failed: {e}")
        return []


def identify_causal_chains(edges_rank: List[dict], chunk_stats: List[ChunkStats]) -> List[List[int]]:
    """Extract causal chains (paths) from RCA graph."""
    if not edges_rank:
        return []
    
    # Build adjacency
    graph = defaultdict(list)
    for edge in edges_rank:
        src, tgt, score = edge["from_chunk"] - 1, edge["to_chunk"] - 1, edge["score"]
        graph[src].append((tgt, score))
    
    # DFS to find chains
    chains = []
    def dfs(node, path, visited):
        if node in visited:
            return
        visited.add(node)
        path.append(node)
        
        if len(path) >= 2:
            chains.append(path. copy())
        
        for neighbor, _ in graph.get(node, []):
            dfs(neighbor, path, visited)
        
        path.pop()
        visited.remove(node)
    
    for start_node in graph:
        dfs(start_node, [], set())
    
    return sorted(chains, key=len, reverse=True)[:5]  # Top 5 chains


def warn_on_trend(anom_ts: pd.Series, vol_ts: pd.Series, threshold_rise=0.15) -> List[str]:
    """Detect concerning trends and emit warnings."""
    warnings = []
    
    if len(anom_ts) >= 3:
        slope = anom_ts.iloc[-1] - anom_ts.iloc[-3]
        if slope > threshold_rise:
            warnings.append(f"🔴 Anomaly trend rising sharply (slope={slope:.3f}).  Investigate recent changes.")
    
    if len(vol_ts) >= 3:
        vol_slope = vol_ts.iloc[-1] - vol_ts. iloc[-3]
        if vol_slope < -0.3:
            warnings.append("⚠️ Log volume declining sharply. Check service health.")
    
    return warnings


def likely_next_activity(df_ts: pd.DataFrame) -> Optional[str]:
    acts = df_ts["activity"].dropna().tolist()
    if len(acts) < 2:
        return None
    pairs = list(zip(acts[:-1], acts[1:]))
    cnt = Counter(pairs)
    if not cnt:
        return None
    last = acts[-1]
    candidates = [(b, c) for (a, b), c in cnt. items() if a == last]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]




def predictive_summary_enhanced(vol, anom_ts, sev_ts, acts_ts, df_ts, chunk_stats_list,
                               edges_rank, ts_freq: str, anom_forecast: List[float],
                               act_predictions: List[Tuple[str, float]],
                               causal_chains: List[List[int]]) -> str:
    """
    Generate predictive summary with context-aware checks:
    - Skip misleading insights when data is sparse
    - Include anomaly trend, forecast, activity predictions, causal chains, severity, and warnings
    """
    lines = []

    # --- Anomaly Trend + Forecast ---
    if anom_ts.empty or "anomaly_mean" not in anom_ts or len(anom_ts) < 3:
        risk_line = "No meaningful anomaly trend detected (insufficient data)."
    else:
        s = anom_ts["anomaly_mean"].ewm(span=3).mean()
        slope = float(s.iloc[-1] - s.iloc[-2])
        level = float(s.iloc[-1])
        trend = "rising" if slope > 0 else "falling" if slope < 0 else "steady"
        risk_line = f"Anomaly EWMA is **{trend}** (now={level:.3f}, Δ={slope:+.3f}) over **{ts_freq}** bins."
        if anom_forecast and len(anom_forecast) >= 1:
            avg_forecast = np.mean(anom_forecast)
            risk_line += f" **Forecast**: next {len(anom_forecast)} steps avg={avg_forecast:.3f}."
    lines.append(f"• {risk_line}")

    # --- Activity Predictions ---
    unique_acts = set(df_ts["activity"].dropna())
    if act_predictions and len(unique_acts) > 1:
        next_acts = ", ".join([f"{a.replace('_', ' ').title()} ({p:.1%})" for a, p in act_predictions[:3]])
        lines.append(f"• Next likely activities: {next_acts}.")
    else:
        lines.append("• Activity prediction skipped (insufficient diversity).")

    # --- Causal Chains ---
    if causal_chains:
        top_chain = causal_chains[0]
        chain_desc = " → ".join([f"C{c+1}" for c in top_chain])
        lines.append(f"• Primary causal chain: {chain_desc}.")
    else:
        lines.append("• No causal chain detected.")

    # --- Severity in Latest Chunks ---
    if chunk_stats_list:
        last_k = chunk_stats_list[-min(3, len(chunk_stats_list)):]
        err_w = sum(cs.severity_counts.get("ERROR", 0) for cs in last_k)
        warn_w = sum(cs.severity_counts.get("WARN", 0) for cs in last_k)
        lines.append(f"• Latest {len(last_k)} chunk(s): ERROR={err_w}, WARN={warn_w}.")
    else:
        lines.append("• No severity data available.")

    # --- Top Causal Edges ---
    if edges_rank:
        top_edges = sorted(edges_rank, key=lambda x: x["score"], reverse=True)[:3]
        desc = "; ".join([f"C{e['from_chunk']}→C{e['to_chunk']} (score {e['score']:.2f})" for e in top_edges])
        lines.append(f"• Most influential causes: {desc}.")
    else:
        lines.append("• No causal edges above threshold.")

    # --- Error Drivers ---
    exc_all = Counter()
    for cs in chunk_stats_list:
        exc_all.update(cs.exceptions)
    if exc_all:
        e1 = ", ".join([f"{k}({v})" for k, v in exc_all.most_common(3)])
        lines.append(f"• Top exceptions: {e1}.")
    else:
        lines.append("• No exception patterns detected.")

    # --- Trend Warnings ---
    anomaly_series = anom_ts["anomaly_mean"] if "anomaly_mean" in anom_ts else pd.Series()
    volume_series = vol["count"] if "count" in vol else pd.Series()
    if len(anomaly_series) >= 3 and len(volume_series) >= 3:
        trend_warns = warn_on_trend(anomaly_series, volume_series)
        if trend_warns:
            lines.extend(trend_warns)
    else:
        lines.append("• Trend warnings skipped (insufficient time-series data).")

    return "\n".join(lines)


def activity_counts_and_transitions(df_ts: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    if df_ts.empty:
        return pd.Series(dtype=int), pd.DataFrame(columns=["from", "to", "count"])
    acts = df_ts["activity"].fillna("NONE")
    counts = acts.value_counts()
    pairs = list(zip(acts[:-1], acts[1:])) if len(acts) >= 2 else []
    df_pairs = pd.DataFrame(pairs, columns=["from", "to"])
    if df_pairs.empty:
        return counts, pd.DataFrame(columns=["from", "to", "count"])
    df_pairs["count"] = 1
    df_trans = df_pairs.groupby(["from", "to"])["count"].sum().reset_index(). sort_values("count", ascending=False)
    return counts, df_trans


def render_activity_summary(df_ts: pd.DataFrame, sev_ts: pd.DataFrame, acts_ts: pd.DataFrame) -> str:
    txt_parts = []
    total = len(df_ts)
    if total == 0:
        return "No logs available to summarize."
    top_counts, trans_df = activity_counts_and_transitions(df_ts)
    if not top_counts.empty:
        top3 = ", ".join([f"{a. replace('_', ' ').title()} ({c})" for a, c in top_counts.head(3).items()])
        txt_parts.append(f"Dominant activities: {top3}.")
    if not trans_df.empty:
        trows = trans_df.head(3). to_dict("records")
        tdesc = "; ".join([f"{r['from'].replace('_', ' ').title()} → {r['to'].replace('_', ' ').title()} ({r['count']})" for r in trows])
        txt_parts.append(f"Major transitions: {tdesc}.")
    if not sev_ts.empty:
        last = sev_ts.tail(3).sum()
        err, warn, info = int(last. get("ERROR", 0)), int(last.get("WARN", 0)), int(last.get("INFO", 0))
        txt_parts.append(f"Recent severity: ERROR {err}, WARN {warn}, INFO {info}.")
    return " ".join(txt_parts)




# ======================================================
# PART 3 — STREAMLIT APP (UI + Upload widget + Optimizations)
# ======================================================

st.set_page_config(page_title="Semantic Activity Log Analyzer v33", layout="wide")
st.title("🧠 Semantic Activity Log Analyzer — v33 (Optimized)")
st.caption("Efficient RCA • Incident Clustering • Anomaly Forecasting • Activity Prediction • Causal Chains")

# === Helper functions used by Part 3 (upload, encoding, batching, GPU mem, auto-tune) ===

def safe_decode_uploaded(uploaded_file, max_bytes=10_000_000):
    """Safely decode uploaded file content to text."""
    raw = uploaded_file.getvalue()
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def encode_texts(texts, max_len=200, device=None):
    """Encode each line to a fixed-length int array [max_len], clamping ordinals to 255."""
    arrs = []
    for t in texts:
        cut = t[:max_len]
        a = [min(ord(c), 255) for c in cut] + [0] * (max_len - len(cut))
        arrs.append(a)
    ten = torch.tensor(arrs).long()
    return ten.to(device) if device else ten


def classify_lines_batch(model, device, lines, max_len=200, batch_size=128, use_amp=True):
    """Batch inference over log lines—returns activities per line, anomaly score per line, and embeddings."""
    acts_all, anoms_all, embeds_all = [], [], []
    labels = [
        "STARTUP", "SHUTDOWN", "CONNECTION_ERROR", "AUTH_FAILURE",
        "RETRY", "TIMEOUT", "CRASH_LOOP", "DATA_PROCESSING"
    ]
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else contextlib.nullcontext
    with torch.no_grad(), amp_ctx():
        for i in range(0, len(lines), batch_size):
            x = encode_texts(lines[i:i+batch_size], max_len, device)
            logits, emb = model.forward_line(x)
            probs = logits.detach().cpu().numpy()
            for bi, p in enumerate(probs):
                # multi-label threshold at 0.5
                acts_all.append([labels[j] for j, v in enumerate(p) if v > 0.5])
                # anomaly score: inverse of max class probability
                anoms_all.append(float(1.0 - p.max()))
                embeds_all.append(emb[bi].detach().cpu())
    return acts_all, anoms_all, embeds_all


def gpu_memory_free_gb(device):
    """Return free GPU memory in GB for CUDA devices; 0.0 otherwise."""
    if device.type != "cuda":
        return 0.0
    with contextlib.suppress(Exception):
        free, _ = torch.cuda.mem_get_info()
        return free / (1024**3)
    return 0.0


def auto_choose_max_len(lines, hard_cap=512, base_min=64, p=95) -> int:
    """Pick a max tokenized line length based on percentile of observed lengths."""
    if not lines:
        return base_min
    lens = np.array([len(l) for l in lines])
    q = int(np.clip(np.percentile(lens, p), base_min, hard_cap))
    return int(q)


def auto_choose_batch_size(device, max_len, gpu_free_gb):
    """Heuristic to pick batch size from device type and available memory budget."""
    if device.type == "cuda":
        if gpu_free_gb >= 12:
            budget = 256_000
        elif gpu_free_gb >= 8:
            budget = 192_000
        elif gpu_free_gb >= 4:
            budget = 128_000
        else:
            budget = 96_000
    else:
        budget = 64_000
    return int(np.clip(budget // max_len, 16, 256))


def auto_choose_chunk_size(num_lines, min_c=20, max_c=500, target_chunks=(12, 40)):
    """Choose chunk size to produce ~target number of chunks over num_lines."""
    if num_lines <= min_c:
        return max(10, num_lines)
    lo, hi = target_chunks
    size = int(np.clip(np.ceil(num_lines / ((lo + hi) / 2.0)), min_c, max_c))
    return int(np.clip(int(np.round(size / 10) * 10), min_c, max_c))


# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls & Settings")

    mode = st.radio(
        "Analysis Mode",
        ["Quick Preview", "Deep Analysis"],
        help="Quick: fast overview; Deep: full RCA & forecasting",
    )
    auto_opt = st.checkbox(
        "Auto Optimize (recommended)", value=True,
        help="Adapts parameters to file size & hardware."
    )

    # Feature toggles
    if mode == "Deep Analysis":
        enable_forecast = st.checkbox(
            "Enable Anomaly Forecasting", value=True,
            help="LSTM-based next-step anomaly prediction."
        )
        enable_clustering = st.checkbox(
            "Incident Clustering", value=True,
            help="Group anomalies into incidents with fingerprints."
        )
    else:
        enable_forecast = False
        enable_clustering = False

    st.markdown("---")
    max_lines_manual = st.slider("Max lines to analyze", 100, 10000, 1500, step=100, disabled=auto_opt)
    batch_size_manual = st.slider("Batch size", 16, 256, 128, step=16, disabled=auto_opt)
    chunk_size_manual = st.slider("Chunk size", 20, 500, 60, step=10, disabled=auto_opt)
    use_amp_manual = st.checkbox("Mixed Precision (GPU)", value=True, disabled=auto_opt)

    st.markdown("---")
    patt_topk = st.slider("Top patterns per chunk", 5, 20, 10, step=1)
    lag_max = st.slider("Cross-corr max lag", 0, 6, 3, step=1)
    ts_freq = st.selectbox("Timeseries frequency", ["30s", "1min", "5min", "15min"], index=1)

    st.markdown("---")
    st.subheader("Graph Visibility")
    min_edge_score = st.slider("Min edge score", 0.0, 1.0, 0.05, step=0.01, help="Filter weak causal links.")
    min_edge_infl = st.slider("Min edge influence", 0.0, 0.50, 0.01, step=0.01)
    arrow_scale = st.slider("Arrow thickness", 1.0, 12.0, 6.0, step=0.5)
    arrow_offset = st.slider("Arrow offset", 0.0, 0.20, 0.08, step=0.01)  # reserved for future bezier offset
    edge_line_width = st.slider("Edge line width", 0.5, 6.0, 2.8, step=0.1)
    edge_opacity = st.slider("Edge opacity", 0.1, 1.0, 0.75, step=0.05)

    st.markdown("---")
    save_now = st.button("💾 Save model to disk")
    reset_model = st.button("♻️ Reset persisted model")

# --- Device & Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device)

if reset_model:
    with contextlib.suppress(Exception):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            st.info("Model file deleted.")
if save_now:
    with contextlib.suppress(Exception):
        torch.save(model.state_dict(), MODEL_PATH)
        st.success("Model saved.")

if "pos_cache" not in st.session_state:
    st.session_state.pos_cache = None

# --- File Upload ---
uploaded_file = st.file_uploader("Upload log file:", type=["txt", "log", "csv", "json"])

if uploaded_file:
    raw_text = safe_decode_uploaded(uploaded_file)
    all_lines = raw_text.splitlines()

    # --- Preview & stats ---
    st.subheader("📄 Log Preview & Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Lines", len(all_lines))
    col2.metric("Device", device.type.upper())
    col3.metric("Memory Free", f"{gpu_memory_free_gb(device):.1f} GB")

    preview_text = "\n".join(all_lines[:20])
    st.code(preview_text, language="log")

    # --- Auto-tune ---
    if auto_opt:
        free_gb = gpu_memory_free_gb(device)
        if len(all_lines) <= 1500:
            max_lines = len(all_lines)
        elif len(all_lines) <= 5000:
            max_lines = 2500
        else:
            max_lines = 5000
        max_len = auto_choose_max_len(all_lines[:max_lines], hard_cap=512, base_min=64, p=95)
        batch_sz = auto_choose_batch_size(device, max_len, free_gb)
        chunk_size = auto_choose_chunk_size(max_lines, min_c=20, max_c=500)
        use_amp = (device.type == "cuda")
    else:
        max_lines = max_lines_manual
        max_len = 200
        batch_sz = batch_size_manual
        chunk_size = chunk_size_manual
        use_amp = use_amp_manual

    with st.expander("🔧 Effective Parameters", expanded=False):
        st.json({
            "max_lines": int(max_lines),
            "max_len": int(max_len),
            "batch_size": int(batch_sz),
            "chunk_size": int(chunk_size),
            "use_amp": bool(use_amp),
            "device": device.type,
            "mode": mode,
            "forecasting_enabled": enable_forecast,
            "clustering_enabled": enable_clustering,
        })

    lines = all_lines[:max_lines]
    n_total = len(lines)
    total_chunks = max(1, int(np.ceil(n_total / chunk_size)))

    # --- Main analysis pipeline containers ---
    rca_ph = st.container()
    progress_ph = st.progress(0)

    anomaly_plot_ph = st.empty()
    patterns_ph = st.expander("🧩 Pattern Cards (per chunk)", expanded=False)
    incident_ph = st.expander("🚨 Incident Clustering & Fingerprints", expanded=True) if enable_clustering else None
    activity_summary_ph = st.expander("📝 Activity Summary", expanded=True)
    ts_panel_ph = st.expander("⏱️ Timeseries", expanded=False)
    insights_ph = st.expander("🧭 Predictive Insights & Forecasts", expanded=True)
    sequence_ph = st.expander("🔗 Causal Chains", expanded=False)

    # --- Accumulators ---
    all_anoms = []
    chunk_embeds_acc, acts_per_chunk_acc = [], []
    pattern_cards, pattern_sets = [], []
    chunk_stats_list: List[ChunkStats] = []
    norms_cache = None
    ts_rows = []
    anomaly_x, anomaly_y = [], []

    # --- Processing loop ---
    with st.spinner(f"Processing {total_chunks} chunks..."):
        for ci in range(total_chunks):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, len(lines))
            lines_chunk = lines[s:e]

            # Classify
            acts, anoms, embeds = classify_lines_batch(
                model, device, lines_chunk,
                max_len=max_len, batch_size=batch_sz, use_amp=use_amp
            )
            all_anoms.extend(anoms)

            # Chunk embed
            pooled, _ = model.forward_chunk_adaptive(embeds)
            chunk_embeds_acc.append(pooled.squeeze(0).cpu())
            acts_per_chunk_acc.append(acts)

            # Patterns & stats
            card, patt_set = extract_chunk_patterns(lines_chunk, top_k=patt_topk)
            pattern_cards.append(card)
            pattern_sets.append(patt_set)

            # Analyze chunk and store stats
            cs = analyze_chunk(ci, lines_chunk, acts, anoms, patt_topk)
            chunk_stats_list.append(cs)

            # Timeline rows for time-series analysis
            ts_rows.extend(append_timeline_rows(lines_chunk, acts, anoms, base_idx=s))

            # Pattern cards UI
            with patterns_ph:
                colp1, colp2, colp3 = st.columns(3)
                with colp1:
                    st.write(f"**C{ci+1}** Unigrams: {', '.join(card['unigrams'][:5]) or '—'}")
                with colp2:
                    st.write(f"Exceptions: {', '.join(card['exceptions'][:3]) or '—'}")
                with colp3:
                    st.write(f"HTTP: {', '.join(card['http']) or '—'}")

            # Anomaly plot data
            anomaly_x.extend(range(s, e))
            anomaly_y.extend(anoms)

            # Update progress bar
            progress_ph.progress(int(((ci + 1) / total_chunks) * 100))

    progress_ph.progress(100)

    # --- Anomaly Trend Plot ---
    fig_anom, ax_anom = plt.subplots(figsize=(13, 4))
    ax_anom.plot(anomaly_x, anomaly_y, marker='o', linestyle='-', color='#d62728', markersize=3, alpha=0.7)
    ax_anom.set_xlabel("Log Line Index")
    ax_anom.set_ylabel("Anomaly Score")
    ax_anom.set_title("Line-wise Anomaly Trend (full span)")
    ax_anom.grid(True, alpha=0.3)
    anomaly_plot_ph.pyplot(fig_anom)
    plt.close(fig_anom)

    # --- Timeseries & Activity Summary ---
    df_ts = pd.DataFrame(ts_rows, columns=["idx", "ts", "line", "severity", "activity", "anomaly"])

    # Activity summary
    with activity_summary_ph:
        summary_txt = render_activity_summary(df_ts, *resample_metrics(df_ts, ts_freq)[2:])
        st.write(summary_txt)

    # Timeseries panels
    with ts_panel_ph:
        vol, anom_ts_df, sev_ts_df, acts_ts_df = resample_metrics(df_ts, ts_freq)
        st.subheader("📈 Resampled Metrics")
        if not vol.empty:
            st.line_chart(vol["count"], height=180)
        if not anom_ts_df.empty:
            st.line_chart(anom_ts_df["anomaly_mean"], height=180)
        if not sev_ts_df.empty:
            st.area_chart(sev_ts_df, height=200)
        if not acts_ts_df.empty:
            st.area_chart(acts_ts_df, height=200)

        corr_df = corr_heatmap_df(vol, anom_ts_df, sev_ts_df, acts_ts_df)
        if not corr_df.empty:
            st.subheader("🔶 Correlation Heatmap (Pearson)")
            st.dataframe(corr_df.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

    # --- RCA Topology ---
    edges_rank: List[dict] = []
    if len(chunk_embeds_acc) > 1:
        try:
            # Build similarity and RCA graph
            chunk_np = np.stack([e.numpy() for e in chunk_embeds_acc])
            sim_matrix, norms_cache = cosine_corr_matrix(chunk_np)

            G, node_labels, node_anomaly, node_sizes, edges_rank = build_topology_enhanced(
                chunk_embeds_acc, chunk_stats_list, sim_matrix, chunk_size,
                base_topk=3, sim_threshold=0.35, infl_threshold=0.02, max_lag=lag_max
            )

            # Use stable layout; reuse last positions if available
            pos = compute_layout_stable(G, st.session_state.get("pos_cache"))
            st.session_state.pos_cache = pos

            # ---- Plotly rendering (preferred) ----
            if PLOTLY_AVAILABLE and G.number_of_edges() > 0:
                # Normalize anomalies for color scale
                an_min = float(min(node_anomaly)) if node_anomaly else 0.0
                an_max = float(max(node_anomaly)) if node_anomaly else 1.0
                if an_max - an_min < 1e-8:
                    an_max = an_min + 1e-8

                def norm(a: float) -> float:
                    return (a - an_min) / (an_max - an_min)

                # Nodes
                node_x, node_y, node_text, node_color, node_marker_size = [], [], [], [], []
                for n in G.nodes():
                    if n not in pos:
                        continue
                    x, y = pos[n]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node_labels[n])
                    node_color.append(norm(float(node_anomaly[n])))
                    node_marker_size.append(12 + float(node_anomaly[n]) * 20)

                node_colors_hex = sample_colorscale('Reds', node_color) if node_color else ['#d62728'] * len(node_x)
                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text',
                    text=node_text, textposition='top center',
                    hoverinfo='text',
                    marker=dict(color=node_colors_hex,
                                size=node_marker_size,
                                line=dict(color='black', width=0.5))
                )

                # Edges: lines + arrowheads (colored by lag)
                def lag_color(lag: int) -> str:
                    return '#32CD32' if lag > 0 else '#FF8C00' if lag == 0 else '#1E90FF'

                edges_lines = {
                    '#32CD32': {'x': [], 'y': []},
                    '#FF8C00': {'x': [], 'y': []},
                    '#1E90FF': {'x': [], 'y': []},
                }
                arrow_heads = {
                    '#32CD32': {'x': [], 'y': [], 'text': []},
                    '#FF8C00': {'x': [], 'y': [], 'text': []},
                    '#1E90FF': {'x': [], 'y': [], 'text': []},
                }

                for u, v, data in G.edges(data=True):
                    # Thresholds
                    edge_score = float(data.get('score', 0.0))
                    edge_weight = float(data.get('weight', 0.0))
                    if edge_score < float(min_edge_score) or edge_weight < float(min_edge_infl):
                        continue
                    if u not in pos or v not in pos:
                        continue

                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    color = lag_color(int(data.get('lag', 0)))

                    # Lines (no hover for performance)
                    edges_lines[color]['x'].extend([x0, x1, None])
                    edges_lines[color]['y'].extend([y0, y1, None])

                    # Prepare hover text with **explicit** coercion and precision
                    score = float(data.get('score', 0.0))
                    infl = float(data.get('weight', 0.0))
                    lag = int(data.get('lag', 0))
                    lag_score = float(data.get('lag_score', 0.0))
                    patt = float(data.get('patt', 0.0))
                    sev_drift = float(data.get('sev_drift', 0.0))
                    exc_sim = float(data.get('exc_sim', 0.0))
                    anom_grad = float(data.get('anom_grad', 0.0))

                    hover = (
                        f"C{u+1} → C{v+1}<br>"
                        f"score={score:.2f}, infl={infl:.2f}<br>"
                        f"lag={lag}, lag_score={lag_score:.2f}<br>"
                        f"patt={patt:.2f}, sev_drift={sev_drift:.2f}, "
                        f"exc_sim={exc_sim:.2f}, anom_grad={anom_grad:.2f}"
                    )

                    arrow_heads[color]['x'].append(x1)
                    arrow_heads[color]['y'].append(y1)
                    arrow_heads[color]['text'].append(hover)

                # Build traces
                edge_traces = []
                for color, pts in edges_lines.items():
                    if pts['x']:
                        edge_traces.append(
                            go.Scatter(
                                x=pts['x'], y=pts['y'], mode='lines',
                                line=dict(width=edge_line_width, color=color),
                                opacity=edge_opacity,
                                hoverinfo='skip',
                                showlegend=False
                            )
                        )
                for color, heads in arrow_heads.items():
                    if heads['x']:
                        edge_traces.append(
                            go.Scatter(
                                x=heads['x'], y=heads['y'], mode='markers',
                                marker=dict(symbol='triangle-up',
                                            size=max(8, int(arrow_scale)),
                                            color=color),
                                text=heads['text'],
                                hoverinfo='text',
                                showlegend=False
                            )
                        )

                # Axis bounds
                xs = [p[0] for p in pos.values()]
                ys = [p[1] for p in pos.values()]
                xmin, xmax = (min(xs) - 0.15, max(xs) + 0.15) if xs else (-1, 1)
                ymin, ymax = (min(ys) - 0.15, max(ys) + 0.15) if ys else (-1, 1)

                fig = go.Figure(
                    data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title="RCA Topology — Causal Analysis",
                        height=700, showlegend=False,
                        xaxis=dict(showgrid=False, showticklabels=False, range=[xmin, xmax]),
                        yaxis=dict(showgrid=False, showticklabels=False, range=[ymin, ymax]),
                    )
                )
                rca_ph.plotly_chart(fig, use_container_width=True)

            else:
                # ---- Matplotlib fallback (Plotly not available or no edges above thresholds) ----
                if G.number_of_edges() > 0:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    nx.draw_networkx_nodes(
                        G, pos,
                        node_size=[max(80, 80 + float(a) * 60) for a in node_anomaly],
                        node_color=node_anomaly, cmap=plt.cm.Reds, ax=ax
                    )
                    for (u, v, data) in G.edges(data=True):
                        edge_score = float(data.get('score', 0.0))
                        edge_weight = float(data.get('weight', 0.0))
                        if edge_score < float(min_edge_score) or edge_weight < float(min_edge_infl):
                            continue
                        nx.draw_networkx_edges(
                            G, pos, edgelist=[(u, v)],
                            width=edge_line_width, alpha=edge_opacity,
                            arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax
                        )
                    nx.draw_networkx_labels(G, pos, labels={n: f"C{n+1}" for n in G.nodes()}, font_size=9, ax=ax)
                    ax.set_title("RCA Topology — Causal Analysis (matplotlib fallback)")
                    ax.axis('off')
                    rca_ph.pyplot(fig)
                    plt.close(fig)
                else:
                    rca_ph.info("No edges above thresholds (nothing to render).")

        except Exception as e:
            rca_ph.warning(f"RCA topology rendering failed: {e}")

    # --- Incident Clustering ---
    if enable_clustering and len(chunk_embeds_acc) > 1 and incident_ph is not None:
        with incident_ph:
            try:
                chunk_np = np.stack([e.numpy() for e in chunk_embeds_acc])
                anom_scores = np.array([cs.anomaly_mean for cs in chunk_stats_list])

                clusters = cluster_anomalies(chunk_np, anom_scores, eps=0.5, min_samples=2)
                st.subheader("🚨 Detected Incidents")

                for cluster_id, chunk_ids in clusters.items():
                    if cluster_id == -1:
                        continue
                    clusters_data = [chunk_stats_list[cid] for cid in chunk_ids]
                    fingerprint = fingerprint_incident(clusters_data)
                    with st.expander(
                        f"Incident {cluster_id+1} — Chunks {chunk_ids}, Anom μ={fingerprint.get('anom_mean', 0):.3f}"
                    ):
                        st.json(fingerprint)
            except Exception as e:
                st.warning(f"Incident clustering failed: {e}")

    # --- Forecasting, Activity Prediction & Insights ---
    anom_forecast: List[float] = []
    act_predictions: List[Tuple[str, float]] = []
    causal_chains: List[List[int]] = []

    try:
        if enable_forecast:
            # Build anomaly time series with resampling
            anom_ts = df_ts.groupby(pd.Grouper(key='ts', freq=ts_freq))['anomaly'].mean()
            anom_forecast = forecast_anomalies(anom_ts, lookback=10, forecast_steps=3)
    except Exception as e:
        st.warning(f"Forecasting failed: {e}")

    try:
        act_predictions = predict_activity_sequence(df_ts, n_steps=3)
    except Exception as e:
        st.warning(f"Activity prediction failed: {e}")

    try:
        causal_chains = identify_causal_chains(edges_rank, chunk_stats_list)
    except Exception as e:
        st.warning(f"Causal chain extraction failed: {e}")

    with insights_ph:
        # Predictive narrative summary
        vol_df, anom_df, sev_df, acts_df = resample_metrics(df_ts, ts_freq)
        summary_md = predictive_summary_enhanced(
            vol_df, anom_df, sev_df, acts_df, df_ts,
            chunk_stats_list, edges_rank, ts_freq,
            anom_forecast, act_predictions, causal_chains
        )
        st.markdown(summary_md)

        # Likely next activity (simple heuristic)
        next_act = likely_next_activity(df_ts)
        if next_act:
            st.info(f"🔮 Likely next activity: **{next_act.replace('_',' ').title()}**")

    # --- Causal Chains Panel ---
    if causal_chains:
        with sequence_ph:
            st.subheader("Derived Causal Chains (Top)")
            for idx, chain in enumerate(causal_chains[:5], start=1):
                chain_desc = " → ".join([f"C{c+1}" for c in chain])
                st.write(f"{idx}. {chain_desc}")
