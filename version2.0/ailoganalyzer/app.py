
# app.py — Semantic Activity Log Analyzer (single-file v32)
# Central Directed RCA Topology • Activity Summary • Predictive Insights • Timeseries • Correlation • Error breakup
# Model persistence • Self-optimizing batching • Learned attention pooler • Pattern miner

# ======================================================
# PART 0 — Imports
# ======================================================
import os
import contextlib
import re
import time
import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    import plotly.express as px
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
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

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
    """Content-aware attention pooling."""
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
        return self.norm(summary), weights


class AdaptiveHierarchicalTransformer(nn.Module):
    """Line-level transformer encodes lines; chunk-level aggregates."""
    def __init__(self, vocab_size=256, embed_dim=512, n_heads=16,
                 line_layers=4, chunk_layers=2, max_summary_tokens=4,
                 router_threshold=0.25, pool_heads=8, pool_init_mode="mean"):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_summary_tokens = max_summary_tokens
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.line_pos = PositionalEncoding(embed_dim)
        self.line_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, embed_dim*4, batch_first=True, activation="gelu"),
            num_layers=line_layers
        )

        self.chunk_pos = PositionalEncoding(embed_dim)
        self.chunk_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, n_heads, embed_dim*4, batch_first=True, activation="gelu"),
            num_layers=chunk_layers
        )

        self.norm_chunk = nn.LayerNorm(embed_dim)
        self.fc_line = nn.Linear(embed_dim, 8)  # 8 activity labels

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
        )
        summary_tokens, weights = self.attn_pooler(lines_tensor)
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
    model = AdaptiveHierarchicalTransformer(
        router_threshold=router_threshold,
        pool_heads=pool_heads,
        pool_init_mode=pool_init_mode,
        max_summary_tokens=max_summary_tokens
    )
    if os.path.exists(MODEL_PATH):
        with contextlib.suppress(Exception):
            sd = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(sd)
            print("✅ Loaded model from disk.")
    with contextlib.suppress(Exception):
        model = torch.compile(model)
    model.eval()
    model.to(device)
    if not os.path.exists(MODEL_PATH):
        with contextlib.suppress(Exception):
            torch.save(model.state_dict(), MODEL_PATH)
            print("💾 Model initialized and saved.")
    return model


# ---------- I/O and batching helpers ----------
def safe_decode_uploaded(uploaded_file, max_bytes=10_000_000):
    raw = uploaded_file.getvalue()
    if len(raw) > max_bytes: raw = raw[:max_bytes]
    for enc in ("utf-8","utf-16","latin-1"):
        try: return raw.decode(enc)
        except Exception: continue
    return raw.decode("utf-8", errors="ignore")

def try_parse_json_lines(raw_text: str) -> Optional[List[str]]:
    """Accepts a JSON array of logs: {'timestamp','level','message'} and converts to plain lines."""
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            lines = []
            for x in obj:
                ts = x.get("timestamp", "")
                lvl = x.get("level", "")
                msg = x.get("message", "")
                lines.append(f"{ts} {lvl} {msg}")
            return lines
    except Exception:
        return None
    return None

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
# PART 2 — ANALYTICS (Temporal + RCA + Pattern Miner)
# ======================================================
TIMESTAMP_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?",
    r"\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s[+\-]\d{4}",
    r"\w{3}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}",
    r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}",
    r"\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}",
]

def first_ts_match(line: str, extra_regex: Optional[str] = None):
    if extra_regex:
        m = re.search(extra_regex, line)
        if m: return m.group(0)
    for pat in TIMESTAMP_PATTERNS:
        m = re.search(pat, line)
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

TOKEN_RE = re.compile(r"[A-Za-z0-9\.\-_:/]+")
EXC_RE   = re.compile(r"\b([A-Za-z0-9_.]+Exception)\b")
HTTP_RE  = re.compile(r"\b([45]\d{2})\b")
ERR_RE   = re.compile(r"\bERR[ _\-]?\d+\b", re.IGNORECASE)

def extract_chunk_patterns(lines_chunk, top_k=10):
    uni = Counter(); bi = Counter(); tri = Counter()
    exc = Counter(); http = Counter(); errc = Counter()
    for line in lines_chunk:
        tokens = [t.lower() for t in TOKEN_RE.findall(line)]
        if tokens:
            uni.update(tokens)
            bi.update([" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)])
            tri.update([" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)])
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
    pattern_set = set(card["unigrams"] + card["bigrams"] + card["trigrams"] +
                      card["exceptions"] + card["http"] + card["errors"])
    return card, pattern_set

@dataclass
class ChunkStats:
    index: int
    severity_counts: Dict[str,int]
    exceptions: Counter
    http_codes: Counter
    error_codes: Counter
    token_top: Dict[str,List[str]]
    anomaly_mean: float
    anomaly_window: np.ndarray
    acts_flat: List[str]
    pattern_set: set

def analyze_chunk(ci, lines_chunk, acts_segment, anoms_segment, patt_topk=10) -> ChunkStats:
    sev_counts = Counter(severity_of(l) for l in lines_chunk)
    card, patt_set = extract_chunk_patterns(lines_chunk, top_k=patt_topk)
    acts_flat = [a for line_acts in acts_segment for a in line_acts]
    exc = Counter(EXC_RE.findall("\n".join(lines_chunk)))
    http = Counter(HTTP_RE.findall("\n".join(lines_chunk)))
    errc = Counter(ERR_RE.findall("\n".join(lines_chunk)))
    return ChunkStats(ci, dict(sev_counts), exc, http, errc, card,
                      float(np.mean(anoms_segment)) if anoms_segment else 0.0,
                      np.asarray(anoms_segment, dtype=float), acts_flat, patt_set)

def severity_drift(sev_i, sev_j):
    def sev_score(sev): return (sev.get("ERROR",0)+0.5*sev.get("WARN",0))/(sum(sev.values()) or 1)
    return max(sev_score(sev_j)-sev_score(sev_i),0.0)

def jaccard_overlap(a,b): return len(a & b)/len(a|b) if a and b else 0.0

# ---- Robust lagged cross-correlation (length-safe) ----
def cross_corr_lags(a, b, max_lag=3, min_overlap=3):
    """
    Robust lagged cross-correlation:
    - normalizes both vectors (z-score) with 1e-8 epsilon
    - for each lag, trims BOTH slices to the exact same length m
    - returns (best_lag, best_score) where score is mean(aa * bb)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size < min_overlap or b.size < min_overlap:
        return 0, 0.0

    a_n = (a - a.mean()) / (a.std() + 1e-8)
    b_n = (b - b.mean()) / (b.std() + 1e-8)

    best_lag, best_score = 0, -1.0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a_n[-lag:]  # shift a forward
            m = min(aa.size, b_n.size)
            if m < min_overlap:
                continue
            aa = aa[:m]
            bb = b_n[:m]
        elif lag > 0:
            bb = b_n[lag:]  # shift b forward
            m = min(bb.size, a_n.size)
            if m < min_overlap:
                continue
            bb = bb[:m]
            aa = a_n[:m]
        else:
            m = min(a_n.size, b_n.size)
            if m < min_overlap:
                continue
            aa = a_n[-m:]
            bb = b_n[-m:]

        score = float(np.mean(aa * bb))
        if score > best_score:
            best_lag, best_score = lag, score

    return best_lag, best_score

def cause_score(sim,ai,aj,lag_score,lag,patt_sim,sev_drift,
                w_sim=0.45,w_grad=0.20,w_lag=0.15,w_patt=0.10,w_sev=0.10):
    grad=max(aj-ai,0.0)
    lag_bonus=lag_score*(1.0 if lag>0 else 0.7 if lag==0 else 0.5)
    return w_sim*sim+w_grad*grad+w_lag*lag_bonus+w_patt*patt_sim+w_sev*sev_drift

def cosine_corr_matrix(embeds_np,norms_cache=None):
    norms=np.linalg.norm(embeds_np,axis=1,keepdims=True)
    norms=np.maximum(norms,1e-8)
    sim=(embeds_np@embeds_np.T)/(norms@norms.T)
    return np.clip(sim,-1.0,1.0),norms

# ---- Stable layout helper (reuses prior positions & adds new nodes) ----
def compute_layout_stable(G: nx.Graph, pos_cache: Optional[Dict]=None, seed: int=SEED):
    """
    Recompute a spring layout while reusing previous positions for nodes that still exist.
    Any new nodes get initialized automatically; old nodes help stabilize the layout.
    """
    try:
        if not pos_cache:
            return nx.spring_layout(G, seed=seed, k=0.95, iterations=350)
        pos_init = {n: pos_cache[n] for n in G.nodes() if n in pos_cache}
        return nx.spring_layout(G, seed=seed, k=0.95, iterations=350, pos=pos_init)
    except Exception:
        with contextlib.suppress(Exception):
            return nx.circular_layout(G)
        return {n: (0.0, 0.0) for n in G.nodes()}

def build_topology(chunk_embeds,chunk_stats,sim_matrix,chunk_size,base_topk,sim_threshold,infl_threshold,max_lag):
    G=nx.DiGraph();C=len(chunk_embeds);G.add_nodes_from(range(C))
    node_labels,node_anomaly,node_sizes={},[],[]
    for idx in range(C):
        cs=chunk_stats[idx]
        top_acts=", ".join([a.replace("_"," ").title() for a,_ in Counter(cs.acts_flat).most_common(3)]) or "No dominant activity"
        sev_str=" / ".join(f"{k}:{cs.severity_counts.get(k,0)}" for k in ("ERROR","WARN","INFO"))
        node_labels[idx]=f"{top_acts}\nAnom μ:{cs.anomaly_mean:.3f}\nSev {sev_str}"
        node_anomaly.append(cs.anomaly_mean)
        node_sizes.append(800+cs.anomaly_mean*1200)
    edges_rank=[]
    for i in range(C):
        ai=node_anomaly[i];wi=chunk_stats[i].anomaly_window;candidates=[]
        for j in range(C):
            if i==j:continue
            sim=float(sim_matrix[i,j])
            if sim<sim_threshold:continue
            aj=node_anomaly[j];wj=chunk_stats[j].anomaly_window
            # Defensive: compute lag safely; fall back if any issue
            try:
                lag,lag_score=cross_corr_lags(wi,wj,max_lag=max_lag) if wi.size>=3 and wj.size>=3 else (0,0.0)
            except Exception:
                lag,lag_score = 0,0.0
            patt_sim=jaccard_overlap(chunk_stats[i].pattern_set,chunk_stats[j].pattern_set)
            sev_drv=severity_drift(chunk_stats[i].severity_counts,chunk_stats[j].severity_counts)
            score=cause_score(sim,ai,aj,lag_score,lag,patt_sim,sev_drv)
            infl=max(0.0,sim*max(aj-ai,0.0))
            if score>=infl_threshold:
                evidence={"sim":sim,"lag":lag,"lag_score":lag_score,"patt":patt_sim,"sev_drift":sev_drv,"anom_grad":max(aj-ai,0.0)}
                candidates.append((j,score,infl,evidence))
        candidates.sort(key=lambda t:t[1],reverse=True)
        for j,score,infl,ev in candidates[:base_topk]:
            G.add_edge(i,j,score=score,weight=max(0.01,infl),**ev)
            edges_rank.append({"from_chunk":i+1,"to_chunk":j+1,"score":score,"influence":infl,**ev})
    return G,node_labels,node_anomaly,node_sizes,edges_rank

# ---- Time series helpers ----
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
    vol  = d["line"].resample(freq).count().rename("count").to_frame()
    anom = d["anomaly"].resample(freq).mean().rename("anomaly_mean").to_frame()
    sev  = pd.get_dummies(d["severity"]).resample(freq).sum().fillna(0)
    acts = pd.get_dummies(d["activity"]).resample(freq).sum().fillna(0)
    return vol, anom, sev, acts

def corr_heatmap_df(vol, anom, sev, acts):
    df = pd.concat([vol, anom, sev, acts], axis=1).fillna(0)
    if df.empty: return pd.DataFrame()
    return df.corr(method="pearson")

# ---- Predictive insights (heuristics) ----
def likely_next_activity(df_ts: pd.DataFrame) -> Optional[str]:
    acts = df_ts["activity"].dropna().tolist()
    if len(acts) < 2:
        return None
    pairs = list(zip(acts[:-1], acts[1:]))
    cnt = Counter(pairs)
    if not cnt:
        return None
    last = acts[-1]
    candidates = [(b, c) for (a, b), c in cnt.items() if a == last]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]

def predictive_summary(vol, anom_ts, sev_ts, acts_ts, df_ts, chunk_stats_list, edges_rank, ts_freq: str) -> str:
    lines = []
    # 1) Trend on anomaly mean (EWMA slope)
    risk_line = "Risk trend: unavailable"
    if not anom_ts.empty:
        s = anom_ts["anomaly_mean"].ewm(span=3).mean()
        if len(s) >= 2:
            slope = float(s.iloc[-1] - s.iloc[-2])
            level = float(s.iloc[-1])
            trend = "rising" if slope > 0.0 else "falling" if slope < 0.0 else "steady"
            risk_line = f"Anomaly EWMA is **{trend}** (last={level:.3f}, Δ={slope:+.3f}) over **{ts_freq}** bins."
    lines.append(f"• {risk_line}")

    # 2) Severity drift in the latest chunks
    if chunk_stats_list:
        last_k = chunk_stats_list[-min(3, len(chunk_stats_list)):]
        err_w = sum(cs.severity_counts.get("ERROR", 0) for cs in last_k)
        warn_w = sum(cs.severity_counts.get("WARN", 0) for cs in last_k)
        lines.append(f"• Latest {len(last_k)} chunk(s): ERROR={err_w}, WARN={warn_w}.")

    # 3) Top causal edges
    if edges_rank:
        top_edges = sorted(edges_rank, key=lambda x: x["score"], reverse=True)[:3]
        desc = "; ".join([f"C{e['from_chunk']}→C{e['to_chunk']} (score {e['score']:.2f}, lag {e['lag']}, sim {e['sim']:.2f})" for e in top_edges])
        lines.append(f"• Most influential causes: {desc}.")
    else:
        lines.append("• No causal edges above thresholds—lower min score/influence to explore hidden relations.")

    # 4) Likely next activity
    nxt = likely_next_activity(df_ts) if not df_ts.empty else None
    if nxt and nxt != "NONE":
        lines.append(f"• Likely next activity: **{nxt.replace('_',' ').title()}**.")
    else:
        lines.append("• Next activity prediction unavailable (insufficient transitions).")

    # 5) Top error drivers
    exc_all = Counter(); http_all = Counter(); errc_all = Counter()
    for cs in chunk_stats_list:
        exc_all.update(cs.exceptions)
        http_all.update(cs.http_codes)
        errc_all.update(cs.error_codes)
    if exc_all:
        e1 = ", ".join([f"{k}({v})" for k, v in exc_all.most_common(3)])
        lines.append(f"• Exceptions trending: {e1}.")
    if http_all:
        h1 = ", ".join([f"{k}({v})" for k, v in http_all.most_common(3)])
        lines.append(f"• HTTP status spikes: {h1}.")
    if errc_all:
        c1 = ", ".join([f"{k}({v})" for k, v in errc_all.most_common(3)])
        lines.append(f"• Custom error codes: {c1}.")

    # 6) Suggested actions (pattern-based hints)
    hints = []
    if any("AUTH_FAILURE" in a for cs in chunk_stats_list for a in cs.acts_flat):
        hints.append("Validate credentials/keys; check IAM policies and token TTL/rotation.")
    if any("TIMEOUT" in a for cs in chunk_stats_list for a in cs.acts_flat):
        hints.append("Investigate downstream latency; tune timeouts; add circuit breakers.")
    if any("CRASH_LOOP" in a for cs in chunk_stats_list for a in cs.acts_flat):
        hints.append("Inspect recent deploys/config; verify liveness/readiness; cap restart backoff.")
    if any(code in ("500","502","503","504") for cs in chunk_stats_list for code in cs.http_codes):
        hints.append("Scale affected service; verify DB/connectivity; enable graceful degradation.")
    if hints:
        lines.append("• Actions: " + " ".join([f"— {h}" for h in hints]))
    return "\n".join(lines)

# ---- Activity Summary helpers ----
def activity_counts_and_transitions(df_ts: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    if df_ts.empty:
        return pd.Series(dtype=int), pd.DataFrame(columns=["from","to","count"])
    acts = df_ts["activity"].fillna("NONE")
    counts = acts.value_counts()
    pairs = list(zip(acts[:-1], acts[1:])) if len(acts) >= 2 else []
    df_pairs = pd.DataFrame(pairs, columns=["from","to"])
    if df_pairs.empty:
        return counts, pd.DataFrame(columns=["from","to","count"])
    df_pairs["count"] = 1
    df_trans = df_pairs.groupby(["from","to"])["count"].sum().reset_index().sort_values("count", ascending=False)
    return counts, df_trans

def render_activity_summary(df_ts: pd.DataFrame, sev_ts: pd.DataFrame, acts_ts: pd.DataFrame) -> str:
    txt_parts = []
    total = len(df_ts)
    if total == 0:
        return "No logs available to summarize."
    top_counts, trans_df = activity_counts_and_transitions(df_ts)
    if not top_counts.empty:
        top3 = ", ".join([f"{a.replace('_',' ').title()} ({c})" for a,c in top_counts.head(3).items()])
        txt_parts.append(f"Dominant activities: {top3}.")
    if not trans_df.empty:
        trows = trans_df.head(3).to_dict("records")
        tdesc = "; ".join([f"{r['from'].replace('_',' ').title()} → {r['to'].replace('_',' ').title()} ({r['count']})" for r in trows])
        txt_parts.append(f"Major transitions: {tdesc}.")
    if not sev_ts.empty:
        last = sev_ts.tail(3).sum()
        err, warn, info = int(last.get("ERROR",0)), int(last.get("WARN",0)), int(last.get("INFO",0))
        txt_parts.append(f"Recent severity mix (last 3 bins): ERROR {err}, WARN {warn}, INFO {info}.")
    if not acts_ts.empty:
        agg = acts_ts.sum().sort_values(ascending=False).head(3)
        time_top = ", ".join([f"{a.replace('_',' ').title()} ({int(v)})" for a,v in agg.items()])
        txt_parts.append(f"Frequent activities over time: {time_top}.")
    return " ".join(txt_parts)

# ======================================================
# PART 3 — STREAMLIT APP (UI + Upload widget)
# ======================================================
st.set_page_config(page_title="Semantic Activity Log Analyzer", layout="wide")
st.title("🧠 Semantic Activity Log Analyzer — v32")
st.caption("Central Directed RCA • Activity Summary • Predictive Insights • Timeseries • Correlation • Error breakup • Persisted model")

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
    ts_freq   = st.selectbox("Timeseries frequency", ["30s","1min","5min","15min"], index=1)
    st.markdown("---")
    st.markdown("### Graph visibility")
    min_edge_score = st.slider("Min edge score to draw", 0.00, 1.00, 0.05, step=0.01)
    min_edge_infl  = st.slider("Min edge influence to draw", 0.00, 0.50, 0.01, step=0.01)
    arrow_scale    = st.slider("Arrow thickness scale", 1.0, 12.0, 6.0, step=0.5)
    arrow_offset   = st.slider("Arrow offset from node (0–0.2)", 0.00, 0.20, 0.08, step=0.01,
                               help="Pulls arrowhead back from target node so it stays visible.")
    edge_line_width = st.slider("Base edge line width", 0.5, 6.0, 2.8, step=0.1)
    edge_opacity    = st.slider("Edge line opacity", 0.1, 1.0, 0.75, step=0.05)
    st.markdown("---")
    # Persistence actions
    save_now = st.button("💾 Save current model to disk")
    reset_model = st.button("♻️ Reset persisted model (delete file)")

# 👉 Upload widget in MAIN PANE
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

# Session state for layout caching (stable topology)
if "pos_cache" not in st.session_state:
    st.session_state["pos_cache"] = None

if uploaded_file:
    raw_text = safe_decode_uploaded(uploaded_file)

    # Support JSON array logs → plain lines
    parsed_lines = try_parse_json_lines(raw_text)
    if parsed_lines:
        all_lines = parsed_lines
        preview_text = "\n".join(all_lines[:50])
    else:
        all_lines = raw_text.splitlines()
        preview_text = raw_text[:2000]

    st.subheader("📄 Raw Log Preview")
    st.code(preview_text)

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
            "timeseries_freq": ts_freq,
            "model_path": MODEL_PATH,
            "model_cached": os.path.exists(MODEL_PATH),
        })

    # --- CENTRAL RCA PANEL ---
    rca_ph = st.container()  # topology first
    # --- Additional panels ---
    progress_ph         = st.progress(0)
    anomaly_plot_ph     = st.empty()
    patterns_ph         = st.expander("🧩 Pattern Cards (per chunk)", expanded=False)
    activity_summary_ph = st.expander("📝 Activity Summary (counts • transitions • narrative)", expanded=True)
    ts_panel_ph         = st.expander("⏱️ Timeseries (volume, anomaly, severity, activities)", expanded=False)
    corr_panel_ph       = st.expander("📈 Correlation (resampled metrics)", expanded=False)
    error_breakdown_ph  = st.expander("🚨 Error Breakup (exceptions / HTTP / custom codes)", expanded=False)
    insights_ph         = st.expander("🧭 Predictive Insights (auto-generated)", expanded=True)
    sequence_ph         = st.expander("🔗 Sequence views (causal chains / Sankey)", expanded=False)

    # --- Accumulators ---
    lines = all_lines[:max_lines]
    all_anoms = []
    chunk_embeds_acc, acts_per_chunk_acc = [], []
    pattern_cards, pattern_sets = [], []
    chunk_stats_list: List[ChunkStats] = []
    norms_cache = None
    total_chunks = max(1, int(np.ceil(len(lines) / chunk_size)))
    redraw_every = max(1, total_chunks // 8)

    anomaly_x, anomaly_y = [], []
    ts_rows = []

    for ci in range(total_chunks):
        s, e = ci * chunk_size, min((ci + 1) * chunk_size, len(lines))
        lines_chunk = lines[s:e]

        # 1) Classify this chunk
        acts, anoms, embeds = classify_lines_batch(model, device, lines_chunk,
                                                   max_len=max_len, batch_size=batch_sz, use_amp=use_amp)
        all_anoms.extend(anoms)

        # 2) Chunk embed (pooled)
        pooled, _ = model.forward_chunk_adaptive(embeds)
        chunk_embeds_acc.append(pooled.squeeze(0).cpu())
        acts_per_chunk_acc.append(acts)

        # 3) Dynamic pattern mining & Chunk Analyzer
        card, patt_set = extract_chunk_patterns(lines_chunk, top_k=patt_topk)
        pattern_cards.append(card)
        pattern_sets.append(patt_set)

        cs = analyze_chunk(ci, lines_chunk, acts, anoms, patt_topk)
        chunk_stats_list.append(cs)

        # 4) Timeseries rows
        ts_rows.extend(append_timeline_rows(lines_chunk, acts, anoms, base_idx=s, extra_regex=None))

        # 5) Pattern Cards UI
        with patterns_ph:
            st.markdown(f"**Chunk {ci+1}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Unigrams:", ", ".join(card["unigrams"]) or "—")
                st.write("Bigrams:", ", ".join(card["bigrams"]) or "—")
            with col2:
                st.write("Trigrams:", ", ".join(card["trigrams"]) or "—")
                st.write("Exceptions:", ", ".join(card["exceptions"]) or "—")
            with col3:
                st.write("HTTP Codes:", ", ".join(card["http"]) or "—")
                st.write("Error Codes:", ", ".join(card["errors"]) or "—")

        # 6) Anomaly trend (partial)
        anomaly_x.extend(range(s, e))
        anomaly_y.extend(anoms)
        fig_tr, ax_tr = plt.subplots(figsize=(12, 3.6))
        ax_tr.plot(anomaly_x, anomaly_y, marker='o', linestyle='-', color='red', markersize=2)
        ax_tr.set_xlabel("Log Line Index"); ax_tr.set_ylabel("Anomaly Score")
        ax_tr.set_title("Line‑wise Anomaly Trend (partial)")
        anomaly_plot_ph.pyplot(fig_tr)

        # 7) RCA / topology (partial cadence; central render)
        if (ci % redraw_every == 0) or (ci == total_chunks - 1):
            C = len(chunk_embeds_acc)
            if C > 1:
                chunk_np = np.stack([e.numpy() for e in chunk_embeds_acc])  # [C, D]

                # Norms cache reset if shape changed
                if (norms_cache is None) or (norms_cache.shape[0] != chunk_np.shape[0]):
                    norms_cache = None
                sim_matrix, norms_cache = cosine_corr_matrix(chunk_np, norms_cache)

                # Build topology with multi-evidence
                G, node_labels, node_anomaly, node_sizes, edges_rank = build_topology(
                    chunk_embeds_acc, chunk_stats_list, sim_matrix, chunk_size,
                    base_topk, sim_threshold, infl_threshold, max_lag=lag_max
                )

                # Compute layout (stable; handles node changes)
                pos = compute_layout_stable(G, st.session_state.get("pos_cache"))
                st.session_state["pos_cache"] = pos

                # --- Central RCA Topology: Plotly (build traces from lists)
                if G.number_of_edges() > 0:
                    safe_nodes = [n for n in G.nodes() if n in pos]
                    xs = [pos[n][0] for n in safe_nodes]
                    ys = [pos[n][1] for n in safe_nodes]
                    xmin, xmax = (min(xs)-0.15 if xs else -1), (max(xs)+0.15 if xs else 1)
                    ymin, ymax = (min(ys)-0.15 if ys else -1), (max(ys)+0.15 if ys else 1)

                    an_min, an_max = (float(min(node_anomaly)) if node_anomaly else 0.0), (float(max(node_anomaly)) if node_anomaly else 1.0)
                    if an_max - an_min < 1e-8: an_max = an_min + 1e-8
                    def norm_color(a): return (a - an_min) / (an_max - an_min)

                    if plotly_ok:
                        # Node trace
                        node_x, node_y, node_text, node_color_vals, node_sizes_plotly = [], [], [], [], []
                        for n in G.nodes():
                            if n not in pos:  # guard missing keys
                                continue
                            x, y = pos[n]
                            node_x.append(x); node_y.append(y)
                            node_text.append(node_labels[n])
                            node_color_vals.append(norm_color(node_anomaly[n]))
                            node_sizes_plotly.append(14 + node_anomaly[n] * 20)
                        node_colors = sample_colorscale('Reds', node_color_vals) if node_color_vals else ['#d62728']*len(node_x)
                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers+text',
                            text=node_text, textposition='top center',
                            hoverinfo='text',
                            marker=dict(color=node_colors, size=node_sizes_plotly or [14]*len(node_x),
                                        line=dict(color='black', width=0.5))
                        )

                        def lag_color(lag: int) -> str:
                            return '#32CD32' if lag > 0 else '#FF8C00' if lag == 0 else '#1E90FF'

                        # Filter edges by thresholds + position guard
                        visible_edges = []
                        for u, v, data in G.edges(data=True):
                            if data.get('score', 0) < float(min_edge_score):   # score filter
                                continue
                            if data.get('weight', 0) < float(min_edge_infl):   # influence filter
                                continue
                            if (u not in pos) or (v not in pos):
                                continue
                            visible_edges.append((u, v, data))

                        # Accumulate per color (lists, not tuples)
                        edges_by_color = {
                            '#32CD32': {'x': [], 'y': []},
                            '#FF8C00': {'x': [], 'y': []},
                            '#1E90FF': {'x': [], 'y': []},
                        }
                        for (u, v, data) in visible_edges:
                            x0, y0 = pos[u]; x1, y1 = pos[v]
                            color = lag_color(data.get('lag', 0))
                            edges_by_color[color]['x'].extend([x0, x1, None])
                            edges_by_color[color]['y'].extend([y0, y1, None])

                        # Arrow annotations (axes-bound)
                        ann = []
                        def arrow_endpoint(x0, y0, x1, y1, offset=0.08):
                            dx, dy = x1 - x0, y1 - y0
                            L = max(np.hypot(dx, dy), 1e-6)
                            ux, uy = dx / L, dy / L
                            return x1 - ux * offset, y1 - uy * offset
                        for u, v, data in visible_edges:
                            x0, y0 = pos[u]; x1, y1 = pos[v]
                            ax, ay = arrow_endpoint(x0, y0, x1, y1, offset=float(arrow_offset))
                            width = max(2.0, float(arrow_scale) * float(data.get('weight', 0.05)))
                            ann.append(dict(
                                x=x1, y=y1, xref='x', yref='y',
                                ax=ax, ay=ay, axref='x', ayref='y',
                                showarrow=True, arrowhead=3, arrowsize=1,
                                arrowwidth=width, arrowcolor=lag_color(data.get('lag', 0)),
                                opacity=0.98
                            ))

                        # Hover midpoints
                        midx, midy, midtxt = [], [], []
                        for (u, v, data) in visible_edges:
                            x0, y0 = pos[u]; x1, y1 = pos[v]
                            midx.append((x0 + x1) / 2); midy.append((y0 + y1) / 2)
                            midtxt.append(
                                f"Cause → Effect<br>From: C{u+1} → To: C{v+1}<br>"
                                f"Score: {data.get('score',0):.3f} | Influence: {data.get('weight',0):.3f}<br>"
                                f"Sim: {data.get('sim',0):.2f} | Lag: {data.get('lag',0)} (r={data.get('lag_score',0):.2f})<br>"
                                f"AnomΔ: {data.get('anom_grad',0):.2f} | SevΔ: {data.get('sev_drift',0):.2f} | Patt: {data.get('patt',0):.2f}"
                            )
                        hover_trace = go.Scatter(
                            x=midx, y=midy, mode='markers',
                            marker=dict(size=12, color='rgba(0,0,0,0)'),
                            hoverinfo='text', text=midtxt,
                            showlegend=False
                        )

                        # Build color traces from accumulated lists
                        color_traces = []
                        for color_hex, pts in edges_by_color.items():
                            if pts['x']:
                                color_traces.append(
                                    go.Scatter(
                                        x=pts['x'], y=pts['y'], mode='lines',
                                        line=dict(width=edge_line_width, color=color_hex),
                                        opacity=float(edge_opacity),
                                        hoverinfo='none',
                                        showlegend=False
                                    )
                                )

                        fig = go.Figure(
                            data=color_traces + [node_trace, hover_trace],
                            layout=go.Layout(
                                title="RCA Topology — Directed cause → effect (multi‑evidence)",
                                showlegend=False, height=650,
                                template='plotly_white',
                                margin=dict(l=10, r=10, t=48, b=10),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[xmin, xmax]),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[ymin, ymax]),
                                annotations=ann,
                                hovermode='closest'
                            )
                        )
                        rca_ph.plotly_chart(fig, use_container_width=True)

                    else:
                        # Matplotlib fallback with visible edges
                        fig2, ax2 = plt.subplots(figsize=(12, 7))
                        cmap = plt.cm.Reds
                        norm_colors = [(a - an_min) / (an_max - an_min) for a in node_anomaly]
                        nx.draw_networkx_nodes(G, pos, node_color=[cmap(c) for c in norm_colors],
                                               node_size=node_sizes, ax=ax2)
                        def lag_color(lag): return "#32CD32" if lag>0 else "#FF8C00" if lag==0 else "#1E90FF"
                        edge_widths = [max(1.8, 10 * G[u][v]['weight']) for u, v in G.edges()]
                        for (u, v, data), w in zip(G.edges(data=True), edge_widths):
                            if (u not in pos) or (v not in pos):
                                continue
                            nx.draw_networkx_edges(
                                G, pos, edgelist=[(u,v)],
                                edge_color=lag_color(data.get('lag',0)),
                                width=w, arrows=True, arrowstyle='-|>', arrowsize=30, ax=ax2,
                                connectionstyle="arc3,rad=0.06"
                            )
                        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_color='black', ax=ax2)
                        ax2.set_title("RCA Topology — Directed cause → effect (multi‑evidence)")
                        ax2.axis('off')
                        rca_ph.pyplot(fig2)

                else:
                    rca_ph.info("No edges above thresholds yet. Reduce 'Min edge score' or 'Min influence' to reveal relations.")

                # --- Sequence / edges table
                if edges_rank:
                    df_edges = pd.DataFrame(edges_rank).sort_values(by="score", ascending=False)
                    with sequence_ph:
                        st.subheader("🔗 Top causal chains (sorted by score)")
                        st.dataframe(df_edges.head(20), use_container_width=True)
                        with contextlib.suppress(Exception):
                            csv_bytes = df_edges.to_csv(index=False).encode("utf-8")
                            st.download_button("⬇️ Download RCA edges (CSV)", csv_bytes, "rca_edges.csv", "text/csv")

                        # Optional Sankey
                        if plotly_ok and len(df_edges) > 0:
                            sankey_df = df_edges.copy()
                            sankey_df["from"] = sankey_df["from_chunk"].astype(str)
                            sankey_df["to"]   = sankey_df["to_chunk"].astype(str)
                            nodes = sorted(set(sankey_df["from"]) | set(sankey_df["to"]))
                            index = {n:i for i,n in enumerate(nodes)}
                            sankey = go.Figure(data=[go.Sankey(
                                node=dict(label=[f"C{n}" for n in nodes], pad=15, thickness=20),
                                link=dict(
                                    source=[index[x] for x in sankey_df["from"]],
                                    target=[index[y] for y in sankey_df["to"]],
                                    value=sankey_df["influence"].clip(lower=0.001),
                                    hovertemplate=("C%{source.label} → C%{target.label}<br>"
                                                   "Score %{customdata:.3f} | Influence %{value:.3f}<extra></extra>"),
                                    customdata=sankey_df["score"]
                                )
                            )])
                            sankey.update_layout(title_text="Sequence flow (Sankey) — influence-weighted", height=420)
                            sequence_ph.plotly_chart(sankey, use_container_width=True)

        progress_ph.progress(int(((ci + 1) / total_chunks) * 100))

    # ===== After processing all chunks: Activity Summary, Timeseries, Correlation, Error breakup, Predictive =====
    df_ts = pd.DataFrame(ts_rows, columns=["idx","ts","line","severity","activity","anomaly"])

    # -- Activity Summary
    with activity_summary_ph:
        counts, trans_df = activity_counts_and_transitions(df_ts)
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Top activities")
            if counts.empty:
                st.write("—")
            else:
                st.write(pd.DataFrame(counts, columns=["count"]))
        with colB:
            st.subheader("Top transitions")
            if trans_df.empty:
                st.write("—")
            else:
                st.write(trans_df.head(15))

        # Narrative
        vol, anom_ts, sev_ts, acts_ts = resample_metrics(df_ts, freq=ts_freq)
        narrative = render_activity_summary(df_ts, sev_ts, acts_ts)
        st.markdown(f"**Summary:** {narrative}")

    # -- Timeseries resample & visuals (collapsed by default)
    vol, anom_ts, sev_ts, acts_ts = resample_metrics(df_ts, freq=ts_freq)
    with ts_panel_ph:
        if plotly_ok and not vol.empty:
            fig_vol = px.line(vol.reset_index(), x="ts", y="count", title=f"Log volume ({ts_freq} bins)")
            fig_an  = px.line(anom_ts.reset_index(), x="ts", y="anomaly_mean", title=f"Anomaly mean ({ts_freq} bins)")
            st.plotly_chart(fig_vol, use_container_width=True)
            st.plotly_chart(fig_an, use_container_width=True)
            if not sev_ts.empty:
                sev_long = sev_ts.reset_index().melt(id_vars="ts", var_name="severity", value_name="count")
                fig_sev = px.area(sev_long, x="ts", y="count", color="severity",
                                  title=f"Severity counts over time ({ts_freq})", groupnorm="fraction")
                st.plotly_chart(fig_sev, use_container_width=True)
            if not acts_ts.empty:
                acts_long = acts_ts.reset_index().melt(id_vars="ts", var_name="activity", value_name="count")
                top_acts = acts_long.groupby("activity")["count"].sum().sort_values(ascending=False).head(6).index
                acts_filtered = acts_long[acts_long["activity"].isin(top_acts)]
                fig_act = px.line(acts_filtered, x="ts", y="count", color="activity",
                                  title=f"Top activity counts over time ({ts_freq})")
                st.plotly_chart(fig_act, use_container_width=True)
        else:
            st.info("Timeseries plots need valid timestamps and (optionally) Plotly. No timestamps found or Plotly unavailable.")

    # -- Correlation heatmap (resampled metrics)
    with corr_panel_ph:
        corr_df = corr_heatmap_df(vol, anom_ts, sev_ts, acts_ts)
        if not corr_df.empty:
            if plotly_ok:
                fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto",
                                     title="Pearson correlation among resampled metrics")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                fig_c, ax_c = plt.subplots(figsize=(8, 6))
                im = ax_c.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
                ax_c.set_xticks(range(len(corr_df.columns))); ax_c.set_xticklabels(corr_df.columns, rotation=90)
                ax_c.set_yticks(range(len(corr_df.index)));   ax_c.set_yticklabels(corr_df.index)
                ax_c.set_title("Correlation among resampled metrics")
                fig_c.colorbar(im, ax=ax_c)
                st.pyplot(fig_c)
        else:
            st.info("Correlation heatmap unavailable: not enough resampled metrics (timestamps may be missing).")

    # -- Error breakup (overall + per-chunk severity stacks)
    with error_breakdown_ph:
        exc_all = Counter(); http_all = Counter(); errc_all = Counter()
        for cs in chunk_stats_list:
            exc_all.update(cs.exceptions)
            http_all.update(cs.http_codes)
            errc_all.update(cs.error_codes)

        def plot_bar(counter: Counter, title: str):
            if not counter:
                st.write(f"{title}: —")
                return
            s = pd.Series(counter).sort_values(ascending=False).head(20)
            if plotly_ok:
                fig_b = px.bar(x=s.index, y=s.values, title=title)
                fig_b.update_layout(xaxis_title="Code/Exception", yaxis_title="Count")
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                fig_b, ax_b = plt.subplots(figsize=(10, 4))
                ax_b.bar(s.index, s.values, color="#d62728")
                ax_b.set_title(title); ax_b.set_ylabel("Count"); ax_b.tick_params(axis='x', rotation=75)
                st.pyplot(fig_b)

        plot_bar(exc_all, "Exceptions (top 20)")
        plot_bar(http_all, "HTTP Status (top 20)")
        plot_bar(errc_all, "Custom error codes (top 20)")

        sev_by_chunk = pd.DataFrame([
            {"chunk": cs.index, **{k: cs.severity_counts.get(k, 0) for k in ["ERROR","WARN","INFO","OTHER"]}}
            for cs in chunk_stats_list
        ])
        if not sev_by_chunk.empty:
            sev_long = sev_by_chunk.melt(id_vars="chunk", var_name="severity", value_name="count")
            if plotly_ok:
                fig_stack = px.bar(sev_long, x="chunk", y="count", color="severity",
                                   title="Severity breakup per chunk", barmode="stack")
                st.plotly_chart(fig_stack, use_container_width=True)
            else:
                fig_s, ax_s = plt.subplots(figsize=(12, 4))
                bottom = np.zeros(len(sev_by_chunk))
                colors = {"ERROR":"#d62728","WARN":"#ff7f0e","INFO":"#2ca02c","OTHER":"#7f7f7f"}
                for sev in ["ERROR","WARN","INFO","OTHER"]:
                    vals = sev_by_chunk[sev].values
                    ax_s.bar(sev_by_chunk["chunk"], vals, bottom=bottom, color=colors[sev], label=sev)
                    bottom += vals
                ax_s.set_title("Severity breakup per chunk"); ax_s.set_xlabel("Chunk"); ax_s.set_ylabel("Count")
                ax_s.legend()
                st.pyplot(fig_s)

    # -- Predictive Insights (summary)
    with insights_ph:
        summary_text = predictive_summary(vol, anom_ts, sev_ts, acts_ts, df_ts, chunk_stats_list, edges_rank if 'edges_rank' in locals() else [], ts_freq)
        st.markdown(summary_text)
        with contextlib.suppress(Exception):
            st.download_button("⬇️ Download Predictive Insights (Markdown)", summary_text.encode("utf-8"), "predictive_insights.md", "text/markdown")

    st.success("✅ Analysis complete — stable topology, activity summary, and predictive insights generated.")

else:
    st.info("Upload a log file (txt/log/csv/json) to begin the analysis.")
