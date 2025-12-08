# ============================================================
# app.py — SHATCAR Log Analyzer (spaCy + Transformer + Chat)
# ============================================================

import os
import re
import json
import sqlite3
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Optional robust date parsing
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DB_PATH = "logs.db"
SPACY_MODEL = "en_core_web_sm"

SHATCAR_MODEL_PATH = "shatcar_model.pt"
SHATCAR_VOCAB_PATH = "shatcar_vocab.json"

SEM_DIM = 300
STRUCT_MAX_LEN = 32

SHATCAR_D_MODEL = 192
SHATCAR_NUM_CLASSES = 2

MAX_SERVICES = 200
MAX_LEVEL_EMBED = 10

SAFE_MIN_TS = datetime(1970, 1, 1)
SAFE_MAX_TS = datetime(2100, 1, 1)


# -----------------------------------------------------------------------------
# BASIC STRUCTURES
# -----------------------------------------------------------------------------
@dataclass
class LogEvent:
    raw: str
    timestamp: Optional[datetime]
    level: Optional[str]
    service: Optional[str]
    trace_id: Optional[str]
    message: str
    line_no: int


def normalize_ts(ts: Optional[datetime]) -> datetime:
    """Normalize timestamps to naive datetimes and clamp to safe range."""
    if ts is None:
        return SAFE_MIN_TS
    if ts.tzinfo is not None:
        ts = ts.astimezone().replace(tzinfo=None)
    if ts.year < SAFE_MIN_TS.year:
        ts = SAFE_MIN_TS
    if ts.year > SAFE_MAX_TS.year:
        ts = SAFE_MAX_TS
    return ts


# -----------------------------------------------------------------------------
# DB HELPERS
# -----------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            line TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_logs_to_db(lines: List[str]):
    if not lines:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executemany("INSERT INTO logs(line) VALUES (?)", [(l,) for l in lines])
    conn.commit()
    conn.close()


def load_logs_from_db() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT line FROM logs")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]


# -----------------------------------------------------------------------------
# LOG PARSING (TEXT + JSON)
# -----------------------------------------------------------------------------
TIMESTAMP_REGEXES = [
    r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?)",
    r"^\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)(?:\s|$)",
    r"^\s*(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})",
]

JSON_TIMESTAMP_KEYS = ["timestamp", "time", "ts"]
JSON_LEVEL_KEYS = ["level", "severity", "loglevel"]
JSON_SERVICE_KEYS = ["service", "svc", "module"]
JSON_TRACE_KEYS = ["trace_id", "trace", "req_id", "request_id", "correlation_id"]
JSON_MESSAGE_KEYS = ["message", "msg", "event", "log", "text"]


def parse_timestamp(line: str) -> Tuple[Optional[datetime], str]:
    """Extract timestamp at beginning if present, return (datetime, rest)."""
    for pat in TIMESTAMP_REGEXES:
        m = re.search(pat, line)
        if m:
            ts_str = m.group(1)
            rest = line[m.end():].lstrip()
            dt = None
            if HAS_DATEUTIL:
                try:
                    dt = date_parser.parse(ts_str)
                except Exception:
                    dt = None
            else:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    dt = None
            return dt, rest

    if HAS_DATEUTIL:
        try:
            dt = date_parser.parse(line, fuzzy=True)
            return dt, line
        except Exception:
            pass

    return None, line


def parse_level(line: str) -> Optional[str]:
    """
    Robust log level detection:
    - INFO, WARN, WARNING, ERROR, DEBUG, TRACE, CRITICAL, FATAL
    - [INFO], level=INFO, severity=ERROR, log_level=DEBUG, etc.
    """
    m = re.search(r"\b(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)\b", line, re.IGNORECASE)
    if m:
        lvl = m.group(1).upper()
        return "WARN" if lvl == "WARNING" else lvl

    m = re.search(r"[\[\{\(<](INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)[\]\}\)>]", line, re.IGNORECASE)
    if m:
        lvl = m.group(1).upper()
        return "WARN" if lvl == "WARNING" else lvl

    m = re.search(
        r"(level|severity|log[_\-]?level)\s*[:=]\s*(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)",
        line,
        re.IGNORECASE,
    )
    if m:
        lvl = m.group(2).upper()
        return "WARN" if lvl == "WARNING" else lvl

    return None


SERVICE_PATTERNS = [
    re.compile(r"\bsvc=([A-Za-z0-9_\-\.]+)\b"),
    re.compile(r"\bservice=([A-Za-z0-9_\-\.]+)\b"),
    re.compile(r"^\s*\[([A-Za-z0-9_\-\.]+)\]"),
]

TRACE_PATTERNS = [
    re.compile(r"\b(trace_id|trace|request_id|req_id|corr_id|correlation_id)=([A-Za-z0-9\-\_:]+)\b")
]


def parse_service(line: str) -> Optional[str]:
    for pat in SERVICE_PATTERNS:
        m = pat.search(line)
        if m:
            return m.group(1)
    return None


def parse_trace_id(line: str) -> Optional[str]:
    for pat in TRACE_PATTERNS:
        m = pat.search(line)
        if m:
            return m.group(2)
    return None


def split_message(line: str) -> str:
    s = re.sub(r"\[(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)\]", "", line, flags=re.IGNORECASE)
    s = re.sub(
        r"\b(level|severity|log[_\-]?level)\s*[:=]\s*(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)\b",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s or line.strip()


def try_parse_json_log(line: str) -> Optional[LogEvent]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    # Timestamp
    ts = None
    for k in JSON_TIMESTAMP_KEYS:
        if k in obj:
            v = obj[k]
            try:
                if isinstance(v, (int, float)):
                    ts = datetime.fromtimestamp(v / 1000.0)
                elif isinstance(v, str):
                    ts = date_parser.parse(v) if HAS_DATEUTIL else datetime.fromisoformat(v.replace("Z", "+00:00"))
            except Exception:
                ts = None
            break

    # Level
    lvl = None
    for k in JSON_LEVEL_KEYS:
        if k in obj:
            val = str(obj[k]).upper()
            if val in ["INFO", "WARN", "WARNING", "ERROR", "DEBUG", "TRACE", "CRITICAL", "FATAL"]:
                lvl = "WARN" if val == "WARNING" else val
            break

    # Service
    svc = None
    for k in JSON_SERVICE_KEYS:
        if k in obj:
            svc = str(obj[k])
            break

    # Trace
    trace = None
    for k in JSON_TRACE_KEYS:
        if k in obj:
            trace = str(obj[k])
            break

    # Message
    msg = None
    for k in JSON_MESSAGE_KEYS:
        if k in obj:
            msg = str(obj[k])
            break
    if msg is None:
        msg = json.dumps(obj)

    return LogEvent(
        raw=line.strip(),
        timestamp=ts,
        level=lvl,
        service=svc,
        trace_id=trace,
        message=msg,
        line_no=-1,
    )


def parse_logs(lines: List[str]) -> List[LogEvent]:
    events: List[LogEvent] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue

        j = try_parse_json_log(s)
        if j is not None:
            j.line_no = i
            events.append(j)
            continue

        ts, rem = parse_timestamp(s)
        lvl = parse_level(s)
        svc = parse_service(rem)
        trace = parse_trace_id(rem)
        msg = split_message(rem)

        events.append(
            LogEvent(
                raw=s,
                timestamp=ts,
                level=lvl,
                service=svc,
                trace_id=trace,
                message=msg,
                line_no=i,
            )
        )

    events.sort(key=lambda e: (normalize_ts(e.timestamp), e.line_no))
    return events


# -----------------------------------------------------------------------------
# TEMPORAL ANALYTICS (SAFE)
# -----------------------------------------------------------------------------
def build_time_buckets(events: List[LogEvent], bucket: str = "minute"):
    """
    Safe bucketing:
    - If any timestamp exists: use datetime buckets (minute resolution), clamped to safe range.
    - If no timestamps: use integer buckets based on line index.
    """
    if not events:
        return {}, True

    have_any_ts = any(e.timestamp is not None for e in events)
    buckets = defaultdict(lambda: {"total": 0, "errors": 0})

    if have_any_ts:
        for e in events:
            if e.timestamp is None:
                key = SAFE_MIN_TS
            else:
                ts = normalize_ts(e.timestamp)
                key = ts.replace(second=0, microsecond=0)
            buckets[key]["total"] += 1
            if e.level in ("ERROR", "CRITICAL", "FATAL"):
                buckets[key]["errors"] += 1
    else:
        for e in events:
            key = e.line_no // 20
            buckets[key]["total"] += 1
            if e.level in ("ERROR", "CRITICAL", "FATAL"):
                buckets[key]["errors"] += 1

    return buckets, have_any_ts


def plot_temporal_analytics(events: List[LogEvent]):
    buckets, is_time = build_time_buckets(events)
    if not buckets:
        st.info("No temporal data.")
        return

    keys = sorted(buckets.keys())
    totals = [buckets[k]["total"] for k in keys]
    errors = [buckets[k]["errors"] for k in keys]

    plt.figure(figsize=(10, 4))

    if is_time:
        xs = [mdates.date2num(k) for k in keys]
        plt.plot_date(xs, totals, "-", label="Total logs")
        plt.plot_date(xs, errors, "-", label="Error logs")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gcf().autofmt_xdate()
        plt.xlabel("Time")
    else:
        plt.plot(keys, totals, "-", label="Total logs")
        plt.plot(keys, errors, "-", label="Error logs")
        plt.xlabel("Bucket index (by lines)")

    plt.ylabel("Count")
    plt.title("Log Volume & Errors Over Time")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()


# -----------------------------------------------------------------------------
# STRUCTURAL PATTERNS
# -----------------------------------------------------------------------------
NUM_PATTERN = re.compile(r"\b\d+\b")
HEX_PATTERN = re.compile(r"\b0x[0-9A-Fa-f]+\b")
UUID_PATTERN = re.compile(r"\b[0-9a-fA-F\-]{32,36}\b")
IP_PATTERN = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")


def message_template(msg: str) -> str:
    s = HEX_PATTERN.sub("<HEX>", msg)
    s = UUID_PATTERN.sub("<UUID>", s)
    s = IP_PATTERN.sub("<IP>", s)
    s = NUM_PATTERN.sub("<NUM>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_structural_patterns(events: List[LogEvent]):
    counter = Counter()
    examples: Dict[str, str] = {}
    for e in events:
        t = message_template(e.message)
        counter[t] += 1
        if t not in examples:
            examples[t] = e.message

    rows = []
    for tmpl, cnt in counter.most_common(15):
        rows.append(
            {
                "Template": tmpl,
                "Count": cnt,
                "Example": examples[tmpl][:200],
            }
        )
    return rows


# -----------------------------------------------------------------------------
# ROOT CAUSE ANALYSIS (HEURISTIC)
# -----------------------------------------------------------------------------
def root_cause_analysis(events: List[LogEvent]) -> Dict[str, Any]:
    traces: Dict[str, List[LogEvent]] = defaultdict(list)
    for e in events:
        key = e.trace_id or "__NO_TRACE__"
        traces[key].append(e)

    root_services = Counter()
    root_messages = Counter()
    per_trace_results = []

    for trace_id, evs in traces.items():
        evs_sorted = sorted(evs, key=lambda e: (normalize_ts(e.timestamp), e.line_no))
        root_event = None
        for e in evs_sorted:
            if e.level in ("ERROR", "CRITICAL", "FATAL"):
                root_event = e
                break
        if root_event:
            svc = root_event.service or "UNK"
            msg = root_event.message[:200]
            root_services[svc] += 1
            root_messages[msg] += 1
            per_trace_results.append(
                {
                    "trace_id": trace_id,
                    "service": svc,
                    "message": msg,
                    "time": root_event.timestamp.isoformat() if root_event.timestamp else None,
                }
            )

    return {
        "per_trace": per_trace_results,
        "top_services": root_services.most_common(10),
        "top_messages": root_messages.most_common(10),
    }


# -----------------------------------------------------------------------------
# BASELINE TOPOLOGY (TRACE-BASED)
# -----------------------------------------------------------------------------
def build_topology_baseline(events: List[LogEvent]) -> nx.DiGraph:
    G = nx.DiGraph()
    traces: Dict[str, List[LogEvent]] = defaultdict(list)

    for e in events:
        key = e.trace_id or "__NOTRACE__"
        traces[key].append(e)

    for tid, evs in traces.items():
        evs = sorted(evs, key=lambda e: (normalize_ts(e.timestamp), e.line_no))
        last = None
        for e in evs:
            s = e.service or "UNK"
            if s not in G:
                G.add_node(s)
            if last and last != s:
                if G.has_edge(last, s):
                    G[last][s]["weight"] += 1
                else:
                    G.add_edge(last, s, weight=1)
            last = s

    return G


def draw_topology(G: nx.DiGraph):
    if not G or len(G.nodes()) == 0:
        st.info("No topology detected.")
        return

    plt.figure(figsize=(13, 9))
    plt.axis("off")

    if len(G.nodes()) <= 3:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.9, seed=42)

    def color(s: str):
        s = s.lower()
        if "gateway" in s or "api" in s:
            return "#ff7f0e"
        if "auth" in s:
            return "#2ca02c"
        if "db" in s:
            return "#d62728"
        if "cache" in s:
            return "#9467bd"
        if "web" in s:
            return "#1f77b4"
        return "#7f7f7f"

    node_colors = [color(n) for n in G.nodes()]
    weights = [G[u][v].get("weight", 0.1) for u, v in G.edges()]

    if weights:
        maxw = max(weights)
        widths = [2 + 6 * (w / maxw) for w in weights]
    else:
        widths = 1

    edge_colors = ["#d62728" if G[u][v].get("weight", 0) > 0.7 else "#555555" for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800)
    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
    )
    nx.draw_networkx_labels(G, pos, font_size=13, font_weight="bold")

    edge_labels = {(u, v): round(G[u][v].get("weight", 0.0), 2) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)

    st.pyplot(plt.gcf())
    plt.close()


# -----------------------------------------------------------------------------
# spaCy DEEP PARSING + SEMANTIC VECTOR
# -----------------------------------------------------------------------------
def advanced_spacy_parse(msg: str, nlp):
    """
    Deep semantic extraction using spaCy:
    - Nouns
    - Verbs
    - Subject-Verb-Object triples
    """
    doc = nlp(msg)

    nouns = []
    verbs = []
    svos = []

    for token in doc:
        if token.dep_ in ("ROOT", "conj") and token.pos_ == "VERB":
            subj = [w.text for w in token.lefts if w.dep_ == "nsubj"]
            obj = [w.text for w in token.rights if w.dep_ in ("dobj", "attr", "pobj")]
            if subj and obj:
                svos.append((subj[0], token.lemma_, obj[0]))

    for tok in doc:
        if tok.pos_ == "NOUN" and not tok.is_stop:
            nouns.append(tok.lemma_.lower())
        if tok.pos_ == "VERB" and not tok.is_stop:
            verbs.append(tok.lemma_.lower())

    return {"nouns": nouns, "verbs": verbs, "svos": svos}


def semantic_vector(msg: str, nlp) -> List[float]:
    """
    Hybrid semantic vector:
      - spaCy dense vector (if available)
      - plus hashed contributions from nouns/verbs/SVOs
    """
    doc = nlp(msg)
    if doc.vector is not None and len(doc.vector) > 0:
        base = doc.vector.tolist()
        if len(base) > SEM_DIM:
            base = base[:SEM_DIM]
        else:
            base = base + [0.0] * (SEM_DIM - len(base))
    else:
        base = [0.0] * SEM_DIM

    info = advanced_spacy_parse(msg, nlp)

    extra = [0.0] * SEM_DIM
    for n in info["nouns"][:10]:
        idx = abs(hash("N:" + n)) % SEM_DIM
        extra[idx] += 1.0
    for v in info["verbs"][:10]:
        idx = abs(hash("V:" + v)) % SEM_DIM
        extra[idx] += 1.5
    for s, v, o in info["svos"][:5]:
        idx = abs(hash(f"SVO:{s}:{v}:{o}")) % SEM_DIM
        extra[idx] += 2.0

    combined = [a + b for a, b in zip(base, extra)]
    norm = (sum(x * x for x in combined) ** 0.5) or 1.0
    return [x / norm for x in combined]


# -----------------------------------------------------------------------------
# SHATCAR TRANSFORMER
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class SHATCAR(nn.Module):
    """
    Structural-Hierarchical Adaptive Transformer with Context-Aware Routing.
    Stronger attention:
      - d_model = 192, nhead = 6
      - 2 encoder layers per level, 3 levels
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = SHATCAR_D_MODEL,
        nhead: int = 6,
        num_levels: int = 3,
        num_classes: int = SHATCAR_NUM_CLASSES,
        max_len: int = STRUCT_MAX_LEN,
        max_services: int = MAX_SERVICES,
        max_levels_embed: int = MAX_LEVEL_EMBED,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_levels = num_levels

        self.struct_embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)

        self.semantic_proj = nn.Linear(SEM_DIM, d_model, bias=False)
        self.level_embed = nn.Embedding(max_levels_embed, d_model)
        self.service_embed = nn.Embedding(max_services, d_model)
        self.time_proj = nn.Linear(4, d_model)

        self.levels = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.2,
                    batch_first=True,
                ),
                num_layers=2,
            )
            for _ in range(num_levels)
        ])

        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_levels),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(
        self,
        struct_ids: torch.Tensor,
        mask: torch.Tensor,
        semantic_vecs: torch.Tensor,
        level_ids: torch.Tensor,
        service_ids: torch.Tensor,
        time_feats: torch.Tensor,
    ):
        x = self.struct_embed(struct_ids)  # (B,L,D)
        x = self.pos(x)

        sem = self.semantic_proj(semantic_vecs)
        lvl = self.level_embed(level_ids)
        svc = self.service_embed(service_ids)
        tim = self.time_proj(time_feats)

        context = sem + lvl + svc + tim  # (B,D)
        fused = x + context.unsqueeze(1)

        gating = self.router(context)  # (B, num_levels)

        outputs = []
        src_key_padding_mask = (mask == 0)
        for level in self.levels:
            out = level(fused, src_key_padding_mask=src_key_padding_mask)
            outputs.append(out)

        stacked = torch.stack(outputs, dim=1)  # (B,num_levels,L,D)
        gating = gating.view(-1, self.num_levels, 1, 1)
        fused_final = (stacked * gating).sum(dim=1)  # (B,L,D)

        mask_f = mask.unsqueeze(-1).float()
        summed = (fused_final * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom  # (B,D)

        pooled = self.dropout(pooled)
        logits = self.cls(pooled)
        return logits

    def pooled_embedding(
        self,
        struct_ids: torch.Tensor,
        mask: torch.Tensor,
        semantic_vecs: torch.Tensor,
        level_ids: torch.Tensor,
        service_ids: torch.Tensor,
        time_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Return pooled latent embedding (B,D) instead of logits."""
        x = self.struct_embed(struct_ids)
        x = self.pos(x)

        sem = self.semantic_proj(semantic_vecs)
        lvl = self.level_embed(level_ids)
        svc = self.service_embed(service_ids)
        tim = self.time_proj(time_feats)

        context = sem + lvl + svc + tim
        fused = x + context.unsqueeze(1)

        gating = self.router(context)

        outputs = []
        src_key_padding_mask = (mask == 0)
        for level in self.levels:
            out = level(fused, src_key_padding_mask=src_key_padding_mask)
            outputs.append(out)

        stacked = torch.stack(outputs, dim=1)
        gating = gating.view(-1, self.num_levels, 1, 1)
        fused_final = (stacked * gating).sum(dim=1)

        mask_f = mask.unsqueeze(-1).float()
        summed = (fused_final * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return pooled  # (B,D)


# -----------------------------------------------------------------------------
# SHATCAR DATA HELPERS
# -----------------------------------------------------------------------------
def build_shatcar_vocab(events: List[LogEvent]) -> Dict[str, int]:
    freq = Counter()
    for e in events:
        tmpl = message_template(e.message)
        for t in tmpl.split():
            freq[t] += 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok in freq.keys():
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def template_to_ids(tmpl: str, vocab: Dict[str, int], max_len: int = STRUCT_MAX_LEN):
    tokens = tmpl.split()
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab["<PAD>"])
        mask.append(0)
    return ids, mask


def level_to_id(level: Optional[str]) -> int:
    lvl = (level or "UNKNOWN").upper()
    mapping = {
        "DEBUG": 0,
        "INFO": 1,
        "WARN": 2,
        "WARNING": 2,
        "ERROR": 3,
        "CRITICAL": 4,
        "FATAL": 5,
        "TRACE": 6,
        "UNKNOWN": 7,
    }
    return mapping.get(lvl, 7)


def build_service_mapping(events: List[LogEvent]) -> Dict[str, int]:
    services = sorted({e.service or "UNK" for e in events})
    mapping = {}
    idx = 0
    for s in services:
        if idx >= MAX_SERVICES - 1:
            break
        mapping[s] = idx
        idx += 1
    mapping["__OTHER__"] = MAX_SERVICES - 1
    return mapping


def service_to_id(service: Optional[str], mapping: Dict[str, int]) -> int:
    s = service or "UNK"
    return mapping.get(s, mapping["__OTHER__"])


def time_features(ts: Optional[datetime]) -> List[float]:
    if ts is None:
        return [0.0, 0.0, 0.0, 0.0]
    ts = normalize_ts(ts)
    hour = ts.hour / 23.0
    minute = ts.minute / 59.0
    second = ts.second / 59.0
    is_weekend = 1.0 if ts.weekday() >= 5 else 0.0
    return [hour, minute, second, is_weekend]


def get_shatcar_dataset(events: List[LogEvent], vocab: Dict[str, int], nlp):
    if not events:
        return None

    svc_map = build_service_mapping(events)

    X_ids, X_masks = [], []
    sems, lvl_ids, svc_ids, time_feats, labels = [], [], [], [], []

    for e in events:
        tmpl = message_template(e.message)
        if not tmpl:
            continue
        ids, mask = template_to_ids(tmpl, vocab)
        X_ids.append(ids)
        X_masks.append(mask)
        sems.append(semantic_vector(e.message, nlp))
        lvl_ids.append(level_to_id(e.level))
        svc_ids.append(service_to_id(e.service, svc_map))
        time_feats.append(time_features(e.timestamp))
        labels.append(1 if e.level in ("ERROR", "CRITICAL", "FATAL") else 0)

    if not X_ids:
        return None

    X_ids = torch.tensor(X_ids, dtype=torch.long)
    X_masks = torch.tensor(X_masks, dtype=torch.long)
    sems = torch.tensor(sems, dtype=torch.float32)
    lvl_ids = torch.tensor(lvl_ids, dtype=torch.long)
    svc_ids = torch.tensor(svc_ids, dtype=torch.long)
    time_feats = torch.tensor(time_feats, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "X_ids": X_ids,
        "X_masks": X_masks,
        "sems": sems,
        "lvl_ids": lvl_ids,
        "svc_ids": svc_ids,
        "time_feats": time_feats,
        "y": labels,
        "svc_map": svc_map,
    }


def load_shatcar_vocab() -> Optional[Dict[str, int]]:
    if not os.path.exists(SHATCAR_VOCAB_PATH):
        return None
    with open(SHATCAR_VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_shatcar_vocab(vocab: Dict[str, int]):
    with open(SHATCAR_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f)


def load_shatcar_model(vocab_size: int, device: torch.device) -> SHATCAR:
    model = SHATCAR(vocab_size=vocab_size)
    model.to(device)
    if os.path.exists(SHATCAR_MODEL_PATH):
        state = torch.load(SHATCAR_MODEL_PATH, map_location=device)
        model.load_state_dict(state)
    return model


def save_shatcar_model(model: SHATCAR):
    torch.save(model.state_dict(), SHATCAR_MODEL_PATH)


def train_shatcar(events: List[LogEvent], nlp, epochs: int, batch_size: int, device: torch.device):
    vocab = load_shatcar_vocab()
    if vocab is None:
        vocab = build_shatcar_vocab(events)
        save_shatcar_vocab(vocab)

    dataset = get_shatcar_dataset(events, vocab, nlp)
    if dataset is None:
        st.error("SHATCAR dataset is empty. Cannot train.")
        return None, vocab, 0.0

    X_ids = dataset["X_ids"]
    X_masks = dataset["X_masks"]
    sems = dataset["sems"]
    lvl_ids = dataset["lvl_ids"]
    svc_ids = dataset["svc_ids"]
    time_feats = dataset["time_feats"]
    y = dataset["y"]

    N = X_ids.size(0)
    if N == 0:
        st.error("No samples in SHATCAR dataset after preprocessing.")
        return None, vocab, 0.0

    model = load_shatcar_model(vocab_size=len(vocab), device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N)
        X_ids = X_ids[perm]
        X_masks = X_masks[perm]
        sems = sems[perm]
        lvl_ids = lvl_ids[perm]
        svc_ids = svc_ids[perm]
        time_feats = time_feats[perm]
        y = y[perm]

        total_loss = 0.0
        total_correct = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b_ids = X_ids[start:end].to(device)
            b_masks = X_masks[start:end].to(device)
            b_sems = sems[start:end].to(device)
            b_lvl = lvl_ids[start:end].to(device)
            b_svc = svc_ids[start:end].to(device)
            b_time = time_feats[start:end].to(device)
            b_y = y[start:end].to(device)

            logits = model(b_ids, b_masks, b_sems, b_lvl, b_svc, b_time)
            loss = F.cross_entropy(logits, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (end - start)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == b_y).sum().item()

        avg_loss = total_loss / N
        acc = total_correct / N
        st.write(f"Epoch {ep+1}/{epochs} • loss={avg_loss:.4f} • acc={acc:.3f}")

    try:
        save_shatcar_model(model)
        save_shatcar_vocab(vocab)
        st.success(f"SHATCAR model saved → {SHATCAR_MODEL_PATH}, vocab → {SHATCAR_VOCAB_PATH}")
    except Exception as e:
        st.error(f"Failed to save SHATCAR model or vocab: {e}")

    return model, vocab, acc


def run_shatcar_inference(events: List[LogEvent], nlp, device: torch.device):
    vocab = load_shatcar_vocab()
    if vocab is None or not os.path.exists(SHATCAR_MODEL_PATH):
        st.info("SHATCAR model/vocab not found. Train it in the last tab.")
        return

    model = load_shatcar_model(vocab_size=len(vocab), device=device)
    model.eval()

    subset = events[-50:] if len(events) > 50 else events
    if not subset:
        st.info("No events to score.")
        return

    svc_map = build_service_mapping(subset)

    X_ids, X_masks, sems, lvl_ids, svc_ids, time_feats = [], [], [], [], [], []
    for e in subset:
        tmpl = message_template(e.message)
        ids, mask = template_to_ids(tmpl, vocab)
        X_ids.append(ids)
        X_masks.append(mask)
        sems.append(semantic_vector(e.message, nlp))
        lvl_ids.append(level_to_id(e.level))
        svc_ids.append(service_to_id(e.service, svc_map))
        time_feats.append(time_features(e.timestamp))

    X_ids = torch.tensor(X_ids, dtype=torch.long).to(device)
    X_masks = torch.tensor(X_masks, dtype=torch.long).to(device)
    sems = torch.tensor(sems, dtype=torch.float32).to(device)
    lvl_ids = torch.tensor(lvl_ids, dtype=torch.long).to(device)
    svc_ids = torch.tensor(svc_ids, dtype=torch.long).to(device)
    time_feats = torch.tensor(time_feats, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_ids, X_masks, sems, lvl_ids, svc_ids, time_feats)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()

    rows = []
    for e, p in zip(subset, probs):
        rows.append(
            {
                "Line": e.line_no,
                "Level": e.level or "UNKNOWN",
                "Service": e.service or "UNK",
                "ErrorProb(SHATCAR)": round(p, 3),
                "Message": e.message[:120],
            }
        )
    st.table(rows)


# -----------------------------------------------------------------------------
# TRANSFORMER-DRIVEN TOPOLOGY
# -----------------------------------------------------------------------------
def build_transformer_topology(
    events: List[LogEvent],
    model: SHATCAR,
    vocab: Dict[str, int],
    nlp,
    device: torch.device,
) -> nx.DiGraph:
    G = nx.DiGraph()
    if not events:
        return G

    svc_map_global = build_service_mapping(events)
    svc_groups: Dict[str, List[LogEvent]] = defaultdict(list)
    for e in events:
        s = e.service or "UNK"
        svc_groups[s].append(e)

    model.eval()
    svc_emb = {}

    for svc, svc_events in svc_groups.items():
        subset = svc_events[-30:] if len(svc_events) > 30 else svc_events
        if not subset:
            continue

        X_ids, X_masks, sems, lvl_ids, svc_ids, time_feats = [], [], [], [], [], []
        for e in subset:
            tmpl = message_template(e.message)
            ids, mask = template_to_ids(tmpl, vocab)
            X_ids.append(ids)
            X_masks.append(mask)
            sems.append(semantic_vector(e.message, nlp))
            lvl_ids.append(level_to_id(e.level))
            svc_ids.append(service_to_id(e.service, svc_map_global))
            time_feats.append(time_features(e.timestamp))

        X_ids_t = torch.tensor(X_ids, dtype=torch.long).to(device)
        X_masks_t = torch.tensor(X_masks, dtype=torch.long).to(device)
        sems_t = torch.tensor(sems, dtype=torch.float32).to(device)
        lvl_ids_t = torch.tensor(lvl_ids, dtype=torch.long).to(device)
        svc_ids_t = torch.tensor(svc_ids, dtype=torch.long).to(device)
        time_feats_t = torch.tensor(time_feats, dtype=torch.float32).to(device)

        with torch.no_grad():
            pooled = model.pooled_embedding(X_ids_t, X_masks_t, sems_t, lvl_ids_t, svc_ids_t, time_feats_t)
            svc_emb[svc] = pooled.mean(dim=0).cpu()

    services = list(svc_emb.keys())
    for svc in services:
        G.add_node(svc)

    for i in range(len(services)):
        for j in range(len(services)):
            if i == j:
                continue
            s1, s2 = services[i], services[j]
            v1, v2 = svc_emb[s1], svc_emb[s2]
            sim = F.cosine_similarity(v1, v2, dim=0).item()
            sim = max(-1.0, min(1.0, sim))
            w = (sim + 1.0) / 2.0  # 0..1
            if w > 0.35:
                G.add_edge(s1, s2, weight=round(w, 3))

    return G


# -----------------------------------------------------------------------------
# SECURE HYBRID SUMMARY (spaCy + TRANSFORMER)
# -----------------------------------------------------------------------------
def secure_transformer_summary(
    events: List[LogEvent],
    model: SHATCAR,
    vocab: Dict[str, int],
    nlp,
    device: torch.device,
) -> str:
    if not events:
        return "No logs found."

    subset = events[-300:] if len(events) > 300 else events
    svc_map = build_service_mapping(subset)

    X_ids, X_masks, sems, lvl_ids, svc_ids, time_feats = [], [], [], [], [], []
    for e in subset:
        tmpl = message_template(e.message)
        ids, mask = template_to_ids(tmpl, vocab)
        X_ids.append(ids)
        X_masks.append(mask)
        sems.append(semantic_vector(e.message, nlp))
        lvl_ids.append(level_to_id(e.level))
        svc_ids.append(service_to_id(e.service, svc_map))
        time_feats.append(time_features(e.timestamp))

    X_ids_t = torch.tensor(X_ids, dtype=torch.long).to(device)
    X_masks_t = torch.tensor(X_masks, dtype=torch.long).to(device)
    sems_t = torch.tensor(sems, dtype=torch.float32).to(device)
    lvl_ids_t = torch.tensor(lvl_ids, dtype=torch.long).to(device)
    svc_ids_t = torch.tensor(svc_ids, dtype=torch.long).to(device)
    time_feats_t = torch.tensor(time_feats, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        embs = model.pooled_embedding(X_ids_t, X_masks_t, sems_t, lvl_ids_t, svc_ids_t, time_feats_t)

    B = embs.size(0)
    if B > 1:
        norm_embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-6)
        sim_mat = torch.matmul(norm_embs, norm_embs.T)
        avg_sim = (sim_mat.sum() - torch.diag(sim_mat).sum()) / (B * (B - 1))
        avg_sim = avg_sim.item()
    else:
        avg_sim = 1.0

    if avg_sim > 0.8:
        complexity_text = "Logs describe one unified behavior pattern."
    elif avg_sim > 0.5:
        complexity_text = "Logs show related flows with moderate variation."
    else:
        complexity_text = "Multiple distinct behaviors are present."

    full_text = "\n".join(e.message for e in subset)
    doc = nlp(full_text)

    nouns = Counter()
    verbs = Counter()
    svos = Counter()

    for sent in doc.sents:
        info = advanced_spacy_parse(sent.text, nlp)
        for n in info["nouns"]:
            nouns[n] += 1
        for v in info["verbs"]:
            verbs[v] += 1
        for triple in info["svos"]:
            svos[triple] += 1

    top_nouns = ", ".join(w for w, _ in nouns.most_common(5)) or "None"
    top_verbs = ", ".join(w for w, _ in verbs.most_common(5)) or "None"
    top_svos = "; ".join(f"{s}->{v}->{o}" for (s, v, o), _ in svos.most_common(3)) or "None"

    svc_counts = Counter(e.service or "UNK" for e in subset)
    svc_text = ", ".join(f"{s}({c})" for s, c in svc_counts.most_common(3)) or "None"

    errors = [e for e in subset if e.level in ("ERROR", "CRITICAL", "FATAL")]
    err_rate = len(errors) / len(subset)
    if err_rate == 0:
        err_text = "Errors are absent in this slice."
    elif err_rate < 0.1:
        err_text = "Issues exist but are minor."
    elif err_rate < 0.3:
        err_text = "System shows emerging instabilities."
    else:
        err_text = "System might be experiencing widespread failures."

    return f"""
Transformer + spaCy Summary
============================

Behavior Complexity:
{complexity_text}

Service Activity:
{svc_text}

Error Condition:
{err_text}

Dominant Topics (nouns):
{top_nouns}

Key Operations (verbs):
{top_verbs}

Role-Action-Object Patterns (SVO):
{top_svos}

This summary fuses:
- SHATCAR transformer latent behavior embeddings
- Deep spaCy linguistic parsing (nouns, verbs, SVO)
- No external LLMs, fully local and secure.
""".strip()


# -----------------------------------------------------------------------------
# HEURISTIC FALLBACK SUMMARY
# -----------------------------------------------------------------------------
def build_summary(events: List[LogEvent], nlp) -> str:
    if not events:
        return "No logs found."

    services = Counter(e.service or "UNK" for e in events)
    errors = [e for e in events if e.level in ("ERROR", "CRITICAL", "FATAL")]
    templates = Counter(message_template(e.message) for e in events)

    timestamps = [normalize_ts(e.timestamp) for e in events if e.timestamp]
    if timestamps:
        span = (timestamps[-1] - timestamps[0]).total_seconds()
        if span < 60:
            tl = "Logs occurred in a short burst."
        elif span < 300:
            tl = "Logs span a few minutes."
        else:
            tl = "Logs span a long-running window."
    else:
        tl = "Timestamps missing or inconsistent."

    if not errors:
        err_text = "No errors detected."
    else:
        err_services = Counter(e.service or "UNK" for e in errors)
        main_svc = err_services.most_common(1)[0][0]
        err_text = f"{len(errors)} errors detected, mainly in `{main_svc}`."

    top_services = services.most_common(3)
    top_templates = templates.most_common(5)

    template_lines = []
    for t, c in top_templates:
        friendly = (
            t.replace("<NUM>", "numbers")
            .replace("<UUID>", "UUIDs")
            .replace("<HEX>", "hex values")
            .replace("<IP>", "IP addresses")
        )
        template_lines.append(f"- {friendly} (~{c})")

    summary = f"""
Heuristic Summary (fallback)
----------------------------
Log span: {tl}

Top services:
{", ".join(f"{svc} ({cnt})" for svc, cnt in top_services)}

Errors:
{err_text}

Common log patterns:
{chr(10).join(template_lines) if template_lines else "None"}
""".strip()
    return summary


# -----------------------------------------------------------------------------
# SAFE EMBEDDING FOR CHAT (no tensor size mismatch)
# -----------------------------------------------------------------------------
def safe_embed_message(
    model: SHATCAR,
    vocab: Dict[str, int],
    nlp,
    device: torch.device,
    msg: str,
    level: Optional[str] = "INFO",
    service: Optional[str] = "CHAT",
    svc_map: Optional[Dict[str, int]] = None,
    ts: Optional[datetime] = None,
) -> torch.Tensor:
    """
    Produce a pooled embedding for any message (log or query) with safe shapes.
    Shapes:
      - X_ids: (1, STRUCT_MAX_LEN)
      - mask:  (1, STRUCT_MAX_LEN)
      - sems:  (1, SEM_DIM)
      - lvl:   (1,)
      - svc:   (1,)
      - time:  (1,4)
    """
    tmpl = message_template(msg)
    ids, mask = template_to_ids(tmpl, vocab)

    X_ids = torch.tensor([ids], dtype=torch.long).to(device)
    X_masks = torch.tensor([mask], dtype=torch.long).to(device)

    sem = semantic_vector(msg, nlp)
    sem = sem[:SEM_DIM] if len(sem) >= SEM_DIM else sem + [0.0] * (SEM_DIM - len(sem))
    sems = torch.tensor([sem], dtype=torch.float32).to(device)

    lvl_ids = torch.tensor([level_to_id(level)], dtype=torch.long).to(device)

    if svc_map is None:
        svc_ids = torch.tensor([0], dtype=torch.long).to(device)
    else:
        sid = service_to_id(service, svc_map)
        svc_ids = torch.tensor([sid], dtype=torch.long).to(device)

    tf = time_features(ts)
    time_feats = torch.tensor([tf], dtype=torch.float32).to(device)

    with torch.no_grad():
        emb = model.pooled_embedding(X_ids, X_masks, sems, lvl_ids, svc_ids, time_feats)  # (1,D)

    emb = emb[0]
    emb = emb / (emb.norm() + 1e-6)
    return emb


# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SHATCAR Log Analyzer", layout="wide")
st.title("🔍 SHATCAR Log Analyzer (spaCy + Transformer + Chat)")
st.caption(
    "Upload any log file (text or JSON lines) → adaptive parsing → temporal analytics → "
    "secure transformer+spaCy summarization → RCA → transformer-driven topology → structural patterns → SHATCAR training → chat with the model."
)

# Init DB
init_db()

# Load spaCy
@st.cache_resource
def load_spacy_model():
    return spacy.load(SPACY_MODEL, disable=["textcat"])


nlp = load_spacy_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sidebar input
st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader("Upload log file", type=["log", "txt"])

st.sidebar.caption(f"CWD: {os.getcwd()}")
st.sidebar.caption(f"Writable: {os.access(os.getcwd(), os.W_OK)}")

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = [ln.rstrip("\n") for ln in raw_text.splitlines() if ln.strip()]
    save_logs_to_db(lines)
    st.sidebar.success(f"Saved {len(lines)} lines into local log store.")

if st.sidebar.button("Clear stored logs"):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()
    st.sidebar.warning("Log store cleared. Upload again to continue.")
    st.stop()

raw_logs = load_logs_from_db()
if not raw_logs:
    st.info("No logs stored yet. Upload a .log or .txt file to begin.")
    st.stop()

st.sidebar.write(f"📦 Total stored log lines: **{len(raw_logs)}**")

events = parse_logs(raw_logs)

levels = Counter(e.level or "UNKNOWN" for e in events)
services = Counter(e.service or "UNK" for e in events)
errors = sum(1 for e in events if e.level in ("ERROR", "CRITICAL", "FATAL"))

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total log lines", len(events))
with c2:
    st.metric("Error / critical lines", errors)
with c3:
    st.metric("Distinct services", len(services))

st.write("### Level distribution")
st.bar_chart({k: v for k, v in levels.items()})

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "⏱ Temporal Analytics",
        "📝 Secure Transformer Summary",
        "🧠 RCA",
        "📡 Transformer Topology",
        "🧩 Structural Patterns",
        "📜 Raw Logs",
        "🧬 SHATCAR Transformer",
        "🗨️ Converse With Model",
    ]
)

with tab1:
    st.subheader("Temporal Analytics")
    plot_temporal_analytics(events)

with tab2:
    st.subheader("Secure Transformer-Based Summary (spaCy + SHATCAR)")
    vocab = load_shatcar_vocab()
    if vocab is None or not os.path.exists(SHATCAR_MODEL_PATH):
        st.warning("SHATCAR transformer is not trained yet. Showing heuristic layman summary instead.")
        summary_text = build_summary(events, nlp)
    else:
        model = load_shatcar_model(vocab_size=len(vocab), device=device)
        summary_text = secure_transformer_summary(events, model, vocab, nlp, device)
    st.text(summary_text)

with tab3:
    st.subheader("Heuristic Root Cause Analysis")
    rca = root_cause_analysis(events)
    if not rca["per_trace"]:
        st.info("No ERROR/CRITICAL/FATAL logs found for RCA.")
    else:
        st.write("**Top suspected root-cause services:**")
        st.table([{"Service": svc, "Count": cnt} for svc, cnt in rca["top_services"]])

        st.write("**Top recurring root-cause messages (truncated):**")
        st.table([{"Message": msg, "Count": cnt} for msg, cnt in rca["top_messages"]])

        with st.expander("Per-trace root cause details"):
            for item in rca["per_trace"]:
                st.markdown(
                    f"- Trace: `{item['trace_id']}` | Service: `{item['service']}` | Time: `{item['time']}`\n\n"
                    f"  → {item['message']}"
                )

with tab4:
    st.subheader("Transformer-Driven Service Topology (SHATCAR)")
    vocab = load_shatcar_vocab()
    if vocab is None or not os.path.exists(SHATCAR_MODEL_PATH):
        st.info(
            "SHATCAR model/vocab not found yet. Train the transformer in the last tab. "
            "Showing baseline trace-based topology instead."
        )
        baseline_graph = build_topology_baseline(events)
        draw_topology(baseline_graph)
    else:
        model = load_shatcar_model(vocab_size=len(vocab), device=device)
        topo_graph = build_transformer_topology(events, model, vocab, nlp, device)
        if len(topo_graph.nodes()) == 0:
            st.info(
                "SHATCAR topology could not infer strong relationships. "
                "Showing baseline trace-based topology instead."
            )
            baseline_graph = build_topology_baseline(events)
            draw_topology(baseline_graph)
        else:
            draw_topology(topo_graph)

with tab5:
    st.subheader("Structural Log Patterns (Templates)")
    patterns = extract_structural_patterns(events)
    if patterns:
        st.table(patterns)
    else:
        st.info("No structural patterns extracted.")

with tab6:
    st.subheader("Raw Logs")
    st.text("\n".join(raw_logs))

with tab7:
    st.subheader("Unified SHATCAR Transformer (Strong Attention)")
    st.write(f"Using device: `{device}`")

    epochs = st.slider("Training epochs", 1, 10, 3)
    batch_size = st.slider("Batch size", 8, 128, 32, step=8)

    if st.button("Train / Update SHATCAR model"):
        st.write(f"Training SHATCAR on {len(events)} parsed events...")
        model, vocab, acc = train_shatcar(events, nlp, epochs=epochs, batch_size=batch_size, device=device)
        if model is None:
            st.warning("Training did not complete (likely due to empty dataset).")
        else:
            st.success(f"SHATCAR trained. Final training accuracy: {acc:.3f}")

    st.markdown("---")
    st.write("**Score Recent Logs (SHATCAR structural error probability)**")
    if st.button("Run SHATCAR inference on latest logs"):
        run_shatcar_inference(events, nlp, device=device)

with tab8:
    st.subheader("🗨️ Converse With The Model (spaCy + SHATCAR)")
    st.caption("Ask questions about logs, anomalies, services, RCA, or summaries. Fully local, no LLM.")

    vocab = load_shatcar_vocab()
    model_exists = vocab is not None and os.path.exists(SHATCAR_MODEL_PATH)

    user_query = st.text_area(
        "Ask anything about the logs:",
        placeholder=(
            "examples:\n"
            "- What caused the errors?\n"
            "- Which service is unstable?\n"
            "- Summarize actions of auth service\n"
            "- Show anomalies between 10-11 pm\n"
            "- What patterns does the transformer detect?"
        ),
        height=140,
    )

    if st.button("Ask"):
        if not user_query.strip():
            st.warning("Please enter a query.")
            st.stop()

        qdoc = nlp(user_query)
        q_nouns = [t.lemma_.lower() for t in qdoc if t.pos_ == "NOUN" and not t.is_stop]
        q_verbs = [t.lemma_.lower() for t in qdoc if t.pos_ == "VERB" and not t.is_stop]

        st.markdown("#### 🔍 Query Linguistic Breakdown")
        st.write(f"**Nouns:** {q_nouns}")
        st.write(f"**Verbs:** {q_verbs}")
        st.write("---")

        svc_map = build_service_mapping(events)

        if model_exists:
            st.markdown("#### 🤖 Transformer Semantic Search")
            model = load_shatcar_model(vocab_size=len(vocab), device=device)
            model.eval()

            query_emb = safe_embed_message(
                model,
                vocab,
                nlp,
                device,
                user_query,
                level="INFO",
                service="CHAT_QUERY",
                svc_map=svc_map,
                ts=None,
            )

            match_rows = []
            with torch.no_grad():
                for e in events:
                    log_emb = safe_embed_message(
                        model,
                        vocab,
                        nlp,
                        device,
                        e.message,
                        level=e.level or "INFO",
                        service=e.service or "UNK",
                        svc_map=svc_map,
                        ts=e.timestamp,
                    )
                    sim = float(torch.dot(query_emb, log_emb))
                    match_rows.append((sim, e))

            match_rows.sort(key=lambda x: x[0], reverse=True)
            top_matches = match_rows[:5]

            table_data = [
                {
                    "Similarity": round(sim, 3),
                    "Service": e.service or "UNK",
                    "Level": e.level or "UNK",
                    "Message": e.message[:120],
                }
                for sim, e in top_matches
            ]
            st.table(table_data)
            st.markdown("---")
        else:
            st.info("Transformer not trained yet → using heuristic query handler only.")

        st.markdown("#### 🧠 Deterministic Reasoning (No LLM)")
        q = user_query.lower()
        if "cause" in q or "why" in q or "root" in q:
            intent = "root_cause"
        elif "summar" in q:
            intent = "summary"
        elif "error" in q or "fail" in q:
            intent = "error_analysis"
        elif "service" in q:
            intent = "service_info"
        elif "pattern" in q or "structure" in q:
            intent = "patterns"
        else:
            intent = "general"

        if intent == "root_cause":
            rca = root_cause_analysis(events)
            if not rca["top_messages"]:
                st.write("No errors detected. System appears healthy.")
            else:
                svc = rca["top_services"][0][0]
                msg = rca["top_messages"][0][0]
                st.write(f"**Likely cause:** Service `{svc}` with message: “{msg}”")

        elif intent == "summary":
            if model_exists:
                model = load_shatcar_model(len(vocab), device=device)
                out = secure_transformer_summary(events, model, vocab, nlp, device)
            else:
                out = build_summary(events, nlp)
            st.text(out)

        elif intent == "error_analysis":
            errs = [e for e in events if e.level in ("ERROR", "CRITICAL", "FATAL")]
            if not errs:
                st.write("No critical errors found.")
            else:
                svc_counts = Counter(e.service or "UNK" for e in errs)
                st.write("**Error-heavy services:**")
                for s, c in svc_counts.most_common():
                    st.write(f"- `{s}`: {c} errors")

        elif intent == "service_info":
            svc_counts = Counter(e.service or "UNK" for e in events)
            st.write("**Service activity:**")
            for s, c in svc_counts.most_common(5):
                st.write(f"- `{s}`: {c} logs")

        elif intent == "patterns":
            rows = extract_structural_patterns(events)
            st.table(rows)

        else:
            st.write("I analyzed your query using semantics and role patterns.")
            st.write("Try asking about *root cause*, *errors*, *patterns*, *summaries*, or *service activity*.")
