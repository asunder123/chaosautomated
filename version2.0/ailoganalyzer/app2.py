##############################################
# app.py — PART 1/3
# Parsing, Storage, Analytics, Structure Engine
##############################################

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


# ==============================================================
#                 GLOBAL CONFIG
# ==============================================================
DB_PATH = "logs.db"
SPACY_MODEL = "en_core_web_sm"

SHATCAR_MODEL_PATH = "shatcar_model.pt"
SHATCAR_VOCAB_PATH = "shatcar_vocab.json"

SEM_DIM = 300
STRUCT_MAX_LEN = 32

# NEW stronger model dims
SHATCAR_D_MODEL = 192   # was 96
SHATCAR_NUM_CLASSES = 2

MAX_SERVICES = 200
MAX_LEVEL_EMBED = 10


# ==============================================================
#                 TIME PARSING UTILITIES
# ==============================================================
try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False


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
    """Safely converts timestamp to offset-naive."""
    if ts is None:
        return datetime.min.replace(tzinfo=None)
    if ts.tzinfo is not None:
        return ts.astimezone().replace(tzinfo=None)
    return ts


# ==============================================================
#                 SQLITE STORAGE
# ==============================================================
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

# ============================================================================
#                   ROOT CAUSE ANALYSIS (HEURISTIC)
# ============================================================================
def root_cause_analysis(events: List[LogEvent]) -> Dict[str, Any]:
    """
    Heuristic root cause detection:
    - Group events by trace_id
    - Identify the earliest ERROR/CRITICAL/FATAL within each trace
    - Aggregate most common error services and messages
    """
    traces: Dict[str, List[LogEvent]] = defaultdict(list)

    # Group events by trace_id (fallback "__NO_TRACE__")
    for e in events:
        key = e.trace_id or "__NO_TRACE__"
        traces[key].append(e)

    root_services = Counter()
    root_messages = Counter()
    per_trace_results = []

    for trace_id, evs in traces.items():
        # Sort inside trace chronologically
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

            per_trace_results.append({
                "trace_id": trace_id,
                "service": svc,
                "message": msg,
                "time": (
                    root_event.timestamp.isoformat()
                    if root_event.timestamp else None
                ),
            })

    # Return structured RCA data
    return {
        "per_trace": per_trace_results,
        "top_services": root_services.most_common(10),
        "top_messages": root_messages.most_common(10),
    }


def load_logs_from_db() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT line FROM logs")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]


# ==============================================================
#             LOG PARSING — TEXT + JSON
# ==============================================================
TIMESTAMP_REGEXES = [
    r"^\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})?)",
    r"^\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)(?:\s|$)",
    r"^\s*(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})",
]

LEVEL_REGEX = re.compile(r"\b(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)\b")

SERVICE_PATTERNS = [
    re.compile(r"\bsvc=([A-Za-z0-9_\-\.]+)\b"),
    re.compile(r"\bservice=([A-Za-z0-9_\-\.]+)\b"),
    re.compile(r"^\s*\[([A-Za-z0-9_\-\.]+)\]"),
]

TRACE_PATTERNS = [
    re.compile(r"\b(trace_id|trace|request_id|req_id|corr_id|correlation_id)=([A-Za-z0-9\-\_:]+)\b")
]


def parse_timestamp(line: str) -> Tuple[Optional[datetime], str]:
    """Extract timestamp if present, return (datetime, rest_of_line)."""
    for pat in TIMESTAMP_REGEXES:
        m = re.search(pat, line)
        if m:
            ts_str = m.group(1)
            rest = line[m.end():].lstrip()
            if HAS_DATEUTIL:
                try:
                    return date_parser.parse(ts_str), rest
                except:
                    return None, rest
            else:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    return dt, rest
                except:
                    return None, rest

    if HAS_DATEUTIL:
        try:
            dt = date_parser.parse(line, fuzzy=True)
            return dt, line
        except:
            return None, line

    return None, line


def parse_level(line: str) -> Optional[str]:
    """
    Robust log level detector:
    - Works even after timestamp removal
    - Supports: INFO, info, Info, [INFO], level=INFO, severity=ERROR, etc.
    - Never returns UNKNOWN if a real level exists.
    """

    # 1) direct simple patterns
    m = re.search(r"\b(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)\b", line, re.IGNORECASE)
    if m:
        lvl = m.group(1).upper()
        return "WARN" if lvl == "WARNING" else lvl

    # 2) bracketed: [INFO], {ERROR}, <DEBUG>
    m = re.search(r"[\[\{\(<](INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)[\]\}\)>]", line, re.IGNORECASE)
    if m:
        lvl = m.group(1).upper()
        return "WARN" if lvl == "WARNING" else lvl

    # 3) key=value: level=INFO severity=ERROR log_level=DEBUG
    m = re.search(r"(level|severity|log[_\-]?level)\s*[:=]\s*(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)", 
                  line, re.IGNORECASE)
    if m:
        lvl = m.group(2).upper()
        return "WARN" if lvl == "WARNING" else lvl

    return None   # allows UNKNOWN fallback



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
    """Remove level tags etc. at left side."""
    s = line
    s = re.sub(r"\[(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)\]", "", s)
    s = re.sub(r"\blevel=(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|CRITICAL|FATAL)\b", "", s)
    return s.strip() or line.strip()


# JSON keys
JSON_TIMESTAMP_KEYS = ["timestamp", "time", "ts"]
JSON_LEVEL_KEYS = ["level", "severity", "loglevel"]
JSON_SERVICE_KEYS = ["service", "svc", "module"]
JSON_TRACE_KEYS = ["trace_id", "trace", "req_id", "request_id"]
JSON_MESSAGE_KEYS = ["message", "msg", "event", "log"]


def try_parse_json_log(line: str) -> Optional[LogEvent]:
    """Try to parse a JSON log line. If valid dict, convert to LogEvent."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    ts = None
    for k in JSON_TIMESTAMP_KEYS:
        if k in obj:
            v = obj[k]
            if isinstance(v, (float, int)):
                try:
                    ts = datetime.fromtimestamp(v / 1000.0)
                except:
                    ts = None
            elif isinstance(v, str):
                try:
                    ts = date_parser.parse(v)
                except:
                    ts = None
            break

    lvl = None
    for k in JSON_LEVEL_KEYS:
        if k in obj:
            lvl = str(obj[k]).upper()
            if lvl == "WARNING":
                lvl = "WARN"
            break

    svc = None
    for k in JSON_SERVICE_KEYS:
        if k in obj:
            svc = str(obj[k])
            break

    trace = None
    for k in JSON_TRACE_KEYS:
        if k in obj:
            trace = str(obj[k])
            break

    msg = None
    for k in JSON_MESSAGE_KEYS:
        if k in obj:
            msg = str(obj[k])
            break
    if msg is None:
        msg = json.dumps(obj)

    return LogEvent(
        raw=line,
        timestamp=ts,
        level=lvl,
        service=svc,
        trace_id=trace,
        message=msg,
        line_no=-1
    )


def parse_logs(lines: List[str]) -> List[LogEvent]:
    events = []
    for i, line in enumerate(lines):
        s = line.strip()

        j = try_parse_json_log(s)
        if j:
            j.line_no = i
            events.append(j)
            continue

        ts, rem = parse_timestamp(s)
        lvl = parse_level(s)
        svc = parse_service(s)
        trace = parse_trace_id(s)
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


# ==============================================================
#              TEMPORAL ANALYTICS
# ==============================================================
def build_time_buckets(events: List[LogEvent], bucket="minute"):
    """
    Fully safe bucketing:
    - Never returns invalid datetimes (matplotlib ordinal-safe)
    - Missing timestamps get substituted with SAFE_MIN_TS
    - Ensures uniform key types
    """
    SAFE_MIN_TS = datetime(1970, 1, 1)

    if not events:
        return {}, True

    have_any_ts = any(e.timestamp is not None for e in events)

    buckets = defaultdict(lambda: {"total": 0, "errors": 0})

    if have_any_ts:
        # ALL keys will be datetime
        for e in events:
            if e.timestamp is None:
                key = SAFE_MIN_TS
            else:
                ts = normalize_ts(e.timestamp)
                # sanitize all timestamps
                if ts.year < 1970:
                    ts = SAFE_MIN_TS
                if ts.year > 2100:
                    ts = datetime(2100, 1, 1)
                key = ts.replace(second=0, microsecond=0)

            buckets[key]["total"] += 1
            if e.level in ("ERROR", "CRITICAL", "FATAL"):
                buckets[key]["errors"] += 1
    else:
        # use int buckets
        for e in events:
            key = e.line_no // 20
            buckets[key]["total"] += 1
            if e.level in ("ERROR", "CRITICAL", "FATAL"):
                buckets[key]["errors"] += 1

    return buckets, have_any_ts



def plot_temporal_analytics(events):
    buckets, is_time = build_time_buckets(events)
    if not buckets:
        st.info("No temporal data.")
        return

    keys = list(buckets.keys())
    totals = [buckets[k]["total"] for k in keys]
    errors = [buckets[k]["errors"] for k in keys]

    plt.figure(figsize=(10, 4))

    if is_time:
        # Convert datetimes to matplotlib ordinals safely
        safe_x = []
        for dt in keys:
            try:
                safe_x.append(mdates.date2num(dt))
            except:
                # If ANY error persists, sanitize to safe minimum
                safe_x.append(mdates.date2num(datetime(1970, 1, 1)))

        plt.plot_date(safe_x, totals, "-")
        plt.plot_date(safe_x, errors, "-")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gcf().autofmt_xdate()
    else:
        # Keys are integers
        plt.plot(keys, totals, "-")
        plt.plot(keys, errors, "-")

    plt.xlabel("Time" if is_time else "Line Buckets")
    plt.ylabel("Count")
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.close()



# ==============================================================
#         BASELINE TOPOLOGY (fallback)
# ==============================================================
def build_topology_baseline(events: List[LogEvent]) -> nx.DiGraph:
    G = nx.DiGraph()
    traces = defaultdict(list)

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

    # Stable layout: if small graph use shell_layout, else spring_layout
    if len(G.nodes()) <= 3:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.9, seed=42)

    # Node colors
    def color(s):
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

    # Edge weights
    weights = [G[u][v].get("weight", 0.1) for u, v in G.edges()]
    if weights:
        maxw = max(weights)
        widths = [2 + 6 * (w / maxw) for w in weights]
    else:
        widths = 1

    # Edge colors: highlight strong edges
    edge_colors = [
        "#d62728" if G[u][v].get("weight", 0) > 0.7 else "#555555"
        for u, v in G.edges()
    ]

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

    # Edge labels (weights)
    edge_labels = {(u, v): round(G[u][v]["weight"], 2) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)

    st.pyplot(plt.gcf())
    plt.close()

# ==============================================================
#             STRUCTURAL PATTERNS (TEMPLATES)
# ==============================================================
NUM_PATTERN = re.compile(r"\b\d+\b")
HEX_PATTERN = re.compile(r"\b0x[0-9A-Fa-f]+\b")
UUID_PATTERN = re.compile(r"\b[0-9a-fA-F\-]{32,36}\b")
IP_PATTERN = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")


def message_template(msg: str) -> str:
    s = NUM_PATTERN.sub("<NUM>", msg)
    s = HEX_PATTERN.sub("<HEX>", s)
    s = UUID_PATTERN.sub("<UUID>", s)
    s = IP_PATTERN.sub("<IP>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_structural_patterns(events: List[LogEvent]):
    counter = Counter()
    examples = {}

    for e in events:
        t = message_template(e.message)
        counter[t] += 1
        if t not in examples:
            examples[t] = e.message

    rows = []
    for tmpl, cnt in counter.most_common(15):
        rows.append({
            "Template": tmpl,
            "Count": cnt,
            "Example": examples[tmpl][:200],
        })
    return rows


# ==============================================================
#      HEURISTIC FALLBACK SUMMARY (used if SHATCAR missing)
# ==============================================================
def build_summary(events: List[LogEvent], nlp) -> str:
    if not events:
        return "No logs found."

    services = Counter(e.service or "UNK" for e in events)
    levels = Counter(e.level or "UNKNOWN" for e in events)
    errors = [e for e in events if e.level in ("ERROR", "CRITICAL", "FATAL")]

    templates = Counter(message_template(e.message) for e in events)

    top_services = services.most_common(3)
    top_templates = templates.most_common(5)

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

    template_lines = []
    for t, c in top_templates:
        friendly = t.replace("<NUM>", "numbers") \
                    .replace("<UUID>", "UUIDs") \
                    .replace("<HEX>", "hex values") \
                    .replace("<IP>", "IP addresses")
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
##############################################
# app.py — PART 2/3
# Upgraded SHATCAR + Training + Topology + Secure Summary
##############################################

# ============================================================================
#                      SHATCAR (UNIFIED TRANSFORMER)
# ============================================================================

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

    Stronger attention than base version:
      - d_model = 192, nhead = 6 (32-dim heads)
      - 2 encoder layers per level
      - dropout for robustness
      - semantic + level + service + time context fused into token space
      - CAR gating across levels
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

        # Structural token embedding + positional encoding
        self.struct_embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)

        # Context channels
        self.semantic_proj = nn.Linear(SEM_DIM, d_model, bias=False)
        self.level_embed = nn.Embedding(max_levels_embed, d_model)
        self.service_embed = nn.Embedding(max_services, d_model)
        self.time_proj = nn.Linear(4, d_model)

        # Stronger hierarchical transformer per level (2 layers)
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

        # Context-aware router (CAR)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_levels),
            nn.Softmax(dim=-1),
        )

        # Classification head (error vs non-error)
        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(
        self,
        struct_ids: torch.Tensor,   # (B, L)
        mask: torch.Tensor,         # (B, L)
        semantic_vecs: torch.Tensor,# (B, SEM_DIM)
        level_ids: torch.Tensor,    # (B,)
        service_ids: torch.Tensor,  # (B,)
        time_feats: torch.Tensor,   # (B, 4)
    ):
        x = self.struct_embed(struct_ids)  # (B, L, D)
        x = self.pos(x)

        # Context fusion
        sem = self.semantic_proj(semantic_vecs)   # (B, D)
        lvl = self.level_embed(level_ids)         # (B, D)
        svc = self.service_embed(service_ids)     # (B, D)
        tim = self.time_proj(time_feats)          # (B, D)

        context = sem + lvl + svc + tim          # (B, D)
        context_expanded = context.unsqueeze(1).repeat(1, x.size(1), 1)
        fused = x + context_expanded             # (B, L, D)

        # Context-aware routing weights over levels
        gating = self.router(context)            # (B, num_levels)

        # Multi-level encoding
        outputs = []
        src_key_padding_mask = (mask == 0)
        for level in self.levels:
            out = level(fused, src_key_padding_mask=src_key_padding_mask)
            outputs.append(out)

        stacked = torch.stack(outputs, dim=1)    # (B, num_levels, L, D)
        gating = gating.view(-1, self.num_levels, 1, 1)
        fused_final = (stacked * gating).sum(dim=1)  # (B, L, D)

        mask_f = mask.unsqueeze(-1).float()
        summed = (fused_final * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom                  # (B, D)

        pooled = self.dropout(pooled)
        logits = self.cls(pooled)                # (B, num_classes)
        return logits


# ============================================================================
#                 SHATCAR DATA PREP & HELPERS
# ============================================================================
def build_shatcar_vocab(events: List[LogEvent]) -> Dict[str, int]:
    freq = Counter()
    for e in events:
        tmpl = message_template(e.message)
        for t in tmpl.split():
            freq[t] += 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in freq.items():
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
    hour = ts.hour / 23.0 if ts.hour is not None else 0.0
    minute = ts.minute / 59.0 if ts.minute is not None else 0.0
    second = ts.second / 59.0 if ts.second is not None else 0.0
    is_weekend = 1.0 if ts.weekday() >= 5 else 0.0
    return [hour, minute, second, is_weekend]


def semantic_vector(msg: str, nlp) -> List[float]:
    """
    300-dim semantic vector:
     - prefer spaCy's dense vector if available
     - fallback: hashed bag-of-words (normalized)
    """
    doc = nlp(msg)
    if getattr(doc, "vector", None) is not None and doc.vector.size > 0:
        v = doc.vector.tolist()
        if len(v) >= SEM_DIM:
            return v[:SEM_DIM]
        else:
            return v + [0.0] * (SEM_DIM - len(v))
    # fallback
    v = [0.0] * SEM_DIM
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        h = hash(tok.lemma_.lower())
        idx = h % SEM_DIM
        v[idx] += 1.0
    norm = (sum(x * x for x in v) ** 0.5) or 1.0
    return [x / norm for x in v]


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


# ============================================================================
#                      TRAIN / UPDATE SHATCAR
# ============================================================================
def train_shatcar(events: List[LogEvent], nlp, epochs: int, batch_size: int, device: torch.device):
    vocab = load_shatcar_vocab()
    if vocab is None:
        vocab = build_shatcar_vocab(events)
        save_shatcar_vocab(vocab)

    dataset = get_shatcar_dataset(events, vocab, nlp)
    if dataset is None:
        st.error("SHATCAR dataset is empty (no tokenizable log messages). Cannot train.")
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
            b_sem = sems[start:end].to(device)
            b_lvl = lvl_ids[start:end].to(device)
            b_svc = svc_ids[start:end].to(device)
            b_time = time_feats[start:end].to(device)
            b_y = y[start:end].to(device)

            logits = model(b_ids, b_masks, b_sem, b_lvl, b_svc, b_time)
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

    # Save model + vocab robustly
    try:
        save_shatcar_model(model)
        save_shatcar_vocab(vocab)
        st.success(f"SHATCAR model saved to `{SHATCAR_MODEL_PATH}` and vocab to `{SHATCAR_VOCAB_PATH}`")
    except Exception as e:
        st.error(f"Failed to save SHATCAR model or vocab: {e}")

    return model, vocab, acc


# ============================================================================
#                      SHATCAR INFERENCE
# ============================================================================
def run_shatcar_inference(events: List[LogEvent], nlp, device: torch.device):
    vocab = load_shatcar_vocab()
    if vocab is None or not os.path.exists(SHATCAR_MODEL_PATH):
        st.info("Unified SHATCAR model or vocab not found. Train it first.")
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
        rows.append({
            "Line": e.line_no,
            "Level": e.level or "UNKNOWN",
            "Service": e.service or "UNK",
            "ErrorProb(SHATCAR)": round(p, 3),
            "Message": e.message[:120],
        })
    st.table(rows)


# ============================================================================
#            TRANSFORMER-DRIVEN SERVICE TOPOLOGY (SHATCAR-based)
# ============================================================================
def build_transformer_topology(
    events: List[LogEvent],
    model: SHATCAR,
    vocab: Dict[str, int],
    nlp,
    device: torch.device
) -> nx.DiGraph:
    """
    FIXED: Transformer-driven topology that ALWAYS renders cleanly.

    Improvements:
    - Ensures nodes always added first
    - Avoids collapsing graph when embeddings are uniform
    - Normalizes similarities into readable 0–1 range
    - Requires minimum activity for edges
    - Handles NaNs or infs gracefully
    - Guarantees directional edges but prevents over-connection
    """

    G = nx.DiGraph()
    if not events:
        return G

    # Group by service
    svc_map = build_service_mapping(events)
    svc_groups: Dict[str, List[LogEvent]] = defaultdict(list)
    for e in events:
        svc = e.service or "UNK"
        svc_groups[svc].append(e)

    # Add all service nodes FIRST to avoid collapsing
    for svc in svc_groups.keys():
        G.add_node(svc)

    model.eval()
    svc_emb = {}
    svc_car = {}

    # Compute embeddings per service
    for svc, svc_events in svc_groups.items():
        # Take last 30 logs to represent service
        subset = svc_events[-30:] if len(svc_events) > 30 else svc_events

        if not subset:
            continue

        # Prepare model inputs
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

        # Convert to tensors
        X_ids = torch.tensor(X_ids, dtype=torch.long).to(device)
        X_masks = torch.tensor(X_masks, dtype=torch.long).to(device)
        sems = torch.tensor(sems, dtype=torch.float32).to(device)
        lvl_ids = torch.tensor(lvl_ids, dtype=torch.long).to(device)
        svc_ids = torch.tensor(svc_ids, dtype=torch.long).to(device)
        time_feats = torch.tensor(time_feats, dtype=torch.float32).to(device)

        with torch.no_grad():
            # get pooled embeddings
            logits = model(X_ids, X_masks, sems, lvl_ids, svc_ids, time_feats)
            pooled = logits  # (B,2)

        # Mean embed per service
        svc_emb[svc] = pooled.mean(dim=0).float().cpu()

        # Get CAR weights too
        with torch.no_grad():
            context = (
                model.semantic_proj(sems)
                + model.level_embed(lvl_ids)
                + model.service_embed(svc_ids)
                + model.time_proj(time_feats)
            )
            car = model.router(context).mean(dim=0).float().cpu()
        svc_car[svc] = car

    services = list(svc_emb.keys())

    # Compute edges between services
    for i in range(len(services)):
        for j in range(len(services)):
            if i == j:
                continue

            s1 = services[i]
            s2 = services[j]

            e1, e2 = svc_emb[s1], svc_emb[s2]
            c1, c2 = svc_car[s1], svc_car[s2]

            # Cosine similarities
            sim_emb = F.cosine_similarity(e1, e2, dim=0).item()
            sim_car = F.cosine_similarity(c1, c2, dim=0).item()

            # Clamp to readable range
            sim_emb = max(-1, min(sim_emb, 1))
            sim_car = max(-1, min(sim_car, 1))

            # Normalize into 0–1 graph weights
            w = (sim_emb + 1) / 2 * 0.6 + (sim_car + 1) / 2 * 0.4

            # Avoid NaNs
            if w != w:
                continue

            # Threshold: only show meaningful edges
            if w > 0.35:  # tuned threshold for readability
                G.add_edge(s1, s2, weight=float(round(w, 3)))

    return G


# ============================================================================
#        SECURE TRANSFORMER-BASED SUMMARY (SHATCAR + spaCy)
# ============================================================================
def secure_transformer_summary(
    events: List[LogEvent],
    model: SHATCAR,
    vocab: Dict[str, int],
    nlp,
    device: torch.device
) -> str:
    """
    Secure, transformer-driven summarizer:
      - uses SHATCAR latent embeddings of last ~300 logs
      - analyzes similarity structure in embedding space
      - uses spaCy nouns/verbs to extract key terms
      - composes a layman, privacy-preserving explanation
    """
    if not events:
        return "No log activity detected."

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
        h = model.struct_embed(X_ids_t)
        h = model.pos(h)
        context = (
            model.semantic_proj(sems_t)
            + model.level_embed(lvl_ids_t)
            + model.service_embed(svc_ids_t)
            + model.time_proj(time_feats_t)
        )
        fused = h + context.unsqueeze(1)

        level_outputs = []
        src_key_padding_mask = (X_masks_t == 0)
        for layer in model.levels:
            lvl_h = layer(fused, src_key_padding_mask=src_key_padding_mask)
            level_outputs.append(lvl_h)

        mask_f = X_masks_t.unsqueeze(-1).float()
        pooled_per_level = []
        for lvl_h in level_outputs:
            summed = (lvl_h * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled_per_level.append(summed / denom)

        embs = sum(pooled_per_level) / len(pooled_per_level)  # (B, D)

    # Analyze variety in embedding space
    B = embs.size(0)
    if B > 1:
        norm_embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-6)
        sim_mat = torch.matmul(norm_embs, norm_embs.T)
        upper = sim_mat[torch.triu(torch.ones_like(sim_mat), diagonal=1) == 1]
        avg_sim = upper.mean().item() if upper.numel() > 0 else 1.0
    else:
        avg_sim = 1.0

    if avg_sim > 0.75:
        variety_text = "Logs are highly similar, suggesting one main scenario or flow."
    elif avg_sim > 0.45:
        variety_text = "Logs show related scenarios with some variation in behavior."
    else:
        variety_text = "Logs seem to cover several distinct behavior patterns."

    services = Counter(e.service or "UNK" for e in subset)
    errors = [e for e in subset if e.level in ("ERROR", "CRITICAL", "FATAL")]
    top_services = services.most_common(3)
    svc_summary = (
        ", ".join(f"`{s}` ({c} logs)" for s, c in top_services)
        if top_services else "No identifiable services"
    )

    error_rate = len(errors) / len(subset)
    if len(errors) == 0:
        error_story = "No significant errors were observed in this slice of logs."
    elif error_rate < 0.05:
        error_story = f"A few errors occurred ({len(errors)}), but they are relatively infrequent."
    elif error_rate < 0.2:
        error_story = f"Errors are noticeable ({len(errors)} events) and may indicate emerging issues."
    else:
        error_story = f"Errors are quite frequent ({len(errors)} events), suggesting instability or failures."

    def top_terms(logs: List[LogEvent], n: int = 6):
        if not logs:
            return []
        text = "\n".join(e.message for e in logs)
        doc = nlp(text)
        counts = Counter()
        for tok in doc:
            if tok.is_stop or tok.is_punct or tok.is_space:
                continue
            if tok.pos_ not in ("NOUN", "VERB"):
                continue
            lemma = tok.lemma_.lower()
            counts[lemma] += 1
        return [w for w, _ in counts.most_common(n)]

    error_terms = top_terms(errors, 6)
    normal_terms = top_terms([e for e in subset if e not in errors], 6)

    error_terms_text = ", ".join(error_terms) if error_terms else "no dominant error keywords"
    normal_terms_text = ", ".join(normal_terms) if normal_terms else "no dominant normal-operation keywords"

    timestamps = [normalize_ts(e.timestamp) for e in subset if e.timestamp]
    if timestamps:
        span = (timestamps[-1] - timestamps[0]).total_seconds()
        if span < 60:
            timeline_text = "Activity is concentrated within a very short time window."
        elif span < 300:
            timeline_text = "Activity progresses over a few minutes, showing a short-lived scenario."
        else:
            timeline_text = "Activity spans a longer period, indicating ongoing system behavior."
    else:
        timeline_text = "Timing information is limited or missing."

    summary = f"""
Transformer-Based Secure Summary
--------------------------------
The SHATCAR transformer encoded {len(subset)} recent log entries into a latent behavior space.

Overall Variety:
{variety_text}

Timeline:
{timeline_text}

Service Activity:
The most active services in this window are: {svc_summary}.

Error Behavior:
{error_story}
Error-related messages commonly involve: {error_terms_text}.
Normal-operation logs often mention: {normal_terms_text}.

Interpretation:
This summary is based on transformer embeddings plus spaCy's vocabulary signals.
It describes, in human terms, how focused or diverse recent activity is, which services
are most involved, and whether the system appears stable or trending towards failure.
    """.strip()

    return summary
##############################################
# app.py — PART 3/3
# Streamlit UI
##############################################

st.set_page_config(page_title="SHATCAR Log Analyzer", layout="wide")
st.title("🔍 SHATCAR Log Analyzer (Strong-Attention SHATCAR)")
st.caption(
    "Upload any log file (text or JSON lines) → adaptive parsing → temporal analytics → "
    "secure transformer-based summarization → RCA → transformer-driven topology → structural patterns → SHATCAR training & scoring."
)

# Init DB
init_db()

# Load spaCy (cached)
@st.cache_resource
def load_spacy_model():
    return spacy.load(SPACY_MODEL, disable=["parser", "textcat"])

nlp = load_spacy_model()

# Global device
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

# Load stored logs
raw_logs = load_logs_from_db()
if not raw_logs:
    st.info("No logs stored yet. Upload a .log or .txt file to begin.")
    st.stop()

st.sidebar.write(f"📦 Total stored log lines: **{len(raw_logs)}**")

# Parse logs
events = parse_logs(raw_logs)

# High-level stats
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

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "⏱ Temporal Analytics",
        "📝 Secure Transformer Summary",
        "🧠 RCA",
        "📡 Transformer Topology",
        "🧩 Structural Patterns",
        "📜 Raw Logs",
        "🧬 SHATCAR Transformer",
    ]
)

with tab1:
    st.subheader("Temporal Analytics")
    plot_temporal_analytics(events)

with tab2:
    st.subheader("Secure Transformer-Based Summary (SHATCAR-driven)")
    vocab = load_shatcar_vocab()
    if vocab is None or not os.path.exists(SHATCAR_MODEL_PATH):
        st.warning(
            "The SHATCAR transformer is not trained yet. "
            "Using heuristic layman summary instead."
        )
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
                    f"- Trace: `{item['trace_id']}` | Service: `{item['service']}` | "
                    f"Time: `{item['time']}`\n\n"
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
                "SHATCAR topology could not infer any strong relationships. "
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
    st.subheader("Unified SHATCAR Transformer (Strong-Attention)")
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
