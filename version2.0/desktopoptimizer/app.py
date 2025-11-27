import os
import time
import platform
import subprocess
from datetime import datetime
from typing import TypedDict, Dict, Any, List, Optional

import streamlit as st
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from langgraph.graph import StateGraph, END


# ============================================================
# GLOBAL CONFIG
# ============================================================

DEVICE = torch.device("cpu")

CNN_MODEL_PATH = "multihead_cnn_fragility_model.pth"
WORD_LM_PATH = "tiny_word_lm.pth"

FEATURE_DIM = 6              # cpu, mem, disk, net_sent, net_recv, procs
NUM_OUTPUT_HEADS = 5         # cpu_pressure, mem_pressure, disk_pressure, proc_overload, fragility_score

CNN_EPOCHS = 12
CNN_LR = 1e-3
CNN_BATCH_SIZE = 16

WORD_LM_EPOCHS = 30
WORD_LM_LR = 1e-3

MAX_AGENT_ITERATIONS = 2  # planner loops


# ============================================================
# TINY LM CORPUS (PRETRAINED IN-APP)
# ============================================================

LM_CORPUS: List[str] = [
    "Close heavy applications to reduce CPU usage.",
    "Free memory by restarting long running processes.",
    "Too many background tasks reduce system stability.",
    "Limit parallel CPU intensive tasks to avoid thermal throttling.",
    "Terminate runaway processes that consume sustained high CPU.",
    "Avoid running multiple antivirus engines concurrently.",
    "Use power saving profiles on laptops to limit CPU spikes.",
    "Clear temporary files and caches to free memory.",
    "Restart memory hungry applications to reclaim RAM.",
    "Reducing open browser tabs lowers memory usage.",
    "Heavy IDEs and browsers combined can starve memory.",
    "Disable unused plugins and extensions to reduce memory load.",
    "Clear temporary and log files to lower disk pressure.",
    "Keep at least twenty percent free disk space.",
    "Storing large files externally reduces disk contention.",
    "Do not defrag SSDs.",
    "Limit automatic cloud syncs during heavy loads.",
    "Pause large downloads when low latency is required.",
    "Stop background updates on constrained networks.",
    "Disable unnecessary startup programs.",
    "Turn off unused background services.",
    "Use lightweight editors if running many local services.",
    "Close heavy UI effects on weak GPUs.",
    "Disable hardware acceleration if glitchy.",
    "Close recording overlays during games.",
    "Reboot regularly to clear memory leaks.",
]


# ============================================================
# WORD TOKENIZER + LSTM LM
# ============================================================

class WordTokenizer:
    def __init__(self, corpus: List[str]):
        words = set()
        for t in corpus:
            for w in t.lower().split():
                words.add(w)

        self.word2idx = {w: i + 1 for i, w in enumerate(sorted(words))}
        self.word2idx["<PAD>"] = 0
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(
            [self.word2idx.get(w.lower(), 0) for w in text.split()],
            dtype=torch.long,
        )

    def decode(self, idxs) -> str:
        return " ".join(self.idx2word.get(int(i), "<UNK>") for i in idxs)


class TinyWordLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def train_word_lm(model: TinyWordLSTM, tokenizer: WordTokenizer,
                  epochs: int = WORD_LM_EPOCHS, lr: float = WORD_LM_LR) -> TinyWordLSTM:
    dataset = []

    for line in LM_CORPUS:
        ids = tokenizer.encode(line)
        if len(ids) < 2:
            continue
        dataset.append((ids[:-1], ids[1:]))

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in dataset:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            opt.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            loss.backward()
            opt.step()

    return model


def lm_word_generate(model: TinyWordLSTM, tokenizer: WordTokenizer,
                     prompt: str, max_words: int = 40, temp: float = 0.7) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = torch.tensor([0])
    seq = ids.tolist()
    hidden = None

    for _ in range(max_words):
        x = torch.tensor([[seq[-1]]])
        logits, hidden = model(x, hidden)
        logits = logits[0, -1] / temp
        prob = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(prob, 1).item()
        seq.append(nxt)

    return tokenizer.decode(seq)


def load_or_train_word_lm() -> (TinyWordLSTM, WordTokenizer):
    tokenizer = WordTokenizer(LM_CORPUS)
    model = TinyWordLSTM(tokenizer.vocab_size).to(DEVICE)

    if os.path.exists(WORD_LM_PATH):
        model.load_state_dict(torch.load(WORD_LM_PATH, map_location=DEVICE))
    else:
        model = train_word_lm(model, tokenizer)
        torch.save(model.state_dict(), WORD_LM_PATH)

    return model, tokenizer


# ============================================================
# LM RECOMMENDATIONS & QUERY (CONDITIONED ON MULTI-HEAD SCORES)
# ============================================================

def generate_recommendations_from_lm(metrics: Dict[str, float],
                                     head_scores: Dict[str, float]) -> str:
    model, tok = load_or_train_word_lm()
    prompt = (
        f"CPU {metrics['cpu']:.1f}% (pressure {head_scores['cpu_pressure']:.2f}), "
        f"Memory {metrics['mem']:.1f}% (pressure {head_scores['mem_pressure']:.2f}), "
        f"Disk {metrics['disk']:.1f}% (pressure {head_scores['disk_pressure']:.2f}), "
        f"Processes {metrics['procs']} (pressure {head_scores['proc_overload']:.2f}), "
        f"overall fragility score {head_scores['fragility_score']:.2f}. "
        f"To improve system performance you should"
    )
    out = lm_word_generate(model, tok, prompt, 40)
    if "you should" in out:
        rec = out.split("you should", 1)[1].strip()
    else:
        rec = out.strip()
    rec = rec.replace("<PAD>", "").strip()
    if not rec.endswith("."):
        rec += "."
    return rec


def lm_query_with_context(query: str,
                          metrics: Dict[str, float],
                          head_scores: Dict[str, float]) -> str:
    model, tok = load_or_train_word_lm()
    prompt = (
        f"CPU pressure {head_scores.get('cpu_pressure',0):.2f}, "
        f"memory pressure {head_scores.get('mem_pressure',0):.2f}, "
        f"disk pressure {head_scores.get('disk_pressure',0):.2f}, "
        f"process overload {head_scores.get('proc_overload',0):.2f}, "
        f"fragility {head_scores.get('fragility_score',0):.2f}. "
        f"User asks: {query} You should"
    )
    out = lm_word_generate(model, tok, prompt, 40)
    if "you should" in out:
        resp = out.split("you should", 1)[1].strip()
    else:
        resp = out.strip()
    resp = resp.replace("<PAD>", "").strip()
    if not resp.endswith("."):
        resp += "."
    return resp


# ============================================================
# REMEDIATION COMMANDS (DISPLAY-ONLY, MAY INCLUDE DANGEROUS ONES)
# ============================================================

def generate_auto_remediation_commands(metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    cpu, mem, disk, procs = metrics["cpu"], metrics["mem"], metrics["disk"], metrics["procs"]
    cmds: List[Dict[str, Any]] = []

    if cpu > 80:
        cmds.append({
            "issue": "High CPU",
            "windows": [
                "tasklist",
                "wmic process where \"CPUUsage>20\" delete   (DANGEROUS - DO NOT RUN DIRECTLY)",
            ],
            "linux": [
                "ps -eo pid,cmd,%cpu --sort=-%cpu | head",
                "kill -9 <PID>   # DANGEROUS",
            ],
            "mac": [
                "ps aux | sort -nrk 3,3 | head",
                "kill -9 <PID>   # DANGEROUS",
            ],
        })

    if mem > 80:
        cmds.append({
            "issue": "High Memory",
            "windows": [
                "tasklist /FI \"MEMUSAGE gt 50000\"",
                "del /q/f/s %TEMP%\\*   (DANGEROUS - deletes files)",
            ],
            "linux": [
                "ps aux --sort=-%mem | head",
                "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches   # DANGEROUS",
            ],
            "mac": [
                "ps aux --sort -rss | head",
                "sudo purge   # DANGEROUS",
            ],
        })

    if disk > 85:
        cmds.append({
            "issue": "High Disk Usage",
            "windows": [
                "wmic logicaldisk get size,freespace,caption",
                "cleanmgr /sagerun:1   (DANGEROUS - cleans files)",
            ],
            "linux": [
                "df -h",
                "sudo apt autoremove -y   # DANGEROUS",
            ],
            "mac": [
                "df -h",
                "sudo rm -rf /Library/Caches/*   # VERY DANGEROUS",
            ],
        })

    if not cmds:
        return [{
            "issue": "Healthy",
            "windows": ["No manual remediation required."],
            "linux": ["No manual remediation required."],
            "mac": ["No manual remediation required."],
        }]
    return cmds


# ============================================================
# SAFE ACTION MAPPING DRIVEN BY MULTI-HEAD SCORES
# ============================================================

def map_scores_to_actions(head_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []

    cpu_p = head_scores.get("cpu_pressure", 0.0)
    mem_p = head_scores.get("mem_pressure", 0.0)
    disk_p = head_scores.get("disk_pressure", 0.0)
    proc_p = head_scores.get("proc_overload", 0.0)

    if cpu_p > 0.6:
        actions.append({
            "issue": "CPU diagnostics",
            "description": f"CPU pressure {cpu_p:.2f} – inspect CPU load.",
            "safe_windows": [
                "wmic cpu get loadpercentage",
                "tasklist /FO TABLE /NH",
            ],
            "safe_linux": [
                "ps -eo pid,cmd,%cpu --sort=-%cpu | head",
                "uptime",
            ],
            "safe_mac": [
                "ps aux | sort -nrk 3,3 | head",
                "uptime",
            ],
        })

    if mem_p > 0.6:
        actions.append({
            "issue": "Memory diagnostics",
            "description": f"Memory pressure {mem_p:.2f} – inspect RAM usage.",
            "safe_windows": [
                "wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value",
            ],
            "safe_linux": [
                "free -h",
            ],
            "safe_mac": [
                "vm_stat",
            ],
        })

    if disk_p > 0.6:
        actions.append({
            "issue": "Disk diagnostics",
            "description": f"Disk pressure {disk_p:.2f} – inspect disk usage.",
            "safe_windows": [
                "wmic logicaldisk get size,freespace,caption",
            ],
            "safe_linux": [
                "df -h",
            ],
            "safe_mac": [
                "df -h",
            ],
        })

    if proc_p > 0.6:
        actions.append({
            "issue": "Process diagnostics",
            "description": f"Process overload {proc_p:.2f} – inspect process list.",
            "safe_windows": [
                "tasklist /FO TABLE /NH",
            ],
            "safe_linux": [
                "ps aux --sort=-%cpu | head",
            ],
            "safe_mac": [
                "ps aux --sort -nrk 3,3 | head",
            ],
        })

    if not actions:
        actions.append({
            "issue": "General diagnostics",
            "description": "General system snapshot.",
            "safe_windows": ["wmic cpu get loadpercentage"],
            "safe_linux": ["uptime"],
            "safe_mac": ["uptime"],
        })

    return actions


def execute_safe_commands_for_current_os(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    os_name = platform.system().lower()
    os_key = (
        "safe_windows" if "windows" in os_name
        else "safe_linux" if "linux" in os_name
        else "safe_mac"
    )

    logs: List[Dict[str, Any]] = []

    for action in actions:
        safe_cmds = action.get(os_key, [])
        for cmd in safe_cmds:
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                logs.append({
                    "issue": action["issue"],
                    "command": cmd,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                })
            except Exception as e:
                logs.append({
                    "issue": action["issue"],
                    "command": cmd,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Execution error: {e}",
                })

    return logs


# ============================================================
# MULTI-HEAD CNN MODEL (REGRESSION 0–1)
# ============================================================

class MultiHeadFragilityCNN(nn.Module):
    def __init__(self, nf: int):
        super().__init__()
        self.c1 = nn.Conv1d(1, 16, 3, padding=1)
        self.b1 = nn.BatchNorm1d(16)
        self.c2 = nn.Conv1d(16, 32, 3, padding=1)
        self.b2 = nn.BatchNorm1d(32)
        self.c3 = nn.Conv1d(32, 32, 3, padding=1)
        self.b3 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, NUM_OUTPUT_HEADS)  # cpu, mem, disk, procs, fragility

    def forward(self, x):
        x = torch.relu(self.b1(self.c1(x)))
        x = torch.relu(self.b2(self.c2(x)))
        x = torch.relu(self.b3(self.c3(x)))
        x = self.pool(x).squeeze(-1)
        out = torch.sigmoid(self.fc(x))
        return out  # (batch, 5) in [0,1]


def load_cnn(nf: int) -> MultiHeadFragilityCNN:
    m = MultiHeadFragilityCNN(nf)
    if os.path.exists(CNN_MODEL_PATH):
        m.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
    return m


def save_cnn(m: MultiHeadFragilityCNN):
    torch.save(m.state_dict(), CNN_MODEL_PATH)


def train_cnn(model: MultiHeadFragilityCNN, X: np.ndarray, Y: np.ndarray) -> MultiHeadFragilityCNN:
    """
    Strongly validated training: ensures shapes are (N,FEATURE_DIM) and (N,NUM_OUTPUT_HEADS)
    to avoid tensor size mismatch.
    """
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,features), got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N,{NUM_OUTPUT_HEADS}), got shape {Y.shape}")

    if X.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"X must have feature dimension {FEATURE_DIM}, got {X.shape[1]}"
        )
    if Y.shape[1] != NUM_OUTPUT_HEADS:
        raise ValueError(
            f"Y must have {NUM_OUTPUT_HEADS} output heads, got {Y.shape[1]}"
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have same number of samples. "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}"
        )

    model.train()
    opt = optim.Adam(model.parameters(), lr=CNN_LR)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X).float().unsqueeze(1)  # (N,1,features)
    Y_t = torch.from_numpy(Y).float()               # (N,5)

    ds = TensorDataset(X_t, Y_t)
    dl = DataLoader(ds, batch_size=CNN_BATCH_SIZE, shuffle=True)

    for _ in range(CNN_EPOCHS):
        for bx, by in dl:
            opt.zero_grad()
            out = model(bx)  # (B,5)
            if out.shape != by.shape:
                raise RuntimeError(
                    f"Shape mismatch inside training: out={out.shape}, target={by.shape}"
                )
            loss = loss_fn(out, by)
            loss.backward()
            opt.step()

    return model


def predict_multihead(model: MultiHeadFragilityCNN, fv: np.ndarray) -> Dict[str, float]:
    fv = np.array(fv, dtype=np.float32).reshape(-1)
    if fv.shape[0] != FEATURE_DIM:
        # Fallback: flat zero scores if feature vector is wrong length
        return {
            "cpu_pressure": 0.0,
            "mem_pressure": 0.0,
            "disk_pressure": 0.0,
            "proc_overload": 0.0,
            "fragility_score": 0.0,
        }

    model.eval()
    x = torch.tensor(fv, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        out = model(x)[0].cpu().numpy()
    cpu_p, mem_p, disk_p, proc_p, frag = out.tolist()
    return {
        "cpu_pressure": float(cpu_p),
        "mem_pressure": float(mem_p),
        "disk_pressure": float(disk_p),
        "proc_overload": float(proc_p),
        "fragility_score": float(frag),
    }


def frag_label_from_score(score: float) -> str:
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Medium"
    else:
        return "High"


# ============================================================
# METRICS + PSEUDO LABELS FOR MULTI-HEAD TRAINING (FROM PSUTIL)
# ============================================================

def get_metrics() -> (Dict[str, float], np.ndarray):
    cpu = psutil.cpu_percent(0.3)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    net = psutil.net_io_counters()
    procs = len(psutil.pids())

    m = {
        "cpu": cpu,
        "mem": mem,
        "disk": disk,
        "net_sent": net.bytes_sent / 1e6,
        "net_recv": net.bytes_recv / 1e6,
        "procs": procs,
    }
    fv = np.array([
        cpu / 100,
        mem / 100,
        disk / 100,
        min(m["net_sent"] / 1000, 1),
        min(m["net_recv"] / 1000, 1),
        min(procs / 1000, 1),
    ], dtype=np.float32)
    return m, fv


def pseudo_multihead_targets(metrics: Dict[str, float]) -> np.ndarray:
    """
    Derive pseudo labels from psutil metrics.
    Each in [0,1]: cpu_pressure, mem_pressure, disk_pressure, proc_overload, fragility_score
    """
    cpu = metrics["cpu"]
    mem = metrics["mem"]
    disk = metrics["disk"]
    procs = metrics["procs"]

    cpu_p = np.clip((cpu - 40) / 60, 0, 1)
    mem_p = np.clip((mem - 40) / 60, 0, 1)
    disk_p = np.clip((disk - 50) / 50, 0, 1)
    proc_p = np.clip((procs - 200) / 800, 0, 1)

    frag = np.clip(0.35 * cpu_p + 0.35 * mem_p + 0.3 * disk_p, 0, 1)

    return np.array([cpu_p, mem_p, disk_p, proc_p, frag], dtype=np.float32)


# ============================================================
# LANGGRAPH AGENT STATE & NODES
# ============================================================

class AgentState(TypedDict, total=False):
    mode: str
    goal: str
    plan: List[str]
    iteration: int
    metrics: Dict[str, float]
    feature_vector: List[float]
    head_scores: Dict[str, float]
    fragility_label: str
    recommendations: str
    remediation_commands: List[Dict[str, Any]]
    action_plan: List[Dict[str, Any]]
    execution_logs: List[Dict[str, Any]]
    user_query: str
    lm_response: str


def node_planner(s: AgentState) -> AgentState:
    if s.get("mode") == "lm_query":
        s["goal"] = "answer_query"
        s["plan"] = ["read_metrics", "cnn_predict", "lm_query"]
    else:
        s["mode"] = "analysis"
        s["goal"] = "reduce_fragility"
        s["plan"] = [
            "read_metrics",
            "cnn_predict",
            "generate_recommendations",
            "generate_remediation",
            "map_actions",
            "safe_execute",
            "check_progress",
            "lm_query",
        ]
    s["iteration"] = s.get("iteration", 0)
    return s


def node_read_metrics(s: AgentState) -> AgentState:
    m, fv = get_metrics()
    s["metrics"] = m
    s["feature_vector"] = fv.tolist()
    return s


def node_cnn_predict(s: AgentState) -> AgentState:
    if not os.path.exists(CNN_MODEL_PATH):
        return s
    fv = np.array(s.get("feature_vector", []), dtype=np.float32)
    model = load_cnn(FEATURE_DIM)
    scores = predict_multihead(model, fv)
    s["head_scores"] = scores
    s["fragility_label"] = frag_label_from_score(scores["fragility_score"])
    return s


def node_generate_recommendations(s: AgentState) -> AgentState:
    if s.get("mode") != "analysis":
        return s
    metrics = s.get("metrics")
    head_scores = s.get("head_scores")
    if metrics is None or head_scores is None:
        return s
    s["recommendations"] = generate_recommendations_from_lm(metrics, head_scores)
    return s


def node_generate_remediation(s: AgentState) -> AgentState:
    if s.get("mode") != "analysis":
        return s
    metrics = s.get("metrics")
    if metrics is None:
        return s
    s["remediation_commands"] = generate_auto_remediation_commands(metrics)
    return s


def node_map_actions(s: AgentState) -> AgentState:
    if s.get("mode") != "analysis":
        return s
    head_scores = s.get("head_scores")
    if head_scores is None:
        return s
    s["action_plan"] = map_scores_to_actions(head_scores)
    return s


def node_safe_execute(s: AgentState) -> AgentState:
    if s.get("mode") != "analysis":
        return s
    actions = s.get("action_plan", [])
    if not actions:
        return s
    s["execution_logs"] = execute_safe_commands_for_current_os(actions)
    return s


def node_check_progress(s: AgentState) -> AgentState:
    s["iteration"] = s.get("iteration", 0) + 1
    return s


def next_after_check(s: AgentState) -> str:
    if s.get("mode") != "analysis":
        return "end"
    head_scores = s.get("head_scores", {})
    frag = head_scores.get("fragility_score", 0)
    iteration = s.get("iteration", 0)
    if frag > 0.7 and iteration < MAX_AGENT_ITERATIONS:
        return "loop"
    return "end"


def node_lm_query(s: AgentState) -> AgentState:
    q = s.get("user_query", "")
    if not q:
        return s
    metrics = s.get("metrics", {})
    head_scores = s.get("head_scores", {})
    s["lm_response"] = lm_query_with_context(q, metrics, head_scores)
    return s


# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("planner", node_planner)
graph.add_node("read_metrics", node_read_metrics)
graph.add_node("cnn_predict", node_cnn_predict)
graph.add_node("generate_recommendations", node_generate_recommendations)
graph.add_node("generate_remediation", node_generate_remediation)
graph.add_node("map_actions", node_map_actions)
graph.add_node("safe_execute", node_safe_execute)
graph.add_node("check_progress", node_check_progress)
graph.add_node("lm_query", node_lm_query)

graph.set_entry_point("planner")
graph.add_edge("planner", "read_metrics")
graph.add_edge("read_metrics", "cnn_predict")
graph.add_edge("cnn_predict", "generate_recommendations")
graph.add_edge("generate_recommendations", "generate_remediation")
graph.add_edge("generate_remediation", "map_actions")
graph.add_edge("map_actions", "safe_execute")
graph.add_edge("safe_execute", "check_progress")

graph.add_conditional_edges("check_progress", next_after_check, {
    "loop": "read_metrics",
    "end": "lm_query",
})

graph.add_edge("lm_query", END)

agent = graph.compile()


# ============================================================
# REAL-TIME SPINNING RUNNER
# ============================================================

def run_agent_with_progress(input_state: AgentState) -> AgentState:
    st.write("### 🚀 Agent Execution Trace")
    log_box = st.empty()
    collected: List[str] = []

    def log(msg: str):
        collected.append(msg)
        html = "<br>".join([f"🟦 {m}" for m in collected])
        log_box.markdown(html, unsafe_allow_html=True)

    with st.spinner("🤖 Agent running…"):
        for event in agent.stream(input_state):
            for node_name, _ in event.items():
                log(f"Running node: **{node_name}**")
                time.sleep(0.25)

        final = agent.invoke(input_state)
        log("🎉 **Execution complete.**")

    return final


# ============================================================
# STREAMLIT STATE INIT (WITH X/Y SYNCHRONISATION)
# ============================================================

def init_state():
    if "X" not in st.session_state:
        st.session_state.X = []
    if "Y" not in st.session_state:
        st.session_state.Y = []
    if "last_agent_state" not in st.session_state:
        st.session_state.last_agent_state = None
    if "lm_history" not in st.session_state:
        st.session_state.lm_history = []

    # Auto-fix any length mismatch from previous runs
    if len(st.session_state.X) != len(st.session_state.Y):
        n = min(len(st.session_state.X), len(st.session_state.Y))
        st.session_state.X = st.session_state.X[:n]
        st.session_state.Y = st.session_state.Y[:n]


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(page_title="Agentic Desktop Optimizer (Multi-Head CNN)", layout="wide")
    init_state()

    st.title("🧠 Desktop Optimizer — Multi-Head CNN + Tiny LM + LangGraph")
    live_metrics, live_fv = get_metrics()

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        if st.button("📸 Capture sample"):
            # Always capture metrics and targets together
            m, fv = get_metrics()
            target = pseudo_multihead_targets(m)
            st.session_state.X.append(fv)
            st.session_state.Y.append(target)

            # Hard sync X/Y lengths
            n = min(len(st.session_state.X), len(st.session_state.Y))
            st.session_state.X = st.session_state.X[:n]
            st.session_state.Y = st.session_state.Y[:n]

            st.success(f"Captured sample #{n}")

        if st.button("🧹 Clear samples"):
            st.session_state.X = []
            st.session_state.Y = []
            st.warning("Cleared all samples.")

        st.subheader("Train Multi-Head CNN")
        if st.button("🧬 Train Model"):
            if len(st.session_state.X) < 5:
                st.error("Need at least 5 samples.")
            else:
                try:
                    # Normalize X
                    X_clean = []
                    for i, fv in enumerate(st.session_state.X):
                        fv_arr = np.array(fv, dtype=np.float32).reshape(-1)
                        if fv_arr.shape[0] != FEATURE_DIM:
                            st.error(
                                f"Sample {i} has feature length {fv_arr.shape[0]} (expected {FEATURE_DIM}). "
                                "Try clearing samples and capturing again."
                            )
                            st.stop()
                        X_clean.append(fv_arr)
                    X = np.stack(X_clean)  # (N,6)

                    # Normalize Y
                    Y_clean = []
                    for i, t in enumerate(st.session_state.Y):
                        t_arr = np.array(t, dtype=np.float32).reshape(-1)
                        if t_arr.shape[0] != NUM_OUTPUT_HEADS:
                            st.error(
                                f"Target {i} has length {t_arr.shape[0]} (expected {NUM_OUTPUT_HEADS}). "
                                "Try clearing samples and capturing again."
                            )
                            st.stop()
                        Y_clean.append(t_arr)
                    Y = np.stack(Y_clean)  # (N,5)

                    st.write(f"Training CNN with X.shape={X.shape}, Y.shape={Y.shape}")
                    model = load_cnn(FEATURE_DIM)
                    with st.spinner("Training multi-head CNN..."):
                        model = train_cnn(model, X, Y)
                        save_cnn(model)
                    st.success("Multi-head CNN trained and saved.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        st.subheader("Run Agent")
        analyze = st.button("🔍 Run Multi-Step Agent")
        auto = st.checkbox("Auto-refresh metrics (5s)", False)

    if auto:
        st.experimental_set_query_params(t=time.time())

    c1, c2 = st.columns(2)

    # Live metrics
    with c1:
        st.subheader("📊 Live Metrics")
        st.metric("CPU %", f"{live_metrics['cpu']:.1f}")
        st.metric("Memory %", f"{live_metrics['mem']:.1f}")
        st.metric("Disk %", f"{live_metrics['disk']:.1f}")
        st.metric("Processes", f"{live_metrics['procs']}")
        st.caption(f"Updated at {datetime.now().strftime('%H:%M:%S')}")

    # Agent results
    with c2:
        st.subheader("🧠 Agent Results (Multi-Head Driven)")

        if analyze:
            init_agent_state: AgentState = {"mode": "analysis"}
            result = run_agent_with_progress(init_agent_state)
            st.session_state.last_agent_state = result

        a: Optional[AgentState] = st.session_state.last_agent_state
        if a and "metrics" in a:
            st.write(f"**Goal:** `{a.get('goal')}`")
            st.write(f"**Plan:** `{a.get('plan')}`")
            st.write(f"**Iterations:** `{a.get('iteration')}`")

            head_scores = a.get("head_scores", {})
            if head_scores:
                st.subheader("Multi-Head Scores")
                st.json(head_scores)
                st.write(
                    f"Fragility label: **{a.get('fragility_label','Unknown')}** "
                    f"(score {head_scores.get('fragility_score',0):.2f})"
                )

            with st.expander("Metrics used by agent"):
                st.json(a.get("metrics", {}))

            if "recommendations" in a:
                st.subheader("💡 LM Recommendations")
                st.write(a["recommendations"])

            if "remediation_commands" in a:
                st.subheader("🛠 Remediation Suggestions (NOT auto-run)")
                for block in a["remediation_commands"]:
                    st.write(f"#### Issue: {block['issue']}")
                    st.write("**Windows:**")
                    st.code("\n".join(block["windows"]), language="powershell")
                    st.write("**Linux:**")
                    st.code("\n".join(block["linux"]), language="bash")
                    st.write("**macOS:**")
                    st.code("\n".join(block["mac"]), language="bash")

            if "action_plan" in a:
                st.subheader("🔗 SAFE Actions from Multi-Head Scores")
                for act in a["action_plan"]:
                    st.write(f"**Issue:** {act['issue']}")
                    st.write(f"*{act['description']}*")
                    st.code(
                        f"Windows: {act['safe_windows']}\n"
                        f"Linux:   {act['safe_linux']}\n"
                        f"macOS:   {act['safe_mac']}",
                        language="text",
                    )

            if "execution_logs" in a:
                st.subheader("📟 Executed SAFE Commands Output")
                for log in a["execution_logs"]:
                    st.write(f"**[{log['issue']}]** `{log['command']}`")
                    st.text(f"Return code: {log['returncode']}")
                    if log["stdout"]:
                        st.text("STDOUT:")
                        st.code(log["stdout"])
                    if log["stderr"]:
                        st.text("STDERR:")
                        st.code(log["stderr"])
        else:
            st.info("Train the multi-head CNN (optional) and run the agent to see results.")

    # LM chat
    st.markdown("---")
    st.subheader("💬 Tiny LM Chat (via Agent, Multi-Head aware)")

    q = st.text_input("Ask about optimization / tuning:")
    if st.button("Send to Agentic LM"):
        if q.strip():
            qs: AgentState = {"mode": "lm_query", "user_query": q.strip()}
            res = run_agent_with_progress(qs)
            ans = res.get("lm_response", "")
            st.session_state.lm_history.append(("You", q))
            st.session_state.lm_history.append(("LM", ans))
        else:
            st.warning("Enter a question.")

    if st.session_state.lm_history:
        st.write("### Conversation")
        for who, msg in st.session_state.lm_history:
            st.markdown(f"**{who}:** {msg}")


if __name__ == "__main__":
    main()
