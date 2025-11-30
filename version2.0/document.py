import os
import math
import re
import random
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# Optional PDF parsing
try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


# ==========================================================
# UI THEME (Dark)
# ==========================================================
st.set_page_config(page_title="Doc QA – Compressed + Fact-Locked Generator", layout="wide")
st.markdown("""
<style>
html, body, .stApp { background-color:#202123 !important; color:#ECECEC !important; }
.main-container { max-width: 1100px; margin:auto; padding:8px; }
section[data-testid="stSidebar"] { background-color:#202123 !important; border-right:1px solid #2f3136 !important; }
section[data-testid="stSidebar"] * { color:#ECECEC !important; }
textarea, input[type="text"] { background-color:#343541 !important; color:#ECECEC !important; border-radius:8px; border:1px solid #565869; }
textarea::placeholder, input[type="text"]::placeholder { color:#9FA0A5 !important; }
[data-testid="stFileUploader"] { background-color:#343541 !important; border-radius:8px; border:1px dashed #565869 !important; }
[data-testid="stFileUploader"] * { color:#ECECEC !important; }
.stButton > button { background-color:#10A37F !important; color:white !important; border-radius:6px; border:none; }
.debug-box { background:#1f1f1f;border-radius:8px;border:1px solid #333;padding:8px;font-size:13px;color:#ddd; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# TOKENIZATION / SENTENCE SPLIT
# ==========================================================
PAD, UNK, CLS, SEP = 0, 1, 2, 3

STOPWORDS = set("""
the a an and or of in on for with is are was were be being been to from at by as this that it its
they them he she we you i their our your his her which who what when where why how into over under
""".split())


def simple_tokenize(text: str) -> List[str]:
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return text.split()


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"[.!?]\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 4]


def build_vocab(sentences: List[str], max_vocab: int = 20000) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for s in sentences:
        for t in simple_tokenize(s):
            freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
    vocab: Dict[str, int] = {}
    idx = 4  # 0-3 reserved
    for tok, _ in sorted_tokens:
        if idx >= max_vocab:
            break
        vocab[tok] = idx
        idx += 1
    return vocab


# ==========================================================
# HEURISTIC SCORE FOR COMPRESSION
# ==========================================================
CAUSAL_CUES = {
    "because", "due", "therefore", "hence", "result", "resulted", "caused", "cause",
    "impact", "effect", "lead", "led", "consequence", "reason", "main",
    "issue", "problem", "solution", "risk", "bottleneck", "failure"
}


def heuristic_sentence_score(sentences: List[str]) -> List[float]:
    freq: Dict[str, int] = {}
    for s in sentences:
        for w in simple_tokenize(s):
            freq[w] = freq.get(w, 0) + 1

    max_freq = max(freq.values()) if freq else 1
    scores: List[float] = []

    for s in sentences:
        toks = simple_tokenize(s)
        if not toks:
            scores.append(0.0)
            continue

        length = len(toks)
        if length < 5 or length > 60:
            length_score = 0.2
        else:
            length_score = min(1.0, length / 30.0)

        rare_sum = 0.0
        rare_count = 0
        for w in toks:
            if w in STOPWORDS:
                continue
            f = freq.get(w, 1)
            rarity = 1.0 - (f / max_freq)
            rare_sum += rarity
            rare_count += 1
        rare_score = (rare_sum / rare_count) if rare_count > 0 else 0.0

        has_digit = any(any(c.isdigit() for c in w) for w in toks)
        numeric_score = 0.2 if has_digit else 0.0

        has_cue = any(w in CAUSAL_CUES for w in toks)
        cue_score = 0.25 if has_cue else 0.0

        base = 0.45 * length_score + 0.35 * rare_score + numeric_score + cue_score
        base = max(0.0, min(1.5, base)) / 1.5
        scores.append(base)

    return scores


# ==========================================================
# POSITIONAL ENCODING
# ==========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ==========================================================
# SENTENCE EMBEDDER (compression + clustering)
# ==========================================================
class SentenceEmbedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.emb(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        h = self.enc(h)
        pooled = h.mean(dim=1)
        score = self.head(pooled).squeeze(-1)
        return score, pooled


def encode_sentence_tokens(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    toks = simple_tokenize(text)
    ids = [vocab.get(t, UNK) for t in toks[:max_len]]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def train_sentence_embedder(
    model: SentenceEmbedder,
    sentences: List[str],
    heur_scores: List[float],
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device,
    epochs: int,
) -> List[float]:
    X = []
    y = []
    for s, h in zip(sentences, heur_scores):
        X.append(encode_sentence_tokens(s, vocab, max_len))
        y.append(h)
    if not X:
        return []
    X = torch.stack(X).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []
    model.train()
    for ep in range(epochs):
        idx = torch.randperm(X.size(0), device=device)
        xb = X[idx]
        yb = y[idx]
        opt.zero_grad()
        pred, _ = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        st.write(f"[Embedder] Epoch {ep+1}/{epochs} — MSE: {loss.item():.4f}")
    return losses


def get_model_scores(
    model: SentenceEmbedder,
    sentences: List[str],
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device,
) -> List[float]:
    model.eval()
    scores = []
    with torch.no_grad():
        for s in sentences:
            x = encode_sentence_tokens(s, vocab, max_len).unsqueeze(0).to(device)
            raw, _ = model(x)
            val = raw.item()
            val = 1.0 / (1.0 + math.exp(-val))
            scores.append(val)
    return scores


def get_sentence_embeddings(
    model: SentenceEmbedder,
    sentences: List[str],
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    embs = []
    with torch.no_grad():
        for s in sentences:
            x = encode_sentence_tokens(s, vocab, max_len).unsqueeze(0).to(device)
            _, pooled = model(x)
            v = pooled.squeeze(0)
            v = v / (v.norm() + 1e-8)
            embs.append(v.cpu())
    if not embs:
        return torch.empty(0, model.d_model)
    return torch.stack(embs, dim=0)


# ==========================================================
# STRICT CLUSTERING (DE-DUP)
# ==========================================================
def cluster_and_deduplicate(
    sentences: List[str],
    scores: List[float],
    embeddings: torch.Tensor,
    sim_thresh: float = 0.75
) -> List[int]:
    n = len(sentences)
    if n == 0:
        return []
    used = [False] * n
    selected = []
    for i in range(n):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, n):
            if used[j]:
                continue
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i], embeddings[j], dim=0
            ).item()
            if sim > sim_thresh:
                cluster.append(j)
                used[j] = True
        best_idx = max(cluster, key=lambda idx: scores[idx])
        selected.append(best_idx)
    return sorted(selected)


# ==========================================================
# CROSS-ENCODER B (QUERY vs SENTENCE)
# ==========================================================
class CrossEncoderB(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 96, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=384,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.cls_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        h = self.enc(h)
        cls = h[:, 0, :]
        logit = self.cls_head(cls).squeeze(-1)
        return logit


def encode_pair(q: str, s: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    q_toks = [vocab.get(t, UNK) for t in simple_tokenize(q)]
    s_toks = [vocab.get(t, UNK) for t in simple_tokenize(s)]
    ids = [CLS] + q_toks + [SEP] + s_toks
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def build_cross_pairs(sentences: List[str], vocab: Dict[str,int], max_len:int):
    if len(sentences) < 2:
        return None, None
    X = []
    y = []
    idxs = list(range(len(sentences)))
    for i, s in enumerate(sentences):
        X.append(encode_pair(s, s, vocab, max_len))
        y.append(1.0)
        others = [j for j in idxs if j != i]
        if others:
            j = random.choice(others)
            X.append(encode_pair(s, sentences[j], vocab, max_len))
            y.append(0.0)
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)


def train_cross_encoder(
    model: CrossEncoderB,
    sentences: List[str],
    vocab: Dict[str,int],
    max_len:int,
    device: torch.device,
    epochs:int,
) -> List[float]:
    X, y = build_cross_pairs(sentences, vocab, max_len)
    if X is None:
        return []
    X = X.to(device)
    y = y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    model.train()
    for ep in range(epochs):
        idx = torch.randperm(X.size(0), device=device)
        xb = X[idx]
        yb = y[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        st.write(f"[Cross-Encoder] Epoch {ep+1}/{epochs} — Loss: {loss.item():.4f}")
    return losses


def score_query_against_sentences(
    query: str,
    sentences: List[str],
    model: CrossEncoderB,
    vocab: Dict[str,int],
    max_len:int,
    device: torch.device,
) -> List[float]:
    model.eval()
    scores = []
    with torch.no_grad():
        for s in sentences:
            x = encode_pair(query, s, vocab, max_len).unsqueeze(0).to(device)
            logit = model(x).item()
            prob = 1.0 / (1.0 + math.exp(-logit))
            scores.append(prob)
    return scores


# ==========================================================
# FACT-LOCKED GENERATIVE LM
# ==========================================================
GLUE_WORDS = set("""
because since so therefore thus overall basically clearly simply actually mainly mostly
also however moreover additionally in summary in short in essence here this that it they we you
the a an of in on to for with from by is are was were be being been
""".split())


class TinyFactLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, max_len: int = 96, n_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=192,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.size()
        h = self.emb(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        h = self.enc(h, mask)
        logits = self.lm_head(h)
        return logits


def build_lm_vocab(sentences: List[str], extra_tokens: List[str], max_vocab: int = 15000) -> Dict[str,int]:
    freq: Dict[str, int] = {}
    for s in sentences:
        for t in simple_tokenize(s):
            freq[t] = freq.get(t, 0) + 1
    for t in extra_tokens:
        freq[t] = freq.get(t, 0) + 1

    sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
    vocab: Dict[str,int] = {"<bos>": 4, "<eos>": 5}  # 0-3 reserved
    idx = 6
    for tok, _ in sorted_tokens:
        if tok in vocab:
            continue
        if idx >= max_vocab:
            break
        vocab[tok] = idx
        idx += 1
    return vocab


def encode_lm(text: str, vocab: Dict[str,int], max_len: int) -> torch.Tensor:
    toks = ["<bos>"] + simple_tokenize(text) + ["<eos>"]
    ids = [vocab.get(t, UNK) for t in toks[:max_len]]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def train_tiny_fact_lm(
    model: TinyFactLM,
    sentences: List[str],
    vocab: Dict[str,int],
    max_len: int,
    device: torch.device,
    epochs: int = 2,
) -> List[float]:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for ep in range(epochs):
        total = 0.0
        count = 0
        for s in sentences:
            ids = encode_lm(s, vocab, max_len).to(device)
            x = ids[:-1].unsqueeze(0)
            y = ids[1:]
            if (y != PAD).sum().item() == 0:
                continue
            opt.zero_grad()
            logits = model(x)[0]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y,
                ignore_index=PAD,
            )
            loss.backward()
            opt.step()
            total += loss.item()
            count += 1
        if count > 0:
            avg = total / count
            losses.append(avg)
            st.write(f"[TinyFactLM] Epoch {ep+1}/{epochs} — Loss: {avg:.4f}")
    return losses


def build_allowed_ids_for_fact(
    fact_text: str,
    vocab: Dict[str,int],
) -> List[int]:
    fact_tokens = set(simple_tokenize(fact_text))
    allowed = set()
    for t in fact_tokens:
        if t in vocab:
            allowed.add(vocab[t])
    for t in GLUE_WORDS:
        if t in vocab:
            allowed.add(vocab[t])
    allowed.update([
        vocab.get("<bos>", -1),
        vocab.get("<eos>", -1),
        PAD,
    ])
    return [i for i in allowed if i is not None and i >= 0]


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def generate_fact_locked(
    model: TinyFactLM,
    prompt: str,
    fact_text: str,
    vocab: Dict[str,int],
    max_len: int,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.75,
    top_k: int = 12,
) -> str:
    model.eval()
    inv_vocab = {idx: tok for tok, idx in vocab.items()}

    x = encode_lm(prompt, vocab, max_len).to(device)
    seq = x.clone()

    allowed_ids = build_allowed_ids_for_fact(fact_text, vocab)
    max_vocab_id = max(inv_vocab.keys()) if inv_vocab else 0
    allowed_mask = torch.zeros(max_vocab_id + 1, dtype=torch.bool)
    for i in allowed_ids:
        if 0 <= i < allowed_mask.numel():
            allowed_mask[i] = True

    with torch.no_grad():
        for _ in range(max_new_tokens):
            length = (seq != PAD).sum().item()
            if length <= 0:
                length = 1
            inp = seq[:length].unsqueeze(0)
            logits = model(inp)[:, -1, :].squeeze(0)

            logits = logits.clone()
            if allowed_mask.numel() >= logits.size(0):
                logits[~allowed_mask[:logits.size(0)]] -= 10.0

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)

            logits = top_k_filter(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            if next_id == vocab.get("<eos>", -1):
                break
            if length < max_len:
                seq[length] = next_id
            else:
                break

    ids = seq.tolist()
    ids = [i for i in ids if i != PAD]
    tokens = []
    started = False
    for idx in ids:
        tok = inv_vocab.get(idx, "")
        if tok == "<bos>":
            started = True
            continue
        if tok == "<eos>":
            break
        if not started:
            continue
        tokens.append(tok)
    return " ".join(tokens)


# ==========================================================
# DYNAMIC TOP-K FACT FUSION + GENERATION
# ==========================================================
def choose_dynamic_top_k(
    scores: List[float],
    sentences: List[str],
    query: str,
    max_k: int = 3,
) -> List[Tuple[int, str]]:
    if not sentences:
        return []

    q = query.lower().strip()
    first_two = q.split()[:2]
    start = " ".join(first_two)

    # Default
    K = max_k

    # More precise mapping
    if start.startswith("what is") or start.startswith("who is") or "define" in q:
        K = 1
    elif "what happened" in q or "summarize" in q or "describe" in q:
        K = min(2, max_k)
    elif q.startswith("why") or q.startswith("how") or "explain" in q or "reason" in q:
        K = max_k
    else:
        K = min(2, max_k)

    K = max(1, min(K, len(sentences)))

    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:K]
    ranked = sorted(ranked)  # keep document order
    return [(i, sentences[i]) for i in ranked]


def humanize_and_generate(
    fused_fact_text: str,
    query: str,
    lm_model: Optional[TinyFactLM],
    lm_vocab: Optional[Dict[str,int]],
    lm_max_len: int,
    device: torch.device,
) -> str:
    base = fused_fact_text.strip()
    if not base:
        return "I couldn't find specific supporting sentences in the compressed summary."

    if base[-1] not in ".!?":
        base += "."

    q = query.lower().strip()

    # Conversational, paragraph-like prompts (Style 3)
    if q.startswith("why"):
        prompt = (
            f"From the document summary, the main reasons mentioned are as follows: {base} "
            f"Let me explain in a clear and connected way what led to this outcome and how the different points relate."
        )
    elif q.startswith("how"):
        prompt = (
            f"According to the summary, the process or outcome happens due to the following details: {base} "
            f"Let me walk through what happens step by step, tying the causes and effects together."
        )
    elif "what happened" in q or "summarize" in q or "describe" in q:
        prompt = (
            f"The summary describes the situation with these key points: {base} "
            f"I'll now describe the overall picture in a short narrative, keeping it faithful to these facts."
        )
    elif q.startswith("what"):
        prompt = (
            f"From the summary, the important details about this topic are: {base} "
            f"I will restate these points in a clear, conversational way without adding new information."
        )
    else:
        prompt = (
            f"Based on the compressed summary, the relevant details are: {base} "
            f"Let me explain the key idea in a concise, paragraph-style answer grounded only in these facts."
        )

    if lm_model is None or lm_vocab is None:
        return prompt

    gen = generate_fact_locked(
        lm_model,
        prompt,
        fused_fact_text,
        lm_vocab,
        lm_max_len,
        device,
        max_new_tokens=80,
        temperature=0.75,
        top_k=12,
    )

    if not gen or len(gen.split()) < 5:
        return prompt

    gen = gen.strip()
    if gen and not gen[0].isupper():
        gen = gen[0].upper() + gen[1:]
    if gen[-1] not in ".!?":
        gen += "."
    return gen


# ==========================================================
# FILE LOADER
# ==========================================================
def load_document(file) -> str:
    if not file:
        return ""
    name = file.name.lower()
    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf") and HAS_PYPDF:
        try:
            reader = PyPDF2.PdfReader(file)
            parts = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(parts)
        except Exception:
            return ""
    return ""


# ==========================================================
# SESSION STATE
# ==========================================================
if "S" not in st.session_state:
    st.session_state.S = {
        "raw_text": "",
        "all_sents": [],
        "heur_scores": [],
        "embed_vocab": None,
        "embed_model": None,
        "embed_max_len": 64,
        "compressed_summary": "",
        "compressed_sents": [],
        "cross_vocab": None,
        "cross_model": None,
        "cross_max_len": 96,
        "lm_vocab": None,
        "lm_model": None,
        "lm_max_len": 96,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "messages": [],
        "last_file": None,
        "last_epochs": None,
        "last_summary_len": None,
    }

S = st.session_state.S


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.header("Upload & Settings")
    up = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
    summary_len = st.slider("Compressed summary sentences", 5, 30, 12)
    epochs = st.slider("Training epochs (per doc)", 1, 5, 2)
    debug = st.checkbox("Show debug info", False)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("🧠 Doc QA – Multi-Sentence, Fact-Locked Conversational Answers (Offline)")
st.write(
    "- Hybrid compression (heuristic + learned)\n"
    "- Very strict clustering (cos_sim > 0.75) to remove redundancy\n"
    "- Cross-Encoder B for dynamic Top-K sentence retrieval\n"
    "- Tiny fact-locked LM for multi-sentence, logical, conversational answers."
)


# ==========================================================
# PIPELINE BUILD
# ==========================================================
def need_rebuild(name: Optional[str], ep: int, slen: int) -> bool:
    if name is None:
        return False
    return (
        S["last_file"] != name
        or S["last_epochs"] != ep
        or S["last_summary_len"] != slen
    )


if up and need_rebuild(up.name, epochs, summary_len):
    S["last_file"] = up.name
    S["last_epochs"] = epochs
    S["last_summary_len"] = summary_len
    S["messages"] = []

    raw = load_document(up).strip()
    S["raw_text"] = raw
    S["all_sents"] = []
    S["compressed_summary"] = ""
    S["compressed_sents"] = []
    S["embed_vocab"] = None
    S["embed_model"] = None
    S["cross_vocab"] = None
    S["cross_model"] = None
    S["lm_vocab"] = None
    S["lm_model"] = None

    if not raw:
        st.warning("Could not extract readable text from the document.")
    else:
        all_sents = sentence_split(raw)
        if len(all_sents) > 800:
            all_sents = all_sents[:800]
        S["all_sents"] = all_sents

        if not all_sents:
            st.warning("No usable sentences found in document.")
        else:
            device = S["device"]

            # Stage 1: heuristic
            with st.spinner("Scoring sentences (heuristic)..."):
                heur_scores = heuristic_sentence_score(all_sents)
            S["heur_scores"] = heur_scores

            # Stage 1b: sentence embedder
            embed_max_len = 40
            S["embed_max_len"] = embed_max_len
            embed_vocab = build_vocab(all_sents)
            S["embed_vocab"] = embed_vocab
            embed_vocab_size = len(embed_vocab) + 4

            embed_model = SentenceEmbedder(
                vocab_size=embed_vocab_size,
                d_model=64,
                max_len=embed_max_len,
            ).to(device)

            safe_name = re.sub(r"[^0-9a-zA-Z._-]", "_", up.name)
            embed_path = f"embed_{safe_name}.pt"

            if os.path.exists(embed_path):
                try:
                    embed_model.load_state_dict(torch.load(embed_path, map_location=device))
                    st.info(f"Loaded existing sentence embedder: {embed_path}")
                except Exception as e:
                    st.warning(f"Failed to load sentence embedder: {e}; retraining...")

            with st.spinner("Training sentence embedder on heuristic scores..."):
                emb_losses = train_sentence_embedder(
                    embed_model,
                    all_sents,
                    heur_scores,
                    embed_vocab,
                    embed_max_len,
                    device,
                    epochs,
                )
                try:
                    torch.save(embed_model.state_dict(), embed_path)
                    st.success(f"Sentence embedder saved as {embed_path}")
                except Exception as e:
                    st.warning(f"Could not save sentence embedder: {e}")
                if emb_losses:
                    st.line_chart(emb_losses)

            S["embed_model"] = embed_model

            # Stage 1c: hybrid scores + clustering
            with st.spinner("Computing hybrid compression scores..."):
                model_scores = get_model_scores(
                    embed_model,
                    all_sents,
                    embed_vocab,
                    embed_max_len,
                    device,
                )
                combined = [0.6 * h + 0.4 * m for h, m in zip(heur_scores, model_scores)]

            pre_k = min(summary_len * 3, len(all_sents))
            top_k = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:pre_k]
            candidate_sents = [all_sents[i] for i in top_k]

            with st.spinner("Computing embeddings for clustering..."):
                emb_candidates = get_sentence_embeddings(
                    embed_model,
                    candidate_sents,
                    embed_vocab,
                    embed_max_len,
                    device,
                )

            clustered_idx = cluster_and_deduplicate(
                candidate_sents,
                [combined[i] for i in top_k],
                emb_candidates,
                sim_thresh=0.75,
            )

            final_idxs = clustered_idx[:summary_len]
            compressed_sents = [candidate_sents[i] for i in final_idxs]

            S["compressed_sents"] = compressed_sents
            S["compressed_summary"] = " ".join(compressed_sents)

            # Stage 2: cross-encoder
            cross_train_sents = list(dict.fromkeys(compressed_sents + all_sents))
            avg_tokens = max(
                1,
                sum(len(simple_tokenize(s)) for s in cross_train_sents) // len(cross_train_sents),
            )
            cross_max_len = max(48, min(128, 2 * avg_tokens + 3))
            S["cross_max_len"] = cross_max_len

            cross_vocab = build_vocab(cross_train_sents)
            S["cross_vocab"] = cross_vocab
            cross_vocab_size = len(cross_vocab) + 4

            cross_model = CrossEncoderB(
                vocab_size=cross_vocab_size,
                d_model=96,
                max_len=cross_max_len,
            ).to(device)

            cross_path = f"crossB_{safe_name}.pt"
            if os.path.exists(cross_path):
                try:
                    cross_model.load_state_dict(
                        torch.load(cross_path, map_location=device)
                    )
                    st.info(f"Loaded existing cross-encoder: {cross_path}")
                except Exception as e:
                    st.warning(f"Failed to load cross-encoder: {e}; retraining...")

            with st.spinner("Training cross-encoder on compressed+doc sentences..."):
                cross_losses = train_cross_encoder(
                    cross_model,
                    cross_train_sents,
                    cross_vocab,
                    cross_max_len,
                    device,
                    epochs,
                )
                try:
                    torch.save(cross_model.state_dict(), cross_path)
                    st.success(f"Cross-encoder saved as {cross_path}")
                except Exception as e:
                    st.warning(f"Could not save cross-encoder: {e}")
                if cross_losses:
                    st.line_chart(cross_losses)

            S["cross_model"] = cross_model

            # Stage 3: tiny fact-locked LM
            if compressed_sents:
                lm_max_len = 96
                S["lm_max_len"] = lm_max_len
                extra_tokens = ["context", "summary", "reason", "cause", "process", "steps"]
                lm_vocab = build_lm_vocab(compressed_sents, extra_tokens)
                S["lm_vocab"] = lm_vocab
                lm_vocab_size = max(lm_vocab.values()) + 1

                lm_model = TinyFactLM(
                    vocab_size=lm_vocab_size,
                    d_model=64,
                    max_len=lm_max_len,
                    n_layers=3,
                ).to(device)

                lm_path = f"lm_fact_{safe_name}.pt"
                if os.path.exists(lm_path):
                    try:
                        lm_model.load_state_dict(torch.load(lm_path, map_location=device))
                        st.info(f"Loaded existing fact-locked LM: {lm_path}")
                    except Exception as e:
                        st.warning(f"Failed to load fact-locked LM: {e}; retraining...")

                with st.spinner("Training fact-locked LM on compressed sentences..."):
                    lm_losses = train_tiny_fact_lm(
                        lm_model,
                        compressed_sents,
                        lm_vocab,
                        lm_max_len,
                        device,
                        epochs=max(1, epochs - 1),
                    )
                    try:
                        torch.save(lm_model.state_dict(), lm_path)
                        st.success(f"Fact-locked LM saved as {lm_path}")
                    except Exception as e:
                        st.warning(f"Could not save fact-locked LM: {e}")
                    if lm_losses:
                        st.line_chart(lm_losses)

                S["lm_model"] = lm_model

            st.success("Pipeline ready: compression + clustering + cross-encoder + fact-locked generator.")


# ==========================================================
# SHOW COMPRESSED SUMMARY
# ==========================================================
if S["compressed_summary"]:
    st.markdown("### 📄 Compressed Summary (De-duplicated High-Information Sentences)")
    st.text_area("Compressed Summary", S["compressed_summary"], height=220)


# ==========================================================
# CHAT
# ==========================================================
st.markdown("### 💬 Ask about the compressed summary")

for role, msg in S["messages"]:
    with st.chat_message(role):
        st.markdown(msg)

user_q = st.chat_input("Ask a question about the compressed summary...")

if user_q:
    S["messages"].append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    if not S["compressed_sents"]:
        answer = "I don't have a compressed summary yet. Please upload a document."
    elif S["cross_model"] is None or S["cross_vocab"] is None:
        answer = "The semantic scorer is not ready. Try re-uploading the document."
    else:
        try:
            scores = score_query_against_sentences(
                user_q,
                S["compressed_sents"],
                S["cross_model"],
                S["cross_vocab"],
                S["cross_max_len"],
                S["device"],
            )

            top_items = choose_dynamic_top_k(scores, S["compressed_sents"], user_q, max_k=3)
            if not top_items:
                answer = "I couldn't identify any specific sentences in the summary for that question."
            else:
                _, top_sents = zip(*top_items)
                fused_fact = ". ".join(s.strip().rstrip(".") for s in top_sents) + "."
                answer = humanize_and_generate(
                    fused_fact,
                    user_q,
                    S["lm_model"],
                    S["lm_vocab"],
                    S["lm_max_len"],
                    S["device"],
                )

            if debug:
                st.markdown("#### 🔍 Debug: Sentence Scores & Selected")
                lines = []
                selected_indices = {i for i, _ in top_items} if top_items else set()
                for i, s in enumerate(S["compressed_sents"]):
                    mark = "⭐" if i in selected_indices else "•"
                    lines.append(f"{mark} score={scores[i]:.4f}: \"{s}\"")
                st.markdown(
                    "<div class='debug-box'>" + "<br>".join(lines) + "</div>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            answer = f"Internal error while answering: `{e}`"

    S["messages"].append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)

st.markdown("</div>", unsafe_allow_html=True)
