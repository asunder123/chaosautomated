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

PAD, UNK, CLS, SEP = 0, 1, 2, 3

STOPWORDS = set("""
the a an and or of in on for with is are was were be being been to from at by as this that it its
they them he she we you i their our your his her which who what when where why how into over under
""".split())

CAUSAL_CUES = {
    "because", "due", "therefore", "hence", "result", "resulted", "caused", "cause",
    "impact", "effect", "lead", "led", "consequence", "reason", "main",
    "issue", "problem", "solution", "risk", "bottleneck", "failure"
}

GLUE_WORDS = set("""
because since so therefore thus overall basically clearly simply actually mainly mostly generally
also however moreover additionally furthermore similarly likewise in contrast on the other hand 
nonetheless nevertheless even so as a result consequently subsequently afterward beforehand meanwhile
in summary in short in essence in conclusion ultimately importantly notably specifically particularly
firstly secondly thirdly then next afterwards following eventually earlier later previously finally
that this these those it they we you one here there such as for example for instance to illustrate 
in other words in simpler terms put differently stated differently that is in fact indeed essentially
primarily fundamentally inherently overall collectively together combined jointly connected linked
thereafter henceforth thereby accordingly proportionally conversely 
""".split())

ROLE_WORDS = {
    "manager","director","engineer","analyst","author","lead","owner","architect",
    "team","department","committee","group","organization","stakeholder","supervisor"
}

# ---------------- basic text utils ----------------
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

# ---------------- question type & semantics ----------------
def detect_question_type(q: str) -> str:
    q = q.lower().strip()
    if q.startswith("who") or q.startswith("whom"):
        return "who"
    if q.startswith("why") or "reason" in q:
        return "why"
    if q.startswith("how"):
        return "how"
    if q.startswith("what happened") or "summarize" in q or "describe" in q:
        return "what_happened"
    if q.startswith("what"):
        return "what"
    return "general"

def tag_sentence_semantics(s: str) -> List[str]:
    """
    Tag a sentence as what/how/why/who (can be multiple).
    """
    tags = []
    lower = s.lower()

    # WHY: causal
    if any(cue in lower for cue in ["because", "due to", "so that", "therefore", "hence", "as a result", "resulted in", "led to"]):
        tags.append("why")

    # HOW: process
    if any(cue in lower for cue in [
        "first", "second", "third", "then", "next", "after that",
        "step", "process", "procedure", "by doing", "by using"
    ]):
        tags.append("how")

    # WHAT: definitional / descriptive
    if any(phrase in lower for phrase in [
        "is defined as", "refers to", "is described as",
        "this document describes", "this section covers", "the purpose of"
    ]) or " is " in lower[:80]:
        tags.append("what")

    # WHO: people / roles
    toks = s.split()
    caps = [t for t in toks[1:] if t and t[0].isupper() and t.lower() not in STOPWORDS]
    roles = [t for t in toks if t.lower() in ROLE_WORDS]
    if caps or roles:
        tags.append("who")

    if not tags:
        tags.append("what")
    return tags

def extract_people_or_roles(sentences: List[str]) -> List[str]:
    results = []
    for s in sentences:
        toks = s.split()
        if not toks:
            continue
        caps = [t for t in toks[1:] if t and t[0].isupper() and t.lower() not in STOPWORDS]
        roles = [t for t in toks if t.lower() in ROLE_WORDS]
        if caps or roles:
            results.append(s)
    return results

# ---------------- models ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class SentenceEmbedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
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
    scores: List[float],
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device,
    epochs: int = 2,
):
    if not sentences:
        return
    X = torch.stack([encode_sentence_tokens(s, vocab, max_len) for s in sentences]).to(device)
    y = torch.tensor(scores, dtype=torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(X.size(0), device=device)
        xb, yb = X[idx], y[idx]
        opt.zero_grad()
        pred, _ = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()

def get_sentence_embeddings(
    model: SentenceEmbedder,
    sentences: List[str],
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    if not sentences:
        return torch.empty(0, model.d_model)
    model.eval()
    embs = []
    with torch.no_grad():
        for s in sentences:
            x = encode_sentence_tokens(s, vocab, max_len).unsqueeze(0).to(device)
            _, pooled = model(x)
            v = pooled.squeeze(0)
            v = v / (v.norm() + 1e-8)
            embs.append(v.cpu())
    return torch.stack(embs, dim=0)

def get_model_scores(
    model: SentenceEmbedder,
    sentences: List[str],
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device,
) -> List[float]:
    model.eval()
    out = []
    with torch.no_grad():
        for s in sentences:
            x = encode_sentence_tokens(s, vocab, max_len).unsqueeze(0).to(device)
            raw, _ = model(x)
            val = raw.item()
            val = 1.0 / (1.0 + math.exp(-val))
            out.append(val)
    return out

def cluster_and_deduplicate(
    sentences: List[str],
    scores: List[float],
    embeddings: torch.Tensor,
    sim_thresh: float = 0.75,
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
            sim = torch.nn.functional.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
            if sim > sim_thresh:
                cluster.append(j)
                used[j] = True
        best_idx = max(cluster, key=lambda idx: scores[idx])
        selected.append(best_idx)
    return sorted(selected)

class CrossEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 96, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=384, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.cls_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        h = self.enc(h)
        cls = h[:, 0, :]
        return self.cls_head(cls).squeeze(-1)

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
    X, y = [], []
    idxs = list(range(len(sentences)))
    for i, s in enumerate(sentences):
        X.append(encode_pair(s, s, vocab, max_len)); y.append(1.0)
        others = [j for j in idxs if j != i]
        if others:
            j = random.choice(others)
            X.append(encode_pair(s, sentences[j], vocab, max_len)); y.append(0.0)
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)

def train_cross_encoder(
    model: CrossEncoder,
    sentences: List[str],
    vocab: Dict[str,int],
    max_len:int,
    device: torch.device,
    epochs:int = 2,
):
    X, y = build_cross_pairs(sentences, vocab, max_len)
    if X is None:
        return
    X, y = X.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(X.size(0), device=device)
        xb, yb = X[idx], y[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

def score_query_against_sentences(
    query: str,
    sentences: List[str],
    model: CrossEncoder,
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

class TinyFactLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, max_len: int = 96, n_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=192, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.size()
        h = self.emb(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        h = self.enc(h, mask)
        return self.lm_head(h)

def build_lm_vocab(sentences: List[str], extra_tokens: List[str], max_vocab: int = 15000) -> Dict[str,int]:
    freq: Dict[str, int] = {}
    for s in sentences:
        for t in simple_tokenize(s):
            freq[t] = freq.get(t, 0) + 1
    for t in extra_tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
    vocab: Dict[str,int] = {"<bos>": 4, "<eos>": 5}
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
):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
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

def weight_glue_words(vocab: Dict[str,int], weight: float = 1.8):
    weighted_ids = set()
    for w in GLUE_WORDS:
        if w in vocab:
            weighted_ids.add(vocab[w])
    return weighted_ids

def extract_contextual_synonyms(sentences: List[str]) -> set:
    freq = {}
    for s in sentences:
        for t in simple_tokenize(s):
            if len(t) > 3 and t not in STOPWORDS:
                freq[t] = freq.get(t, 0) + 1
    sorted_ctx = sorted(freq.items(), key=lambda x: -x[1])[:30]
    return set(t for t, _ in sorted_ctx)

def generate_topic_continuity_tokens(sentences: List[str]) -> set:
    continuity = set()
    for s in sentences:
        toks = simple_tokenize(s)
        if len(toks) > 1:
            continuity.add("the " + toks[0])
        if len(toks) > 2:
            continuity.add("the " + toks[0] + " " + toks[1])
    return continuity

def build_allowed_ids_for_fact(
    fact_text: str,
    vocab: Dict[str,int],
    compressed_sents: List[str],
):
    fact_tokens = set(simple_tokenize(fact_text))
    ctx_syn = extract_contextual_synonyms(compressed_sents)
    topic_tokens = generate_topic_continuity_tokens(compressed_sents)
    allowed_tokens = fact_tokens | GLUE_WORDS | ctx_syn | topic_tokens
    allowed = set()
    for tok in allowed_tokens:
        if tok in vocab:
            allowed.add(vocab[tok])
    weighted_glue_ids = weight_glue_words(vocab, weight=1.8)
    allowed.update([
        vocab.get("<bos>", -1),
        vocab.get("<eos>", -1),
        PAD,
    ])
    return {
        "allowed_ids": [i for i in allowed if i >= 0],
        "weighted_glue_ids": weighted_glue_ids,
    }

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
    compressed_sents: List[str],
    vocab: Dict[str,int],
    max_len: int,
    device: torch.device,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_k: int = 20,
) -> str:
    model.eval()
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    x = encode_lm(prompt, vocab, max_len).to(device)
    seq = x.clone()

    info = build_allowed_ids_for_fact(fact_text, vocab, compressed_sents)
    allowed_ids = info["allowed_ids"]
    weighted_glue = info["weighted_glue_ids"]

    max_vocab_id = max(inv_vocab.keys()) if inv_vocab else 0
    allowed_mask = torch.zeros(max_vocab_id + 1, dtype=torch.bool)
    for i in allowed_ids:
        if 0 <= i < allowed_mask.numel():
            allowed_mask[i] = True

    bigram_history = set()

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

            for idx in weighted_glue:
                if idx < logits.size(0):
                    logits[idx] += 0.45

            recent_window = seq[max(0, length - 14):length].tolist()
            recent_counts = {}
            for tid in recent_window:
                if tid != PAD and tid != vocab.get("<bos>", -1):
                    recent_counts[tid] = recent_counts.get(tid, 0) + 1
            for tid, c in recent_counts.items():
                if tid < logits.size(0) and c > 1:
                    logits[tid] -= 0.65 * (c - 1)

            if length >= 1:
                prev_tid = int(seq[length - 1].item())
                for cand in range(logits.size(0)):
                    if (prev_tid, cand) in bigram_history:
                        logits[cand] -= 1.2

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)
            logits = top_k_filter(logits, top_k)
            probs = F.softmax(logits, dim=-1)

            next_id = int(torch.multinomial(probs, num_samples=1).item())
            if length >= 1 and (int(seq[length - 1].item()), next_id) in bigram_history:
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_id == vocab.get("<eos>", -1):
                break
            if length < max_len:
                if length >= 1:
                    bigram_history.add((int(seq[length - 1].item()), next_id))
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

def choose_dynamic_top_k(
    query: str,
    sentences: List[str],
    scores: List[float],
    max_k: int,
    semantic_buckets: Dict[str, List[int]],
) -> List[Tuple[int, str]]:
    if not sentences:
        return []

    qtype = detect_question_type(query)

    # 1) choose candidate indices from buckets
    if qtype == "who":
        candidate_idxs = semantic_buckets.get("who", [])
        if not candidate_idxs:
            candidate_idxs = semantic_buckets.get("what", list(range(len(sentences))))
    elif qtype == "why":
        candidate_idxs = semantic_buckets.get("why", [])
        if not candidate_idxs:
            candidate_idxs = semantic_buckets.get("what", list(range(len(sentences))))
    elif qtype == "how":
        candidate_idxs = semantic_buckets.get("how", [])
        if not candidate_idxs:
            candidate_idxs = semantic_buckets.get("what", list(range(len(sentences))))
    elif qtype == "what_happened":
        candidate_idxs = list(set(
            semantic_buckets.get("how", []) +
            semantic_buckets.get("why", []) +
            semantic_buckets.get("what", [])
        ))
        if not candidate_idxs:
            candidate_idxs = list(range(len(sentences)))
    elif qtype == "what":
        candidate_idxs = semantic_buckets.get("what", [])
        if not candidate_idxs:
            candidate_idxs = list(range(len(sentences)))
    else:
        candidate_idxs = list(range(len(sentences)))

    if not candidate_idxs:
        candidate_idxs = list(range(len(sentences)))

    # 2) K per type
    if qtype == "who":
        K = 2
    elif qtype in ("why", "how"):
        K = max_k
    elif qtype == "what_happened":
        K = min(2, max_k)
    elif qtype == "what":
        K = 1
    else:
        K = min(2, max_k)

    K = max(1, min(K, len(candidate_idxs)))

    ranked = sorted(candidate_idxs, key=lambda i: scores[i], reverse=True)[:K]
    ranked = sorted(ranked)
    return [(i, sentences[i]) for i in ranked]

def humanize_and_generate(
    fused_fact_text: str,
    query: str,
    lm_model: Optional[TinyFactLM],
    lm_vocab: Optional[Dict[str,int]],
    lm_max_len: int,
    compressed_sents: List[str],
    device: torch.device,
) -> str:
    base = fused_fact_text.strip()
    if not base:
        return "I couldn't find specific supporting sentences in the compressed summary."
    if base[-1] not in ".!?":
        base += "."

    qtype = detect_question_type(query)

    if qtype == "who":
        prompt = (
            f"In the summary, the references to people or roles are: {base} "
            f"Explain who is involved, what their role or contribution is, and how they relate to "
            f"the main situation. Write this as an expressive, clear paragraph without adding "
            f"any information that isn't present in these facts."
        )
    elif qtype == "why":
        prompt = (
            f"From the summary, the reasons given are: {base} "
            f"Start by stating the main cause, then describe how each supporting detail leads "
            f"to that outcome, and end with a single sentence that ties them into one explanation."
        )
    elif qtype == "how":
        prompt = (
            f"The document describes the mechanism or process using these points: {base} "
            f"Explain step by step how this unfolds, showing how each part leads into the next, "
            f"and end with what this sequence means overall."
        )
    elif qtype == "what_happened":
        prompt = (
            f"The summary outlines what occurred using these facts: {base} "
            f"Give a short, narrative-style description of the situation, the key events or "
            f"conditions, and the final result, staying fully faithful to these facts."
        )
    elif qtype == "what":
        prompt = (
            f"The key information about this topic in the summary is: {base} "
            f"Rephrase this in a clean, definitional paragraph: open with what it is, then add "
            f"one or two supporting details, and close with why it matters in the document's context."
        )
    else:
        prompt = (
            f"According to the compressed summary, the relevant facts are: {base} "
            f"Turn these into a cohesive, conversational explanation: open with the main idea, "
            f"weave in the supporting points in a logical order, and end with a concise insight "
            f"that stays fully faithful to these facts."
        )

    if lm_model is None or lm_vocab is None:
        return prompt

    gen = generate_fact_locked(
        lm_model,
        prompt,
        fused_fact_text,
        compressed_sents,
        lm_vocab,
        lm_max_len,
        device,
        max_new_tokens=120,
        temperature=0.9,
        top_k=20,
    )

    if not gen or len(gen.split()) < 5:
        return prompt

    gen = gen.strip()
    if gen and not gen[0].isupper():
        gen = gen[0].upper() + gen[1:]
    if gen[-1] not in ".!?":
        gen += "."
    return gen

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

# ---------------- Streamlit UI & state ----------------
st.set_page_config(page_title="Hierarchical Doc QA", layout="wide")
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

if "S" not in st.session_state:
    st.session_state.S = {
        "raw_text": "",
        "all_sents": [],
        "compressed_sents": [],
        "compressed_summary": "",
        "semantic_tags": [],
        "semantic_buckets": {"what": [], "how": [], "why": [], "who": []},
        "embed_vocab": None,
        "embed_model": None,
        "embed_max_len": 48,
        "cross_vocab": None,
        "cross_model": None,
        "cross_max_len": 96,
        "lm_vocab": None,
        "lm_model": None,
        "lm_max_len": 96,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "messages": [],
        "last_file": None,
        "last_summary_len": None,
        "last_epochs": None,
    }

S = st.session_state.S

with st.sidebar:
    st.header("Upload & Settings")
    up = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
    summary_len = st.slider("Compressed summary sentences", 5, 30, 12)
    epochs = st.slider("Training epochs (per doc)", 1, 4, 2)
    debug = st.checkbox("Show debug info", False)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("🧠 Hierarchical WHAT / HOW / WHY / WHO Doc Chatbot (Offline PyTorch)")
st.write(
    "- Compresses document into de-duplicated key sentences\n"
    "- Tags each sentence with what/how/why/who semantics\n"
    "- Uses a small cross-encoder + hierarchical buckets for retrieval\n"
    "- Generates expressive, fact-locked answers guided by the question type"
)

def need_rebuild(name: Optional[str], ep: int, slen: int) -> bool:
    if name is None:
        return False
    return (
        S["last_file"] != name
        or S["last_summary_len"] != slen
        or S["last_epochs"] != ep
    )

# ---------------- pipeline build ----------------
if up and need_rebuild(up.name, epochs, summary_len):
    S["last_file"] = up.name
    S["last_summary_len"] = summary_len
    S["last_epochs"] = epochs
    S["messages"] = []
    S["compressed_sents"] = []
    S["compressed_summary"] = ""
    S["semantic_tags"] = []
    S["semantic_buckets"] = {"what": [], "how": [], "why": [], "who": []}
    S["embed_vocab"] = None
    S["embed_model"] = None
    S["cross_vocab"] = None
    S["cross_model"] = None
    S["lm_vocab"] = None
    S["lm_model"] = None

    raw = load_document(up).strip()
    S["raw_text"] = raw
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
            with st.spinner("Scoring and compressing sentences..."):
                heur = heuristic_sentence_score(all_sents)
                embed_vocab = build_vocab(all_sents)
                S["embed_vocab"] = embed_vocab

                embed_model = SentenceEmbedder(
                    vocab_size=len(embed_vocab) + 6,
                    d_model=64,
                    max_len=S["embed_max_len"],
                ).to(device)
                train_sentence_embedder(
                    embed_model,
                    all_sents,
                    heur,
                    embed_vocab,
                    S["embed_max_len"],
                    device,
                    epochs=epochs,
                )
                S["embed_model"] = embed_model

                model_scores = get_model_scores(
                    embed_model,
                    all_sents,
                    embed_vocab,
                    S["embed_max_len"],
                    device,
                )
                combined = [0.6 * h + 0.4 * m for h, m in zip(heur, model_scores)]

                pre_k = min(summary_len * 3, len(all_sents))
                top_idx = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:pre_k]
                cand_sents = [all_sents[i] for i in top_idx]

                emb_cands = get_sentence_embeddings(
                    embed_model,
                    cand_sents,
                    embed_vocab,
                    S["embed_max_len"],
                    device,
                )
                clustered = cluster_and_deduplicate(
                    cand_sents,
                    [combined[i] for i in top_idx],
                    emb_cands,
                    sim_thresh=0.75,
                )
                final_idx = clustered[:summary_len]
                compressed_sents = [cand_sents[i] for i in final_idx]

                S["compressed_sents"] = compressed_sents
                S["compressed_summary"] = " ".join(compressed_sents)

                tags_per_sent = []
                buckets = {"what": [], "how": [], "why": [], "who": []}
                for idx, sent in enumerate(compressed_sents):
                    tags = tag_sentence_semantics(sent)
                    tags_per_sent.append(tags)
                    for t in tags:
                        if t in buckets:
                            buckets[t].append(idx)
                S["semantic_tags"] = tags_per_sent
                S["semantic_buckets"] = buckets

            cross_train_sents = list(dict.fromkeys(compressed_sents + all_sents))
            cross_vocab = build_vocab(cross_train_sents)
            S["cross_vocab"] = cross_vocab
            avg_tokens = max(
                1,
                sum(len(simple_tokenize(s)) for s in cross_train_sents) // len(cross_train_sents),
            )
            S["cross_max_len"] = max(48, min(128, 2 * avg_tokens + 3))

            cross_model = CrossEncoder(
                vocab_size=len(cross_vocab) + 6,
                d_model=96,
                max_len=S["cross_max_len"],
            ).to(device)
            with st.spinner("Training semantic scorer..."):
                train_cross_encoder(
                    cross_model,
                    cross_train_sents,
                    cross_vocab,
                    S["cross_max_len"],
                    device,
                    epochs=epochs,
                )
            S["cross_model"] = cross_model

            if compressed_sents:
                S["lm_max_len"] = 96
                lm_vocab = build_lm_vocab(
                    compressed_sents,
                    extra_tokens=["context", "summary", "reason", "cause", "process", "steps"],
                )
                S["lm_vocab"] = lm_vocab
                lm_model = TinyFactLM(
                    vocab_size=max(lm_vocab.values()) + 1,
                    d_model=64,
                    max_len=S["lm_max_len"],
                    n_layers=3,
                ).to(device)
                with st.spinner("Training expressive tiny LM on compressed summary..."):
                    train_tiny_fact_lm(
                        lm_model,
                        compressed_sents,
                        lm_vocab,
                        S["lm_max_len"],
                        device,
                        epochs=max(1, epochs),
                    )
                S["lm_model"] = lm_model

            st.success("Document processed. Summary and models are ready.")

# ---------------- summary display ----------------
if S.get("compressed_summary"):
    st.markdown("### 📄 Compressed Summary")
    st.text_area("Summary", S["compressed_summary"], height=220)

# ---------------- chat ----------------
st.markdown("### 💬 Ask about the compressed summary")
for role, msg in S["messages"]:
    with st.chat_message(role):
        st.markdown(msg)

user_q = st.chat_input("Ask a question about the document...")

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
            top_items = choose_dynamic_top_k(
                user_q,
                S["compressed_sents"],
                scores,
                max_k=3,
                semantic_buckets=S.get("semantic_buckets", {"what": [], "how": [], "why": [], "who": []}),
            )
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
                    S["compressed_sents"],
                    S["device"],
                )
            if debug:
                st.markdown("#### 🔍 Debug: Sentence Scores & Selected")
                lines = []
                selected_indices = {i for i, _ in top_items} if top_items else set()
                for i, s in enumerate(S["compressed_sents"]):
                    mark = "⭐" if i in selected_indices else "•"
                    lines.append(f"{mark} {scores[i]:.4f}: {s}")
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
