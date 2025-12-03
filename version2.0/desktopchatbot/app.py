##############################################################
# OFFLINE SEMANTICALLY-GATED RAG (FAISS + Transformer Failover)
# Now with PDF + TXT ingestion
##############################################################

import os
from typing import List, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

##############################################################
# SAFE PAGE CONFIG
##############################################################
def safe_page_config():
    try:
        st.set_page_config(
            page_title="Semantic-Gated RAG (TXT + PDF)",
            layout="wide"
        )
    except:
        pass

safe_page_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################
# SPACY
##############################################################
import spacy

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        return None

nlp = load_spacy_model()

##############################################################
# FAISS
##############################################################
try:
    import faiss
except:
    faiss = None
    st.warning("FAISS not installed — fallback to brute force.")

##############################################################
# PRIMARY ENGINE: FAISS RAG
##############################################################
class ExtractiveRAG:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.sentences = []
        self.index = None
        self.vecs = None
        self.dim = None

    def _embed(self, text):
        doc = self.nlp(text)
        vec = doc.vector
        if vec is None or vec.shape[0] == 0:
            return np.zeros(self.dim if self.dim else 300, dtype="float32")
        return vec.astype("float32")

    def build(self, corpus):
        sents = []
        for para in corpus.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            doc = self.nlp(para)
            for s in doc.sents:
                t = s.text.strip()
                if len(t) > 25:
                    sents.append(t)

        if not sents:
            sents = [p.strip() for p in corpus.split("\n\n") if p.strip()]

        if not sents:
            raise ValueError("No valid text found in documents.")

        vecs = np.vstack([self._embed(s) for s in sents])
        self.dim = vecs.shape[1]

        if faiss:
            faiss.normalize_L2(vecs)
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(vecs)
            self.index = idx
        else:
            self.vecs = vecs

        self.sentences = sents
        return len(sents)

    def search(self, query, top_k=8):
        qv = self._embed(query).reshape(1, -1)

        if faiss and self.index:
            faiss.normalize_L2(qv)
            k = min(top_k, len(self.sentences))
            D, I = self.index.search(qv, k)
            return [(self.sentences[i], float(s)) for s, i in zip(D[0], I[0])]

        # brute force
        vecs = self.vecs
        qn = qv / (np.linalg.norm(qv) + 1e-8)
        mn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        scores = (mn @ qn.T).ravel()
        idx = np.argsort(-scores)[:8]
        return [(self.sentences[i], float(scores[i])) for i in idx]

##############################################################
# KEYWORD MATCHING
##############################################################
def extract_keywords(query, nlp):
    doc = nlp(query)
    keys = set()
    for t in doc:
        if t.pos_ in ("NOUN", "PROPN", "VERB") and not t.is_stop:
            keys.add(t.lemma_.lower())
    for ent in doc.ents:
        keys.add(ent.text.lower())
    return list(keys)

def keyword_score(keys, sent, nlp):
    if not keys:
        return 0.0
    doc = nlp(sent)
    lemmas = {t.lemma_.lower() for t in doc}
    s = sent.lower()
    score = 0.0
    for k in keys:
        if k in lemmas:
            score += 1.0
        elif k in s:
            score += 0.8
    return score / max(1, len(keys))

##############################################################
# PRIMARY BEST SENTENCE
##############################################################
def choose_primary(query, cands, nlp):
    keys = extract_keywords(query, nlp)
    best_sent = None
    best_score = -1e9
    best_sem = 0
    best_kw = 0
    for sent, sem in cands:
        kw = keyword_score(keys, sent, nlp)
        combined = 0.7 * sem + 0.3 * kw
        if combined > best_score:
            best_score = combined
            best_sent = sent
            best_sem = sem
            best_kw = kw
    return best_sent, best_sem, best_kw

##############################################################
# SEMANTIC GATING
##############################################################
def cos_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

SEM_THRESHOLD = 0.18

def sem_gate(query, sentence, rag):
    q = rag._embed(query)
    s = rag._embed(sentence)
    sim = cos_sim(q, s)
    return sim >= SEM_THRESHOLD, sim

##############################################################
# FALLBACK TRANSFORMER (restricted)
##############################################################
class TinyCrossEncoder(nn.Module):
    def __init__(self, vocab=30000, hidden=128, heads=4, layers=2, max_len=128):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab, hidden)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            activation="gelu",
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls = nn.Linear(hidden, 1)

    def forward(self, ids):
        x = self.emb(ids)
        x = self.enc(x)
        return self.cls(x[:, 0, :]).squeeze(-1)

class SimpleTokenizer:
    def __init__(self, max_len=128):
        self.max_len = max_len
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.next_id = 2

    def encode(self, text):
        toks = []
        for w in text.lower().split():
            if w not in self.vocab:
                self.vocab[w] = self.next_id
                self.next_id += 1
            toks.append(self.vocab[w])
        toks = toks[: self.max_len - 1]
        toks = [0] + toks
        toks += [0] * (self.max_len - len(toks))
        return torch.tensor(toks, dtype=torch.long).unsqueeze(0)

@st.cache_resource
def load_ce():
    tok = SimpleTokenizer()
    model = TinyCrossEncoder().to(device)
    model.eval()
    return tok, model

tok_ce, ce_model = load_ce()

def fallback_transformer(query, candidates, rag):
    valid = []
    for sent, sem in candidates:
        ok, _ = sem_gate(query, sent, rag)
        if ok:
            valid.append((sent, sem))
    if not valid:
        return None

    best = None
    best_score = -1e9

    for sent, sem in valid:
        ids = tok_ce.encode(f"[Q]{query}[S]{sent}").to(device)
        with torch.no_grad():
            ce = ce_model(ids).item()
        ce_norm = float(torch.tanh(torch.tensor(ce)))
        score = 0.6 * sem + 0.4 * ce_norm
        if score > best_score:
            best_score = score
            best = sent
    return best

##############################################################
# STREAMLIT UI
##############################################################
st.title("📘 Semantic-Gated RAG (TXT + PDF Offline Chatbot)")

if nlp is None:
    st.error("Install spaCy model: python -m spacy download en_core_web_sm")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "history" not in st.session_state:
    st.session_state.history = []

##############################################################
# UPLOAD TXT and PDF
##############################################################
import pdfplumber

st.subheader("📂 Upload Documents (.txt or .pdf)")

files = st.file_uploader(
    "Choose files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

corpus = ""

if files:
    for f in files:
        name = f.name.lower()
        if name.endswith(".txt"):
            try:
                corpus += f.read().decode("utf-8", "ignore") + "\n"
            except:
                corpus += f.read().decode("latin-1", "ignore") + "\n"

        elif name.endswith(".pdf"):
            try:
                with pdfplumber.open(f) as pdf:
                    for p in pdf.pages:
                        text = p.extract_text() or ""
                        corpus += text + "\n"
            except Exception as e:
                st.error(f"PDF error {f.name}: {e}")

    if corpus.strip():
        st.success("Documents loaded. Click Build Index.")
    else:
        st.error("No extractable text found.")

##############################################################
# BUILD INDEX
##############################################################
if st.button("🔧 Build Index"):
    if not corpus.strip():
        st.error("Upload documents first.")
    else:
        rag = ExtractiveRAG(nlp)
        with st.spinner("Indexing..."):
            n = rag.build(corpus)
        st.session_state.rag = rag
        st.success(f"Indexed {n} sentences!")

##############################################################
# CHAT
##############################################################
st.subheader("💬 Ask a question")

for r, t in st.session_state.history:
    st.chat_message(r).write(t)

q = st.chat_input("Ask something...")

if q:
    st.session_state.history.append(("user", q))
    st.chat_message("user").write(q)

    if st.session_state.rag is None:
        ans = "Please build the index first."
    else:
        rag = st.session_state.rag
        cands = rag.search(q, top_k=8)

        # PRIMARY
        ps, p_sem, p_kw = choose_primary(q, cands, nlp)
        ps_ok, ps_sim = sem_gate(q, ps, rag)

        if ps_ok:
            ans = ps
        else:
            fb = fallback_transformer(q, cands, rag)
            if fb:
                fb_ok, fb_sim = sem_gate(q, fb, rag)
                if fb_ok and fb_sim >= ps_sim:
                    ans = fb
                else:
                    ans = ps
            else:
                ans = ps

    st.chat_message("assistant").write(ans)
    st.session_state.history.append(("assistant", ans))
