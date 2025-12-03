##############################################################
# Offline Semantic-Gated RAG v2.1 (TXT + PDF)
# - FAISS primary semantic engine
# - Sliding-window sentence chunks
# - Sentence quality scoring
# - Context-anchored scoring (nouns/verbs/entities/chunks)
# - Strict dependency alignment (subject/verb/object)
# - Semantic gating (with anchor requirement)
# - Tiny transformer fallback (restricted)
# - Anti-collapse memory
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
            page_title="Semantic-Gated RAG v2.1 (TXT + PDF)",
            layout="wide"
        )
    except Exception:
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
    except Exception:
        return None

nlp = load_spacy_model()

##############################################################
# FAISS
##############################################################
try:
    import faiss
except Exception:
    faiss = None
    st.warning("FAISS not installed — falling back to brute force similarity.")

##############################################################
# PDF PARSING
##############################################################
import pdfplumber

##############################################################
# RAG INDEX WITH QUALITY + SLIDING WINDOWS
##############################################################
class ExtractiveRAG:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.chunks: List[str] = []
        self.quality: List[float] = []
        self.index = None
        self.vecs = None
        self.dim = None

    def _embed(self, text: str) -> np.ndarray:
        doc = self.nlp(text)
        vec = doc.vector
        if vec is None or vec.shape[0] == 0:
            return np.zeros(self.dim if self.dim else 300, dtype="float32")
        return vec.astype("float32")

    def _sentence_quality(self, text: str) -> float:
        """
        Structural quality:
        - reward: normal-length, has subject+verb
        - penalize: very short/long, headers, boilerplate
        """
        doc = self.nlp(text)
        tokens = [t for t in doc if not t.is_space]
        words = [t for t in tokens if t.is_alpha]

        if not words:
            return -1.0

        n_words = len(words)
        q = 0.0

        # length
        if 8 <= n_words <= 40:
            q += 0.6
        elif n_words < 5:
            q -= 0.5
        elif n_words > 60:
            q -= 0.3

        # subject + verb
        has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc)
        has_verb = any(t.pos_ == "VERB" for t in doc)
        if has_subj and has_verb:
            q += 0.4
        else:
            q -= 0.2

        text_strip = text.strip()
        upper_ratio = sum(1 for c in text_strip if c.isupper()) / max(len(text_strip), 1)
        if upper_ratio > 0.7:
            q -= 0.5

        lower = text_strip.lower()
        if any(k in lower for k in ["copyright", "all rights reserved", "page "]):
            q -= 0.5

        if len(text_strip) < 60 and (
            text_strip.endswith(":")
            or any(k in lower for k in ["chapter", "section", "table "])
        ):
            q -= 0.3

        return max(-1.0, min(1.0, q))

    def build(self, corpus: str) -> int:
        """
        Build chunks:
        - single sentences
        - 2- and 3-sentence sliding windows
        Then index.
        """
        if self.nlp is None:
            raise RuntimeError("spaCy model not loaded.")

        chunk_set = set()

        for para in corpus.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            doc = self.nlp(para)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]

            # single sentences
            for s in sents:
                if len(s) > 15:
                    chunk_set.add(s)

            # 2-sentence windows
            for i in range(len(sents)):
                if i + 1 < len(sents):
                    w2 = sents[i] + " " + sents[i+1]
                    if len(w2) > 25:
                        chunk_set.add(w2)
                if i + 2 < len(sents):
                    w3 = sents[i] + " " + sents[i+1] + " " + sents[i+2]
                    if len(w3) > 35:
                        chunk_set.add(w3)

        if not chunk_set:
            chunk_set = {p.strip() for p in corpus.split("\n\n") if p.strip()}

        chunks = list(chunk_set)
        if not chunks:
            raise ValueError("No valid text found in documents.")

        qualities = []
        vecs = []
        for ch in chunks:
            q = self._sentence_quality(ch)
            qualities.append(q)
            vecs.append(self._embed(ch))
        vecs = np.vstack(vecs)
        self.dim = vecs.shape[1]

        self.quality = qualities
        self.chunks = chunks

        if faiss:
            faiss.normalize_L2(vecs)
            index = faiss.IndexFlatIP(self.dim)
            index.add(vecs)
            self.index = index
            self.vecs = None
        else:
            self.vecs = vecs
            self.index = None

        return len(self.chunks)

    def search(self, query: str, top_k: int = 12) -> List[Tuple[str, float, int]]:
        if not self.chunks:
            return []

        qv = self._embed(query).reshape(1, -1)

        if faiss and self.index is not None:
            faiss.normalize_L2(qv)
            k = min(top_k, len(self.chunks))
            D, I = self.index.search(qv, k)
            return [(self.chunks[i], float(s), int(i)) for s, i in zip(D[0], I[0])]

        # brute-force fallback
        vecs = self.vecs
        qn = qv / (np.linalg.norm(qv) + 1e-8)
        mn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        scores = (mn @ qn.T).ravel()
        idxs = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i]), int(i)) for i in idxs]

##############################################################
# DEPENDENCY SIGNATURE & STRICT MATCH
##############################################################
def dep_signature(doc):
    subs = set()
    verbs = set()
    objs = set()
    attrs = set()
    for t in doc:
        lemma = t.lemma_.lower()
        if t.dep_ in ("nsubj", "nsubjpass"):
            subs.add(lemma)
        if t.pos_ == "VERB":
            verbs.add(lemma)
        if t.dep_ in ("dobj", "obj", "pobj"):
            objs.add(lemma)
        if t.pos_ in ("ADJ", "ADV"):
            attrs.add(lemma)
    return {"subs": subs, "verbs": verbs, "objs": objs, "attrs": attrs}

def dep_match_score(sig_q, sig_s):
    """
    Stricter, more discriminative dependency alignment.
    Ensures question structure aligns with answer structure.
    """
    subj_overlap = len(sig_q["subs"] & sig_s["subs"])
    obj_overlap = len(sig_q["objs"] & sig_s["objs"])
    verb_overlap = len(sig_q["verbs"] & sig_s["verbs"])
    attr_overlap = len(sig_q["attrs"] & sig_s["attrs"])

    score = (
        0.45 * (subj_overlap > 0) +
        0.30 * (obj_overlap > 0) +
        0.20 * (verb_overlap > 0) +
        0.05 * (attr_overlap > 0)
    )
    return score

##############################################################
# CONTEXT ANCHOR SCORE (NARROW CONTEXT)
##############################################################
def context_anchor_score(query: str, sentence: str, nlp_model) -> float:
    """
    Strong context anchoring:
    - main nouns
    - verbs
    - multi-word noun chunks
    - named entities
    """
    qdoc = nlp_model(query)
    sdoc = nlp_model(sentence)

    q_nouns = {t.lemma_.lower() for t in qdoc if t.pos_ in ("NOUN", "PROPN")}
    q_verbs = {t.lemma_.lower() for t in qdoc if t.pos_ == "VERB"}
    q_chunks = {chunk.text.lower() for chunk in qdoc.noun_chunks}
    q_ents = {ent.text.lower() for ent in qdoc.ents}

    sent_lower = sentence.lower()
    s_lemmas = {t.lemma_.lower() for t in sdoc}

    score = 0.0

    # nouns
    for n in q_nouns:
        if n in s_lemmas or n in sent_lower:
            score += 1.2

    # verbs
    for v in q_verbs:
        if v in s_lemmas:
            score += 1.0

    # chunks
    for ch in q_chunks:
        if ch in sent_lower:
            score += 1.5

    # entities
    for ent in q_ents:
        if ent in sent_lower:
            score += 1.5

    denom = max(1, len(q_nouns) + len(q_verbs) + len(q_chunks) + len(q_ents))
    return score / denom

##############################################################
# PRIMARY SCORING + ANTI-COLLAPSE (NARROWED)
##############################################################
def choose_primary(
    query: str,
    candidates: List[Tuple[str, float, int]],
    rag: ExtractiveRAG,
    prev_query: str = None,
    prev_answer: str = None
):
    """
    Score each chunk using:
      - sem (FAISS)
      - context_anchor_score
      - dep_match_score
      - sentence_quality
    Also avoid repeating same wrong answer across unrelated queries.
    """
    if not candidates:
        return "", 0.0

    q_doc = nlp(query)
    q_sig = dep_signature(q_doc)

    scored = []
    for text, sem, idx in candidates:
        doc = nlp(text)
        s_sig = dep_signature(doc)
        dep_s = dep_match_score(q_sig, s_sig)
        anchor = context_anchor_score(query, text, nlp)
        qual = rag.quality[idx]

        combined = (
            0.55 * sem +
            0.25 * anchor +
            0.15 * dep_s +
            0.05 * qual
        )

        scored.append({
            "text": text,
            "sem": sem,
            "anchor": anchor,
            "dep": dep_s,
            "qual": qual,
            "combined": combined
        })

    scored.sort(key=lambda x: x["combined"], reverse=True)

    # semantic gradient check between top-2
    if len(scored) > 1:
        grad = scored[0]["sem"] - scored[1]["sem"]
        if grad < 0.03 and scored[1]["combined"] > scored[0]["combined"]:
            top_candidate = scored[1]
        else:
            top_candidate = scored[0]
    else:
        top_candidate = scored[0]

    # anti-collapse: avoid repeating same answer if queries differ
    if prev_answer and prev_query:
        prev_doc = nlp(prev_query)
        q_sim = q_doc.similarity(prev_doc)
        if top_candidate["text"] == prev_answer and q_sim < 0.8 and len(scored) > 1:
            # pick next if nearly as good
            second = scored[1]
            if second["combined"] >= top_candidate["combined"] - 0.15:
                top_candidate = second

    return top_candidate["text"], top_candidate["sem"]

##############################################################
# SEMANTIC GATING (WITH ANCHOR REQUIREMENT)
##############################################################
def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

SEM_THRESHOLD = 0.18

def passes_semantic_gate(query: str, sentence: str, rag: ExtractiveRAG):
    if not sentence:
        return False, 0.0
    qv = rag._embed(query)
    sv = rag._embed(sentence)
    sim = cosine_sim(qv, sv)

    # strict context anchor requirement
    anchor = context_anchor_score(query, sentence, nlp)
    if anchor < 0.05:
        return False, sim

    return sim >= SEM_THRESHOLD, sim

##############################################################
# TINY TRANSFORMER RERANKER (RESTRICTED)
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

    def encode(self, text: str):
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

def pos_sequence(doc):
    return " ".join(t.pos_ for t in doc)

def fallback_transformer(query: str, candidates: List[Tuple[str, float, int]], rag: ExtractiveRAG):
    """
    Rerank only semantically + anchor-valid candidates.
    Transformer effect is small and cannot override FAISS semantics.
    """
    q_doc = nlp(query)
    q_pos = pos_sequence(q_doc)

    valid = []
    for text, sem, idx in candidates:
        ok, _ = passes_semantic_gate(query, text, rag)
        if ok:
            valid.append((text, sem))

    if not valid:
        return None

    best_text = None
    best_score = -1e9

    for text, sem in valid:
        s_doc = nlp(text)
        s_pos = pos_sequence(s_doc)

        pair = f"[Q] {query} [POSQ] {q_pos} [S] {text} [POSS] {s_pos}"
        ids = tok_ce.encode(pair).to(device)
        with torch.no_grad():
            ce = ce_model(ids).item()
        ce_norm = float(torch.tanh(torch.tensor(ce)))

        combined = 0.8 * sem + 0.2 * ce_norm

        if combined > best_score:
            best_score = combined
            best_text = text

    return best_text

##############################################################
# STREAMLIT UI
##############################################################
st.title("🤖 Semantic-Gated Offline RAG v2.1 (TXT + PDF)")

st.markdown("""
- **FAISS** is the main semantic engine  
- **Context anchors** (nouns/verbs/entities/chunks) keep answers tightly on-topic  
- **Dependency matching** aligns subject–verb–object with the question  
- **Semantic gating** with anchor requirement prevents off-context answers  
- Tiny transformer only re-ranks *within* semantically valid candidates  
- Single-sentence, extractive, offline  
""")

if nlp is None:
    st.error("spaCy model `en_core_web_sm` not installed. Run: `python -m spacy download en_core_web_sm`")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "history" not in st.session_state:
    st.session_state.history = []
if "prev_query" not in st.session_state:
    st.session_state.prev_query = None
if "prev_answer" not in st.session_state:
    st.session_state.prev_answer = None

##############################################################
# UPLOAD TXT + PDF
##############################################################
st.subheader("📂 Upload .txt or .pdf files")

files = st.file_uploader(
    "Upload documents",
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
            except Exception:
                corpus += f.read().decode("latin-1", "ignore") + "\n"
        elif name.endswith(".pdf"):
            try:
                with pdfplumber.open(f) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        corpus += text + "\n"
            except Exception as e:
                st.error(f"Error reading PDF {f.name}: {e}")

    if corpus.strip():
        st.success("Documents loaded. Click **Build Index**.")
    else:
        st.error("No extractable text found in uploaded documents.")

##############################################################
# BUILD INDEX
##############################################################
if st.button("🔧 Build Index"):
    if not corpus.strip():
        st.error("Upload some documents first.")
    elif nlp is None:
        st.error("spaCy model not available.")
    else:
        rag = ExtractiveRAG(nlp)
        with st.spinner("Indexing sentences and sliding windows..."):
            try:
                n_chunks = rag.build(corpus)
                st.session_state.rag = rag
                st.success(f"Index built with {n_chunks} chunks.")
            except Exception as e:
                st.error(f"Indexing error: {e}")

##############################################################
# CHAT
##############################################################
st.subheader("💬 Ask a question")

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

query = st.chat_input("Ask something based on your documents...")

if query:
    st.session_state.history.append(("user", query))
    st.chat_message("user").write(query)

    if st.session_state.rag is None:
        answer = "Please upload documents and build the index first."
    else:
        rag = st.session_state.rag
        candidates = rag.search(query, top_k=12)

        primary_sent, primary_sem = choose_primary(
            query,
            candidates,
            rag,
            prev_query=st.session_state.prev_query,
            prev_answer=st.session_state.prev_answer
        )

        ok_primary, primary_sim = passes_semantic_gate(query, primary_sent, rag)

        if ok_primary:
            answer = primary_sent
        else:
            fb = fallback_transformer(query, candidates, rag)
            if fb:
                ok_fb, fb_sim = passes_semantic_gate(query, fb, rag)
                if ok_fb and fb_sim >= primary_sim:
                    answer = fb
                else:
                    answer = primary_sent
            else:
                answer = primary_sent

    st.chat_message("assistant").write(answer)
    st.session_state.history.append(("assistant", answer))

    st.session_state.prev_query = query
    st.session_state.prev_answer = answer
