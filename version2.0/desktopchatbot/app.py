
import os
import math
import psutil
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import faiss
import spacy
from langgraph.graph import Graph
import string

# ================================
# DARK MODE CONFIG
# ================================
st.set_page_config(page_title="Enterprise Mobility RAG Assistant", layout="wide")

DARK_CSS = """
<style>
body, html, .stApp {
    background-color: #0D0D0D !important;
    color: white !important;
}
input, textarea, select {
    background-color: #1E1E1E !important;
    color: white !important;
    border-radius: 8px;
}
div.stButton > button {
    background-color: #333333 !important;
    color: white !important;
    border: 1px solid #555555 !important;
    border-radius: 6px;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ================================
# ENTERPRISE APP DETECTION
# ================================
ENTERPRISE_APPS = ["GTD", "GetTALENT", "Workday", "SuccessFactors"]

def get_desktop_context():
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        detected_apps = []
        for p in psutil.process_iter(["name"]):
            pname = p.info.get("name", "")
            for app in ENTERPRISE_APPS:
                if app.lower() in pname.lower():
                    detected_apps.append(app)
        txt = f"System Context:\nCPU {cpu:.1f}% | RAM {ram:.1f}% | Disk {disk:.1f}%\n"
        if detected_apps:
            txt += f"Detected Enterprise Apps: {', '.join(set(detected_apps))}\n"
        return txt
    except:
        return "System context unavailable."

# ================================
# FILE LOADING
# ================================
def load_text(upload):
    name = upload.name.lower()
    try:
        if name.endswith(".txt"):
            return upload.read().decode("utf-8", "ignore")
        if name.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(upload)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if name.endswith(".docx"):
            import docx
            d = docx.Document(upload)
            return "\n".join(p.text for p in d.paragraphs)
    except:
        return ""
    return ""

# ================================
# TOKENIZER / VOCAB
# ================================
SPECIAL = {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3}

def tokenize(text): return text.split()

def build_vocab(texts):
    freq = {}
    for t in texts:
        for tok in tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1
    token2id = dict(SPECIAL)
    id2token = {v: k for k, v in token2id.items()}
    idx = len(token2id)
    for tok in freq.keys():
        token2id[tok] = idx
        id2token[idx] = tok
        idx += 1
    return token2id, id2token

def encode(text, token2id):
    ids = [SPECIAL["BOS"]] + [token2id.get(t, SPECIAL["UNK"]) for t in tokenize(text)] + [SPECIAL["EOS"]]
    return ids

# ================================
# DATASET
# ================================
class LMData(Dataset):
    def __init__(self, ids, seq_len):
        self.data = [ids[i:i+seq_len] for i in range(len(ids)-seq_len)]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        return seq[:-1], seq[1:]

# ================================
# TRANSFORMER MODEL
# ================================
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TinyEncLM(nn.Module):
    def __init__(self, vocab_size, d=256, heads=8, layers=6, ff=512):
        super().__init__()
        self.vocab = vocab_size
        self.d = d
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = PosEnc(d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.fc = nn.Linear(d, vocab_size)
    def mask(self, n, device):
        m = torch.triu(torch.ones(n, n, device=device), 1)
        return m.masked_fill(m == 1, float("-inf"))
    def forward(self, x):
        x = self.emb(x)*math.sqrt(self.d)
        x = self.pos(x)
        mask = self.mask(x.size(1), x.device)
        h = self.enc(x, mask=mask)
        return self.fc(h)
    @torch.no_grad()
    def encode(self, x):
        x = self.emb(x)*math.sqrt(self.d)
        x = self.pos(x)
        mask = self.mask(x.size(1), x.device)
        h = self.enc(x, mask=mask)
        return h.mean(dim=1)

# ================================
# TRAINING (Mini-Batch)
# ================================
def train_lm(model, dataset, epochs=3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, model.vocab), y.reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {ep+1}/{epochs} - Loss {total/len(loader):.4f}")

# ================================
# TOPIC SPOTTING
# ================================
nlp = spacy.load("en_core_web_sm")
def extract_topics(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "SKILL", "PROJECT"]]

# ================================
# FAISS INDEX
# ================================
def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.numpy())
    return index

def search_faiss(index, q_emb, topk=5):
    D, I = index.search(q_emb.numpy().reshape(1, -1), topk)
    return I[0], D[0]

# ================================
# RERANKING + ANSWER EXTRACTION
# ================================
def lexical_overlap_score(question, chunk):
    table = str.maketrans("", "", string.punctuation)
    q_toks = [w.lower().translate(table) for w in question.split() if w.strip()]
    c_toks = [w.lower().translate(table) for w in chunk.split() if w.strip()]
    q_set, c_set = set(q_toks), set(c_toks)
    if not q_set or not c_set: return 0.0
    return len(q_set & c_set) / len(q_set | c_set)

def rerank_candidates(question, candidates):
    reranked = []
    for chunk, base_score in candidates:
        lex_score = lexical_overlap_score(question, chunk)
        final_score = 0.75 * base_score + 0.25 * lex_score
        reranked.append((chunk, final_score))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:3]

def extract_answer(question, chunks):
    table = str.maketrans("", "", string.punctuation)
    q_toks = set(w.lower().translate(table) for w in question.split() if len(w.strip()) > 2)
    snippets = []
    for chunk, _ in chunks:
        sentences = chunk.split(".")
        for sent in sentences:
            s = sent.strip()
            if not s: continue
            s_tokens = set(w.lower().translate(table) for w in s.split() if w.strip())
            if q_toks & s_tokens:
                snippets.append(s)
    return " ".join(snippets[:3]) if snippets else chunks[0][0]

# ================================
# LANGGRAPH ORCHESTRATION
# ================================
def chunk_documents(texts):
    chunks = []
    for d in texts:
        toks = d.split()
        for i in range(0, len(toks), 80):
            chunk = " ".join(toks[i:i+80])
            if chunk.strip():
                chunks.append(chunk)
    return chunks

def train_mini_batches(all_ids, token2id):
    model = TinyEncLM(vocab_size=len(token2id))
    batch_size = 500
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        dataset = LMData(batch_ids, seq_len=64)
        if len(dataset) > 5:
            train_lm(model, dataset, epochs=3)
    return model

def embed_chunks(model, chunks, token2id):
    embs = []
    for i in range(0, len(chunks), 16):
        batch = chunks[i:i+16]
        ids_list = [encode(t, token2id) for t in batch]
        max_len = max(len(x) for x in ids_list)
        padded = [x+[SPECIAL["PAD"]]*(max_len-len(x)) for x in ids_list]
        x = torch.tensor(padded, dtype=torch.long)
        e = model.encode(x)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)

# ================================
# STREAMLIT UI
# ================================
st.title("🚀 Enterprise Mobility RAG Assistant")

uploads = st.file_uploader("Upload documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if st.button("Build Index"):
    if not uploads:
        st.error("Upload documents first.")
    else:
        doc_texts = [load_text(f) for f in uploads if load_text(f).strip()]
        system_ctx = get_desktop_context()
        all_texts = doc_texts + [system_ctx]
        token2id, id2token = build_vocab(all_texts)
        all_ids = []
        for t in all_texts: all_ids.extend(encode(t, token2id))

        # LangGraph pipeline
        graph = Graph()
        graph.add_node("chunking", lambda: chunk_documents(doc_texts))
        graph.add_node("train", lambda: train_mini_batches(all_ids, token2id))
        graph.add_node("embedding", None)
        graph.add_node("faiss", None)

        chunks = chunk_documents(doc_texts)
        model = train_mini_batches(all_ids, token2id)
        chunk_embs = embed_chunks(model, chunks, token2id)
        index = build_faiss_index(chunk_embs)

        st.session_state.model = model
        st.session_state.token2id = token2id
        st.session_state.chunks = chunks
        st.session_state.chunk_embs = chunk_embs
        st.session_state.faiss_index = index
        st.success(f"Indexed {len(chunks)} chunks.")

query = st.text_input("Ask a question:")
if st.button("Search"):
    if st.session_state.faiss_index is None:
        st.error("Index not ready.")
    else:
        q_ids = encode(query, st.session_state.token2id)
        q_tensor = torch.tensor([q_ids], dtype=torch.long)
        q_emb = st.session_state.model.encode(q_tensor)
        idxs, scores = search_faiss(st.session_state.faiss_index, q_emb)
        candidates = [(st.session_state.chunks[i], float(scores[j])) for j, i in enumerate(idxs)]
        reranked = rerank_candidates(query, candidates)
        answer = extract_answer(query, reranked)
        st.write("### Answer:")
        st.write(answer)
        st.write("### Top Chunks:")
        for chunk, score in reranked:
            st.write(f"Score: {score:.4f}")
            st.write(chunk)
