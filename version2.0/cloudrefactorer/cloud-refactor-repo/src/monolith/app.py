import math
import os
import psutil
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================================
# DARK MODE CONFIG
# ================================
st.set_page_config(page_title="Tiny Transformer RAG Assistant", layout="wide")

DARK_CSS = """
<style>
body {
    background-color: #0D0D0D !important;
    color: white !important;
}
html, .stApp {
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


# ============================================================
# OPTIONAL SAFE IMPORTS (no crashes if missing)
# ============================================================

try:
    import PyPDF2
    PDF_ENABLED = True
except Exception:
    PDF_ENABLED = False

try:
    import docx
    DOCX_ENABLED = True
except Exception:
    DOCX_ENABLED = False


# ============================================================
# SYSTEM CONTEXT
# ============================================================

def get_desktop_context():
    """Return a textual snapshot of system metrics."""
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        processes = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                processes.append(p.info)
            except Exception:
                pass

        processes = sorted(
            processes, key=lambda x: x.get("cpu_percent", 0.0), reverse=True
        )[:8]

        txt = f"System Context:\nCPU {cpu:.1f}% | RAM {ram:.1f}% | Disk {disk:.1f}%\nTop Processes:\n"
        for p in processes:
            txt += (
                f"- {p.get('name')} | CPU {p.get('cpu_percent',0.0):.1f}% | "
                f"MEM {p.get('memory_percent',0.0):.1f}%\n"
            )
        return txt
    except Exception:
        return "System context unavailable."


# ============================================================
# FILE LOADING
# ============================================================

def load_text(upload):
    """Read text from txt/pdf/doc/docx; return empty string on failure."""
    name = upload.name.lower()
    try:
        if name.endswith(".txt"):
            return upload.read().decode("utf-8", "ignore")
        if name.endswith(".pdf") and PDF_ENABLED:
            reader = PyPDF2.PdfReader(upload)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if name.endswith(".docx") and DOCX_ENABLED:
            d = docx.Document(upload)
            return "\n".join(p.text for p in d.paragraphs)
        if name.endswith(".doc"):
            return upload.read().decode("utf-8", "ignore")
    except Exception:
        return ""
    return ""


# ============================================================
# TOKENIZER / VOCAB
# ============================================================

SPECIAL = {"PAD": 0, "BOS": 1, "EOS": 2, "UNK": 3}

def tokenize(text: str):
    return text.split()

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
    ids = [SPECIAL["BOS"]]
    for t in tokenize(text):
        ids.append(token2id.get(t, SPECIAL["UNK"]))
    ids.append(SPECIAL["EOS"])
    return ids

def decode(ids, id2token):
    return " ".join(id2token.get(i, "UNK") for i in ids if i > 3)


# ============================================================
# DATASET
# ============================================================

class LMData(Dataset):
    def __init__(self, ids, seq_len: int):
        self.data = []
        for i in range(len(ids) - seq_len):
            self.data.append(ids[i:i+seq_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        return seq[:-1], seq[1:]


# ============================================================
# TRANSFORMER (LM + encoder, WIDER)
# ============================================================

class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TinyEncLM(nn.Module):
    """
    Wider transformer: d=256, heads=8, layers=6, ff=512
    Used as LM for training + encoder for embeddings.
    """
    def __init__(self, vocab_size, d=256, heads=8, layers=6, ff=512):
        super().__init__()
        self.vocab = vocab_size
        self.d = d
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = PosEnc(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.1,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.fc = nn.Linear(d, vocab_size)

    def mask(self, n, device):
        m = torch.triu(torch.ones(n, n, device=device), 1)
        return m.masked_fill(m == 1, float("-inf"))

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d)
        x = self.pos(x)
        mask = self.mask(x.size(1), x.device)
        h = self.enc(x, mask=mask)
        return self.fc(h)

    @torch.no_grad()
    def encode(self, x):
        x = self.emb(x) * math.sqrt(self.d)
        x = self.pos(x)
        mask = self.mask(x.size(1), x.device)
        h = self.enc(x, mask=mask)
        return h.mean(dim=1)


# ============================================================
# TRAINING
# ============================================================

def train_lm(model, dataset, epochs: int = 20):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
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


# ============================================================
# CHUNKING + EMBEDDING + 2-STAGE RETRIEVAL
# ============================================================

CHUNK_TOKENS = 80
TOPK_BI = 8   # first-stage bi-encoder candidates
TOPK_FINAL = 3  # final number of chunks after reranking

def build_chunks_and_tags(doc_texts, system_context):
    """
    Build chunks from docs and system context, tagging each chunk as 'doc' or 'system'.
    """
    chunks = []
    chunk_tags = []

    # Document chunks
    for d in doc_texts:
        toks = tokenize(d)
        for i in range(0, len(toks), CHUNK_TOKENS):
            part = toks[i:i+CHUNK_TOKENS]
            if part:
                chunks.append(" ".join(part))
                chunk_tags.append("doc")

    # System context chunks (tagged separately)
    s_toks = tokenize(system_context)
    for i in range(0, len(s_toks), CHUNK_TOKENS):
        part = s_toks[i:i+CHUNK_TOKENS]
        if part:
            chunks.append(" ".join(part))
            chunk_tags.append("system")

    return chunks, chunk_tags

@torch.no_grad()
def embed_chunks(model, chunks, token2id):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    embs = []
    for i in range(0, len(chunks), 16):
        batch = chunks[i:i+16]
        ids_list = [encode(t, token2id) for t in batch]
        max_len = max(len(x) for x in ids_list)
        padded = [x + [SPECIAL["PAD"]] * (max_len - len(x)) for x in ids_list]
        x = torch.tensor(padded, dtype=torch.long, device=dev)
        e = model.encode(x)
        embs.append(e.cpu())
    if not embs:
        return torch.empty(0, model.d)
    return torch.cat(embs, dim=0)

@torch.no_grad()
def embed_query(model, query, token2id):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = encode(query, token2id)
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    e = model.encode(x)
    return e[0].cpu()

def lexical_overlap_score(question: str, chunk: str) -> float:
    """
    Simple lexical overlap: Jaccard over tokens (lowercased, stripped punctuation).
    """
    import string
    table = str.maketrans("", "", string.punctuation)
    q_toks = [w.lower().translate(table) for w in question.split() if w.strip()]
    c_toks = [w.lower().translate(table) for w in chunk.split() if w.strip()]

    q_set = set(q_toks)
    c_set = set(c_toks)
    if not q_set or not c_set:
        return 0.0
    inter = len(q_set & c_set)
    union = len(q_set | c_set)
    if union == 0:
        return 0.0
    return inter / union

def cross_encoder_rerank(question, candidates):
    """
    "Cross-encoder-style" reranker that combines:
      - base semantic score from bi-encoder
      - lexical overlap between question and chunk
    This is a lightweight approximation to a cross-encoder.
    candidates: list of (chunk_text, base_score, tag)
    """
    reranked = []
    for chunk_text, base_score, tag in candidates:
        lex_score = lexical_overlap_score(question, chunk_text)
        # Weighted combination: mostly semantic, some lexical
        final_score = 0.75 * base_score + 0.25 * lex_score
        reranked.append((chunk_text, final_score, tag))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:TOPK_FINAL]

def retrieve(model, query, token2id, chunks, chunk_embs, chunk_tags):
    """
    Two-stage retrieval:
      1. Bi-encoder: transformer embedding cosine similarity → top-K candidates
      2. Cross-encoder-style rerank: semantic + lexical overlap → final K
      System chunks are only used if question is system-related.
    """
    if chunk_embs is None or chunk_embs.shape[0] == 0 or not chunks:
        return []

    # Determine if user wants system context
    sys_terms = ["cpu", "ram", "disk", "memory", "process", "psutil", "system"]
    q_lower = query.lower()
    wants_system = any(w in q_lower for w in sys_terms)

    # Bi-encoder stage
    q_emb = embed_query(model, query, token2id)
    qn = q_emb / (q_emb.norm() + 1e-8)
    cn = chunk_embs / (chunk_embs.norm(dim=1, keepdim=True) + 1e-8)
    sims = torch.mv(cn, qn)

    vals, idx = torch.topk(sims, k=min(TOPK_BI, sims.numel()))
    candidates = []
    for s, i in zip(vals.tolist(), idx.tolist()):
        tag = chunk_tags[i]
        # Skip system chunks unless clearly asked
        if tag == "system" and not wants_system:
            continue
        if s > 0.0:
            candidates.append((chunks[i], s, tag))

    if not candidates:
        return []

    # Cross-encoder-style reranking
    reranked = cross_encoder_rerank(query, candidates)
    return reranked


def extract_answer(question, retrieved):
    """
    Extractive answer: pick sentences from retrieved doc chunks,
    prioritizing those that share words with the question.
    """
    if not retrieved:
        return "I do not know based on the available documents."

    import string
    table = str.maketrans("", "", string.punctuation)
    q_toks = set(
        w.lower().translate(table)
        for w in question.split()
        if len(w.strip()) > 2
    )

    snippets = []
    for chunk, score, tag in retrieved:
        sentences = chunk.split(".")
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            s_tokens = set(
                w.lower().translate(table)
                for w in s.split()
                if w.strip()
            )
            if q_toks & s_tokens:
                snippets.append(s)

    if snippets:
        return " ".join(snippets[:5])

    # fallback: just return top chunk
    best_chunk, _, _ = retrieved[0]
    return best_chunk.strip()


# ============================================================
# STREAMLIT UI – CHAT
# ============================================================

st.title("🌓 Tiny Transformer RAG Assistant (Wider + Cross-Encoder Rerank)")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "model" not in st.session_state:
    st.session_state.model = None

if "token2id" not in st.session_state:
    st.session_state.token2id = None
if "id2token" not in st.session_state:
    st.session_state.id2token = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embs" not in st.session_state:
    st.session_state.chunk_embs = None
if "chunk_tags" not in st.session_state:
    st.session_state.chunk_tags = []

# Load saved model/index if present
if (
    st.session_state.model is None
    and os.path.exists("tiny_enc_lm.pth")
    and os.path.exists("tiny_enc_meta.pth")
):
    try:
        meta = torch.load("tiny_enc_meta.pth", map_location="cpu")
        st.session_state.token2id = meta["token2id"]
        st.session_state.id2token = meta["id2token"]
        st.session_state.chunks = meta["chunk_texts"]
        st.session_state.chunk_embs = meta["chunk_embs"]
        st.session_state.chunk_tags = meta.get(
            "chunk_tags", ["doc"] * len(st.session_state.chunks)
        )

        model = TinyEncLM(vocab_size=len(st.session_state.token2id))
        model.load_state_dict(torch.load("tiny_enc_lm.pth", map_location="cpu"))
        st.session_state.model = model
        st.success("Loaded saved transformer + index.")
    except Exception:
        st.warning("Could not load saved model/index; please retrain.")


# Chat bubble renderer
chat_placeholder = st.empty()

def render_chat():
    html = ""
    for m in st.session_state.history:
        if m["role"] == "user":
            html += f"""
            <div style="background:#007AFF;padding:10px;border-radius:10px;
            margin:8px;color:white;max-width:70%;text-align:right;margin-left:30%;">
            {m['text']}
            </div>
            """
        else:
            html += f"""
            <div style="background:#333333;padding:10px;border-radius:10px;
            margin:8px;color:white;max-width:70%;text-align:left;">
            {m['text']}
            </div>
            """
    chat_placeholder.markdown(html, unsafe_allow_html=True)

render_chat()


# ============================================================
# TRAIN / BUILD INDEX
# ============================================================

st.subheader("📄 Upload & Train")

uploads = st.file_uploader(
    "Upload documents to train and index",
    type=["txt", "pdf", "doc", "docx"],
    accept_multiple_files=True,
)

if st.button("Train / Rebuild Model"):
    if not uploads:
        st.error("Please upload at least one document.")
    else:
        doc_texts = [load_text(f) for f in uploads]
        doc_texts = [d for d in doc_texts if d.strip()]

        if not doc_texts:
            st.error("No usable text extracted from uploaded files.")
        else:
            system_ctx = get_desktop_context()
            # For LM training, include system context so model knows those tokens
            training_docs = doc_texts + [system_ctx]

            token2id, id2token = build_vocab(training_docs)
            st.session_state.token2id = token2id
            st.session_state.id2token = id2token

            # Build chunks and tags (docs vs system)
            chunks, chunk_tags = build_chunks_and_tags(doc_texts, system_ctx)
            st.session_state.chunks = chunks
            st.session_state.chunk_tags = chunk_tags

            # Build LM training sequence
            all_ids = []
            for d in training_docs:
                all_ids.extend(encode(d, token2id))

            if len(all_ids) < 20:
                st.error("Not enough tokens to train. Add more/larger documents.")
            else:
                seq_len = min(80, max(8, len(all_ids) // 3))
                dataset = LMData(all_ids, seq_len)

                if len(dataset) < 5:
                    st.error("Not enough training samples; add more text.")
                else:
                    st.success(
                        f"Training dataset has {len(dataset)} samples (seq_len={seq_len})."
                    )

                    model = TinyEncLM(vocab_size=len(token2id))
                    st.info("Training wider tiny transformer (20 epochs)...")
                    train_lm(model, dataset, epochs=20)
                    st.session_state.model = model

                    st.info("Computing semantic embeddings for chunks...")
                    chunk_embs = embed_chunks(model, chunks, token2id)
                    st.session_state.chunk_embs = chunk_embs

                    # Save everything
                    torch.save(model.state_dict(), "tiny_enc_lm.pth")
                    torch.save(
                        {
                            "token2id": token2id,
                            "id2token": id2token,
                            "chunk_texts": chunks,
                            "chunk_embs": chunk_embs,
                            "chunk_tags": chunk_tags,
                        },
                        "tiny_enc_meta.pth",
                    )
                    st.success("Model trained and index saved.")


# ============================================================
# CHAT INPUT (SAFE CLEARING VIA DYNAMIC KEY)
# ============================================================

st.subheader("💬 Chat")

if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0

def handle_send():
    key = f"chat_input_{st.session_state.chat_input_key}"
    msg = st.session_state.get(key, "").strip()
    if not msg:
        return

    # Add user message
    st.session_state.history.append({"role": "user", "text": msg})

    # Generate reply
    if (
        st.session_state.model is None
        or st.session_state.token2id is None
        or st.session_state.chunk_embs is None
        or st.session_state.chunk_embs.shape[0] == 0
        or not st.session_state.chunks
    ):
        bot = "Model/index not ready. Upload docs and click 'Train / Rebuild Model'."
    else:
        retrieved = retrieve(
            st.session_state.model,
            msg,
            st.session_state.token2id,
            st.session_state.chunks,
            st.session_state.chunk_embs,
            st.session_state.chunk_tags,
        )
        bot = extract_answer(msg, retrieved)

    st.session_state.history.append({"role": "bot", "text": bot})

    # Bump key → new empty input widget next rerun
    st.session_state.chat_input_key += 1
    render_chat()


col1, col2 = st.columns([5, 1])
with col1:
    st.text_input(
        "Message:",
        key=f"chat_input_{st.session_state.chat_input_key}",
        on_change=handle_send,
    )
with col2:
    if st.button("Send"):
        handle_send()