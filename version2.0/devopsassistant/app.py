
# ----------------------------------------------------------
# STREAMLIT APP: CLI Generator with Local Embeddings + RAG + Incremental Retraining
# ----------------------------------------------------------
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import faiss
import numpy as np
import os
import pickle
from torch.nn.utils.rnn import pad_sequence

# ==========================================================
# CONFIG
# ==========================================================
MODEL_DIR = "models"
CORRECTION_DIR = "corrections"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CORRECTION_DIR, exist_ok=True)

DOMAINS = ["aws", "k8s", "docker"]

# ==========================================================
# DATASETS
# ==========================================================
DATASETS = {
    "aws": [
        ("list all s3 buckets", "aws s3 ls"),
        ("create s3 bucket mybucket in us-east-1", "aws s3 mb s3://mybucket --region us-east-1"),
        ("delete s3 bucket mybucket", "aws s3 rb s3://mybucket --force"),
        ("upload file.txt to s3 bucket mybucket", "aws s3 cp file.txt s3://mybucket"),
        ("list ec2 instances", "aws ec2 describe-instances"),
        ("start ec2 instance i-12345", "aws ec2 start-instances --instance-ids i-12345"),
        ("stop ec2 instance i-12345", "aws ec2 stop-instances --instance-ids i-12345"),
        ("configure AWS CLI profile", "aws configure"),
        ("check AWS CLI version", "aws --version"),
    ],
    "k8s": [
        ("list pods", "kubectl get pods"),
        ("list pods in all namespaces", "kubectl get pods -A"),
        ("describe pod mypod", "kubectl describe pod mypod"),
        ("delete pod mypod", "kubectl delete pod mypod"),
        ("scale deployment myapp to 3 replicas", "kubectl scale deployment myapp --replicas=3"),
        ("apply manifest file.yaml", "kubectl apply -f file.yaml"),
        ("get logs of pod mypod", "kubectl logs mypod"),
        ("exec into pod mypod", "kubectl exec -it mypod -- /bin/bash"),
        ("check kubectl version", "kubectl version"),
    ],
    "docker": [
        ("show docker images", "docker images"),
        ("remove docker image myimage", "docker rmi myimage"),
        ("build docker image from dockerfile", "docker build -t myimage ."),
        ("run container from image", "docker run -d myimage"),
        ("stop docker container", "docker stop <container_id>"),
        ("list running containers", "docker ps"),
        ("exec into container", "docker exec -it <container_id> /bin/bash"),
        ("check docker version", "docker --version"),
    ]
}

# ==========================================================
# Tokenizer
# ==========================================================
class Tokenizer:
    def __init__(self, texts):
        base = ["<pad>", "<bos>", "<eos>", "<unk>"]
        vocab = set(base)
        for t in texts:
            for tok in t.lower().split():
                vocab.add(tok)
        self.vocab = sorted(vocab)
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.pad = self.stoi["<pad>"]
        self.bos = self.stoi["<bos>"]
        self.eos = self.stoi["<eos>"]
        self.unk = self.stoi["<unk>"]

    def encode(self, text):
        ids = [self.stoi.get(tok, self.unk) for tok in text.lower().split()]
        return [self.bos] + ids + [self.eos]

    def decode(self, ids):
        toks = []
        for i in ids:
            t = self.itos.get(int(i), "<unk>")
            if t not in ["<bos>", "<eos>", "<pad>"]:
                toks.append(t)
        return " ".join(toks)

    @property
    def vocab_size(self):
        return len(self.vocab)

# ==========================================================
# Local Embedding Model
# ==========================================================
class PromptEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        emb = self.embedding(token_ids)
        return emb.mean(dim=0)  # Mean pooling

# ==========================================================
# Transformer Model
# ==========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[: x.size(0)]

class CLITransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dim=512, nhead=8, num_layers=6, dropout=0.3):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, dim)
        self.tgt_emb = nn.Embedding(tgt_vocab, dim)
        self.pos = PositionalEncoding(dim)

        enc_layer = nn.TransformerEncoderLayer(dim, nhead, 1024, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        dec_layer = nn.TransformerDecoderLayer(dim, nhead, 1024, dropout=dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        self.fc = nn.Linear(dim, tgt_vocab)

    def forward(self, src, tgt, mask):
        mem = self.encoder(self.pos(self.src_emb(src)))
        dec = self.decoder(self.pos(self.tgt_emb(tgt)), mem, tgt_mask=mask)
        return self.fc(dec)

    def encode(self, src):
        return self.encoder(self.pos(self.src_emb(src)))

    def decode(self, ys, mem, mask):
        return self.decoder(self.pos(self.tgt_emb(ys)), mem, tgt_mask=mask)

def mask(sz):
    m = torch.triu(torch.ones(sz, sz), 1).bool()
    return m.float().masked_fill(m, float("-inf"))

# ==========================================================
# Training & Retraining
# ==========================================================
def make_batch(data, s_tok, t_tok):
    xs = [torch.tensor(s_tok.encode(s)) for s, _ in data]
    ys = [torch.tensor(t_tok.encode(t)) for _, t in data]
    return pad_sequence(xs, False, s_tok.pad), pad_sequence(ys, False, t_tok.pad)

def train(data, s_tok, t_tok, model=None, epochs=20, lr=5e-4):
    if model is None:
        model = CLITransformer(s_tok.vocab_size, t_tok.vocab_size)

    src_b, tgt_b = make_batch(data, s_tok, t_tok)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=t_tok.pad, label_smoothing=0.1)

    for ep in range(epochs):
        opt.zero_grad()
        inp = tgt_b[:-1]
        out = tgt_b[1:]
        m = mask(inp.size(0))

        pred = model(src_b, inp, m)
        loss = loss_fn(pred.reshape(-1, pred.size(-1)), out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return model

def save_model(domain, model, s_tok, t_tok):
    torch.save(model.state_dict(), f"{MODEL_DIR}/{domain}_model.pt")
    with open(f"{MODEL_DIR}/{domain}_tok.pkl", "wb") as f:
        pickle.dump((s_tok, t_tok), f)

def load_model(domain):
    model_path = f"{MODEL_DIR}/{domain}_model.pt"
    tok_path = f"{MODEL_DIR}/{domain}_tok.pkl"
    if not os.path.exists(model_path):
        return None, None, None
    with open(tok_path, "rb") as f:
        s_tok, t_tok = pickle.load(f)
    model = CLITransformer(s_tok.vocab_size, t_tok.vocab_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, s_tok, t_tok

def retrain_with_corrections(domain):
    correction_file = f"{CORRECTION_DIR}/{domain}_corrections.txt"
    corrections = []
    if os.path.exists(correction_file):
        with open(correction_file, "r") as f:
            for line in f:
                if "|||" in line:
                    p, c = line.strip().split("|||")
                    corrections.append((p, c))

    if not corrections:
        return None

    combined_data = DATASETS[domain] + corrections
    s_tok = Tokenizer([s for s, _ in combined_data])
    t_tok = Tokenizer([t for _, t in combined_data])
    model = train(combined_data, s_tok, t_tok, epochs=30, lr=1e-4)
    save_model(domain, model, s_tok, t_tok)
    return model

# ==========================================================
# FAISS Retrieval with Local Embeddings
# ==========================================================
def compute_embeddings(prompts, tok, embed_model):
    vectors = []
    for p in prompts:
        ids = torch.tensor(tok.encode(p))
        vec = embed_model(ids).detach().numpy()
        vectors.append(vec)
    return np.vstack(vectors)

def build_faiss_index(domain, tok, embed_model):
    data = DATASETS[domain]
    prompts = [p for p, _ in data]
    embeddings = compute_embeddings(prompts, tok, embed_model)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, f"{MODEL_DIR}/{domain}_index.faiss")
    with open(f"{MODEL_DIR}/{domain}_prompts.pkl", "wb") as f:
        pickle.dump(prompts, f)

def retrieve_examples(domain, query, tok, embed_model, k=3):
    index = faiss.read_index(f"{MODEL_DIR}/{domain}_index.faiss")
    with open(f"{MODEL_DIR}/{domain}_prompts.pkl", "rb") as f:
        prompts = pickle.load(f)
    ids = torch.tensor(tok.encode(query))
    query_emb = embed_model(ids).detach().numpy().reshape(1, -1)
    distances, indices = index.search(query_emb, k)
    return [prompts[idx] for idx in indices[0]]

# ==========================================================
# Beam Search with RAG
# ==========================================================
def beam_search(model, s_tok, t_tok, prompt, retrieved, beam_width=3, max_len=15):
    rag_context = prompt + " " + " ".join(retrieved)
    src = torch.tensor(s_tok.encode(rag_context)).unsqueeze(1)
    mem = model.encode(src)

    beams = [(torch.tensor([[t_tok.bos]]), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            m = mask(seq.size(0))
            dec = model.decode(seq, mem, m)
            logits = model.fc(dec[-1])
            probs = torch.log_softmax(logits, dim=-1)

            topk = torch.topk(probs, beam_width)
            for idx, val in zip(topk.indices[0], topk.values[0]):
                new_seq = torch.cat([seq, torch.tensor([[idx]])], dim=0)
                new_beams.append((new_seq, score + val.item()))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        if beams[0][0][-1].item() == t_tok.eos:
            break

    best_seq = beams[0][0].squeeze(1).tolist()
    return t_tok.decode(best_seq)

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.title("⚡ CLI Generator with Local Embeddings + RAG + Incremental Retraining")

if st.button("Train All Models"):
    st.info("Training models and building FAISS indexes...")
    for domain in DOMAINS:
        data = DATASETS[domain]
        s_tok = Tokenizer([s for s, _ in data])
        t_tok = Tokenizer([t for _, t in data])
        model = train(data, s_tok, t_tok, epochs=50)
        save_model(domain, model, s_tok, t_tok)

        # Build FAISS index with local embeddings
        embed_model = PromptEmbedding(s_tok.vocab_size)
        build_faiss_index(domain, s_tok, embed_model)
    st.success("Training and FAISS indexing completed!")

prompt = st.text_input("Describe your task:", value="scale deployment myapp to 3 replicas")
domain = st.selectbox("Select Domain:", DOMAINS)

if st.button("Generate Command"):
    model, s_tok, t_tok = load_model(domain)
    if not model:
        st.error(f"Model for {domain} not trained yet.")
    else:
        embed_model = PromptEmbedding(s_tok.vocab_size)
        retrieved = retrieve_examples(domain, prompt, s_tok, embed_model)
        st.write(f"Retrieved Examples: {retrieved}")
        cmd = beam_search(model, s_tok, t_tok, prompt, retrieved)
        st.success(f"Generated Command: `{cmd}`")

# Correction submission
correction = st.text_input("Provide correct command (optional):", value="")
if st.button("Submit Correction"):
    if correction.strip() and prompt.strip():
        with open(f"{CORRECTION_DIR}/{domain}_corrections.txt", "a") as f:
            f.write(f"{prompt}|||{correction}\n")
        st.info("✅ Correction saved. Retraining model with ALL corrections...")
        model = retrain_with_corrections(domain)
        if model:
            st.success("Model updated with corrections! Try generating again.")
        else:
            st.warning("No corrections found for retraining.")
    else:
        st.warning("Please provide a valid correction and prompt before submitting.")
