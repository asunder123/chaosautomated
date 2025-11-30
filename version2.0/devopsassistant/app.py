
# ----------------------------------------------------------
# STREAMLIT APP: Multi-Model CLI Generator with Adaptive Router & Incremental Retraining
# ----------------------------------------------------------
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_sequence
import os
import pickle
import numpy as np

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
# Domain Keywords for Router
# ==========================================================
domain_keywords = {
    "aws": ["aws", "s3", "ec2", "iam", "lambda", "cloudformation"],
    "k8s": ["kubectl", "pod", "pods", "deployment", "namespace", "rollout"],
    "docker": ["docker", "container", "image", "compose", "volume", "network"]
}

# ==========================================================
# Router Embedding
# ==========================================================
class RouterEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, ids):
        return self.embedding(ids)

router_vocab = sorted(set(word for words in domain_keywords.values() for word in words))
router_stoi = {w: i for i, w in enumerate(router_vocab)}
router_itos = {i: w for w, i in router_stoi.items()}

router_model = RouterEmbedding(len(router_vocab))

def compute_domain_embeddings():
    domain_embeds = {}
    for domain, words in domain_keywords.items():
        ids = torch.tensor([router_stoi.get(w, 0) for w in words])
        emb = router_model(ids).mean(dim=0)
        emb = F.normalize(emb, p=2, dim=0)
        domain_embeds[domain] = emb
    return domain_embeds

domain_embeddings = compute_domain_embeddings()

def refresh_router_embeddings():
    global domain_embeddings
    domain_embeddings = compute_domain_embeddings()

def detect_domain_fuzzy(prompt):
    tokens = [w for w in prompt.lower().split() if w in router_stoi]
    if not tokens:
        return None, 0.0
    ids = torch.tensor([router_stoi.get(w, 0) for w in tokens])
    prompt_emb = router_model(ids).mean(dim=0)
    prompt_emb = F.normalize(prompt_emb, p=2, dim=0)
    sims = {d: torch.cosine_similarity(prompt_emb, emb, dim=0).item() for d, emb in domain_embeddings.items()}
    best_domain = max(sims, key=sims.get)
    return best_domain, sims[best_domain]

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
# Transformer Model
# ==========================================================
class CLITransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dim=512, nhead=8, num_layers=6, dropout=0.4):
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
        return None, None, None

    corrections = list(set(corrections))  # Deduplicate
    combined_data = DATASETS[domain] + corrections
    s_tok = Tokenizer([s for s, _ in combined_data])
    t_tok = Tokenizer([t for _, t in combined_data])

    model = CLITransformer(s_tok.vocab_size, t_tok.vocab_size)
    model = train(combined_data, s_tok, t_tok, model=model, epochs=20, lr=1e-4)

    save_model(domain, model, s_tok, t_tok)
    refresh_router_embeddings()
    return model, s_tok, t_tok

# ==========================================================
# Beam Search with Stronger Constraints
# ==========================================================
def beam_search(model, s_tok, t_tok, prompt, beam_width=3, max_len=15, repetition_penalty=2.0):
    src = torch.tensor(s_tok.encode(prompt)).unsqueeze(1)
    mem = model.encode(src)

    beams = [(torch.tensor([[t_tok.bos]]), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            m = mask(seq.size(0))
            dec = model.decode(seq, mem, m)
            logits = model.fc(dec[-1])
            probs = torch.log_softmax(logits, dim=-1)

            tokens = seq.squeeze().tolist()
            if isinstance(tokens, int):
                tokens = [tokens]

            # Apply repetition penalty
            for token in set(tokens):
                probs[0][token] /= repetition_penalty

            topk = torch.topk(probs, beam_width)
            for idx, val in zip(topk.indices[0], topk.values[0]):
                new_seq = torch.cat([seq, torch.tensor([[idx]])], dim=0)
                # Coverage penalty: discourage repeats
                penalty = 0.1 * (len(tokens) - len(set(tokens)))
                new_beams.append((new_seq, score + val.item() - penalty))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Stop if EOS or repeated token pattern
        tokens = beams[0][0].squeeze().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        if beams[0][0][-1].item() == t_tok.eos or len(set(tokens)) < len(tokens):
            break

    best_seq = beams[0][0].squeeze(1).tolist()
    return t_tok.decode(best_seq), np.exp(beams[0][1] / len(best_seq))

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.title("⚡ Multi-Model CLI Generator with Adaptive Router & Incremental Retraining")

if "prompt" not in st.session_state:
    st.session_state["prompt"] = "scale deployment myapp to 3 replicas"
if "correction_text" not in st.session_state:
    st.session_state["correction_text"] = ""
if "domain" not in st.session_state:
    st.session_state["domain"] = None

if st.button("Train All Models"):
    st.info("Training models for AWS, Kubernetes, and Docker...")
    for domain in DOMAINS:
        data = DATASETS[domain]
        s_tok = Tokenizer([s for s, _ in data])
        t_tok = Tokenizer([t for _, t in data])
        model = train(data, s_tok, t_tok, epochs=50)
        save_model(domain, model, s_tok, t_tok)
    refresh_router_embeddings()
    st.success("Training completed!")

st.session_state["prompt"] = st.text_input("Describe your task:", value=st.session_state["prompt"])

if st.button("Generate Command"):
    prompt = st.session_state["prompt"].strip()
    if not prompt:
        st.error("Please enter a prompt.")
    else:
        domain, confidence = detect_domain_fuzzy(prompt)
        if not domain:
            st.error("Could not detect domain.")
        else:
            st.session_state["domain"] = domain
            st.write(f"Detected Domain: **{domain.upper()}** (Confidence: {confidence:.2f})")
            model, s_tok, t_tok = load_model(domain)
            if not model:
                st.error(f"Model for {domain} not trained yet.")
            else:
                cmd, conf = beam_search(model, s_tok, t_tok, prompt)
                st.success(f"Generated Command: `{cmd}`")
                st.write(f"Generation Confidence: `{conf:.2f}`")

st.session_state["correction_text"] = st.text_input("Provide correct command (optional):", value=st.session_state["correction_text"])

if st.button("Submit Correction"):
    correction = st.session_state["correction_text"].strip()
    prompt = st.session_state["prompt"].strip()

    if correction and prompt:
        if not st.session_state["domain"]:
            st.error("Please generate a command first to detect the domain.")
        else:
            domain = st.session_state["domain"]
            with open(f"{CORRECTION_DIR}/{domain}_corrections.txt", "a") as f:
                f.write(f"{prompt}|||{correction}\n")
            st.info("✅ Correction saved. Retraining model with ALL corrections...")
            model, s_tok, t_tok = retrain_with_corrections(domain)
            if model:
                st.success("Model updated with all corrections! Try generating again.")
            else:
                st.warning("No corrections found for retraining.")
    else:
        st.warning("Please provide a valid correction and prompt before submitting.")

if st.session_state["domain"]:
    domain = st.session_state["domain"]
    st.subheader(f"📜 Saved Corrections for {domain.upper()}")
    correction_file = f"{CORRECTION_DIR}/{domain}_corrections.txt"
    if os.path.exists(correction_file):
        with open(correction_file, "r") as f:
            corrections = f.readlines()
        if corrections:
            for idx, line in enumerate(corrections, 1):
                st.write(f"{idx}. {line.strip()}")
            if st.button("Clear All Corrections"):
                os.remove(correction_file)
                st.warning(f"All corrections for {domain.upper()} have been cleared.")
        else:
            st.write("No corrections saved yet.")
    else:
        st.write("No corrections file found for this domain.")
