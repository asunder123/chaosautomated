
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
import psutil
import pandas as pd
import networkx as nx
from pyvis.network import Network
import torch
import torch.nn as nn
import torch.optim as optim
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Paths & Config
# -----------------------------
CNN_MODEL_PATH = "process_discovery_cnn_v3.pt"
TINY_LM_PATH = "tiny_lm_big.pt"
STATE_PATH = "prev_topology.pkl"
REFRESH_INTERVAL = 15000

# -----------------------------
# CNN Risk Model
# -----------------------------
class ProcessDiscoveryCNN(nn.Module):
    def __init__(self):
        super(ProcessDiscoveryCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

cnn_model = ProcessDiscoveryCNN()
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH))
cnn_model.eval()

# -----------------------------
# Tiny LM with Attention
# -----------------------------
rules = [
    {"input": "cpu > 80", "output": "High CPU usage detected: Consider scaling or killing process"},
    {"input": "ram > 500", "output": "Memory spike: Optimize or restart process"},
    {"input": "risk > 0.7", "output": "High risk: Investigate anomaly and apply mitigation"}
]

class TinyTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>"]
        self.vocab = set(self.special_tokens)
        for r in rules:
            self.vocab.update(r["input"].split())
            self.vocab.update(r["output"].split())
        self.word2idx = {word: idx for idx, word in enumerate(sorted(self.vocab))}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text, max_len=30):
        tokens = ["<SOS>"] + text.split() + ["<EOS>"]
        ids = [self.word2idx.get(t, 0) for t in tokens]
        ids += [self.word2idx["<PAD>"]] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        words = [self.idx2word[i] for i in ids if i != self.word2idx["<PAD>"]]
        return " ".join(words).replace("<SOS>", "").replace("<EOS>", "").strip()

tokenizer = TinyTokenizer()

# Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

# Bigger Tiny LM with Attention
class BiggerTinyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super(BiggerTinyLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention = Attention(hidden_dim)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        embedded_src = self.embedding(src)
        encoder_outputs, hidden = self.encoder(embedded_src)  # hidden shape: (num_layers, batch, hidden_dim)
        outputs = []
        input_token = torch.tensor([[tokenizer.word2idx["<SOS>"]]] * src.size(0)).to(src.device)

        for t in range(30):
            embedded_tgt = self.embedding(input_token)
            decoder_output, hidden = self.decoder(embedded_tgt, hidden)  # Pass full hidden state
            attn_weights = self.attention(hidden[-1], encoder_outputs)  # Use last layer for attention
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            combined = decoder_output.squeeze(1) + context.squeeze(1)
            pred = self.fc(combined)
            outputs.append(pred.unsqueeze(1))

            if tgt is not None and random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t].unsqueeze(1)
            else:
                input_token = pred.argmax(1).unsqueeze(1)

        return torch.cat(outputs, dim=1)

tiny_lm = BiggerTinyLM(vocab_size=len(tokenizer.vocab))

# Auto-train if missing
if not os.path.exists(TINY_LM_PATH):
    st.warning("No pre-trained Tiny LM found. Training a bigger model...")
    optimizer = optim.Adam(tiny_lm.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<PAD>"])
    train_data = []
    for r in rules:
        src = torch.tensor(tokenizer.encode(r["input"])).unsqueeze(0)
        tgt = torch.tensor(tokenizer.encode(r["output"], max_len=30)).unsqueeze(0)
        train_data.append((src, tgt))
    tiny_lm.train()
    for epoch in range(100):
        total_loss = 0
        for src, tgt in train_data:
            optimizer.zero_grad()
            output = tiny_lm(src, tgt)
            loss = criterion(output.view(-1, len(tokenizer.vocab)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    torch.save(tiny_lm.state_dict(), TINY_LM_PATH)
    st.success("✅ Bigger Tiny LM trained and saved!")
else:
    tiny_lm.load_state_dict(torch.load(TINY_LM_PATH))
tiny_lm.eval()

# -----------------------------
# Utility Functions
# -----------------------------
def get_process_data(limit=10):
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        if len(processes) >= limit:
            break
        try:
            info = proc.info
            processes.append({
                'pid': info['pid'],
                'name': info['name'],
                'cpu': info['cpu_percent'],
                'ram': info['memory_info'].rss / (1024 * 1024),
                'net': random.randint(0, 150)
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pd.DataFrame(processes)

def generate_recommendation(condition):
    src = torch.tensor(tokenizer.encode(condition)).unsqueeze(0).to(next(tiny_lm.parameters()).device)
    output = tiny_lm(src, teacher_forcing_ratio=0.0)
    pred_ids = output.argmax(2).squeeze().tolist()
    return tokenizer.decode(pred_ids)

def online_update(model, feedback_data, tokenizer, lr=0.0005):
    if not feedback_data:
        return
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<PAD>"])
    for fb in feedback_data:
        src = torch.tensor(tokenizer.encode(fb["condition"])).unsqueeze(0).to(next(model.parameters()).device)
        tgt = torch.tensor(tokenizer.encode(fb["output"], max_len=30)).unsqueeze(0).to(next(model.parameters()).device)
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output.view(-1, len(tokenizer.vocab)), tgt.view(-1))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), TINY_LM_PATH)
    st.success("✅ Tiny LM updated with feedback!")

# -----------------------------
# LangGraph for Smooth Topology
# -----------------------------
class LangGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.weak_edges = []

    def add_process_node(self, pid, name, cpu, ram, net, risk, cluster):
        color_map = {0: "blue", 1: "orange", 2: "red"}
        color = color_map.get(cluster, "green")
        size = 20 + (cpu / 10)
        tooltip = f"Process: {name}\nCPU: {cpu}%\nRAM: {ram:.1f}MB\nRisk: {risk:.2f}\nCluster: {cluster}"
        self.graph.add_node(pid, label=name, color=color, title=tooltip, size=size)

    def add_packet_flow(self, src_pid, dst_ip):
        self.graph.add_node(dst_ip, label=dst_ip, title="Network Node")
        self.graph.add_edge(src_pid, dst_ip, arrow=True)

    def add_weak_links(self, df):
        similarity_matrix = cosine_similarity(df[['cpu','ram','net','risk']])
        for i, pid1 in enumerate(df['pid']):
            for j, pid2 in enumerate(df['pid']):
                if i != j and similarity_matrix[i, j] < 0.3:
                    self.weak_edges.append((pid1, pid2))

    def render_html(self, prev_positions=None):
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 200,
              "updateInterval": 25
            }
          },
          "edges": {
            "smooth": {
              "type": "dynamic"
            }
          }
        }
        """)
        for node, data in self.graph.nodes(data=True):
            pos = prev_positions.get(node) if prev_positions else None
            net.add_node(node, label=data.get('label', str(node)), title=data.get('title', 'Node'),
                         color=data.get('color', 'blue'), size=data.get('size', 20),
                         x=pos['x'] if pos else None, y=pos['y'] if pos else None)
        for src, dst in self.graph.edges():
            net.add_edge(src, dst, arrows="to", width=2)
        for src, dst in self.weak_edges:
            net.add_edge(src, dst, arrows="to", dashes=True, color="gray", width=1)
        return net.generate_html()

# -----------------------------
# Streamlit UI
# -----------------------------
st_autorefresh(interval=REFRESH_INTERVAL, key="refresh")
st.title("🌐 Smooth Topology & Bigger Tiny LM with Attention")
st.caption("CNN-driven risk + advanced recommendations")

df = get_process_data(limit=10)
sequence_data = torch.tensor(df[['cpu', 'ram', 'net']].values, dtype=torch.float32).unsqueeze(1)
cnn_preds = cnn_model(sequence_data)
clusters = torch.argmax(cnn_preds, dim=1).tolist()
df['cluster'] = clusters
df['risk'] = torch.softmax(cnn_preds, dim=1)[:, 2].tolist()

# Build topology
lg = LangGraph()
for _, row in df.iterrows():
    lg.add_process_node(row['pid'], row['name'], row['cpu'], row['ram'], row['net'], row['risk'], row['cluster'])
connections = psutil.net_connections(kind='inet')
for conn in connections:
    if conn.pid in df['pid'].values and conn.raddr:
        lg.add_packet_flow(conn.pid, conn.raddr.ip)
lg.add_weak_links(df)

prev_state = {}
if os.path.exists(STATE_PATH):
    with open(STATE_PATH, "rb") as f:
        prev_state = pickle.load(f)
prev_positions = prev_state.get('positions', {})
html_content = lg.render_html(prev_positions)
components.html(html_content, height=550, scrolling=True)

positions = {node: {'x': None, 'y': None} for node in df['pid']}
prev_state['positions'] = positions
with open(STATE_PATH, "wb") as f:
    pickle.dump(prev_state, f)

# Correlation Heatmap
st.subheader("📊 Feature Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df[['cpu','ram','net','risk']].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Recommendations
st.subheader("✅ Adaptive Recommendations with Attention")
feedback_data = []

for _, row in df.iterrows():
    if row['risk'] > 0.7:
        condition = "risk > 0.7"
    elif row['cpu'] > 80:
        condition = "cpu > 80"
    elif row['ram'] > 500:
        condition = "ram > 500"
    else:
        condition = "cpu < 30"

    rec = generate_recommendation(condition)
    st.write(f"PID {row['pid']} ({row['name']}): {rec}")

    fb = st.radio(f"Feedback for PID {row['pid']}", ["👍 Good", "👎 Needs Improvement"], key=f"fb_{row['pid']}")
    if fb == "👎 Needs Improvement":
        new_action = st.text_input(f"Suggest better recommendation for PID {row['pid']}", key=f"new_{row['pid']}")
        if new_action:
            feedback_data.append({"condition": condition, "output": new_action})

# Button outside loop
if st.button("Update Model with Feedback"):
    online_update(tiny_lm, feedback_data, tokenizer)
