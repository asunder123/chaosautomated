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
import sqlite3
import os
import datetime

# -----------------------------
# Neural Network Model
# -----------------------------
MODEL_PATH = "failure_predictor.pt"

class FailurePredictor(nn.Module):
    def __init__(self):
        super(FailurePredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Load or initialize model
model = FailurePredictor()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# -----------------------------
# SQLite Setup
# -----------------------------
DB_PATH = "process_intelligence.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS process_data (
    pid INTEGER,
    name TEXT,
    cpu REAL,
    ram REAL,
    net REAL,
    risk REAL,
    system_group TEXT,
    department TEXT,
    location TEXT,
    timestamp TEXT
)
""")
conn.commit()

# -----------------------------
# LangGraph Orchestration
# -----------------------------
class LangGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_process_node(self, pid, name, cpu, ram, net, risk):
        color = "red" if risk > 0.8 else "orange" if risk > 0.5 else "green"
        tooltip = (
            f"Process: {name}\n"
            f"CPU: {cpu}% ({'High' if cpu > 80 else 'OK'})\n"
            f"RAM: {ram:.1f}MB ({'High' if ram > 500 else 'OK'})\n"
            f"Network: {net} connections\n"
            f"Risk Score: {risk:.2f} ({'Critical' if risk > 0.8 else 'Warning' if risk > 0.5 else 'Healthy'})"
        )
        self.graph.add_node(pid, label=name, color=color, title=tooltip)

    def add_packet_flow(self, src_pid, dst_ip):
        self.graph.add_node(dst_ip, label=dst_ip, title="Network Node")
        self.graph.add_edge(src_pid, dst_ip, arrow=True)

    def build_topology(self, df, connections):
        for _, row in df.iterrows():
            features = torch.tensor([[row['cpu'], row['ram'], row['net']]], dtype=torch.float32)
            risk = model(features).item()
            self.add_process_node(row['pid'], row['name'], row['cpu'], row['ram'], row['net'], risk)
        for conn in connections:
            if conn.pid in df['pid'].values and conn.raddr:
                self.add_packet_flow(conn.pid, conn.raddr.ip)

    def render_html(self):
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        for node, data in self.graph.nodes(data=True):
            net.add_node(node, label=data.get('label', str(node)), title=data.get('title', 'Node'), color=data.get('color', 'blue'))
        for src, dst in self.graph.edges():
            net.add_edge(src, dst, arrows="to")
        return net.generate_html()

# -----------------------------
# Data Collection
# -----------------------------
def get_process_data(limit=10):
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        if len(processes) >= limit:
            break
        try:
            info = proc.info
            net_usage = sum(1 for conn in psutil.net_connections() if conn.pid == info['pid'])
            processes.append({
                'pid': info['pid'],
                'name': info['name'],
                'cpu': info['cpu_percent'],
                'ram': info['memory_info'].rss / (1024 * 1024),  # MB
                'net': net_usage,
                'system_group': f"System-{info['pid'] % 3}",
                'department': "Engineering",
                'location': "Hyderabad"
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pd.DataFrame(processes)

def get_network_connections():
    return psutil.net_connections(kind='inet')

# -----------------------------
# Dynamic Recommendations
# -----------------------------
def generate_recommendations(df):
    recs = []
    for _, row in df.iterrows():
        if row['risk'] > 0.8:
            recs.append(f"Terminate {row['name']} (PID {row['pid']}) - Critical risk.")
        elif row['cpu'] > 80:
            recs.append(f"Investigate {row['name']} - High CPU usage.")
        elif row['ram'] > 500:
            recs.append(f"Consider RAM upgrade for {row['name']}.")
        elif row['net'] > 10:
            recs.append(f"Audit network activity for {row['name']}.")
    return recs if recs else ["No immediate actions required."]

# -----------------------------
# Streamlit UI
# -----------------------------
st_autorefresh(interval=5000, key="process_refresh")

st.title("📊 Hierarchical Process Intelligence Dashboard")
st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")

# Fetch data
df = get_process_data(limit=10)
connections = get_network_connections()

# Train model incrementally
if not df.empty:
    X = torch.tensor(df[['cpu', 'ram', 'net']].values, dtype=torch.float32)
    y = torch.tensor([[1 if cpu > 80 else 0] for cpu in df['cpu']], dtype=torch.float32)
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)

# Predict risk and store in DB
df['risk'] = [model(torch.tensor([[row['cpu'], row['ram'], row['net']]], dtype=torch.float32)).item() for _, row in df.iterrows()]
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
for _, row in df.iterrows():
    cursor.execute("""
    INSERT INTO process_data (pid, name, cpu, ram, net, risk, system_group, department, location, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (row['pid'], row['name'], row['cpu'], row['ram'], row['net'], row['risk'], row['system_group'], row['department'], row['location'], timestamp))
conn.commit()

# Hierarchical summaries
st.subheader("Hierarchical Intelligence")
system_summary = pd.read_sql_query("SELECT system_group, AVG(risk) as avg_risk FROM process_data GROUP BY system_group", conn)
dept_summary = pd.read_sql_query("SELECT department, AVG(risk) as avg_risk FROM process_data GROUP BY department", conn)
loc_summary = pd.read_sql_query("SELECT location, AVG(risk) as avg_risk FROM process_data GROUP BY location", conn)

st.write("System-level Risk Summary")
st.dataframe(system_summary)
st.write("Department-level Risk Summary")
st.dataframe(dept_summary)
st.write("Location-level Risk Summary")
st.dataframe(loc_summary)

# Alerts
if len(df[df['risk'] > 0.8]) > 0:
    st.error("⚠ Critical Risk Detected!")
    st.table(df[df['risk'] > 0.8][['pid', 'name', 'cpu', 'ram', 'risk']])
else:
    st.success("✅ No critical risks detected.")

# Recommendations
st.subheader("Dynamic Recommendations")
for rec in generate_recommendations(df):
    st.write(f"- {rec}")

# Topology Viewer
lg = LangGraph()
lg.build_topology(df, connections)
html_content = lg.render_html()

st.subheader("Topology Viewer (Color-coded Risk)")
components.html(html_content, height=550, scrolling=True)

# Risk Table
st.subheader("Process Risk Details")
st.dataframe(df[['pid', 'name', 'cpu', 'ram', 'net', 'risk']].sort_values('risk', ascending=False))