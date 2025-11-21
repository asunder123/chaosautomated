"""
AI-Driven Desktop Observability — Stable Version with Sparklines

Run: streamlit run desktopautoobs.py
"""

import os
import time
import psutil
import sqlite3
import threading
import tempfile
import io
import base64

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# ------------------------ Paths & Model ------------------------
MODEL_DIR = os.path.join(tempfile.gettempdir(), "obs_ai_models")
os.makedirs(MODEL_DIR, exist_ok=True)
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "topology_cnn.keras")

# ------------------------ TensorFlow (optional) ------------------------
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

LATENT = 8
INPUT_LEN = 64

# ------------------------ DB ------------------------
DB_PATH = os.path.join(tempfile.gettempdir(), "obs_metrics.db")
DB = sqlite3.connect(DB_PATH, check_same_thread=False)
LOCK = threading.Lock()
DB.execute("CREATE TABLE IF NOT EXISTS metrics (ts INT, pid INT, name TEXT, cpu REAL, mem REAL)")
DB.commit()

# ------------------------ CNN helpers ------------------------
def load_or_create_cnn():
    if TF_AVAILABLE and os.path.exists(CNN_MODEL_PATH):
        try:
            return tf.keras.models.load_model(CNN_MODEL_PATH)
        except Exception:
            pass
    if not TF_AVAILABLE:
        return None
    model = models.Sequential([
        layers.Input(shape=(INPUT_LEN,1)),
        layers.Conv1D(32,5,activation='relu'),
        layers.MaxPool1D(2),
        layers.Conv1D(64,5,activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(LATENT)
    ])
    model.compile(optimizer='adam', loss='mse')
    try:
        model.save(CNN_MODEL_PATH)
    except Exception:
        pass
    return model

CNN = load_or_create_cnn()

# ------------------------ DB functions ------------------------
def insert_metrics(rows):
    with LOCK:
        DB.executemany("INSERT INTO metrics VALUES (?,?,?,?,?)", rows)
        DB.commit()

def query(window):
    cutoff = int(time.time()) - window
    try:
        df = pd.read_sql_query(f"SELECT * FROM metrics WHERE ts >= {cutoff}", DB)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    return df

# ------------------------ Collector ------------------------
if 'collecting' not in st.session_state:
    st.session_state.collecting = False
if 'collector_thread' not in st.session_state:
    st.session_state.collector_thread = None

def collect_once(top_n=12):
    out = []
    for p in psutil.process_iter(['pid','name','cpu_percent','memory_percent']):
        try:
            info = p.info
            out.append((info['pid'], info.get('name') or str(info['pid']), float(info.get('cpu_percent') or 0), float(info.get('memory_percent') or 0)))
        except Exception:
            pass
    out = sorted(out, key=lambda x: x[2], reverse=True)[:top_n]
    ts = int(time.time())
    rows = [(ts,pid,name,cpu,mem) for pid,name,cpu,mem in out]
    insert_metrics(rows)

def collector_loop(interval, top_n):
    while st.session_state.collecting:
        collect_once(top_n)
        time.sleep(interval)

def start_collector(interval, top_n):
    if st.session_state.collecting:
        return
    st.session_state.collecting = True
    th = threading.Thread(target=collector_loop, args=(interval, top_n), daemon=True)
    st.session_state.collector_thread = th
    th.start()

def stop_collector():
    st.session_state.collecting = False

# ------------------------ Topology ------------------------
def compute_golden_signals(cpu, mem):
    load = max(cpu, mem)
    if load < 40:
        return 'green'
    if load < 70:
        return 'yellow'
    return 'red'

def build_graph(top_n=12):
    procs = []
    for p in psutil.process_iter(['pid','name','cpu_percent','memory_percent','ppid','io_counters']):
        try:
            info = p.info
            io = info.get('io_counters')
            r_bytes = getattr(io, 'read_bytes', 0) if io else 0
            w_bytes = getattr(io, 'write_bytes', 0) if io else 0
            info['r_bytes'] = r_bytes
            info['w_bytes'] = w_bytes
            procs.append(info)
        except Exception:
            pass
    procs = sorted(procs, key=lambda x: float(x.get('cpu_percent') or 0), reverse=True)[:top_n]

    G = nx.DiGraph()
    for pr in procs:
        cpu = float(pr.get('cpu_percent') or 0)
        mem = float(pr.get('memory_percent') or 0)
        health = compute_golden_signals(cpu, mem)
        G.add_node(pr['pid'], label=pr.get('name') or str(pr['pid']), cpu=cpu, mem=mem, health=health, r_bytes=pr.get('r_bytes',0), w_bytes=pr.get('w_bytes',0))

    for pr in procs:
        ppid = pr.get('ppid')
        if ppid in G.nodes:
            G.add_edge(ppid, pr['pid'], title='Parent→Child', color='blue', width=2)

    for a in procs:
        for b in procs:
            if a['pid'] == b['pid']:
                continue
            a_io = (a.get('r_bytes') or 0) + (a.get('w_bytes') or 0)
            b_io = (b.get('r_bytes') or 0) + (b.get('w_bytes') or 0)
            if a_io > 0 and a_io > b_io:
                G.add_edge(a['pid'], b['pid'], title=f"Traffic IO {a_io}", color='orange', width=1)
    return G

# ------------------------ Embeddings ------------------------
def compute_embeddings(G):
    nodes = list(G.nodes())
    if not nodes:
        return {}
    A = nx.to_numpy_array(G)
    X = np.array([[G.nodes[n]['cpu'], G.nodes[n]['mem']] for n in nodes])
    vectors = []
    for i in range(len(nodes)):
        flat = np.concatenate([A[i], X[i]])
        v = np.zeros(INPUT_LEN)
        v[:min(len(flat), INPUT_LEN)] = flat[:INPUT_LEN]
        vectors.append(v)
    vectors = np.array(vectors)
    if CNN is not None:
        try:
            emb = CNN.predict(vectors.reshape((vectors.shape[0], vectors.shape[1], 1)), verbose=0)
            CNN.save(CNN_MODEL_PATH)
        except Exception:
            emb = np.random.randn(len(nodes), LATENT)
    else:
        emb = np.random.randn(len(nodes), LATENT)
    return {node: emb[i] for i, node in enumerate(nodes)}

# ------------------------ Sparklines ------------------------
def make_sparkline(pid, window=120, width=180, height=40):
    cutoff = int(time.time()) - window
    try:
        with LOCK:
            df_pid = pd.read_sql_query(f"SELECT ts,cpu FROM metrics WHERE ts >= {cutoff} AND pid = {int(pid)} ORDER BY ts ASC", DB)
    except Exception:
        return None
    buf = io.BytesIO()
    plt.figure(figsize=(width/100, height/100))
    if df_pid.empty:
        plt.plot([], [])
    else:
        df_pid['ts'] = pd.to_datetime(df_pid['ts'], unit='s')
        plt.plot(df_pid['ts'], df_pid['cpu'], linewidth=1)
        plt.fill_between(df_pid['ts'], df_pid['cpu'], alpha=0.15)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('ascii')

# ------------------------ Rendering ------------------------
def render_topology(G, emb):
    net = Network(height='650px', width='100%', directed=True)
    net.barnes_hut(gravity=-40000, spring_length=200, central_gravity=0.2, damping=0.09)

    for node, data in G.nodes(data=True):
        color = {'green':'#27AE60','yellow':'#F4D03F','red':'#C0392B'}.get(data.get('health','green'), '#27AE60')
        title = f"<b>{data.get('label')}</b><br>CPU: {data.get('cpu',0):.1f}%<br>MEM: {data.get('mem',0):.1f}%<br>Health: {data.get('health','unknown')}"

        try:
            b64 = make_sparkline(node)
            if b64:
                img_html = (
                    f'<div style="width:180px">'
                    f'<b>{data.get("label")}</b><br>'
                    f'<img src="data:image/png;base64,{b64}" style="width:180px;height           )
                title = (
                    img_html +
                    f'<div style="font-size:12px;">CPU: {data.get("cpu",0):.1f}% &nbsp; MEM: {data.get("mem",0):.1f}%'
                    f'<br>Health: {data.get("health","unknown")}'
                    f'<br>Read: {data.get("r_bytes",0):,} bytes<br>Write: {data.get("w_bytes",0):,} bytes</div>'
                )
        except Exception:
            pass

        net.add_node(node, label=str(data.get('label'))[:18], title=title, color=color, size=25)

    for u, v, d in G.edges(data=True):
        traffic = d.get('title','')
        try:
            vol = int(''.join(ch for ch in traffic if ch.isdigit()))
            width = max(1, min(15, vol // 50000))
        except:
            width = d.get('width', 1)
        net.add_edge(u, v, color=d.get('color', 'gray'), title=d.get('title', ''), width=width, arrows='to')

    path = os.path.join(tempfile.gettempdir(), f"golden_topology_{int(time.time())}.html")
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ------------------------ Recommendations ------------------------
def summarize(df):
    if df.empty:
        return 'No data yet'
    agg = df.groupby('pid').agg({'cpu':'mean','mem':'mean','name':'first'}).reset_index()
    top = agg.sort_values('cpu', ascending=False).head(5)
    out = ['Recommendations:']
    for _, r in top.iterrows():
        out.append(f"{r['name']} → CPU {r['cpu']:.1f}% MEM {r['mem']:.1f}%")
    return '\n'.join(out)

# ------------------------ Streamlit UI ------------------------
st.set_page_config(layout='wide')
st.title('AppDynamics-Style Desktop Observability (Stable)')
with st.sidebar:
    st.header('Collector')
    interval = st.number_input('Interval', 1, 10, 2)
    top_n = st.number_input('Top N', 3, 50, 12)
    if st.button('Start'):
        start_collector(interval, top_n)
        st.success('Collector started')
    if st.button('Stop'):
        stop_collector()
        st.warning('Collector stopped')
    auto = st.checkbox('Auto-refresh', value=True)
    refresh = st.number_input('Refresh every (sec)', 1, 10, 3)

left, right = st.columns((1.4, 1))
with left:
    st.subheader('Topology')
    G = build_graph(top_n)
    emb = compute_embeddings(G)
    html = render_topology(G, emb)
    components.html(html, height=650)
with right:
    st.subheader('Metrics (last 5 min)')
    df = query(300)
    if not df.empty:
        agg = df.groupby('ts').cpu.sum().reset_index()
        st.line_chart(agg.set_index('ts'))
        st.dataframe(df.tail(200))
    else:
        st.info('No metrics yet')
    st.subheader('Recommendations')
    st.code(summarize(df))

if auto:
    time.sleep(refresh)
    st.experimental_rerun()