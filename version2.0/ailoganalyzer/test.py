
# app.py
# ------------------------------------------------------------------------------
# Transformer .pt Model Analyzer & Reverse Engineering Aid (Streamlit)
# Safe inspection of TorchScript/state_dict artifacts, human-readable insights,
# complexity & memory estimates, optional smoke-test inference, and a
# reverse-engineering blueprint generator with config export.
# ------------------------------------------------------------------------------

import io
import json
import time
import math
import tempfile
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- UI SETUP ----------
st.set_page_config(page_title="Transformer .pt Reverse Engineering Aid", layout="wide")
st.title("🧭 Transformer `.pt` Reverse Engineering Aid (PyTorch)")
st.caption(
    "Safely load TorchScript or state_dict, extract structure & stats, generate human-readable insights, "
    "estimate complexity, and produce a reconstruction blueprint (model skeleton + config)."
)

# ---------- HELPERS ----------
@st.cache_resource
def _device():
    """CPU for deterministic, safe behavior."""
    return torch.device("cpu")

def safe_load_torchscript(buffer: bytes) -> Tuple[Optional[torch.jit.ScriptModule], str]:
    """Attempt to load as TorchScript (safe)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(buffer)
            path = f.name
        m = torch.jit.load(path, map_location=_device())
        return m, "torchscript"
    except Exception as e:
        return None, f"torchscript: {e}"

def safe_load_state_container(buffer: bytes) -> Tuple[Optional[dict], str]:
    """
    Load any pickled dict safely as possible.
    Prefer weights_only=True (PyTorch 2+). Return the raw container if dict, else None.
    """
    try:
        container = torch.load(io.BytesIO(buffer), map_location=_device(), weights_only=True)  # type: ignore
    except TypeError:
        try:
            container = torch.load(io.BytesIO(buffer), map_location=_device())
        except Exception as e:
            return None, f"state_container: {e}"
    except Exception as e:
        return None, f"state_container: {e}"

    if isinstance(container, dict):
        return container, "container(dict)"
    return None, "not_a_dict"

def extract_primary_state_dict(container: dict) -> Tuple[Optional[Dict[str, torch.Tensor]], str]:
    """
    Given a raw checkpoint-like container, extract the primary state_dict if present,
    otherwise return the container itself if it looks like a pure state_dict.
    """
    if "state_dict" in container and isinstance(container["state_dict"], dict):
        return container["state_dict"], "state_dict(checkpoint)"
    # Heuristic: if values look like Tensors, assume it's a pure state_dict
    if container and all(isinstance(v, torch.Tensor) for v in container.values() if v is not None):
        return container, "state_dict"
    return None, "no_state_dict_found"

def parameter_stats_tensors(named_tensors: List[tuple]) -> pd.DataFrame:
    """Generic tensor stats from (name, tensor) pairs."""
    rows = []
    for k, t in named_tensors:
        if not isinstance(t, torch.Tensor):
            continue
        np_t = t.detach().cpu().numpy()
        rows.append({
            "name": k,
            "shape": list(np_t.shape),
            "ndim": int(np_t.ndim),
            "numel": int(np_t.size),
            "dtype": str(t.dtype).replace("torch.", ""),
            "mean": float(np.mean(np_t)),
            "std": float(np.std(np_t)),
            "min": float(np.min(np_t)),
            "max": float(np.max(np_t)),
            "norm2": float(np.linalg.norm(np_t))
        })
    return pd.DataFrame(rows)

def parameter_stats(sd: Dict[str, torch.Tensor]) -> pd.DataFrame:
    return parameter_stats_tensors(list(sd.items()))

def bytes_for_params(df: pd.DataFrame) -> int:
    dtype_sizes = {
        "float32": 4, "bfloat16": 2, "float16": 2, "float64": 8,
        "int8": 1, "int16": 2, "int32": 4, "int64": 8,
        "uint8": 1, "bool": 1
    }
    total = 0
    for _, r in df.iterrows():
        total += r["numel"] * dtype_sizes.get(str(r["dtype"]), 4)
    return int(total)

def infer_transformer_config_from_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Heuristic config extraction from common key patterns."""
    cfg: Dict[str, Any] = {}
    # Embedding matrix
    emb_w = None
    for k, v in sd.items():
        if v is None or not isinstance(v, torch.Tensor):
            continue
        if ("embedding" in k or "embeddings.word_embeddings.weight" in k or
            "tok_embeddings.weight" in k or "token_embedding.weight" in k or "wte" in k) and v.ndim == 2:
            emb_w = v
            break
    if emb_w is not None:
        cfg["vocab_size"], cfg["d_model_guess"] = emb_w.shape

    # Positional embeddings (if any)
    pos_keys = [k for k in sd.keys() if any(x in k.lower() for x in ["position_embeddings", "pos_embedding", "wpe"])]
    for k in pos_keys:
        t = sd[k]
        if isinstance(t, torch.Tensor) and t.ndim in (1, 2):
            cfg["max_position_embeddings"] = int(t.shape[0]) if t.ndim == 2 else int(t.shape[0])

    # Layers count via block patterns
    block_ids = set()
    for k in sd.keys():
        for token in ["encoder.layer.", "decoder.layer.", "transformer.h.", "layers.", "blocks."]:
            if token in k:
                try:
                    idx_str = k.split(token)[1].split(".")[0]
                    block_ids.add(int(idx_str))
                except Exception:
                    pass
    if block_ids:
        cfg["num_layers_guess"] = max(block_ids) + 1

    # Linear shapes to guess d_model and MLP width
    linear_ws = [v.shape for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim == 2 and ".weight" in k]
    if linear_ws:
        dims = np.array(linear_ws)
        min_dims, counts = np.unique(dims.min(axis=1), return_counts=True)
        max_dims, counts2 = np.unique(dims.max(axis=1), return_counts=True)
        cfg["d_model_linear_mode"] = int(min_dims[np.argmax(counts)])
        cfg["mlp_dim_mode"] = int(max_dims[np.argmax(counts2)])
    return cfg

def flops_estimate_transformer(L: int, d_model: int, n_heads: int, mlp_ratio: float, n_layers: int, batch_size: int = 1) -> Dict[str, Any]:
    """Approximate FLOPs per forward pass for a standard block."""
    B = batch_size
    attn = B * (L**2) * d_model
    proj = 4 * B * L * (d_model**2)
    mlp = 2 * B * L * d_model * int(mlp_ratio * d_model)
    per_layer = attn + proj + mlp
    total = per_layer * n_layers
    return {
        "per_layer_flops": int(per_layer),
        "total_flops": int(total),
        "components": {"attn": int(attn), "proj": int(proj), "mlp": int(mlp)}
    }

def activation_memory_bytes(B: int, L: int, d_model: int, dtype_bytes: int = 4, n_layers: int = 1) -> int:
    """Rule-of-thumb: ~6 * d_model activations per token per layer."""
    per_layer = B * L * d_model * 6 * dtype_bytes
    return int(per_layer * n_layers)

def detect_architecture(sd: Dict[str, torch.Tensor], names: List[str]) -> Tuple[str, str]:
    """
    Return (arch_type, rationale).
    Heuristics on common naming patterns seen in GPT/BERT/ViT families.
    """
    lname = [n.lower() for n in names]
    rationale = []
    # GPT-like (decoder-only): transformer.h.N, attn, mlp, wte, wpe
    if any("transformer.h." in n for n in names) or ("wte" in " ".join(lname) and "wpe" in " ".join(lname)):
        rationale.append("Found 'transformer.h.' blocks and token/pos embeddings typical of GPT-style decoders.")
        return "GPT-like (decoder-only, causal)", " ".join(rationale)
    # BERT-like (encoder-only): encoder.layer.N.attention, embeddings.word_embeddings
    if any("encoder.layer." in n for n in names) and any("embeddings.word_embeddings" in n for n in names):
        rationale.append("Found 'encoder.layer.*.attention' and 'embeddings.word_embeddings' typical of BERT encoders.")
        return "BERT-like (encoder-only, bidirectional)", " ".join(rationale)
    # ViT: patch_embed/proj, cls_token, pos_embed
    if any("patch_embed" in n for n in lname) or any("cls_token" in n for n in lname) or any("pos_embed" in n for n in lname):
        rationale.append("Found 'patch_embed'/'cls_token'/'pos_embed' patterns typical of ViT.")
        return "ViT-like (vision transformer)", " ".join(rationale)
    rationale.append("Could not match clear GPT/BERT/ViT patterns. Treat as custom Transformer.")
    return "Unknown/Custom Transformer", " ".join(rationale)

def generate_skeleton(arch_type: str, cfg: Dict[str, Any], d_model: int, n_layers: int, n_heads: int, mlp_ratio: float) -> str:
    """
    Generate a minimal reconstruction blueprint using stock PyTorch transformer blocks.
    This is a starting point to refactor into your exact architecture.
    """
    vocab_size = int(cfg.get("vocab_size", 30522))
    max_pos = int(cfg.get("max_position_embeddings", 2048))

    if arch_type.startswith("GPT-like"):
        # Decoder-only skeleton
        return f'''import torch
import torch.nn as nn

class ReverseEngineeredGPT(nn.Module):
    def __init__(self, vocab_size={vocab_size}, d_model={d_model}, num_layers={n_layers},
                 num_heads={n_heads}, mlp_ratio={mlp_ratio}, max_positions={max_pos}):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_positions, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads,
                                           dim_feedforward=int(mlp_ratio*d_model), batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # tie weights manually if needed

    def forward(self, input_ids):
        B, L = input_ids.size()
        pos = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=input_ids.device)*float('-inf'), diagonal=1)
        # decode with a dummy memory (decoder-only can be emulated with self-attn in each layer)
        # Here we pass x as both tgt and memory to reuse the API; custom modules would separate them
        x = self.decoder(tgt=x, memory=x, tgt_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
'''
    if arch_type.startswith("BERT-like"):
        # Encoder-only skeleton
        return f'''import torch
import torch.nn as nn

class ReverseEngineeredBERT(nn.Module):
    def __init__(self, vocab_size={vocab_size}, d_model={d_model}, num_layers={n_layers},
                 num_heads={n_heads}, mlp_ratio={mlp_ratio}, max_positions={max_pos}):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_positions, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                           dim_feedforward=int(mlp_ratio*d_model), batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, vocab_size, bias=False)  # placeholder

    def forward(self, input_ids):
        B, L = input_ids.size()
        pos = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.word_embeddings(input_ids) + self.position_embeddings(pos)
        x = self.layer_norm(self.drop(x))
        x = self.encoder(x)  # bidirectional
        logits = self.cls_head(x)  # task-specific head to be replaced
        return logits
'''
    if arch_type.startswith("ViT-like"):
        # Vision skeleton (very generic)
        return f'''import torch
import torch.nn as nn

class ReverseEngineeredViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, d_model={d_model}, num_layers={n_layers},
                 num_heads={n_heads}, mlp_ratio={mlp_ratio}, num_classes=1000):
        super().__init__()
        self.patch = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + num_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                           dim_feedforward=int(mlp_ratio*d_model), batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)  # (B, d_model, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS
        return self.head(x)
'''
    # Unknown/custom: provide a generic encoder skeleton
    return f'''import torch
import torch.nn as nn

class ReverseEngineeredTransformer(nn.Module):
    def __init__(self, vocab_size={vocab_size}, d_model={d_model}, num_layers={n_layers},
                 num_heads={n_heads}, mlp_ratio={mlp_ratio}, max_positions={max_pos}):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_positions, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                           dim_feedforward=int(mlp_ratio*d_model), batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, L = input_ids.size()
        pos = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.embed(input_ids) + self.pos(pos)
        x = self.encoder(x)
        x = self.norm(x)
        return self.out(x)
'''

def extract_training_hints(raw_container: Optional[dict]) -> Dict[str, Any]:
    """
    Inspect raw checkpoint container (if present) for optimizer clues.
    Returns { 'optimizer': str|None, 'has_state': bool, 'keys': [...] }
    """
    hints = {"optimizer": None, "has_state": False, "keys": []}
    if not isinstance(raw_container, dict):
        return hints
    keys = list(raw_container.keys())
    hints["keys"] = keys
    # Common patterns where optimizer is stored
    for k in keys:
        kl = k.lower()
        if "optimizer" in kl or "optim_state" in kl:
            hints["optimizer"] = "Adam/AdamW (guessed)" if "adam" in str(raw_container.get(k, "")).lower() else "Unknown optimizer"
            hints["has_state"] = True
            break
    # Some frameworks store 'amp' or 'scaler' for mixed precision
    for k in keys:
        if "scaler" in k.lower():
            hints["mixed_precision"] = True
    return hints

def build_markdown_report(
    artifact_type: Optional[str],
    total_params: Optional[int],
    total_bytes: Optional[int],
    dtype_choice: str,
    cfg_guess: Dict[str, Any],
    d_model: Optional[int],
    n_layers: Optional[int],
    n_heads: int,
    mlp_ratio: float,
    flops: Optional[Dict[str, Any]],
    act_mem: Optional[int],
    insights: List[str],
    recommendations: List[str],
    warnings: List[str],
    arch_type: Optional[str],
    arch_rationale: Optional[str],
    training_hints: Dict[str, Any]
) -> str:
    """Create a human-readable Markdown report."""
    lines = []
    lines.append("# Transformer Model Analysis & Reverse Engineering Report")
    lines.append("")
    lines.append(f"**Artifact type:** `{artifact_type}`")
    if total_params is not None:
        lines.append(f"**Total parameters:** {total_params:,}")
    if total_bytes is not None:
        lines.append(f"**Parameter memory (approx):** {total_bytes/1e6:.2f} MB ({dtype_choice})")
    lines.append("")
    if arch_type:
        lines.append(f"**Inferred architecture:** {arch_type}")
        if arch_rationale:
            lines.append(f"_Rationale_: {arch_rationale}")
        lines.append("")
    if cfg_guess:
        lines.append("## Heuristic Config Guess")
        lines.append("```json")
        lines.append(json.dumps(cfg_guess, indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## Complexity & Memory Estimates")
    lines.append(f"- d_model: {d_model}")
    lines.append(f"- layers: {n_layers}")
    lines.append(f"- heads: {n_heads}")
    lines.append(f"- MLP ratio: {mlp_ratio}")
    if flops is not None:
        lines.append(f"- Per‑layer FLOPs: {flops['per_layer_flops']:,}")
        lines.append(f"- Total FLOPs/forward: {flops['total_flops']:,}")
    if act_mem is not None:
        lines.append(f"- Activation memory (approx): {act_mem/1e6:.2f} MB")
    lines.append("")
    if insights:
        lines.append("## Key Insights")
        for i in insights:
            lines.append(f"- {i}")
        lines.append("")
    if recommendations:
        lines.append("## Suggested Improvements")
        for r in recommendations:
            lines.append(f"- {r}")
        lines.append("")
    if warnings:
        lines.append("## Warnings / Potential Issues")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    if training_hints:
        lines.append("## Training Hints (from checkpoint container, if present)")
        lines.append("```json")
        lines.append(json.dumps(training_hints, indent=2))
        lines.append("```")
        lines.append("")
    lines.append("> Generated by Streamlit analyzer on CPU using safe loading (TorchScript first, then weights‑only state_dict).")
    return "\n".join(lines)

# ---------- SIDEBAR INPUTS ----------
st.sidebar.header("Inference/Complexity Settings")
B = st.sidebar.number_input("Batch size (B)", min_value=1, value=1, step=1)
L = st.sidebar.number_input("Sequence length (L)", min_value=8, value=128, step=8)
d_model_override = st.sidebar.number_input("Override d_model (if unknown)", min_value=0, value=0, step=8)
n_heads = st.sidebar.number_input("Number of attention heads (estimate)", min_value=1, value=8, step=1)
mlp_ratio = st.sidebar.number_input("MLP expansion ratio (≈4 for GPT/BERT)", min_value=1.0, value=4.0, step=0.5)
n_layers_override = st.sidebar.number_input("Override #layers (if unknown)", min_value=0, value=0, step=1)
dtype_choice = st.sidebar.selectbox("Compute dtype (for memory est.)", ["float32", "bfloat16", "float16"])
dtype_bytes = {"float32": 4, "bfloat16": 2, "float16": 2}[dtype_choice]

st.sidebar.header("Optional tokenizer")
tok_json = st.sidebar.file_uploader("Upload tokenizer.json (optional)", type=["json"])
vocab_json = st.sidebar.file_uploader("Upload vocab.json (optional)", type=["json"])
merges_txt = st.sidebar.file_uploader("Upload merges.txt (optional)", type=["txt"])
sample_prompt = st.sidebar.text_area("Sample prompt (optional)", "Hello world from Capgemini!")

# ---------- MAIN UPLOAD ----------
uploaded = st.file_uploader("Upload your `.pt` model file", type=["pt", "pth"], accept_multiple_files=False)

artifact_type: Optional[str] = None
jit_module: Optional[torch.jit.ScriptModule] = None
state_dict: Optional[Dict[str, torch.Tensor]] = None
checkpoint_container: Optional[dict] = None
status_msgs: List[str] = []

if uploaded is not None:
    buffer = uploaded.getvalue()

    # Try TorchScript first
    m, msg = safe_load_torchscript(buffer)
    if isinstance(m, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
        jit_module = m
        artifact_type = "torchscript"
        status_msgs.append("Loaded as TorchScript.")
    else:
        status_msgs.append(f"TorchScript load attempt: {msg}")

    # If not TS, try generic container (checkpoint or state_dict)
    if jit_module is None:
        container, msg2 = safe_load_state_container(buffer)
        if isinstance(container, dict):
            checkpoint_container = container.copy()
            sd, kind = extract_primary_state_dict(container)
            if isinstance(sd, dict):
                state_dict = sd
                artifact_type = kind
                status_msgs.append(f"Extracted {kind} from container.")
            else:
                status_msgs.append(f"Could not extract primary state_dict ({kind}).")
        else:
            status_msgs.append(f"Container load attempt: {msg2}")

    # No inline ternary
    if artifact_type:
        st.success(f"Artifact type: {artifact_type}")
    else:
        st.error("Could not load as TorchScript or state_dict.")

    with st.expander("Load attempts & messages"):
        for s in status_msgs:
            st.write("•", s)

# ---------- ANALYTICS ----------
df_stats: Optional[pd.DataFrame] = None
total_params: Optional[int] = None
total_bytes: Optional[int] = None
cfg_guess: Dict[str, Any] = {}
d_model: Optional[int] = None
n_layers: Optional[int] = None
flops: Optional[Dict[str, Any]] = None
act_mem: Optional[int] = None

if artifact_type == "torchscript" and jit_module is not None:
    st.subheader("🧩 TorchScript Module Summary")

    def module_tree(mod):
        rows = []
        for name, sub in mod.named_modules():
            rows.append({"module": name, "type": type(sub).__name__})
        return pd.DataFrame(rows)

    df_tree = module_tree(jit_module)
    st.dataframe(df_tree, use_container_width=True)

    # Parameter stats from named_parameters
    named_params = list(jit_module.named_parameters())
    if named_params:
        df_stats = parameter_stats_tensors(named_params)
        st.subheader("📊 Weights Analytics (TorchScript)")
        st.dataframe(df_stats.head(200), use_container_width=True)
        total_params = int(df_stats["numel"].sum())
        total_bytes = bytes_for_params(df_stats)

    st.info("For complexity, set B, L, d_model, heads, mlp_ratio in the sidebar. TorchScript inference requires known input signatures.")

elif state_dict is not None:
    st.subheader("📊 Weights Analytics (state_dict)")
    df_stats = parameter_stats(state_dict)
    st.dataframe(df_stats.head(200), use_container_width=True)

    total_params = int(df_stats["numel"].sum())
    total_bytes = bytes_for_params(df_stats)
    st.markdown(f"**Total parameters:** {total_params:,}  \n**Parameter memory (approx):** {total_bytes/1e6:.2f} MB ({dtype_choice})")

    # Plots: distributions & layer share
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].hist(df_stats["mean"], bins=50, color="#4e79a7"); ax[0].set_title("Per‑tensor mean")
    ax[1].hist(df_stats["std"], bins=50, color="#f28e2b"); ax[1].set_title("Per‑tensor std")
    ax[2].hist(df_stats["norm2"], bins=50, color="#59a14f"); ax[2].set_title("Per‑tensor ||W||₂")
    st.pyplot(fig)

    def classify(k):
        if "emb" in k or "embedding" in k: return "Embedding"
        if "attn" in k or "mha" in k or "multihead" in k: return "Attention"
        if "norm" in k or "layernorm" in k or "ln_" in k: return "LayerNorm"
        if "proj" in k or ("linear" in k) or (".weight" in k and state_dict[k].ndim == 2): return "Linear"
        return "Other"
    df_stats["layer_class"] = df_stats["name"].map(classify)
    layer_hist = df_stats.groupby("layer_class")["numel"].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(layer_hist["layer_class"], layer_hist["numel"]/1e6, color="#e15759")
    ax2.set_ylabel("Parameters (M)")
    ax2.set_title("Layer Class Parameter Share")
    st.pyplot(fig2)

    # Heuristic config
    cfg_guess = infer_transformer_config_from_keys(state_dict)
    st.markdown("### 🧠 Config Guess (heuristic)")
    st.json(cfg_guess)

# ---------- COMPLEXITY & MEMORY ----------
if df_stats is not None:
    d_model = d_model_override if d_model_override > 0 else int(cfg_guess.get("d_model_guess", cfg_guess.get("d_model_linear_mode", 512)))
    n_layers = n_layers_override if n_layers_override > 0 else int(cfg_guess.get("num_layers_guess", 12))
    flops = flops_estimate_transformer(L=L, d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, n_layers=n_layers, batch_size=B)
    act_mem = activation_memory_bytes(B=B, L=L, d_model=d_model, dtype_bytes=dtype_bytes, n_layers=n_layers)

    st.markdown("### ⚙️ Complexity & Memory (estimates)")
    st.write(f"- **d_model:** {d_model}, **layers:** {n_layers}, **heads:** {n_heads}, **MLP ratio:** {mlp_ratio}")
    st.write(f"- **Per‑layer FLOPs:** {flops['per_layer_flops']:,}")
    st.write(f"- **Total FLOPs / forward:** {flops['total_flops']:,}")
    st.write(f"- **Activation memory (approx):** {act_mem/1e6:.2f} MB ({dtype_choice})")

# ---------- HUMAN‑READABLE INSIGHTS ----------
st.subheader("💡 Insights & Improvement Directions")
insights: List[str] = []
recommendations: List[str] = []
warnings: List[str] = []

if df_stats is not None and total_params is not None:
    total_params_m = total_params / 1e6
    # Model size bucket
    if total_params_m < 50:
        insights.append(f"Model size is **small** ({total_params_m:.1f}M params)—good for edge/low‑latency use.")
        recommendations.append("If accuracy is insufficient, consider modestly increasing depth or width.")
    elif total_params_m < 500:
        insights.append(f"Model size is **moderate** ({total_params_m:.1f}M params)—fits many enterprise NLP workloads.")
        recommendations.append("If latency is a concern, explore pruning or 8‑bit/4‑bit quantization (AWQ/GPTQ).")
    else:
        insights.append(f"Model size is **very large** ({total_params_m:.1f}M params)—GPU/accelerator likely needed.")
        recommendations.append("Consider distillation or parameter‑efficient fine‑tuning (LoRA/adapters).")

    # Weight distribution health
    mean_abs = float(abs(df_stats["mean"]).mean())
    std_mean = float(df_stats["std"].mean())
    mean_std_ratio = std_mean / (mean_abs + 1e-8)
    if mean_std_ratio < 0.1:
        insights.append("Weights show **low variance relative to mean** → potential underfitting or over‑smoothed updates.")
        recommendations.append("Review initialization, optimizer settings, and consider layer‑wise LR or warmup schedules.")
    elif mean_std_ratio > 10:
        insights.append("Weights show **very high variance** → risk of training instability.")
        recommendations.append("Enable gradient clipping; check normalization layers and LR schedule.")

    # Outlier check (rough)
    outlier_counts = 0
    sample = df_stats.sample(min(200, len(df_stats)), random_state=42)  # speed
    for _, r in sample.iterrows():
        try:
            if abs(r["max"]) > 5 * (r["std"] + 1e-8):
                outlier_counts += 1
        except Exception:
            pass
    if outlier_counts > len(sample) * 0.2:
        warnings.append("Many tensors have extreme values (|w| >> σ).")
        recommendations.append("Verify loss scaling/precision; consider lower LR, better weight decay, or re‑init unstable layers.")

    # Dtype mix
    dtype_mix = df_stats["dtype"].value_counts()
    if len(dtype_mix) > 1:
        insights.append(f"Mixed parameter dtypes detected: {', '.join(map(str, dtype_mix.index.tolist()))}.")
        recommendations.append("For inference, unify to a single dtype (fp32/bf16/fp16) to avoid conversion overheads.")

    # Normalization presence
    has_ln = any(("norm" in n.lower() or "layernorm" in n.lower() or "ln_" in n.lower()) for n in df_stats["name"])
    if not has_ln:
        warnings.append("No LayerNorm detected—check architecture; transformers rely on normalization.")
        recommendations.append("Ensure normalization (LayerNorm/RMSNorm) is correctly placed and configured.")

    # Layer composition & share
    if "layer_class" in df_stats.columns:
        share = df_stats.groupby("layer_class")["numel"].sum()
        total_numel = float(df_stats["numel"].sum())
        for cls in ["Attention", "Linear", "Embedding", "LayerNorm"]:
            if cls in share.index:
                pct = 100.0 * float(share[cls]) / total_numel
                insights.append(f"**{cls}** holds ~{pct:.1f}% of parameters.")
        # Heavy Linear share → suggestions
        if "Linear" in share.index and (share["Linear"] / total_numel) > 0.6:
            recommendations.append("Linear/MLP is parameter‑heavy—apply low‑rank adapters or structured pruning on MLP layers.")

    # Complexity/memory interpretations
    if flops is not None:
        if flops['total_flops'] > 1e11:
            insights.append(f"Compute cost is **high** (~{flops['total_flops']:,} FLOPs per forward) → CPU inference may be slow.")
            recommendations.append("Use mixed precision (fp16/bf16) and reduce sequence length where possible.")
        else:
            insights.append(f"Compute cost is **manageable** (~{flops['total_flops']:,} FLOPs per forward).")
    if act_mem is not None:
        if act_mem > 2e9:
            insights.append(f"Activation memory is **large** (~{act_mem/1e6:.1f} MB) → could hit RAM limits.")
            recommendations.append("For training, use gradient checkpointing; for inference, reduce batch size/sequence length.")

# Display insights
if insights:
    st.markdown("### Key Insights")
    for i in insights:
        st.write("✅", i)

if recommendations:
    st.markdown("### Suggested Improvements")
    for r in recommendations:
        st.write("➡️", r)

if warnings:
    st.markdown("### Warnings / Potential Issues")
    for w in warnings:
        st.write("⚠️", w)

# ---------- OPTIONAL: TEXT INTERACTION ----------
st.subheader("🗣️ Optional: Minimal Text Interaction")
st.caption("Provide TorchScript or a safe Model class to run a synthetic forward with token IDs.")
run_interaction = st.checkbox("Attempt minimal interaction (requires TorchScript or known Model class)", value=False)

model_class_code = st.text_area(
    "Paste your Model class (Python) ONLY if you want to instantiate and run a forward.",
    placeholder="class MyTransformer(torch.nn.Module):\n    ...",
    help="WARNING: Executing arbitrary code is risky. Prefer TorchScript for safe inference."
)

if run_interaction:
    if artifact_type == "torchscript" and jit_module is not None:
        st.info("Attempting TorchScript forward on synthetic input (int64 token IDs).")
        try:
            dummy = torch.randint(0, 10000, (B, L), dtype=torch.long, device=_device())
            start = time.time()
            out = jit_module(dummy)
            dur = time.time() - start
            if isinstance(out, torch.Tensor):
                st.write("Output tensor shape:", list(out.shape))
                st.write("dtype:", str(out.dtype))
            elif isinstance(out, (tuple, list)):
                shapes = [list(t.shape) for t in out if isinstance(t, torch.Tensor)]
                st.write("Tuple/list output shapes:", shapes)
            else:
                st.write("Output (type):", type(out).__name__)
            st.success(f"Forward OK in {dur:.3f}s (CPU).")
        except Exception as e:
            st.error(f"TorchScript forward failed: {e}")

    elif state_dict is not None and model_class_code.strip():
        st.warning("You opted in to execute a custom Model class. Proceed with caution.")
        try:
            ns: Dict[str, Any] = {}
            exec(model_class_code, {"torch": torch, "nn": torch.nn}, ns)
            # Find a Module subclass
            cls = None
            for k, v in ns.items():
                if isinstance(v, type) and issubclass(v, torch.nn.Module):
                    cls = v; break
            if cls is None:
                st.error("Could not find a torch.nn.Module subclass in the provided code.")
            else:
                model = cls()  # User must define defaults consistent with state_dict
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                st.write("Missing keys:", missing)
                st.write("Unexpected keys:", unexpected)
                model.eval()
                with torch.no_grad():
                    dummy = torch.randint(0, 10000, (B, L), dtype=torch.long, device=_device())
                    start = time.time()
                    out = model(dummy)
                    dur = time.time() - start
                if isinstance(out, torch.Tensor):
                    st.write("Output tensor shape:", list(out.shape))
                else:
                    st.write("Output (type):", type(out).__name__)
                st.success(f"Forward OK in {dur:.3f}s (CPU).")
        except Exception as e:
            st.error(f"Custom class execution failed: {e}")
    else:
        st.info("Provide TorchScript or a safe Model class to run interaction.")

# ---------- REVERSE ENGINEERING AID ----------
st.subheader("🛠 Reverse Engineering Aid")

arch_type: Optional[str] = None
arch_rationale: Optional[str] = None
training_hints: Dict[str, Any] = {}

if df_stats is not None:
    names = df_stats["name"].tolist()
    arch_type, arch_rationale = detect_architecture(state_dict if state_dict is not None else {}, names)
    st.write(f"**Likely architecture type:** {arch_type}")
    with st.expander("Why this guess?"):
        st.caption(arch_rationale)

    # Training hints (if raw checkpoint container exists)
    training_hints = extract_training_hints(checkpoint_container)
    if training_hints.get("has_state", False):
        st.markdown("**Optimizer / Training artifacts detected** (from container):")
        st.json(training_hints)

    # Generate skeleton
    if df_stats is not None and (d_model is not None) and (n_layers is not None):
        skeleton = generate_skeleton(
            arch_type=arch_type,
            cfg=cfg_guess,
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            mlp_ratio=float(mlp_ratio)
        )
        st.markdown("### Reconstruction Blueprint (editable)")
        st.code(skeleton, language="python")
        st.download_button(
            "⬇️ Download reconstruction_blueprint.py",
            data=skeleton,
            file_name="reconstruction_blueprint.py"
        )
        st.download_button(
            "⬇️ Download config_guess.json",
            data=json.dumps({
                "vocab_size": int(cfg_guess.get("vocab_size", 30522)),
                "d_model": int(d_model),
                "num_layers": int(n_layers),
                "num_heads": int(n_heads),
                "mlp_ratio": float(mlp_ratio),
                "max_position_embeddings": int(cfg_guess.get("max_position_embeddings", 2048))
            }, indent=2),
            file_name="config_guess.json"
        )
    else:
        st.info("Insufficient info to build a skeleton (need d_model and num_layers). Set overrides in the sidebar if unknown.")

# ---------- EXPORTS ----------
st.subheader("🧾 Export Analysis Report")
if (uploaded is not None) and (artifact_type in ("torchscript", "state_dict", "state_dict(checkpoint)")) and df_stats is not None:
    report = {
        "artifact_type": artifact_type,
        "load_messages": status_msgs,
        "complexity": flops if flops is not None else None,
        "activation_memory_bytes": act_mem if act_mem is not None else None,
        "config_guess": cfg_guess if cfg_guess else None,
        "total_params": int(total_params) if total_params is not None else None,
        "dtype_choice_for_mem_est": dtype_choice,
        "insights": insights,
        "recommendations": recommendations,
        "warnings": warnings,
        "inferred_architecture": arch_type,
        "arch_rationale": arch_rationale,
        "training_hints": training_hints,
    }
    md_report = build_markdown_report(
        artifact_type=artifact_type,
        total_params=total_params,
        total_bytes=bytes_for_params(df_stats) if df_stats is not None else None,
        dtype_choice=dtype_choice,
        cfg_guess=cfg_guess,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=int(n_heads),
        mlp_ratio=float(mlp_ratio),
        flops=flops,
        act_mem=act_mem,
        insights=insights,
        recommendations=recommendations,
        warnings=warnings,
        arch_type=arch_type,
        arch_rationale=arch_rationale,
        training_hints=training_hints
    )

    st.download_button(
        "⬇️ Save report.json",
        data=json.dumps(report, indent=2),
        file_name="model_report.json"
    )
    st.download_button(
        "⬇️ Save report.md",
        data=md_report,
        file_name="model_report.md"
    )

st.write("---")
st.caption(
    "Notes: This is a heuristic aid. Skeletons use stock PyTorch Transformer blocks as a starting point—"
    "adapt attention masking, residual placement, norms (Pre/Post/RMS), and heads to your exact architecture."
)
