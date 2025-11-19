# -------------------- PART 1 --------------------
import streamlit as st
import os
from pathlib import Path
import shutil
import time
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Try ML dependencies
try:
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    from joblib import dump, load
    SKLEARN_OK = True
except:
    SKLEARN_OK = False

# -------------------------------------------------------------
# Directories for Optimizer State
# -------------------------------------------------------------
APP_DIR = Path.home() / ".desktop_optimizer"
CHECKPOINTS_DIR = APP_DIR / "checkpoints"
MODELS_DIR = APP_DIR / "models"
LOGS_DIR = APP_DIR / "logs"
QUARANTINE_DIR = APP_DIR / "quarantine"

for d in (APP_DIR, CHECKPOINTS_DIR, MODELS_DIR, LOGS_DIR, QUARANTINE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "online_regressor.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_CSV = LOGS_DIR / "metrics.csv"

# -------------------------------------------------------------
# Safety
# -------------------------------------------------------------
CRITICAL_PATHS = [
    "c:\\windows",
    "c:\\program files",
    "/usr",
    "/etc",
    "/bin",
    "/sbin",
    "/system",
    "/library",
    "/boot",
    "/dev",
    "/proc",
]

def is_safe_path(path: str) -> bool:
    p = str(Path(path).resolve()).lower()
    return not any(c in p for c in CRITICAL_PATHS)

# -------------------------------------------------------------
# Optimized Auto-Discovery with Depth Limit + Cache
# -------------------------------------------------------------
COMMON_TEMP_LOCATIONS = [
    str(Path("/tmp")) if os.name != "nt" else None,
    str(Path.home() / "AppData/Local/Temp") if os.name == "nt" else None,
    str(Path.home() / ".cache"),
]

@st.cache_data(ttl=600)
def discover_candidate_paths(min_files=10, min_mb=5):
    paths = set()

    for p in COMMON_TEMP_LOCATIONS:
        if p:
            paths.add(p)

    for child in Path.home().iterdir():
        if child.is_dir():
            paths.add(str(child))

    results = []
    progress = st.progress(0)
    total = len(paths)

    def scan_dir(pth):
        p = Path(pth)
        if not p.exists() or not p.is_dir() or not is_safe_path(pth):
            return None

        total_files = 0
        total_size = 0
        ages = []
        now = time.time()

        for root, dirs, files in os.walk(p):
            depth = root.count(os.sep) - str(p).count(os.sep)
            if depth > 2:  # Limit depth
                dirs[:] = []
                continue

            for f in files:
                fp = Path(root) / f
                try:
                    stinfo = fp.stat()
                    total_files += 1
                    total_size += stinfo.st_size
                    ages.append((now - stinfo.st_mtime) / 86400)
                except:
                    pass

        if total_files < min_files and total_size / (1024**2) < min_mb:
            return None

        return {
            "path": pth,
            "num_files": total_files,
            "total_size_mb": total_size / (1024**2),
            "avg_file_age_days": np.mean(ages) if ages else 0,
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = []
        for idx, p in enumerate(paths):
            futures.append(ex.submit(scan_dir, p))
            progress.progress((idx + 1) / total)

        for f in futures:
            r = f.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["total_size_mb"], reverse=True)
    return results

# -------------------------------------------------------------
# Checkpoint Snapshot System
# -------------------------------------------------------------
def snapshot_folder(path):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = {"timestamp": now, "path": path, "items": []}
    total_files = 0
    total_size = 0

    for root, dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                stinfo = fp.stat()
                total_files += 1
                total_size += stinfo.st_size
                if len(snap["items"]) < 3000:
                    snap["items"].append({
                        "relpath": str(fp.relative_to(path)),
                        "size": stinfo.st_size,
                        "mtime": stinfo.st_mtime,
                    })
            except:
                pass

    snap["summary"] = {
        "total_files": total_files,
        "total_size_mb": total_size / (1024**2),
    }

    out = CHECKPOINTS_DIR / f"snap_{Path(path).name}_{now}.json"
    with open(out, "w") as fh:
        json.dump(snap, fh, indent=2)

    return snap

# -------------------------------------------------------------
# Online Learning Model
# -------------------------------------------------------------
class OnlineRegressor:
    def __init__(self):
        self.feature_names = [
            "total_size_mb",
            "num_files",
            "avg_file_age_days",
            "disk_free_gb",
        ]

        if SKLEARN_OK and MODEL_PATH.exists() and SCALER_PATH.exists():
            self.model = load(MODEL_PATH)
            self.scaler = load(SCALER_PATH)
            self.initialized = True
        elif SKLEARN_OK:
            self.model = SGDRegressor(max_iter=2000)
            self.scaler = StandardScaler()
            X0 = np.array([[1,1,1,100]])
            y0 = np.array([0])
            self.scaler.fit(X0)
            self.model.partial_fit(self.scaler.transform(X0), y0)
            self.initialized = True
        else:
            self.model = {"w": np.zeros(4), "b": 0.0, "lr": 1e-4}
            self.scaler = {"mean": np.zeros(4), "std": np.ones(4)}
            self.initialized = True

    def predict(self, f):
        X = np.array([[f[k] for k in self.feature_names]])
        if SKLEARN_OK:
            return float(self.model.predict(self.scaler.transform(X))[0])
        else:
            Xs = (X - self.scaler["mean"]) / (self.scaler["std"]+1e-9)
            return float(np.dot(self.model["w"], Xs[0]) + self.model["b"])

    def update(self, f, actual):
        X = np.array([[f[k] for k in self.feature_names]])
        y = np.array([actual])

        if SKLEARN_OK:
            try:
                self.model.partial_fit(self.scaler.transform(X), y)
                dump(self.model, MODEL_PATH)
                dump(self.scaler, SCALER_PATH)
            except:
                pass
            return

        Xs = (X[0] - self.scaler["mean"]) / (self.scaler["std"] + 1e-9)
        pred = float(np.dot(self.model["w"], Xs) + self.model["b"])
        err = pred - actual
        self.model["w"] -= self.model["lr"] * err * Xs
        self.model["b"] -= self.model["lr"] * err

REGRESSOR = OnlineRegressor()

# -------------------- PART 2 --------------------
def move_to_quarantine(src):
    qrun = QUARANTINE_DIR / f"run_{int(time.time())}"
    qrun.mkdir(exist_ok=True)
    dst = qrun / Path(src).name
    try:
        shutil.move(src, dst)
        return True, str(dst)
    except:
        try:
            shutil.copy2(src, dst)
            os.remove(src)
            return True, str(dst)
        except Exception as e:
            return False, str(e)

def safe_select_files(path):
    out = []
    now = time.time()
    for root, dirs, files in os.walk(path):
        depth = root.count(os.sep) - str(path).count(os.sep)
        if depth > 2:  # Limit depth for file selection
            dirs[:] = []
            continue
        for f in files:
            fp = os.path.join(root, f)
            try:
                stinfo = os.stat(fp)
                age_days = (now - stinfo.st_mtime)/86400
                if age_days >= 5 or stinfo.st_size >= (1*1024*1024):
                    out.append(fp)
            except:
                pass
    return out

def log_metrics(record: dict):
    if Path(METRICS_CSV).exists():
        try:
            df = pd.read_csv(METRICS_CSV)
        except:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(METRICS_CSV, index=False)

# Streamlit UI
st.set_page_config(page_title="AI Holistic Optimizer", layout="wide")
st.title("🧠 Auto-Discovery Holistic Desktop Optimizer")

tab1, tab2, tab3 = st.tabs(["Discovery", "Quarantine", "Metrics"])

if "scan_triggered" not in st.session_state:
    st.session_state.scan_triggered = False

# -------------------- DISCOVERY TAB --------------------
with tab1:
    st.sidebar.header("Discovery Settings")
    min_files = st.sidebar.number_input("Minimum file count", 1, 999999, 10)
    min_mb = st.sidebar.number_input("Minimum size (MB)", 1, 999999, 10)
    mode = st.sidebar.selectbox("Mode", [
        "Recommend Only",
        "Auto-Optimize (Quarantine)",
        "Auto-Purge (Permanent)"
    ])
    workers = st.sidebar.slider("Parallel workers", 2, 64, 16)

    if st.sidebar.button("Discover & Run"):
        st.session_state.scan_triggered = True

    if st.session_state.scan_triggered:
        with st.spinner("Scanning directories..."):
            candidates = discover_candidate_paths(min_files=min_files, min_mb=min_mb)

        if not candidates:
            st.warning("No candidates found.")
            st.session_state.scan_triggered = False
            st.stop()

        st.success(f"Found {len(candidates)} candidates.")
        st.session_state.scan_triggered = False

        disktotal, diskused, diskfree = shutil.disk_usage(str(Path.home()))
        diskfree_gb = diskfree / (1024**3)

        rows = []
        for c in candidates:
            feat = {
                "total_size_mb": c["total_size_mb"],
                "num_files": c["num_files"],
                "avg_file_age_days": c["avg_file_age_days"],
                "disk_free_gb": diskfree_gb,
            }
            pred = REGRESSOR.predict(feat)
            rows.append({
                "path": c["path"],
                "files": c["num_files"],
                "size_mb": round(c["total_size_mb"],2),
                "avg_age_days": round(c["avg_file_age_days"],2),
                "predicted_reclaim_mb": round(pred,2)
            })

        df = pd.DataFrame(rows)
        st.dataframe(df)

        if mode != "Recommend Only":
            st.subheader("⚙ Running Optimization")
            for item in rows:
                path = item["path"]
                st.write(f"### Processing {path}")
                snap = snapshot_folder(path)
                files_to_move = safe_select_files(path)
                st.write(f"{len(files_to_move)} items selected")

                moved_bytes = 0
                moved_count = 0

                def worker(fp):
                    ok, dst = move_to_quarantine(fp)
                    if ok:
                        size = Path(dst).stat().st_size if Path(dst).exists() else 0
                        return True, size
                    return False, 0

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(worker, fp) for fp in files_to_move]
                    for f in futures:
                        success, size = f.result()
                        if success:
                            moved_count += 1
                            moved_bytes += size

                reclaimed_mb = moved_bytes / (1024**2)
                st.write(f"**Moved {moved_count} files → {reclaimed_mb:.2f} MB reclaimed**")

                # Log metrics
                log_metrics({
                    "timestamp": datetime.now().isoformat(),
                    "path": path,
                    "action": mode,
                    "pre_total_mb": snap["summary"]["total_size_mb"],
                    "reclaimed_mb": reclaimed_mb,
                    "moved_count": moved_count
                })

                # Update ML model
                feat = {
                    "total_size_mb": snap["summary"]["total_size_mb"],
                    "num_files": snap["summary"]["total_files"],
                    "avg_file_age_days": item["avg_age_days"],
                    "disk_free_gb": diskfree_gb,
                }
                REGRESSOR.update(feat, reclaimed_mb)

                if mode == "Auto-Purge (Permanent)":
                    shutil.rmtree(QUARANTINE_DIR, ignore_errors=True)
                    QUARANTINE_DIR.mkdir(exist_ok=True)
                    st.write("🗑 Purged permanently.")

            st.success("🎉 Optimization Completed!")

# -------------------- QUARANTINE TAB --------------------
with tab2:
    st.subheader("Quarantine Management")
    runs = sorted([p for p in QUARANTINE_DIR.iterdir() if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        st.info("Quarantine is empty.")
    else:
        for r in runs:
            with st.expander(f"Run: {r.name} — {datetime.fromtimestamp(r.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"):
                preview = []
                size = 0
                count = 0
                for root, dirs, files in os.walk(r):
                    for f in files[:50]:
                        fp = Path(root) / f
                        preview.append(str(fp.relative_to(r)))
                        size += fp.stat().st_size
                        count += 1
                st.write(f"{count} items, total {size/1024**2:.2f} MB")
                st.write(preview[:50])

                col1, col2 = st.columns([1, 1])
                if col1.button(f"Restore {r.name}"):
                    restore_dir = Path.home() / "restored_from_optimizer"
                    restore_dir.mkdir(exist_ok=True)
                    restored = 0
                    for root, dirs, files in os.walk(r):
                        for f in files:
                            src = Path(root) / f
                            dst = restore_dir / f
                            try:
                                shutil.move(str(src), str(dst))
                                restored += 1
                            except Exception as e:
                                st.warning(f"Failed to restore {src}: {e}")
                    st.success(f"Restored {restored} files to {restore_dir}")

                if col2.button(f"Delete {r.name}"):
                    shutil.rmtree(r, ignore_errors=True)
                    st.success(f"Deleted {r.name}")

# -------------------- METRICS TAB --------------------
with tab3:
    st.subheader("Metrics & Visualization")
    if Path(METRICS_CSV).exists():
        dfm = pd.read_csv(METRICS_CSV)
        st.write("Recent metrics (latest 20 rows):")
        st.dataframe(dfm.tail(20))

        if not dfm.empty:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(pd.to_datetime(dfm['timestamp']), dfm['reclaimed_mb'].astype(float), marker='o')
                ax.set_xlabel("Timestamp")
                ax.set_ylabel("Reclaimed MB")
                ax.set_title("Reclaimed Space Over Time")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Plotting failed: {e}")
    else:
        st.info("No metrics logged yet. Run optimization to generate data.")
