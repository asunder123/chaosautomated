
import streamlit as st
import boto3
import json
import re
import zipfile
from pathlib import Path
from langgraph.graph import StateGraph, END
from typing import TypedDict

# ============================================================
# STREAMLIT AWS LOGIN UI
# ============================================================

st.set_page_config(page_title="Python Modularizer (AWS Login)", layout="wide")
st.title("🔐 AWS Login for Bedrock Access")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "aws_session" not in st.session_state:
    st.session_state.aws_session = None
if "bedrock" not in st.session_state:
    st.session_state.bedrock = None

# ------------------------------
#  Login UI (visible until authenticated)
# ------------------------------
if not st.session_state.authenticated:

    st.subheader("Enter AWS Credentials")

    access_key = st.text_input("AWS Access Key ID", type="password")
    secret_key = st.text_input("AWS Secret Access Key", type="password")
    session_token = st.text_input("(Optional) AWS Session Token", type="password")
    region = st.text_input("AWS Region", value="us-east-1")

    if st.button("Sign In"):
        try:
            # Create AWS Session
            session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token if session_token else None,
                region_name=region
            )

            # Test Bedrock client
            bedrock = session.client("bedrock-runtime")

            # Dummy request to verify credentials
            test_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Say 'ready'"}]}]
            }

            response = bedrock.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                accept="application/json",
                contentType="application/json",
                body=json.dumps(test_payload)
            )

            # If response is good — save session and authenticate
            st.session_state.authenticated = True
            st.session_state.aws_session = session
            st.session_state.bedrock = bedrock

            st.success("AWS Authentication Successful! 🎉")

        except Exception as e:
            st.error(f"Login Failed: {e}")

    st.stop()

# ============================================================
# AUTHENTICATED — SHOW MODULARIZER UI
# ============================================================

st.success("Authenticated to AWS Bedrock ✔")
st.title("🧩 Python Code Modularizer (LangGraph + AWS Bedrock)")

bedrock = st.session_state.bedrock

# ============================================================
# BEDROCK CALL
# ============================================================

def call_bedrock(prompt: str):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        accept="application/json",
        contentType="application/json",
        body=json.dumps(payload)
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# ============================================================
# LANGGRAPH
# ============================================================

class CodeState(TypedDict):
    code: str
    imports: str
    config: str
    classes: str
    functions: str
    main: str

def extract_sections(state: CodeState):
    code = state["code"]

    prompt = f"""
Split this Python code into:
- IMPORTS
- CONFIG
- CLASSES
- FUNCTIONS
- MAIN EXECUTION BLOCK

Return ONLY valid JSON with keys:
imports, config, classes, functions, main.

Code:
{code}
"""

    try:
        output = call_bedrock(prompt)
        parsed = json.loads(output)
        state.update(parsed)
    except:
        # Regex fallback
        state["imports"] = "\n".join([l for l in code.splitlines()
                                      if l.startswith("import") or l.startswith("from")])
        state["classes"] = "\n".join(re.findall(r"class .*?:[\s\S]*?(?=\n\S|$)", code))
        state["functions"] = "\n".join(re.findall(r"def .*?:[\s\S]*?(?=\n\S|$)", code))
        state["main"] = "\n".join([l for l in code.splitlines() if "if __name__" in l])
        state["config"] = ""
    return state

def refine_sections(state: CodeState):
    for key in ["imports", "config", "classes", "functions", "main"]:
        section = state[key]
        prompt = f"""
Refine this Python {key} section:

- Make clean & modular
- Improve readability
- PEP8 compliant
- Avoid logic changes unless needed

Section:
{section}
"""
        state[key] = call_bedrock(prompt)
    return state

def build_graph():
    workflow = StateGraph(CodeState)
    workflow.add_node("extract", extract_sections)
    workflow.add_node("refine", refine_sections)
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "refine")
    workflow.add_edge("refine", END)
    return workflow.compile()

graph = build_graph()

# ============================================================
# CLEANUP LOGIC: Remove extra text, keep only Python code
# ============================================================

def cleanup_python(text: str) -> str:
    # Remove markdown fences and non-code commentary
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:python)?", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned)

    # Keep only lines that look like Python code
    lines = cleaned.splitlines()
    python_lines = []
    for line in lines:
        if (line.strip().startswith(("import", "from", "class ", "def ", "if __name__", "@")) or
            re.match(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=.*", line) or
            line.strip().startswith(("print", "return", "#")) or
            line.strip() == "" or
            line.strip().startswith(("for ", "while ", "try", "except", "with "))):
            python_lines.append(line)

    return "\n".join(python_lines).strip()

# ============================================================
# README GENERATION
# ============================================================

def generate_readme(sections: dict) -> str:
    prompt = f"""
Generate a README.md summarizing the modular structure of this Python project.
Explain each section (imports, config, classes, functions, main) briefly.
Use markdown headings and bullet points.

Sections:
{json.dumps(sections)}
"""
    return call_bedrock(prompt)

# ============================================================
# UI — FILE INPUT
# ============================================================

uploaded = st.file_uploader("Upload Python File", type=["py"])

if uploaded:
    code = uploaded.read().decode("utf-8")
    st.subheader("📄 Original Code")
    st.code(code, language="python")

    if st.button("Modularize Code"):
        with st.spinner("Processing via Claude 3 Haiku (Bedrock)…"):
            result = graph.invoke({"code": code})

        st.success("Modularization Complete ✔")

        tabs = st.tabs(["Imports", "Config", "Classes", "Functions", "Main"])
        keys = ["imports", "config", "classes", "functions", "main"]

        cleaned_sections = {}
        for tab, key in zip(tabs, keys):
            with tab:
                st.subheader(key.title())
                cleaned = cleanup_python(result[key])
                cleaned_sections[key] = cleaned
                st.code(cleaned, language="python")

        # ============================================================
        # Generate README.md using Claude
        # ============================================================
        st.subheader("📖 Generating README.md...")
        readme_content = generate_readme(cleaned_sections)
        st.markdown(readme_content)

        # ============================================================
        # EXPORT MODULARIZED STUB + README AS ZIP
        # ============================================================
        st.subheader("⬇️ Export Modularized Stub")
        workdir = Path("modularized_stub")
        workdir.mkdir(exist_ok=True)

        # Create files for each section with cleanup
        file_map = {
            "imports.py": cleaned_sections["imports"],
            "config.py": cleaned_sections["config"],
            "classes.py": cleaned_sections["classes"],
            "functions.py": cleaned_sections["functions"],
            "main.py": cleaned_sections["main"],
            "README.md": readme_content
        }

        for fname, content in file_map.items():
            (workdir / fname).write_text(content)

        # Create ZIP
        zip_path = workdir / "modularized_stub.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in workdir.iterdir():
                if file.is_file() and file.name != "modularized_stub.zip":
                    zipf.write(file, file.name)

        with open(zip_path, "rb") as f:
            st.download_button("Download Modularized Stub ZIP", f, file_name="modularized_stub.zip")
