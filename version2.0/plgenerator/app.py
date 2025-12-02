import streamlit as st
from e79_codegen import transpile_to_e79
from e79_interpreter import run_e79
import base64


# ------------------------------------------------------
# Utility: download link
# ------------------------------------------------------
def make_download(label, filename, text):
    b64 = base64.b64encode(text.encode()).decode()
    st.markdown(
        f'<a download="{filename}" href="data:text/plain;base64,{b64}">{label}</a>',
        unsafe_allow_html=True
    )


# ------------------------------------------------------
# Codegen: E79 → C (stub)
# ------------------------------------------------------
def e79_to_c(src: str):
    out = ["#include <stdio.h>", ""]
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("fn "):
            name = s[3:].split("(")[0].strip()
            out.append(f"double {name}() {{")
        elif s.startswith("return"):
            val = s.replace("return", "").strip()
            out.append(f"    return {val};\n}}")
        elif "=" in s:
            v, e = s.split("=", 1)
            out.append(f"    double {v.strip()} = {e.strip()};")
    return "\n".join(out)


# ------------------------------------------------------
# Codegen: E79 → LLVM pseudo IR
# ------------------------------------------------------
def e79_to_llvm(src: str):
    out = ["; PSEUDO LLVM IR", ""]
    for line in src.splitlines():
        s = line.strip()
        if "=" in s:
            left, right = s.split("=", 1)
            out.append(f"%{left.strip()} = add double {right.strip()}")
        if s.startswith("return"):
            val = s.replace("return", "").strip()
            out.append(f"ret double {val}")
    return "\n".join(out)


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.set_page_config(page_title="E79 IDE", layout="wide")
st.title("⚡ E79 Language IDE — Python → E79 → Run → C → LLVM")

tabs = st.tabs(
    ["Python → E79", "E79 Editor", "Run E79", "Downloads", "C & LLVM Output"]
)

# ---------------- TAB 1 ----------------
with tabs[0]:
    st.header("Transpile Python → E79")

    py = st.text_area(
        "Python code:",
        """def test(a, b):
    c = a + b
    return c
""",
        height=200,
    )

    if st.button("Convert to E79"):
        try:
            e79 = transpile_to_e79(py)
            st.session_state["e79"] = e79
            st.code(e79)
        except Exception as e:
            st.error(e)

# ---------------- TAB 2 ----------------
with tabs[1]:
    st.header("Edit E79 Code")
    code = st.session_state.get(
        "e79",
        "fn test(a: auto, b: auto) -> auto:\n    c = a + b\n    return c\n",
    )
    st.session_state["e79"] = st.text_area("E79 Code:", code, height=300)

# ---------------- TAB 3 ----------------
with tabs[2]:
    st.header("Run E79 (like Python)")

    code_to_run = st.text_area(
        "E79 code to execute:",
        st.session_state.get("e79", ""),
        height=240,
    )

    fn = st.text_input("Function:", "test")
    args = st.text_input("Arguments:", "2, 3")

    if st.button("Run E79 Code"):
        try:
            parsed = [] if not args.strip() else [eval(x) for x in args.split(",")]
            result = run_e79(code_to_run, call=fn, args=parsed)
            st.success(f"Output = {result}")
        except Exception as e:
            st.error(e)

# ---------------- TAB 4 ----------------
with tabs[3]:
    st.header("Download generated stubs")

    e79 = st.session_state.get("e79", "")
    c_code = e79_to_c(e79)
    ll = e79_to_llvm(e79)

    make_download("Download E79 file", "program.e79", e79)
    make_download("Download C stub", "program.c", c_code)
    make_download("Download LLVM IR", "program.ll", ll)

# ---------------- TAB 5 ----------------
with tabs[4]:
    st.header("C Stub Output")
    st.code(e79_to_c(st.session_state.get("e79", "")))

    st.header("LLVM IR Output")
    st.code(e79_to_llvm(st.session_state.get("e79", "")))
