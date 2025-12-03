import streamlit as st
from e79_codegen import transpile_to_e79
from e79_interpreter import compile_e79, run_e79_function


st.set_page_config(page_title="E79 Rule Engine IDE", layout="wide")
st.title("⚡ E79 Rule Engine IDE — Python → E79 → AST-compiled Rules")


tab1, tab2, tab3 = st.tabs(
    ["Python → E79 (Rules)", "E79 Editor", "Compile & Run Rule"]
)

# ======================================================
# TAB 1 — Python → E79 (Rules)
# ======================================================
with tab1:
    st.header("Define Rules as Python Functions")

    default_py = """def risk_rule(score, exposure):
    high = score > 0.8 and exposure > 1_000_000
    med = score > 0.5 and exposure > 500_000
    label = "LOW"
    label = "MEDIUM" if med else label
    label = "HIGH" if high else label
    return label
"""

    py_code = st.text_area(
        "Python rule function(s):",
        st.session_state.get("py_rules", default_py),
        height=260,
    )
    st.session_state["py_rules"] = py_code

    if st.button("Transpile to E79"):
        try:
            e79_code = transpile_to_e79(py_code)
            st.session_state["e79"] = e79_code
            st.session_state["compiled_rules"] = None
            st.success("Transpiled to E79:")
            st.code(e79_code, language="text")
        except Exception as e:
            st.error(f"Transpilation error: {e}")

# ======================================================
# TAB 2 — E79 Editor
# ======================================================
with tab2:
    st.header("Edit E79 Rule Code")

    default_e79 = st.session_state.get(
        "e79",
        """fn risk_rule(score: auto, exposure: auto) -> auto:
    high = score > 0.8 and exposure > 1000000
    med = score > 0.5 and exposure > 500000
    label = "LOW"
    label = "MEDIUM" if med else label
    label = "HIGH" if high else label
    return label
""",
    )

    e79_code = st.text_area(
        "E79 Code:",
        default_e79,
        height=300,
        key="e79_editor",
    )
    st.session_state["e79"] = e79_code

# ======================================================
# TAB 3 — Compile & Run Rule
# ======================================================
with tab3:
    st.header("Compile E79 Rules and Run on a Context")

    e79_src = st.session_state.get("e79", "")
    if not e79_src.strip():
        st.info("No E79 code available yet. Use the first tab to generate it.")
    else:
        if st.button("Compile E79 Rules"):
            try:
                compiled = compile_e79(e79_src)
                st.session_state["compiled_rules"] = compiled
                st.success(
                    "Rules compiled. Available functions: "
                    + ", ".join(compiled.keys())
                )
            except Exception as e:
                st.error(f"Compile error: {e}")

        compiled = st.session_state.get("compiled_rules")

        rule_name = st.text_input("Rule / function name to run:", "risk_rule")

        ctx_default = "score=0.9, exposure=2000000"
        ctx_str = st.text_input(
            "Context (key=value, comma separated):",
            ctx_default,
        )

        if st.button("Run Rule"):
            if compiled is None:
                # try compiling on the fly
                try:
                    compiled = compile_e79(e79_src)
                    st.session_state["compiled_rules"] = compiled
                except Exception as e:
                    st.error(f"Compile error: {e}")
                    compiled = None

            if compiled is not None:
                try:
                    context = {}
                    if ctx_str.strip():
                        parts = ctx_str.split(",")
                        for p in parts:
                            if "=" not in p:
                                continue
                            k, v = p.split("=", 1)
                            key = k.strip()
                            val_str = v.strip()
                            # interpret values as Python literals
                            val = eval(val_str)
                            context[key] = val

                    result = run_e79_function(compiled, rule_name, **context)
                    st.success(f"Rule result: {result}")

                except Exception as e:
                    st.error(f"Runtime error: {e}")
