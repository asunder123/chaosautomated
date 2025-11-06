import os
import json
import boto3
import streamlit as st
import graphviz
from typing import Dict, Any, Optional
from pydantic import BaseModel
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(page_title="LangGraph Chaos Executor", layout="wide")
st.title("🧠 LangGraph Chaos Executor (Claude + AWS FIS)")
st.caption("AWS Bedrock → LangGraph → Auto FIS Template → Validation → Execution → Feedback")

# ------------------------------------------------------------
# Sidebar credentials + config
# ------------------------------------------------------------
with st.sidebar:
    st.header("🔐 AWS Login")
    access_key = st.text_input("Access Key ID", type="password")
    secret_key = st.text_input("Secret Access Key", type="password")
    session_token = st.text_input("Session Token (optional)", type="password")
    region = st.text_input("AWS Region", value="us-east-1")

    st.header("🧠 Bedrock Model")
    model_id = st.text_input("Preferred Model ID", value="anthropic.claude-3-5-sonnet-20241022-v2:0")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.number_input("Max Tokens", 256, 4096, 2048, step=256)

    st.header("⚙️ Execution Mode")
    dry_run = st.toggle("🧪 Dry Run Mode (no real FIS execution)", value=True)

if not access_key or not secret_key:
    st.warning("Please enter AWS credentials to continue.")
    st.stop()

# ------------------------------------------------------------
# AWS session
# ------------------------------------------------------------
try:
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token or None,
        region_name=region,
    )
    fis = session.client("fis")
    st.sidebar.success("✅ AWS session established")
except Exception as e:
    st.sidebar.error(f"❌ AWS connection failed: {e}")
    st.stop()

# ------------------------------------------------------------
# Initialize Bedrock LLM
# ------------------------------------------------------------
def find_supported_model(preferred_id: str) -> str:
    bedrock = session.client("bedrock-runtime")
    try:
        body = json.dumps({"inputText": "ping"})
        bedrock.invoke_model(modelId=preferred_id, body=body, contentType="application/json", accept="application/json")
        return preferred_id
    except Exception:
        return "anthropic.claude-3-haiku-20240307-v1:0"

supported_model = find_supported_model(model_id)

llm = ChatBedrockConverse(
    model_id=supported_model,
    region_name=region,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    aws_session_token=session_token or None,
    temperature=temperature,
    max_tokens=max_tokens,
)

# ------------------------------------------------------------
# State schema
# ------------------------------------------------------------
class ChaosState(BaseModel):
    goal: str
    constraints: Optional[str] = None
    env: Optional[str] = None
    dry_run: bool = True
    hypothesis: Optional[str] = None
    plan: Optional[str] = None
    fis_template_raw: Optional[str] = None
    fis_template: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def ask_llm(system: str, user: str) -> str:
    msgs = [("system", system), ("human", user)]
    resp: AIMessage = llm.invoke(msgs)
    return resp.content if isinstance(resp.content, str) else str(resp)

def parse_json_block(s: str):
    s = s.strip()
    if "```" in s:
        s = s.split("```", 1)[-1].replace("json", "").strip()
        if "```" in s:
            s = s.split("```")[0]
    try:
        return json.loads(s)
    except Exception:
        return None

# ------------------------------------------------------------
# Auto-correct FIS template schema
# ------------------------------------------------------------
def correct_fis_template(original: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "description": original.get("description", "Chaos experiment for ECS resilience"),
        "roleArn": original.get("roleArn", "arn:aws:iam::123456789012:role/FISRole"),
        "actions": {
            "terminateInstances": {
                "actionId": "aws:ec2:terminate-instances",
                "parameters": {
                    "instanceIds": ",".join(original.get("actions", [{}])[0].get("parameters", {}).get("instanceIds", []))
                },
                "targets": {
                    "InstancesTarget": "InstancesInAZ"
                }
            }
        },
        "targets": {
            "InstancesInAZ": {
                "resourceType": "aws:ec2:instance",
                "filters": [
                    {
                        "path": "Placement.AvailabilityZone",
                        "values": ["us-west-2a"]
                    }
                ]
            }
        },
        "stopConditions": [
            {
                "source": "aws:cloudwatch:alarm",
                "value": "arn:aws:cloudwatch:us-west-2:123456789012:alarm:ECSFailoverAlarm"
            }
        ]
    }

# ------------------------------------------------------------
# LangGraph nodes
# ------------------------------------------------------------
def node_hypothesis(state: ChaosState) -> ChaosState:
    state.hypothesis = ask_llm("You are an AWS Chaos Engineer. Given a goal, propose a testable hypothesis.",
                               f"Goal: {state.goal}\nConstraints: {state.constraints or '(none)'}")
    return state

def node_plan(state: ChaosState) -> ChaosState:
    state.plan = ask_llm("Generate a numbered AWS chaos plan (actions, guardrails, stop conditions).",
                         state.hypothesis or "")
    return state

def node_template(state: ChaosState) -> ChaosState:
    result = ask_llm("Convert the plan into a valid AWS FIS JSON template.", state.plan or "")
    state.fis_template_raw = result
    parsed = parse_json_block(result)
    state.fis_template = parsed if parsed else None
    return state

def node_validate(state: ChaosState) -> ChaosState:
    t = state.fis_template
    if not t:
        state.validation = {"ok": False, "reason": "Invalid template"}
    else:
        missing = [k for k in ["description", "roleArn", "actions", "targets", "stopConditions"] if k not in t]
        state.validation = {"ok": not missing, "reason": ("Missing " + ", ".join(missing)) if missing else "Valid"}
    return state

def node_execute(state: ChaosState) -> ChaosState:
    if state.dry_run:
        state.execution = {"mode": "dry-run", "message": "No experiment executed."}
    else:
        try:
            resp = fis.create_experiment_template(**state.fis_template)
            state.execution = {"mode": "live", "result": resp}
        except Exception as e:
            state.execution = {"mode": "live", "error": str(e)}
    return state

def node_feedback(state: ChaosState) -> ChaosState:
    error_context = ""
    if state.validation and not state.validation.get("ok"):
        error_context = f"Validation failed: {state.validation.get('reason')}"
    elif state.execution and "error" in state.execution:
        error_context = f"Execution error: {state.execution.get('error')}"
    state.feedback = ask_llm("Analyze the chaos experiment and suggest improvements. "
                             "Consider these issues:\n" + error_context,
                             json.dumps(state.execution or {}, indent=2))
    return state

# ------------------------------------------------------------
# Build LangGraph
# ------------------------------------------------------------
graph = StateGraph(state_schema=ChaosState)
graph.add_node("hypothesis", node_hypothesis)
graph.add_node("plan", node_plan)
graph.add_node("template", node_template)
graph.add_node("validate", node_validate)
graph.add_node("execute", node_execute)
graph.add_node("feedback", node_feedback)

graph.set_entry_point("hypothesis")
graph.add_edge("hypothesis", "plan")
graph.add_edge("plan", "template")
graph.add_edge("template", "validate")
graph.add_edge("validate", "execute")
graph.add_edge("execute", "feedback")
workflow = graph.compile()

# ------------------------------------------------------------
# Dynamic DAG Renderer
# ------------------------------------------------------------
def render_dag(statuses):
    status_colors = {"completed": "green", "in_progress": "yellow", "failed": "red", "pending": "gray"}
    dag = graphviz.Digraph()
    dag.attr(rankdir='LR', size='10')
    for node, status in statuses.items():
        dag.node(node, label=node.capitalize(), style="filled", fillcolor=status_colors.get(status, "gray"))
    for src, dst in [("hypothesis", "plan"), ("plan", "template"), ("template", "validate"),
                     ("validate", "execute"), ("execute", "feedback")]:
        dag.edge(src, dst)
    st.graphviz_chart(dag)

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.subheader("⚙️ Chaos Workflow")
goal = st.text_area("Chaos Engineering Goal", placeholder="e.g. Test ECS resilience under AZ failure")
constraints = st.text_area("Constraints / Guardrails", "(none)")
env = st.text_input("Environment (optional)", "")

if st.button("🚀 Run LangGraph Workflow"):
    if not goal.strip():
        st.warning("Please enter a goal.")
    else:
        statuses = {n: "pending" for n in ["hypothesis", "plan", "template", "validate", "execute", "feedback"]}
        render_dag(statuses)
        initial = ChaosState(goal=goal.strip(), constraints=constraints.strip(), env=env.strip(), dry_run=dry_run)

        for stage in ["hypothesis", "plan", "template", "validate", "execute", "feedback"]:
            statuses[stage] = "in_progress"
            render_dag(statuses)
            st.info(f"Running stage: {stage}")
            result = workflow.invoke(initial, start_at=stage, end_at=stage)

            # Auto-correct template if validation fails
            if stage == "validate" and result.get("validation") and not result["validation"].get("ok"):
                st.warning("Validation failed. Auto-correcting template...")
                corrected = correct_fis_template(result.get("fis_template", {}))
                result["fis_template"] = corrected
                result["validation"] = {"ok": True, "reason": "Corrected schema applied"}

            # Retry execution if previous error
            if stage == "execute" and result.get("execution") and "error" in result["execution"]:
                st.warning("Execution failed. Retrying with corrected template...")
                corrected = correct_fis_template(result.get("fis_template", {}))
                try:
                    resp = fis.create_experiment_template(**corrected)
                    result["execution"] = {"mode": "live", "result": resp}
                except Exception as e:
                    result["execution"] = {"mode": "live", "error": str(e)}

            # Determine status
            if stage == "validate" and not result["validation"].get("ok"):
                statuses[stage] = "failed"
            elif stage == "execute" and "error" in result["execution"]:
                statuses[stage] = "failed"
            else:
                statuses[stage] = "completed"
            render_dag(statuses)

            # Show relevant info
            if stage == "hypothesis":
                st.markdown(f"**Hypothesis:** {result.get('hypothesis', '—')}")
            elif stage == "plan":
                st.markdown(f"**Plan:**\n{result.get('plan', '—')}")
            elif stage == "template":
                st.code(result.get('fis_template_raw', 'Template generation failed'), language="json")
            elif stage == "validate":
                st.markdown(f"**Validation:** {result.get('validation', {})}")
            elif stage == "execute":
                st.markdown(f"**Execution:** {result.get('execution', {})}")
            elif stage == "feedback":
                st.markdown(f"**Feedback:** {result.get('feedback', '—')}")

        st.success("✅ Workflow complete")