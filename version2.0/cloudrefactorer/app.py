
#!/usr/bin/env python
# app.py
#
# Streamlit + LangGraph + AWS Bedrock (Claude 3 Haiku)
# Monolith → Cloud Factory Refactor Planner (AWS-native edition)
# Advanced Multi-Cloud Deployment Stub Generator (AWS/GCP/Azure)

import os
import json
import ast
import boto3
from typing import TypedDict, Dict, Any, Optional, List

import streamlit as st
from langgraph.graph import StateGraph, END

# ==============================
# 1. LangGraph State
# ==============================

class RefactorState(TypedDict, total=False):
    raw_code: str
    parsed_code: Dict[str, Any]
    domain_model: Dict[str, Any]
    bedrock_response_raw: str
    structured_plan: Dict[str, Any]
    cloud_factory_mapping: Dict[str, Any]
    deployment_blueprint: str
    error: Optional[str]


# ==============================
# 2. Utility Functions
# ==============================

def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {"raw_text": text}


def parse_python_code(source: str) -> Dict[str, Any]:
    result = {
        "num_lines": len(source.splitlines()),
        "functions": [],
        "classes": [],
        "imports": [],
        "probable_endpoints": [],
        "db_keywords": [],
        "http_keywords": [],
    }

    try:
        tree = ast.parse(source)
    except SyntaxError:
        lower = source.lower()
        result["db_keywords"] = [k for k in ["select","insert","update","delete","join"] if k in lower]
        result["http_keywords"] = [k for k in ["flask","django","fastapi","route"] if k in lower]
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            result["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            result["classes"].append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            result["imports"].append(module)

    lower = source.lower()
    for marker in ["@app.route", "/api/", "router.", "flask", "fastapi"]:
        if marker in lower:
            result["probable_endpoints"].append(marker)

    result["db_keywords"] = [k for k in ["select","insert","update","delete","join"] if k in lower]
    result["http_keywords"] = [k for k in ["flask","django","fastapi","request","response"] if k in lower]

    return result


def infer_domain(parsed_code: Dict[str, Any]) -> Dict[str, Any]:
    funcs = parsed_code.get("functions", [])
    api_funcs = [f for f in funcs if "route" in f.lower() or "api" in f.lower()]
    data_funcs = [f for f in funcs if "db" in f.lower() or "repo" in f.lower()]
    core_funcs = [f for f in funcs if f not in api_funcs + data_funcs]

    domains = []
    if data_funcs:
        domains.append({"name": "data-layer", "functions": data_funcs})
    if api_funcs:
        domains.append({"name": "api-layer", "functions": api_funcs})
    if core_funcs:
        domains.append({"name": "core-domain", "functions": core_funcs})

    return {
        "candidate_domains": domains,
        "notes": "Heuristic grouping; will be refined by Claude Haiku."
    }


# ==============================
# 3. Bedrock Helper
# ==============================

def get_bedrock_client():
    try:
        session = boto3.Session(
            aws_access_key_id=st.session_state.get("aws_access_key_id"),
            aws_secret_access_key=st.session_state.get("aws_secret_access_key"),
            aws_session_token=None,
            region_name=st.session_state.get("aws_region", "us-east-1"),
        )
        client = session.client("bedrock-runtime")
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Bedrock client: {e}")


# ==============================
# 4. LangGraph Node Logic
# ==============================

def node_parse(state: RefactorState) -> RefactorState:
    code = state.get("raw_code", "")
    if not code:
        return {**state, "error": "No code provided."}
    parsed = parse_python_code(code)
    return {**state, "parsed_code": parsed}


def node_domain(state: RefactorState) -> RefactorState:
    parsed = state.get("parsed_code")
    if not parsed:
        return {**state, "error": "Missing parsed code."}
    domains = infer_domain(parsed)
    return {**state, "domain_model": domains}


def node_bedrock_haiku(state: RefactorState) -> RefactorState:
    code = state.get("raw_code", "")
    parsed = state.get("parsed_code", {})
    domain = state.get("domain_model", {})

    client = get_bedrock_client()
    snippet = code[:8000]

    prompt = f"""
You are an elite cloud modernization expert.
Generate a structured JSON-only response that outlines a complete
migration plan for converting a monolithic application into a 
Cloud-Factory-ready AWS-native architecture.

STRICT JSON RESPONSE with keys:
- current_diagnostic
- target_architecture
- phased_roadmap (array of {{phase, objective, steps[]}})
- readiness_scores
- cloud_factory_mapping
- deployment_blueprint

--- MONOLITH CODE SNIPPET ---
{snippet}

--- PARSED CODE ---
{json.dumps(parsed,indent=2)}

--- DOMAIN MODEL ---
{json.dumps(domain,indent=2)}
"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        raw = json.loads(response["body"].read().decode("utf-8"))
        text = raw["content"][0]["text"]
    except Exception as e:
        return {**state, "error": f"Bedrock request failure: {e}"}

    structured = safe_json_loads(text)
    return {**state, "bedrock_response_raw": text, "structured_plan": structured}


def node_cloud_factory(state: RefactorState) -> RefactorState:
    structured = state.get("structured_plan", {})
    cf = structured.get("cloud_factory_mapping", {"services": []})
    return {**state, "cloud_factory_mapping": cf}


def node_blueprint(state: RefactorState) -> RefactorState:
    plan = state.get("structured_plan", {})
    cf = state.get("cloud_factory_mapping", {})
    phased = plan.get("phased_roadmap", [])
    target_arch = plan.get("target_architecture", "")
    readiness = plan.get("readiness_scores", {})

    lines = ["# AWS Cloud Factory Deployment Blueprint\n\n", "## Target Architecture\n"]
    lines.append(json.dumps(target_arch, indent=2) + "\n\n" if isinstance(target_arch, (dict, list)) else str(target_arch) + "\n\n")

    lines.append("## Phased Roadmap\n")
    if isinstance(phased, list):
        for p in phased:
            lines.append(f"### {p.get('phase','Unnamed Phase')}")
            obj = p.get("objective","")
            lines.append(json.dumps(obj, indent=2) if isinstance(obj,(dict,list)) else f"**Objective:** {obj}")
            for s in p.get("steps", []):
                lines.append(f"- {s}")
            lines.append("")
    else:
        lines.append(json.dumps(phased, indent=2))

    lines.append("## Cloud Factory Services\n")
    for svc in cf.get("services", []):
        lines.append(f"### {svc.get('name','Unnamed Service')}")
        lines.append(f"- compute: {svc.get('recommended_compute','N/A')}")
        lines.append(f"- deps: {svc.get('dependencies','N/A')}")
        lines.append("")

    lines.append("## Readiness Scores\n")
    if isinstance(readiness, dict):
        for k,v in readiness.items():
            if isinstance(v, dict):
                lines.append(f"- {k}: {v.get('score','N/A')} / 100")
                notes = v.get("notes","")
                lines.append(json.dumps(notes, indent=2) if isinstance(notes,(dict,list)) else f"  - {notes}")
            else:
                lines.append(f"- {k}: {v}")
            lines.append("")
    else:
        lines.append(json.dumps(readiness, indent=2))

    return {**state, "deployment_blueprint": "\n".join(lines)}


# ==============================
# 5. Advanced Multi-Cloud Stub Generator
# ==============================

def generate_stub(plan: Dict[str, Any], provider: str) -> str:
    cf = plan.get("cloud_factory_mapping", {})
    services = cf.get("services", [])

    if provider == "AWS":
        lines = [
            "AWSTemplateFormatVersion: '2010-09-09'",
            "Description: Advanced AWS Deployment Template",
            "Resources:",
            "  VPC:",
            "    Type: AWS::EC2::VPC",
            "    Properties:",
            "      CidrBlock: 10.0.0.0/16",
            "  PublicSubnet:",
            "    Type: AWS::EC2::Subnet",
            "    Properties:",
            "      VpcId: !Ref VPC",
            "      CidrBlock: 10.0.1.0/24",
            "  SecurityGroup:",
            "    Type: AWS::EC2::SecurityGroup",
            "    Properties:",
            "      GroupDescription: Allow HTTP/HTTPS",
            "      VpcId: !Ref VPC"
        ]
        for svc in services:
            name = svc.get("name","AppInstance").replace(" ","")
            compute = svc.get("recommended_compute","t3.micro")
            lines.append(f"  {name}:")
            lines.append("    Type: AWS::EC2::Instance")
            lines.append("    Properties:")
            lines.append(f"      InstanceType: {compute}")
            lines.append("      ImageId: ami-xxxxxxxx")
            lines.append("      SubnetId: !Ref PublicSubnet")
            lines.append("      SecurityGroupIds: [!Ref SecurityGroup]")
        lines.append("Outputs:")
        lines.append("  VPCId:")
        lines.append("    Value: !Ref VPC")
        return "\n".join(lines)

    elif provider == "GCP":
        lines = [
            "resources:",
            "- name: vpc-network",
            "  type: compute.v1.network",
            "  properties:",
            "    autoCreateSubnetworks: true"
        ]
        for svc in services:
            name = svc.get("name","app-instance").replace(" ","-")
            compute = svc.get("recommended_compute","n1-standard-1")
            lines.append(f"- name: {name}")
            lines.append("  type: compute.v1.instance")
            lines.append("  properties:")
            lines.append(f"    machineType: zones/us-central1-a/machineTypes/{compute}")
            lines.append("    disks:")
            lines.append("    - boot: true")
            lines.append("      autoDelete: true")
            lines.append("      initializeParams:")
            lines.append("        sourceImage: projects/debian-cloud/global/images/family/debian-11")
            lines.append("    networkInterfaces:")
            lines.append("    - network: $(ref.vpc-network.selfLink)")
        return "\n".join(lines)

    elif provider == "Azure":
        lines = [
            "{",
            '  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",',
            '  "contentVersion": "1.0.0.0",',
            '  "resources": [',
            '    { "type": "Microsoft.Network/virtualNetworks", "name": "vnet", "properties": { "addressSpace": { "addressPrefixes": ["10.0.0.0/16"] } } },'
        ]
        for svc in services:
            name = svc.get("name","AppVM").replace(" ","")
            compute = svc.get("recommended_compute","Standard_B1s")
            lines.append("    {")
            lines.append('      "type": "Microsoft.Compute/virtualMachines",')
            lines.append(f'      "name": "{name}",')
            lines.append('      "properties": {')
            lines.append(f'        "hardwareProfile": {{"vmSize": "{compute}"}}')
            lines.append("      }")
            lines.append("    },")
        lines.append("  ]")
        lines.append("}")
        return "\n".join(lines)

    return "# Unsupported provider"


# ==============================
# 6. Build LangGraph
# ==============================

def build_graph():
    g = StateGraph(RefactorState)
    g.add_node("parse", node_parse)
    g.add_node("domain", node_domain)
    g.add_node("haiku", node_bedrock_haiku)
    g.add_node("cf_map", node_cloud_factory)
    g.add_node("blueprint", node_blueprint)
    g.set_entry_point("parse")
    g.add_edge("parse","domain")
    g.add_edge("domain","haiku")
    g.add_edge("haiku","cf_map")
    g.add_edge("cf_map","blueprint")
    g.add_edge("blueprint", END)
    return g.compile()

GRAPH = build_graph()


# ==============================
# 7. Streamlit UI
# ==============================

def main():
    st.set_page_config(page_title="Cloud Refactor Planner", layout="wide")
    st.title("🏗️ Monolith → Cloud Factory Refactor Planner (AWS Bedrock Version)")

    st.sidebar.header("🔐 AWS Credentials")
    st.session_state.setdefault("aws_access_key_id","")
    st.session_state.setdefault("aws_secret_access_key","")
    st.session_state.setdefault("aws_region","us-east-1")

    st.session_state["aws_access_key_id"] = st.sidebar.text_input("AWS Access Key ID", value=st.session_state["aws_access_key_id"])
    st.session_state["aws_secret_access_key"] = st.sidebar.text_input("AWS Secret Access Key", value=st.session_state["aws_secret_access_key"], type="password")
    st.session_state["aws_region"] = st.sidebar.text_input("AWS Region", value=st.session_state["aws_region"])
    st.sidebar.success("Credentials stored in session.")

    st.markdown("### Upload or paste your monolithic code")
    code_text = st.text_area("Monolith Code", height=300)

    if st.button("🚀 Generate Cloud Factory Refactor Plan"):
        if not code_text.strip():
            st.error("Please provide some code.")
            st.stop()

        with st.spinner("Running Bedrock Claude Haiku…"):
            state = GRAPH.invoke({"raw_code": code_text})

        if state.get("error"):
            st.error(state["error"])
            st.stop()

        tab1, tab2, tab3, tab4 = st.tabs(["Parsed", "Claude JSON", "Cloud Factory Mapping", "Blueprint"])
        with tab1:
            st.json(state.get("parsed_code"))
            st.json(state.get("domain_model"))
        with tab2:
            st.text(state.get("bedrock_response_raw"))
            st.json(state.get("structured_plan"))
        with tab3:
            st.json(state.get("cloud_factory_mapping"))
        with tab4:
            st.markdown(state.get("deployment_blueprint"))

        # Multi-Cloud Stub Download
        st.markdown("### Download Advanced Deployment Stub")
        provider = st.selectbox("Choose Cloud Provider", ["AWS","GCP","Azure"])
        stub = generate_stub(state.get("structured_plan", {}), provider)
        st.download_button(
            label=f"Download {provider} Deployment Stub",
            data=stub,
            file_name=f"{provider.lower()}_deployment_stub.yaml" if provider!="Azure" else f"{provider.lower()}_deployment_stub.json"
        )

if __name__ == "__main__":
    main()
