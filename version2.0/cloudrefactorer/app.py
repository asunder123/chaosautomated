
#!/usr/bin/env python
# app.py
#
# Streamlit + AWS Bedrock + GitHub Repo Generator
# Monolith → Cloud Factory Refactor Planner with CI/CD Ready Repo

import os
import json
import ast
import boto3
import zipfile
from typing import TypedDict, Dict, Any, Optional
import streamlit as st

# ==============================
# 1. State Definition
# ==============================
class RefactorState(TypedDict, total=False):
    raw_code: str
    parsed_code: Dict[str, Any]
    domain_model: Dict[str, Any]
    bedrock_response_raw: str
    structured_plan: Dict[str, Any]
    cloud_factory_mapping: Dict[str, Any]
    enriched_plan: Dict[str, Any]
    deployment_blueprint: str
    repo_zip: str
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
    result = {"num_lines": len(source.splitlines()), "functions": [], "classes": [], "imports": []}
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                result["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                result["classes"].append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                result["imports"].append(node.module or "")
    except SyntaxError:
        pass
    return result

def infer_domain(parsed_code: Dict[str, Any]) -> Dict[str, Any]:
    funcs = parsed_code.get("functions", [])
    api_funcs = [f for f in funcs if "api" in f.lower() or "route" in f.lower()]
    data_funcs = [f for f in funcs if "db" in f.lower() or "repo" in f.lower()]
    core_funcs = [f for f in funcs if f not in api_funcs + data_funcs]
    return {"candidate_domains": [
        {"name": "api-layer", "functions": api_funcs},
        {"name": "data-layer", "functions": data_funcs},
        {"name": "core-domain", "functions": core_funcs}
    ]}

def get_bedrock_client():
    session = boto3.Session(
        aws_access_key_id=st.session_state.get("aws_access_key_id"),
        aws_secret_access_key=st.session_state.get("aws_secret_access_key"),
        region_name=st.session_state.get("aws_region", "us-east-1"),
    )
    return session.client("bedrock-runtime")

# ==============================
# 3. Pipeline Nodes
# ==============================
def node_parse(state: RefactorState) -> RefactorState:
    return {**state, "parsed_code": parse_python_code(state["raw_code"])}

def node_domain(state: RefactorState) -> RefactorState:
    return {**state, "domain_model": infer_domain(state["parsed_code"])}

def node_bedrock_haiku(state: RefactorState) -> RefactorState:
    client = get_bedrock_client()
    prompt = f"""
Generate JSON migration plan for AWS-native architecture.
Keys: current_diagnostic, target_architecture, phased_roadmap, readiness_scores, cloud_factory_mapping, deployment_blueprint.
Code snippet:
{state['raw_code']}
Parsed:
{json.dumps(state['parsed_code'], indent=2)}
Domain:
{json.dumps(state['domain_model'], indent=2)}
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
        return {**state, "bedrock_response_raw": text, "structured_plan": safe_json_loads(text)}
    except Exception as e:
        return {**state, "error": str(e)}

def node_cloud_factory(state: RefactorState) -> RefactorState:
    cf = state["structured_plan"].get("cloud_factory_mapping", {"services": []})
    return {**state, "cloud_factory_mapping": cf}

def node_enrich_plan(state: RefactorState) -> RefactorState:
    plan = state["structured_plan"]
    plan["additional_resources"] = {
        "Networking": {"InternetGateway": {"Type": "AWS::EC2::InternetGateway"}},
        "Compute": {"AutoScalingGroup": {"Type": "AWS::AutoScaling::AutoScalingGroup"}},
        "Storage": {"S3Bucket": {"Type": "AWS::S3::Bucket"}},
        "IAM": {"InstanceRole": {"Type": "AWS::IAM::Role"}},
        "Monitoring": {"CloudWatchAlarm": {"Type": "AWS::CloudWatch::Alarm"}}
    }
    return {**state, "enriched_plan": plan}

def node_blueprint(state: RefactorState) -> RefactorState:
    bp = f"# Deployment Blueprint\n\n{json.dumps(state['enriched_plan'], indent=2)}"
    return {**state, "deployment_blueprint": bp}

def generate_stub(plan: Dict[str, Any]) -> str:
    lines = ["AWSTemplateFormatVersion: '2010-09-09'", "Resources:"]
    for category, resources in plan.get("additional_resources", {}).items():
        for name, details in resources.items():
            lines.append(f"  {name}:")
            lines.append(f"    Type: {details['Type']}")
    return "\n".join(lines)

def node_generate_repo(state: RefactorState) -> RefactorState:
    repo_path = "cloud-refactor-repo"
    os.makedirs(f"{repo_path}/infrastructure/aws", exist_ok=True)
    os.makedirs(f"{repo_path}/.github/workflows", exist_ok=True)
    os.makedirs(f"{repo_path}/src/monolith", exist_ok=True)
    os.makedirs(f"{repo_path}/src/refactored", exist_ok=True)
    os.makedirs(f"{repo_path}/config", exist_ok=True)

    # Save original code with UTF-8 encoding
    with open(f"{repo_path}/src/monolith/app.py", "w", encoding="utf-8") as f:
        f.write(state["raw_code"])

    # Dockerfile
    dockerfile = """FROM python:3.11-slim
WORKDIR /app
COPY src/monolith/ /app
RUN pip install -r requirements.txt || true
CMD ["python", "app.py"]
"""
    with open(f"{repo_path}/config/Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile)

    # docker-compose.yaml
    compose = """version: '3.8'
services:
  app:
    build: ./config
    ports:
      - "8080:8080"
    environment:
      - ENV=production
"""
    with open(f"{repo_path}/config/docker-compose.yaml", "w", encoding="utf-8") as f:
        f.write(compose)

    # app-config.json
    config_json = {"ENV": "production", "AWS_REGION": st.session_state.get("aws_region", "us-east-1")}
    with open(f"{repo_path}/config/app-config.json", "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    # CloudFormation template
    with open(f"{repo_path}/infrastructure/aws/cloudformation.yaml", "w", encoding="utf-8") as f:
        f.write(generate_stub(state["enriched_plan"]))

    # README
    with open(f"{repo_path}/README.md", "w", encoding="utf-8") as f:
        f.write(state["deployment_blueprint"])

    # GitHub Actions workflow
    workflow = """name: Deploy
on: [push]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Login to ECR
        run: |
          aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_REPO_URI>
      - name: Build and Push Docker Image
        run: |
          docker build -t cloud-refactor-app ./config
          docker tag cloud-refactor-app:latest <ECR_REPO_URI>:latest
          docker push <ECR_REPO_URI>:latest
      - name: Deploy CloudFormation
        run: |
          aws cloudformation deploy \
            --template-file infrastructure/aws/cloudformation.yaml \
            --stack-name refactor-stack \
            --capabilities CAPABILITY_IAM
"""
    with open(f"{repo_path}/.github/workflows/deploy.yml", "w", encoding="utf-8") as f:
        f.write(workflow)

    # Zip repo
    zip_name = "cloud-refactor-repo.zip"
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, _, files in os.walk(repo_path):
            for file in files:
                zipf.write(os.path.join(root, file))
    return {**state, "repo_zip": zip_name}

# ==============================
# 4. Streamlit UI
# ==============================
def main():
    st.set_page_config(page_title="Cloud Refactor Planner", layout="wide")
    st.title("🏗️ Monolith → Cloud Factory Refactor Planner")

    # AWS Credentials
    st.sidebar.header("AWS Credentials")
    st.session_state.setdefault("aws_access_key_id", "")
    st.session_state.setdefault("aws_secret_access_key", "")
    st.session_state.setdefault("aws_region", "us-east-1")
    st.session_state["aws_access_key_id"] = st.sidebar.text_input("Access Key", st.session_state["aws_access_key_id"])
    st.session_state["aws_secret_access_key"] = st.sidebar.text_input("Secret Key", st.session_state["aws_secret_access_key"], type="password")
    st.session_state["aws_region"] = st.sidebar.text_input("Region", st.session_state["aws_region"])

    code_text = st.text_area("Paste Monolith Code", height=300)

    if st.button("🚀 Generate Plan"):
        if not code_text.strip():
            st.error("Please provide code.")
            st.stop()

        progress = st.progress(0)
        state = {"raw_code": code_text}

        progress.progress(20, "Parsing...")
        state = node_parse(state)
        st.json(state["parsed_code"])

        progress.progress(40, "Inferring domain...")
        state = node_domain(state)
        st.json(state["domain_model"])

        progress.progress(60, "Calling Claude Haiku...")
        state = node_bedrock_haiku(state)
        st.json(state["structured_plan"])

        progress.progress(75, "Cloud Factory mapping...")
        state = node_cloud_factory(state)
        st.json(state["cloud_factory_mapping"])

        progress.progress(85, "Enriching plan...")
        state = node_enrich_plan(state)
        st.json(state["enriched_plan"])

        progress.progress(95, "Generating blueprint...")
        state = node_blueprint(state)
        st.markdown(state["deployment_blueprint"])

        progress.progress(100, "Creating GitHub repo...")
        state = node_generate_repo(state)
        st.success("✅ Repo ready!")
        with open(state["repo_zip"], "rb") as f:
            st.download_button("Download GitHub Repo ZIP", f, file_name="cloud-refactor-repo.zip")

if __name__ == "__main__":
    main()
