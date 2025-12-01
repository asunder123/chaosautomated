
import os
import re
import json
import streamlit as st
import boto3
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage

# ----------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------
st.set_page_config(page_title="☁️ Cloud Migration Planner", layout="wide")
st.title("☁️ Cloud Migration Planner (Claude via AWS Bedrock)")

# AWS Credentials Input
st.subheader("🔐 AWS Authentication")
aws_access_key = st.text_input("AWS Access Key ID", type="password")
aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
aws_region = st.text_input("AWS Region", value="us-east-1")

# Model Selection
model_choice = st.selectbox(
    "Choose Claude Model",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0"
    ]
)

# ----------------------------------------------------------
# AWS Login
# ----------------------------------------------------------
if st.button("Login to AWS"):
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        st.session_state["aws_session"] = session
        st.success("✅ AWS Authentication Successful!")
    except ClientError as e:
        st.error(f"Authentication failed: {e}")

# ----------------------------------------------------------
# Claude via LangChain Bedrock Wrapper
# ----------------------------------------------------------
def generate_plan(infra_details: str, constraints: str, model_id: str) -> str:
    try:
        bedrock_client = st.session_state["aws_session"].client("bedrock-runtime")
        llm = ChatBedrockConverse(
            client=bedrock_client,
            model=model_id,
            max_tokens=1024,
            temperature=0.2
        )

        messages = [
            SystemMessage(content="You are an expert AWS cloud migration planner."),
            HumanMessage(content=f"""
            You are a cloud migration expert. Given the infrastructure:
            {infra_details}
            and constraints:
            {constraints}

            Generate 3 optimized cloud migration scenarios for AWS with:
            - Approach (Lift-and-shift, Re-platform, Re-architect)
            - Estimated cost
            - Timeline
            - Risks
            - Compliance considerations

            Format clearly with headings for each scenario and subheadings for each category.
            """)
        ]

        response = llm.invoke(messages)
        return response.content if response else "⚠️ No response generated. Check AWS Bedrock setup."
    except Exception as e:
        return f"Error invoking model: {e}"

# ----------------------------------------------------------
# Migration Planner Logic
# ----------------------------------------------------------
st.subheader("📋 Migration Details")
infra_details = st.text_area("Enter current infrastructure details (VMs, DBs, apps):")
constraints = st.text_area("Enter constraints (budget, downtime, compliance):")

if st.button("Generate Migration Plan"):
    if "aws_session" not in st.session_state:
        st.error("Please log in to AWS first.")
    elif not infra_details or not constraints:
        st.warning("Please provide infrastructure details and constraints.")
    else:
        with st.spinner("Generating migration scenarios..."):
            raw_result = generate_plan(infra_details, constraints, model_choice)

        st.session_state["migration_plan"] = raw_result
        st.success("✅ Migration Plan Generated!")

        # ----------------------------------------------------------
        # Parsing Logic for Better Readability
        # ----------------------------------------------------------
        st.markdown("### 📑 Migration Scenarios")
        scenarios = re.split(r"\n\s*\d+\.\s*", raw_result.strip())  # Split by numbered scenarios
        for i, scenario in enumerate(scenarios):
            if scenario.strip():
                st.markdown(f"#### 🔹 Scenario {i+1}")
                # Further split by categories
                parts = re.split(r"(?i)(Approach|Estimated cost|Timeline|Risks|Compliance)", scenario)
                for j in range(1, len(parts), 2):
                    category = parts[j].strip()
                    content = parts[j+1].strip()
                    st.write(f"**{category}:** {content}")
                st.divider()

# ----------------------------------------------------------
# Download Plan
# ----------------------------------------------------------
if "migration_plan" in st.session_state:
    st.download_button(
        "Download Migration Plan",
        data=st.session_state["migration_plan"],
        file_name="migration_plan.txt"
    )
