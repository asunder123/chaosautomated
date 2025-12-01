
import json
import re
import streamlit as st
import boto3
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage

# ----------------------------------------------------------
# STREAMLIT UI CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="☁️ Cloud Migration Planner", layout="wide")
st.markdown("<h1 style='text-align:center;'>☁️ Cloud Migration Planner</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Powered by Claude via AWS Bedrock</p>", unsafe_allow_html=True)
st.divider()

# AWS Authentication
with st.expander("🔐 AWS Authentication", expanded=True):
    aws_access_key = st.text_input("AWS Access Key ID", type="password")
    aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
    aws_region = st.text_input("AWS Region", value="us-east-1")

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

# Model Selection
model_choice = st.selectbox(
    "🤖 Choose Claude Model",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0"
    ]
)

# ----------------------------------------------------------
# Claude Streaming Logic
# ----------------------------------------------------------
def stream_plan(infra_details: str, constraints: str, model_id: str):
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
        Given the infrastructure:
        {infra_details}
        and constraints:
        {constraints}

        Generate 3 optimized AWS migration scenarios with:
        - Approach
        - Estimated cost
        - Timeline
        - Risks
        - Compliance considerations

        Return ONLY valid JSON. Do not include any text outside the JSON.
        Format:
        [
          {{
            "approach": "...",
            "cost": "...",
            "timeline": "...",
            "risks": "...",
            "compliance": "..."
          }}
        ]
        """)
    ]

    return llm.stream(messages)  # Streaming generator

# ----------------------------------------------------------
# Helper: Extract JSON safely
# ----------------------------------------------------------
def extract_json(raw_text):
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    return match.group(0) if match else None

# ----------------------------------------------------------
# Migration Planner UI
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
        st.info("Streaming response from Claude...")
        placeholder = st.empty()
        progress = st.progress(0)
        streamed_text = ""
        token_count = 0

        # Stream tokens live
        for chunk in stream_plan(infra_details, constraints, model_choice):
            # Safely extract text from chunk
            if hasattr(chunk, "content") and chunk.content:
                streamed_text += "".join([part.get("text", "") for part in chunk.content])
            elif hasattr(chunk, "text"):
                streamed_text += chunk.text

            token_count += 1
            progress.progress(min(token_count / 100, 1.0))  # Simulate progress
            placeholder.markdown(f"```\n{streamed_text}\n```")

        progress.empty()  # Remove progress bar after streaming

        # Try parsing JSON after streaming completes
        json_text = extract_json(streamed_text)
        if json_text:
            try:
                scenarios = json.loads(json_text)
                st.session_state["migration_plan"] = scenarios
                st.success("✅ Migration Plan Generated!")

                # ----------------------------------------------------------
                # Enhanced UX Display
                # ----------------------------------------------------------
                st.markdown("### 📑 Migration Scenarios")
                for i, scenario in enumerate(scenarios, start=1):
                    with st.expander(f"🔹 Scenario {i}: {scenario.get('approach','Unknown')}", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"<span style='color:#2E86C1;font-weight:bold;'>Approach:</span> {scenario.get('approach','N/A')}", unsafe_allow_html=True)
                            st.markdown(f"<span style='color:#28B463;font-weight:bold;'>Cost:</span> {scenario.get('cost','N/A')}", unsafe_allow_html=True)
                            st.markdown(f"<span style='color:#F39C12;font-weight:bold;'>Timeline:</span> {scenario.get('timeline','N/A')}", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<span style='color:#C0392B;font-weight:bold;'>Risks:</span> {scenario.get('risks','N/A')}", unsafe_allow_html=True)
                            st.markdown(f"<span style='color:#8E44AD;font-weight:bold;'>Compliance:</span> {scenario.get('compliance','N/A')}", unsafe_allow_html=True)
                        st.divider()
            except json.JSONDecodeError:
                st.error("Failed to parse JSON. Showing raw streamed response:")
                st.write(streamed_text)
        else:
            st.error("Claude did not return valid JSON. Showing raw streamed response:")
            st.write(streamed_text)

# ----------------------------------------------------------
# Download Button
# ----------------------------------------------------------
if "migration_plan" in st.session_state:
    st.download_button(
        "⬇️ Download Migration Plan",
        data=json.dumps(st.session_state["migration_plan"], indent=2),
        file_name="migration_plan.json"
    )
