Here’s a **README.md** for your **Chaos Orchestrator — Claude on Bedrock + CrewAI Executor** project that explains what makes it unique:

***

# 🧠 Chaos Orchestrator

**Anthropic Claude on AWS Bedrock + CrewAI Executor for Automated Chaos Engineering**

***

## ✅ **What Makes This Unique**

1.  **Claude-Powered Chaos Workflow**
    *   Uses **Anthropic Claude via AWS Bedrock** to:
        *   Generate **chaos experiment plans**.
        *   Convert plans into **AWS Fault Injection Simulator (FIS) templates**.
        *   Validate templates for **security and compliance**.
        *   Auto-generate **Python (boto3)** and **Terraform (HCL)** code artifacts.

2.  **CrewAI-Style Execution**
    *   Executes generated artifacts in a **safe sandbox**:
        *   Runs Python scripts locally.
        *   Validates Terraform configurations (`init`, `fmt`, `validate`).
    *   Summarizes execution results for quick feedback.

3.  **Multi-Step Orchestration**
    *   Interactive workflow with **five key steps**:
        *   **Plan** → **Generate FIS** → **Validate** → **Codegen** → **Execute**
    *   Optional **Improve Plan** step using validation feedback.

4.  **Context-Aware Generation**
    *   Injects **AWS account details, IAM roles, and tagging standards** into Claude prompts.
    *   Dynamically replaces placeholders like `${roleArn}` with real or dummy values.

5.  **Robust JSON Parsing**
    *   Handles Claude’s output gracefully:
        *   Sanitizes triple-quoted strings.
        *   Extracts Python and HCL code blocks.
        *   Falls back to loose JSON parsing if needed.

6.  **Enterprise-Grade Features**
    *   Secure AWS login via **STS identity check**.
    *   Supports **Claude model selection** for cost-performance trade-offs.
    *   Fully integrated with **AWS Bedrock runtime**.

***

## 🛠 **Tech Stack**

*   **Streamlit** – Interactive UI
*   **AWS Bedrock + Claude** – AI-driven chaos planning
*   **CrewAI Executor** – Safe artifact execution
*   **Terraform CLI** – Infrastructure validation
*   **Python** – Core orchestration logic

***

## 🔍 **How It Works**

1.  **Login to AWS**
    *   Enter region, access key, secret key, and optional session token.
2.  **Describe Chaos Scenario**
    *   Example:
            Simulate EC2 network latency on staging ASG for 2 minutes
3.  **Run Workflow**
    *   **Step 1: Plan** → Claude generates chaos plan JSON.
    *   **Step 2: Generate FIS** → Converts plan to AWS FIS template.
    *   **Step 3: Validate** → Security audit of FIS template.
    *   **Step 4: Codegen** → Generates Python boto3 and Terraform code.
    *   **Step 5: Execute** → Runs generated code in sandbox.
4.  **Improve Plan**
    *   Uses validation feedback to refine chaos plan.

***

## 📦 **Installation**

```bash
pip install streamlit boto3 langchain-aws
terraform --version  # Ensure Terraform CLI is installed
streamlit run app.py
```

***

## ✅ **Why Use This?**

*   **Automates chaos engineering** from planning to execution.
*   **Ensures safety and compliance** with validation step.
*   **Bridges AI and DevOps** for faster, smarter infrastructure testing.

***
