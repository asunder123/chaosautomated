Here’s a **README.md** for your **Terraform Orchestrator — LangGraph + Claude (Bedrock) with IAM Preflight & Recursive Self-Heal** project that explains what makes it unique:

***

# 🧠 Terraform Orchestrator

**LangGraph + Claude (AWS Bedrock) with IAM Preflight & Recursive Self-Healing**

***

## ✅ **What Makes This Unique**

1.  **Agentic Workflow with LangGraph**
    *   Implements a **graph-based orchestration** for Terraform lifecycle:
            codegen → init → validate → plan ⇄ repair → IAM preflight → apply ⇄ repair → verify → output
    *   Supports **recursive self-healing loops** for both `plan` and `apply` stages until success or max attempts.

2.  **Claude-Powered Intelligence**
    *   Uses **Anthropic Claude 3 Haiku via AWS Bedrock** for:
        *   Generating Terraform code from natural language prompts.
        *   Repairing errors during `validate`, `plan`, and `apply`.
        *   Injecting missing **IAM roles, policies, and attachments** before apply.

3.  **IAM-Aware Preflight**
    *   Claude inspects Terraform files and **adds required IAM resources** automatically.
    *   Ensures compliance with AWS permissions for EC2, S3, CloudFront, etc.

4.  **Robust JSON Parsing**
    *   Extracts valid JSON from Claude output even if wrapped in markdown or code blocks.
    *   Writes files directly to disk with proper structure.

5.  **Cross-Platform Terraform Setup**
    *   Auto-downloads and installs **Terraform CLI** if missing.
    *   Handles Windows, macOS, and Linux seamlessly.

6.  **Streamlit UI with Live Progress**
    *   Displays:
        *   Real-time progress bar.
        *   Claude raw JSON output for debugging.
        *   Per-step logs (stdout/stderr).
    *   Allows manual destroy for cleanup.

7.  **Enterprise-Grade Features**
    *   Secure AWS credential validation via STS.
    *   Configurable retry limits for `plan` and `apply`.
    *   Safe environment variable injection for AWS keys.

***

## 🛠 **Tech Stack**

*   **Streamlit** – Interactive dashboard
*   **AWS Bedrock + Claude 3 Haiku** – AI-driven Terraform generation and repair
*   **LangGraph** – Agent orchestration with conditional edges
*   **Terraform CLI** – Infrastructure provisioning
*   **Python** – Core logic and orchestration

***

## 🔍 **How It Works**

1.  **Validate AWS Credentials**
    *   Enter region, access key, secret key, and optional session token.
2.  **Describe Infrastructure**
    *   Example:
            Create VPC with 2 subnets, EC2 instance, and S3 bucket using Terraform.
3.  **Run LangGraph Workflow**
    *   Steps:
        *   **Codegen** → Claude generates Terraform files.
        *   **Init & Validate** → Terraform initialization and syntax check.
        *   **Plan** → Generates execution plan.
        *   **Repair Loop** → Claude fixes errors until plan succeeds or retries exhausted.
        *   **IAM Preflight** → Adds missing IAM roles/policies.
        *   **Apply** → Deploys infrastructure.
        *   **Repair Loop** → Fixes apply errors if needed.
        *   **Verify & Output** → Displays Terraform outputs.
4.  **Destroy Resources**
    *   Manual or automated cleanup after apply.

***

## 📦 **Installation**

```bash
pip install streamlit boto3 langgraph
terraform --version  # Ensure Terraform CLI is installed or let app auto-install
streamlit run app.py
```

***

## ✅ **Why Use This?**

*   **Automates Terraform generation and deployment** from natural language.
*   **Self-healing orchestration** for reliable infrastructure provisioning.
*   **IAM-aware automation** for secure and compliant deployments.


