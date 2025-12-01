Here’s a **README.md** for your **Terraform Orchestrator — LangGraph + Claude + Self-Healing** project that explains what makes it unique:

***

# 🌍 Terraform Orchestrator — LangGraph + Claude + Self-Healing

**AI-driven Infrastructure-as-Code Workflow with Real-Time Validation and Recovery**

***

## ✅ **What Makes This Unique**

1.  **Claude-Powered Code Generation**
    *   Uses **Anthropic Claude via AWS Bedrock** to generate Terraform configurations from natural language prompts.
    *   Supports **multi-step reasoning** for complex infrastructure setups.

2.  **LangGraph Workflow Orchestration**
    *   Implements a **linear graph execution model** for Terraform lifecycle:
        *   `codegen → init → validate → plan → apply → verify → output`
    *   Streams execution state **step-by-step** for transparency.

3.  **Self-Healing Workflow**
    *   Detects non-recoverable errors and offers **automatic retry**.
    *   Deduplicates Terraform providers to prevent conflicts.

4.  **End-to-End Automation**
    *   Handles **AWS authentication**, **Terraform binary setup**, and **environment configuration**.
    *   Generates **ZIP archive of Terraform files** for easy download.

5.  **Dynamic UI**
    *   Built with **Streamlit** for an interactive experience:
        *   Progress bar for workflow steps.
        *   Real-time execution logs.
        *   Expandable sections for Terraform code preview.

6.  **Enterprise-Grade Features**
    *   Secure AWS credential handling.
    *   Supports **Claude model selection** for cost-performance optimization.
    *   Compatible with **Terraform CLI** for production deployments.

***

## 🛠 **Tech Stack**

*   **Streamlit** – Interactive dashboard
*   **AWS Bedrock + Claude** – AI-powered code generation
*   **LangGraph** – Workflow orchestration
*   **Terraform CLI** – Infrastructure provisioning
*   **Python** – Core logic and orchestration

***

## 🔍 **How It Works**

1.  **Authenticate with AWS**
    *   Enter region, access key, secret key, and optional session token.
2.  **Describe Infrastructure**
    *   Provide a natural language prompt (e.g., “Create an EC2 instance with S3 bucket and IAM role”).
3.  **Run Orchestrator**
    *   Executes workflow:
        *   **Codegen**: Claude generates Terraform files.
        *   **Init**: Initializes Terraform.
        *   **Validate**: Claude validates syntax and structure.
        *   **Plan**: Generates execution plan.
        *   **Apply**: Deploys infrastructure.
        *   **Verify**: Confirms resources.
        *   **Output**: Displays results and Terraform files.
4.  **Download Files**
    *   Export all generated Terraform files as a ZIP archive.

***

## 📦 **Installation**

```bash
pip install streamlit boto3 langchain-aws
streamlit run app.py
```

***

## ✅ **Why Use This?**

*   **Natural Language → Infrastructure-as-Code** in minutes.
*   **Self-healing orchestration** for reliable deployments.
*   **Enterprise-ready** with AWS Bedrock integration and Terraform compliance.

***



