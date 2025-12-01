Here’s a **README.md** for your **LangGraph-Driven Autonomous Terraform Engine (RAG + Claude)** project that explains what makes it unique:

***

# 🌐 LangGraph Terraform Orchestration with RAG

**Autonomous Infrastructure Provisioning using Claude (AWS Bedrock) + LangGraph + RAG Context**

***

## ✅ **What Makes This Unique**

1.  **AI-Powered Terraform Automation**
    *   Uses **Anthropic Claude via AWS Bedrock** to:
        *   Generate Terraform HCL from natural language prompts.
        *   Fix validation errors automatically until success.
    *   Fully autonomous provisioning with minimal human intervention.

2.  **LangGraph Workflow Orchestration**
    *   Implements a **graph-based agent workflow**:
            GenerateHCL → ValidateHCL ⇄ FixHCL → TerraformStep (init → plan → apply)
    *   Recursive self-healing loop for validation and apply errors.

3.  **RAG (Retrieval-Augmented Generation) Context**
    *   Fetches **best practices and reusable snippets** from an S3 bucket.
    *   Injects context into Claude prompts for **more accurate and compliant Terraform code**.

4.  **Cross-Platform Terraform Execution**
    *   Runs `terraform init`, `validate`, `plan`, and `apply` automatically.
    *   Displays real-time status and error logs in Streamlit.

5.  **Enterprise-Grade Features**
    *   Secure AWS login and STS identity validation.
    *   Configurable retry limits for fixing errors.
    *   Safe environment variable injection for AWS credentials.

6.  **Streamlit UI**
    *   Simple, interactive interface:
        *   AWS login form.
        *   Infrastructure prompt input.
        *   Live progress and final Terraform code display.

***

## 🛠 **Tech Stack**

*   **Streamlit** – Interactive dashboard
*   **AWS Bedrock + Claude** – AI-driven Terraform generation and repair
*   **LangGraph** – Workflow orchestration
*   **Terraform CLI** – Infrastructure provisioning
*   **Python** – Core logic and orchestration
*   **S3 (RAG)** – Context retrieval for best practices

***

## 🔍 **How It Works**

1.  **Login to AWS**
    *   Enter Access Key, Secret Key, and Region.
2.  **Describe Infrastructure**
    *   Example:
            1 EC2 instance and 1 S3 bucket.
3.  **Fetch RAG Context**
    *   Retrieves snippets from S3 bucket (`rag-terraform-context`) for enhanced code generation.
4.  **Run LangGraph Workflow**
    *   Steps:
        *   **GenerateHCL** → Claude creates Terraform code.
        *   **ValidateHCL** → Checks syntax.
        *   **FixHCL** → Claude repairs errors recursively.
        *   **TerraformStep** → Executes init, plan, and apply.
5.  **Display Results**
    *   Shows final Terraform code and provisioning status.

***

## 📦 **Installation**

```bash
pip install streamlit boto3 langgraph
terraform --version  # Ensure Terraform CLI is installed
streamlit run app.py
```

***

## ✅ **Why Use This?**

*   **End-to-end automation** for Terraform provisioning.
*   **Self-healing orchestration** for error-free deployments.
*   **Context-aware generation** using RAG for best practices.

***
