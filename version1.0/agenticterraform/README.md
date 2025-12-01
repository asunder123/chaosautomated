Here’s a **README.md** for your **Agentic Modular Terraform Orchestrator — Bedrock Claude 3 Haiku + LangGraph** project that explains what makes it unique:

***

# 🧠 Agentic Modular Terraform Orchestrator

**Claude 3 Haiku on AWS Bedrock + LangGraph for Self-Healing Infrastructure-as-Code**

***

## ✅ **What Makes This Unique**

1.  **Claude-Powered Modular Terraform Generation**
    *   Uses **Anthropic Claude 3 Haiku via AWS Bedrock** to generate a **fully modular Terraform project**:
        *   `main.tf` at root
        *   `modules/<service>/main.tf` (+ `variables.tf` if needed)
    *   Ensures **provider region correctness** and unique resource naming.

2.  **Agentic Workflow with LangGraph**
    *   Implements a **multi-step agent** that:
        *   Generates Terraform code.
        *   Runs `terraform init`, `validate`, `plan`, `apply`.
        *   Repairs errors automatically using Claude until success or max retries.
    *   **Self-healing loop**:
        *   Up to **50 repair attempts** for `plan` and `apply` steps.

3.  **Error-Aware Repair**
    *   Claude acts as a **critic agent**:
        *   Reads Terraform error logs.
        *   Suggests **minimal fixes** for broken files.
        *   Updates only necessary files in the modular structure.

4.  **Enterprise-Grade Automation**
    *   Auto-downloads and installs **Terraform CLI** if missing.
    *   Secure AWS credential handling.
    *   Optional **auto-destroy** after successful apply.

5.  **Interactive Streamlit UI**
    *   Displays:
        *   Generated Terraform files.
        *   Workflow logs with stdout/stderr for each step.
    *   Allows manual destroy for cleanup.

6.  **Robust Execution**
    *   Handles platform detection (Windows/Linux/macOS).
    *   Configurable recursion and retry limits.
    *   Safe environment variable injection for AWS credentials.

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
            Create VPC with 2 subnets, EC2 instance, and S3 bucket using modules.
3.  **Run Modular Workflow**
    *   Steps:
        *   **Codegen** → Claude generates modular Terraform files.
        *   **Init** → Initializes Terraform.
        *   **Validate** → Checks syntax.
        *   **Plan** → Generates execution plan.
        *   **Critic Loop** → Repairs errors until plan succeeds or retries exhausted.
        *   **Apply** → Deploys infrastructure.
        *   **Critic Loop** → Repairs apply errors if needed.
        *   **Output** → Displays Terraform outputs.
4.  **Auto-Destroy (Optional)**
    *   Cleans up resources after successful apply.

***

## 📦 **Installation**

```bash
pip install streamlit boto3 langgraph
streamlit run app.py
```

Ensure **Terraform CLI** is installed or let the app auto-install it.

***

## ✅ **Why Use This?**

*   **Automates modular Terraform generation** from natural language.
*   **Self-healing orchestration** for reliable deployments.
*   **Enterprise-ready** with AWS Bedrock integration and Terraform compliance.

***

