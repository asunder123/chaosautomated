
# Cloud Refactor Planner

## Overview
This project provides an **automated refactoring planner** for monolithic applications, leveraging:
- **Streamlit** for an interactive UI
- **AWS Bedrock (Claude Haiku)** for generating migration plans
- **GitHub Actions** for CI/CD
- **CloudFormation** for infrastructure provisioning
- **Docker** for containerization

The tool analyzes your monolithic code, generates a **Cloud Factory-ready architecture plan**, and creates a **GitHub-ready repository** with:
- Original monolith code
- Deployment blueprint
- Infrastructure templates
- CI/CD workflow
- Docker configuration

---

## Features
- **Code Parsing:** Extracts functions, classes, and imports from your monolith.
- **Domain Inference:** Identifies API, data, and core layers.
- **AI-Powered Migration Plan:** Uses AWS Bedrock Claude Haiku to generate structured JSON plans.
- **Infrastructure Templates:** Creates AWS CloudFormation stubs with networking, compute, storage, IAM, and monitoring.
- **Containerization:** Adds Dockerfile and docker-compose for app deployment.
- **CI/CD Pipeline:** GitHub Actions workflow for building Docker images, pushing to ECR, and deploying via CloudFormation.
- **Downloadable Repo:** Bundles everything into a ZIP for easy GitHub integration.

---

## Repository Structure
