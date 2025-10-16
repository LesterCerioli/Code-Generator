# CodeGen Microservice

## 🚀 Overview
**CodeGen Microservice** is a web tool designed to **generate source code automatically from business requirements**.  
It produces code that follows **best practices** and **robust software architectures**, tailored to the requirements provided by the user.

The solution integrates seamlessly with **GitHub**, creating repositories and publishing generated code using the user’s GitHub credentials.  
Built with **Python 3.12**, **FastAPI**, and **Docker**, this service is designed to be deployed in the cloud and made available for everyone.

---

## ✨ Features
- **Code Generation**: Automatically generate code based on business requirements.
- **Best Practices**: Output follows clean architecture and robust design principles.
- **GitHub Integration**: Create repositories and push generated code directly to GitHub.
- **REST API**: Expose endpoints via FastAPI with interactive Swagger UI documentation.
- **Cloud Ready**: Containerized with Docker for easy deployment on any cloud provider.
- **Scalable**: Designed with modular services and PostgreSQL persistence.

---

## 🛠️ Tech Stack
- **Language**: Python 3.12  
- **Framework**: FastAPI  
- **Database**: PostgreSQL (with native SQL queries)  
- **Containerization**: Docker  
- **Version Control Integration**: GitHub API  
- **Authentication**: GitHub Personal Access Tokens  

---

## 📦 Installation

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- PostgreSQL instance
- GitHub Personal Access Token

### Clone the repository
```bash
git clone https://github.com/your-org/codegen-microservice.git
cd codegen-microservice
