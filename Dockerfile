FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    sqlite3 git nodejs npm && pip install uv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN uv pip install  --system --no-cache .

# Install streamlit-firebase-auth, build frontend, and patch pyproject.toml
# NOTE: This package requires a custom workflow due to upstream pyproject.toml issues.
# 1. Build the frontend to ensure the required "build" directory exists,
#    otherwise the SDK fails with: "No such component directory: .../frontend/build"
# 2. Patch pyproject.toml:
#    - Remove invalid [project].requires entries (PEP 621 does not allow this)
#    - Change build-backend to setuptools (required for correct installation)
#    - Ensure build-system.requires = ["setuptools", "wheel"]
# NOTE: This custom build workflow is only required for Docker Compose builds, 
# not when building directly in VSCode DevContainers.
RUN git clone https://github.com/munaita-0/streamlit-firebase-auth.git /tmp/sfa && \
    cp /app/MANIFEST.in /tmp/sfa/ && \
    cd /tmp/sfa/streamlit_firebase_auth/frontend && npm install && npm run build && \
    cd /tmp/sfa && \
    # Remove [project].requires array (invalid in PEP 621)
    awk 'BEGIN {in_proj=0; in_req=0} /^\[project\]/ {in_proj=1} in_proj && /^\[/ && $0 !~ /^\[project\]/ {in_proj=0} in_proj && /^\s*requires = \[/ {in_req=1; next} in_req && /^\s*\]/ {in_req=0; next} in_req {next} {print}' /tmp/sfa/pyproject.toml > /tmp/sfa/pyproject.fixed.toml && \
    mv /tmp/sfa/pyproject.fixed.toml /tmp/sfa/pyproject.toml && \
    # Use setuptools as build backend and ensure correct dependencies
    sed -i 's/hatchling\.build/setuptools.build_meta/' /tmp/sfa/pyproject.toml && \
    sed -i 's/hatchling/setuptools", "wheel/' /tmp/sfa/pyproject.toml && \
    cd /tmp/sfa && uv pip install --system .

CMD ["streamlit", "run", "src/chunk_recall_trainer/main.py", "--server.port=8501", "--server.enableCORS=false", "--browser.gatherUsageStats=false"]