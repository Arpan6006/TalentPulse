# ── Multi-Service Container Configuration ──────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="TalentPulse Enterprise ML Platform"
LABEL description="Production ML microservice & interactive workforce retention intelligence dashboard"

# ── System Dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python Dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Application Source ────────────────────────────────────────────────────
COPY . .

# ── Expose Service Ports ───────────────────────────────────────────────────────
# 8000: FastAPI REST Microservice
# 8501: Streamlit Executive Dashboard
EXPOSE 8000 8501

# ── Healthcheck ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8501/_stcore/health || exit 1

# ── Default Execution ──────────────────────────────────────────────────────────
# Default to Streamlit dashboard; overridden by docker-compose for FastAPI microservice
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
