# =============================================================================
# Content-Based Recommendation API (FastAPI + uvicorn)
#
# Build context is the REPO ROOT, not this folder — config.py resolves
# DATA_DIR via BASE_DIR.parent, i.e. it expects centralized_data/ as a
# sibling of content_apis/, not nested inside it. Build with:
#   docker build -f content_apis/Dockerfile -t content-api .
# (docker-compose.yml at the repo root does this already.)
# =============================================================================
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# -----------------------------------------------------------------------------
# System dependencies + Microsoft ODBC Driver 17 for SQL Server (required by
# pyodbc / the mssql+pyodbc connection strings used in config.py)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        gnupg \
        unixodbc \
        unixodbc-dev \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -sSL https://packages.microsoft.com/config/debian/12/prod.list -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 \
    && apt-get purge -y --auto-remove gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
COPY content_apis/requirements.txt content_apis/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r content_apis/requirements.txt

# -----------------------------------------------------------------------------
# Application code
# centralized_data/ (root-level sibling) and content_apis/data/ (feature
# matrix cache, recomputed after each retraining cutover) are runtime
# artifacts mounted as volumes — see docker-compose.yml.
# -----------------------------------------------------------------------------
COPY content_apis/ content_apis/
RUN mkdir -p centralized_data content_apis/data/content_data/feature_matrix

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

WORKDIR /app/content_apis

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
