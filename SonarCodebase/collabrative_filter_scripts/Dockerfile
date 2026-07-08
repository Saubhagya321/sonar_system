# =============================================================================
# Collaborative-Filtering Recommendation API (FastAPI + uvicorn)
#
# Build context is the REPO ROOT, not this folder — config.py resolves
# BASE_PATH via Path(__file__).resolve().parents[2] / "centralized_data",
# i.e. it expects centralized_data/ as a sibling of collabrative_filter_scripts/,
# not nested inside it.  Build with:
#   docker build -f collabrative_filter_scripts/Dockerfile -t collaborative-filter-api .
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
# Python dependencies (no torch needed — this is a scikit-learn ensemble, not
# LightGCN)
# -----------------------------------------------------------------------------
COPY collabrative_filter_scripts/requirements.txt collabrative_filter_scripts/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r collabrative_filter_scripts/requirements.txt

# -----------------------------------------------------------------------------
# Application code
# centralized_data/, collabrative_filter_scripts/{data,output,models pickles}
# are runtime artifacts mounted as volumes — see docker-compose.yml.
# -----------------------------------------------------------------------------
COPY collabrative_filter_scripts/ collabrative_filter_scripts/
RUN mkdir -p centralized_data \
        collabrative_filter_scripts/output \
        collabrative_filter_scripts/data

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

WORKDIR /app/collabrative_filter_scripts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
