# =============================================================================
# Recommendation Combiner API (Flask + gunicorn)
#
# Build context is THIS folder — unlike collab-api/content-api, config.py's
# DATA_DIR is local (BASE_DIR / "data"), not anchored to a repo-root
# centralized_data/ sibling, so no root-context build is needed. Build with:
#   docker build -t combiner-api .
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
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Application code
# data/, weights/, logs/, batch_logs/ are runtime artifacts (client data,
# learned feedback weights, log files) mounted as volumes — see
# docker-compose.yml. routes.py is the live entrypoint (module-level
# `app = create_app()`); combiner_api_final_backup.py is a superseded copy
# and is not what's served here.
# -----------------------------------------------------------------------------
COPY . .
RUN mkdir -p data weights logs batch_logs

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "1", "--threads", "4", "--timeout", "120", "routes:app"]
