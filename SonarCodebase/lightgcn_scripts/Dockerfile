# =============================================================================
# LightGCN Vehicle-Chauffeur Recommendation API
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
# CPU-only torch build — this is a serving-time API, no GPU inference needed.
# Must be installed before the general requirements.txt so pip doesn't
# resolve torch from the default (CUDA-bundled) PyPI index instead.
# -----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Application code
# data/, models/, output/, pipeline_data/ are intentionally NOT copied in —
# they're runtime artifacts (hundreds of MB, environment-specific) mounted
# as volumes at container run time. See docker-compose.yml.
# -----------------------------------------------------------------------------
COPY . .
RUN mkdir -p data models output pipeline_data recommendations_output

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# gunicorn (already in requirements.txt) instead of Flask's dev server.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "flask_api_server:app"]
