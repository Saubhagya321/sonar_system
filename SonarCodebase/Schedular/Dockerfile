# =============================================================================
# Port Scheduler — long-running APScheduler process (not an HTTP server).
# Schedules per-port batch-process calls against combiner-api and runs the
# UpdateBookingFeedVehicleType stored procedure ahead of them daily.
# =============================================================================
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# -----------------------------------------------------------------------------
# System dependencies + Microsoft ODBC Driver 17 for SQL Server (required by
# pyodbc / the mssql+pyodbc connection string used in config.py)
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
# -----------------------------------------------------------------------------
COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# No EXPOSE/HEALTHCHECK — this is a BlockingScheduler process, not a server.
# Liveness can be checked via `docker ps` / `docker logs`, not an HTTP probe.
CMD ["python3", "scheduler_advance.py"]
