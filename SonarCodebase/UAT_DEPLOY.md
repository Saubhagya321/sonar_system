# UAT Deployment Runbook

Step-by-step guide for deploying the dockerized Sonar stack on the client's
Windows GCP VM. Assumes images are already pushed to ghcr.io — this runbook
only covers pulling and running them on the VM. For the initial build/push
side, see `DEPLOYMENT.md`.

## What you need with you before going on-site

- [ ] The ghcr.io PAT (`ghp_...` classic token with `read:packages` scope
      minimum). Same token used for the dev push. Store in a password manager
      or on your phone — not in email.
- [ ] Two files on a USB / OneDrive / accessible via RDP transfer:
      - `SonarCodebase/docker-compose.yml`
      - `SonarCodebase/.env`
- [ ] Credentials for the UAT SQL Server (`migration_svc` password) if
      different from dev's `q3tech@1234`.
- [ ] The UAT SQL Server hostname/IP and confirmation whether it lives on the
      same VM or a separate machine.
- [ ] This file, printed or downloaded — client Wi-Fi may block Claude Code
      / GitHub.

## Time budget

| Phase                              | Time      | Notes                                     |
|------------------------------------|-----------|-------------------------------------------|
| Prereqs (Docker Desktop install)   | 30–60 min | Skip if already installed                 |
| SQL Server host-side setup         | 20–30 min | Skip if DB is on a separate managed box   |
| File transfer + `.env` edit        | 5 min     |                                           |
| `docker login ghcr.io`             | 1 min     |                                           |
| `docker compose pull` (~15 GB)     | 15–40 min | Slowest step, depends on VM bandwidth     |
| `docker compose up -d`             | 2–5 min   |                                           |
| Smoke tests                        | 10 min    |                                           |
| Buffer for surprises               | 30 min    |                                           |
| **Total (realistic)**              | **1.5–3h**|                                           |

Budget half a day for first-time-on-this-VM deploy; ~1 hour for repeat
deploys.

---

## Phase 1 — Prereqs

Open **PowerShell as Administrator** on the VM.

```powershell
# Confirm Docker is installed and reachable
docker version

# Confirm outbound to ghcr.io works (corporate firewall check)
curl.exe -I https://ghcr.io      # expect 200 or 301
```

If Docker is missing: install Docker Desktop for Windows from
docker.com/products/docker-desktop, restart Windows, launch it once so the
daemon starts, then continue.

---

## Phase 2 — SQL Server host-side setup

**Skip this phase if the UAT DB is on a separate box you don't manage** — in
that case, just make sure someone has already done the equivalent of steps
2–5 in `DEPLOYMENT.md` on that server, and you have the connection details.

If the SQL Server is on THIS VM, follow `DEPLOYMENT.md` sections 2–5 verbatim.
Summary of what those sections do:

1. **Mixed Mode auth** — SSMS → right-click server → Properties → Security →
   *SQL Server and Windows Authentication mode* → restart SQL Server service.
2. **Create `migration_svc` login** — SSMS → Security → Logins → New Login,
   SQL Authentication, password (e.g. `q3tech@1234`), map to both
   `TransElite.Q3` and `Q3_Recommendation` with `db_datareader` +
   `db_datawriter`.
3. **Enable TCP/IP** — SQL Server Configuration Manager → Network Config →
   TCP/IP: Enabled → IPAll TCP Port `1433` → restart service.
4. **Firewall inbound rule** — from admin PowerShell:
   ```powershell
   New-NetFirewallRule -DisplayName "SQL Server (Docker) 1433" `
       -Direction Inbound -Protocol TCP -LocalPort 1433 -Action Allow `
       -Profile Domain,Private,Public
   ```

Verify:
```powershell
Test-NetConnection localhost -Port 1433   # expect TcpTestSucceeded : True
```

---

## Phase 3 — Transfer the 2 files

Put `docker-compose.yml` and `.env` into a folder on the VM, e.g.
`C:\sonar\`. Any transfer method works: RDP clipboard/drag, shared drive,
USB, or `scp` if the VM has SSH.

You do NOT need to copy source code, Dockerfiles, or any other repo
contents. The images have everything baked in.

---

## Phase 4 — Configure `.env` for UAT

Open `C:\sonar\.env` in Notepad. Edit **only** these lines:

```
DB_HOST=<UAT SQL Server hostname or IP>
DB_PASS=<UAT password, URL-encoded (@ = %40)>
```

Rules:
- If SQL Server is on the **same VM**: `DB_HOST=host.docker.internal`
- If SQL Server is on **another box**: `DB_HOST=<hostname or IP>` (e.g.
  `uat-sql.internal` or `10.0.0.5`)
- Special characters in the password must be URL-encoded because it goes
  into a SQLAlchemy URL. `@` becomes `%40`, `#` becomes `%23`, etc.

Everything else in the file stays as-is.

---

## Phase 5 — Log in, pull, run

```powershell
cd C:\sonar

# 5.1 Log in to ghcr.io (once per VM lifetime — Docker Desktop persists it)
docker login ghcr.io -u Saubhagya321
# Paste the ghp_... PAT when prompted. Expect "Login Succeeded".

# 5.2 Pull all 7 images (~15 GB total, slowest step)
docker compose pull

# 5.3 Start the stack in detached mode
docker compose up -d

# 5.4 Confirm all services are up
docker compose ps
```

Expected state after `docker compose ps`:
- 6 services `Up` / `running` / `healthy`
- `orchestrator` shows `Exited (0)` — that is CORRECT. It's a one-shot job
  that runs `main_orchestrator.py` and exits; it's not a long-running server.
  You invoke it explicitly via `docker compose run --rm orchestrator ...`
  when you want a retraining run.

---

## Phase 6 — Smoke tests

### 6.1 Each API responds
```powershell
curl.exe http://localhost:5000/               # lightgcn — expect JSON/HTML
curl.exe http://localhost:8000/docs           # collab FastAPI Swagger UI
curl.exe http://localhost:8003/docs           # content FastAPI Swagger UI
curl.exe http://localhost:8001/               # combiner (Flask)
curl.exe http://localhost/                    # frontend (nginx serves React)
```

### 6.2 DB connectivity per service
```powershell
docker compose logs lightgcn-api | Select-Object -Last 30
docker compose logs collab-api    | Select-Object -Last 30
docker compose logs content-api   | Select-Object -Last 30
docker compose logs combiner-api  | Select-Object -Last 30
docker compose logs scheduler     | Select-Object -Last 30
```

Watch for: no `pyodbc.OperationalError`, no `Login timeout expired`, no
`Login failed for user 'migration_svc'`. Absence of DB errors = passing.

### 6.3 End-to-end recommendation call
Use a real client/port ID from the UAT DB (ask client for a test value):
```powershell
curl.exe -X POST http://localhost:8001/api/v1/recommendations `
  -H "Content-Type: application/json" `
  -d '{\"port_id\": 1, \"client_id\": 1}'
```
Expect JSON with fused recommendations from lightgcn + collab + content.

### 6.4 Orchestrator dry-run (safe — doesn't retrain)
```powershell
docker compose run --rm orchestrator --run-now --dry-run
docker compose run --rm orchestrator --status
```

### 6.5 Frontend in a browser
Open `http://<VM-IP>/` from a machine that can reach the VM. Log in,
trigger a recommendation from the UI, confirm the results render.

---

## Common failure modes

| Symptom                                          | Cause                                          | Fix                                                              |
|--------------------------------------------------|------------------------------------------------|------------------------------------------------------------------|
| `docker compose pull` → `unauthorized`           | Not logged into ghcr.io / PAT expired          | `docker login ghcr.io -u Saubhagya321`                           |
| `docker compose pull` → connection reset         | Corporate firewall blocks ghcr.io              | Ask IT to allowlist `ghcr.io` + `pkg-containers.githubusercontent.com` |
| Services start but log `Login timeout expired`   | Firewall blocking 1433 or wrong DB_HOST        | `Test-NetConnection <DB_HOST> -Port 1433` from the VM            |
| `Login failed for user 'migration_svc'`          | Wrong password, or login not created on UAT DB | Verify Phase 2 was done on the actual UAT SQL Server             |
| `Cannot open database "TransElite.Q3"`           | `migration_svc` missing role mapping on DB     | SSMS → Login properties → User Mapping → tick both DBs + roles   |
| Combiner returns empty recommendations           | `Client.csv` missing in `centralized_data/`    | Copy the file into the mounted volume path                       |
| Frontend loads but API calls fail with CORS/URL  | `VITE_API_URL` baked at build time is wrong    | Rebuild frontend with `--build-arg VITE_API_URL=<UAT-URL>` on dev, re-push |

---

## Before you leave the client office

```powershell
# Capture logs for remote debugging later
docker compose logs > C:\sonar\uat-startup.log
docker compose ps  > C:\sonar\uat-ps.txt
```

Email both files to yourself. If something breaks after you leave, you'll be
glad you have them.

## Rolling updates later (from your dev machine)

```bash
# On dev: rebuild + push new images
docker compose build
docker compose push

# On UAT VM: pull the new images + recreate containers
docker compose pull
docker compose up -d
```

Compose only recreates containers whose image changed, so unchanged services
keep running.
