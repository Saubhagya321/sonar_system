# Docker Deployment Notes

Running log of what it takes to get the 5 Docker images (`lightgcn-api`,
`collab-api`, `content-api`, `combiner-api`, `retraining-orchestrator`)
running against a real SQL Server on a new machine/VM. Append new sections
here as new steps get figured out — this is meant to be the single place
that explains *why* each deployment step exists, not just the command.

Repo root (this machine): `SonarCodebase\`
Compose file: `docker-compose.yml` (repo root)

---

## 1. Why containers can't reach SQL Server via `localhost`

Every `.env` file's `DB_CONNECTION_STRING` currently points at `localhost`
with `trusted_connection=yes` (Windows Integrated Auth) — correct for
running the scripts directly on the host with SSMS, wrong once the same
code runs inside a Docker container.

Two separate problems, both must be fixed:

1. **`localhost` means "this container", not the host machine.** Docker
   gives every container its own network namespace — `localhost` inside a
   container's network namespace refers to itself, never the machine
   Docker is running on.
2. **`trusted_connection=yes` (Windows Integrated Auth) does not work from
   Linux containers.** Trusted auth relies on the OS-level Windows
   security context (the calling process's Windows identity, verified via
   the domain/local machine). A Linux container has no Windows identity to
   present, so this style of connection is not available to it — SQL
   Server Authentication (username + password) has to be used instead.

This applies whenever SQL Server runs **natively on the same machine as
Docker** (our case: dev laptop today, target VM later — SSMS + the DB are
installed directly on the OS, not containerized). If SQL Server ever moves
to its own separate server, the fix is simpler: just point
`DB_CONNECTION_STRING` at that server's real hostname/IP instead of
`localhost` — no Docker-side config needed, since Docker's default network
already routes outbound to the LAN.

---

## 2. Enable Mixed Mode Authentication in SQL Server (SSMS)

Needed once per machine (dev laptop now, then again on the deployment VM).

1. Open **SQL Server Management Studio**, connect to the instance with your
   current (Windows Auth) login as you normally do.
2. In **Object Explorer**, right-click the server instance (the top-level
   node, e.g. `YOURMACHINE\SQLEXPRESS` or just the instance name) →
   **Properties**.
3. Go to the **Security** page (left-hand list).
4. Under "Server authentication", switch from
   *"Windows Authentication mode"* to
   **"SQL Server and Windows Authentication mode"** (this is "mixed mode" —
   it keeps Windows Auth working for existing tools like SSMS *and* adds
   SQL logins for anything that can't do Windows Auth, like our containers).
5. Click **OK**.
6. SSMS will prompt that this requires a service restart — it does not
   apply live.
7. Restart the SQL Server service:
   - Easiest: open **SQL Server Configuration Manager** → **SQL Server
     Services** → right-click **SQL Server (<instance name>)** → **Restart**.
   - Or via `services.msc` → find `SQL Server (<instance name>)` → Restart.

## 3. Create a SQL login for the containers to use

Do this after mixed mode is enabled and the service has restarted.

1. In SSMS Object Explorer: **Security** → right-click **Logins** →
   **New Login...**
2. Login name: pick something identifying, e.g. `docker_app`. (Actually used
   on this machine: `migration_svc`.)
3. Select **SQL Server authentication**, set a password (uncheck "Enforce
   password expiration" for a service account — it shouldn't lock itself
   out).
4. On the **User Mapping** page: check the databases this account needs
   (`TransElite.Q3`, `Q3_Recommendation` — see table in `COMMANDS.md` for
   which folder uses which DB), and grant at least `db_datareader` /
   `db_datawriter` (add more roles only if a specific script needs them,
   e.g. stored-procedure execute rights for
   `comnbiner_script_final/connect_stored_procedure.py`).
5. Click OK. Test it: New Query window → disconnect → reconnect using
   **SQL Server Authentication** with `migration_svc` / the password, confirm
   it connects and can query the mapped databases.

**Status: done** on the dev machine — login `migration_svc` created, mapped to
both `TransElite.Q3` and `Q3_Recommendation` with `db_datareader`/`db_datawriter`.

## 4. Enable the TCP/IP protocol

SSMS on the same machine as the DB often connects via Shared Memory or
Named Pipes, neither of which a container can use. TCP/IP must be turned on
explicitly.

1. Open **SQL Server Configuration Manager**.
2. **SQL Server Network Configuration** → **Protocols for <instance name>**.
3. Right-click **TCP/IP** → **Enable** (if not already).
4. Double-click **TCP/IP** → **IP Addresses** tab → scroll to **IPAll** →
   confirm/set **TCP Port** to `1433` (default).
5. Restart the SQL Server service again (same as step 2.7) for this to
   take effect.

## 5. Allow the port through Windows Firewall

1. Open **Windows Defender Firewall with Advanced Security**.
2. **Inbound Rules** → **New Rule...**
3. Rule type: **Port** → **TCP** → **Specific local ports**: `1433`.
4. **Allow the connection** → apply to all profiles (Domain/Private/Public)
   unless your org's policy says otherwise → name it something like
   `SQL Server (Docker) 1433`.
5. This is needed because Docker Desktop's containers reach the host over
   a virtual network adapter, which Windows Firewall treats like any other
   inbound connection to the SQL Server port.

## 6. Update `DB_CONNECTION_STRING` in each `.env`

Once steps 2–5 are done, the connection string changes in two ways from
what's there today:
- Host: `localhost` → **`host.docker.internal`** (Docker Desktop's
  built-in DNS name that always resolves to the host machine from inside
  any container — works identically on the dev laptop and the VM, as long
  as Docker Desktop is what's running Docker there too).
- Auth: `trusted_connection=yes` → `migration_svc:<password>` in the URL
  (SQL auth, matching the login created in step 3), plus
  `Encrypt=yes&TrustServerCertificate=yes` (note: `yes`/`no`, not
  `True`/`False` — that's the ADO.NET convention, not this pyodbc dialect).
  Any special character in the password must be URL-encoded (`@` → `%40`).

Actual value applied (same shape in all 5 files, only host/db differ):
```
mssql+pyodbc://migration_svc:q3tech%401234@host.docker.internal:1433/<DB_NAME>?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes
```

Files updated (all previously used `localhost` + `trusted_connection=yes`):
- `collabrative_filter_scripts/.env` — DB `Q3_Recommendation`
- `lightgcn_scripts/.env` — DB `TransElite.Q3`
- `content_apis/.env` — DB `TransElite.Q3`
- `comnbiner_script_final/.env` — DB `Q3_Recommendation` (also updated
  `DB_SERVER`/`DB_TRUSTED_CONNECTION` companion vars in the same file)
- `retraining_pipeline/.env` — DB `TransElite.Q3`

This is a `.env`-only change — none of these are baked into the Docker
images (`docker-compose.yml` bind-mounts each folder's `.env` at
container-start time), so no image rebuild is needed after editing them.

---

## Open items / not yet done

- [x] Step 2 (mixed mode) done on this dev machine.
- [x] Step 3 (SQL login `migration_svc`) done on this dev machine.
- [x] Step 4 (TCP/IP protocol) — verified already enabled, static port 1433,
      and confirmed actively `LISTEN`ing on `0.0.0.0:1433` via
      `Get-NetTCPConnection`. Instance is the default instance
      (`MSSQL16.MSSQLSERVER`, i.e. SQL Server 2022), no named-instance
      port lookup needed.
- [x] Step 5 (Windows Firewall inbound rule for port 1433) — no rule existed
      (confirmed via `Get-NetFirewallRule`), so created one:
      `New-NetFirewallRule -DisplayName "SQL Server (Docker) 1433" -Direction
      Inbound -Protocol TCP -LocalPort 1433 -Action Allow -Profile
      Domain,Private,Public` (required a one-time UAC-elevated PowerShell
      session — this needs local admin rights, so on the deployment VM
      whoever runs it must have admin access there too).
- [x] `.env` files updated on this dev machine (this section).
- [ ] Same steps (2–5, `.env` update) need repeating on the deployment VM
      once it's provisioned.
- [x] Verified each serving image's DB health check against the new
      connection string (`docker compose up -d`, checked each `/health`):
      `lightgcn-api`, `collab-api`, `content-api`, `combiner-api` all
      `healthy`. `combiner-api`'s `/health` explicitly showed
      `"database":{"status":"connected"}` — direct proof the new SQL-auth
      connection string works end to end.

## 7. Permission gap: SQL login mapped to a DB ≠ role actually applied

Hit this during verification: `lightgcn-api` came up connected (engine
initialized, `/health` returned 200) but logged
`SELECT permission was denied on the object 'Vehicle', database
'TransElite.Q3', schema 'dbo'` and loaded 0 port/vehicle/chauffeur mappings.

Root cause, found via `sqlcmd`:
```sql
SELECT dp.name, r.name AS role_name
FROM sys.database_principals dp
LEFT JOIN sys.database_role_members drm ON dp.principal_id = drm.member_principal_id
LEFT JOIN sys.database_principals r ON drm.role_principal_id = r.principal_id
WHERE dp.name = 'migration_svc';
```
run against each database showed `migration_svc` had `db_datareader` /
`db_datawriter` (even `db_owner`) in `Q3_Recommendation`, but **no role
membership at all** in `TransElite.Q3` — the login existed there (SSMS's
User Mapping page had created the user) but the role checkboxes hadn't
actually taken effect for that database. Being mapped to a database and
being a member of a role in that database are two separate things — always
verify both, not just that the login can connect.

Fix (run once per affected database):
```sql
USE [TransElite.Q3];
ALTER ROLE db_datareader ADD MEMBER migration_svc;
ALTER ROLE db_datawriter ADD MEMBER migration_svc;
```
Then restart the affected container so it reloads any data it cached 0/empty
results for on first (failed) load — a running process won't retry
automatically. Confirmed fixed: `lightgcn-api` restart picked up 98 ports
with vehicle mappings, 5 with chauffeur mappings, `ENGINE READY!`, container
`healthy`.

**Takeaway for the VM setup:** after creating the login there, explicitly
re-run the `sys.database_role_members` check above for every database in the
User Mapping table (`TransElite.Q3`, `Q3_Recommendation`) — don't just trust
that checking the boxes in SSMS applied cleanly.

---

## 8. Bugs found/fixed getting a full retraining run to complete

Hit these running the monthly retraining pipeline (`docker compose run --rm
orchestrator --run-now`) end to end for the first time. None of these are
Docker-specific, but they only surfaced once a real run pushed real data
volumes through the pipeline — worth keeping here since the next full run
(or the next person debugging a stall) will hit the same shape of problem.

### 8.1 SQL parameter chunking (`pyodbc` / SQL Server parameter limit)

Large `IN (...)` queries built from a full list of client/booking IDs in one
shot hit SQL Server's parameter limit and failed outright once the dataset
was big enough (fine on small dev data, breaks on real volume). Fixed by
chunking the ID list into batches before building each query, matching the
same pattern already used elsewhere in the codebase for large `IN` clauses.

### 8.2 `shutil.copy`/`copy2` fail with `PermissionError` on the bind-mounted staging volume

`step_cutover` in `retraining_pipeline/main_orchestrator.py` promotes staged
artifacts to their live paths with a plain file copy. Two rounds of this bug:

1. `shutil.copy2()` (copies content + metadata, including `utime`) failed —
   the bind-mounted Docker volume doesn't allow the container to set file
   times/permissions on it the way `copy2` expects.
2. Switching to `shutil.copy()` (still copies *mode*, i.e. calls
   `copymode()` → `chmod()` internally) failed for the same underlying
   reason — `copy()` is not a pure content copy either.

Fix: use `shutil.copyfile()` in both `_cp()` and `_cpdir()` — the only one of
the three that copies file *content only*, no metadata/permission syscalls:
```python
shutil.copyfile(src, dst)   # not copy() or copy2()
```
Confirmed fixed by resuming a partially-failed cutover directly from already
-staged artifacts (via `StagingLayout.from_root()`), then via a full clean
run end to end.

**Takeaway:** on any bind-mounted volume (Docker bind mount, some network
filesystems), prefer `shutil.copyfile()` over `copy`/`copy2` unless you've
confirmed the mount actually supports `chmod`/`utime`.

### 8.3 O(n) full-table-scan anti-pattern, found twice (weight optimization code)

Two separate functions filtered a full DataFrame (all clients' ground-truth
or history rows) *inside* a per-client, per-combination loop, instead of
partitioning once up front — turning what should be an O(n) pass into
O(n × combos × clients). At real data volume this looked like a hang (long
gaps between visible log lines) but was actually just slow, CPU-active work.

Fixed both by grouping once into a `Dict[client_id, DataFrame]` before the
loop, then doing O(1) dict lookups (`.get(client_id, empty_df)`) inside it:
- First instance: (see prior session, not reflected in this log at time of
  writing — same pattern, already fixed before this section was added).
- Second instance: `comnbiner_script_final/grid_search.py`,
  `GridSearchOptimizer.evaluate_weights()`/`grid_search()` — pre-grouped
  `ground_truth_df` into `gt_by_client` once per `grid_search()` call.
  Confirmed fixed: GLOBAL CREDIT (124 clients × 72 weight combinations)
  dropped from an indeterminately long stall to 11m40s.

**Takeaway:** if a grid search / weight optimization loop is filtering a
full-size DataFrame inside its innermost loop, group-by once outside the
loop first — this bug pattern recurs, so check for it specifically whenever
adding a new per-client/per-combo evaluation function.

### 8.4 `CLIENT_TYPE_FILE` wiring — orchestrator container had no populated file to point at

`comnbiner_script_final/.env` sets `CLIENT_TYPE_FILE` to its own
`data/client_final.csv`, but nothing populates that file inside the
orchestrator's container — the actual `ID` + `RefClientType` columns it
needs live in `Client.csv`, already exported into the shared
`centralized_data` volume by `db_pull`. Fixed by overriding
`CLIENT_TYPE_FILE` for the `orchestrator` service in `docker-compose.yml` to
point at the mounted `centralized_data` copy instead:
```yaml
environment:
  CLIENT_TYPE_FILE: /app/centralized_data/Client.csv
```
This only affects the orchestrator container's env — `combiner-api` (the
standalone serving container) still reads its own `.env`'s
`data/client_final.csv` default, which is a separate, currently-open gap
(see section 9 below).

### 8.5 Per-model retraining CLI flag (`--steps`)

Added to `retraining_pipeline/main_orchestrator.py` so a single model can be
retrained (or just cutover/reload re-run) without re-running the full
multi-hour pipeline — useful for exactly the kind of partial-failure resume
situation in 8.2 above. See `--help` output for usage; key points:
- `--steps STEP1,STEP2,...` accepts aliases (`lightgcn`, `collab`, `combiner`,
  `features`, `reload`, etc.) and always executes in canonical pipeline order
  regardless of the order given.
- `--staging-version VERSION_TAG` targets a specific staging directory
  (defaults to the current month) — lets you resume against artifacts a
  prior run already staged, without recomputing them.
- Prerequisite checks fail fast with an itemized list of what's missing
  (e.g. `combiner` requested without `lightgcn`/`collaborative` recommendation
  CSVs staged) rather than partway through the step.
- Verified live: dry-run `cutover` against an already-staged month passed
  cleanly; a deliberately nonexistent `--staging-version` and a bogus step
  name both failed fast with exit code 1 and clear error messages.

---

## 9. Open items carried forward

- [x] `combiner-api` (the standalone serving container, not the orchestrator)
      had no `centralized_data` volume mount in `docker-compose.yml`, so its
      own live `load_client_types()` call was silently falling back to
      `comnbiner_script_final/data/client_final.csv` (never populated) →
      0 client types loaded at startup, no crash/error surfaced. Fixed:
      added the `centralized_data` volume mount and a `CLIENT_TYPE_FILE`
      override pointing at `centralized_data/Client.csv`, same as the
      orchestrator (8.4). Confirmed via container recreate + logs:
      `Loaded client data: 6228 clients` (was 0 before).
      **Separate, still-open gap found along the way:** the *sensitivity
      levels* loader (`DataLoader.load_sensitivity_levels()`, same file)
      reads `client_final.csv`/`chauff_final.csv`/`vehicle_final.csv` via
      bare relative paths (`Path('client_final.csv')`, not
      config/`CLIENT_TYPE_FILE`-driven), so it still warns "not found" and
      loads 0 sensitivity mappings — unrelated to this fix, not yet
      addressed.
- [x] `Schedular/` is now dockerized. Added `Schedular/Dockerfile` (same
      shape as `lightgcn_scripts/Dockerfile` — Debian slim + msodbcsql17 for
      pyodbc, non-root `appuser`; no `EXPOSE`/`HEALTHCHECK` since this is a
      `BlockingScheduler` process, not an HTTP server) and
      `Schedular/.dockerignore`. Added a `scheduler` service to
      `docker-compose.yml`, overriding `Schedular/.env`'s bare-host defaults
      the same way `orchestrator`/`combiner-api` override theirs:
      - `API_URL` → `http://combiner-api:8001/api/v1/batch-process` (service
        name instead of `127.0.0.1`)
      - `DB_CONNECTION_STRING` → `host.docker.internal` + `migration_svc` SQL
        auth, database `Q3_Recommendation` (matches the original `.env`
        value — `PORT_DB_NAME=TransElite.Q3` is a separate cross-database
        table qualifier used inside `load_ports()`'s query, not the
        connection's own database, so it was left unchanged)

      Verified live end-to-end (`docker compose build scheduler` →
      `docker compose up -d scheduler`): connected to the DB, loaded 14
      ports, ran the `UpdateBookingFeedVehicleType` stored procedure
      successfully, and — since `Schedular/.env` currently has
      `TEST_MODE=true` — two real port jobs (`SYD`, `MEL`) fired minutes
      later and both got HTTP 200 back from `combiner-api` over the compose
      network. Stopped the container afterward since test mode fires a job
      every ~15s indefinitely.
      **Before running this in production:** flip `TEST_MODE` to `false` in
      `Schedular/.env` (or override it per-environment) — left as `true`
      right now purely to make the smoke test finish in minutes instead of
      waiting for real 20:00-local port times.
