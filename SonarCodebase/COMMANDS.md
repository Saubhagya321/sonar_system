# Command Reference

Every command needed to run the 4 recommendation APIs and their retraining
pipelines on this box — individually and as a master pipeline.

This replaces `global_readme.md`, which was written for a normal
`python -m venv` / `Scripts\Activate.ps1` setup. **That setup doesn't exist
here.** See [`INSTALLATION.md`](INSTALLATION.md) for why: this box only has
Python 3.14 and a Group Policy blocking MSI installers, so every environment
below is an **embeddable-Python zip** extracted by hand, not a real venv.
Two things follow from that, everywhere below:

- The interpreter is at the **venv root**, not in a `Scripts\` subfolder:
  `.\<venvname>\python.exe`, not `.\<venvname>\Scripts\python.exe`.
- Activation is `.\<venvname>\Activate.ps1` (a hand-written script, not the
  real-venv one) — there is no `python -m venv` step, ever. If a venv folder
  is ever missing/corrupt, rebuild it per `INSTALLATION.md` section
  "Setting up a new environment from scratch", not with `python -m venv`.

Repo root: `C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase`

## Module overview

| Module | Folder | Venv | Port | Framework | DB |
|---|---|---|---|---|---|
| Collaborative Filtering | `collabrative_filter_scripts` | `colabvenv` | 8000 | FastAPI/uvicorn | `Q3_Recommendation` |
| LightGCN | `lightgcn_scripts` | `gcnvenv` | 5000 | Flask | `TransElite.Q3` |
| Content-Based | `content_apis` | `contvenv` | 8003 | FastAPI/uvicorn | `TransElite.Q3` |
| Combiner (fusion) | `comnbiner_script_final` | `combvenv` | 8001 | Flask | `Q3_Recommendation` |
| Retraining Pipeline (master) | `retraining_pipeline` | `pipelinevenv` | — | — | `TransElite.Q3` |

`comnbiner_script_final` also has an older venv folder, `reqcomb` — unused,
`combvenv` is current. Don't delete `reqcomb` without checking nothing still
points at it.

---

## 1. Activating an environment

Two equivalent ways to use a module's venv — pick whichever suits the
situation:

```powershell
# A) Activate, then use bare `python` (interactive / debugging)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\Activate.ps1
python -m api.app
deactivate

# B) Call the venv's python.exe directly, no activation (scripts / services)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\python.exe -m api.app
```

All commands below use style (A) for readability; swap in style (B) if
you're scripting/automating a launch.

---

## 2. Run the APIs (individually)

### Collaborative Filtering API — port 8000
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\Activate.ps1
python -m api.app
# or, with auto-reload:
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Loads ~4M bookings + ~5.2M chauffeur-status rows at startup — first request
readiness can take several minutes.

### LightGCN API — port 5000
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\lightgcn_scripts
.\gcnvenv\Activate.ps1
python flask_api_server.py --host 0.0.0.0 --port 5000
# optional flags:
python flask_api_server.py --host 0.0.0.0 --port 5000 --debug
python flask_api_server.py --host 0.0.0.0 --port 5000 --model-path <path\to\model.pt>
```
Unpickles a multi-GB pipeline/model file at startup — make sure there's
enough free RAM before starting it alongside other heavy apps.

### Content-Based API — port 8003
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\content_apis
.\contvenv\Activate.ps1
$env:PYTHONUTF8 = '1'   # required — without it, a non-ASCII char in a log line 500s every request
python main.py
# or:
python -m uvicorn main:app --host 0.0.0.0 --port 8003
```

### Combiner API — port 8001
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\comnbiner_script_final
.\combvenv\Activate.ps1
python routes.py                       # dev server, port 8001 (or $env:PORT if set)

# production (gunicorn is Linux-only; use waitress on Windows):
pip install waitress
waitress-serve --host=0.0.0.0 --port=8001 routes:app
```

### Start all 4 together (4 terminals)
```powershell
# Terminal 1 — Collaborative Filtering (8000)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\Activate.ps1
python -m api.app

# Terminal 2 — LightGCN (5000)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\lightgcn_scripts
.\gcnvenv\Activate.ps1
python flask_api_server.py --host 0.0.0.0 --port 5000

# Terminal 3 — Content-Based (8003)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\content_apis
.\contvenv\Activate.ps1
$env:PYTHONUTF8 = '1'
python main.py

# Terminal 4 — Combiner (8001)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\comnbiner_script_final
.\combvenv\Activate.ps1
python routes.py
```
Start LightGCN before opening anything else memory-heavy (e.g. SSMS) — it
only needs the large working set during model load, not once it's serving.

---

## 3. Run training — individually, per module

### Collaborative Filtering
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\Activate.ps1

python orchestrator.py                                    # full pipeline, defaults
python orchestrator.py --retrain                           # force retrain, ignore cached model
python orchestrator.py --n_recommendations 200 --batch_size 300
python orchestrator.py --step 1                            # data loading only
python orchestrator.py --step 2                            # + preprocessing
python orchestrator.py --step 3                            # + model training
python orchestrator.py --step 4                            # generate recommendations only (needs steps 1-3 output already saved)
```

### LightGCN
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\lightgcn_scripts
.\gcnvenv\Activate.ps1

python orchestrator.py --mode full                          # full pipeline (train + save model)
python orchestrator.py --mode full --skip-training           # load existing model, skip training
python orchestrator.py --mode full --model-path <path.pt>     # custom model save/load path
python orchestrator.py --mode test                           # quick smoke test (PipelineTester.quick_test)
```
`--mode train_only` and `--mode recommendations_only` are accepted by the CLI
but are stubs / documented as "programmatic use only" in the code — they do
not run anything useful from the command line. Use `--mode full
--skip-training` to load-and-serve without retraining.

Equivalent one-liner used elsewhere in this repo (same `full` pipeline, called
as a function instead of via argparse):
```powershell
python -c "from orchestrator import run_complete_pipeline; run_complete_pipeline()"
```

### Content-Based
No standalone retraining CLI. The feature matrix
(`content_apis\data\content_data\feature_matrix\feature_matrix.npy`) is built
automatically the first time it's missing when the API loads. To force a
rebuild, delete the cached file and restart the API:
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\content_apis
Remove-Item .\data\content_data\feature_matrix\feature_matrix.npy -Confirm:$false
.\contvenv\Activate.ps1
$env:PYTHONUTF8 = '1'
python main.py     # rebuilds feature_matrix.npy on first request
```

### Combiner (weight optimization / grid search)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\comnbiner_script_final
.\combvenv\Activate.ps1

python grid_search.py          # monthly weight optimization, direct
python main.py gridsearch      # same thing via the CLI wrapper
python main.py health          # health check — pings all 4 module APIs, must be running first
python main.py quickstart      # prints a setup checklist
```

---

## 4. Run the whole retraining pipeline (master orchestrator)

`retraining_pipeline\main_orchestrator.py` is the single entry point that
drives all 3 model pipelines plus the combiner as one staged run: `db_pull` →
`feature_engineering` → `retrain_lightgcn` → `retrain_collaborative` →
`combiner_update` → (only if all 5 succeed) `cutover` → `signal_apis_reload`.
Everything is written to a `staging/YYYY_MM/` folder first; live-serving
files (models, weights, feature data) are only overwritten by the final
`cutover` step, so a failed run never takes down the running APIs.

```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\retraining_pipeline
.\pipelinevenv\Activate.ps1

python main_orchestrator.py --run-now                # run the full pipeline immediately
python main_orchestrator.py --run-now --dry-run       # walk through every step, write nothing
python main_orchestrator.py --schedule                # start APScheduler (runs 1st of Jan/Apr/Jul/Oct, 02:00 UTC)
python main_orchestrator.py --status                  # print raw version registry JSON
python main_orchestrator.py --list-versions            # print a table of stored versions
python main_orchestrator.py --rollback 2026_06        # restore live APIs to a previous version
python main_orchestrator.py --rollback initial        # restore the pre-retraining baseline
```

There is no per-step flag (no `--pipeline lightgcn` — that's from the old,
stale `global_readme.md`; the current script has no such option). To retrain
one model only, use the per-module commands in section 3 instead of the
master orchestrator, which always runs all steps together.

The `feature_engineering` folder (`content_FE.py`, `db_data_fetch.py`,
`user_item_FE.py`) has no standalone CLI — `main_orchestrator.py` loads these
modules directly as part of its `feature_engineering` step. There's nothing
to run there by hand.

---

## 5. Useful one-offs

```powershell
# Verify DB connectivity from any of the 5 venvs, before debugging application errors:
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase
.\collabrative_filter_scripts\colabvenv\python.exe test_db_connection.py
```
