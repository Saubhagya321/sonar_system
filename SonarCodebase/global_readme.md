# Sonar Recommendation System — Global Command Reference

This file lists every command needed to set up, retrain, and serve the four recommendation pipelines in this repo, plus the master pipeline that ties them together. All commands are PowerShell, run from the repo root unless a `cd` is shown.

Repo root: `C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase`

Modules:
| Module | Folder | Model | API framework | Existing venv |
|---|---|---|---|---|
| Collaborative Filtering | `collabrative_filter_scripts` | Ensemble CF | FastAPI | `colabvenv` |
| LightGCN | `lightgcn_scripts` | Heterogeneous GNN | Flask | `gcnvenv` |
| Content-Based | `content_apis` | Item/user content similarity | FastAPI | `contvenv` |
| Combiner (fusion) | `comnbiner_script_final` | RRF + weighted fusion of the 3 above | Flask | `combvenv` (current), `reqcomb` (older, still present) |
| Retraining Pipeline (master) | `retraining_pipeline` | Orchestrates the 3 training pipelines above | — | none yet, create one |

---

## 1. Navigate (cd) commands

```powershell
# Repo root
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase

# Individual modules
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\lightgcn_scripts
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\content_apis
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\comnbiner_script_final
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\retraining_pipeline
```

---

## 2. Virtual environment setup

Each module has its own venv already created. Activate the existing one, or recreate it from `requirements.txt` if it's missing/broken.

### Collaborative Filtering (`colabvenv`)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\Scripts\Activate.ps1
# If the venv doesn't exist yet:
python -m venv colabvenv
.\colabvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### LightGCN (`gcnvenv`)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\lightgcn_scripts
.\gcnvenv\Scripts\Activate.ps1
# If the venv doesn't exist yet:
python -m venv gcnvenv
.\gcnvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Content-Based (`contvenv`)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\content_apis
.\contvenv\Scripts\Activate.ps1
# If the venv doesn't exist yet:
python -m venv contvenv
.\contvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Combiner (`combvenv`)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\comnbiner_script_final
.\combvenv\Scripts\Activate.ps1
# If the venv doesn't exist yet:
python -m venv combvenv
.\combvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Note: an older venv named `reqcomb` also exists in this folder. `combvenv` is the current one — don't delete `reqcomb` without checking it isn't still referenced somewhere.

### Retraining Pipeline (master orchestrator) — no venv exists yet
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\retraining_pipeline
python -m venv pipelinevenv
.\pipelinevenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 3. Run retraining — individually, per module

### Collaborative Filtering
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\Scripts\Activate.ps1

python orchestrator.py                                   # full pipeline, defaults
python orchestrator.py --retrain                          # force retrain, ignore cached model
python orchestrator.py --n_recommendations 200 --batch_size 300
python orchestrator.py --step 1                           # data loading only
python orchestrator.py --step 2                           # preprocessing only
python orchestrator.py --step 3                           # model training only
python orchestrator.py --step 4                           # generate recommendations only
```

### LightGCN
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\lightgcn_scripts
.\gcnvenv\Scripts\Activate.ps1

python -c "from orchestrator import run_complete_pipeline; run_complete_pipeline()"
```

### Content-Based
There is no standalone retraining CLI for this module. The feature matrix (`content_apis\data\content_data\feature_matrix\feature_matrix.npy`) is built automatically the first time it's missing when the API loads a model. To force a rebuild, delete the cached file and restart the API:
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\content_apis
Remove-Item .\data\content_data\feature_matrix\feature_matrix.npy -Confirm:$false
.\contvenv\Scripts\Activate.ps1
python main.py     # rebuilds feature_matrix.npy on first request
```

### Combiner (weight optimization / grid search)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\comnbiner_script_final
.\combvenv\Scripts\Activate.ps1

python grid_search.py          # monthly weight optimization, direct
python main.py gridsearch      # same thing via the CLI wrapper
python main.py health          # health check
python main.py quickstart      # setup checklist
```

---

## 4. Run APIs — individually, per module

### Collaborative Filtering API (FastAPI, port 8000)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\Scripts\Activate.ps1
python -m api.app
# or:
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### LightGCN API (Flask, port 5000)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\lightgcn_scripts
.\gcnvenv\Scripts\Activate.ps1
python flask_api_server.py --host 0.0.0.0 --port 5000
# optional debug mode:
python flask_api_server.py --host 0.0.0.0 --port 5000 --debug
```

### Content-Based API (FastAPI, port 8003)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\content_apis
.\contvenv\Scripts\Activate.ps1
python main.py
# or:
python -m uvicorn main:app --host 0.0.0.0 --port 8003
```

### Combiner API (Flask, port 8001)
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\comnbiner_script_final
.\combvenv\Scripts\Activate.ps1
python routes.py                       # Flask dev server on port 8001 (or $env:PORT if set)

# production (gunicorn is Linux-only; on Windows use waitress instead):
pip install waitress
waitress-serve --host=0.0.0.0 --port=8001 routes:app
```

---

## 5. Run the whole retraining pipeline (global / master orchestrator)

`retraining_pipeline\sub_orchestrator.py` is the master entry point. It dynamically loads and runs the collaborative, lightgcn, and combiner orchestrators from one place.

```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\retraining_pipeline
.\pipelinevenv\Scripts\Activate.ps1

python sub_orchestrator.py --list                                   # show registered pipelines

python sub_orchestrator.py --pipeline lightgcn                      # run lightgcn only
python sub_orchestrator.py --pipeline lightgcn --skip-training      # load existing model, skip training

python sub_orchestrator.py --pipeline collaborative --retrain
python sub_orchestrator.py --pipeline collaborative --n-recommendations 50 --batch-size 100

python sub_orchestrator.py --pipeline combiner                      # runs grid_search.run_monthly_optimization
```

Run all three sequentially for a full end-to-end retrain:
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\retraining_pipeline
.\pipelinevenv\Scripts\Activate.ps1

python sub_orchestrator.py --pipeline lightgcn
python sub_orchestrator.py --pipeline collaborative --retrain
python sub_orchestrator.py --pipeline combiner
```

Optional: run the shared feature-engineering step first (DB fetch + content features + user-item features) before the three model pipelines:
```powershell
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\retraining_pipeline\feature_engineering
python orchestrator.py
```

**Known bug to fix before relying on `--pipeline lightgcn`:** in `sub_orchestrator.py`, `PIPELINE_REGISTRY["lightgcn"]["module"]` is set to `"orchestator"` (a typo, missing the second `r`), but the real file on disk is `lightgcn_scripts\orchestrator.py`. As written today, `python sub_orchestrator.py --pipeline lightgcn` will raise `FileNotFoundError`. Until that's fixed, use the direct LightGCN command from section 3 instead.

---

## 6. Start all APIs together (4 separate terminals)

```powershell
# Terminal 1 — Collaborative Filtering API (port 8000)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\collabrative_filter_scripts
.\colabvenv\Scripts\Activate.ps1
python -m api.app

# Terminal 2 — LightGCN API (port 5000)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\lightgcn_scripts
.\gcnvenv\Scripts\Activate.ps1
python flask_api_server.py --host 0.0.0.0 --port 5000

# Terminal 3 — Content-Based API (port 8003)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\content_apis
.\contvenv\Scripts\Activate.ps1
python main.py

# Terminal 4 — Combiner API (port 8001)
cd C:\Users\Q3Admin\Desktop\SonarCodebase\SonarCodebase\comnbiner_script_final
.\combvenv\Scripts\Activate.ps1
python routes.py
```
