# Installation Guide

This repo has 5 independent Python sub-projects, each with its own isolated
environment:

| Project | Env folder | DB target |
|---|---|---|
| `collabrative_filter_scripts` | `colabvenv` | `Q3_Recommendation` |
| `lightgcn_scripts` | `gcnvenv` | `TransElite.Q3` (role: `booking`) |
| `content_apis` | `contvenv` | `TransElite.Q3` |
| `comnbiner_script_final` | `combvenv` | `Q3_Recommendation` |
| `retraining_pipeline` | `pipelinevenv` | `TransElite.Q3` |

## Why these aren't normal venvs

This machine only has Python 3.14 installed, and a Group Policy on this
machine blocks all MSI-based installers (`winget`, the official python.org
installer — both fail with error `0x80070659`, "forbidden by system policy").
That ruled out installing Python 3.12 the normal way, and Python 3.14 is too
new for several pinned scientific packages (e.g. `numpy==1.26.2` has no
wheels for it).

The workaround: the **embeddable zip** distribution of Python is just a zip
of files — no installer, no MSI, nothing Windows Installer touches. Each
project's env folder above is a full, independent, self-contained Python
3.12.10 install produced by extracting that zip — not a `venv`-module
virtual environment (a real `venv` needs `ensurepip`, which the embeddable
distribution deliberately omits).

Practically, this means:
- No system-wide Python was installed or changed.
- Each project's env folder (`colabvenv`, `gcnvenv`, etc.) contains a
  complete copy of the Python runtime plus its own `Lib\site-packages`.
- There's no `Scripts\Activate.ps1` like a real venv — instead each env
  folder has an `Activate.ps1` we wrote by hand (see below).

## Using an existing environment

```powershell
cd <project_folder>          # e.g. collabrative_filter_scripts
.\<venvname>\Activate.ps1    # e.g. .\colabvenv\Activate.ps1
python your_script.py
deactivate
```

While active, `python`, `pip`, and any installed console scripts (`uvicorn`,
`flask`, `jupyter`, etc.) resolve to that project's environment. `deactivate`
restores your normal `PATH` and prompt.

## Setting up a new environment from scratch

Use this if you need to rebuild one of the 5 envs, or add a 6th project.

```powershell
# 1. Download the embeddable zip and pip's bootstrap script
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" -OutFile "python312-embed.zip"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "get-pip.py"

# 2. Extract into the project's venv folder
Expand-Archive -Path "python312-embed.zip" -DestinationPath "<project>\<venvname>" -Force
```

3. **Enable site-packages support** (required for pip to work at all). Edit
   `<project>\<venvname>\python312._pth` and change:
   ```
   #import site
   ```
   to:
   ```
   import site
   ```

```powershell
# 4. Bootstrap pip (embeddable Python ships without pip or ensurepip)
& "<project>\<venvname>\python.exe" "get-pip.py" --no-warn-script-location

# 5. Install build tools (needed for any package with no prebuilt wheel)
& "<project>\<venvname>\python.exe" -m pip install setuptools wheel

# 6. Install the project's actual dependencies
& "<project>\<venvname>\python.exe" -m pip install -r "<project>\requirements.txt"
```

7. **Fix local-package imports.** The embeddable distribution's `._pth` file
   puts the interpreter in an "isolated" mode that does **not** add the
   running script's own directory to `sys.path`, and **ignores `PYTHONPATH`
   entirely** — both things a normal Python install does automatically. Any
   project using local imports (e.g. `from config.config import Config`)
   will fail with `ModuleNotFoundError` without this fix. Create
   `<project>\<venvname>\Lib\site-packages\sitecustomize.py`:
   ```python
   import os
   import sys

   if sys.argv and sys.argv[0]:
       _script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
       if _script_dir and _script_dir not in sys.path:
           sys.path.insert(0, _script_dir)
   ```
   `site.py` auto-imports `sitecustomize.py` on interpreter startup, so this
   restores normal CPython import behavior transparently for every script.

8. **(Optional) Copy `Activate.ps1`** from one of the existing venv folders
   (e.g. `collabrative_filter_scripts\colabvenv\Activate.ps1`) into the new
   one, so you get `.\<venvname>\Activate.ps1` / `deactivate` like a real venv.
   It works unmodified — it derives the venv name from its own folder name.

## Database configuration (`.env` files)

Each project reads `DB_CONNECTION_STRING` from an `.env` file in its own root
folder (loaded via `python-dotenv`). All 5 currently use **Windows
Authentication** against a local SQL Server instance:

```
DB_CONNECTION_STRING=mssql+pyodbc://localhost/<database>?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes
```

- `collabrative_filter_scripts/.env`, `comnbiner_script_final/.env` → `Q3_Recommendation`
- `content_apis/.env`, `retraining_pipeline/.env` → `TransElite.Q3`
- `lightgcn_scripts/.env` → `DB_SERVER=localhost` + `DB_TRUSTED_CONNECTION=true`
  (builds its own connection string per DB role via `get_db_connection_string()`)

A standalone connectivity check script is at the repo root:
`test_db_connection.py` — run it with any of the 5 venvs' `python.exe` to
verify the DB is reachable before debugging application-level issues.

