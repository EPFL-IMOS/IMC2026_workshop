# Installation

The `gnn` and `picid` directories are independent Python projects. Run the
commands below from the repository root; do not create a shared root
environment.

## 1. Install the prerequisites

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Confirm that uv is available:

```bash
uv --version
```

## 2. Install the project environments

Sync both environments from their committed lockfiles:

```bash
(cd gnn && uv sync --locked)
(cd picid && uv sync --locked)
```

uv installs the required Python versions and creates these environments:

- `gnn/.venv` using Python 3.11.12
- `picid/.venv` using Python 3.12

## 3. Select a notebook kernel

In VS Code or another notebook editor, select the environment belonging to the
notebook:

- GNN: `gnn/.venv/bin/python`
- PICID: `picid/.venv/bin/python`

On Windows, the equivalent paths end in `.venv\Scripts\python.exe`.

See [README.md](README.md) for commands that execute the notebooks into
separate `*.evaluated.ipynb` files.
