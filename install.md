# Installation

## Install the UV Package Manager
In this workshop, we use the uv package manager.
Please install [UV](https://docs.astral.sh/uv/) first.

You need [Git](https://github.com/git-guides/install-git) to clone (to your computer) the repository.

We recommend to use [VSCode](https://code.visualstudio.com/download) as a coding environment.

## Install the GNN environment
Each sub-project has its own environment. Sync the locked GNN environment from
the `gnn` directory:

```bash
cd gnn
uv sync
```

In VSCode, select `gnn/.venv/bin/python` as the notebook kernel. Commands can
also be run without activating the environment by prefixing them with `uv run`.

## Install the PICID environment
Sync the independent PICID environment from its sub-project directory:

```bash
cd picid
uv sync
```

In VSCode, select `picid/.venv/bin/python` as the notebook kernel.
