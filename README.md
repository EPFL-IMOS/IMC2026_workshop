# IMC Workshop Notebooks

This repository contains two independent notebook projects. Each project uses
its own `uv` environment.

## GNN

```bash
cd gnn
uv sync
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --output 01_gnn_basics_solution.evaluated.ipynb \
  01_gnn_basics_solution.ipynb
```

The exercise notebook contains incomplete tasks and will only run fully after
they have been completed.

## PICID

```bash
cd picid
uv sync
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --output 00_cstr_degradation_simulator.evaluated.ipynb \
  00_cstr_degradation_simulator.ipynb

uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --output 01_picid_rul_experiment.evaluated.ipynb \
  01_picid_rul_experiment.ipynb
```

Each command writes an evaluated copy and does not overwrite the source
notebook.
