# 👺 Graph Operator Basis Learning & Inference (GOBLIN) 🧌

Code for the paper **Can Graph Foundation Models Generalize Over Architecture?** (ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling).

---

## Setup

```bash
export TORCH_VARIANT=cu118   # or cpu
conda create -n goblin python=3.10 -y && conda activate goblin

python -m pip install --upgrade pip setuptools wheel
pip install "setuptools<70" packaging
pip install "torch==2.1.*" torchvision torchaudio torchdata \
  --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}
pip install "numpy<2"
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+${TORCH_VARIANT}.html
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html   # omit -f flag for cpu
pip uninstall -y torchdata
pip install torchdata==0.7.1 --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}
conda install -y cudatoolkit=11.8 -c nvidia   # omit for cpu

pip install matplotlib pandas scikit-learn pyyaml "lightning==2.*" pydantic wandb rich \
  hydra-core einops ogb rootutils hydra_colorlog codetiming humanfriendly

# May need to run this on new terminals
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Reproducing results

Pre-computed results are included in `results/`. To generate all paper figures and tables immediately:

```bash
python notebooks/get_figures_and_tables.py
```

To re-run experiments from scratch:

```bash
# 0. Pre-compute CityNetwork shortest-path distances (required before city experiments)
python notebooks/compute_citynetwork_khops.py

# 1. GOBLIN — one call per seed (0–4); GPU required for most datasets
python notebooks/run_goblin_exps.py --seed 0

# 2. GraphAny operator variants
python notebooks/run_graphany_exps.py

# 3. MeanGNN, GAT, GraphAny on CityNetworks
python notebooks/run_city_exps.py

# 4. TS-GNN (GPU + triton required; trains on Cora then evaluates)
python notebooks/run_tsgnn_exps.py

# 5. Figures and tables
python notebooks/get_figures_and_tables.py
python notebooks/get_range_figures.py
```

| Script | Produces |
|---|---|
| `run_goblin_exps.py` | Tables 1–3, Figure 1 (GOBLIN column) |
| `run_graphany_exps.py` | Table 1, Figure 1 (GraphAny variants) |
| `run_city_exps.py` | Table 3 (MeanGNN, GAT, GraphAny) |
| `run_tsgnn_exps.py` | Table 3 and Figure 1 (TS-GAT), Table 5 with `--num_layers 16` |
| `get_range_figures.py` | Figure 2 |
| `get_figures_and_tables.py` | All figures and tables |

**Notes:** All-pairs shortest-path and operator caches are computed on first use and cached automatically. GraphAny's operator cache can be disk-intensive on first run.

**TS-GNN:** `tsgnn/` contains the minimal [EquivarianceEverywhere](https://github.com/Saro00/EquivarianceEverywhere) code needed to reproduce results. It uses Triton kernels and requires a CUDA GPU. Set `TSGNN_DATA_DIR` to a directory for dataset/LS caches, and optionally `TSGNN_MODELS_DIR` to reuse pre-trained checkpoints.

---

## Exploration

`notebooks/train_eval_goblin.py` runs GOBLIN with configurable hyperparameters; defaults match the paper config.

```bash
python notebooks/train_eval_goblin.py
python notebooks/train_eval_goblin.py --eval_dataset Chameleon --basis_size 5
```

GraphAny experiments can be run directly via `graphany/run.py` (Hydra). GraphAny code is adapted from [their repository](https://github.com/DeepGraphLearning/GraphAny).
