# Can Graph Foundation Models Generalize Over Architecture?

Code for reproducing results in the paper, submitted to the ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling.

### Setup
```
# CPU-only (uses pyenv)
export TORCH_VARIANT=cpu
pyenv install 3.10.13
pyenv local 3.10.13
python -m venv .venv
source .venv/bin/activate

# OR: CUDA 11.8 (uses conda)
export TORCH_VARIANT=cu118
conda create -n graphany python=3.10 -y
conda activate graphany

python -m pip install --upgrade pip setuptools wheel
pip install "setuptools<70" packaging
pip install \
  "torch==2.1.*" \
  torchvision \
  torchaudio \
  torchdata \
  --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}
pip install "numpy<2"
pip install torch-geometric \
  -f https://data.pyg.org/whl/torch-2.1.0+${TORCH_VARIANT}.html
if [ "$TORCH_VARIANT" = "cpu" ]; then
  pip install dgl
else
  pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
fi
pip uninstall -y torchdata
pip install torchdata==0.7.1 \
  --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}
if [ "$TORCH_VARIANT" = "cu118" ]; then
  conda install -y cudatoolkit=11.8 -c nvidia
fi

pip install \
  matplotlib \
  pandas \
  scikit-learn \
  pyyaml \
  "lightning==2.*" \
  pydantic \
  wandb \
  rich \
  hydra-core \
  einops \
  ogb \
  rootutils \
  hydra_colorlog \
  codetiming \
  humanfriendly
```

### Results
Results for GOBLIN (Table 2 and Table 1 rightmost column) can be reproduced using `notebooks/train_eval_goblin.py`, which will train a GOBLIN instance and evaluate it on given datasets (HopSign or benchmarks). Comment/uncomment eval datasets and vary hyperparameters as desired; the defaults are those used in the paper. 
```
python notebooks/train_eval_goblin.py
```

Results for GraphAny (Table 1) can be reproduced using `notebooks/hopsign_results_graphany.py`, which will train a GraphAny instance with given hyperparameters defined in the notebook and then evaluate on HopSign and real-world benchmarks. GraphAny code is adapted from [their repo](https://github.com/DeepGraphLearning/GraphAny).

**Notes** 
- Generating all-pairs shortest path distances can be time-consuming and memory-intensive, but they are cached
- Evaluation on larger graphs can be slow if not using a GPU; we used H100s for CoPhysics and Questions and A10s elsewhere
- GraphAny code caches computed operators, so can be time and disk memory intensive the first time it is run, especially for large eval graphs