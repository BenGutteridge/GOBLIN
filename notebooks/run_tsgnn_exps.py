"""
TS-GNN baselines — Figure 1, Table 3.

Trains TS-MeanGNN and TS-GAT on Cora, evaluates on HopSign (1–8) and CityNetworks.
Requires triton and a CUDA GPU.

Usage:
    python notebooks/run_tsgnn_exps.py
    python notebooks/run_tsgnn_exps.py --num_layers 16  # Table 5 (appendix)
    python notebooks/run_tsgnn_exps.py --dry_run
    python notebooks/run_tsgnn_exps.py --table_only

Set TSGNN_DATA_DIR for LS/dataset cache; TSGNN_MODELS_DIR to reuse checkpoints.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
while not (ROOT / "goblin").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsgnn.data import (load_hopsign, load_city, prepare_data, split_and_transform,
                        _make_transform)
from tsgnn.models.gfm import GFM, GFMArgs
from tsgnn.models.gnn_type import GNNType
from tsgnn.utils import set_seed, coo_to_csr, accuracy, SEEDS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from goblin.config import DATA_CACHE
HOPSIGN_CACHE = DATA_CACHE / "goblin_khopsign"

# LS cache and PyG dataset cache. Override with TSGNN_DATA_DIR env var.
TSGNN_DATA_DIR = Path(os.environ.get("TSGNN_DATA_DIR", str(ROOT / "data" / "tsgnn")))

# Saved model checkpoints. Override with TSGNN_MODELS_DIR env var.
TSGNN_MODELS_DIR = Path(os.environ.get("TSGNN_MODELS_DIR", str(ROOT / "ckpts" / "tsgnn")))

RESULTS_DIR = ROOT / "results" / "tsgnn"

# Model configs: (gnn_type, hid, num_layers, ls_layers, lp_ratio, epochs, lr)
CONFIGS = {
    "TS-MeanGNN": {
        "gnn_type": GNNType.MEAN_GNN,
        "hid_channel": 16,
        "ls_num_layers": 1,
        "lp_ratio": 0.4,
        "max_epochs": 2000,
        "lr": 0.01,
    },
    "TS-GAT": {
        "gnn_type": GNNType.GAT,
        "hid_channel": 16,
        "ls_num_layers": 1,
        "lp_ratio": 0.25,
        "max_epochs": 2000,
        "lr": 0.03,
    },
}

HOPSIGN_DATASETS = [f"hopsign{k}" for k in range(1, 9)]
CITY_DATASETS = ["paris", "shanghai", "la", "london"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def saved_model_dir(cfg: dict, num_layers: int) -> Path:
    """Construct the saved_models subdirectory name matching the original
    EquivarianceEverywhere naming convention:
        trainset1_1_{GNN_TYPE}_{hid}_{layers}_{ls_layers}_{lp}_{epochs}_{lr}
    """
    parts = [
        "trainset1", "1",
        cfg["gnn_type"].name,
        str(cfg["hid_channel"]),
        str(num_layers),
        str(cfg["ls_num_layers"]),
        str(cfg["lp_ratio"]),
        str(cfg["max_epochs"]),
        str(cfg["lr"]),
    ]
    return TSGNN_MODELS_DIR / "_".join(parts)


def load_dataset(ds_name: str) -> torch.Tensor:
    """Load a dataset by name. Returns prepared Data object."""
    if ds_name.startswith("hopsign"):
        k = int(ds_name.replace("hopsign", ""))
        data = load_hopsign(k, str(HOPSIGN_CACHE))
    elif ds_name in CITY_DATASETS:
        data = load_city(ds_name, str(TSGNN_DATA_DIR))
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")
    return prepare_data(data)


def evaluate_single(data, model, device, mask_str: str):
    """Run forward pass and compute accuracy for a given split mask."""
    model.eval()
    train_y = copy.deepcopy(data.y_mat)
    train_y[~data.train_mask] = 0
    with torch.no_grad():
        scores, _ = model(
            data.x, train_y=train_y,
            xy_conversions=data.xy_conversions, is_batch=False,
            device=device, edge_index=data.edge_index,
            rowptr=data.rowptr, indices=data.indices,
        )
    mask = getattr(data, f'{mask_str}_mask').to(device)
    loss = nn.CrossEntropyLoss()(scores[mask], data.y.to(device)[mask]).item()
    acc = accuracy(preds=scores[mask], targets=data.y.to(device)[mask])
    return loss, acc


def load_cora_base() -> object:
    """Load and pre-process Cora for TS-GNN training (matches EE trainset1 setup)."""
    from torch_geometric.datasets import Planetoid
    root = TSGNN_DATA_DIR / "cora"
    data = Planetoid(root=str(root), name='cora', transform=_make_transform())[0]
    return prepare_data(data)


def train_model_config(cfg: dict, num_layers: int, device):
    """Train a model config for all seeds on Cora. Saves to TSGNN_MODELS_DIR."""
    model_dir = saved_model_dir(cfg, num_layers)
    gfm_args = GFMArgs(
        gnn_type=cfg["gnn_type"],
        hid_channel=cfg["hid_channel"],
        num_layers=num_layers,
        ls_num_layers=cfg["ls_num_layers"],
        lp_ratio=cfg["lp_ratio"],
    )

    print(f"  Loading Cora for training...")
    cora_base = load_cora_base()

    for seed in SEEDS:
        model_path = model_dir / f"Seed{seed}.pt"
        if model_path.exists():
            print(f"  Seed {seed}: checkpoint exists, skipping")
            continue

        print(f"  Seed {seed}: training for {cfg['max_epochs']} epochs...")
        set_seed(seed)

        data = split_and_transform(
            cora_base, seed=seed, ls_num_layers=cfg["ls_num_layers"],
            dataset_name="cora", ls_cache_root=str(TSGNN_DATA_DIR),
        )
        rowptr, indices = coo_to_csr(
            data.edge_index[0], data.edge_index[1], num_nodes=data.x.shape[0])
        data.edge_index = []
        data.rowptr = rowptr
        data.indices = indices

        model = GFM(gfm_args=gfm_args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

        for epoch in tqdm(range(cfg["max_epochs"]), desc=f"Seed {seed}", leave=False):
            torch.cuda.empty_cache()
            # Train step
            model.train()
            optimizer.zero_grad()
            train_y = copy.deepcopy(data.y_mat)
            train_y[~data.train_mask] = 0
            scores, gt_mask = model(
                data.x, train_y=train_y,
                xy_conversions=data.xy_conversions, is_batch=True, device=device,
                edge_index=data.edge_index, rowptr=data.rowptr, indices=data.indices,
            )
            gt_mask = gt_mask.to(device)
            loss = nn.CrossEntropyLoss()(scores[gt_mask], data.y.to(device)[gt_mask])
            loss.backward()
            optimizer.step()

        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.cpu().state_dict(), model_path)
        print(f"  Seed {seed}: saved to {model_path}")
        torch.cuda.empty_cache()


def run_eval(ds_name: str, cfg: dict, num_layers: int, device, dry_run: bool):
    """Evaluate a model config on a dataset across all seeds. Returns per-seed test accs."""
    model_dir = saved_model_dir(cfg, num_layers)
    gfm_args = GFMArgs(
        gnn_type=cfg["gnn_type"],
        hid_channel=cfg["hid_channel"],
        num_layers=num_layers,
        ls_num_layers=cfg["ls_num_layers"],
        lp_ratio=cfg["lp_ratio"],
    )

    data_base = load_dataset(ds_name)
    per_seed_accs = []

    for seed in SEEDS:
        model_path = model_dir / f"Seed{seed}.pt"
        ls_cache_exists = (TSGNN_DATA_DIR / ds_name /
                           f"{data_base.x.shape[1]}feat" / f"seed{seed}").exists()

        if not model_path.exists():
            print(f"    Seed {seed}: model not found at {model_path}, skipping")
            continue
        if not ls_cache_exists and not dry_run:
            print(f"    Seed {seed}: LS cache not found for {ds_name}, will compute")

        if dry_run:
            print(f"    Seed {seed}: would evaluate {model_path}")
            continue

        set_seed(seed)
        data = split_and_transform(
            data_base, seed=seed, ls_num_layers=cfg["ls_num_layers"],
            dataset_name=ds_name, ls_cache_root=str(TSGNN_DATA_DIR),
        )

        # Convert to CSR for Triton kernels
        rowptr, indices = coo_to_csr(
            data.edge_index[0], data.edge_index[1], num_nodes=data.x.shape[0])
        data.edge_index = []
        data.rowptr = rowptr
        data.indices = indices

        model = GFM(gfm_args=gfm_args)
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)

        _, test_acc = evaluate_single(data, model, device, 'test')
        per_seed_accs.append(test_acc * 100)
        print(f"    Seed {seed}: test_acc = {test_acc * 100:.2f}%")
        torch.cuda.empty_cache()

    return per_seed_accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all(num_layers: int, dry_run: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not dry_run:
        print(f"Using device: {device}")
        print(f"Models dir: {TSGNN_MODELS_DIR}")
        print(f"LS cache dir: {TSGNN_DATA_DIR}")

    all_results = {}

    for model_name, cfg in CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"{model_name} (L={num_layers})")
        print(f"{'=' * 60}")

        model_dir = saved_model_dir(cfg, num_layers)
        if not model_dir.exists() or not any(model_dir.glob("Seed*.pt")):
            if dry_run:
                print(f"  Would train {model_name} on Cora (no checkpoints found)")
            else:
                print(f"  No checkpoints found — training {model_name} on Cora...")
                train_model_config(cfg, num_layers, device)

        results = {}

        # HopSign
        print(f"\n  HopSign 1-8:")
        for ds in HOPSIGN_DATASETS:
            k = ds.replace("hopsign", "")
            print(f"  {k}HopSign:")
            accs = run_eval(ds, cfg, num_layers, device, dry_run)
            if accs:
                results[ds] = {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "per_seed": accs,
                }

        # CityNetworks
        print(f"\n  CityNetworks:")
        for ds in CITY_DATASETS:
            city_label = f"City{ds.capitalize()}"
            print(f"  {city_label}:")
            accs = run_eval(ds, cfg, num_layers, device, dry_run)
            if accs:
                results[ds] = {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "per_seed": accs,
                }

        all_results[model_name] = results

    return all_results


def save_results(results: dict, num_layers: int):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_L{num_layers}" if num_layers != 2 else ""
    out_path = RESULTS_DIR / f"tsgnn_results{suffix}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out_path


def print_table(results: dict, num_layers: int):
    print(f"\n{'=' * 70}")
    print(f"TS-GNN Results (L={num_layers}) — test accuracy %")
    print(f"{'=' * 70}")

    header = f"{'Dataset':15s}"
    for model_name in results:
        header += f"  {model_name:>16s}"
    print(header)
    print("-" * len(header))

    # HopSign
    for k in range(1, 9):
        ds = f"hopsign{k}"
        row = f"{k}HopSign        "
        for model_name, model_results in results.items():
            if ds in model_results:
                r = model_results[ds]
                row += f"  {r['mean']:5.2f} ± {r['std']:4.2f}  "
            else:
                row += f"  {'N/A':>16s}"
        print(row)

    print("-" * len(header))

    # Cities
    for ds in CITY_DATASETS:
        city_label = f"City{ds.capitalize()}"
        row = f"{city_label:15s}"
        for model_name, model_results in results.items():
            if ds in model_results:
                r = model_results[ds]
                row += f"  {r['mean']:5.2f} ± {r['std']:4.2f}  "
            else:
                row += f"  {'N/A':>16s}"
        print(row)


def load_existing_results(num_layers: int):
    suffix = f"_L{num_layers}" if num_layers != 2 else ""
    path = RESULTS_DIR / f"tsgnn_results{suffix}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of GFM layers (2 for main results, 16 for Table 5)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without running")
    parser.add_argument("--table_only", action="store_true",
                        help="Just print results table from saved JSON")
    args = parser.parse_args()

    if args.table_only:
        results = load_existing_results(args.num_layers)
        if results is None:
            print("No saved results found. Run evaluation first.")
            sys.exit(1)
        print_table(results, args.num_layers)
        sys.exit(0)

    results = run_all(num_layers=args.num_layers, dry_run=args.dry_run)

    if not args.dry_run and results:
        out_path = save_results(results, args.num_layers)
        print_table(results, args.num_layers)
