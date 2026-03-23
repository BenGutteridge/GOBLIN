"""
CityNetworks baselines: MeanGNN, GAT, GraphAny, and GOBLIN (Table 3).

Usage:
    python notebooks/run_city_exps.py
    python notebooks/run_city_exps.py --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
while not (ROOT / "goblin").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goblin.config import DATA_CACHE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CITIES  = ["CityParis", "CityShanghai", "CityLA", "CityLondon"]
GA_SEEDS = [0, 1, 2, 3]           # GraphAny: 4 seeds (paper uses these)
GNN_SEEDS = [0, 1, 2, 3, 4]       # GNN: 5 training seeds (paper uses these)

# GraphAny: results are stored per-run by graphany/run.py under data_cache/results/
GRAPHANY_RESULTS_DIR = DATA_CACHE / "results"

# GNN: results stored here (one JSON per city, already contains all seeds)
GNN_RESULTS_DIR    = ROOT / "results" / "gnn"
GOBLIN_RESULTS_DIR = ROOT / "results" / "goblin"
GOBLIN_NAME        = "canonical"
GOBLIN_SEEDS       = list(range(5))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], dry_run: bool, cwd: Path = ROOT):
    print("  $", " ".join(cmd))
    if not dry_run:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{os.environ.get('CONDA_PREFIX','')}/lib:{env.get('LD_LIBRARY_PATH','')}"
        result = subprocess.run(cmd, cwd=cwd, env=env)
        if result.returncode != 0:
            print(f"  WARNING: command exited with code {result.returncode}")


def graphany_result_exists(city: str, seed: int) -> bool:
    """Check if any result file exists for this city/seed combo in data_cache/results/."""
    d = GRAPHANY_RESULTS_DIR / f"Cora_X_{city}"
    if not d.exists():
        return False
    # graphany/run.py names results {dataset}_{feat}_{pred}_{hash}_results.json;
    # we check for any file that matches seed by loading hparams.
    for f in d.glob("*_results.json"):
        try:
            data = json.load(open(f))
            hp = data.get("hparams", data.get("cfg", {}))
            if int(hp.get("seed", -1)) == seed:
                return True
        except Exception:
            pass
    return False


def load_graphany_accs(city: str) -> list[float]:
    """Load all seed test accuracies for a city from the results cache."""
    d = GRAPHANY_RESULTS_DIR / f"Cora_X_{city}"
    if not d.exists():
        return []
    key = f"ind/{city.lower()[:4]}_test_acc"
    accs = []
    for f in sorted(d.glob("*_results.json")):
        try:
            data = json.load(open(f))
            hp = data.get("hparams", data.get("cfg", {}))
            if int(hp.get("seed", -1)) in GA_SEEDS:
                v = data.get(key)
                if v is not None:
                    accs.append(float(v))
        except Exception:
            pass
    return accs


def gnn_result_exists(model: str, city: str) -> bool:
    return (GNN_RESULTS_DIR / model / f"{city}.json").exists()


def load_gnn_accs(model: str, city: str):
    p = GNN_RESULTS_DIR / model / f"{city}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    tr = d["test_results"]
    return tr["mean_pct"], tr["std_pct"]


def load_goblin_accs(city: str) -> list[float]:
    """Load GOBLIN city test accs from multi-seed canonical results."""
    key_map = {
        "CityParis":    "CityParis",
        "CityShanghai": "CityShanghai",
        "CityLA":       "CityLA",
        "CityLondon":   "CityLondon",
    }
    import torch
    accs = []
    for seed in GOBLIN_SEEDS:
        p = GOBLIN_RESULTS_DIR / f"{GOBLIN_NAME}_seed{seed}.pt"
        if not p.exists():
            continue
        data = torch.load(p, map_location="cpu", weights_only=False)
        ev = data.get("eval", {})
        city_key = key_map[city]
        if city_key in ev:
            v = ev[city_key].get("test_acc")
            if v is not None:
                accs.append(float(v) * 100)
    return accs


# ---------------------------------------------------------------------------
# 1. GraphAny
# ---------------------------------------------------------------------------

def run_graphany(dry_run: bool):
    print("\n" + "=" * 60)
    print("GraphAny — Cora-pretrained, 4 seeds × 4 cities")
    print("=" * 60)
    skipped = 0
    for city in CITIES:
        for seed in GA_SEEDS:
            if graphany_result_exists(city, seed):
                skipped += 1
                continue
            print(f"\n  {city}  seed={seed}")
            run_cmd([
                "python", "graphany/run.py",
                f"dataset=Cora_X_{city}",
                f"seed={seed}",
                "max_khops=null",
                "max_output_nodes_samples=10",
                "prev_ckpt=checkpoints/graph_any_cora.pt",
                "total_steps=0",
                "feat_chn=X+L1+L2+H1+H2",
                "pred_chn=X+L1+L2",
            ], dry_run=dry_run)
    if skipped:
        print(f"  ({skipped}/{len(CITIES) * len(GA_SEEDS)} runs already cached — skipped)")


# ---------------------------------------------------------------------------
# 2. GNN baselines (MeanGNN + GAT)
# ---------------------------------------------------------------------------

def run_gnn_baselines(dry_run: bool):
    print("\n" + "=" * 60)
    print("GNN baselines — MeanGNN and GAT (L=2), 5 training seeds × 4 cities")
    print("Each run does a 4-config grid search then evaluates 5 seeds.")
    print("=" * 60)
    skipped = 0
    for model, num_layers in [("meangnn", 2), ("gat", 2)]:
        for city in CITIES:
            out_path = GNN_RESULTS_DIR / model / f"{city}.json"
            if out_path.exists():
                skipped += 1
                continue
            print(f"\n  {model} L={num_layers}  {city}")
            run_cmd([
                "python", "notebooks/train_eval_gnn_city.py",
                "--model", model,
                "--dataset", city,
                "--num_layers", str(num_layers),
            ], dry_run=dry_run)
    if skipped:
        print(f"  ({skipped}/{len(CITIES) * 2} runs already cached — skipped)")


# ---------------------------------------------------------------------------
# 3. Results table
# ---------------------------------------------------------------------------

def print_table():
    print("\n" + "=" * 70)
    print("CityNetworks results — mean ± std % accuracy over seeds")
    print("=" * 70)

    header = f"{'':12s}  {'MeanGNN':>14s}  {'GAT':>14s}  {'GraphAny':>14s}  {'GOBLIN':>14s}"
    print(header)
    print("-" * len(header))

    all_means = {m: [] for m in ["meangnn", "gat", "graphany", "goblin"]}

    for city in CITIES:
        label = city.replace("City", "")

        # MeanGNN
        gnn_m = load_gnn_accs("meangnn", city)
        gnn_g = load_gnn_accs("gat", city)
        ga_accs = load_graphany_accs(city)
        gob_accs = load_goblin_accs(city)

        def fmt(pair):
            if pair is None:
                return "     N/A      "
            return f"{pair[0]:5.2f} ± {pair[1]:4.2f}"

        def fmt_list(accs):
            if not accs:
                return "     N/A      "
            return f"{np.mean(accs):5.2f} ± {np.std(accs):4.2f}"

        print(f"{label:12s}  {fmt(gnn_m):>14s}  {fmt(gnn_g):>14s}  {fmt_list(ga_accs):>14s}  {fmt_list(gob_accs):>14s}")

        if gnn_m:   all_means["meangnn"].append(gnn_m[0])
        if gnn_g:   all_means["gat"].append(gnn_g[0])
        if ga_accs: all_means["graphany"].append(np.mean(ga_accs))
        if gob_accs: all_means["goblin"].append(np.mean(gob_accs))

    print("-" * len(header))
    def avg_fmt(model):
        vals = all_means[model]
        return f"{np.mean(vals):5.2f}" if vals else "  N/A "
    print(f"{'Average':12s}  {avg_fmt('meangnn'):>14s}  {avg_fmt('gat'):>14s}  "
          f"{avg_fmt('graphany'):>14s}  {avg_fmt('goblin'):>14s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--table_only", action="store_true",
                        help="Skip experiments, just print results table")
    args = parser.parse_args()

    if not args.table_only:
        run_graphany(dry_run=args.dry_run)
        run_gnn_baselines(dry_run=args.dry_run)

    if not args.dry_run:
        print_table()
