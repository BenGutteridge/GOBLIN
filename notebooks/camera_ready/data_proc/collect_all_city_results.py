"""
Collect all CityNetworks results into camera_ready_exps_results/.

Produces:
  camera_ready_exps_results/
    all_results.json        -- full per-seed data for all models + datasets
    summary.csv             -- wide table: model x dataset, mean ± std

Models collected:
  - GraphAny (Cora pretrained, zero-shot)     -- 4 seeds = 4 split/init seeds
  - GOBLIN   (Cora trained, zero-shot)        -- 4 seeds = 4 operator search seeds
  - MeanGNN  L=2  (directly trained)          -- 4 training seeds, fixed splits
  - MeanGNN  L=16 (directly trained)          -- 4 training seeds, fixed splits
  - GAT      L=2  (directly trained, 8 heads) -- 4 training seeds, fixed splits
  - GAT      L=16 (directly trained, 8 heads) -- 4 training seeds, fixed splits
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_CACHE = Path("/data-gauss/bengut/GraphAny-Range/data_cache")
DATASETS = ["CityParis", "CityShanghai", "CityLA", "CityLondon"]
SEEDS = [0, 1, 2, 3]

OUT_DIR = ROOT / "camera_ready_exps_results"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_graphany():
    """
    Results: DATA_CACHE/results/Cora_X_{dataset}/*_results.json
    One JSON per (seed, dataset). Seed identified via hparams.seed.
    Key metric: ind/city_test_acc (percentage, already ×100).
    """
    model_results = {}
    for dataset in DATASETS:
        results_dir = DATA_CACHE / "results" / f"Cora_X_{dataset}"
        seed_accs = {}
        for json_path in sorted(results_dir.glob("*_results.json")):
            d = json.load(open(json_path))
            seed = d.get("hparams", {}).get("seed")
            if seed not in SEEDS:
                continue
            acc = d.get("ind/city_test_acc")
            if acc is not None:
                seed_accs[seed] = float(acc)
        model_results[dataset] = {
            "per_seed": seed_accs,
            "seeds_present": sorted(seed_accs.keys()),
        }
    return model_results


def collect_goblin():
    """
    Results: output/results/goblin/{hparam_hash}/{dataset}.pt
    One .pt per (seed, dataset). Seed identified via stored hparams.seed.
    Key metric: full_results[dataset]["test/acc"] (fraction → ×100).
    """
    import torch
    # Stub dgl to avoid CUDA library requirement on login node
    import types
    if "dgl" not in sys.modules:
        sys.modules["dgl"] = types.ModuleType("dgl")

    results_root = ROOT / "output" / "results" / "goblin"
    model_results = {ds: {"per_seed": {}} for ds in DATASETS}

    for hash_dir in sorted(results_root.iterdir()):
        if not hash_dir.is_dir():
            continue
        for pt_path in sorted(hash_dir.glob("*.pt")):
            dataset = pt_path.stem
            if dataset not in DATASETS:
                continue
            try:
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  [GOBLIN] Failed {pt_path.name}: {e}")
                continue
            seed = d.get("hparams", {}).get("seed")
            if seed not in SEEDS:
                continue
            acc = d.get(dataset, {}).get("test/acc")
            if acc is not None:
                model_results[dataset]["per_seed"][seed] = float(acc) * 100

    for ds in DATASETS:
        sd = model_results[ds]["per_seed"]
        model_results[ds]["seeds_present"] = sorted(sd.keys())
    return model_results


def collect_gnn(model_dir: str):
    """
    Results: output/results/gnn/{model_dir}/{dataset}.json
    Contains accs_pct (list, one per training seed, in seed order).
    """
    results_root = ROOT / "output" / "results" / "gnn" / model_dir
    model_results = {}
    for dataset in DATASETS:
        p = results_root / f"{dataset}.json"
        if not p.exists():
            model_results[dataset] = {"per_seed": {}, "seeds_present": [], "best_hparams": None, "notes": "missing"}
            continue
        d = json.load(open(p))
        tr = d["test_results"]
        accs = tr["accs_pct"]
        seed_list = tr["seeds"]
        model_results[dataset] = {
            "per_seed": {s: float(a) for s, a in zip(seed_list, accs)},
            "seeds_present": seed_list,
            "best_hparams": d.get("best_hparams"),
        }
    return model_results


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def compute_stats(per_seed: dict) -> dict:
    vals = list(per_seed.values())
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "n": len(vals),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODELS = {
    "GraphAny_Cora":  ("GraphAny (Cora)", collect_graphany,  None),
    "GOBLIN_Cora":    ("GOBLIN (Cora)",   collect_goblin,    None),
    "MeanGNN_L2":     ("MeanGNN L=2",     collect_gnn,       "meangnn_L2"),
    "MeanGNN_L16":    ("MeanGNN L=16",    collect_gnn,       "meangnn_L16"),
    "GAT_L2_H8":      ("GAT L=2 H=8",    collect_gnn,       "gat_L2_H8"),
    "GAT_L16_H8":     ("GAT L=16 H=8",   collect_gnn,       "gat_L16_H8"),
}

all_results = {}

print("Collecting results...")
for key, (label, fn, arg) in MODELS.items():
    print(f"  {label}...")
    data = fn(arg) if arg is not None else fn()
    # Attach stats
    for ds in DATASETS:
        if ds not in data:
            data[ds] = {"per_seed": {}, "seeds_present": []}
        data[ds]["stats"] = compute_stats(data[ds]["per_seed"])
    all_results[key] = {"label": label, "datasets": data}

# ---- Save JSON -------------------------------------------------------------
json_path = OUT_DIR / "all_results.json"
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved full results to {json_path}")

# ---- Save CSV --------------------------------------------------------------
import csv

csv_path = OUT_DIR / "summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    # Header
    seed_cols = [f"seed{s}" for s in SEEDS]
    writer.writerow(["model", "dataset"] + seed_cols + ["mean", "std", "n", "best_hparams"])
    for key, (label, _, _) in MODELS.items():
        for ds in DATASETS:
            ds_data = all_results[key]["datasets"][ds]
            per_seed = ds_data["per_seed"]
            stats = ds_data["stats"]
            seed_vals = [per_seed.get(s, "") for s in SEEDS]
            hparams = ds_data.get("best_hparams") or ""
            writer.writerow(
                [label, ds]
                + [f"{v:.6f}" if isinstance(v, float) else v for v in seed_vals]
                + [
                    f"{stats['mean']:.6f}" if stats["mean"] is not None else "",
                    f"{stats['std']:.6f}" if stats["std"] is not None else "",
                    stats["n"],
                    json.dumps(hparams) if hparams else "",
                ]
            )
print(f"Saved summary CSV to {csv_path}")

# ---- Print table -----------------------------------------------------------
W = 20
SEP = "=" * (14 + W * len(MODELS))
print(f"\n{SEP}")
print("CityNetworks test accuracy (%) — mean ± std over seeds")
print(SEP)
print(f"{'Dataset':<14}" + "".join(f"{all_results[k]['label']:<{W}}" for k in MODELS))
print("-" * (14 + W * len(MODELS)))
for ds in DATASETS:
    row = f"{ds:<14}"
    for key in MODELS:
        st = all_results[key]["datasets"][ds]["stats"]
        if st["mean"] is not None:
            row += f"{st['mean']:.2f} ± {st['std']:.2f}".ljust(W)
        else:
            row += f"{'N/A':<{W}}"
    print(row)
print(SEP)
