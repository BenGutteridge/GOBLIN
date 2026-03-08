"""
Collect citynetworks results for GOBLIN (Cora-trained) and GraphAny (Cora-pretrained)
across 4 seeds and 4 city datasets. Prints mean ± std table.

Usage:
    python notebooks/collect_citynetworks_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

DATA_CACHE = Path("/data-gauss/bengut/GraphAny-Range/data_cache")

DATASETS = ["CityParis", "CityShanghai", "CityLA", "CityLondon"]
SEEDS = [0, 1, 2, 3]


def collect_graphany(seeds=SEEDS, datasets=DATASETS):
    """
    Results: DATA_CACHE/results/Cora_X_{dataset}/*_results.json
    Key metric: `ind/city_test_acc`
    Each JSON's hparams.seed identifies the seed.
    """
    results = {s: {} for s in seeds}
    for dataset in datasets:
        results_dir = DATA_CACHE / "results" / f"Cora_X_{dataset}"
        if not results_dir.exists():
            print(f"[GraphAny] Missing: {results_dir}")
            continue
        for json_path in sorted(results_dir.glob("*_results.json")):
            with open(json_path) as f:
                d = json.load(f)
            seed = d.get("hparams", {}).get("seed")
            if seed not in seeds:
                continue
            acc = d.get("ind/city_test_acc")
            if acc is not None:
                results[seed][dataset] = acc
    return results


def collect_goblin(seeds=SEEDS, datasets=DATASETS):
    """
    Results: output/results/goblin/{hparam_hash}/{dataset}.pt
    Key metric: full_results[dataset]["test/acc"] (as fraction, multiply by 100)
    """
    import torch

    results_root = ROOT / "output" / "results" / "goblin"
    results = {s: {} for s in seeds}

    for hash_dir in sorted(results_root.iterdir()):
        if not hash_dir.is_dir():
            continue
        for pt_path in sorted(hash_dir.glob("*.pt")):
            dataset = pt_path.stem
            if dataset not in datasets:
                continue
            try:
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[GOBLIN] Failed to load {pt_path}: {e}")
                continue
            seed = d.get("hparams", {}).get("seed")
            if seed not in seeds:
                continue
            ds_result = d.get(dataset, {})
            acc = ds_result.get("test/acc")
            if acc is not None:
                results[seed][dataset] = round(float(acc) * 100, 2)

    return results


def print_table(ga, go, seeds=SEEDS, datasets=DATASETS):
    def fmt(accs_by_seed):
        valid = [accs_by_seed[s] for s in seeds if accs_by_seed.get(s) is not None]
        if not valid:
            return "N/A"
        mean, std = np.mean(valid), np.std(valid)
        tag = f" ({len(valid)}/{len(seeds)})" if len(valid) < len(seeds) else ""
        return f"{mean:.1f} ± {std:.1f}{tag}"

    print("\n" + "=" * 66)
    print("CityNetworks test accuracy (%), mean ± std over 4 seeds")
    print("Trained on Cora, zero-shot transfer to city networks")
    print("=" * 66)
    print(f"{'Dataset':<16}{'GraphAny (Cora)':<26}{'GOBLIN (Cora)':<24}")
    print("-" * 66)
    for ds in datasets:
        ga_by_seed = {s: ga[s].get(ds) for s in seeds}
        go_by_seed = {s: go[s].get(ds) for s in seeds}
        print(f"{ds:<16}{fmt(ga_by_seed):<26}{fmt(go_by_seed):<24}")
    print("=" * 66)

    # Overall mean across datasets
    ga_all = [ga[s].get(ds) for s in seeds for ds in datasets if ga[s].get(ds) is not None]
    go_all = [go[s].get(ds) for s in seeds for ds in datasets if go[s].get(ds) is not None]
    print(f"{'Mean':<16}{np.mean(ga_all):.1f} ± {np.std(ga_all):.1f}{'':>10}{np.mean(go_all):.1f} ± {np.std(go_all):.1f}")
    print("=" * 66)

    print("\nPer-seed breakdown:")
    header = f"{'Dataset':<14}" + "".join(f"{'GA s'+str(s):<9}" for s in seeds) + "  " + "".join(f"{'GO s'+str(s):<9}" for s in seeds)
    print(header)
    print("-" * len(header))
    for ds in datasets:
        ga_row = "".join(f"{ga[s].get(ds, 'N/A')!s:<9}" for s in seeds)
        go_row = "".join(f"{go[s].get(ds, 'N/A')!s:<9}" for s in seeds)
        print(f"{ds:<14}{ga_row}  {go_row}")


if __name__ == "__main__":
    print("Collecting GraphAny results...")
    ga = collect_graphany()
    print("Collecting GOBLIN results...")
    go = collect_goblin()
    print_table(ga, go)

    ga_count = sum(v is not None for sv in ga.values() for v in sv.values())
    go_count = sum(v is not None for sv in go.values() for v in sv.values())
    total = len(SEEDS) * len(DATASETS)
    print(f"\nGraphAny: {ga_count}/{total} results  |  GOBLIN: {go_count}/{total} results")
