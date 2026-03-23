# %% [markdown]
# # GOBLIN — Tables 1–3, Figure 1
#
# Trains on Cora, evaluates on HopSign (1–8), 25 benchmarks, and CityNetworks.
# Run once per seed (0–4); GPU required.
#
# Usage:
#   python notebooks/run_goblin_exps.py --seed 0
#   python notebooks/run_goblin_exps.py --seed 0 --name my_run
#
# Results:     results/goblin/{name}_seed{N}.pt
# Checkpoints: ckpts/goblin/{name}/seed{N}.pt

# %% [markdown]
# ## Imports and paths

# %%
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path(".").resolve()
sys.path.insert(0, str(ROOT))

from goblin import GOBLIN, OperatorSearchConfig, ExpertsDeepSetConfig, MultiSearchConfig
from goblin.config import DATA_CACHE
from goblin.data import (
    load_graph_dataset,
    build_and_cache_distance_operators,
    build_and_cache_city_distance_operators,
    build_hopsign_dataset,
    compute_mean_spd,
    recommended_max_dist,
)

# %% [markdown]
# ## Hyperparameters — loaded from notebooks/goblin_hparams.json

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Training seed (0–4)")
parser.add_argument("--name", type=str, default="canonical",
                    help="Name for output files, e.g. 'canonical' (default) or a custom label")
parser.add_argument("--hparams", type=str, default=None,
                    help="Path to a custom hparams JSON file (default: notebooks/goblin_hparams.json)")
args, _ = parser.parse_known_args()

configs_path = Path(args.hparams) if args.hparams else ROOT / "notebooks" / "goblin_hparams.json"
with open(configs_path) as f:
    p = json.load(f)

name = args.name
seed = args.seed

print(f"Run name: {name}")
print(f"Seed: {seed}")
print(f"Training mode: {p['training_mode']}  basis_size={p['basis_size']}  sigma={p['sigma']}")

# %% [markdown]
# ## Datasets to evaluate

# %%
CITY_DATASETS = {"CityParis", "CityShanghai", "CityLA", "CityLondon"}
HOPSIGN_DATASETS = [f"{k}HopSign" for k in range(1, 9)]

EVAL_DS = [
    # Synthetic: k-hop sign classification
    "1HopSign", "2HopSign", "3HopSign", "4HopSign",
    "5HopSign", "6HopSign", "7HopSign", "8HopSign",
    # Standard benchmarks (25 total)
    "AirBrazil", "AirUS", "AirEU",
    "Cornell", "Texas", "Wisconsin",
    "Chameleon", "Wiki", "Squirrel", "Actor",
    "Citeseer", "BlogCatalog", "WkCS",
    "Tolokers", "AmzComp", "AmzPhoto",
    "Minesweeper", "DBLP", "CoCS",
    "Pubmed", "FCora", "Roman", "AmzRatings",
    "CoPhysics", "Questions",  # large; may need CPU (see note above)
    # Long-range city tasks
    "CityParis", "CityShanghai", "CityLA", "CityLondon",
]

lingauss_cache_dir = DATA_CACHE / "apspd/lingauss"
hopsign_cache_dir  = DATA_CACHE / "goblin_khopsign"

# %% [markdown]
# ## Output paths

# %%
out_dir  = ROOT / "results" / "goblin"
ckpt_dir = ROOT / "ckpts" / "goblin" / name
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

results_path = out_dir / f"{name}_seed{seed}.pt"
ckpt_path    = ckpt_dir / f"seed{seed}.pt"
print(f"Results: {results_path}")

# %% [markdown]
# ## GPU memory check

# %%
if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"GPU free: {free_gb:.1f} GB")
    if free_gb < 4.0:
        print("Less than 4 GB free — exiting.")
        sys.exit(1)
else:
    print("No GPU detected — running on CPU.")

# %% [markdown]
# ## Config helpers

# %%
def resolve_bounds(mean_spd: float):
    sf_mu  = p["spd_scale_factor_mu"]
    sf_tau = p["spd_scale_factor_tau"]
    mu_max      = sf_mu  * mean_spd if sf_mu  > 0 else p["mu_max"]
    tausqrt_max = sf_tau * mean_spd if sf_tau > 0 else p["tausqrt_max"]
    return mu_max, tausqrt_max


def make_configs(mu_max: float, tausqrt_max: float):
    op_cfg = OperatorSearchConfig(
        families=["heat", "gaussian"],
        bo_objective=p["bo_objective"],
        ucb_beta=p["ucb_beta"],
        mu_min=p["mu_min"], mu_max=mu_max, mu_num=p["mu_num"],
        tausqrt_min=p["tausqrt_min"], tausqrt_max=tausqrt_max, tausqrt_num=p["tausqrt_num"],
        rbf_length_scale=p["rbf_length_scale"],
        white_noise=p["white_noise"],
    )
    multi_cfg = MultiSearchConfig(
        n_samples=p["n_samples"],
        basis_size=p["basis_size"],
        enforce_family_coverage=p["enforce_family_coverage"],
        include_fixed_ops=["L2"] if p.get("num_fixed_operators", 0) > 0 else [],
        basis_selection_rule=p["basis_selection_rule"],
        diversity_lambda=p["diversity_lambda"],
        mu_anchors=p["mu_anchors"],
        tausqrt_anchors=p["tausqrt_anchors"],
    )
    deepset_cfg = ExpertsDeepSetConfig(
        hidden_dim=p["hidden_dim"],
        attn_temp=p["attn_temp"],
        num_deepset_layers=p["num_deepset_layers"],
        num_head_layers=p["num_head_layers"],
        epochs=p["epochs"],
        lr=p["lr"],
        dropout=p["dropout"],
        score_feature=p["score_feature"],
        weight_selection=p["weight_selection"],
        feature_set_size=p["feature_set_size"],
        n_batches=p["n_batches"],
        batch_size=p["batch_size"],
        n_ref_per_class=p["n_ref_per_class"],
        pool_size=p["pool_size"],
        n_ops_per_batch=p["n_ops_per_batch"],
    )
    return op_cfg, multi_cfg, deepset_cfg


def prep_benchmark(ds: str, split_seed: int = None):
    if split_seed is None:
        split_seed = 0 if ds in CITY_DATASETS else seed
    data, X, apd, y_class, y_onehot, splits, C = load_graph_dataset(
        name=ds, root=ROOT / "data/goblin", seed=split_seed,
        compute_all_pairs_dist=(ds not in CITY_DATASETS),
    )
    if ds in CITY_DATASETS:
        build_and_cache_city_distance_operators(
            city_name=ds, cache_dir=lingauss_cache_dir / ds, max_k=20,
        )
    elif apd is not None:
        build_and_cache_distance_operators(
            apd, max_dist=recommended_max_dist(compute_mean_spd(apd)),
            cache_dir=lingauss_cache_dir / ds,
        )
    mean_spd = compute_mean_spd(apd, cache_dir=lingauss_cache_dir / ds if apd is None else None)
    return data, X, apd, y_class, y_onehot, splits, C, mean_spd


def prep_hopsign(k: int):
    ds_name = f"{k}HopSign"
    ds = build_hopsign_dataset(N=1000, radius=0.1, k=k, label_noise=0.5)
    mean_spd = compute_mean_spd(ds["all_pairs_dist"])
    build_and_cache_distance_operators(
        ds["all_pairs_dist"], max_dist=recommended_max_dist(mean_spd),
        cache_dir=hopsign_cache_dir / ds_name,
    )
    return ds, mean_spd

# %% [markdown]
# ## Load or initialise results

# %%
if results_path.exists():
    full_results = torch.load(results_path, map_location="cpu", weights_only=False)
    print(f"Resuming — {len(full_results.get('eval', {}))} datasets already done.")
else:
    full_results = {"name": name, "seed": seed, "hparams": p, "eval": {}}

already_done = set(full_results.get("eval", {}).keys())

# %% [markdown]
# ## Train (skipped if checkpoint already exists)

# %%
TRAIN_DS = "Cora"

if ckpt_path.exists():
    print("Checkpoint found — skipping training.")
    data, X, apd, y_class, y_onehot, splits, C, train_mean_spd = prep_benchmark(TRAIN_DS)
    mu_max, tausqrt_max = resolve_bounds(train_mean_spd)
    _, _, deepset_cfg = make_configs(mu_max, tausqrt_max)
else:
    print(f"\nTraining on {TRAIN_DS}  [mode={p['training_mode']}  seed={seed}]")
    t0 = time.perf_counter()

    data, X, apd, y_class, y_onehot, splits, C, train_mean_spd = prep_benchmark(TRAIN_DS)
    mu_max, tausqrt_max = resolve_bounds(train_mean_spd)
    op_cfg, multi_cfg, deepset_cfg = make_configs(mu_max, tausqrt_max)

    goblin_train = GOBLIN(
        data=data, X=X, all_pairs_dist=apd,
        y_class=y_class, y_onehot=y_onehot, splits=splits,
        sigma=p["sigma"], C=C,
        operator_search_cfg=op_cfg, multi_search_cfg=multi_cfg, deepset_cfg=deepset_cfg,
        seed=seed, dataset_name=TRAIN_DS,
    )
    goblin_train.run_operator_search()
    train_basis = goblin_train.select_basis()
    print(f"Train basis ({len(train_basis)}): {train_basis}")

    if p["training_mode"] == "pool":
        train_metrics = goblin_train.train_deepset_pool(verbose=False)
    elif p["training_mode"] == "stochastic":
        train_metrics = goblin_train.train_deepset_stochastic(verbose=False)
    else:
        train_metrics = goblin_train.train_deepset(verbose=False)

    train_time = time.perf_counter() - t0
    print(f"Training done in {train_time:.1f}s — val/acc={train_metrics.get('val/acc', float('nan')):.4f}")

    goblin_train.save_deepset(ckpt_path)
    del goblin_train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    full_results["train_basis"]   = [str(op) for op in train_basis]
    full_results["train_metrics"] = {k: v for k, v in train_metrics.items() if not isinstance(v, torch.Tensor)}
    full_results["train_time_s"]  = train_time
    full_results.setdefault("eval", {})
    torch.save(full_results, results_path)

# %% [markdown]
# ## Evaluate on all datasets

# %%
todo = [ds for ds in EVAL_DS if ds not in already_done]
print(f"Evaluating {len(todo)} datasets (of {len(EVAL_DS)} total)...")

for ds in tqdm(todo):
    print(f"\n--- {ds} ---")
    row = {}
    eval_goblin = None
    try:
        if ds.endswith("HopSign"):
            k = int(ds[0])
            hops_ds, mean_spd = prep_hopsign(k)
            mu_max_e, tausqrt_max_e = resolve_bounds(mean_spd)
            op_cfg_e, multi_cfg_e, _ = make_configs(mu_max_e, tausqrt_max_e)
            eval_goblin = GOBLIN(
                data=hops_ds["data"], X=hops_ds["X"],
                all_pairs_dist=hops_ds["all_pairs_dist"],
                y_class=hops_ds["y_class"], y_onehot=hops_ds["y_onehot"],
                splits=hops_ds["splits"],
                sigma=p["sigma"], C=2,
                operator_search_cfg=op_cfg_e, multi_search_cfg=multi_cfg_e,
                deepset_cfg=deepset_cfg, seed=seed, dataset_name=ds,
            )
        else:
            data_e, X_e, apd_e, y_class_e, y_onehot_e, splits_e, C_e, mean_spd = prep_benchmark(ds)
            mu_max_e, tausqrt_max_e = resolve_bounds(mean_spd)
            op_cfg_e, multi_cfg_e, _ = make_configs(mu_max_e, tausqrt_max_e)
            eval_goblin = GOBLIN(
                data=data_e, X=X_e, all_pairs_dist=apd_e,
                y_class=y_class_e, y_onehot=y_onehot_e, splits=splits_e,
                sigma=p["sigma"], C=C_e,
                operator_search_cfg=op_cfg_e, multi_search_cfg=multi_cfg_e,
                deepset_cfg=deepset_cfg, seed=seed, dataset_name=ds,
            )

        eval_goblin.run_operator_search()
        basis_e = eval_goblin.select_basis()
        row["basis"] = [str(op) for op in basis_e]

        eval_goblin.load_deepset(ckpt_path)
        metrics_e = eval_goblin.eval_deepset(splits=["val", "test"])
        row["metrics"]  = {k: v for k, v in metrics_e.items() if not isinstance(v, torch.Tensor)}
        row["val_acc"]  = metrics_e.get("val/acc",  float("nan"))
        row["test_acc"] = metrics_e.get("test/acc", float("nan"))
        print(f"  val={row['val_acc']:.4f}  test={row['test_acc']:.4f}  basis={basis_e}")

    except Exception as e:
        import traceback
        row["error"]     = str(e)
        row["traceback"] = traceback.format_exc()
        print(f"  ERROR: {e}")
    finally:
        if eval_goblin is not None:
            del eval_goblin
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_results["eval"][ds] = row
    torch.save(full_results, results_path)

# %% [markdown]
# ## Summary

# %%
print(f"\nSummary  name={name}  seed={seed}")
print("-" * 50)
for ds, row in full_results.get("eval", {}).items():
    if "error" in row:
        print(f"  {ds:<20} ERROR: {row['error'][:60]}")
    else:
        print(f"  {ds:<20} test={row.get('test_acc', float('nan')):.4f}")
print(f"\nResults saved to {results_path}")
