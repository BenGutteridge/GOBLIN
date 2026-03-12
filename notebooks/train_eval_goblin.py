import argparse
import dataclasses
import numpy as np
import hashlib
import json
import sys
from pathlib import Path
from tqdm import tqdm
import torch

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from goblin import GOBLIN, OperatorSearchConfig, ExpertsDeepSetConfig, MultiSearchConfig
from goblin.config import DATA_CACHE
from goblin.data import (
    load_graph_dataset,
    build_hopsign_dataset,
    build_and_cache_distance_operators,
    build_and_cache_city_distance_operators,
    compute_mean_spd,
    recommended_max_dist,
)

hopsign_cache_dir = DATA_CACHE / "goblin_khopsign"
lingauss_cache_dir = DATA_CACHE / "apspd/lingauss"

hparams = {
    "train_ds": "Cora",
    "seed": 0,
    "bo_objective": "trimmed_20",
    "basis_size": 5,
    "basis_selection_rule": "greedy_diversity",  # "top_k" | "greedy_diversity"
    "enforce_family_coverage": False,
    "n_samples": 15,    # total linear GNN solves including anchors
    "ucb_beta": 1.0,    # β in acq = -µ_GP + β·σ_GP; 0 = pure exploitation
    # LinGauss
    "mu_min": 0.0,
    "mu_max": 8.0,
    "mu_num": 250,
    "sigma": 0.5,
    # LinHeat (search is over tausqrt = sqrt(tau))
    "tausqrt_min": 0.0,
    "tausqrt_max": 5.0,
    "tausqrt_num": 50,
    "diversity_lambda": 0.2,  # λ for greedy_diversity rule (0 = pure top-k)
    "mu_anchors": 1.0,         # gaussian family anchors: float, list[float], int (N auto-spaced), or None
    "tausqrt_anchors": 1.5,   # heat family anchors: float, list[float], int (N auto-spaced), or None
    "num_fixed_operators": 1,
    "fixed_operators": ["L1"],
    # Adaptive search bounds (0 = use fixed bounds; >0 = scale by mean SPD)
    "mu_max_spd_sf": 0.0,       # mu_max      = mu_max_spd_sf      * mean_SPD (or mu_max      if 0)
    "tausqrt_max_spd_sf": 0.0,  # tausqrt_max = tausqrt_max_spd_sf * mean_SPD (or tausqrt_max if 0)
    # GP
    "rbf_length_scale": 1.0,
    "white_noise": 0.2,
    # DeepSet
    "lr": 3e-4,
    "dropout": 0.0,
    "hidden_dim": 32,
    "attn_temp": 10.0,
    "num_deepset_layers": 2,
    "num_head_layers": 1,
    "epochs": 500,
    "score_feature": "none",  # "none" | "trimmed" | "trimmed_and_lower_half"
    "weight_selection": "current",   # "current" | "pre_filter" | "mask_by_deepset"
    "feature_set_size": "all",       # "all" | "top_half" | "basis_size"
    # stochastic training (set to False to use fixed full-batch training)
    "stochastic_training": False,
    "n_batches": 1000,
    "batch_size": 40, # for Cora with very few labels
    "n_ref_per_class": 5,
}

# Uncomment as required, or pass --eval_dataset via CLI

_parser = argparse.ArgumentParser()
_parser.add_argument("--eval_dataset", type=str, default=None)
for _key, _val in hparams.items():
    if isinstance(_val, bool):
        _parser.add_argument(f"--{_key}", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None)
    elif isinstance(_val, list):
        _parser.add_argument(f"--{_key}", nargs='+', default=None)
    else:
        _parser.add_argument(f"--{_key}", type=type(_val), default=None)
_args, _ = _parser.parse_known_args()

eval_ds = [
    # # HopSign
    # "1HopSign",
    # "2HopSign",
    # "3HopSign",
    # "4HopSign",
    # "5HopSign",
    # "6HopSign",
    # "7HopSign",
    # "8HopSign",
    #
    # # Benchmarks
    "AirBrazil",
    # "AirUS",
    # "AirEU",
    # "Cornell",
    # "Texas",
    # "Wisconsin",
    # "Chameleon",
    # "Wiki",
    # "Squirrel",
    # "Actor",
    # "Citeseer",
    # "BlogCatalog",
    # "WkCS",
    # "Tolokers",
    # "AmzComp",
    # "AmzPhoto",
    # "Minesweeper",
    # "DBLP",
    # "CoCS",
    # "Pubmed",
    # "FCora",
    # "Roman",
    # "AmzRatings",
    # "CoPhysics",
    # "Questions",
    #
    # CityNetworks
    # "CityParis",
    # "CityShanghai",
    # "CityLA",
    # "CityLondon",
]

if _args.eval_dataset is not None:
    eval_ds = [_args.eval_dataset]

p = dict(hparams)
for _key in hparams:
    _val = getattr(_args, _key)
    if _val is not None:
        p[_key] = _val
hparam_str = json.dumps(p, sort_keys=True)
hparam_hash = hashlib.md5(hparam_str.encode()).hexdigest()

model_ckpt_path = Path(f"ckpts/goblin/{hparam_hash}.pt")
# Use per-dataset results path when eval_dataset is specified, to avoid race
# conditions when multiple parallel jobs share the same hparam_hash.
if _args.eval_dataset is not None:
    results_path = Path(f"output/results/goblin/{hparam_hash}/{_args.eval_dataset}.pt")
else:
    results_path = Path(f"output/results/goblin/{hparam_hash}.pt")
results_path.parent.mkdir(parents=True, exist_ok=True)
model_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

CITY_DATASETS = {"CityParis", "CityShanghai", "CityLA", "CityLondon"}

print("Hparam hash:", hparam_hash)
print("Sampled hyperparameters:", p)
print("Model checkpoint path:", model_ckpt_path)
print("Results path:", results_path)

data, X, all_pairs_dist, y_class, y_onehot, splits, C = load_graph_dataset(
    name=p["train_ds"],
    root=Path("data/goblin"),
    seed=p["seed"],
    compute_all_pairs_dist=(p["train_ds"] not in CITY_DATASETS),
)

if p["train_ds"] in CITY_DATASETS:
    build_and_cache_city_distance_operators(
        city_name=p["train_ds"],
        cache_dir=lingauss_cache_dir / p["train_ds"],
        max_k=20,
    )
elif all_pairs_dist is not None:  # None means M_dist_k cache already sufficient
    _train_mean_spd_for_cache = compute_mean_spd(all_pairs_dist)
    build_and_cache_distance_operators(
        all_pairs_dist,
        max_dist=recommended_max_dist(_train_mean_spd_for_cache),  # previously set to 10
        cache_dir=lingauss_cache_dir / p["train_ds"],
    )

_train_mean_spd = compute_mean_spd(
    all_pairs_dist,
    cache_dir=lingauss_cache_dir / p["train_ds"] if all_pairs_dist is None else None,
)
_train_mu_max = p["mu_max_spd_sf"] * _train_mean_spd if p["mu_max_spd_sf"] > 0 else p["mu_max"]
_train_tausqrt_max = p["tausqrt_max_spd_sf"] * _train_mean_spd if p["tausqrt_max_spd_sf"] > 0 else p["tausqrt_max"]
print(f"Train mean SPD: {_train_mean_spd:.3f} → mu_max={_train_mu_max:.3f}, tausqrt_max={_train_tausqrt_max:.3f}")

operator_cfg = OperatorSearchConfig(
    families=["heat", "gaussian"],
    bo_objective=p["bo_objective"],
    ucb_beta=p["ucb_beta"],
    mu_min=p["mu_min"],
    mu_max=_train_mu_max,
    mu_num=p["mu_num"],
    tausqrt_min=p["tausqrt_min"],
    tausqrt_max=_train_tausqrt_max,
    tausqrt_num=p["tausqrt_num"],
    rbf_length_scale=p["rbf_length_scale"],
    white_noise=p["white_noise"],
)

multi_search_cfg = MultiSearchConfig(
    n_samples=p["n_samples"],
    basis_size=p["basis_size"],
    enforce_family_coverage=p["enforce_family_coverage"],
    include_fixed_ops=p["fixed_operators"],
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
)

goblin = GOBLIN(
    data=data,
    X=X,
    all_pairs_dist=all_pairs_dist,
    y_class=y_class,
    y_onehot=y_onehot,
    splits=splits,
    sigma=p["sigma"],
    C=C,
    operator_search_cfg=operator_cfg,
    multi_search_cfg=multi_search_cfg,
    deepset_cfg=deepset_cfg,
    seed=p["seed"],
    dataset_name=p["train_ds"],
)

search_out = goblin.run_operator_search()
print("Sampled ops:", search_out["sampled_ops"])

# ---- Basis selection ----
basis = goblin.select_basis()
print("Final basis:", basis)

# ---- Train DeepSet ----
if p["stochastic_training"]:
    metrics = goblin.train_deepset_stochastic(verbose=True)
else:
    metrics = goblin.train_deepset(verbose=True)

goblin.save_deepset(model_ckpt_path)

# ###### EVAL ########
full_results = torch.load(results_path) if results_path.exists() else {}
full_results.update({"ckpt_path": str(model_ckpt_path), "hparams": p, "hash": hparam_hash,
                      "loss_curve": goblin._last_loss_curve, "train_metrics": metrics})

# Evaluate on HopSign
N = 1000
radius = 0.1
label_noise = 0.5
seed = p["seed"]

for k in tqdm(range(1, int(p["mu_max"]) + 1)):
    ds_name = f"{k}HopSign"
    if ds_name not in eval_ds:
        continue

    test_dataset = build_hopsign_dataset(
        N=N,
        radius=radius,
        k=k,
        label_noise=label_noise,
    )

    _mean_spd = compute_mean_spd(test_dataset["all_pairs_dist"])
    build_and_cache_distance_operators(
        test_dataset["all_pairs_dist"],
        max_dist=recommended_max_dist(_mean_spd),  # previously set to 10
        cache_dir=hopsign_cache_dir / ds_name,
    )
    _mu_max = p["mu_max_spd_sf"] * _mean_spd if p["mu_max_spd_sf"] > 0 else p["mu_max"]
    _tausqrt_max = p["tausqrt_max_spd_sf"] * _mean_spd if p["tausqrt_max_spd_sf"] > 0 else p["tausqrt_max"]

    eval_goblin = GOBLIN(
        data=test_dataset["data"],
        X=test_dataset["X"],
        all_pairs_dist=test_dataset["all_pairs_dist"],
        y_class=test_dataset["y_class"],
        y_onehot=test_dataset["y_onehot"],
        splits=test_dataset["splits"],
        sigma=p["sigma"],
        C=2,
        operator_search_cfg=dataclasses.replace(operator_cfg, mu_max=_mu_max, tausqrt_max=_tausqrt_max),
        multi_search_cfg=multi_search_cfg,
        deepset_cfg=deepset_cfg,
        seed=p["seed"],
        dataset_name=ds_name,
    )

    search_out = eval_goblin.run_operator_search()
    basis = eval_goblin.select_basis()

    eval_goblin.load_deepset(model_ckpt_path)

    results = eval_goblin.eval_deepset(splits=["val", "test"])

    full_results[f"{k}HopSign"] = results | {"basis": basis}


# Now onto the benchmark datasets

for ds in tqdm(eval_ds):
    if ds.endswith("HopSign"):
        continue

    data, X, all_pairs_dist, y_class, y_onehot, splits, C = load_graph_dataset(
        name=ds,
        root=Path("data/goblin"),
        seed=p["seed"],
        compute_all_pairs_dist=(ds not in CITY_DATASETS),
    )

    if ds in CITY_DATASETS:
        build_and_cache_city_distance_operators(
            city_name=ds,
            cache_dir=lingauss_cache_dir / ds,
            max_k=20,
        )
    elif all_pairs_dist is not None:  # None means M_dist_k cache already sufficient
        _mean_spd_for_cache = compute_mean_spd(all_pairs_dist)
        build_and_cache_distance_operators(
            all_pairs_dist,
            max_dist=recommended_max_dist(_mean_spd_for_cache),  # previously set to 10
            cache_dir=lingauss_cache_dir / ds,
        )

    _mean_spd = compute_mean_spd(
        all_pairs_dist,
        cache_dir=lingauss_cache_dir / ds if all_pairs_dist is None else None,
    )
    _mu_max = p["mu_max_spd_sf"] * _mean_spd if p["mu_max_spd_sf"] > 0 else p["mu_max"]
    _tausqrt_max = p["tausqrt_max_spd_sf"] * _mean_spd if p["tausqrt_max_spd_sf"] > 0 else p["tausqrt_max"]

    eval_goblin = GOBLIN(
        data=data,
        X=X,
        all_pairs_dist=all_pairs_dist,
        y_class=y_class,
        y_onehot=y_onehot,
        splits=splits,
        sigma=p["sigma"],
        C=C,
        operator_search_cfg=dataclasses.replace(operator_cfg, mu_max=_mu_max, tausqrt_max=_tausqrt_max),
        multi_search_cfg=multi_search_cfg,
        deepset_cfg=deepset_cfg,
        seed=p["seed"],
        dataset_name=ds,
    )

    search_out = eval_goblin.run_operator_search()
    basis = eval_goblin.select_basis()

    eval_goblin.load_deepset(model_ckpt_path)

    results = eval_goblin.eval_deepset(splits=["val", "test"])

    full_results[ds] = results | {"basis": basis}

    torch.save(full_results, results_path)


torch.save(full_results, results_path)
print(f"Saved results to {results_path}")
