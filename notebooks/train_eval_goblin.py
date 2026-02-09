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
from goblin.data import (
    load_graph_dataset,
    build_hopsign_dataset,
    build_and_cache_distance_operators,
)


hparams = {
    "train_ds": "Cora",
    "seed": 0,
    "bo_objective": "trimmed_20",
    "basis_size": 2,
    "mix_strategy": "balanced",
    "bias_prob": 0.50,
    "basis_selection_rule": "diverse_top_k",
    "enforce_family_coverage": False,
    "num_explore_steps": 9,
    "num_exploit_steps": 6,
    # LinGauss
    "mu_min": 0.0,
    "mu_max": 8.0,
    "mu_num": 250,
    "sigma": 0.5,
    "search_min_sep_mu": 0.2,
    "basis_min_sep_mu": 0.5,
    # LinHeat
    "tau_min": 0.0,
    "tau_max": 5.0,
    "tau_num": 50,
    "search_min_sep_tau": 0.1,
    "basis_min_sep_tau": 1.0,
    "num_fixed_operators": 1,
    "fixed_operators": ["L1"],
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
}

# Uncomment as required

eval_ds = [
    # # HopSign
    "1HopSign",
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
]

p = hparams
hparam_str = json.dumps(p, sort_keys=True)
hparam_hash = hashlib.md5(hparam_str.encode()).hexdigest()

model_ckpt_path = Path(f"ckpts/goblin/{hparam_hash}.pt")
results_path = Path(f"output/results/goblin/{hparam_hash}.pt")
results_path.parent.mkdir(parents=True, exist_ok=True)
model_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

print("Hparam hash:", hparam_hash)
print("Sampled hyperparameters:", p)
print("Model checkpoint path:", model_ckpt_path)
print("Results path:", results_path)

data, X, all_pairs_dist, y_class, y_onehot, splits, C = load_graph_dataset(
    name=p["train_ds"],
    root=Path("data/goblin"),
    seed=p["seed"],
    compute_all_pairs_dist=True,
)

build_and_cache_distance_operators(
    all_pairs_dist,
    max_dist=10,
    cache_dir=f"data_cache/lingauss/{p['train_ds']}",
)

operator_cfg = OperatorSearchConfig(
    families=["heat", "gaussian"],
    bo_objective=p["bo_objective"],
    num_explore_steps=p["num_explore_steps"],
    num_exploit_steps=p["num_exploit_steps"],
    mu_min=p["mu_min"],
    mu_max=p["mu_max"],
    mu_num=p["mu_num"],
    tau_min=p["tau_min"],
    tau_max=p["tau_max"],
    tau_num=p["tau_num"],
    rbf_length_scale=p["rbf_length_scale"],
    white_noise=p["white_noise"],
)

multi_search_cfg = MultiSearchConfig(
    total_steps=(p["num_explore_steps"] + p["num_exploit_steps"]),
    mix_strategy=p["mix_strategy"],
    basis_size=p["basis_size"],
    basis_min_sep_mu=p["basis_min_sep_mu"],
    basis_min_sep_tau=p["basis_min_sep_tau"],
    bias_prob=p["bias_prob"],
    enforce_family_coverage=p["enforce_family_coverage"],
    include_fixed_ops=p["fixed_operators"],
    basis_selection_rule=p["basis_selection_rule"],
)

deepset_cfg = ExpertsDeepSetConfig(
    hidden_dim=p["hidden_dim"],
    attn_temp=p["attn_temp"],
    num_deepset_layers=p["num_deepset_layers"],
    num_head_layers=p["num_head_layers"],
    epochs=p["epochs"],
    lr=p["lr"],
    dropout=p["dropout"],
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
metrics = goblin.train_deepset(verbose=True)

goblin.save_deepset(model_ckpt_path)

# ###### EVAL ########
full_results = {"ckpt_path": str(model_ckpt_path), "hparams": p, "hash": hparam_hash}

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

    build_and_cache_distance_operators(
        test_dataset["all_pairs_dist"],
        max_dist=10,
        cache_dir=f"data_cache/lingauss/{ds_name}",
    )

    eval_goblin = GOBLIN(
        data=test_dataset["data"],
        X=test_dataset["X"],
        all_pairs_dist=test_dataset["all_pairs_dist"],
        y_class=test_dataset["y_class"],
        y_onehot=test_dataset["y_onehot"],
        splits=test_dataset["splits"],
        sigma=p["sigma"],
        C=2,
        operator_search_cfg=operator_cfg,
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
        compute_all_pairs_dist=True,
    )

    build_and_cache_distance_operators(
        all_pairs_dist,
        max_dist=10,
        cache_dir=f"data_cache/lingauss/{ds}",
    )

    eval_goblin = GOBLIN(
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
