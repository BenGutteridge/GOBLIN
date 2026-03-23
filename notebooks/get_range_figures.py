# %%
# Figure 2 — operator range histogram for the canonical GOBLIN config (5 seeds).
#
# Usage:
#   python notebooks/get_range_figures.py
#
# Outputs: figures/benchmark_ranges_*.pdf, figures/range_vs_delta_scatter*.pdf
# Range cache: data_cache/ranges_goblin.pt  (delete to force recompute)

# %% [markdown]
# ## Imports & setup

# %%
import gzip
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

# Stub dgl to avoid CUDA library errors on login nodes
if "dgl" not in sys.modules:
    sys.modules["dgl"] = types.ModuleType("dgl")

ROOT = Path(__file__).resolve().parent
while not (ROOT / "goblin").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goblin.data import load_graph_dataset, build_hopsign_dataset

# %% [markdown]
# ## Paths, dataset lists, config

# %%
from goblin.config import DATA_CACHE
KHOP_DIR    = DATA_CACHE / "citynetwork_apspd"
RESULTS_DIR = ROOT / "results" / "goblin"
GOBLIN_NAME = "canonical"
SEEDS       = list(range(5))

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CACHE_DIR  = ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "ranges_goblin.pt"

CITY_DATASETS = ["CityLA", "CityLondon", "CityParis", "CityShanghai"]
BENCHMARK_DS  = [
    "AirBrazil", "AirUS", "AirEU",
    "Cornell", "Texas", "Wisconsin",
    "Chameleon", "Wiki", "Squirrel", "Actor",
    "Citeseer", "BlogCatalog", "WkCS",
    "Tolokers", "AmzComp", "AmzPhoto",
    "Minesweeper", "DBLP", "CoCS",
    "Pubmed", "FCora", "Roman", "AmzRatings",
    "CoPhysics", "Questions",
]
HOPSIGN_DS = [f"{k}HopSign" for k in range(1, 9)]
ALL_DS = BENCHMARK_DS + CITY_DATASETS + HOPSIGN_DS  # 37 total


# %% [markdown]
# ## Helper functions (operator parsing, CSR, BFS, range estimators, graph loading)

# %%
# ---------------------------------------------------------------------------
# Operator string parsing
# ---------------------------------------------------------------------------

def parse_op(op_str: str):
    """Parse 'OpId(family=..., param=...)' → (family, param)."""
    m = re.search(r"family='(\w+)'", op_str)
    family = m.group(1) if m else "unknown"
    m = re.search(r"param=([^\)]+)", op_str)
    raw = m.group(1).strip().strip("'\"") if m else "0"
    try:
        param = float(raw)
    except ValueError:
        param = raw  # e.g. 'L2'
    return family, param


# %%
# ---------------------------------------------------------------------------
# CSR construction from edge_index
# ---------------------------------------------------------------------------

def build_csr(edge_index: torch.Tensor, N: int):
    """
    Build sorted CSR (a_src, a_dst, ptr) from a 2×E edge_index.
    Removes self-loops.
    """
    src = edge_index[0].long()
    dst = edge_index[1].long()
    mask = src != dst
    src, dst = src[mask], dst[mask]
    order = torch.argsort(src)
    a_src = src[order]
    a_dst = dst[order]
    deg = torch.bincount(a_src, minlength=N)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)])
    return a_src, a_dst, ptr


# %%
# ---------------------------------------------------------------------------
# Single-source BFS
# ---------------------------------------------------------------------------

def bfs(u: int, a_dst: torch.Tensor, ptr: torch.Tensor, N: int,
        max_k: int = 20) -> torch.Tensor:
    """BFS from node u. Returns hop-distance vector of length N (-1 = unreachable)."""
    dist = torch.full((N,), -1, dtype=torch.long)
    dist[u] = 0
    frontier = torch.tensor([u], dtype=torch.long)
    for k in range(1, max_k + 1):
        lo, hi = ptr[frontier], ptr[frontier + 1]
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            break
        bid = torch.repeat_interleave(torch.arange(len(frontier)), counts)
        cum = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        nbrs = a_dst[lo[bid] + torch.arange(total) - cum[bid]]
        new_nodes = nbrs[dist[nbrs] == -1].unique()
        if new_nodes.numel() == 0:
            break
        dist[new_nodes] = k
        frontier = new_nodes
    return dist


# %%
# ---------------------------------------------------------------------------
# Sparse operator range estimators
# ---------------------------------------------------------------------------

def sparse_heat_range(a_src: torch.Tensor, a_dst: torch.Tensor,
                      ptr: torch.Tensor, N: int,
                      tau: float, N_eff: int = 200, K: int = 10,
                      seed: int = 42) -> float:
    """
    Estimate expected hop-distance under exp(-tau * L_sym) via sparse
    Taylor expansion + BFS, averaged over N_eff sampled source nodes.
    """
    deg = (ptr[1:] - ptr[:-1]).float()
    d_invsqrt = (deg + 1e-8).sqrt().reciprocal()
    gen = torch.Generator().manual_seed(seed)
    sample = torch.randperm(N, generator=gen)[:min(N_eff, N)]
    rhos = []
    for u in sample.tolist():
        dist = bfs(u, a_dst, ptr, N)
        x = torch.zeros(N); x[u] = 1.0
        result = x.clone(); term = x.clone()
        for k in range(1, K + 1):
            Av = torch.zeros(N)
            Av.scatter_add_(0, a_src, (d_invsqrt * term)[a_dst])
            term = (-tau / k) * (term - d_invsqrt * Av)
            result = result + term
        S_row = result.abs()
        total = S_row.sum().item()
        if total < 1e-12:
            continue
        d_float = dist.float().clamp(min=0)
        rhos.append((S_row * d_float).sum().item() / total)
    return float(np.mean(rhos)) if rhos else float("nan")


def sparse_l2_range(a_dst: torch.Tensor, ptr: torch.Tensor, N: int,
                    N_eff: int = 200, seed: int = 42) -> float:
    """
    Estimate expected hop-distance under A_rw^2 (2-step random walk) + BFS,
    averaged over N_eff sampled source nodes.
    """
    deg = (ptr[1:] - ptr[:-1]).float().clamp(min=1)
    gen = torch.Generator().manual_seed(seed)
    sample = torch.randperm(N, generator=gen)[:min(N_eff, N)]
    rhos = []
    for u in sample.tolist():
        dist = bfs(u, a_dst, ptr, N)
        lo_u = int(ptr[u].item()); hi_u = int(ptr[u + 1].item())
        nbrs_u = a_dst[lo_u:hi_u]
        w_row = torch.zeros(N)
        for w in nbrs_u.tolist():
            lo_w = int(ptr[w].item()); hi_w = int(ptr[w + 1].item())
            nbrs_w = a_dst[lo_w:hi_w]
            w_row[nbrs_w] += 1.0 / (deg[u].item() * deg[w].item())
        total = w_row.sum().item()
        if total < 1e-12:
            continue
        d_float = dist.float().clamp(min=0)
        rhos.append((w_row * d_float).sum().item() / total)
    return float(np.mean(rhos)) if rhos else float("nan")


def op_range(family: str, param, a_src, a_dst, ptr, N, N_eff: int) -> float:
    if family == "gaussian":
        return float(param)
    if family == "heat":
        return sparse_heat_range(a_src, a_dst, ptr, N, tau=float(param), N_eff=N_eff)
    if family == "fixed":  # L2
        return sparse_l2_range(a_dst, ptr, N, N_eff=N_eff)
    return float("nan")


# %%
# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_benchmark_csr(ds: str):
    data, X, apd, y_class, y_onehot, splits, C = load_graph_dataset(
        name=ds, root=ROOT / "data/goblin", seed=0,
        compute_all_pairs_dist=False,
    )
    N = int(X.shape[0])
    ei = data.edge_index
    a_src, a_dst, ptr = build_csr(ei, N)
    return a_src, a_dst, ptr, N


def load_hopsign_csr(k: int):
    ds = build_hopsign_dataset(N=1000, radius=0.1, k=k, label_noise=0.5, topology_seed=0)
    N = int(ds["X"].shape[0])
    ei = ds["data"].edge_index
    a_src, a_dst, ptr = build_csr(ei, N)
    return a_src, a_dst, ptr, N


def load_city_csr(city: str):
    p = KHOP_DIR / f"{city}_khop_01.pt"
    with gzip.open(str(p), "rb") as f:
        d = torch.load(f, weights_only=True)
    row, col = d["row"].long(), d["col"].long()
    # Undirected: add both directions
    src = torch.cat([row, col])
    dst = torch.cat([col, row])
    N = int(d["N"])
    order = torch.argsort(src)
    a_src = src[order]; a_dst = dst[order]
    deg = torch.bincount(a_src, minlength=N)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)])
    return a_src, a_dst, ptr, N


# %% [markdown]
# ## Load or compute alpha-weighted ranges (cached)
# Loads from `data_cache/ranges_goblin.pt` if present, otherwise runs
# BFS+Taylor expansion for all 37 datasets. Each range is the alpha-weighted mean
# of operator ranges, averaged over 5 seeds.

# %%
# ---------------------------------------------------------------------------
# Main range computation
# ---------------------------------------------------------------------------

def compute_all_ranges(results: dict) -> dict:
    """
    Compute {ds: {"mean": float, "std": float}} for all 37 datasets.
    Range per seed = unweighted mean of per-operator ranges.
    Mean/std across 5 seeds.
    """
    all_ranges = {}

    ds_groups = [
        ("Benchmarks",   BENCHMARK_DS,  load_benchmark_csr, lambda ds: ds),
        ("kHopSign",      HOPSIGN_DS,    lambda ds: load_hopsign_csr(int(ds[0])), lambda ds: ds),
        ("CityNetworks",CITY_DATASETS, load_city_csr, lambda ds: ds),
    ]

    for group_label, ds_list, loader_fn, name_fn in ds_groups:
        print(f"\n{'='*50}\n{group_label}\n{'='*50}")
        for ds in tqdm(ds_list, desc=group_label):
            try:
                a_src, a_dst, ptr, N = loader_fn(ds)
            except Exception as e:
                print(f"  {ds}: LOAD ERROR — {e}")
                continue

            # N_eff: use all nodes for small graphs, sample for large ones
            N_eff = N if N <= 500 else (500 if N <= 5000 else 200)

            # Cache of (family, round(param)) → range, shared across seeds
            op_cache: dict = {}

            seed_ranges = []
            for seed in SEEDS:
                ev = results[seed]["eval"]
                if ds not in ev or "error" in ev[ds] or not ev[ds].get("basis"):
                    continue
                row = ev[ds]
                basis = row["basis"]
                # Alpha weights from DeepSet; fall back to uniform if absent
                alpha = row.get("mean_alpha", None)
                if alpha is not None:
                    alpha = alpha.float().numpy()
                    alpha = alpha / alpha.sum()  # normalise
                else:
                    alpha = np.ones(len(basis)) / len(basis)

                op_rs = []
                for op_str in basis:
                    family, param = parse_op(op_str)
                    key = (family, round(float(param), 6) if isinstance(param, float) else param)
                    if key not in op_cache:
                        op_cache[key] = op_range(family, param, a_src, a_dst, ptr, N, N_eff)
                    op_rs.append(op_cache[key])

                # Alpha-weighted mean (skip if any nan — shouldn't happen)
                op_rs = np.array(op_rs)
                valid = ~np.isnan(op_rs)
                if valid.any():
                    w = alpha[valid]; w = w / w.sum()
                    seed_ranges.append(float((op_rs[valid] * w).sum()))

            if seed_ranges:
                all_ranges[ds] = {
                    "mean": float(np.mean(seed_ranges)),
                    "std":  float(np.std(seed_ranges)),
                }
                print(f"  {ds:<22} {all_ranges[ds]['mean']:.3f} ± {all_ranges[ds]['std']:.3f}")
            else:
                print(f"  {ds:<22} no data")

            # Save incrementally after each dataset
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(all_ranges, CACHE_FILE)

    return all_ranges


# ---------------------------------------------------------------------------
# Load or compute
# ---------------------------------------------------------------------------

results = {s: torch.load(RESULTS_DIR / f"{GOBLIN_NAME}_seed{s}.pt",
                          map_location="cpu", weights_only=False)
           for s in SEEDS}

if CACHE_FILE.exists():
    all_ranges = torch.load(CACHE_FILE, weights_only=False)
    print(f"Loaded cached ranges ({len(all_ranges)}/37 datasets) from {CACHE_FILE}")
    for ds, v in all_ranges.items():
        print(f"  {ds:<22} {v['mean']:.3f} ± {v['std']:.3f}")
else:
    print("Computing ranges...")
    all_ranges = compute_all_ranges(results)
    print(f"\nSaved ranges for {len(all_ranges)} datasets to {CACHE_FILE}")


# %% [markdown]
# ## Compute median summary bars for benchmarks and city networks

# %%
# ---------------------------------------------------------------------------
# Average bars
# ---------------------------------------------------------------------------

bench_means = [all_ranges[ds]["mean"] for ds in BENCHMARK_DS if ds in all_ranges]
bench_stds  = [all_ranges[ds]["std"]  for ds in BENCHMARK_DS if ds in all_ranges]
city_means  = [all_ranges[ds]["mean"] for ds in CITY_DATASETS if ds in all_ranges]
city_stds   = [all_ranges[ds]["std"]  for ds in CITY_DATASETS if ds in all_ranges]

avg_bench = {"median": float(np.median(bench_means)), "std": float(np.mean(bench_stds))}
avg_city  = {"median": float(np.median(city_means)),  "std": float(np.mean(city_stds))}


# %% [markdown]
# ## Figure: alpha-weighted range for all 37 datasets + benchmark/city medians
# Bars show mean ± std over 5 seeds. Median bars (green) use median over datasets
# to avoid outlier inflation from Minesweeper (σ=29) and AmzRatings (σ=12).
# → `benchmark_ranges_full.pdf`

# %%
# ---------------------------------------------------------------------------
# Full plot — all 37 datasets + 2 average bars
# ---------------------------------------------------------------------------

ordered_names  = []
ordered_means  = []
ordered_stds   = []
ordered_colors = []

for ds in BENCHMARK_DS:
    if ds not in all_ranges:
        continue
    ordered_names.append(ds)
    ordered_means.append(all_ranges[ds]["mean"])
    ordered_stds.append(all_ranges[ds]["std"])
    ordered_colors.append("tab:blue")
ordered_names.append("Median")
ordered_means.append(avg_bench["median"])
ordered_stds.append(avg_bench["std"])
ordered_colors.append("tab:green")

for ds in CITY_DATASETS:
    if ds not in all_ranges:
        continue
    ordered_names.append(ds)
    ordered_means.append(all_ranges[ds]["mean"])
    ordered_stds.append(all_ranges[ds]["std"])
    ordered_colors.append("tab:red")
ordered_names.append("Median")
ordered_means.append(avg_city["median"])
ordered_stds.append(avg_city["std"])
ordered_colors.append("tab:green")

for ds in HOPSIGN_DS:
    if ds not in all_ranges:
        continue
    ordered_names.append(ds)
    ordered_means.append(all_ranges[ds]["mean"])
    ordered_stds.append(all_ranges[ds]["std"])
    ordered_colors.append("tab:orange")

from matplotlib.patches import Patch

_n_full = len(ordered_names)
fig, ax = plt.subplots(figsize=(8, 2.0 * _n_full / 12))
y_pos = list(range(_n_full))
ax.barh(y_pos, ordered_means, xerr=ordered_stds, capsize=2,
        color=ordered_colors, alpha=0.85, height=0.85)
ax.set_yticks(y_pos)
ax.set_yticklabels(ordered_names, fontsize=7)
ax.set_xlabel("Range of GOBLIN", fontsize=8)
ax.tick_params(axis="x", labelsize=7)
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
for lbl in ax.get_yticklabels():
    if lbl.get_text() == "Median":
        lbl.set_fontweight("bold")
ax.legend(handles=[
    Patch(color="tab:blue",   label="Benchmarks (25)"),
    Patch(color="tab:red",    label="City networks (4)"),
    Patch(color="tab:orange", label="HopSign (8)"),
    Patch(color="tab:green",  label="Median"),
], loc="upper right", fontsize=7)
plt.tight_layout(pad=0.4)
out_full = FIGURES_DIR / "benchmark_ranges_full.pdf"
plt.savefig(out_full, bbox_inches="tight")
plt.show()
print(f"Saved: {out_full}")


# %% [markdown]
# ## Figure: small paper figure — representative subset of datasets
# Compact version intended for inclusion in the paper.
# → `benchmark_ranges_small.pdf`

# %%
# ---------------------------------------------------------------------------
# Small paper figure — representative subset
# ---------------------------------------------------------------------------

# Top-5 benchmarks by GOBLIN Δ vs best baseline (highest positive delta)
MEDIAN_LABEL = "Median"
small_names = [
    "AirBrazil",       # +11.54
    "Wisconsin",       # +8.23
    "AirUS",           # +7.85
    "Squirrel",        # +6.28
    "Chameleon",       # +3.73
    MEDIAN_LABEL,
    "CityLA", "CityParis",
    "2HopSign", "4HopSign", "6HopSign", "8HopSign",
]

small_means_v = []
small_stds_v  = []
small_cols_v  = []
for k in small_names:
    if k == MEDIAN_LABEL:
        small_means_v.append(avg_bench["median"]); small_stds_v.append(avg_bench["std"])
        small_cols_v.append("tab:green")
    elif k not in all_ranges:
        continue
    elif "HopSign" in k:
        small_means_v.append(all_ranges[k]["mean"]); small_stds_v.append(all_ranges[k]["std"])
        small_cols_v.append("tab:orange")
    elif "City" in k:
        small_means_v.append(all_ranges[k]["mean"]); small_stds_v.append(all_ranges[k]["std"])
        small_cols_v.append("tab:red")
    else:
        small_means_v.append(all_ranges[k]["mean"]); small_stds_v.append(all_ranges[k]["std"])
        small_cols_v.append("tab:blue")

small_names_clean = [k for k in small_names if k == MEDIAN_LABEL or k in all_ranges]

fig, ax = plt.subplots(figsize=(1.7, 2))
ax.barh(small_names_clean, small_means_v, xerr=small_stds_v, capsize=2,
        color=small_cols_v, height=0.85)
ax.set_xlabel("Range", fontsize=8)
ax.tick_params(axis="both", labelsize=7)
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
# Bold the Median tick label
for lbl in ax.get_yticklabels():
    if lbl.get_text() == MEDIAN_LABEL:
        lbl.set_fontweight("bold")
plt.tight_layout(pad=0.4)
out_small = FIGURES_DIR / "benchmark_ranges_small.pdf"
plt.savefig(out_small, bbox_inches="tight")
plt.show()
print(f"Saved: {out_small}")


# %% [markdown]
# ## Figure: small poster figure — same subset minus CityParis, 4/6HopSign
# Transparent background variant for poster overlay.
# → `benchmark_ranges_poster.pdf`

# %%
# ---------------------------------------------------------------------------
# Small poster figure — representative subset (no CityParis, 4HopSign, 6HopSign)
# ---------------------------------------------------------------------------

poster_names = [
    "AirBrazil",       # +11.54
    "Wisconsin",       # +8.23
    "AirUS",           # +7.85
    "Squirrel",        # +6.28
    "Chameleon",       # +3.73
    MEDIAN_LABEL,
    "CityLA",
    "2HopSign", "8HopSign",
]

poster_means_v = []
poster_stds_v  = []
poster_cols_v  = []
for k in poster_names:
    if k == MEDIAN_LABEL:
        poster_means_v.append(avg_bench["median"]); poster_stds_v.append(avg_bench["std"])
        poster_cols_v.append("tab:green")
    elif k not in all_ranges:
        continue
    elif "HopSign" in k:
        poster_means_v.append(all_ranges[k]["mean"]); poster_stds_v.append(all_ranges[k]["std"])
        poster_cols_v.append("tab:orange")
    elif "City" in k:
        poster_means_v.append(all_ranges[k]["mean"]); poster_stds_v.append(all_ranges[k]["std"])
        poster_cols_v.append("tab:red")
    else:
        poster_means_v.append(all_ranges[k]["mean"]); poster_stds_v.append(all_ranges[k]["std"])
        poster_cols_v.append("tab:blue")

poster_names_clean = [k for k in poster_names if k == MEDIAN_LABEL or k in all_ranges]

fig, ax = plt.subplots(figsize=(1.7, 1.6))
fig.patch.set_alpha(0.0)
ax.set_facecolor("none")
ax.barh(poster_names_clean, poster_means_v, xerr=poster_stds_v, capsize=2,
        color=poster_cols_v, height=0.85)
ax.set_xlabel("Range", fontsize=8)
ax.tick_params(axis="both", labelsize=7)
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
# Bold the Median tick label
for lbl in ax.get_yticklabels():
    if lbl.get_text() == MEDIAN_LABEL:
        lbl.set_fontweight("bold")
plt.tight_layout(pad=0.4)
out_poster = FIGURES_DIR / "benchmark_ranges_poster.pdf"
plt.savefig(out_poster, bbox_inches="tight", transparent=True)
plt.show()
print(f"Saved: {out_poster}")


# %% [markdown]
# ## Load or compute per-operator range cache
# Cached per unique (family, param) key that appeared across any seed.
# Used as a lookup table for the best-operator plot below.

# %%
# ---------------------------------------------------------------------------
# Per-operator range cache
# Compute & cache individual operator ranges (family, param, alpha, range)
# per dataset, averaged across seeds.
# Cache: data_cache/ranges_per_op_goblin.pt
# ---------------------------------------------------------------------------

PER_OP_CACHE = CACHE_DIR / "ranges_per_op_goblin.pt"

def op_label(family: str, param) -> str:
    if family == "fixed":
        return "L2"
    if family == "gaussian":
        return f"μ={float(param):.2f}"
    if family == "heat":
        return f"√τ={float(param):.2f}"
    return str(param)

def compute_per_op_ranges(results: dict) -> dict:
    """
    Returns {ds: [{"label": str, "mean_alpha": float, "mean_range": float, "std_range": float}]}
    sorted by descending mean_alpha.
    """
    per_op = {}
    ds_groups = [
        (BENCHMARK_DS,  load_benchmark_csr, lambda ds: ds),
        (CITY_DATASETS, load_city_csr,      lambda ds: ds),
        (HOPSIGN_DS,    lambda ds: load_hopsign_csr(int(ds[0])), lambda ds: ds),
    ]
    for ds_list, loader_fn, _ in ds_groups:
        for ds in tqdm(ds_list, desc="per-op ranges"):
            try:
                a_src, a_dst, ptr, N = loader_fn(ds)
            except Exception as e:
                print(f"  {ds}: LOAD ERROR — {e}")
                continue
            N_eff = N if N <= 500 else (500 if N <= 5000 else 200)
            op_cache: dict = {}

            # Collect per-seed (op_key → [range]), (op_key → [alpha])
            from collections import defaultdict
            key_ranges = defaultdict(list)
            key_alphas = defaultdict(list)
            key_labels = {}

            for seed in SEEDS:
                ev = results[seed]["eval"]
                if ds not in ev or "error" in ev[ds] or not ev[ds].get("basis"):
                    continue
                row = ev[ds]
                basis = row["basis"]
                alpha = row.get("mean_alpha", None)
                if alpha is not None:
                    alpha_np = alpha.float().numpy()
                else:
                    alpha_np = np.ones(len(basis)) / len(basis)

                for i, op_str in enumerate(basis):
                    family, param = parse_op(op_str)
                    key = (family, round(float(param), 6) if isinstance(param, float) else param)
                    if key not in op_cache:
                        op_cache[key] = op_range(family, param, a_src, a_dst, ptr, N, N_eff)
                    key_ranges[key].append(op_cache[key])
                    key_alphas[key].append(float(alpha_np[i]))
                    key_labels[key] = op_label(family, param)

            if not key_ranges:
                continue

            ops = []
            for key in key_ranges:
                ops.append({
                    "label":      key_labels[key],
                    "mean_alpha": float(np.mean(key_alphas[key])),
                    "mean_range": float(np.mean(key_ranges[key])),
                    "std_range":  float(np.std(key_ranges[key])),
                })
            # Sort by descending alpha weight
            ops.sort(key=lambda x: -x["mean_alpha"])
            per_op[ds] = ops

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(per_op, PER_OP_CACHE)
    return per_op


if PER_OP_CACHE.exists():
    per_op_ranges = torch.load(PER_OP_CACHE, weights_only=False)
    print(f"Loaded per-op cache ({len(per_op_ranges)} datasets)")
else:
    print("Computing per-operator ranges...")
    per_op_ranges = compute_per_op_ranges(results)
    print(f"Saved per-op ranges for {len(per_op_ranges)} datasets")

# %% [markdown]
# ## Figure: range of the single best-performing operator per dataset
# For each dataset, finds the operator (across all seeds × all basis positions)
# with the highest individual test accuracy, then plots its range.
# Colour = operator family (red=L2, blue=Heat, green=Gaussian).
# → `benchmark_ranges_best_op.pdf`

# %%
# ---------------------------------------------------------------------------
# Best-operator plot — all 37 datasets
# For each dataset, find the single operator with the highest individual
# test accuracy across ALL seeds and ALL basis positions.
# Plot its range, coloured by family, labelled with operator descriptor.
# ---------------------------------------------------------------------------

# Build label → range lookup from the per_op cache.
# Also store remapped entries so new label formats (μ=, √τ=) resolve old cache entries (σ=, τ=).
_op_range_lookup: dict = {}
for ds_ops in per_op_ranges.values():
    for op in ds_ops:
        lbl = op["label"]
        rng = op.get("mean_range", op.get("range", float("nan")))
        _op_range_lookup[lbl] = rng
        if lbl.startswith("σ="):
            _op_range_lookup["μ=" + lbl[2:]] = rng
        elif lbl.startswith("τ="):
            _op_range_lookup["√τ=" + lbl[2:]] = rng

FAMILY_COLORS = {"fixed": "tab:red", "heat": "tab:blue", "gaussian": "tab:green"}

def best_op_for_ds(ds: str):
    """Return (label, family, range, best_acc) for the best individual GOBLIN operator."""
    best_acc, best_label, best_family, best_range = -1, None, None, float("nan")
    for seed in SEEDS:
        ev = results[seed]["eval"]
        if ds not in ev or not ev[ds].get("basis"):
            continue
        row    = ev[ds]
        basis  = row["basis"]
        metrics = row.get("metrics", {})
        for i, op_str in enumerate(basis):
            acc = metrics.get(f"test/acc/expert{i}", None)
            if acc is None:
                continue
            if acc > best_acc:
                family, param = parse_op(op_str)
                lbl = op_label(family, param)
                rng = float(param) if family == "gaussian" \
                      else _op_range_lookup.get(lbl, float("nan"))
                best_acc, best_label, best_family, best_range = acc, lbl, family, rng
    return best_label, best_family, best_range, best_acc

ds_order_best  = [ds for ds in BENCHMARK_DS  if ds in all_ranges]
ds_order_best += [ds for ds in CITY_DATASETS if ds in all_ranges]
ds_order_best += [ds for ds in HOPSIGN_DS   if ds in all_ranges]

names_b, means_b, cols_b, labels_b = [], [], [], []
for ds in ds_order_best:
    lbl, fam, rng, acc = best_op_for_ds(ds)
    if lbl is None:
        continue
    names_b.append(ds)
    means_b.append(rng)
    cols_b.append(FAMILY_COLORS.get(fam, "gray"))
    labels_b.append(lbl)

_n_best = len(names_b)
fig, ax = plt.subplots(figsize=(8, 2.0 * _n_best / 12))
y_pos = list(range(_n_best))
ax.barh(y_pos, means_b, color=cols_b, alpha=0.85, height=0.85)
for i, (w, lbl) in enumerate(zip(means_b, labels_b)):
    ax.text(w + 0.05, i, lbl, va="center", ha="left", fontsize=7)
ax.set_yticks(y_pos)
ax.set_yticklabels(names_b, fontsize=7)
ax.tick_params(axis="x", labelsize=7)
ax.invert_yaxis()
ax.set_xlabel("Range of best individual GOBLIN operator", fontsize=8)
ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[
    Patch(color="tab:red",   label="$A^2$"),
    Patch(color="tab:blue",  label="LinHeat ($\\sqrt{\\tau}$)"),
    Patch(color="tab:green", label="LinGauss ($\\mu$)"),
], fontsize=7, loc="upper right")
plt.tight_layout(pad=0.4)
out_best = FIGURES_DIR / "benchmark_ranges_best_op.pdf"
plt.savefig(out_best, bbox_inches="tight")
plt.show()
print(f"Saved: {out_best}")


# %% [markdown]
# ## Figure: scatter — GOBLIN performance delta vs operator range
# x = alpha-weighted range (mean over 5 seeds), y = GOBLIN test acc minus best
# non-GOBLIN baseline (all methods). Includes 25 benchmarks + 4 city networks.
# Pearson r and Spearman ρ both shown. Marker size ∝ 1/range_std.
# → `range_vs_delta_scatter.pdf`

# %%
# ---------------------------------------------------------------------------
# Correlation: GOBLIN Δ vs range
# Δ = GOBLIN test_acc minus best non-GOBLIN baseline, on 25 benchmarks.
# Scatter plot with linear fit. Pearson r (raw) + Spearman r (rank, robust).
# Error bars on x = range std across seeds.
# Marker size ∝ 1/range_std (larger = more stable estimate) to reflect variance.
# ---------------------------------------------------------------------------

from scipy import stats

# Deltas vs best non-GOBLIN model (all methods including MPNNs)
GOBLIN_DELTAS = {
    "WkCS":        -11.57,
    "Wiki":        -10.14,
    "Actor":        -7.23,
    "Roman":        -7.32,
    "BlogCatalog":  -8.77,
    "FCora":        -4.67,
    "Minesweeper":  -3.15,
    "CoPhysics":    -3.03,
    "Texas":        -3.24,
    "Citeseer":     -3.58,
    "Cornell":      -2.16,
    "AmzComp":      -2.12,
    "DBLP":         -1.80,
    "Pubmed":       -1.76,
    "AmzPhoto":     -1.28,
    "CoCS":         -1.28,
    "Questions":    -0.90,
    "Tolokers":     +0.07,
    "AirEU":         0.00,
    "AmzRatings":   +1.64,
    "Chameleon":    +3.73,
    "Squirrel":     +6.28,
    "AirUS":        +7.85,
    "Wisconsin":    +8.23,
    "AirBrazil":   +11.54,
    # City networks
    "CityParis":   +3.80,
    "CityShanghai": +0.67,
    "CityLA":      +4.95,
    "CityLondon":  +0.02,
}

ds_corr   = [ds for ds in GOBLIN_DELTAS if ds in all_ranges]
x_range   = np.array([all_ranges[ds]["mean"] for ds in ds_corr])
x_std     = np.array([all_ranges[ds]["std"]  for ds in ds_corr])
y_delta   = np.array([GOBLIN_DELTAS[ds]       for ds in ds_corr])

# Linear fit
slope, intercept, r, p, _ = stats.linregress(x_range, y_delta)
rho, p_spearman            = stats.spearmanr(x_range, y_delta)
x_fit = np.linspace(x_range.min(), x_range.max(), 200)
y_fit = slope * x_fit + intercept

fig, ax = plt.subplots(figsize=(6, 8))
ax.scatter(x_range, y_delta, s=50, c="steelblue", zorder=3, alpha=0.85, marker="o")
ax.errorbar(x_range, y_delta, xerr=x_std, fmt="none", ecolor="gray",
            alpha=0.4, zorder=2, capsize=2)
ax.plot(x_fit, y_fit, "r--", linewidth=1.2,
        label=f"fit  r={r:.2f} (p={p:.3f})\nSpearman ρ={rho:.2f} (p={p_spearman:.3f})")
ax.axhline(0, color="k", linewidth=0.7, linestyle=":")

for ds, xv, yv in zip(ds_corr, x_range, y_delta):
    ax.annotate(ds, (xv, yv), textcoords="offset points", xytext=(4, 2),
                fontsize=9, alpha=0.8)

ax.set_xlabel("GOBLIN range (α-weighted, mean over 5 seeds)", fontsize=9)
ax.set_ylabel("GOBLIN Δ vs best baseline (pp)", fontsize=9)
ax.legend(fontsize=9)
ax.grid(linestyle="--", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out_corr = FIGURES_DIR / "range_vs_delta_scatter.pdf"
plt.savefig(out_corr, bbox_inches="tight")
plt.show()
print(f"Saved: {out_corr}")
print(f"Pearson r={r:.3f}  p={p:.4f}")
print(f"Spearman ρ={rho:.3f}  p={p_spearman:.4f}")

# %%
