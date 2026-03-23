# %% [markdown]
# # GOBLIN — Figure 1, Tables 1–3
#
# Produces main paper figures and tables from cached results.
# Run the four run_*.py scripts first.
#
# Usage:
#   python notebooks/get_figures_and_tables.py

# %% [markdown]
# ## Imports and paths

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goblin.config import DATA_CACHE

FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# GraphAny operator result directories
HOPSIGN_GRAPHANY_DIR = DATA_CACHE / "camera_ready" / "graphany_ops_hopsign" / "CoraXHopSign"
BENCH_GRAPHANY_DIR   = DATA_CACHE / "camera_ready" / "graphany_ops_25bench" / "CoraX25Benchmarks"
CITY_RESULTS_DIR     = DATA_CACHE / "results"
GNN_RESULTS_DIR      = ROOT / "results" / "gnn"
TSGNN_FILE           = ROOT / "results" / "tsgnn" / "tsgnn_results.json"

# GOBLIN result files
GOBLIN_NAME  = "canonical"
GOBLIN_SEEDS = [0, 1, 2, 3, 4]
GOBLIN_DIR   = ROOT / "results" / "goblin"

KS = list(range(1, 9))  # k in 1–8HopSign

# %% [markdown]
# ## Dataset lists

# %%
HOPSIGN_DS = [f"{k}HopSign" for k in KS]

# Table 2 order: alphabetical, matching paper
BENCH_DS = [
    "Actor", "AirBrazil", "AirEU", "AirUS",
    "AmzComp", "AmzPhoto", "AmzRatings",
    "BlogCatalog", "Chameleon", "Citeseer",
    "CoCS", "CoPhysics", "Cornell", "DBLP", "FCora",
    "Minesweeper", "Pubmed", "Questions",
    "Roman", "Squirrel", "Texas", "Tolokers",
    "Wiki", "Wisconsin", "WkCS",
]
CITY_DS = ["CityParis", "CityShanghai", "CityLA", "CityLondon"]
ALL_GOBLIN_DS = HOPSIGN_DS + BENCH_DS + CITY_DS

# GraphAny channel → display label
FEAT_CHN_TO_LABEL = {
    "X+L1+L2+H1+H2":        r"$A, A^2$ (GraphAny)",
    "X+L1+L2+L3+L4+H1+H2": r"$A, A^2, A^3, A^4$",
    "X+N1+N2+N3+N4":        r"$A, A_2, A_3, A_4$",
    "X+N1+N2+N3d+NdR":      r"HopBins",
    "X+N1+DS+DM+DL":        r"HeatKernel",
}
LABEL_ORDER = [
    r"$A, A^2$ (GraphAny)",
    r"$A, A^2, A^3, A^4$",
    r"$A, A_2, A_3, A_4$",
    r"HopBins",
    r"HeatKernel",
]

# %% [markdown]
# ## Load GOBLIN multi-seed results

# %%
goblin_accs = {ds: [] for ds in ALL_GOBLIN_DS}

for seed in GOBLIN_SEEDS:
    path = GOBLIN_DIR / f"{GOBLIN_NAME}_seed{seed}.pt"
    if not path.exists():
        print(f"WARNING: Missing seed={seed}: {path}")
        continue
    r = torch.load(path, map_location="cpu", weights_only=False)
    for ds in ALL_GOBLIN_DS:
        v = r.get("eval", {}).get(ds, {}).get("test_acc", float("nan"))
        goblin_accs[ds].append(float(v) * 100 if v == v else float("nan"))

goblin_mean = {ds: float(np.nanmean(goblin_accs[ds])) for ds in ALL_GOBLIN_DS}
goblin_std  = {ds: float(np.nanstd(goblin_accs[ds]))  for ds in ALL_GOBLIN_DS}
n_seeds = sum(1 for s in GOBLIN_SEEDS
              if (GOBLIN_DIR / f"{GOBLIN_NAME}_seed{s}.pt").exists())
print(f"GOBLIN: {n_seeds}/{len(GOBLIN_SEEDS)} seeds loaded.")

# %% [markdown]
# ## Verify GOBLIN HopSign means

# %%
PUBLISHED_HOPSIGN_MEANS = [90.80, 96.40, 91.20, 93.20, 98.40, 98.00, 90.40, 91.60]
print("HopSign verification (loaded vs published):")
all_ok = True
for k, ds in enumerate(HOPSIGN_DS):
    ok = abs(goblin_mean[ds] - PUBLISHED_HOPSIGN_MEANS[k]) < 0.05
    if not ok:
        all_ok = False
    print(f"  {ds}: {goblin_mean[ds]:.2f}  published={PUBLISHED_HOPSIGN_MEANS[k]:.2f}"
          + ("" if ok else "  << MISMATCH"))
print("All match." if all_ok else "WARNING: mismatches found — check seed result files.")
GOBLIN_HOPSIGN_MEANS = [goblin_mean[ds] for ds in HOPSIGN_DS]

# %% [markdown]
# ## Load GraphAny operator results

# %%
def load_graphany_operators(results_dir: Path):
    """Load per-variant result entries from a GraphAny result directory."""
    files = list(results_dir.glob("*_results.json"))
    if not files:
        print(f"  No result files in {results_dir}")
        return {}
    print(f"  Found {len(files)} files in {results_dir.name}")
    data = {}
    for f in sorted(files):
        entry = json.load(open(f))
        hparams = entry.get("hparams", entry.get("cfg", {}))
        label = FEAT_CHN_TO_LABEL.get(hparams.get("feat_chn", ""))
        if label is None:
            continue
        data[label] = entry
    return data


print("GraphAny HopSign results:")
raw_hopsign = load_graphany_operators(HOPSIGN_GRAPHANY_DIR)
GRAPHANY_HOPSIGN = {}
for label in LABEL_ORDER:
    if label not in raw_hopsign:
        print(f"  WARNING: {label!r} missing")
        continue
    entry = raw_hopsign[label]
    hop_accs = [entry.get(f"hopsR/{k}hop_test_acc", float("nan")) for k in KS]
    GRAPHANY_HOPSIGN[label] = hop_accs
    print(f"  {label:30s}  k=1:{hop_accs[0]:.1f}  avg:{np.nanmean(hop_accs):.1f}")

print("\nGraphAny 25-benchmark results:")
raw_bench = load_graphany_operators(BENCH_GRAPHANY_DIR)
GRAPHANY_BENCH_AVG = {}
for label in LABEL_ORDER:
    if label not in raw_bench:
        print(f"  WARNING: {label!r} missing")
        continue
    avg = raw_bench[label].get("ind_test_acc", float("nan"))
    GRAPHANY_BENCH_AVG[label] = avg
    print(f"  {label:30s}  avg:{avg:.2f}%")

# %% [markdown]
# ## Load TS-GNN results (HopSign + CityNetworks)

# %%
def load_tsgnn_results(path: Path) -> dict:
    """Load TS-GNN results from run_tsgnn_exps.py output JSON.
    Returns {model_name: {dataset_name: {mean, std, per_seed}}}."""
    if not path.exists():
        print(f"  TS-GNN file not found: {path}")
        return {}
    return json.load(open(path))


tsgnn_results = load_tsgnn_results(TSGNN_FILE)

# Extract HopSign data for plotting
tsgnn_hopsign = {}
for model_name, ds_results in tsgnn_results.items():
    tsgnn_hopsign[model_name] = {}
    for k in KS:
        ds = f"hopsign{k}"
        if ds in ds_results:
            tsgnn_hopsign[model_name][k] = (ds_results[ds]["mean"], ds_results[ds]["std"])

# Extract City data
tsgnn_city = {}
for model_name, ds_results in tsgnn_results.items():
    tsgnn_city[model_name] = {}
    for city_ds in CITY_DS:
        city_key = city_ds.replace("City", "").lower()
        if city_key in ds_results:
            r = ds_results[city_key]
            tsgnn_city[model_name][city_ds] = (r["mean"], r["std"])

print(f"TS-GNN models loaded: {list(tsgnn_results.keys())}")
for mn in tsgnn_results:
    n_hop = sum(1 for k in KS if f"hopsign{k}" in tsgnn_results[mn])
    n_city = sum(1 for c in CITY_DS if c.replace("City", "").lower() in tsgnn_results[mn])
    print(f"  {mn}: {n_hop} HopSign, {n_city} CityNetworks")

# %% [markdown]
# ## Load CityNetworks results (MeanGNN, GAT, GraphAny)

# %%
def load_gnn_city(model: str, city: str):
    """Load (mean%, std%) from results/gnn/{model}/{city}.json."""
    p = GNN_RESULTS_DIR / model / f"{city}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    tr = d["test_results"]
    return tr["mean_pct"], tr["std_pct"]


def load_graphany_city(city: str, seeds=(0, 1, 2, 3)):
    """Load all seed test accs for a city from data_cache/results/Cora_X_{city}/."""
    d = CITY_RESULTS_DIR / f"Cora_X_{city}"
    if not d.exists():
        return None
    key = f"ind/{city.lower()[:4]}_test_acc"
    accs = []
    for f in sorted(d.glob("*_results.json")):
        try:
            data = json.load(open(f))
            hp = data.get("hparams", data.get("cfg", {}))
            if int(hp.get("seed", -1)) in seeds:
                v = data.get(key)
                if v is not None:
                    accs.append(float(v))
        except Exception:
            pass
    return accs if accs else None


def get_tsgat_city(city: str):
    """Get TS-GAT city results from loaded tsgnn_results."""
    return tsgnn_city.get("TS-GAT", {}).get(city)

# %% [markdown]
# ## Figure 1 — kHopSign line plot

# %%
OPERATOR_COLORS = [
    "#1f77b4",  # blue
    "#4e9fd4",  # light blue
    "#9467bd",  # purple
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]


def plot_line(ax, ks, vals, label, color, zorder=2, linestyle="-",
              marker="o", markersize=6):
    ax.plot(ks, vals, marker=marker, label=label, color=color,
            linewidth=1.8, linestyle=linestyle, zorder=zorder, markersize=markersize)


def plot_line_stats(ax, ks, k_to_stats, label, color, zorder=2, marker="o"):
    ks_p  = np.array([k for k in ks if k in k_to_stats])
    means = np.array([k_to_stats[k][0] for k in ks if k in k_to_stats])
    ax.plot(ks_p, means, marker=marker, label=label, color=color,
            linewidth=1.8, linestyle="-", zorder=zorder)


plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(8, 3))
ks = np.array(KS)

plot_line(ax, ks, GOBLIN_HOPSIGN_MEANS, label="GOBLIN", color="#d62728",
          linestyle="-", marker="*", markersize=11, zorder=5)

if "TS-GAT" in tsgnn_hopsign:
    plot_line_stats(ax, ks, tsgnn_hopsign["TS-GAT"], label="TS-GAT",
                    color="#ff7f0e", zorder=4)

for (label, vals), color in zip(GRAPHANY_HOPSIGN.items(), OPERATOR_COLORS):
    plot_line(ax, ks, vals, label=label, color=color, marker="s", zorder=2)

ax.axhline(50, color="gray", linestyle=":", linewidth=1.2, label="Random (50%)")
ax.set_xlabel("$k$HopSign")
ax.set_ylabel("Test accuracy (%)")
ax.set_xticks(KS)
ax.grid(True, alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
          ncol=1, fontsize=13, frameon=True, borderpad=0.4, handlelength=1.5)
fig.tight_layout()

out = FIGURES_DIR / "hopsign_lineplot.pdf"
fig.savefig(out, bbox_inches="tight", transparent=True)
print(f"Saved → {out}")
plt.show()

# %% [markdown]
# ## Table 1 — GraphAny operator basis: avg 25-benchmark accuracy
#
# Shown next to Figure 1 in the paper.
# Displays raw average accuracy and delta vs standard GraphAny (A, A²).

# %%
baseline_label = r"$A, A^2$ (GraphAny)"
baseline_score = GRAPHANY_BENCH_AVG.get(baseline_label, float("nan"))

print("Table 1 — GraphAny operator basis: avg 25-benchmark accuracy (Δ vs baseline)\n")
print(f"{'Operator basis':<32s}  {'Avg acc (%)':>11s}  {'Δ':>7s}")
print("-" * 56)
for label in LABEL_ORDER:
    score = GRAPHANY_BENCH_AVG.get(label, float("nan"))
    delta = score - baseline_score
    sign  = "+" if delta >= 0 else ""
    d_str = f"{sign}{delta:.2f}" if score == score else "N/A"
    s_str = f"{score:.2f}" if score == score else "N/A"
    print(f"{label:<32s}  {s_str:>11s}  {d_str:>7s}")

# Markdown version
print("\nMarkdown:\n")
print(f"| Operator basis | Avg. Bench Acc. (%) | Δ (%) |")
print(f"|----------------|:-------------------:|:-----:|")
for label in LABEL_ORDER:
    score = GRAPHANY_BENCH_AVG.get(label, float("nan"))
    delta = score - baseline_score
    sign  = "+" if delta >= 0 else ""
    d_str = f"{sign}{delta:.2f}" if score == score else "N/A"
    s_str = f"{score:.2f}" if score == score else "N/A"
    print(f"| {label} | {s_str} | {d_str} |")

# %% [markdown]
# ## Table 2 — GOBLIN 25-benchmark results (GOBLIN column only)
#
# Alphabetical order matching paper.  Other method columns (MeanGNN, GAT,
# GraphAny, TS-Mean) are reproduced from prior work and not shown here.

# %%
def fmt(ds: str) -> str:
    m, s = goblin_mean.get(ds, float("nan")), goblin_std.get(ds, float("nan"))
    n = len([x for x in goblin_accs[ds] if x == x])
    if m != m:
        return "N/A"
    note = f" (n={n})" if n < len(GOBLIN_SEEDS) else ""
    return f"{m:.2f} ± {s:.2f}{note}"


# Average ± std reported in paper = mean of per-dataset means ± mean of per-dataset stds.
# (The ± reflects per-dataset reproducibility, not variability of the overall average.)
bench_avg     = float(np.nanmean([goblin_mean[ds] for ds in BENCH_DS]))
bench_avg_std = float(np.nanmean([goblin_std[ds]  for ds in BENCH_DS]))

print("Table 2 — GOBLIN 25-benchmark results (mean ± std, 5 seeds)\n")
print(f"{'Dataset':<18s}  {'GOBLIN (%)':>14s}")
print("-" * 36)
for ds in BENCH_DS:
    print(f"{ds:<18s}  {fmt(ds):>14s}")
print("-" * 36)
print(f"{'Average':<18s}  {bench_avg:>8.2f} ± {bench_avg_std:4.2f}")

# Markdown version
print("\nMarkdown:\n")
print("| Dataset | GOBLIN (%) |")
print("|---------|:----------:|")
for ds in BENCH_DS:
    print(f"| {ds} | {fmt(ds)} |")
print(f"| **Average** | **{bench_avg:.2f} ± {bench_avg_std:.2f}** |")

# %% [markdown]
# ## Table 3 — CityNetworks results
#
# Columns: MeanGNN, GAT, GraphAny, TS-GAT, GOBLIN.
# TS-GAT values from run_tsgnn_exps.py results.

# %%
print("Table 3 — CityNetworks results (mean ± std %)\n")
header = f"{'':12s}  {'MeanGNN':>14s}  {'GAT':>14s}  {'GraphAny':>14s}  {'TS-GAT':>12s}  {'GOBLIN':>14s}"
print(header)
print("-" * len(header))

city_labels = {"CityParis": "Paris", "CityShanghai": "Shanghai",
               "CityLA": "LA", "CityLondon": "London"}

all_means = {m: [] for m in ["meangnn", "gat", "graphany", "tsgat", "goblin"]}
all_stds  = {m: [] for m in ["meangnn", "gat", "graphany", "tsgat", "goblin"]}

for city in CITY_DS:
    label = city_labels[city]

    gnn_m = load_gnn_city("meangnn", city)
    gnn_g = load_gnn_city("gat", city)
    ga    = load_graphany_city(city)
    tsgat = get_tsgat_city(city)
    gob_m = goblin_mean[city]
    gob_s = goblin_std[city]

    def f2(pair): return f"{pair[0]:5.2f} ± {pair[1]:4.2f}" if pair else "   N/A       "
    def fl(accs): return f"{np.mean(accs):5.2f} ± {np.std(accs):4.2f}" if accs else "   N/A       "
    tsgat_str = f2(tsgat) if tsgat else "     N/A     "

    print(f"{label:12s}  {f2(gnn_m):>14s}  {f2(gnn_g):>14s}  {fl(ga):>14s}  "
          f"{tsgat_str:>12s}  {gob_m:>6.2f} ± {gob_s:>4.2f}")

    if gnn_m:  all_means["meangnn"].append(gnn_m[0]); all_stds["meangnn"].append(gnn_m[1])
    if gnn_g:  all_means["gat"].append(gnn_g[0]); all_stds["gat"].append(gnn_g[1])
    if ga:     all_means["graphany"].append(np.mean(ga)); all_stds["graphany"].append(np.std(ga))
    if tsgat:  all_means["tsgat"].append(tsgat[0]); all_stds["tsgat"].append(tsgat[1])
    all_means["goblin"].append(gob_m); all_stds["goblin"].append(gob_s)

print("-" * len(header))

def avg_fmt(m):
    if not all_means[m]:
        return "  N/A "
    return f"{np.mean(all_means[m]):5.2f} ± {np.mean(all_stds[m]):4.2f}"

print(f"{'Average':12s}  {avg_fmt('meangnn'):>14s}  {avg_fmt('gat'):>14s}  "
      f"{avg_fmt('graphany'):>14s}  {avg_fmt('tsgat'):>12s}  {avg_fmt('goblin'):>14s}")

# Markdown version
print("\nMarkdown:\n")
print("| | MeanGNN | GAT | GraphAny | TS-GAT | **GOBLIN** |")
print("|---|:---:|:---:|:---:|:---:|:---:|")
for city in CITY_DS:
    label = city_labels[city]
    gnn_m = load_gnn_city("meangnn", city)
    gnn_g = load_gnn_city("gat", city)
    ga    = load_graphany_city(city)
    tsgat = get_tsgat_city(city)
    def md(pair): return f"{pair[0]:.2f}±{pair[1]:.2f}" if pair else "N/A"
    def mdl(accs): return f"{np.mean(accs):.2f}±{np.std(accs):.2f}" if accs else "N/A"
    ts = md(tsgat) if tsgat else "N/A"
    print(f"| {label} | {md(gnn_m)} | {md(gnn_g)} | {mdl(ga)} | {ts} | "
          f"**{goblin_mean[city]:.2f}±{goblin_std[city]:.2f}** |")
def md_avg(m):
    if not all_means[m]:
        return "N/A"
    return f"{np.mean(all_means[m]):.2f}±{np.mean(all_stds[m]):.2f}"

print(f"| **Average** | {md_avg('meangnn')} | {md_avg('gat')} | "
      f"{md_avg('graphany')} | {md_avg('tsgat')} | "
      f"**{md_avg('goblin')}** |")

# %% [markdown]
# ## Numeric summary — HopSign per-k

# %%
W = 40
print(f"\n{'Model':{W}s}  " + "  ".join(f"k={k}" for k in KS) + "  Avg")
print("-" * (W + 2 + 8 * 7 + 5))
for label, vals in GRAPHANY_HOPSIGN.items():
    print(f"{label:{W}s}  " + "  ".join(f"{v:5.1f}" for v in vals)
          + f"  {np.nanmean(vals):.1f}")
if "TS-GAT" in tsgnn_hopsign:
    row_vals = [tsgnn_hopsign["TS-GAT"][k][0] for k in KS]
    print(f"{'TS-GAT':{W}s}  " + "  ".join(f"{v:5.1f}" for v in row_vals)
          + f"  {np.mean(row_vals):.1f}")
print(f"{'GOBLIN':{W}s}  " + "  ".join(f"{v:5.1f}" for v in GOBLIN_HOPSIGN_MEANS)
      + f"  {np.mean(GOBLIN_HOPSIGN_MEANS):.1f}")

# %%
