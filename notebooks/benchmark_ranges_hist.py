# %%
# Setup
import os
import sys
import gzip
import types
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# Stub dgl before goblin is imported (avoids CUDA library errors on login nodes)
if "dgl" not in sys.modules:
    sys.modules["dgl"] = types.ModuleType("dgl")

# Navigate to repo root and add to sys.path so goblin package is importable
ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
while not (ROOT / "goblin").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRAPHANY_RANGE = Path("/data-gauss/bengut/GraphAny-Range")
KHOP_DIR = GRAPHANY_RANGE / "data_cache" / "citynetwork_apspd"
GOBLIN_RESULTS = ROOT / "output" / "results" / "goblin"
CITY_CACHE = ROOT / "data_cache" / "benchmark_ranges" / "city_ranges.pt"

# %%
# Load existing benchmark ranges (benchmark + HopSign datasets)
bench_dir = GRAPHANY_RANGE / "data_cache" / "benchmark_ranges"
ranges = {}
for fn in os.listdir(bench_dir):
    if fn.endswith(".pt"):
        d = torch.load(bench_dir / fn, weights_only=False)
        if isinstance(d, dict):
            ranges |= d

# Check for expected datasets
benchmark_datasets = [
    "AirBrazil", "AirUS", "AirEU", "Cornell", "Texas", "Wisconsin",
    "Chameleon", "Wiki", "Squirrel", "Actor", "Citeseer", "BlogCatalog",
    "WkCS", "Tolokers", "AmzComp", "AmzPhoto", "Minesweeper", "DBLP",
    "CoCS", "Pubmed", "FCora", "Roman", "AmzRatings", "CoPhysics", "Questions",
] + [f"{k}HopSign" for k in range(1, 9)]

for ds in benchmark_datasets:
    if ds not in ranges:
        print(f"Missing range for {ds}")
    elif np.isnan(ranges[ds]["mean"]):
        print(f"NaN range for {ds}")


# %%
# Helpers for CityNetworks range computation

def load_adj(city):
    """Load 1-hop adjacency from khop file. Returns (src, dst, N) both directions."""
    p = KHOP_DIR / f"{city}_khop_01.pt"
    with gzip.open(str(p), "rb") as f:
        d = torch.load(f, weights_only=True)
    row, col = d["row"].long(), d["col"].long()
    return torch.cat([row, col]), torch.cat([col, row]), d["N"]


def build_csr_ptr(src_sorted, N):
    """Build CSR row-pointer array from a sorted source tensor."""
    deg = torch.bincount(src_sorted, minlength=N)
    return torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)])


def single_source_bfs(u, a_dst, ptr, N, max_k=16):
    """
    BFS from node u using CSR adjacency (ptr, a_dst). Returns distance vector
    of length N; -1 means unreachable within max_k hops.
    """
    dist = torch.full((N,), -1, dtype=torch.long)
    dist[u] = 0
    frontier = torch.tensor([u], dtype=torch.long)
    for k in range(1, max_k + 1):
        lo = ptr[frontier]
        hi = ptr[frontier + 1]
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            break
        block_id = torch.repeat_interleave(torch.arange(len(frontier)), counts)
        cum = torch.zeros(len(counts) + 1, dtype=torch.long)
        cum[1:] = counts.cumsum(0)
        local_off = torch.arange(total) - cum[block_id]
        nbrs = a_dst[lo[block_id] + local_off]
        new_nodes = nbrs[dist[nbrs] == -1].unique()
        if new_nodes.numel() == 0:
            break
        dist[new_nodes] = k
        frontier = new_nodes
    return dist


def heat_range_sparse(city, tau=5.0, N_eff=None, K=10, seed=42):
    """
    Estimate the mean weighted range of heat(tau) on a city graph.
    Uses sparse Taylor expansion exp(-tau * L_sym) + BFS distances.
    Returns (mean_rho, std_rho).
    """
    src, dst, N = load_adj(city)

    if N_eff is None:
        N_eff = min(200, max(100, N // 1000))

    # Degree & D^{-1/2} for L_sym = I - D^{-1/2} A D^{-1/2}
    deg = torch.bincount(src, minlength=N).float()
    d_invsqrt = (deg + 1e-8).sqrt().reciprocal()

    # Build sorted CSR adjacency
    order = torch.argsort(src)
    a_src = src[order]
    a_dst = dst[order]
    ptr = build_csr_ptr(a_src, N)

    # Sample source nodes
    gen = torch.Generator().manual_seed(seed)
    sample = torch.randperm(N, generator=gen)[:N_eff]

    rhos = []
    for u_t in tqdm(sample, desc=f"{city} heat(τ={tau})"):
        u = u_t.item()

        # BFS distances from u (up to 16 hops)
        dist = single_source_bfs(u, a_dst, ptr, N, max_k=16)

        # Sparse Taylor expansion: S_u = exp(-tau * L_sym) @ e_u
        # L_sym @ v = v - d_invsqrt * (A @ (d_invsqrt * v))
        x = torch.zeros(N)
        x[u] = 1.0
        result = x.clone()
        term = x.clone()
        for k in range(1, K + 1):
            Av = torch.zeros(N)
            Av.scatter_add_(0, a_src, (d_invsqrt * term)[a_dst])
            term = (-tau / k) * (term - d_invsqrt * Av)
            result = result + term

        S_row = result.abs()
        total_weight = S_row.sum().item()
        if total_weight < 1e-12:
            continue

        # dist = -1 for unreachable → treat as 0 (negligible heat kernel weight there)
        d_float = dist.float().clamp(min=0)
        rho = (S_row * d_float).sum().item() / total_weight
        rhos.append(rho)

    return float(np.mean(rhos)), float(np.std(rhos))


# %%
# Compute CityNetworks ranges (with caching)

CITY_DATASETS = ["CityParis", "CityShanghai", "CityLA", "CityLondon"]

if CITY_CACHE.exists():
    city_ranges = torch.load(CITY_CACHE, weights_only=False)
    print(f"Loaded cached city ranges from {CITY_CACHE}")
    for city, v in city_ranges.items():
        print(f"  {city}: {v['mean']:.3f} ± {v['std']:.3f}")
else:
    heat_cache = {}  # (city, tau) -> mean_rho

    city_ranges = {}
    for city in CITY_DATASETS:
        seed_ranges = []

        for hash_dir in sorted(GOBLIN_RESULTS.iterdir()):
            if not hash_dir.is_dir():
                continue
            pt_path = hash_dir / f"{city}.pt"
            if not pt_path.exists():
                continue
            try:
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  [warn] {pt_path}: {e}")
                continue

            city_data = d.get(city, {})
            basis = city_data.get("basis", [])
            alpha = city_data.get("test/mean_alpha")
            if alpha is None or not basis:
                continue

            op_rhos = []
            for op in basis:
                if op.family == "gaussian":
                    op_rhos.append(float(op.param))
                elif op.family == "fixed" and op.param == "L1":
                    op_rhos.append(1.0)
                elif op.family == "fixed" and op.param == "L2":
                    op_rhos.append(2.0)
                elif op.family == "heat":
                    tau = float(op.param)
                    key = (city, tau)
                    if key not in heat_cache:
                        mean_rho, _ = heat_range_sparse(city, tau=tau)
                        heat_cache[key] = mean_rho
                    op_rhos.append(heat_cache[key])
                else:
                    print(f"  [warn] unknown operator {op}, skipping")
                    op_rhos.append(float("nan"))

            alpha_np = alpha.float().numpy()
            overall = float(np.dot(alpha_np, op_rhos))
            seed_ranges.append(overall)

        if seed_ranges:
            city_ranges[city] = {
                "mean": float(np.mean(seed_ranges)),
                "std": float(np.std(seed_ranges)),
            }
            print(f"{city}: {city_ranges[city]['mean']:.3f} ± {city_ranges[city]['std']:.3f}  ({len(seed_ranges)} seeds)")

    CITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(city_ranges, CITY_CACHE)
    print(f"Saved city ranges to {CITY_CACHE}")


# %%
# Merge all ranges
all_ranges = {**ranges, **city_ranges}


# %%
# Full plot – all benchmark + HopSign + CityNetworks

names = sorted(k for k in all_ranges if "HopSign" not in k and "City" not in k)
city_names = ["CityLA", "CityLondon", "CityParis", "CityShanghai"]
hop_names = sorted(k for k in all_ranges if "HopSign" in k)
ordered = names + city_names + hop_names

means = [all_ranges[k]["mean"] for k in ordered]
stds  = [all_ranges[k]["std"]  for k in ordered]
cols  = [
    "tab:orange" if "HopSign" in k
    else "tab:red"   if "City" in k
    else "tab:blue"
    for k in ordered
]

fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(ordered, means, xerr=stds, capsize=3, color=cols)
ax.set_xlabel("Range")
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("benchmark_ranges.pdf", bbox_inches="tight")
plt.show()


# %%
# Small plot for paper – sample of benchmark + city + HopSign
real_keys = [k for k in all_ranges if "HopSign" not in k]
avg_mean = float(np.mean([all_ranges[k]["mean"] for k in real_keys]))
avg_std  = float(np.mean([all_ranges[k]["std"]  for k in real_keys]))

small_names = [
    "Actor", "Citeseer", "Roman", "AirBrazil", "Chameleon", "Squirrel",
    "CityShanghai", "CityParis", "CityLA", "CityLondon",
    "Average", "2HopSign", "4HopSign", "6HopSign", "8HopSign",
]
small_means = [avg_mean if k == "Average" else all_ranges[k]["mean"] for k in small_names]
small_stds  = [avg_std  if k == "Average" else all_ranges[k]["std"]  for k in small_names]
small_cols  = [
    "tab:orange" if "HopSign" in k
    else "tab:green" if k == "Average"
    else "tab:red"   if "City" in k
    else "tab:blue"
    for k in small_names
]

fig, ax = plt.subplots(figsize=(1.7, 2.8))
ax.barh(small_names, small_means, xerr=small_stds, capsize=2,
        color=small_cols, height=0.85)
ax.set_xlabel("Range", fontsize=8)
ax.tick_params(axis="both", labelsize=7)
ax.invert_yaxis()
ax.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.4)
plt.savefig("benchmark_ranges_small.pdf", bbox_inches="tight")
plt.show()
# %%
