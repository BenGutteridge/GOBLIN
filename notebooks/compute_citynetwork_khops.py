"""
CityNetworks All-Pairs Shortest-Path Distances (k=1..16)
=========================================================
For each of the four CityNetworks graphs (paris, shanghai, la, london),
computes and saves 16 sparse files:

    DATA_CACHE/citynetwork_apspd/{CityName}_khop_{k:02d}.pt   (k = 1 … 16)

Each .pt is a gzip-compressed torch.save dict:
    "row" : int32 source node indices  (upper triangle only: row < col)
    "col" : int32 target node indices
    "N"   : int    number of nodes

Load back with:
    import gzip, torch
    with gzip.open("CityParis_khop_03.pt", "rb") as f:
        d = torch.load(f)
    N = d["N"]
    row, col = d["row"].long(), d["col"].long()
    # full symmetric pairs (both directions):
    all_row = torch.cat([row, col])
    all_col = torch.cat([col, row])

Checkpointing: files are written after each k. An interrupted run
resumes automatically from the last completed k.

Verification (VERIFY_CORA):
    When VERIFY_CORA=True, generates k-hop files for Cora and exactly
    compares the reconstructed distance matrix against apspd_to_tensor(Cora).
    This validates the k-hop pipeline against a known-correct reference.
"""

import gc
import gzip
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

# ── Paths via goblin config ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goblin.config import DATA_CACHE

CACHE_DIR = DATA_CACHE / "citynetwork_apspd"
DATA_DIR  = DATA_CACHE / "city_networks"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

from torch_geometric.datasets import CityNetwork

# ── Datasets to process ───────────────────────────────────────────────────────
# Comment out any entry to skip it. Already-completed datasets are always skipped
# automatically regardless.
DATASETS = [
    # "CityParis",
    # "CityShanghai",
    # "CityLA",
    # "CityLondon",
]

MAX_K = 16

# (N, undirected_edges) – used only for the upfront size estimate
DATASET_STATS = {
    "CityParis":    (114_127,  182_511),
    "CityShanghai": (183_917,  262_092),
    "CityLA":       (240_587,  341_523),
    "CityLondon":   (568_795,  756_502),
}

_CITY_TO_PYG = {
    "CityParis": "paris",
    "CityShanghai": "shanghai",
    "CityLA": "la",
    "CityLondon": "london",
}


def load_pyg_data(name: str):
    """Return a PyG Data object for the given dataset name."""
    if name in _CITY_TO_PYG:
        return CityNetwork(root=str(DATA_DIR), name=_CITY_TO_PYG[name])[0]
    raise ValueError(f"Unknown dataset: {name}")


# ── Pre-flight estimation ─────────────────────────────────────────────────────
def print_size_estimates(datasets=None):
    """
    Estimate per-k file sizes and peak RAM before running anything.
    """
    if datasets is None:
        datasets = DATASETS
    if not datasets:
        return
    SEP = "─" * 70
    print(SEP)
    print(f"{'PRE-FLIGHT SIZE & MEMORY ESTIMATES':^70}")
    print(SEP)
    print("  Model: 2-D road network → frontier at k ≈ k × avg_deg nodes per source")
    print("         File compression ≈ 3× (gzip on sorted int32 pairs)")
    print()

    for city in datasets:
        N, E = DATASET_STATS[city]
        avg_deg = 2 * E / N
        print(f"  ┌─ {city.upper()}  N={N:,}  E={E:,}  avg_deg={avg_deg:.2f}")
        print(f"  │   {'k':>3}  {'~pairs':>12}  {'raw MB':>9}  {'compressed MB':>15}")

        total_nnz   = 0
        total_comp  = 0.0
        frontier_ks = []

        for k in range(1, MAX_K + 1):
            nnz     = int(N * k * avg_deg / 2)
            raw_mb  = nnz * 2 * 4 / 1e6
            comp_mb = raw_mb / 3.0
            total_nnz  += nnz
            total_comp += comp_mb
            frontier_ks.append(nnz)
            print(f"  │   {k:3d}  {nnz:>12,}  {raw_mb:>9.1f}  {comp_mb:>15.1f}")

        reached_mb  = total_nnz * 2 * 8 / 1e6
        frontier_mb = frontier_ks[-1] * 2 * 8 / 1e6
        peak_ram_gb = (reached_mb + 2 * frontier_mb) / 1e3

        print(f"  │")
        print(f"  │   Total compressed : ~{total_comp / 1e3:.2f} GB")
        print(f"  └─  Peak RAM estimate: ~{peak_ram_gb:.1f} GB  (reached_flat + 2×frontier)")
        print()

    print(SEP)
    print()


print_size_estimates()


# ── Storage helpers ───────────────────────────────────────────────────────────
def khop_path(city: str, k: int) -> Path:
    return CACHE_DIR / f"{city}_khop_{k:02d}.pt"


def save_khop(city: str, k: int, src: torch.Tensor, dst: torch.Tensor, N: int) -> float:
    """
    Save upper-triangle pairs (src < dst) as gzip-compressed .pt. Returns MB on disk.
    src, dst: 1-D int64 tensors of the new k-hop shell (both directions included).
    """
    mask = src < dst
    d = {
        "row": src[mask].to(torch.int32),
        "col": dst[mask].to(torch.int32),
        "N":   N,
    }
    out = khop_path(city, k)
    with gzip.open(str(out), "wb") as f:
        torch.save(d, f)
    return out.stat().st_size / 1e6


def load_khop(city: str, k: int):
    """Load k-hop shell. Returns (src, dst) int64 tensors with BOTH directions."""
    with gzip.open(str(khop_path(city, k)), "rb") as f:
        d = torch.load(f, weights_only=True)
    row = d["row"].long()
    col = d["col"].long()
    return torch.cat([row, col]), torch.cat([col, row])


def completed_ks(city: str) -> set:
    return {k for k in range(1, MAX_K + 1) if khop_path(city, k).exists()}


# ── BFS helpers ───────────────────────────────────────────────────────────────
def build_adj_sorted(data) -> tuple:
    """
    Build symmetrised, self-loop-free adjacency sorted by source node.
    Returns (a_src, a_dst, N): 1-D int64 tensors, sorted by a_src.
    """
    N  = data.num_nodes
    ei = data.edge_index
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    ei = torch.unique(ei, dim=1)
    order = torch.argsort(ei[0])
    ei = ei[:, order]
    return ei[0], ei[1], N


def expand_one_hop(f_src, f_dst, a_src_sorted, a_dst_sorted):
    lo = torch.searchsorted(a_src_sorted, f_dst, right=False)
    hi = torch.searchsorted(a_src_sorted, f_dst, right=True)
    counts = hi - lo

    total = int(counts.sum())
    if total == 0:
        empty = f_src.new_empty(0)
        return empty, empty

    block_id = torch.repeat_interleave(torch.arange(len(f_src), device=f_src.device), counts)
    cum = torch.zeros(len(counts) + 1, dtype=torch.long, device=f_src.device)
    cum[1:] = counts.cumsum(0)
    local_off = torch.arange(total, device=f_src.device) - cum[block_id]
    return f_src[block_id], a_dst_sorted[lo[block_id] + local_off]


def filter_new_frontier(new_src, new_dst, reached_flat, N):
    flat = new_src * N + new_dst
    no_self = new_src != new_dst
    flat = flat[no_self]; new_src = new_src[no_self]; new_dst = new_dst[no_self]

    if flat.numel() == 0:
        return new_src, new_dst, flat

    flat = torch.unique(flat)
    new_src = flat // N
    new_dst = flat % N

    if reached_flat.numel() > 0:
        not_reached = ~torch.isin(flat, reached_flat)
        flat = flat[not_reached]; new_src = new_src[not_reached]; new_dst = new_dst[not_reached]

    return new_src, new_dst, flat


# ── Main BFS computation ──────────────────────────────────────────────────────
def compute_city(city: str, data=None):
    """
    Compute k-hop shell files for `city` up to MAX_K.
    `data`: optional pre-loaded PyG Data object (avoids re-loading).
    """
    done = completed_ks(city)

    if len(done) == MAX_K:
        total = sum(khop_path(city, k).stat().st_size for k in range(1, MAX_K + 1)) / 1e6
        print(f"[{city}] All {MAX_K} k-hops already computed. "
              f"Total on disk: {total:.0f} MB. Skipping.")
        return

    remaining = sorted(set(range(1, MAX_K + 1)) - done)
    print(f"\n{'=' * 65}")
    print(f"[{city.upper()}]")
    print(f"  Completed : {sorted(done)}")
    print(f"  Remaining : {remaining}")

    t0 = time.perf_counter()
    print(f"  Loading {city!r} …")
    if data is None:
        data = load_pyg_data(city)
    a_src, a_dst, N = build_adj_sorted(data)
    print(f"  N={N:,}  adj edges={len(a_src):,}  ({time.perf_counter()-t0:.1f}s)")
    del data

    start_k = min(remaining)

    if start_k == 1:
        f_src, f_dst = a_src.clone(), a_dst.clone()
        mb = save_khop(city, 1, f_src, f_dst, N)
        print(f"  k= 1 saved — {len(f_src) // 2:,} pairs — {mb:.1f} MB")
        reached_flat = torch.unique(f_src * N + f_dst)
        start_k = 2
    else:
        print(f"  Resuming at k={start_k}. Rebuilding state from k=1..{start_k - 1} …")
        reached_flat = torch.empty(0, dtype=torch.long)
        for k_prev in tqdm(range(1, start_k), desc="  loading prior k-hops", leave=False):
            s, d = load_khop(city, k_prev)
            reached_flat = torch.unique(torch.cat([reached_flat, torch.unique(s * N + d)]))
        f_src, f_dst = load_khop(city, start_k - 1)

    step_times = []
    pbar = tqdm(range(start_k, MAX_K + 1), desc=f"  [{city}] BFS")

    for k in pbar:
        if k in done:
            f_src, f_dst = load_khop(city, k)
            pbar.set_postfix(status=f"k={k} already done")
            continue

        t_step = time.perf_counter()
        raw_src, raw_dst = expand_one_hop(f_src, f_dst, a_src, a_dst)
        new_src, new_dst, new_flat = filter_new_frontier(raw_src, raw_dst, reached_flat, N)
        del raw_src, raw_dst

        reached_flat = torch.unique(torch.cat([reached_flat, new_flat]))
        del new_flat

        mb      = save_khop(city, k, new_src, new_dst, N)
        elapsed = time.perf_counter() - t_step
        step_times.append(elapsed)
        eta_s   = (MAX_K - k) * float(torch.tensor(step_times[-5:]).mean())
        pbar.set_postfix(
            pairs=f"{len(new_src) // 2:,}", size_mb=f"{mb:.1f}",
            step_s=f"{elapsed:.1f}", eta_min=f"{eta_s / 60:.1f}",
        )
        f_src, f_dst = new_src, new_dst
        del new_src, new_dst
        gc.collect()

    del reached_flat, f_src, f_dst, a_src, a_dst
    gc.collect()

    total = sum(khop_path(city, k).stat().st_size for k in range(1, MAX_K + 1)) / 1e6
    print(f"\n  [{city.upper()}] DONE — total on disk: {total:.0f} MB\n")


# ── Run ───────────────────────────────────────────────────────────────────────
for name in DATASETS:
    compute_city(name)

if DATASETS:
    print("✓ All datasets complete.")


# ── Sanity checks ─────────────────────────────────────────────────────────────
def load_all_khops_as_dist_matrix(name: str, N: int):
    """Reconstruct a dense (N,N) distance matrix from the k-hop files."""
    dist = torch.full((N, N), -1, dtype=torch.long)
    dist.fill_diagonal_(0)
    for k in range(1, MAX_K + 1):
        p = khop_path(name, k)
        if not p.exists():
            break
        src, dst = load_khop(name, k)
        dist[src, dst] = k
    return dist


def single_source_bfs(src_node, a_src_sorted, a_dst_sorted, N, max_k=MAX_K):
    dist = torch.full((N,), -1, dtype=torch.long)
    dist[src_node] = 0
    frontier = torch.tensor([src_node], dtype=torch.long)
    for k in range(1, max_k + 1):
        lo = torch.searchsorted(a_src_sorted, frontier, right=False)
        hi = torch.searchsorted(a_src_sorted, frontier, right=True)
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            break
        block_id = torch.repeat_interleave(torch.arange(len(frontier)), counts)
        cum = torch.zeros(len(counts) + 1, dtype=torch.long)
        cum[1:] = counts.cumsum(0)
        local_off = torch.arange(total) - cum[block_id]
        nbrs = a_dst_sorted[lo[block_id] + local_off]
        new_nbrs = torch.unique(nbrs[dist[nbrs] == -1])
        if new_nbrs.numel() == 0:
            break
        dist[new_nbrs] = k
        frontier = new_nbrs
    return dist


import random

SEP = "─" * 65
sanity_targets = [ds for ds in DATASETS if all(khop_path(ds, k).exists() for k in range(1, MAX_K + 1))]

if sanity_targets:
    print(SEP)
    print(f"{'SANITY CHECKS':^65}")
    print(SEP)

    for ds in sanity_targets:
        print(f"\n[{ds}]")
        with gzip.open(str(khop_path(ds, 1)), "rb") as f:
            N = torch.load(f, weights_only=True)["N"]

        # Check 1: shells disjoint
        all_flats = [torch.unique(load_khop(ds, k)[0] * N + load_khop(ds, k)[1])
                     for k in range(1, MAX_K + 1)]
        combined = torch.cat(all_flats)
        if torch.unique(combined).numel() == combined.numel():
            print("  ✓ shells are disjoint")
        else:
            print(f"  FAIL: {combined.numel() - torch.unique(combined).numel():,} duplicate pairs")
        del all_flats, combined

        # Check 2: no self-loops
        if not any((load_khop(ds, k)[0] == load_khop(ds, k)[1]).any() for k in range(1, MAX_K + 1)):
            print("  ✓ no self-loops in any shell")
        else:
            print("  FAIL: self-loop found")

        # Check 3: symmetric
        sym_ok = all(
            torch.equal(
                torch.sort(torch.unique(load_khop(ds, k)[0] * N + load_khop(ds, k)[1])).values,
                torch.sort(torch.unique(load_khop(ds, k)[1] * N + load_khop(ds, k)[0])).values,
            )
            for k in range(1, MAX_K + 1)
        )
        print("  ✓ all shells symmetric" if sym_ok else "  FAIL: asymmetry detected")

        # Check 4: BFS spot-check (20 random sources)
        print(f"  spot-checking 20 random single-source BFS …")
        data = load_pyg_data(ds)
        a_src, a_dst, _ = build_adj_sorted(data)
        del data
        random.seed(42)
        recon = load_all_khops_as_dist_matrix(ds, N)
        all_ok = True
        for v in random.sample(range(N), min(20, N)):
            ref = single_source_bfs(v, a_src, a_dst, N)
            reachable = (ref > 0) & (ref <= MAX_K)
            if not torch.equal(recon[v][reachable], ref[reachable]):
                print(f"  FAIL at source {v}")
                all_ok = False
                break
        if all_ok:
            print("  ✓ BFS spot-check passed for 20 sampled sources")

    print(f"\n{SEP}")
    print("Sanity checks complete.")


# ── Cora verification ─────────────────────────────────────────────────────────
# Generates k-hop files for Cora (N=2708, tiny) and exactly compares the
# reconstructed distance matrix against apspd_to_tensor(Cora).
# This validates that the k-hop BFS pipeline is correct.
VERIFY_CORA = True  # set False to skip

if VERIFY_CORA:
    from torch_geometric.datasets import Planetoid
    from goblin.data import apspd_to_tensor

    print(f"\n{SEP}")
    print(f"{'CORA VERIFICATION':^65}")
    print(f"{SEP}")

    cora_data = Planetoid(root=str(DATA_DIR / "Cora"), name="Cora")[0]
    N_cora = cora_data.num_nodes
    print(f"  Cora: N={N_cora}")

    # Generate k-hop files for Cora (same pipeline as city networks)
    compute_city("Cora", data=cora_data)

    # Reconstruct distance matrix from k-hop files
    recon = load_all_khops_as_dist_matrix("Cora", N_cora)

    # Compute ground-truth dense APSPD
    print("  Computing ground-truth APSPD via NetworkX …")
    true_apspd = apspd_to_tensor(cora_data).long()

    # Exact comparison for pairs reachable within MAX_K
    mask = (true_apspd > 0) & (true_apspd <= MAX_K)
    n_reachable = int(mask.sum())
    n_match = int((recon[mask] == true_apspd[mask]).sum())
    n_missing = int(((recon == -1) & mask).sum())
    n_mismatch = int(((recon[mask] != true_apspd[mask]) & (recon[mask] != -1)).sum())

    if n_match == n_reachable and n_missing == 0:
        print(f"  ✓ VERIFY OK — exact match on all {n_reachable:,} reachable pairs (k=1..{MAX_K})")
    else:
        print(f"  FAIL — {n_match:,}/{n_reachable:,} match, "
              f"{n_missing:,} missing, {n_mismatch:,} mismatch")

    print(f"{SEP}\n")
