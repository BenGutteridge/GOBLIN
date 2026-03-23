# %% [markdown]
# # GraphAny Operator Variants — Table 1, Figure 1
#
# Evaluates 5 operator bases on HopSign and 25 benchmarks (trains on Cora).
#
# Usage:
#   python notebooks/run_graphany_exps.py
#   python notebooks/run_graphany_exps.py --variant 2  # single variant

# %% [markdown]
# ## Imports and paths

# %%
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent if "__file__" in dir() else Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goblin.config import DATA_CACHE

# %% [markdown]
# ## Variant definitions
#
# Each entry specifies the graphany/run.py arguments for one operator basis.
# variant_id=0 uses the pre-trained Cora checkpoint (total_steps=0).
# Others train from scratch on Cora (total_steps > 0).

# %%
VARIANTS = [
    # id=0 — standard GraphAny (L1+L2+H1+H2), inference-only from Cora checkpoint
    {
        "id": 0,
        "name": "graphany",
        "extra_args": [
            "prev_ckpt=checkpoints/graph_any_cora.pt",
            "total_steps=0",
        ],
    },
    # id=1 — extended lazy random walk (L1+L2+L3+L4 + high-pass H1+H2)
    {
        "id": 1,
        "name": "graphany_extended",
        "extra_args": [
            "feat_chn=X+L1+L2+L3+L4+H1+H2",
            "pred_chn=X+L1+L2+L3+L4",
            "entropy=1",
            "attn_temp=5",
            "n_hidden=128",
            "n_mlp_layer=3",
            "total_steps=1500",
        ],
    },
    # id=2 — exact k-hop shell operators (N1, N2, N3, N4)
    {
        "id": 2,
        "name": "n1_4",
        "extra_args": [
            "feat_chn=X+N1+N2+N3+N4",
            "pred_chn=X+N1+N2+N3+N4",
            "entropy=2",
            "attn_temp=5",
            "n_hidden=64",
            "n_mlp_layer=3",
            "total_steps=1500",
        ],
    },
    # id=3 — binned hops (near/mid/far bins + range indicator)
    {
        "id": 3,
        "name": "hop_bins",
        "extra_args": [
            "feat_chn=X+N1+N2+N3d+NdR",
            "pred_chn=X+N1+N2+N3d+NdR",
            "entropy=2",
            "attn_temp=5",
            "n_hidden=128",
            "n_mlp_layer=3",
            "total_steps=1000",
        ],
    },
    # id=4 — heat kernel (short + medium + long range heat diffusion)
    {
        "id": 4,
        "name": "heat_kernel",
        "extra_args": [
            "feat_chn=X+N1+DS+DM+DL",
            "pred_chn=X+N1+DS+DM+DL",
            "entropy=2",
            "attn_temp=5",
            "n_hidden=64",
            "n_mlp_layer=2",
            "total_steps=1500",
        ],
    },
]

# %% [markdown]
# ## Argument parsing

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=int, default=None,
                    help="Variant index 0–4. If not set, runs all variants sequentially.")
parser.add_argument("--seed", type=int, default=0, help="Training seed (default 0)")
args, _ = parser.parse_known_args()

variants_to_run = [VARIANTS[args.variant]] if args.variant is not None else VARIANTS
seed = args.seed

# %% [markdown]
# ## Evaluation datasets and output dirs

# %%
# Two evaluation targets
EVAL_TARGETS = [
    {
        "dataset": "CoraXHopSign",
        "out_subdir": "camera_ready/graphany_ops_hopsign",
    },
    {
        "dataset": "CoraX25Benchmarks",
        "out_subdir": "camera_ready/graphany_ops_25bench",
    },
]

# %% [markdown]
# ## Run variants

# %%
def get_feat_chn(variant: dict) -> str:
    """Extract feat_chn from a variant's extra_args, falling back to the standard GraphAny default."""
    for arg in variant["extra_args"]:
        if arg.startswith("feat_chn="):
            return arg.split("=", 1)[1]
    return "X+L1+L2+H1+H2"


def find_cached_result(variant: dict, dataset: str, out_subdir: str):
    """Return path to existing result JSON if present, else None.

    graphany/run.py writes:
      {DATA_CACHE}/{out_subdir}/{dataset}/{dataset}_feat={feat_chn}_pred=*_{hash}_results.json
    We glob on feat_chn (hash is not known until the run itself computes it).
    """
    feat_chn = get_feat_chn(variant)
    result_dir = DATA_CACHE / out_subdir / dataset
    matches = list(result_dir.glob(f"{dataset}_feat={feat_chn}_pred=*_results.json"))
    return matches[0] if matches else None


def run_variant(variant: dict, dataset: str, out_subdir: str, seed: int):
    """Run graphany/run.py for one (variant, dataset) pair.

    If a result JSON already exists (from a previous run), load and print a
    summary of the cached results rather than re-running inference.
    """
    import json as _json
    import numpy as _np

    cached = find_cached_result(variant, dataset, out_subdir)
    if cached is not None:
        data = _json.load(open(cached))
        # Summarise cached metrics for this dataset type
        if "HopSign" in dataset:
            hop_accs = [data.get(f"hopsR/{k}hop_test_acc", float("nan")) for k in range(1, 9)]
            valid = [x for x in hop_accs if x == x]
            summary = f"avg={_np.mean(valid):.1f}%  k=1: {hop_accs[0]:.1f}%" if valid else "no hop metrics"
        else:
            avg = data.get("ind_test_acc", float("nan"))
            summary = f"25-bench avg={avg:.2f}%" if avg == avg else "no ind metric"
        print(f"  [cached] {variant['name']:20s}  {dataset}  {summary}  ({cached.name})")
        return 0

    # No cached result — run graphany/run.py
    cmd = [
        sys.executable, str(ROOT / "graphany" / "run.py"),
        f"dataset={dataset}",
        f"seed={seed}",
        f"results_output={out_subdir}",
    ] + variant["extra_args"]

    print(f"\n{'='*60}")
    print(f"Variant: {variant['name']}  |  Dataset: {dataset}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"WARNING: variant={variant['name']} dataset={dataset} exited with code {result.returncode}")
    return result.returncode


# Run selected variants across both eval datasets
for target in EVAL_TARGETS:
    print(f"\nDataset: {target['dataset']}")
    for variant in variants_to_run:
        run_variant(variant, target["dataset"], target["out_subdir"], seed)

print("\nAll done.")
print(f"HopSign results in:   {DATA_CACHE / 'camera_ready/graphany_ops_hopsign/CoraXHopSign'}")
print(f"25-bench results in:  {DATA_CACHE / 'camera_ready/graphany_ops_25bench/CoraX25Benchmarks'}")
