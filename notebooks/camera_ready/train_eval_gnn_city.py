"""
Standard MPNN baselines (MeanGNN, GAT) trained directly on CityNetworks datasets.

Pipeline:
  1. Grid search over lr x hidden_dim (4 configs, training seed=0, fixed splits).
  2. Select config with best validation accuracy.
  3. Train with best config across 4 training seeds (fixed splits), report mean±std test acc.

Usage:
  python notebooks/train_eval_gnn_city.py --model meangnn --dataset CityParis
  python notebooks/train_eval_gnn_city.py --model gat --dataset CityLondon --num_layers 16

Hparam grid (from paper Table 10):
  lr:         {2e-4, 5e-4}
  hidden_dim: {64, 128}
  epochs:     400       (fixed)
  num_layers: 2 or 16  (CLI arg, default 2)
  MLP layers: 0        (fixed; last conv directly outputs class logits)
  batch:      full     (fixed)
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import add_self_loops

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from goblin.data import load_graph_dataset

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class MeanConv(MessagePassing):
    """
    Single mean-aggregation layer:
      h_v^(l+1) = W * mean_{u in N~(v)} h_u^(l)
    where N~(v) = N(v) ∪ {v}  (self-loop included).
    Row-normalised adjacency, NOT symmetric (GCN) normalisation.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        return self.lin(out)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class MeanGNN(nn.Module):
    """
    L-layer MeanGNN, 0 additional MLP layers.
    Layers 1..L-1: MeanConv(· → hidden) + ReLU
    Layer L:       MeanConv(hidden → num_classes)  (logits, no activation)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1
        dims = [in_channels] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.convs = nn.ModuleList(
            MeanConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs[:-1]:
            if self.use_checkpoint and h.requires_grad:
                h = grad_ckpt(lambda h, c=conv: F.relu(c(h, edge_index)), h, use_reentrant=False)
            else:
                h = F.relu(conv(h, edge_index))
        return self.convs[-1](h, edge_index)


class GAT(nn.Module):
    """
    L-layer GAT, 0 additional MLP layers.
    Layers 1..L-1: GATConv(· → hidden//num_heads, heads=num_heads, concat=True) + ELU  → (N, hidden)
    Layer L:       GATConv(hidden → num_classes, heads=1, concat=False)  → (N, num_classes)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        num_heads: int = 8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1
        head_dim = max(1, hidden_dim // num_heads)
        convs = []
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else head_dim * num_heads
            if i < num_layers - 1:
                convs.append(GATConv(in_dim, head_dim, heads=num_heads, concat=True))
            else:
                convs.append(GATConv(in_dim, num_classes, heads=1, concat=False))
        self.convs = nn.ModuleList(convs)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs[:-1]:
            if self.use_checkpoint and h.requires_grad:
                h = grad_ckpt(lambda h, c=conv: F.elu(c(h, edge_index)), h, use_reentrant=False)
            else:
                h = F.elu(conv(h, edge_index))
        return self.convs[-1](h, edge_index)


def build_model(
    model_name: str,
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    num_layers: int,
    num_heads: int = 8,
    use_checkpoint: bool = False,
) -> nn.Module:
    if model_name == "meangnn":
        return MeanGNN(in_channels, hidden_dim, num_classes, num_layers, use_checkpoint)
    elif model_name == "gat":
        return GAT(in_channels, hidden_dim, num_classes, num_layers, num_heads, use_checkpoint)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    preds = logits[idx].argmax(dim=-1)
    return (preds == y[idx]).float().mean().item()


def train_and_eval(
    model_name: str,
    in_channels: int,
    num_classes: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    hidden_dim: int,
    lr: float,
    epochs: int,
    train_seed: int,
    device: torch.device,
    num_layers: int = 2,
    num_heads: int = 8,
    use_checkpoint: bool = False,
) -> dict:
    """Train one model config; return best-val epoch's val+test accuracies."""
    set_seed(train_seed)

    model = build_model(model_name, in_channels, hidden_dim, num_classes, num_layers, num_heads, use_checkpoint).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = -1.0
    best_test = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(x, edge_index)
        val_acc = accuracy(logits, y, val_idx)
        if val_acc > best_val:
            best_val = val_acc
            best_test = accuracy(logits, y, test_idx)

    return {"val_acc": best_val, "test_acc": best_test}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["meangnn", "gat"], required=True)
    parser.add_argument("--dataset", choices=["CityParis", "CityShanghai", "CityLA", "CityLondon"], required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8, help="GAT attention heads (ignored for MeanGNN)")
    parser.add_argument("--use_checkpoint", action="store_true", help="gradient checkpointing to save GPU memory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads_str = f" heads={args.num_heads}" if args.model == "gat" else ""
    print(f"Model: {args.model} (L={args.num_layers}{heads_str})  |  Dataset: {args.dataset}  |  Device: {device}")

    # ---- Load data (fixed split seed=0) ------------------------------------
    print("Loading dataset...")
    data, X, _, y_class, _, splits, C = load_graph_dataset(
        name=args.dataset,
        root=ROOT / "data" / "goblin",
        seed=0,
        compute_all_pairs_dist=False,
    )

    x = X.float().to(device)
    edge_index = data.edge_index.to(device)
    y = y_class.long().to(device)
    train_idx = splits["train_fit"].to(device)
    val_idx = splits["val"].to(device)
    test_idx = splits["test"].to(device)

    in_channels = x.size(1)
    print(f"N={x.size(0):,}  feat_dim={in_channels}  classes={C}")
    print(f"train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    # ---- Grid search (training seed=0) -------------------------------------
    lr_choices = [2e-4, 5e-4]
    hidden_choices = [64, 128]
    grid = list(product(lr_choices, hidden_choices))

    print(f"\n{'='*60}")
    print(f"Grid search: {len(grid)} configs × {args.epochs} epochs (seed=0, L={args.num_layers})")
    print(f"{'='*60}")

    grid_results = []
    for lr, hidden_dim in grid:
        res = train_and_eval(
            model_name=args.model,
            in_channels=in_channels,
            num_classes=C,
            x=x, edge_index=edge_index, y=y,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            hidden_dim=hidden_dim, lr=lr, epochs=args.epochs,
            train_seed=0, device=device, num_layers=args.num_layers,
            num_heads=args.num_heads, use_checkpoint=args.use_checkpoint,
        )
        grid_results.append({"lr": lr, "hidden_dim": hidden_dim, **res})
        print(f"  lr={lr:.0e}  hidden={hidden_dim:3d}  "
              f"val={res['val_acc']*100:.2f}%  test={res['test_acc']*100:.2f}%")

    best = max(grid_results, key=lambda r: r["val_acc"])
    best_hparams = {"lr": best["lr"], "hidden_dim": best["hidden_dim"]}
    print(f"\nBest config: {best_hparams}  (val={best['val_acc']*100:.2f}%)")

    # ---- Test eval (4 training seeds, fixed splits) ------------------------
    TEST_SEEDS = [0, 1, 2, 3]
    print(f"\n{'='*60}")
    print(f"Test eval: best config, {len(TEST_SEEDS)} training seeds")
    print(f"{'='*60}")

    test_accs = []
    for seed in TEST_SEEDS:
        res = train_and_eval(
            model_name=args.model,
            in_channels=in_channels,
            num_classes=C,
            x=x, edge_index=edge_index, y=y,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            hidden_dim=best_hparams["hidden_dim"],
            lr=best_hparams["lr"],
            epochs=args.epochs,
            train_seed=seed,
            device=device,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            use_checkpoint=args.use_checkpoint,
        )
        test_accs.append(res["test_acc"])
        print(f"  seed={seed}  val={res['val_acc']*100:.2f}%  test={res['test_acc']*100:.2f}%")

    mean_acc = float(np.mean(test_accs)) * 100
    std_acc = float(np.std(test_accs)) * 100
    print(f"\nFinal: {mean_acc:.2f} ± {std_acc:.2f}%  ({args.model} L={args.num_layers} on {args.dataset})")

    # ---- Save results -------------------------------------------------------
    model_tag = f"{args.model}_L{args.num_layers}"
    if args.model == "gat":
        model_tag += f"_H{args.num_heads}"
    out_dir = ROOT / "output" / "results" / "gnn" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.json"
    result = {
        "model": args.model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads if args.model == "gat" else None,
        "dataset": args.dataset,
        "best_hparams": best_hparams,
        "grid_search": grid_results,
        "test_results": {
            "seeds": TEST_SEEDS,
            "accs_pct": [a * 100 for a in test_accs],
            "mean_pct": mean_acc,
            "std_pct": std_acc,
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
