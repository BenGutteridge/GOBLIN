import logging
import os
import os.path
import os.path as osp
from pathlib import Path
from matplotlib import pyplot as plt
import re
import ssl
import math
import sys
import urllib
from torch.nn.functional import one_hot
import dgl
import dgl.function as fn
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold._utils import (
    _binary_search_perplexity as sklearn_binary_search_perplexity,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from graphany.utils import logger, timer
from collections import deque
from time import time
import networkx as nx

from .range import (
    get_linear_gnn_jacobians,
    generate_adj_power_F,
    dgl_to_sparse_rw_adj,
    solve_linear_gnn,
)


def get_entropy_normed_cond_gaussian_prob(X, entropy, metric="euclidean"):
    """
    Parameters
    ----------
    X:              The matrix for pairwise similarity
    entropy:     Perplexity of the conditional prob distribution
    Returns the entropy-normalized conditional gaussian probability based on distances.
    -------
    """

    # Compute pairwise distances
    perplexity = np.exp2(entropy)
    distances = pdist(X, metric=metric)
    distances = squareform(distances)

    # Compute the squared distances
    distances **= 2
    distances = distances.astype(np.float32)
    return sklearn_binary_search_perplexity(distances, perplexity, verbose=0)


def sample_k_nodes_per_label(label, visible_nodes, k, num_class):
    ref_node_idx = [
        (label[visible_nodes] == lbl).nonzero().view(-1) for lbl in range(num_class)
    ]
    sampled_indices = [
        label_indices[torch.randperm(len(label_indices))[:k]]
        for label_indices in ref_node_idx
    ]
    return visible_nodes[torch.cat(sampled_indices)]


def get_data_split_masks(n_nodes, labels, num_train_nodes, label_idx=None, seed=42):
    label_idx = np.arange(n_nodes)
    test_rate_in_labeled_nodes = (len(labels) - num_train_nodes) / len(labels)
    train_idx, test_and_valid_idx = train_test_split(
        label_idx,
        test_size=test_rate_in_labeled_nodes,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=labels[test_and_valid_idx],
    )
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def download_url(url: str, folder: str, log: bool = True, filename=None):
    r"""Modified from torch_geometric.data.download_url

    Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(osp.expanduser(osp.normpath(folder)), exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def load_heterophilous_dataset(url, raw_dir):
    # Wrap Heterophilous to DGL Graph Dataset format https://arxiv.org/pdf/2302.11640.pdf
    download_path = download_url(url, raw_dir)
    data = np.load(download_path)
    node_features = torch.tensor(data["node_features"])
    labels = torch.tensor(data["node_labels"])
    edges = torch.tensor(data["edges"])

    graph = dgl.graph(
        (edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int
    )
    num_classes = len(labels.unique())
    num_targets = 1 if num_classes == 2 else num_classes
    if num_targets == 1:
        labels = labels.float()
    train_masks = torch.tensor(data["train_masks"]).T
    val_masks = torch.tensor(data["val_masks"]).T
    test_masks = torch.tensor(data["test_masks"]).T

    return graph, labels, num_classes, node_features, train_masks, val_masks, test_masks


def get_gft(graph, signal, name, hop):
    A = graph.adj().to_dense()
    assert torch.equal(A, A.T)
    deg = A.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    L = torch.eye(A.size(0)) - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals, eigvecs = torch.linalg.eigh(L)
    yhat = eigvecs.T @ signal

    eigvals, energy = eigvals[1:], yhat[1:] ** 2  # remove the zero-freq component

    # (2) Cumulative energy
    # cum_energy = energy.cumsum(dim=0) / energy.sum()
    plt.plot(eigvals, energy, label=name.split("_")[0], alpha=(hop / 6), color="b")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(f"figures/gft_{name}.png")

    return eigvals, eigvecs, yhat


class CombinedDataset(pl.LightningDataModule):
    def __init__(self, train_ds_dict, eval_ds_dict, cfg):
        super().__init__()
        self.train_ds_dict = train_ds_dict
        self.eval_ds_dict = eval_ds_dict
        self.all_ds = list(self.train_ds_dict.values()) + list(
            self.eval_ds_dict.values()
        )
        self.cfg = cfg

    def to(self, device):
        for ds in self.all_ds:
            ds.to(device)

    def train_dataloader(self):
        sub_dataloaders = {
            name: ds.train_dataloader() for name, ds in self.train_ds_dict.items()
        }
        return pl.utilities.combined_loader.CombinedLoader(sub_dataloaders, "min_size")

    def val_dataloader(self):
        sub_dataloaders = {
            name: ds.val_dataloader() for name, ds in self.eval_ds_dict.items()
        }
        # Use max_size instead of max_size_cycle to avoid repeated evaluation on small datasets
        return pl.utilities.combined_loader.CombinedLoader(sub_dataloaders, "max_size")

    def test_dataloader(self):
        sub_dataloaders = {
            name: ds.test_dataloader() for name, ds in self.eval_ds_dict.items()
        }
        # Use max_size instead of max_size_cycle to avoid repeated evaluation on small datasets
        return pl.utilities.combined_loader.CombinedLoader(sub_dataloaders, "max_size")


class GraphDataset(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        ds_name,
        cache_dir,
        train_batch_size=256,
        val_test_batch_size=256,
        preprocess_device=torch.device("cpu"),
        permute_label=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = ds_name
        self.cache_dir = Path(cache_dir)
        self.train_batch_size = train_batch_size
        self.permute_label = permute_label  # For checking label equivariance
        self.val_test_batch_size = val_test_batch_size
        self.preprocess_device = preprocess_device

        self.data_source, ds_alias = cfg["_ds_meta_data"][ds_name].split(", ")
        self.gidtype = None
        self.dist = None
        self.unmasked_pred = None
        if self.data_source == "pyg":
            components = ds_alias.split(".")
            ds_init_args = {
                "_target_": f"torch_geometric.datasets.{ds_alias}",
                "root": f"{cfg.dirs.data_storage}{self.data_source}/{ds_alias}/",
            }
            if len(components) == 2:  # If sub-dataset
                ds_init_args["_target_"] = f"torch_geometric.datasets.{components[0]}"
                ds_init_args["name"] = components[1]
        elif self.data_source == "dgl":
            ds_init_args = {
                "_target_": f"dgl.data.{ds_alias}",
                "raw_dir": f"{cfg.dirs.data_storage}{self.data_source}/",
            }
        elif self.data_source == "ogb":
            ds_init_args = {
                "_target_": f"ogb.nodeproppred.DglNodePropPredDataset",
                "root": f"{cfg.dirs.data_storage}{self.data_source}/",
                "name": ds_alias,
            }
        elif self.data_source == "heterophilous":
            target = "graphany.data.load_heterophilous_dataset"
            url = f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/{ds_alias}.npz"
            ds_init_args = {
                "_target_": target,
                "raw_dir": f"{cfg.dirs.data_storage}{self.data_source}/",
                "url": url,
            }
        elif self.data_source == "k_hop_sign":
            target = "goblin.data.build_hopsign_dataset_wrapper"
            k = int(self.name.split("HopSign")[0])
            ds_init_args = {
                "k": k,
                "_target_": target,
            }
        else:
            raise NotImplementedError(f"Unsupported {self.data_source=}")
        self.data_init_args = OmegaConf.create(ds_init_args)

        all_channels = "+".join([cfg.feat_chn, cfg.pred_chn])
        sorted_channels = sorted(
            list(set(all_channels.split("+"))), reverse=True
        )  # unique set of feats
        channels_key = "+".join(sorted_channels)

        self.split_index = 0
        (
            self.g,
            self.label,
            self.feat,
            self.train_mask,
            self.val_mask,
            self.test_mask,
            self.num_class,
        ) = self.load_dataset(self.data_init_args)
        self.n_nodes, self.n_edges = self.g.num_nodes(), self.g.num_edges()
        self.cache_f_name = (
            self.cache_dir
            / f"{self.name}_{channels_key}_selfloop={cfg.add_self_loop}_bidirected={cfg.to_bidirected}_split={self.split_index}.pt"
        )

        self.dist_f_name = (
            self.cache_dir
            / f"{self.name}_{channels_key}_selfloop={cfg.add_self_loop}_bidirected={cfg.to_bidirected}_split={self.split_index}_entropy={cfg.entropy}_dist.pt"
        )

        self.gidtype = self.g.idtype
        self.train_indices = self.train_mask.nonzero().view(-1)

        # #  Compute all-pairs shortest path distances
        N, d = self.feat.shape
        torch.manual_seed(cfg.get("sampling_seed", 0))
        self.randomized_node_indices = torch.randperm(
            N
        )  # for sampling for APSPD for large graphs and for range stats
        self.get_all_pairs_shortest_path_distances()
        apspd = self.all_pairs_shortest_path_distances["spd"].reshape(N, -1)
        apspd_idxs = self.all_pairs_shortest_path_distances.get("node_idxs", None)

        # # COMPUTE LINEAR GNN FEATS
        X = self.feat  # [N, d]
        X_ = X.to(self.preprocess_device)
        A_rw = dgl_to_sparse_rw_adj(self.g)  # sparse [N, N]
        Y = one_hot(self.label, self.num_class).float()  # [N, C]
        all_F, all_Y_hat, all_A_k_rw = {}, {}, {}
        for chn in sorted_channels:
            cache_path = (
                self.cache_dir
                / f"{self.name}_{chn}_selfloop={cfg.add_self_loop}_bidirected={cfg.to_bidirected}_split={self.split_index}.pt"
            )
            if cache_path.exists():
                logger.info(
                    f"Loading cached features and preds for channel {chn} from {cache_path}"
                )
                F, Y_hat, A_k_rw = torch.load(cache_path)
                all_F[chn], all_Y_hat[chn] = F, Y_hat
                if A_k_rw is not None and chn[0] == "N" and chn[1:].isnumeric():
                    all_A_k_rw[int(chn[1:])] = A_k_rw
                continue

            # Otherwise, compute
            S = None
            dist_median = int(apspd.median().item())
            dist_mean = apspd.float().mean().item()
            diameter = int(apspd.max().item())
            if chn == "X":
                lp, hops = True, 0
            elif chn[0] == "L":
                lp, hops = True, int(chn[1:])
            elif chn[0] == "H":
                lp, hops = False, int(chn[1:])
            elif chn[0] == "N" and chn[1:].isdigit():  # precise k-hop adjs
                if apspd_idxs is None:  # we have the full [N, N] APSPD
                    k = int(chn[1:])
                    if k in all_A_k_rw:
                        S = all_A_k_rw[k]
                    else:
                        S = get_A_k_rw(apspd, [k])
                        all_A_k_rw[k] = S
                    lp, hops = True, 1  # single MP step on k-hop adj
                else:
                    raise NotImplementedError(
                        "Cannot compute precise k-hop adjs when using sampled APSPD."
                    )
            # Binned k-hop adjs, up to median dist and from median dist to diameter
            elif (
                chn[0] == "N" and chn[1].isdigit() and chn[2:] == "d"
            ):  # some val up to d*
                start_k = int(chn[1])
                k_list = list(range(start_k, dist_median + 1))
                S = get_A_k_rw(apspd, k_list)
                lp, hops = True, 1
            elif (
                chn[0] == "N" and chn[1] == "d" and chn[2:] == "R"
            ):  # the rest of the khops beyond median_dist
                k_list = list(range(dist_median + 1, diameter + 1))
                S = get_A_k_rw(apspd, k_list)
                lp, hops = True, 1

            elif chn[0] == "D":  # diffusion kernel
                tau = {
                    "S": 1,
                    "M": dist_mean**2,
                    "L": min((2 * dist_mean) ** 2, diameter**2),
                }[chn[1:]]
                L_rw = torch.eye(N).to(A_rw.device) - A_rw  # random-walk Laplacian
                if N < 19_750:  # FCora and below
                    S = torch.matrix_exp(-tau * L_rw)  # [N, N] # heat diffusion kernel
                    lp, hops = True, 1
                else:
                    K = min(int(math.ceil(tau + 6 * tau**0.5)), 30)
                    logger.info(
                        f"Large graph detected, computing diffusion kernel approx via truncated Taylor series with K={K} terms..."
                    )
                    F = torch.zeros((N, d), device=A_rw.device)
                    Xk = X
                    coeff = math.exp(-tau)
                    F += coeff * Xk  # k=0 term

                    for k in tqdm(range(1, K + 1)):
                        Xk = torch.sparse.mm(A_rw, Xk)
                        coeff *= tau / k
                        F += coeff * Xk
                    X_ = F
                    lp, hops = True, 0  # return X_ = F below
            else:
                raise ValueError(f"Unknown channel specifier {chn=}")

            F, _ = generate_adj_power_F(
                X=X_,
                A_rw=S if S is not None else A_rw,
                n_hops=hops,
                low_pass=lp,
            )  # F: [N, d]

            _, Y_hat = solve_linear_gnn(
                F=F,
                Y=Y,
                label_idxs=self.train_indices,
            )
            # For checking that the two methods are equivalent (or thereabouts)
            # assert torch.allclose(Y_hat, self.unmasked_pred[chn], atol=1e-4), f"Discrepancy found in Yhat^i for channel {chn}!"

            torch.save((F, Y_hat, S), cache_path)
            logger.info(
                f"Saved cached features and preds for channel {chn} to {cache_path}"
            )
            all_F[chn], all_Y_hat[chn] = F, Y_hat

        self.features = all_F
        self.unmasked_pred = all_Y_hat
        self.dist = self.prepare_dist_features(self.unmasked_pred)  # [N, t(t-1)]

        if not cfg.get("compute_range", False):  # skip range computation stuff
            # Remove the graph, as GraphAny doesn't use it in training
            del self.g
            del self.feat
            torch.cuda.empty_cache()
            return

        dists = self.all_pairs_shortest_path_distances["spd"].reshape(-1, N)
        if (dists < 0).any():
            raise ValueError(
                "All pairs shortest path distances contains -1 (unreachable node pairs). Cannot compute range stats on disconnected graphs."
            )

        range_filepath = (
            self.cache_dir
            / "range_stats"
            / f"{self.name}_{channels_key}_selfloop={cfg.add_self_loop}_bidirected={cfg.to_bidirected}_split={self.split_index}_range.pt"
        )
        ranges = {}
        if os.path.exists(range_filepath):
            logger.info(f"Loading cached range stats from {range_filepath}")
            ranges = torch.load(range_filepath)

        # Compute linear GNNs
        lp_hop_pairs = []
        for chn in sorted_channels:
            if chn == "X":
                lp_hop_pairs.append((chn, True, 0, A_rw))
            elif chn[0] == "L":
                lp_hop_pairs.append((chn, True, int(chn[1:]), A_rw))
            elif chn[0] == "H":
                lp_hop_pairs.append((chn, False, int(chn[1:]), A_rw))
            elif chn[0] == "N":
                k = int(chn[1:])
                lp_hop_pairs.append((chn, True, 1, all_A_k_rw[k]))
            else:
                raise ValueError(f"Unknown channel specifier {chn=}")

        dists_idxs = self.all_pairs_shortest_path_distances.get("node_idxs", None)
        if dists_idxs is None:
            fixed_W_rho_feat_str = "fixed_W_rho_spd"
        else:
            fixed_W_rho_feat_str = f"fixed_W_rho_spd_Ns={len(dists_idxs):04d}"
        ranges[fixed_W_rho_feat_str] = ranges.get(fixed_W_rho_feat_str, {})

        for feat_str, lp, hops, A_rw_i in lp_hop_pairs:
            if feat_str in ranges[fixed_W_rho_feat_str]:
                continue  # Already computed

            _, S = generate_adj_power_F(
                X=self.feat,
                A_rw=A_rw_i,
                n_hops=hops,
                low_pass=lp,
            )  # F: [N, d], S: sparse [N, N]
            S: torch.Tensor = S.to_dense()  # [N, N], I, A, (I-A), etc

            if dists_idxs is None:
                rho: torch.Tensor = (S * dists).abs().sum(dim=1) / S.abs().sum(
                    dim=1
                )  # [N], normalized range per node
                normalizing_c = S.abs().sum(dim=1)
                rho = torch.where(
                    normalizing_c != 0, rho, torch.zeros_like(rho)
                )  # 0/0 -> 0
                ranges[fixed_W_rho_feat_str][feat_str] = rho
            else:

                S_sampled = S[dists_idxs, :]  # [N_s, N]
                rho: torch.Tensor = (S_sampled * dists).abs().sum(
                    dim=1
                ) / S_sampled.abs().sum(
                    dim=1
                )  # [N_s], normalized range per node
                ranges[fixed_W_rho_feat_str][feat_str] = rho
                ranges[fixed_W_rho_feat_str]["node_idxs"] = dists_idxs

        logger.info(f"\nFixed-W LinearGNN ranges for {self.name}:")
        for feat_str, rho in ranges[fixed_W_rho_feat_str].items():
            if feat_str == "node_idxs":
                continue
            logger.info(
                f"{feat_str}:\trange = {rho.mean().item():.3f} +/- {rho.std().item():.3f}"
            )

        # Black-box sampled LinearGNN ranges
        os.makedirs(osp.dirname(range_filepath), exist_ok=True)
        torch.save(ranges, range_filepath)

        correct_out_ch_only: torch.Tensor = self.label.reshape(
            self.n_nodes, -1
        )  # [N, num_sampled_out_ch]
        in_ch = torch.randperm(d)[: min(cfg.max_input_ch_samples, d)]

        range_keys = []
        for i in range(
            0, min(cfg.max_output_nodes_samples, N), cfg.output_nodes_samples
        ):
            l, r = i, min(i + cfg.output_nodes_samples, N + 1)
            range_key = f"linear_gnns_rho_true_spd_out-nodes{l:04d}-{r:04d}_in-ch_{len(in_ch):03d}"
            range_keys.append((l, r, range_key))
            out_nodes = self.randomized_node_indices[l:r]
            out_ch = correct_out_ch_only[out_nodes]
            ranges[range_key] = ranges.get(range_key, {})

            for feat_str, lp, hops, A_rw_i in lp_hop_pairs:
                if feat_str in ranges[range_key]:
                    continue  # Already computed

                t0 = time()
                _, J = get_linear_gnn_jacobians(
                    X=self.feat,
                    Y=one_hot(self.label, self.num_class).float(),
                    A_rw=A_rw_i,
                    label_idxs=self.train_indices,
                    n_hops=hops,
                    low_pass=lp,
                    out_nodes=out_nodes,
                    out_ch=out_ch,
                    in_ch=in_ch,
                    device=self.preprocess_device,
                )  # [N, C, N, d] or [len(out_nodes), len(out_ch), N, d] if sampling

                if "true" in range_key and len(J.shape) == 3:
                    J_abs_summed = J.abs().sum(dim=-1)  # sum over d channel
                else:
                    J_abs_summed = (
                        J.abs().sum(dim=1).sum(dim=-1)
                    )  # sum over C and d channel, -> [N, N]

                # Get correct dists slice depending on if dists and/or Jac nodes are sampled
                if (
                    out_nodes is not None and dists_idxs is None
                ):  # sampled nodes, full dists
                    permuted_dists = dists[out_nodes, :]  # [len(out_nodes), N]
                elif out_nodes is not None:  # sampled nodes, sampled dists
                    assert torch.equal(
                        dists_idxs[:r], self.randomized_node_indices[:r]
                    ), "Node idxs mismatch!"
                    permuted_dists = dists[l:r]  # dists already permuted
                else:  # full nodes, full dists
                    permuted_dists = dists  # [N, N]

                rho_unnormallized = (J_abs_summed * permuted_dists).sum(dim=1)  # [N]
                norm_c = J_abs_summed.sum(dim=1)  # normalize by total sensitivity
                rho = torch.where(
                    norm_c != 0,
                    rho_unnormallized / norm_c,
                    torch.zeros_like(rho_unnormallized),
                )  # 0/0 -> 0

                t = time() - t0
                ranges[range_key][feat_str + "_time"] = t
                ranges[range_key][feat_str] = rho
                ranges[range_key][feat_str + "_node_idxs"] = out_nodes

            torch.save(ranges, range_filepath)
            logger.info(
                f"Saved range stats for {self.name} nodes {l}-{r} to {range_filepath}"
            )

            # Print results
            logger.info(f"\n\nBlack box LinearGNN ranges for {self.name} nodes 0-{r}:")
            for feat_str in sorted_channels:
                all_rho_u = torch.cat(
                    [ranges[rk][feat_str] for _, _, rk in range_keys], dim=0
                )
                rho_mean, rho_std = all_rho_u.mean().item(), all_rho_u.std().item()
                t = sum([ranges[rk][feat_str + "_time"] for _, _, rk in range_keys])
                logger.info(
                    f"{feat_str}:\trange = {rho_mean:.3f} +/- {rho_std:.3f}\t time: {round(t)}s ({t/60:.1f} mins)"
                )

        # Remove the graph, as GraphAny doesn't use it in training
        del self.g
        del self.feat
        torch.cuda.empty_cache()

    def to(self, device):  # Supports nested dictionary
        def to_device(input):
            if input is None:
                return None
            elif isinstance(input, dict):
                return {key: to_device(value) for key, value in input.items()}
            elif isinstance(input, list):
                return [to_device(item) for item in input]
            elif hasattr(input, "to"):
                return input.to(device)
            else:
                return (
                    input  # Return as is if it's not a tensor or any nested structure
                )

        # Apply to_device to all attributes that may contain tensors
        attrs = [
            "label",
            "feat",
            "train_mask",
            "val_mask",
            "test_mask",
            "train_indices",
            "unmasked_pred",
        ]
        for attr in attrs:
            if hasattr(self, attr):
                setattr(self, attr, to_device(getattr(self, attr)))

    def load_dataset(self, data_init_args):
        dataset = instantiate(data_init_args)

        if self.data_source == "ogb":
            split_idx = dataset.get_idx_split()
            train_indices, valid_indices, test_indices = (
                split_idx["train"],
                split_idx["valid"],
                split_idx["test"],
            )
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
            g, label = dataset[0]
            label = label.view(-1)

            def to_mask(indices):
                mask = torch.BoolTensor(g.number_of_nodes()).fill_(False)
                mask[indices] = 1
                return mask

            train_mask, val_mask, test_mask = map(
                to_mask, (train_indices, valid_indices, test_indices)
            )

            num_class = label.max().item() + 1

            feat = g.ndata["feat"]
        elif self.data_source == "heterophilous":
            g, label, num_class, feat, train_mask, val_mask, test_mask = dataset
        elif self.data_source == "dgl":
            g = dataset[0]
            num_class = dataset.num_classes

            # get node feature
            feat = g.ndata["feat"]

            # get data split
            train_mask = g.ndata["train_mask"]
            val_mask = g.ndata["val_mask"]
            test_mask = g.ndata["test_mask"]

            label = g.ndata["label"]
        elif self.data_source == "pyg":
            g = dgl.graph((dataset.edge_index[0], dataset.edge_index[1]))
            n_nodes = dataset.x.shape[0]
            num_class = dataset.num_classes
            # get node feature
            feat = dataset.x
            label = dataset.y

            if (
                hasattr(dataset, "train_mask")
                and hasattr(dataset, "val_mask")
                and hasattr(dataset, "test_mask")
            ):
                train_mask, val_mask, test_mask = (
                    dataset.train_mask,
                    dataset.val_mask,
                    dataset.test_mask,
                )
            else:
                if label.ndim > 1:
                    raise NotImplementedError(
                        "Multi-Label classification currently unsupported."
                    )
                logging.warning(
                    f"No dataset split found for {self.name}, splitting with semi-supervised settings!!"
                )
                train_mask, val_mask, test_mask = get_data_split_masks(
                    n_nodes, label, 20 * num_class, seed=self.cfg.seed
                )

                self.split_index = self.cfg.seed
        elif self.data_source == "k_hop_sign":
            (
                g,
                label,
                num_class,
                feat,
                train_mask,
                val_mask,
                test_mask,
            ) = dataset
        else:
            raise NotImplementedError(f"Unsupported {self.data_source=}")
        if train_mask.ndim == 1:
            pass  # only one train/val/test split
        elif train_mask.ndim == 2:
            # ! Multiple splits
            # Modified: Use the ${seed} split if not specified!
            split_index = self.data_init_args.get("split", self.cfg.seed)
            # Avoid invalid split index
            self.split_index = split_index = split_index % train_mask.ndim
            train_mask = train_mask[:, split_index].squeeze()
            val_mask = val_mask[:, split_index].squeeze()
            if test_mask.ndim == 2:
                test_mask = test_mask[:, split_index].squeeze()
        else:
            raise ValueError("train/val/test masks have more than 2 dimensions")
        print(
            f"{self.name} {g.num_nodes()} {g.num_edges()} {feat.shape[1]} {num_class} {len(train_mask.nonzero())}"
        )

        if self.cfg.add_self_loop:
            g = dgl.add_self_loop(g)
        else:
            g = dgl.remove_self_loop(g)
        if self.cfg.to_bidirected:
            g = dgl.to_bidirected(g)
        g = dgl.to_simple(g)  # Remove duplicate edges.
        return g, label, feat, train_mask, val_mask, test_mask, num_class

    def compute_linear_gnn_logits(
        self, features, n_per_label_examples, visible_nodes, bootstrap=False
    ):
        # Compute and save LinearGNN logits into a dict. Note the computation is on CPU as torch does not support
        # the gelss driver on GPU currently.
        preds = {}
        label, num_class, device = self.label, self.num_class, torch.device("cpu")
        label = label.to(device)
        visible_nodes = visible_nodes.to(device)
        for channel, F in features.items():
            F = F.to(device)
            if bootstrap:
                ref_nodes = sample_k_nodes_per_label(
                    label, visible_nodes, n_per_label_examples, num_class
                )
            else:
                ref_nodes = visible_nodes
            Y_L = torch.nn.functional.one_hot(label[ref_nodes], num_class).float()
            with timer(
                f"Solving with CPU driver (N={len(ref_nodes)}, d={F.shape[1]}, k={num_class})",
                logger.debug,
            ):
                W = torch.linalg.lstsq(
                    F[ref_nodes.cpu()].cpu(), Y_L.cpu(), driver="gelss"
                )[0]
            preds[channel] = F @ W

        return preds

    def compute_channel_logits(self, features, visible_nodes, sample, device):
        pred_logits = self.compute_linear_gnn_logits(
            {
                c: features[c]
                for c in set(self.cfg.feat_channels + self.cfg.pred_channels)
            },
            self.cfg.n_per_label_examples,
            visible_nodes,
            bootstrap=sample,
        )
        return {c: logits.to(device) for c, logits in pred_logits.items()}

    def prepare_dist_features(
        self,
        unmasked_pred: dict[str, torch.Tensor],  # t: [N, C]
    ) -> torch.Tensor:  # -> dists [N, t(t-1)]
        """
        - Calculate distance features based on conditional gaussian probabilities between different channel predictions
        - as in GraphAny paper
        """
        if not os.path.exists(self.dist_f_name):
            with timer(
                f"Computing {self.name} conditional gaussian distances "
                f"and save to {self.dist_f_name}",
                logger.info,
            ):
                # y_feat: n_nodes, n_channels, n_labels
                y_feat = np.stack(
                    [unmasked_pred[c].cpu().numpy() for c in self.cfg.feat_channels],
                    axis=1,
                )
                # Conditional gaussian probability
                bsz, n_channel, n_class = y_feat.shape
                dist_feat_dim = n_channel * (n_channel - 1)
                # Conditional gaussian probability
                cond_gaussian_prob = np.zeros((bsz, n_channel, n_channel))
                for i in range(bsz):
                    cond_gaussian_prob[i, :, :] = get_entropy_normed_cond_gaussian_prob(
                        y_feat[i, :, :], self.cfg.entropy
                    )
                dist = np.zeros((bsz, dist_feat_dim), dtype=np.float32)

                # Compute pairwise distances between channels n_channels(n_channels-1)/2 total features
                pair_index = 0
                for c in range(n_channel):
                    for c_prime in range(n_channel):
                        if c != c_prime:  # Diagonal distances are useless
                            dist[:, pair_index] = cond_gaussian_prob[:, c, c_prime]
                            pair_index += 1

                dist = torch.from_numpy(dist)
                torch.save(dist, self.dist_f_name)
        else:
            dist = torch.load(self.dist_f_name, map_location="cpu")
        return dist

    def train_dataloader(self):
        return DataLoader(
            self.train_mask.nonzero().view(-1),
            batch_size=self.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_mask.nonzero().view(-1), batch_size=self.val_test_batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_mask.nonzero().view(-1), batch_size=self.val_test_batch_size
        )

    def compute_k_hop_apspd_sparse(
        self,
        nhops: int,
        device: torch.device | None = None,
    ):
        """
        Compute exact APSP up to nhops using sparse matrix multiplication.
        Automatically resumes from cached k-hop frontiers if present.
        """
        import re
        from tqdm import tqdm

        assert hasattr(self, "g"), "Graph must exist before calling this method"
        g = self.g
        N = g.num_nodes()

        self.k_hops_cache_dir = self.cache_dir.parent / "apspd" / "k_hops"
        self.k_hops_cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = torch.device("cpu")

        # -----------------------------
        # Detect existing checkpoints
        # -----------------------------
        frontier_re = re.compile(rf"{self.name}_k=(\d+)_frontier\.pt")

        existing = {}
        for p in self.k_hops_cache_dir.glob(f"{self.name}_k=*_frontier.pt"):
            m = frontier_re.match(p.name)
            if m:
                existing[int(m.group(1))] = p

        max_cached_k = max(existing.keys()) if existing else 0

        # -----------------------------
        # Build sparse adjacency matrix
        # -----------------------------
        src, dst = g.edges()
        src = src.to(device)
        dst = dst.to(device)

        A = torch.sparse_coo_tensor(
            torch.stack([src, dst]),
            torch.ones(src.numel(), device=device),
            size=(N, N),
            device=device,
        ).coalesce()

        # -----------------------------
        # Initialise seen / frontier
        # -----------------------------
        all_rows, all_cols, all_vals = [], [], []

        # distance 0: self
        diag = torch.arange(N, device=device)
        all_rows.append(diag)
        all_cols.append(diag)
        all_vals.append(torch.zeros(N, dtype=torch.int8, device=device))

        if max_cached_k > 0:
            # Resume
            frontier = torch.load(
                self.k_hops_cache_dir / f"{self.name}_k={max_cached_k:02d}_frontier.pt",
                map_location=device,
            ).coalesce()

            seen = torch.load(
                self.k_hops_cache_dir / f"{self.name}_k={max_cached_k:02d}_seen.pt",
                map_location=device,
            ).coalesce()

            start_k = max_cached_k + 1
        else:
            # Fresh start
            seen = torch.sparse_coo_tensor(
                torch.stack([diag, diag]),
                torch.ones(N, device=device),
                size=(N, N),
                device=device,
            ).coalesce()

            frontier = A
            start_k = 1

        # -----------------------------
        # Iterate hops
        # -----------------------------
        for k in tqdm(
            range(start_k, nhops + 1),
            desc=f"{self.name}: k-hop expansion",
        ):
            if frontier._nnz() == 0:
                break

            frontier = frontier.coalesce()

            frontier_filepath = (
                self.k_hops_cache_dir / f"{self.name}_k={k:02d}_frontier.pt"
            )

            if frontier_filepath.exists():
                frontier = torch.load(frontier_filepath, map_location=device).coalesce()
            else:
                # Save frontier
                torch.save(frontier, frontier_filepath)

            # Record distances
            r, c = frontier.indices()
            all_rows.append(r)
            all_cols.append(c)
            all_vals.append(
                torch.full((frontier._nnz(),), k, dtype=torch.int8, device=device)
            )

            # Update seen
            seen = torch.sparse_coo_tensor(
                torch.cat([seen.indices(), frontier.indices()], dim=1),
                torch.ones(seen._nnz() + frontier._nnz(), device=device),
                size=(N, N),
                device=device,
            ).coalesce()

            seen_filepath = self.k_hops_cache_dir / f"{self.name}_k={k:02d}_seen.pt"
            if not seen_filepath.exists():
                torch.save(seen, seen_filepath)

            if k == nhops:
                break

            # -------------------------
            # Compute next frontier
            # -------------------------
            with tqdm(
                total=3,
                desc=f"hop {k} ops",
                leave=False,
            ) as inner:

                next_frontier = torch.sparse.mm(A, frontier).coalesce()
                inner.update(1)

                if next_frontier._nnz() == 0:
                    frontier = next_frontier
                    continue

                key = next_frontier.indices()
                lin = key[0] * N + key[1]

                seen_key = seen.indices()
                seen_lin = seen_key[0] * N + seen_key[1]

                mask = ~torch.isin(lin, seen_lin)
                inner.update(1)

                if mask.any():
                    frontier = torch.sparse_coo_tensor(
                        key[:, mask],
                        torch.ones(mask.sum(), device=device),
                        size=(N, N),
                        device=device,
                    ).coalesce()
                else:
                    frontier = torch.sparse_coo_tensor(
                        torch.empty((2, 0), dtype=torch.long, device=device),
                        torch.empty((0,), device=device),
                        size=(N, N),
                        device=device,
                    )

                inner.update(1)

        # -----------------------------
        # Assemble final sparse SPD
        # -----------------------------
        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        vals = torch.cat(all_vals)

        spd_sparse = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            size=(N, N),
            device=device,
        ).coalesce()

        diameter = int(vals.max().item()) if vals.numel() > 0 else 0

        return {
            "spd_sparse": spd_sparse,
            "diameter": diameter,
            "nhops": nhops,
            "cached_up_to": max_cached_k,
        }

    def compute_all_pairs_shortest_path_distances(
        self,
        node_idxs: torch.Tensor,
    ):
        """
        Compute sampled APSP: [N_s x N], equivalent to full_dist[node_idxs]

        Distances:
        - >= 0 : shortest path distance
        - -1   : unreachable
        """

        assert hasattr(self, "g"), "Graph must exist before calling this method"
        g = self.g
        N = g.num_nodes()

        src, dst = g.edges()
        average_deg = g.num_edges() / N

        # ---------------------------------
        # BFS from selected sources only
        # ---------------------------------
        adj = [[] for _ in range(N)]
        for u, v in zip(src.tolist(), dst.tolist()):
            adj[u].append(v)

        N_src = node_idxs.numel()
        dist = torch.full((N_src, N), -1, dtype=torch.int32)

        for row, i in enumerate(
            tqdm(node_idxs.tolist(), desc=f"APSPD BFS for {self.name}")
        ):
            queue = deque([i])
            dist[row, i] = 0

            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[row, v] == -1:
                        dist[row, v] = dist[row, u] + 1
                        queue.append(v)

        # Graph stats (reachable pairs only)
        reachable = dist >= 0
        if (n_unreachable := (~reachable).sum().item()) > 0:
            logger.warning(
                f"{self.name}: Graph is disconnected. "
                f"{n_unreachable} / {N*N} node pairs unreachable. ({100 * n_unreachable / (N*N):.2f}%)"
            )

        diameter = dist[reachable].max().item()
        avg_spd = dist[reachable].float().mean().item()

        # Minimal dtype
        if diameter <= 127:
            spd = dist.to(torch.int8)
        elif diameter <= 32_767:
            spd = dist.to(torch.int16)
        else:
            spd = dist

        return {
            "spd": spd,  # [N x N] or [N_s x N]
            "node_idxs": node_idxs,
            "diameter": int(diameter),
            "average_deg": float(average_deg),
            "avg_spd": float(avg_spd),
        }

    def get_all_pairs_shortest_path_distances(self):
        """Get APSPD for the dataset, or a sampled version, and cache it."""
        N, cfg = self.feat.shape[0], self.cfg
        randomized_node_indices = self.randomized_node_indices

        # Get full all-pairs distances
        logger.info(
            f"Computing all pairs shortest path distances for {self.name} graph with {self.g.num_nodes()} nodes."
        )

        full_apspd_filepath = self.cache_dir.parent / "apspd" / f"{self.name}_apspd.pt"
        if os.path.exists(full_apspd_filepath):
            logger.info(f"Loading cached full APSPD from {full_apspd_filepath}")
            self.all_pairs_shortest_path_distances = torch.load(full_apspd_filepath)
            return

        if cfg.max_khops is not None:
            # Use sparse method for exact APSPD up to max_khops
            apspd_result = self.compute_k_hop_apspd_sparse(cfg.max_khops)
            spd_sparse: torch.sparse_coo_tensor = apspd_result["spd_sparse"]
            spd_dense = torch.full((N, N), -1, dtype=torch.int8)
            spd_dense[spd_sparse.indices()[0], spd_sparse.indices()[1]] = (
                spd_sparse.values()
            )
            self.all_pairs_shortest_path_distances = {
                "spd": spd_dense,
                "average_deg": self.g.num_edges() / N,
                "diameter": apspd_result["diameter"],
                "avg_spd": spd_dense[spd_dense >= 0].float().mean().item(),
                "avg_spd_std": spd_dense[spd_dense >= 0].float().std().item(),
            }
            torch.save(self.all_pairs_shortest_path_distances, full_apspd_filepath)
            logger.info(f"Saved full APSPD to {full_apspd_filepath}")
            return

        # Total number of sampled nodes, number to sample per block
        N_s, N_s_i = cfg.max_output_nodes_samples, cfg.output_nodes_samples
        apspds = []
        for l in range(0, min(N_s, N), N_s_i):
            r = min(l + N_s_i, N)
            sampled_apspd_filepath = (
                self.cache_dir.parent
                / "apspd"
                / f"{self.name}_apspd_sampled_{l:04d}-{r:04d}_seed{cfg.sampling_seed:02d}.pt"
            )
            if os.path.exists(sampled_apspd_filepath):
                logger.info(
                    f"Loading cached sampled APSPD from {sampled_apspd_filepath}"
                )
                apspds.append(torch.load(sampled_apspd_filepath))
                continue

            # Compute chunked APSPD
            node_sample_idxs = randomized_node_indices[l:r]

            ap_spd = self.compute_all_pairs_shortest_path_distances(
                node_idxs=node_sample_idxs,
            )
            torch.save(ap_spd, sampled_apspd_filepath)
            apspds.append(ap_spd)

        # Combine sampled APSPDs
        if r == N:  # we have computed the full spd piecemeal
            full_apspd = torch.zeros((N, N), dtype=apspds[0]["spd"].dtype)
            full_apspd[randomized_node_indices, :] = torch.cat(
                [apspd["spd"] for apspd in apspds], dim=0
            )  # [N, N]
            self.all_pairs_shortest_path_distances = {
                "spd": full_apspd,
                "average_deg": apspds[0]["average_deg"],
                "diameter": full_apspd.max().item(),
                "avg_spd": full_apspd.float()[full_apspd >= 0].mean().item(),
                "avg_spd_std": full_apspd.float()[full_apspd >= 0].std().item(),
            }
            torch.save(self.all_pairs_shortest_path_distances, full_apspd_filepath)
            logger.info(f"Saved full APSPD to {full_apspd_filepath}")
        else:
            full_apspd = torch.cat([apspd["spd"] for apspd in apspds], dim=0)
            self.all_pairs_shortest_path_distances = {
                "spd": full_apspd,
                "average_deg": apspds[0]["average_deg"],
                "diameter": full_apspd.max().item(),
                "avg_spd": full_apspd.float()[full_apspd >= 0].mean().item(),
                "avg_spd_std": full_apspd.float()[full_apspd >= 0].std().item(),
                "node_idxs": randomized_node_indices[:r],
            }
            combined_sampled_apspd_filepath = (
                self.cache_dir
                / "apspd"
                / f"{self.name}_apspd_sampled_{r:04d}_seed{cfg.sampling_seed:02d}.pt"
            )
            torch.save(
                self.all_pairs_shortest_path_distances, combined_sampled_apspd_filepath
            )
            logger.info(
                f"Saved combined sampled APSPD to {combined_sampled_apspd_filepath}"
            )
        return


def get_A_k_rw(apspd: torch.Tensor, k_list: list[int]) -> torch.sparse_coo_tensor:
    A_k = torch.zeros_like(apspd, dtype=torch.float32)
    for k in k_list:
        A_k += (apspd == k).float()
    A_k_rw = torch.sparse_coo_tensor(
        indices=(idx := A_k.nonzero(as_tuple=False).T),
        values=(1.0 / A_k.sum(dim=1).clamp(min=1))[idx[0]],
        size=A_k.shape,
    )
    del A_k
    return A_k_rw
