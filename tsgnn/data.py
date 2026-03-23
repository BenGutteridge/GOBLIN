"""
Dataset loading for TS-GNN reproduction.
Supports only HopSign (1-8) and CityNetworks (Paris, Shanghai, LA, London).
"""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import CityNetwork

from tsgnn.transforms.normalize import normalize, NormalizeTransform
from tsgnn.transforms.least_squares import LeastSquaresTransform


# ---------------------------------------------------------------------------
# Transforms shared across datasets
# ---------------------------------------------------------------------------

class RemoveSelfLoops(T.BaseTransform):
    def forward(self, data: Data) -> Data:
        edge_index = data.edge_index
        mask = edge_index[0] != edge_index[1]
        data.edge_index = edge_index[:, mask]
        return data


def _make_transform():
    return T.Compose([
        T.ToUndirected(),
        RemoveSelfLoops(),
        T.RemoveDuplicatedEdges(),
        NormalizeTransform(),
    ])


# ---------------------------------------------------------------------------
# Splits (matches original EquivarianceEverywhere splits)
# ---------------------------------------------------------------------------

def graphany_mask_splits(n_nodes, labels, num_train_nodes, seed=42):
    label_idx = np.arange(n_nodes)
    test_rate = (len(labels) - num_train_nodes) / len(labels)
    train_idx, test_and_valid_idx = train_test_split(
        label_idx, test_size=test_rate, random_state=seed, shuffle=True,
        stratify=labels)
    valid_idx, test_idx = train_test_split(
        test_and_valid_idx, test_size=0.5, random_state=seed, shuffle=True,
        stratify=labels[test_and_valid_idx])
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def get_masks(data: Data, seed: int):
    if hasattr(data, "train_mask") and hasattr(data, "val_mask") and hasattr(data, "test_mask"):
        return data.train_mask, data.val_mask, data.test_mask
    n_nodes = data.x.shape[0]
    label = data.y
    num_class = label.max().item() + 1
    return graphany_mask_splits(n_nodes, label, num_train_nodes=20 * num_class, seed=seed)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_hopsign(k: int, hopsign_cache_dir: str) -> Data:
    """Load kHopSign dataset from GOBLIN's khopsign cache."""
    cache_path = os.path.join(
        hopsign_cache_dir,
        f"{k}HopSign_N=1000_r=0.1_ln=0.5_seed=0.pt",
    )
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    pyg_data = cache['data']
    X = cache['X']
    y_class = cache['y_class']
    splits = cache['splits']
    N = X.shape[0]
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[splits['train_fit']] = True
    train_mask[splits['train_eval']] = True
    val_mask[splits['val']] = True
    test_mask[splits['test']] = True
    data = Data(
        x=X, y=y_class, edge_index=pyg_data.edge_index,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )
    transform = _make_transform()
    data = transform(data)
    return data


def load_city(city_name: str, pyg_cache_dir: str) -> Data:
    """Load CityNetwork dataset. city_name is one of paris/shanghai/la/london."""
    root = os.path.join(pyg_cache_dir, city_name)
    transform = _make_transform()
    data = CityNetwork(root=root, name=city_name, transform=transform)[0]
    return data


def prepare_data(data: Data) -> Data:
    """Add y_mat attribute (normalized one-hot labels)."""
    data.y = data.y.squeeze(dim=-1)
    y_mat = F.one_hot(data.y).float()
    y_mat = normalize(x=y_mat)
    data.y_mat = y_mat
    return data


def split_and_transform(data: Data, seed: int, ls_num_layers: int,
                        dataset_name: str, ls_cache_root: str) -> Data:
    """Apply train/val/test split and LS transform."""
    train_mask, val_mask, test_mask = get_masks(data=data, seed=seed)
    if train_mask.ndim == 2:
        split_index = seed % train_mask.shape[1]
        train_mask = train_mask[:, split_index].squeeze()
        val_mask = val_mask[:, split_index].squeeze()
        if test_mask.ndim == 2:
            test_mask = test_mask[:, split_index].squeeze()
    dataset_copy = copy.deepcopy(data)
    dataset_copy.train_mask = train_mask
    dataset_copy.val_mask = val_mask
    dataset_copy.test_mask = test_mask
    ls_transform = LeastSquaresTransform(
        ls_num_layers=ls_num_layers, dataset_name=dataset_name,
        num_feat=data.x.shape[1], seed=seed, cache_root=ls_cache_root,
    )
    dataset_copy = ls_transform(dataset_copy)
    return dataset_copy
