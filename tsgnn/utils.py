import os
import random

import numpy as np
import torch
from torch_geometric.utils import index_sort
from torch_geometric.utils.sparse import index2ptr


SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def coo_to_csr(row, col, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(row.max()) + 1
    row, perm = index_sort(row, max_value=num_nodes)
    col = col[perm]
    rowptr = index2ptr(row, num_nodes)
    return rowptr, col


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if preds.dim() == 2:
        pred_labels = preds.argmax(dim=1)
    elif preds.dim() == 1:
        pred_labels = preds
    else:
        raise ValueError(f"Invalid shape for preds: {preds.shape}")
    correct = (pred_labels == targets).sum().item()
    total = targets.numel()
    return correct / total
