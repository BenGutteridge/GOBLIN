import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

EPS = 1E-8


def standardize(x: Tensor) -> Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + EPS)


def normalize(x: Tensor) -> Tensor:
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + EPS)


class NormalizeTransform(BaseTransform):
    def forward(self, data: Data) -> Data:
        data.x = normalize(x=data.x)
        return data
