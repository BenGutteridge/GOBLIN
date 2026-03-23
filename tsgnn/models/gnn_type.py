from enum import Enum, auto
from torch.nn import Module
from torch.nn.modules.linear import init

from tsgnn.models.triton_nn.mean_gnn import TritonMeanGNN
from tsgnn.models.triton_nn.gat import TritonGAT


def get_fan_in(model: Module) -> float:
    return init._calculate_fan_in_and_fan_out(model.aggr_lin.weight)[0]


class GNNType(Enum):
    GAT = auto()
    MEAN_GNN = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return GNNType[s]
        except KeyError:
            raise ValueError(f"Unknown GNNType: {s}")

    def get_module(self, in_channel: int, out_channel: int, triton_on: bool = True):
        if self is GNNType.GAT:
            return TritonGAT(in_channels=in_channel, out_channels=out_channel)
        elif self is GNNType.MEAN_GNN:
            return TritonMeanGNN(in_channels=in_channel, out_channels=out_channel)
        else:
            raise ValueError(f'model {self.name} not supported')

    def uses_triton(self) -> bool:
        return True
