import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .data import get_entropy_normed_cond_gaussian_prob


class GraphAny(nn.Module):
    def __init__(
        self,
        n_hidden,
        feat_channels,
        pred_channels,
        att_temperature,
        entropy=1,
        n_mlp_layer=2,
    ):
        super(GraphAny, self).__init__()
        self.feat_channels = feat_channels  # channels used as features p_u(i|j)
        self.pred_channels = pred_channels  # channels used as predictions to obtain \bar{y}_u. Might be different from feat_channels
        self.entropy = entropy  # ???, but determines \sigma_u^i for similarity feats
        self.att_temperature = att_temperature
        self.dist_feat_dim = len(feat_channels) * (len(feat_channels) - 1)  # t(t-1)

        self.mlp = MLP(
            self.dist_feat_dim,
            n_hidden,
            len(pred_channels),
            n_mlp_layer,
        )

    def log_gaussian_kernel(self, mu, k, sigma=0.1):
        """Gaussian kernel centered at k with bandwidth sigma"""
        return -0.5 * ((mu - k) / sigma) ** 2

    def compute_dist(self, y_feat):
        bsz, n_channel, n_class = y_feat.shape
        # Conditional gaussian probability
        cond_gaussian_prob = np.zeros((bsz, n_channel, n_channel))
        for i in range(bsz):
            cond_gaussian_prob[i, :, :] = get_entropy_normed_cond_gaussian_prob(
                y_feat[i, :, :].cpu().numpy(), self.entropy
            )

        # Compute pairwise distances between channels n_channels(n_channels-1)/2 total features
        dist = np.zeros((bsz, self.dist_feat_dim), dtype=np.float32)

        pair_index = 0
        for c in range(n_channel):
            for c_prime in range(n_channel):
                if c != c_prime:  # Diagonal distances are useless
                    dist[:, pair_index] = cond_gaussian_prob[:, c, c_prime]
                    pair_index += 1

        dist = torch.from_numpy(dist).to(y_feat.device)
        return dist

    def forward(self, logit_dict: dict[str, torch.Tensor], dist=None, **kwargs):
        """
        N: num nodes/batch size
        C: num classes
        t: num LinearGNN channels used for attention

        logit_dict: key: LinearGNN channel, value: prediction \hat{Y}^i of shape [N, C]
        dist: if None, recompute, if precomputed, is [N, t(t-1)]
        """

        # \hat{y}_u in paper, [N, t, C]
        y_feat = torch.stack([logit_dict[c] for c in self.feat_channels], dim=1)
        y_pred = torch.stack([logit_dict[c] for c in self.pred_channels], dim=1)

        ## Fuse y_pred with attentions

        # dist is actually the [N, t(t-1)] similarity feats
        dist = self.compute_dist(y_feat) if dist is None else dist

        N = dist.shape[0]

        # Get normal MLP gate attention logits
        logits = self.mlp(dist)  # [N, t]

        if kwargs.get("one_hot_test_attn", False):
            attention = th.softmax(logits / 1e-8, dim=-1)
        else:
            attention = th.softmax(logits / self.att_temperature, dim=-1)

        # Get \bar{y} as sum of attention-weighted y feats
        fused_y = th.sum(
            rearrange(attention, "n n_channels -> n n_channels 1") * y_pred, dim=1
        )  # Sum over channels, resulting in (batch_size, n_classes)

        return (
            fused_y,  # [N, C]
            attention,  # [N, t], attention per-channel per node
            logits,  # [N, t], pre-softmax attention logits
        )


def study_feat_sizes(
    logits_weighted: dict[str, torch.Tensor],
    attn: torch.Tensor,
    ds_name: str | None = None,
):
    print(
        f"\n=== Per-linear-GNN contribution diagnostics {'for ' + ds_name if ds_name else ''}==="
    )

    for i, (k, logits) in enumerate(logits_weighted.items()):
        # binary margin: logit(correct) - logit(incorrect)
        margin = logits[:, 1] - logits[:, 0]

        mean_abs_margin = margin.abs().mean().item()
        mean_margin = margin.mean().item()
        std_margin = margin.std().item()
        l2_norm = torch.norm(logits).item()

        print(
            f"{k:>3} | \alpha = {attn[i]:.2f} | "
            f"||α·logits||₂ = {l2_norm:7.4f} | "
            f"E[|margin|] = {mean_abs_margin:7.4f} | "
            f"E[margin] = {mean_margin:+7.4f} | "
            f"Std(margin) = {std_margin:7.4f}"
        )


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        dropout=0.5,
        bias=True,
    ):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if n_layers == 1:
            # just linear layer
            self.lins.append(nn.Linear(in_channels, out_channels, bias=bias))
            self.bns.append(nn.BatchNorm1d(out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels, bias=bias))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(n_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels, bias=bias))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            if x.shape[0] > 1:  # Batch norm only if batch_size > 1
                x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
