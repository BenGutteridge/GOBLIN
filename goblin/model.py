from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any, Union, Literal
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data as PygGraph

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from goblin.config import DATA_CACHE
from goblin.data import (
    apply_lin_gaussian_operator,
    apply_lin_heat_operator,
    get_L_sym_eigvals_eigvecs,
    get_fixed_operator,
)
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore", category=ConvergenceWarning, module="sklearn.gaussian_process"
)


# -----------------------------
# Utilities
# -----------------------------
def ensure_empty_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for f in path.glob("*"):
        if f.is_file():
            f.unlink()


def to_numpy_1d(x):
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    return np.array(x).reshape(-1)


def _as_float_list(x):
    return [float(v) for v in x]


def _resolve_anchors(
    anchor_spec: float | list | int | None,
    grid: np.ndarray,
) -> list[float]:
    """Convert an anchor specification to a list of float parameter values.

    Args:
        anchor_spec:
            None        → no anchors (empty list)
            float       → single anchor at that value
            list[float] → explicit anchor positions
            int (n > 0) → n equally-spaced anchors interior to [grid.min, grid.max]
        grid: the parameter grid (used only when anchor_spec is an int)
    """
    if anchor_spec is None:
        return []
    if isinstance(anchor_spec, int):
        n = anchor_spec
        if n <= 0:
            return []
        lo, hi = float(grid.min()), float(grid.max())
        # n interior points; excludes the exact endpoints via linspace(lo, hi, n+2)[1:-1]
        return list(np.linspace(lo, hi, n + 2)[1:-1])
    if isinstance(anchor_spec, (list, tuple)):
        return [float(v) for v in anchor_spec]
    return [float(anchor_spec)]


@dataclass
class OperatorFamily:
    name: str  # "gaussian" or "heat"
    param_name: str  # "mu" or "tau"
    grid: np.ndarray  # 1D grid of parameter values
    eval_fn: Callable[[float, bool], float]


@dataclass(frozen=True)
class OpId:
    family: Literal["gaussian", "heat", "fixed"]
    param: Union[float, str]  # float for heat/gauss, str for fixed ("L1", "H2", ...)


# -----------------------------
# OperatorSearch
# -----------------------------
@dataclass
class OperatorSearchConfig:

    families: list[str]

    # BO settings
    bo_objective: (
        str  # "mean", "trimmed_{%}", "lower_/upper_quartile", "mean_minus_var_{lambda}"
    )
    ucb_beta: float = 1.0  # β in acq = -µ_GP + β·σ_GP; 0 = pure exploitation

    # LinGaussian
    mu_min: float = 0.0
    mu_max: float = 10.0
    mu_num: int = 250

    # LinHeat (search is over tausqrt = sqrt(tau); eval squares before passing to kernel)
    tausqrt_min: float = 0.0
    tausqrt_max: float = 2.5
    tausqrt_num: int = 50

    # GP settings
    rbf_length_scale: float = 1.0
    white_noise: float = 0.2

    # logging
    frames_dir: Path = Path("figures/gif_frames")
    make_gif: bool = False
    gif_name: str = "bo_evolution.gif"
    frame_duration_ms: int = 1500
    last_frame_duration_ms: int = 5000


@dataclass
class MultiSearchConfig:
    n_samples: int        # total linear GNN solve budget, including anchor evaluations
    basis_size: int
    basis_selection_rule: str
    enforce_family_coverage: bool = False  # when picking final basis
    include_fixed_ops: list[str] = None  # e.g. ["X", "L1", "H1"]
    diversity_lambda: float = 0.0  # λ for greedy_diversity rule (0 = pure top-k)
    # Anchor specifications for each family.
    # Accepted types:
    #   None          → no anchors for this family
    #   float         → single anchor at that parameter value
    #   list[float]   → explicit anchor positions (each counts as one sample)
    #   int (n > 0)   → n equally-spaced anchors interior to the resolved parameter range
    mu_anchors: float | list | int | None = 1.0       # gaussian family anchors
    tausqrt_anchors: float | list | int | None = 1.5  # heat family anchors
    sampling_strategy: str = "cross_family_ucb"  # "cross_family_ucb" | "per_family"


class MultiOperatorSearch:
    def __init__(
        self,
        goblin: GOBLIN,
        cfg: MultiSearchConfig,
        searches: dict[str, OperatorSearch],
    ):
        """
        searches: mapping family_name -> OperatorSearch
                  e.g. {"heat": heat_search, "gaussian": gauss_search}
        """
        self.goblin = goblin
        self.cfg = cfg
        self.searches = searches

        self.sampled_ops: list[OpId] = []
        self.sampled_scores: dict[OpId, float] = {}

    # --------------------------------------------------
    # Main UCB loop
    # --------------------------------------------------
    def run(self):
        # ---- Fixed operators ----
        if self.cfg.include_fixed_ops:
            for name in self.cfg.include_fixed_ops:
                op = OpId("fixed", name)
                acc = self.goblin.eval_accuracy_lin_fixed(name)
                self.sampled_ops.append(op)
                self.sampled_scores[op] = acc

        # ---- Anchor initial observations ----
        # Evaluated before the UCB loop; fed into each GP as its first observation.
        # All anchors count toward n_samples.
        n_anchors = 0
        _anchor_specs = {"gaussian": self.cfg.mu_anchors, "heat": self.cfg.tausqrt_anchors}
        for fam_name, anchor_spec in _anchor_specs.items():
            if fam_name not in self.searches:
                continue
            search = self.searches[fam_name]
            anchor_params = _resolve_anchors(anchor_spec, search.family.grid)
            if not anchor_params:
                continue
            lo, hi = float(search.family.grid.min()), float(search.family.grid.max())
            oob = [p for p in anchor_params if p < lo or p > hi]
            if oob:
                warnings.warn(
                    f"{fam_name} anchors {oob} are outside the search grid [{lo:.3g}, {hi:.3g}]; "
                    "they will inform the GP but cannot be re-queried by UCB.",
                    stacklevel=2,
                )
            acc_stds = [search.family.eval_fn(p, standardized=True) for p in anchor_params]
            acc_raws = [search.family.eval_fn(p, standardized=False) for p in anchor_params]
            # Replace the default GP initialization with the anchor observations
            search.param_obs = list(anchor_params)
            search.y_obs = [-s for s in acc_stds]
            for p, acc_raw in zip(anchor_params, acc_raws):
                op = OpId(fam_name, float(p))
                self.sampled_ops.append(op)
                self.sampled_scores[op] = acc_raw
                n_anchors += 1

        # ---- UCB loop ----
        beta = self.goblin.operator_search_cfg.ucb_beta
        n_ucb_steps = self.cfg.n_samples - n_anchors
        if n_ucb_steps < 0:
            warnings.warn(
                f"n_anchors ({n_anchors}) exceeds n_samples ({self.cfg.n_samples}); "
                "UCB loop will be skipped. Increase n_samples or reduce anchor count.",
                stacklevel=2,
            )
            n_ucb_steps = 0

        if self.cfg.sampling_strategy == "per_family":
            # Each family gets an independent UCB budget; no cross-family competition.
            # Prevents a lucky anchor in one family from starving the other.
            active = list(self.searches.items())
            base, rem = divmod(n_ucb_steps, len(active))
            for i, (fam_name, search) in enumerate(active):
                for _ in tqdm(range(base + (1 if i < rem else 0))):
                    param, _ = search.suggest_ucb(beta)
                    search.observe("ucb", param)
                    op = OpId(fam_name, float(param))
                    self.sampled_ops.append(op)
                    self.sampled_scores[op] = search.family.eval_fn(float(param), standardized=False)
        else:
            # Cross-family UCB: all families compete for the shared sample budget.
            for _ in tqdm(range(n_ucb_steps)):
                best_params: dict[str, float] = {}
                best_acqs: dict[str, float] = {}
                for fam_name, search in self.searches.items():
                    param, acq_val = search.suggest_ucb(beta)
                    best_params[fam_name] = param
                    best_acqs[fam_name] = acq_val

                fam = max(best_acqs, key=best_acqs.get)
                param_next = best_params[fam]
                search = self.searches[fam]

                search.observe("ucb", param_next)
                op = OpId(fam, float(param_next))
                acc = search.family.eval_fn(float(param_next), standardized=False)
                self.sampled_ops.append(op)
                self.sampled_scores[op] = acc

        return self.sampled_ops, self.sampled_scores

    # --------------------------------------------------
    # Mixed basis selection
    # --------------------------------------------------
    def select_mixed_basis(self) -> list[OpId]:
        """Select a mixed basis of operators, potentially from different families."""
        basis_size = self.cfg.basis_size  # fixed do not count

        ranked = sorted(
            self.sampled_scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

        chosen: list[OpId] = []

        # Always include fixed ops (anchors)
        for op, _ in ranked:
            if op.family == "fixed":
                chosen.append(op)
                basis_size += 1

        for op, _ in ranked:
            if len(chosen) >= basis_size:
                break
            if op in chosen:
                continue
            chosen.append(op)

        # optional: enforce at least one op per family
        if self.cfg.enforce_family_coverage and self.cfg.basis_size >= len(
            self.families
        ):
            for fam in self.families:
                if not any(o.family == fam for o in chosen):
                    repl = next((o for o, _ in ranked if o.family == fam), None)
                    if repl is not None:
                        chosen[-1] = repl

        # stable dedupe
        out, seen = [], set()
        for o in chosen:
            if o not in seen:
                out.append(o)
                seen.add(o)
        return out


    def select_greedy_diversity_basis(self) -> list[OpId]:
        """Greedy basis selection penalizing cosine similarity of Y_L^eval predictions."""
        lam = self.cfg.diversity_lambda
        eval_idxs = self.goblin.splits["train_eval"]

        def get_pred_vec(op: OpId) -> torch.Tensor | None:
            yhat_full = self.goblin._get_op_yhat_bo(op)  # (N, C)
            flat = yhat_full[eval_idxs].reshape(-1).float()
            norm = flat.norm()
            return flat / norm if norm > 1e-12 else None

        pred_vecs = {op: get_pred_vec(op) for op in self.sampled_ops}

        # Fixed ops always in basis first
        chosen = [op for op in self.sampled_ops if op.family == "fixed"]
        remaining = [op for op in self.sampled_ops if op.family != "fixed"]
        total_size = self.cfg.basis_size + len(chosen)

        while len(chosen) < total_size and remaining:
            best_op, best_score = None, -float("inf")
            for op in remaining:
                score = self.sampled_scores[op]
                if lam > 0 and pred_vecs[op] is not None:
                    chosen_vecs = [pred_vecs[c] for c in chosen if pred_vecs.get(c) is not None]
                    if chosen_vecs:
                        sims = [float((pred_vecs[op] * v).sum()) for v in chosen_vecs]
                        score -= lam * max(sims)
                if score > best_score:
                    best_score, best_op = score, op
            if best_op is None:
                break
            chosen.append(best_op)
            remaining.remove(best_op)

        # stable dedupe
        out, seen = [], set()
        for o in chosen:
            if o not in seen:
                out.append(o)
                seen.add(o)
        return out


class OperatorSearch:
    """
    Two-phase operator search on k:
      - explore: maximin (space-filling)
      - exploit: maximize GP mean with min distance constraint
    Logging: saves frames each iter + GIF.
    """

    def __init__(
        self,
        goblin: GOBLIN,
        family: OperatorFamily,
        config: OperatorSearchConfig,
        initialization_obs: list[float] | None = None,
    ):
        self.goblin = goblin
        self.C = goblin.C
        self.cfg = config
        self.family = family
        self.param_grid = family.grid

        self.seed = goblin.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.kernel = RBF(length_scale=self.cfg.rbf_length_scale) + WhiteKernel(
            noise_level=self.cfg.white_noise
        )

        self.param_obs: list[float] = (
            _as_float_list(initialization_obs)
            if initialization_obs is not None
            else [float(self.param_grid[0])]
        )

        self.y_obs: list[float] = [
            -self.family.eval_fn(p, standardized=True) for p in self.param_obs
        ]  # y = -standardized_acc

        self.selected_exploit: list[float] = []
        self.selected_exploit_acc: list[float] = []

        self.gp: Optional[GaussianProcessRegressor] = None
        self.gp_mean: Optional[np.ndarray] = None
        self.gp_std: Optional[np.ndarray] = None

    # ---- plotting ----
    def _plot_frame(self, t: int, phase: str, param_next: Optional[float] = None):
        mean = self.gp_mean
        std = self.gp_std
        assert mean is not None and std is not None

        # Convert back to plain accuracy for plotting:
        # std_acc = (acc - 1/C)/(1-1/C)  =>  acc = std_acc*(1-1/C) + 1/C
        mean_acc = (-mean) * (1 - 1 / self.C) + 1 / self.C
        std_acc = std * (1 - 1 / self.C)

        plt.figure(figsize=(8, 4))

        plt.scatter(
            self.param_obs,
            [self.family.eval_fn(p, standardized=False) for p in self.param_obs],
            c="red",
            zorder=2,
            label="Sampled (eval acc)",
        )

        # exploit points
        if len(self.selected_exploit) > 0:
            plt.scatter(
                self.selected_exploit,
                self.selected_exploit_acc,
                c="purple",
                s=80,
                edgecolors="black",
                zorder=4,
                label="Exploit samples",
            )
            for param, acc in zip(self.selected_exploit, self.selected_exploit_acc):
                plt.text(
                    param,
                    acc + 0.02,
                    f"{self.family.param_name}={param:.2f}",
                    color="purple",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    zorder=5,
                )

        plt.plot(self.param_grid, mean_acc, label="GP mean (eval acc)")
        plt.fill_between(
            self.param_grid, mean_acc - 2 * std_acc, mean_acc + 2 * std_acc, alpha=0.2
        )

        if param_next is not None:
            plt.axvline(
                param_next,
                color="black",
                linestyle="--",
                alpha=0.6,
                label=f"Next {self.family.param_name} ({phase})",
            )

        plt.xlabel(self.family.param_name)
        plt.ylabel("Eval accuracy")
        plt.ylim(0.4, 1.1)
        plt.title(f"BO iteration {t + 1} [{phase}]")
        plt.legend()
        plt.tight_layout()

        frame_path = self.cfg.frames_dir / f"frame_{t:03d}.png"
        plt.savefig(frame_path)
        plt.close()

    def _make_gif(self, gif_path: Path):
        frames = [
            Image.open(p) for p in sorted(self.cfg.frames_dir.glob("frame_*.png"))
        ]
        if len(frames) == 0:
            raise RuntimeError("No frames found to build a GIF.")

        durations = [self.cfg.frame_duration_ms] * (len(frames) - 1) + [
            self.cfg.last_frame_duration_ms
        ]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )
        return gif_path

    # ---- selection rules ----
    def _fit_gp(self):
        """Fit GP on self.y_obs."""
        gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True)
        gp.fit(np.array(self.param_obs).reshape(-1, 1), np.array(self.y_obs))
        gp_mean, gp_std = gp.predict(self.param_grid.reshape(-1, 1), return_std=True)
        self.gp, self.gp_mean, self.gp_std = gp, gp_mean, gp_std

    def suggest_ucb(self, beta: float) -> tuple[float, float]:
        """Fit GP and return (best_param, acq_value) for cross-family comparison.

        acq(θ) = -gp_mean(θ) + beta * gp_std(θ)
        GP stores y = -standardized_acc, so -gp_mean ≈ standardized_acc.
        Already-sampled points are masked to -inf.
        Returns argmax param and its acquisition value.
        """
        self._fit_gp()
        acq = -self.gp_mean + beta * self.gp_std
        for p in self.param_obs:
            acq[np.isclose(self.param_grid, p, atol=1e-6)] = -np.inf
        best_idx = int(np.argmax(acq))
        return float(self.param_grid[best_idx]), float(acq[best_idx])

    def observe(self, phase: str, param_next: float):
        """Record an observation of the parameter and its evaluation."""
        self.param_obs.append(float(param_next))
        self.y_obs.append(-self.family.eval_fn(float(param_next), standardized=True))
        if phase == "exploit":
            self.selected_exploit.append(float(param_next))
            self.selected_exploit_acc.append(
                self.family.eval_fn(float(param_next), standardized=False)
            )


# -----------------------------
# ExpertsDeepSet
# -----------------------------
@dataclass
class ExpertsDeepSetConfig:
    hidden_dim: int = 32
    attn_temp: float = 5.0
    num_deepset_layers: int = 2
    num_head_layers: int = 2
    lr: float = 1e-3
    epochs: int = 1000
    log_every: int = 200
    dropout: float = 0.0  # TODO implement
    score_feature: str = "none"  # "none" | "trimmed" | "trimmed_and_lower_half"
    weight_selection: str = "current"  # "current" | "pre_filter" | "mask_by_deepset"
    feature_set_size: str = "all"      # "all" | "top_half" | "basis_size"; ignored when weight_selection == "current"
    # stochastic training mode
    n_batches: int = 1000
    batch_size: int = 128
    n_ref_per_class: int = 5
    # random operator pool training
    use_random_op_pool: bool = False
    pool_size: int = 50          # operators in pool (split evenly gaussian/heat)
    n_ops_per_batch: int = 5     # operators sampled from pool per batch


class ExpertsDeepSet(nn.Module):
    """
    DeepSet over operator-expert features
    """

    def __init__(self, feat_dim: int, cfg: ExpertsDeepSetConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg

        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        H = cfg.hidden_dim

        # phi: per-expert embedding
        layers = []
        in_dim = feat_dim
        for _ in range(cfg.num_deepset_layers):
            layers += [
                nn.Linear(in_dim, H),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = H
        self.phi = nn.Sequential(*layers)

        # psi: scoring head takes [e_i, sum e]
        head = []
        head_in = 2 * H
        for _ in range(cfg.num_head_layers - 1):
            head += [
                nn.Linear(head_in, H),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            head_in = H
        head += [nn.Linear(head_in, 1)]
        self.psi = nn.Sequential(*head)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H_0: (N, t, feat_dim) -> alpha: (N, t)
        E = self.phi(H)  # (N,t,hidden)
        E_agg = E.sum(dim=1, keepdim=True)  # (N,1,hidden)
        E_agg = E_agg.expand_as(E)  # repeat along t dim
        scores = self.psi(torch.cat([E, E_agg], dim=-1)).squeeze(
            -1
        )  # [N, t, 2h] -> [N, t]
        alpha = F.softmax(scores / self.cfg.attn_temp, dim=1)
        return alpha

    @staticmethod
    def build_features(
        Yhat: torch.Tensor, score_feat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Yhat: (N, t, C) probabilities from t experts.
        score_feat: optional (t, n_scores) standardized BO scores per operator.
        Returns H_0: (N, t, 4+n_scores) DeepSet input feats.
        """
        N, t, C = Yhat.shape
        diff = (Yhat.unsqueeze(2) - Yhat.unsqueeze(1)).pow(2).sum(dim=-1)  # (N,t,t)
        mask = ~torch.eye(t, device=Yhat.device).bool()
        diff_ij = diff[:, mask].view(N, t, t - 1)

        mean_diff = diff_ij.mean(dim=-1)
        var_diff = diff_ij.var(dim=-1)
        min_diff = diff_ij.min(dim=-1).values
        max_diff = diff_ij.max(dim=-1).values

        H = torch.stack([mean_diff, var_diff, min_diff, max_diff], dim=-1)  # (N, t, 4)
        if score_feat is not None:
            # score_feat: (t, n_scores) -> broadcast to (N, t, n_scores)
            sf = score_feat.unsqueeze(0).expand(N, -1, -1).to(H.device)
            H = torch.cat([H, sf], dim=-1)
        return H

    @staticmethod
    def _apply_weight_selection(
        alpha: torch.Tensor,            # (N, M)
        Yhat_N: torch.Tensor,           # (N, M, C)
        basis_indices: list[int] | None,
        k_mask: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-process alpha for weight selection.

        Returns (Yhat_bar (N,C), alpha_eff (N,k_eff)):
          - pre_filter (basis_indices set): k_eff = len(basis_indices)
          - mask_by_deepset (k_mask set):  k_eff = k_mask (zeros for non-selected)
          - current (neither set):         k_eff = M (no change)
        """
        if basis_indices is not None:
            idx = torch.tensor(basis_indices, device=alpha.device)
            alpha_k = alpha[:, idx]
            alpha_k = alpha_k / alpha_k.sum(dim=1, keepdim=True).clamp(min=1e-8)
            Yhat_bar = (alpha_k.unsqueeze(-1) * Yhat_N[:, idx]).sum(dim=1)
            return Yhat_bar, alpha_k
        elif k_mask is not None:
            k = min(k_mask, alpha.shape[1])
            topk_idx = alpha.topk(k, dim=1).indices
            mask = torch.zeros_like(alpha).scatter_(1, topk_idx, 1.0)
            alpha_m = alpha * mask
            alpha_m = alpha_m / alpha_m.sum(dim=1, keepdim=True).clamp(min=1e-8)
            Yhat_bar = (alpha_m.unsqueeze(-1) * Yhat_N).sum(dim=1)
            return Yhat_bar, alpha_m
        else:
            Yhat_bar = (alpha.unsqueeze(-1) * Yhat_N).sum(dim=1)
            return Yhat_bar, alpha

    def train_model(
        self,
        Yhat: torch.Tensor,
        y_class: torch.Tensor,
        train_idx: torch.Tensor,
        *,
        score_feat: torch.Tensor | None = None,
        verbose: bool = True,
        basis_indices: list[int] | None = None,
        k_mask: int | None = None,
    ) -> list[float]:
        """
        Train on given train_idx split (should be train_eval, with Yhat solved from from train_fit).
        Returns loss_curve: list of per-epoch loss values.
        """
        H = self.build_features(Yhat, score_feat)
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        loss_curve: list[float] = []

        for epoch in range(self.cfg.epochs):
            self.train()
            alpha = self(H[train_idx])
            Yhat_bar, _ = self._apply_weight_selection(alpha, Yhat[train_idx], basis_indices, k_mask)

            loss = self.loss_fn(Yhat_bar, y_class[train_idx])
            # loss = F.nll_loss((Yhat_bar + 1e-8).log(), y_class[train_idx])
            if loss.isnan():
                raise RuntimeError(f"NaN loss at epoch {epoch}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_curve.append(loss.item())
            if verbose and (epoch % self.cfg.log_every == 0):
                print(f"Epoch {epoch}, loss={loss.item():.4f}")

        print(f"Final epoch {epoch}, loss={loss.item():.4f}")
        return loss_curve

    @torch.no_grad()
    def evaluate(
        self,
        Yhat: torch.Tensor,
        y_class: torch.Tensor,
        splits: dict[str, torch.Tensor],
        *,
        score_feat: torch.Tensor | None = None,
        basis_indices: list[int] | None = None,
        k_mask: int | None = None,
    ) -> dict[str, float]:
        self.eval()
        H = self.build_features(Yhat, score_feat)
        res = {}
        for split, split_idx in splits.items():
            alpha_eval = self(H[split_idx])
            Yhat_bar, alpha_eff = self._apply_weight_selection(
                alpha_eval, Yhat[split_idx], basis_indices, k_mask
            )
            preds = Yhat_bar.argmax(dim=1)
            acc = (preds == y_class[split_idx]).float().mean().item()
            res[f"{split}/acc"] = acc
            res[f"{split}/mean_alpha"] = alpha_eff.mean(dim=0).cpu()
            for expert_idx in range(alpha_eff.shape[1]):
                yhat_idx = basis_indices[expert_idx] if basis_indices is not None else expert_idx
                expert_pred = Yhat[:, yhat_idx][split_idx].argmax(dim=1)
                expert_acc = (expert_pred == y_class[split_idx]).float().mean()
                res[f"{split}/acc/expert{expert_idx}"] = expert_acc.item()

            # Include MeanAgg, as a baseline
            Yhat_meanagg = Yhat[split_idx].mean(dim=1)
            meanagg_preds = Yhat_meanagg.argmax(dim=1)
            meanagg_acc = (meanagg_preds == y_class[split_idx]).float().mean().item()
            res[f"{split}/acc/meanagg"] = meanagg_acc

            # ----- Alpha entropy diagnostics (over effective alpha) -----
            eps = 1e-12
            alpha_safe = alpha_eff.clamp(min=eps)

            # Per-node entropy: (N_split,)
            node_entropy = -(alpha_safe * alpha_safe.log()).sum(dim=1)

            # Normalize by log(t) so values are in [0, 1]
            max_entropy = np.log(alpha_eff.shape[1])
            node_entropy_norm = node_entropy / max_entropy

            # Aggregate stats
            res[f"{split}/alpha_entropy_mean"] = node_entropy_norm.mean().item()
            res[f"{split}/alpha_entropy_std"] = node_entropy_norm.std().item()

            # Entropy of mean alpha (global preference)
            mean_alpha = alpha_eff.mean(dim=0)
            mean_alpha_safe = mean_alpha.clamp(min=eps)
            mean_alpha_entropy = -(mean_alpha_safe * mean_alpha_safe.log()).sum()
            mean_alpha_entropy_norm = mean_alpha_entropy / max_entropy

            res[f"{split}/alpha_entropy_mean_alpha"] = mean_alpha_entropy_norm.item()

        return res

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path, map_location: str = "cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))


def sample_nodes_per_class(
    y_class: torch.Tensor,
    pool_idx: torch.Tensor,
    n_per_class: int,
    C: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample up to n_per_class nodes per class from pool_idx. Mirrors GraphAny's
    sample_k_nodes_per_label. Returns concatenated global node indices."""
    sampled = []
    pool_labels = y_class[pool_idx]
    for c in range(C):
        cls_mask = (pool_labels == c).nonzero(as_tuple=False).view(-1)
        if cls_mask.numel() == 0:
            continue
        perm = torch.randperm(cls_mask.numel(), generator=generator)
        sampled.append(pool_idx[cls_mask[perm[:n_per_class]]])
    return torch.cat(sampled) if sampled else torch.empty(0, dtype=torch.long)


# -----------------------------
# GOBLIN (top-level)
# -----------------------------
class GOBLIN:
    """
    Graph Operator Basis Learning & INference.

    One instance per graph:
      - OperatorSearch to find basis ks (or accept externally provided)
      - ExpertsDeepSet to train/eval/serve mixture weights on that graph
    """

    def __init__(
        self,
        *,
        data: PygGraph,
        X: torch.Tensor,  # (N, d)
        all_pairs_dist: torch.Tensor,  # (N, N)
        y_onehot: torch.Tensor,  # (N, C)
        y_class: torch.Tensor,  # (N,) in {0,1}
        splits: dict[str, torch.Tensor],
        sigma: float,
        C: int,
        operator_search_cfg: OperatorSearchConfig,
        multi_search_cfg: MultiSearchConfig,
        deepset_cfg: ExpertsDeepSetConfig,
        initialization_obs: Optional[List[float]] = None,
        seed: int = 0,
        dataset_name: str = "",
        cache_dir: Path | None = None,
    ):
        self.data = data
        self.dataset_name = dataset_name
        self.X = X
        self.all_pairs_dist = all_pairs_dist
        self.A = (all_pairs_dist == 1) if all_pairs_dist is not None else None
        self.N = X.shape[0]
        self.seed = seed
        self.cache_dir = cache_dir if cache_dir is not None else DATA_CACHE
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.y_onehot = y_onehot  # (N, C)
        self.y_class = y_class  # (N,)
        self.splits = splits

        self.sigma = float(sigma) if sigma is not None else None
        self.C = int(C)

        # Cache for LinHeat operators, and move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L_sym, self.L_sym_eigvals, self.L_sym_eigvecs = get_L_sym_eigvals_eigvecs(
            data, N_exact=5000
        )
        self.L_sym = self.L_sym.to(self.device)
        if self.L_sym_eigvals is not None:
            self.L_sym_eigvals = self.L_sym_eigvals.to(self.device)
            self.L_sym_eigvecs = self.L_sym_eigvecs.to(self.device)

        self.operator_search_cfg = operator_search_cfg
        self.bo_objective = operator_search_cfg.bo_objective
        self.basis_selection_rule = multi_search_cfg.basis_selection_rule

        # Cache for expensive-to-compute Yhats.
        # NOTE: only ever written to during operator search, never DeepSet training/eval.
        self._lin_heat_cache: dict[tuple[tuple[str, ...], float], torch.Tensor] = (
            {}
        )  # ((split_str,), tau) -> Yhat  [keys are actual τ = tausqrt**2]
        self._lin_gaussian_cache: dict[tuple[tuple[str, ...], float], torch.Tensor] = {}

        # Create operator families
        self.operator_searches = {}
        families = operator_search_cfg.families
        assert all(
            fam in ("heat", "gaussian") for fam in families
        ), f"Unknown family in {families}"
        family_defs = self.make_operator_families(operator_search_cfg)

        for fam in families:
            self.operator_searches[fam] = OperatorSearch(
                goblin=self,
                family=family_defs[fam],
                config=operator_search_cfg,
                initialization_obs=initialization_obs,
            )
        self.multi_search = MultiOperatorSearch(
            self, multi_search_cfg, self.operator_searches
        )
        self.deepset_cfg = deepset_cfg
        self.deepset: Optional[ExpertsDeepSet] = None

        self.Yhat: Optional[torch.Tensor] = None  # (N,t,C)
        self._basis_Fs: Optional[list] = None     # cached F matrices per basis op
        self._last_loss_curve: list[float] = []
        self._pool_ops: list[OpId] = []
        self._pool_Fs: list[torch.Tensor] = []

        self.basis: list[OpId] | None = (
            None  # e.g. (|"gaussian", 2), ("fixed", "L1"), etc
        )

        # Weight selection state (set by train_deepset, consumed by eval_deepset)
        self._ws_basis_indices: list[int] | None = None
        self._ws_k_mask: int | None = None
        self._ws_score_feat: torch.Tensor | None = None


    def make_operator_families(
        self, cfg: OperatorSearchConfig
    ) -> dict[str, OperatorFamily]:
        """Create all supported operator families."""
        fams = {}
        fams["gaussian"] = OperatorFamily(
            name="gaussian",
            param_name="mu",
            grid=np.linspace(cfg.mu_min, cfg.mu_max, cfg.mu_num),
            eval_fn=lambda mu, standardized: self.eval_accuracy_lin_gaussian(
                mu, standardized
            ),
        )
        TAU_MAX = 10.0
        tausqrt_max = min(cfg.tausqrt_max, TAU_MAX ** 0.5)
        fams["heat"] = OperatorFamily(
            name="heat",
            param_name="tausqrt",
            grid=np.linspace(cfg.tausqrt_min, tausqrt_max, cfg.tausqrt_num),
            eval_fn=lambda tausqrt, standardized: self.eval_accuracy_lin_heat(
                tausqrt**2, standardized
            ),
        )
        return fams

    def run_operator_search(self):
        """
        Run operator search (single or multi-family).
        """
        ops, scores = self.multi_search.run()
        return {
            "sampled_ops": ops,
            "sampled_scores": scores,
        }

    # -----------------------------
    # Solve linear GNN -> logits
    # -----------------------------

    def solve_linear_gnn(self, F: torch.Tensor, splits: list[str]) -> torch.Tensor:
        """
        Given operator * data F=SX for linear GNN FW, fit W on given splits and return Yhat=FW* logits for all nodes.
        """
        # Use all train labels for deepset feats, and train_fit split for operator search
        label_idxs = torch.cat([self.splits[s] for s in splits])
        F_fit = F[label_idxs]  # (n_fit, d)

        # Fit W to predict y (in {-1,+1})
        W = torch.linalg.pinv(F_fit) @ self.y_onehot[label_idxs].float()
        Yhat = F @ W  # (N, C)
        return Yhat

    # -----------------------------
    # Evaluation metrics
    # -----------------------------
    def eval_accuracy_lin_gaussian(
        self,
        mu: float,
        standardized: bool = False,
    ) -> float:
        """Get accuracy of LinGaussian operator with given mu on train_fit split, for BO."""
        # Get Yhat fit on train_fit
        splits = ("train_fit",)
        if (splits, mu) not in self._lin_gaussian_cache:
            F = apply_lin_gaussian_operator(
                # all_pairs_dist=self.all_pairs_dist,
                X=self.X,
                mu=mu,
                sigma=self.sigma,
                cache_dir=(self.cache_dir / f"apspd/lingauss/{self.dataset_name}"),
            )
            Yhat = self.solve_linear_gnn(F=F, splits=list(splits))
            self._lin_gaussian_cache[(splits, mu)] = Yhat
        else:
            Yhat = self._lin_gaussian_cache[(splits, mu)]

        return self.eval_accuracy(Yhat=Yhat, standardized=standardized)

    def eval_accuracy_lin_heat(
        self,
        tau: float,
        standardized: bool = False,
    ) -> float:
        """Get accuracy of LinHeat operator with given tau on train_fit split, for BO."""
        # Get Yhat fit on train_fit
        splits = ("train_fit",)
        if (splits, tau) not in self._lin_heat_cache:
            X = self.X.to(self.L_sym.device)
            F = apply_lin_heat_operator(
                tau=tau,
                L_sym=self.L_sym,
                X=X,
                eigvals=self.L_sym_eigvals,
                eigvecs=self.L_sym_eigvecs,
            )
            Yhat = self.solve_linear_gnn(F=F, splits=list(splits))
            self._lin_heat_cache[(splits, tau)] = Yhat
        else:
            Yhat = self._lin_heat_cache[(splits, tau)]

        return self.eval_accuracy(Yhat=Yhat, standardized=standardized)

    def eval_accuracy_lin_fixed(
        self,
        name: str,
        standardized: bool = False,
    ) -> float:
        """Get accuracy of fixed (X, L1-2, H1-2) operator on train_fit split, for BO."""
        F = get_fixed_operator(self.data, self.X, operator_name=name)
        Yhat = self.solve_linear_gnn(F=F, splits=["train_fit"])

        return self.eval_accuracy(Yhat=Yhat, standardized=standardized)

    def eval_accuracy(
        self,
        Yhat: torch.Tensor,
        standardized: bool = False,
        bo_objective: str | None = None,
    ) -> float:
        """Given a linear GNNs logits [N, C], evaluate accuracy on train_eval split, for BO."""
        # Evaluate on train_eval
        eval_idxs = self.splits["train_eval"]
        logits = Yhat[eval_idxs]
        y_true = self.y_class[eval_idxs]
        preds = logits.argmax(dim=1)  # (N_eval,)
        correct = (preds == y_true).float()

        # For objectives involving sorting by acc
        true_logit = logits[torch.arange(len(y_true)), y_true]

        # best competing logit
        tmp = logits.clone()
        tmp[torch.arange(len(y_true)), y_true] = -float("inf")
        best_other = tmp.max(dim=1).values

        margin = true_logit - best_other  # (N_eval,)

        # sort from confidently correct -> confidently wrong
        order = torch.argsort(margin, descending=True)
        correct_sorted = correct[order]

        n = len(correct_sorted)

        # Specific objectives
        obj = bo_objective if bo_objective is not None else self.bo_objective

        if obj == "mean":
            acc = correct.mean()

        elif obj.startswith("trimmed_"):
            q = int(float(obj.split("_")[1]) / 100 * n)
            keep = correct_sorted[q : n - q]
            acc = keep.mean()

        elif obj == "lower_quartile":
            q = int(0.25 * n)
            keep = correct_sorted[n - q :]
            acc = keep.mean()

        elif obj == "lower_half":
            q = int(0.5 * n)
            keep = correct_sorted[n - q :]
            acc = keep.mean()

        elif obj == "upper_quartile":
            q = int(0.25 * n)
            keep = correct_sorted[:q]
            acc = keep.mean()

        elif obj.startswith("mean_minus_var_"):
            lam = float(obj.split("_")[-1])
            acc = correct.mean() - lam * correct.var(unbiased=False)

        else:
            raise ValueError(f"Unknown BO objective: {obj}")

        if standardized:
            acc_std = (acc - 1 / self.C) / (1 - 1 / self.C)
            return acc_std.item()

        return acc.item()

    def update_basis(self, new_ops: list[OpId], overwrite: bool = False):
        if overwrite or self.basis is None:
            self.basis = []
            self._basis_Fs = None
        assert self.basis is not None
        seen = set(self.basis)
        for op in new_ops:
            if op not in seen:
                self.basis.append(op)
                seen.add(op)

    def select_basis(self) -> list[OpId]:
        """
        Select final operator basis after operator search.
        """
        if self.basis_selection_rule == "diverse_top_k":
            basis = self.multi_search.select_mixed_basis()
        elif self.basis_selection_rule == "greedy_diversity":
            basis = self.multi_search.select_greedy_diversity_basis()
        else:
            raise ValueError(
                f"Unknown basis selection rule: {self.basis_selection_rule}"
            )
        self.update_basis(basis, overwrite=True)
        return basis

    # ---- Score features for DeepSet ----

    def _get_op_yhat_bo(self, op: OpId) -> torch.Tensor:
        """Return Yhat (train_fit split) for a basis op — from BO cache or freshly computed."""
        splits = ("train_fit",)
        if op.family == "gaussian":
            return self._lin_gaussian_cache[(splits, op.param)]
        elif op.family == "heat":
            return self._lin_heat_cache[(splits, op.param ** 2)]
        elif op.family == "fixed":
            F = get_fixed_operator(self.data, self.X, operator_name=op.param)
            return self.solve_linear_gnn(F=F, splits=list(splits))
        else:
            raise ValueError(f"Unknown family: {op.family}")

    def get_score_feat_tensor(self) -> torch.Tensor | None:
        """
        Build (t, n_score_dims) standardized score features for the current basis.
        Returns None if score_feature == "none" or no basis is set.

        Both features use fixed objectives (independent of bo_objective):
          - "trimmed": trimmed_20 mean accuracy, standardized as (acc - 1/C) / (1 - 1/C)
          - "trimmed_and_lower_half": above + lower_half accuracy, standardized

        Yhat is retrieved from the BO cache (train_fit split), so objective re-evaluation
        requires no GNN recomputation.
        """
        sf = self.deepset_cfg.score_feature
        if sf == "none" or self.basis is None:
            return None
        rows = []
        for op in self.basis:
            yhat = self._get_op_yhat_bo(op)
            trimmed = self.eval_accuracy(yhat, bo_objective="trimmed_20", standardized=True)
            if sf == "trimmed":
                rows.append([trimmed])
            else:  # "trimmed_and_lower_half"
                lower = self.eval_accuracy(yhat, bo_objective="lower_half", standardized=True)
                rows.append([trimmed, lower])
        return torch.tensor(rows, dtype=torch.float32)  # (t, 1 or 2)

    def get_score_feat_for_ops(self, ops: list[OpId]) -> torch.Tensor | None:
        """Like get_score_feat_tensor but for an arbitrary ops list."""
        sf = self.deepset_cfg.score_feature
        if sf == "none":
            return None
        rows = []
        for op in ops:
            yhat = self._get_op_yhat_bo(op)
            trimmed = self.eval_accuracy(yhat, bo_objective="trimmed_20", standardized=True)
            if sf == "trimmed":
                rows.append([trimmed])
            else:  # "trimmed_and_lower_half"
                lower = self.eval_accuracy(yhat, bo_objective="lower_half", standardized=True)
                rows.append([trimmed, lower])
        return torch.tensor(rows, dtype=torch.float32)

    def _cosine_dedup(self, ops: list[OpId], threshold: float = 0.99) -> list[OpId]:
        """Remove near-duplicate operators by cosine similarity of train_eval predictions."""
        eval_idxs = self.splits["train_eval"]
        kept: list[OpId] = []
        kept_vecs: list[torch.Tensor] = []
        for op in ops:
            yhat = self._get_op_yhat_bo(op)
            flat = yhat[eval_idxs].reshape(-1).float()
            norm = flat.norm()
            vec = flat / norm if norm > 1e-12 else flat
            if not any(float((vec * kv).sum()) > threshold for kv in kept_vecs):
                kept.append(op)
                kept_vecs.append(vec)
        return kept

    def get_feature_ops(self) -> list[OpId]:
        """Return the feature set ops per weight_selection and feature_set_size config."""
        ws = self.deepset_cfg.weight_selection
        fss = self.deepset_cfg.feature_set_size
        if ws == "current" or fss == "basis_size":
            return list(self.basis)
        dedup_ops = self._cosine_dedup(list(self.multi_search.sampled_ops))
        if fss == "all":
            result = dedup_ops
        elif fss == "top_half":
            fixed = [op for op in dedup_ops if op.family == "fixed"]
            non_fixed = sorted(
                [op for op in dedup_ops if op.family != "fixed"],
                key=lambda op: self.multi_search.sampled_scores[op],
                reverse=True,
            )
            k = max(len(non_fixed) // 2, self.multi_search.cfg.basis_size)
            result = fixed + non_fixed[:k]
        else:
            raise ValueError(f"Unknown feature_set_size: {fss}")
        # Guarantee all basis ops are present — _cosine_dedup may drop them if they
        # are very similar to another sampled op that was kept first.
        basis_set = set(self.basis)
        missing = [op for op in self.basis if op not in set(result)]
        return missing + result

    def get_Yhat_for_ops(self, ops: list[OpId], splits: list[str]) -> torch.Tensor:
        """Like get_Yhat_experts but for an arbitrary ops list. Does not set self.Yhat."""
        Yhat = []
        for op in ops:
            if op.family == "gaussian":
                mu, key = op.param, (tuple(splits), op.param)
                if key in self._lin_gaussian_cache:
                    Yhat.append(self._lin_gaussian_cache[key])
                    continue
                F = apply_lin_gaussian_operator(
                    X=self.X,
                    mu=mu,
                    sigma=self.sigma,
                    cache_dir=(self.cache_dir / f"apspd/lingauss/{self.dataset_name}"),
                )
            elif op.family == "heat":
                tausqrt = op.param
                tau = tausqrt**2
                key = (tuple(splits), tau)
                if key in self._lin_heat_cache:
                    Yhat.append(self._lin_heat_cache[key])
                    continue
                F = apply_lin_heat_operator(
                    tau=tau,
                    L_sym=self.L_sym,
                    X=self.X,
                    eigvals=self.L_sym_eigvals,
                    eigvecs=self.L_sym_eigvecs,
                )
            elif op.family == "fixed":
                F = get_fixed_operator(self.data, self.X, operator_name=op.param)
            else:
                raise ValueError(f"Unknown family: {op.family}")

            Yhat_expert = self.solve_linear_gnn(F=F, splits=splits)
            if op.family == "gaussian":
                self._lin_gaussian_cache[key] = Yhat_expert
            elif op.family == "heat":
                self._lin_heat_cache[key] = Yhat_expert
            Yhat.append(Yhat_expert)

        return torch.stack(Yhat, dim=1)  # (N, M, C)

    # ---- Build LinearGNN experts ----
    def get_Yhat_experts(self, splits: list[str]) -> torch.Tensor:
        """Get linear GNNs Yhat fit to training labels for each basis"""
        if self.basis is None:
            raise ValueError("No basis set. Run select_basis() or set_basis().")

        Yhat = []
        for op in self.basis:
            if op.family == "gaussian":
                # Load if cached
                mu, key = op.param, (tuple(splits), op.param)
                if key in self._lin_gaussian_cache:
                    Yhat_expert = self._lin_gaussian_cache[key]
                    Yhat.append(Yhat_expert)
                    continue
                # Compute
                F = apply_lin_gaussian_operator(
                    X=self.X,
                    mu=mu,
                    sigma=self.sigma,
                    cache_dir=(self.cache_dir / f"apspd/lingauss/{self.dataset_name}"),
                )
            elif op.family == "heat":
                tausqrt = op.param
                tau = tausqrt**2
                key = (tuple(splits), tau)  # cache keyed by actual τ, matching eval_accuracy_lin_heat
                # Load if cached
                if key in self._lin_heat_cache:
                    Yhat_expert = self._lin_heat_cache[key]
                    Yhat.append(Yhat_expert)
                    continue
                F = apply_lin_heat_operator(
                    tau=tau,
                    L_sym=self.L_sym,
                    X=self.X,
                    eigvals=self.L_sym_eigvals,
                    eigvecs=self.L_sym_eigvecs,
                )
            elif op.family == "fixed":
                F = get_fixed_operator(self.data, self.X, operator_name=op.param)
            else:
                raise ValueError(f"Unknown basis family: {op.family}")

            Yhat_expert = self.solve_linear_gnn(F=F, splits=splits)
            if op.family == "gaussian":
                self._lin_gaussian_cache[key] = Yhat_expert
            if op.family == "heat":
                self._lin_heat_cache[key] = Yhat_expert

            Yhat.append(Yhat_expert)  # *******

        Yhat = torch.stack(Yhat, dim=1)
        self.Yhat = Yhat
        return Yhat  # (N, t, C)

    def get_operator_Fs(self) -> list:
        """Compute (or return cached) the raw F matrix for each basis operator.
        F is (N, d) — the operator-transformed features used to fit each linear GNN.
        Cached in self._basis_Fs; invalidated when basis changes via update_basis."""
        if self._basis_Fs is not None:
            return self._basis_Fs
        if self.basis is None:
            raise ValueError("No basis set. Run select_basis() first.")
        Fs = []
        for op in self.basis:
            if op.family == "gaussian":
                F = apply_lin_gaussian_operator(
                    X=self.X,
                    mu=op.param,
                    sigma=self.sigma,
                    cache_dir=(self.cache_dir / f"apspd/lingauss/{self.dataset_name}"),
                )
            elif op.family == "heat":
                F = apply_lin_heat_operator(
                    tau=op.param ** 2,
                    L_sym=self.L_sym,
                    X=self.X.to(self.L_sym.device),
                    eigvals=self.L_sym_eigvals,
                    eigvecs=self.L_sym_eigvecs,
                )
            elif op.family == "fixed":
                F = get_fixed_operator(self.data, self.X, operator_name=op.param)
            else:
                raise ValueError(f"Unknown family: {op.family}")
            Fs.append(F)
        self._basis_Fs = Fs
        return Fs

    def build_random_op_pool(self) -> None:
        """Sample a random pool of operators (uniform over µ/τ grids) and cache their F matrices.
        Called once before train_deepset_pool. Splits pool evenly between gaussian and heat."""
        cfg = self.deepset_cfg
        cfg_s = self.operator_search_cfg
        rng = np.random.default_rng(self.seed)

        ops: list[OpId] = []
        Fs: list[torch.Tensor] = []

        # Gaussian half
        n_gauss = cfg.pool_size // 2
        mus = rng.uniform(cfg_s.mu_min, cfg_s.mu_max, n_gauss)
        for mu in mus:
            F = apply_lin_gaussian_operator(
                X=self.X, mu=float(mu), sigma=self.sigma,
                cache_dir=self.cache_dir / f"apspd/lingauss/{self.dataset_name}",
            )
            ops.append(OpId(family="gaussian", param=float(mu)))
            Fs.append(F)

        # Heat half
        n_heat = cfg.pool_size - n_gauss
        tausqrts = rng.uniform(cfg_s.tausqrt_min, cfg_s.tausqrt_max, n_heat)
        for tausqrt in tausqrts:
            F = apply_lin_heat_operator(
                tau=float(tausqrt) ** 2, L_sym=self.L_sym,
                X=self.X.to(self.L_sym.device),
                eigvals=self.L_sym_eigvals, eigvecs=self.L_sym_eigvecs,
            )
            ops.append(OpId(family="heat", param=float(tausqrt)))
            Fs.append(F)

        self._pool_ops = ops
        self._pool_Fs = Fs
        print(f"Built random op pool: {len(ops)} operators ({n_gauss} Gaussian, {n_heat} Heat)")

    # ---- DeepSet ----
    def init_deepset(self):
        self.get_Yhat_experts(splits=["train_fit"])
        score_feat = self.get_score_feat_tensor()
        feat_dim = ExpertsDeepSet.build_features(self.Yhat, score_feat).shape[-1]
        self.deepset = ExpertsDeepSet(
            feat_dim=feat_dim, cfg=self.deepset_cfg, seed=self.seed
        )
        return self.deepset

    def train_deepset(self, verbose: bool = True) -> Dict[str, float]:
        ws = self.deepset_cfg.weight_selection

        if self.deepset is None:
            self.init_deepset()  # feat_dim = 4 + n_score_dims regardless of feature set size

        train_idx = torch.cat([self.splits[split] for split in ["train_eval"]])

        if ws == "current":
            self.get_Yhat_experts(splits=["train_fit"])
            score_feat = self.get_score_feat_tensor()
            basis_indices, k_mask = None, None
            feature_ops = None
        else:
            feature_ops = self.get_feature_ops()
            self.Yhat = self.get_Yhat_for_ops(feature_ops, splits=["train_fit"])
            score_feat = self.get_score_feat_for_ops(feature_ops)
            if ws == "pre_filter":
                basis_indices = [feature_ops.index(op) for op in self.basis]
                k_mask = None
            else:  # mask_by_deepset
                basis_indices = None
                k_mask = len(self.basis)

        assert self.deepset is not None and self.Yhat is not None

        # Train on *train_eval* labels, with feats derived from Yhat fit to *train_fit* split
        loss_curve = self.deepset.train_model(
            Yhat=self.Yhat,
            y_class=self.y_class,
            train_idx=train_idx,
            score_feat=score_feat,
            verbose=verbose,
            basis_indices=basis_indices,
            k_mask=k_mask,
        )
        self._last_loss_curve = loss_curve

        # Refit on full train set before eval
        if ws == "current":
            self.get_Yhat_experts(splits=["train_fit", "train_eval"])
        else:
            self.Yhat = self.get_Yhat_for_ops(feature_ops, splits=["train_fit", "train_eval"])

        # Store weight selection state for eval_deepset
        self._ws_basis_indices = basis_indices
        self._ws_k_mask = k_mask
        self._ws_score_feat = score_feat

        metrics = self.deepset.evaluate(
            Yhat=self.Yhat,
            y_class=self.y_class,
            splits={"val": self.splits["val"], "train": train_idx},
            score_feat=score_feat,
            basis_indices=basis_indices,
            k_mask=k_mask,
        )
        return metrics

    def train_deepset_stochastic(self, verbose: bool = True) -> Dict[str, float]:
        """Stochastic mini-batch training following GraphAny's approach.

        Each batch:
          1. Randomly partition train_fit ∪ train_eval into Vtarget (batch_size) and a ref pool
          2. Sample n_ref_per_class nodes per class from the ref pool → Vref
          3. For each cached F: refit W = pinv(F[Vref]) @ Y[Vref], compute Yhat = F @ W
          4. Build DeepSet features, forward on Vtarget, backprop

        F matrices are computed once and cached; only W changes each batch (cheap pseudoinverse).
        """
        cfg = self.deepset_cfg
        if self.deepset is None:
            self.init_deepset()
        assert self.deepset is not None

        Fs = self.get_operator_Fs()
        all_train_idx = torch.cat([self.splits["train_fit"], self.splits["train_eval"]])
        score_feat = self.get_score_feat_tensor()
        opt = torch.optim.Adam(self.deepset.parameters(), lr=cfg.lr)
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        loss_curve: list[float] = []

        for batch_idx in range(cfg.n_batches):
            self.deepset.train()

            # Partition: first batch_size → Vtarget, rest → Vref pool
            perm = torch.randperm(all_train_idx.numel(), generator=gen)
            n_target = min(cfg.batch_size, all_train_idx.numel())
            target_idx = all_train_idx[perm[:n_target]]
            ref_pool = all_train_idx[perm[n_target:]]

            ref_idx = sample_nodes_per_class(
                self.y_class, ref_pool, cfg.n_ref_per_class, self.C, gen
            )
            if ref_idx.numel() == 0:
                # edge case: pool empty, fall back to full train set minus target
                ref_idx = sample_nodes_per_class(
                    self.y_class, all_train_idx, cfg.n_ref_per_class, self.C, gen
                )

            # Refit W per operator using Vref
            Yhat_list = []
            for F in Fs:
                F_cpu = F.cpu()
                Y_ref = self.y_onehot[ref_idx].float()
                W = torch.linalg.pinv(F_cpu[ref_idx]) @ Y_ref
                Yhat_list.append(F_cpu @ W)
            Yhat = torch.stack(Yhat_list, dim=1)  # (N, t, C) on CPU

            H = ExpertsDeepSet.build_features(Yhat, score_feat)
            alpha = self.deepset(H[target_idx])
            Yhat_bar = (alpha.unsqueeze(-1) * Yhat[target_idx]).sum(dim=1)
            loss = self.deepset.loss_fn(Yhat_bar, self.y_class[target_idx])

            if loss.isnan():
                raise RuntimeError(f"NaN loss at batch {batch_idx}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_curve.append(loss.item())
            if verbose and batch_idx % cfg.log_every == 0:
                print(f"Batch {batch_idx}, loss={loss.item():.4f}")

        print(f"Final batch {batch_idx}, loss={loss_curve[-1]:.4f}")
        self._last_loss_curve = loss_curve

        # Refit Yhat on full train set for eval (mirrors train_deepset)
        self.get_Yhat_experts(splits=["train_fit", "train_eval"])
        train_idx = torch.cat([self.splits[s] for s in ["train_fit", "train_eval"]])
        return self.deepset.evaluate(
            Yhat=self.Yhat,
            y_class=self.y_class,
            splits={"val": self.splits["val"], "train": train_idx},
            score_feat=score_feat,
        )

    def train_deepset_pool(self, verbose: bool = True) -> Dict[str, float]:
        """Pool-based stochastic training.

        Each batch:
          1. Sample n_ops_per_batch operators uniformly from the random pool
          2. Partition train nodes into Vtarget (batch_size) and Vref pool
          3. Sample n_ref_per_class nodes per class from Vref pool → Vref
          4. Refit W = pinv(F[Vref]) @ Y[Vref] for each selected operator
          5. Compute Yhat = F @ W; optionally compute per-op score feature from accuracy on Vtarget
          6. Build DeepSet features, forward on Vtarget, backprop

        At eval time, reverts to basis operators (from BO) refit on full train set.
        """
        cfg = self.deepset_cfg
        if self.deepset is None:
            self.init_deepset()
        assert self.deepset is not None

        if not self._pool_ops:
            self.build_random_op_pool()

        all_train_idx = torch.cat([self.splits["train_fit"], self.splits["train_eval"]])
        opt = torch.optim.Adam(self.deepset.parameters(), lr=cfg.lr)
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        loss_curve: list[float] = []

        pool_size = len(self._pool_ops)
        n_ops = cfg.n_ops_per_batch
        need_score = cfg.score_feature != "none"
        n_score_dims = 2 if cfg.score_feature == "trimmed_and_lower_half" else 1

        for batch_idx in range(cfg.n_batches):
            self.deepset.train()

            # Sample n_ops_per_batch from pool (without replacement)
            op_perm = torch.randperm(pool_size, generator=gen)[:n_ops]
            Fs_batch = [self._pool_Fs[i].cpu() for i in op_perm.tolist()]

            # Partition labeled nodes: Vtarget | Vref pool
            perm = torch.randperm(all_train_idx.numel(), generator=gen)
            n_target = min(cfg.batch_size, all_train_idx.numel())
            target_idx = all_train_idx[perm[:n_target]]
            ref_pool = all_train_idx[perm[n_target:]]

            ref_idx = sample_nodes_per_class(
                self.y_class, ref_pool, cfg.n_ref_per_class, self.C, gen
            )
            if ref_idx.numel() == 0:
                ref_idx = sample_nodes_per_class(
                    self.y_class, all_train_idx, cfg.n_ref_per_class, self.C, gen
                )

            # Refit W per selected operator on Vref, compute Yhat
            Yhat_list = []
            Y_ref = self.y_onehot[ref_idx].float()
            for F in Fs_batch:
                W = torch.linalg.pinv(F[ref_idx]) @ Y_ref
                Yhat_list.append(F @ W)

            Yhat = torch.stack(Yhat_list, dim=1)  # (N, n_ops, C)

            # Per-batch score features: standardized accuracy on Vtarget
            if need_score:
                rows = []
                y_target = self.y_class[target_idx]
                for Yhat_op in Yhat_list:
                    preds = Yhat_op[target_idx].argmax(-1)
                    acc = (preds == y_target).float().mean().item()
                    acc_std = (acc - 1.0 / self.C) / max(1.0 - 1.0 / self.C, 1e-8)
                    if n_score_dims == 1:
                        rows.append([acc_std])
                    else:
                        rows.append([acc_std, acc_std])  # duplicate for lower_half slot
                score_feat: torch.Tensor | None = torch.tensor(rows, dtype=torch.float32)
            else:
                score_feat = None

            H = ExpertsDeepSet.build_features(Yhat, score_feat)
            alpha = self.deepset(H[target_idx])
            Yhat_bar = (alpha.unsqueeze(-1) * Yhat[target_idx]).sum(dim=1)
            loss = self.deepset.loss_fn(Yhat_bar, self.y_class[target_idx])

            if loss.isnan():
                raise RuntimeError(f"NaN loss at batch {batch_idx}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_curve.append(loss.item())
            if verbose and batch_idx % cfg.log_every == 0:
                print(f"Batch {batch_idx}, loss={loss.item():.4f}")

        print(f"Final batch {batch_idx}, loss={loss_curve[-1]:.4f}")
        self._last_loss_curve = loss_curve

        # Eval using basis operators (from BO), refit on full train set
        self.get_Yhat_experts(splits=["train_fit", "train_eval"])
        train_idx = torch.cat([self.splits[s] for s in ["train_fit", "train_eval"]])
        score_feat_eval = self.get_score_feat_tensor()
        return self.deepset.evaluate(
            Yhat=self.Yhat,
            y_class=self.y_class,
            splits={"val": self.splits["val"], "train": train_idx},
            score_feat=score_feat_eval,
        )

    def eval_deepset(self, splits: list[str]) -> Dict[str, float]:
        if self.deepset is None:
            raise ValueError(
                "DeepSet not initialised. Call init_deepset() or train_deepset() or load_deepset()."
            )
        # Use weight selection state set by train_deepset; fall back to basis score feat
        score_feat = (
            self._ws_score_feat
            if self._ws_score_feat is not None
            else self.get_score_feat_tensor()
        )
        eval_splits = {split: self.splits[split] for split in splits}
        eval_splits["+".join(splits)] = torch.cat(
            [self.splits[split] for split in splits]
        )
        return self.deepset.evaluate(
            Yhat=self.Yhat,
            y_class=self.y_class,
            splits=eval_splits,
            score_feat=score_feat,
            basis_indices=self._ws_basis_indices,
            k_mask=self._ws_k_mask,
        )

    def save_deepset(self, path: Path):
        if self.deepset is None:
            raise ValueError("DeepSet not initialised.")
        self.deepset.save(path)

    def load_deepset(self, path: Path, map_location: str = "cpu"):
        if self.deepset is None:
            self.init_deepset()
        assert self.deepset is not None
        self.deepset.load(path, map_location=map_location)
