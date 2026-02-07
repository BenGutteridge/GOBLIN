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
    num_explore_steps: int
    num_exploit_steps: int

    # LinGaussian
    mu_min: float = 0.0
    mu_max: float = 10.0
    mu_num: int = 250
    search_min_sep_mu: float = 0.2

    # LinHeat
    tau_min: float = 0.0
    tau_max: float = 5.0
    tau_num: int = 50
    search_min_sep_tau: float = 0.1

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
    total_steps: int  # total BO steps across both families
    mix_strategy: str  # "balanced" | "biased" | "adaptive"
    basis_size: int
    basis_min_sep_mu: float
    basis_min_sep_tau: float
    basis_selection_rule: str
    bias_prob: float = 0.5  # used for bias strategies
    enforce_family_coverage: bool = False  # when picking final basis
    include_fixed_ops: list[str] = None  # e.g. ["X", "L1", "H1"]


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
        self.families = list(searches.keys())
        self.rng = np.random.default_rng(goblin.seed)

        self.sampled_ops: list[OpId] = []
        self.sampled_scores: dict[OpId, float] = {}

    # --------------------------------------------------
    # Family selection
    # --------------------------------------------------
    def _choose_family(self, t: int) -> str:
        families = self.families

        # Single-family = degenerate multi-family
        if len(families) == 1:
            return families[0]

        if self.cfg.mix_strategy == "balanced":
            return families[t % len(families)]

        if self.cfg.mix_strategy == "biased":
            # two-family only; bias toward first family
            return (
                families[0] if self.rng.random() < self.cfg.bias_prob else families[1]
            )

        if self.cfg.mix_strategy == "adaptive":
            if t < self.cfg.total_steps // 2:
                return families[t % len(families)]

            # bias toward family with best score so far
            best_by_family = {}
            for f in families:
                best_by_family[f] = max(
                    (v for k, v in self.sampled_scores.items() if k.family == f),
                    default=-1e9,
                )
            return max(best_by_family, key=best_by_family.get)

        raise ValueError(f"Unknown mix_strategy: {self.cfg.mix_strategy}")

    # --------------------------------------------------
    # Main BO loop
    # --------------------------------------------------
    def run(self):
        # ---- Inject fixed operators (optional) ----
        if self.cfg.include_fixed_ops:
            for name in self.cfg.include_fixed_ops:
                op = OpId("fixed", name)
                acc = self.goblin.eval_accuracy_lin_fixed(name)
                self.sampled_ops.append(op)
                self.sampled_scores[op] = acc

        # ---- BO loop ----
        for t in tqdm(range(self.cfg.total_steps)):
            fam = self._choose_family(t)
            search = self.searches[fam]

            phase, param_next = search.suggest_next(
                t=min(t, search.cfg.num_explore_steps + search.cfg.num_exploit_steps)
            )
            if param_next is None:
                continue

            # update GP for that family
            search.observe(phase, param_next)

            # record globally
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

        def ok_sep(op: OpId) -> bool:
            if op.family not in self.searches:
                return True
            for c in chosen:
                if c.family != op.family:
                    continue
                dist = abs(float(c.param) - float(op.param))
                min_sep = (
                    self.cfg.basis_min_sep_tau
                    if op.family == "heat"
                    else self.cfg.basis_min_sep_mu
                )
                if dist < min_sep:
                    return False
            return True

        for op, _ in ranked:
            if len(chosen) >= basis_size:
                break
            if op in chosen:
                continue
            if not ok_sep(op):
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
        gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True)
        gp.fit(np.array(self.param_obs).reshape(-1, 1), np.array(self.y_obs))
        gp_mean, gp_std = gp.predict(self.param_grid.reshape(-1, 1), return_std=True)
        self.gp, self.gp_mean, self.gp_std = gp, gp_mean, gp_std

    def _pick_explore(self) -> float:
        # maximin in k-space (space-filling)
        distances = np.min(
            np.abs(self.param_grid[:, None] - np.array(self.param_obs)[None, :]), axis=1
        )
        # skip already sampled (just in case)
        for idx in np.argsort(-distances):
            k_candidate = float(self.param_grid[idx])
            if not any(np.isclose(k_candidate, self.param_obs, atol=1e-6)):
                return k_candidate
        # fallback (shouldn't happen)
        return float(self.param_grid[np.argmax(distances)])

    def _pick_exploit(self) -> float:
        # maximize mean in original objective: we stored y = -std_acc, so
        # minimizing mean corresponds to maximizing standardized_acc (and thus accuracy).
        assert self.gp_mean is not None
        sorted_idx = np.argsort(self.gp_mean)  # ascending mean -> best

        for idx in sorted_idx:
            param_candidate = float(self.param_grid[idx])

            # skip if already sampled
            if any(np.isclose(param_candidate, self.param_obs, atol=1e-6)):
                continue

            # spacing among exploit points
            min_sep = (
                self.cfg.search_min_sep_tau
                if self.family.name == "heat"
                else self.cfg.search_min_sep_mu
            )
            if all(
                abs(param_candidate - param_prev) >= min_sep
                for param_prev in self.selected_exploit
            ):
                return param_candidate

        # fallback: reuse best (may already be sampled)
        return float(self.param_grid[np.argmin(self.gp_mean)])

    def suggest_next(self, t: int) -> tuple[str, float | None]:
        """
        Returns (phase, param_next). param_next=None means 'final' frame.
        """
        num_total = self.cfg.num_explore_steps + self.cfg.num_exploit_steps

        self._fit_gp()

        if t < self.cfg.num_explore_steps:
            return "explore", self._pick_explore()
        elif t < num_total:
            return "exploit", self._pick_exploit()
        else:
            return "final", None

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
    def build_features(Yhat: torch.Tensor) -> torch.Tensor:
        """
        Yhat: (N, t, C) probabilities from t experts.
        Returns H_0: (N, t, 4) DeepSet input feats, using pairwise squared distances summary stats.
        """
        N, t, C = Yhat.shape
        diff = (Yhat.unsqueeze(2) - Yhat.unsqueeze(1)).pow(2).sum(dim=-1)  # (N,t,t)
        mask = ~torch.eye(t, device=Yhat.device).bool()
        diff_ij = diff[:, mask].view(N, t, t - 1)

        mean_diff = diff_ij.mean(dim=-1)
        var_diff = diff_ij.var(dim=-1)
        min_diff = diff_ij.min(dim=-1).values
        max_diff = diff_ij.max(dim=-1).values

        H = torch.stack([mean_diff, var_diff, min_diff, max_diff], dim=-1)
        return H  # (N, t, 4)

    def train_model(
        self,
        Yhat: torch.Tensor,
        y_class: torch.Tensor,
        train_idx: torch.Tensor,
        *,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train on given train_idx split (should be train_eval, with Yhat solved from from train_fit).
        """
        H = self.build_features(Yhat)
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

        for epoch in range(self.cfg.epochs):
            self.train()
            alpha = self(H[train_idx])
            Yhat_bar = (alpha.unsqueeze(-1) * Yhat[train_idx]).sum(dim=1)

            loss = self.loss_fn(Yhat_bar, y_class[train_idx])
            # loss = F.nll_loss((Yhat_bar + 1e-8).log(), y_class[train_idx])
            if loss.isnan():
                raise RuntimeError(f"NaN loss at epoch {epoch}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            if verbose and (epoch % self.cfg.log_every == 0):
                print(f"Epoch {epoch}, loss={loss.item():.4f}")

        print(f"Final epoch {epoch}, loss={loss.item():.4f}")
        return

    @torch.no_grad()
    def evaluate(
        self,
        Yhat: torch.Tensor,
        y_class: torch.Tensor,
        splits: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        self.eval()
        H = self.build_features(Yhat)
        res = {}
        for split, split_idx in splits.items():
            alpha_eval = self(H[split_idx])
            Yhat_bar = (alpha_eval.unsqueeze(dim=-1) * Yhat[split_idx]).sum(dim=1)
            preds = Yhat_bar.argmax(dim=1)
            acc = (preds == y_class[split_idx]).float().mean().item()
            res[f"{split}/acc"] = acc
            res[f"{split}/mean_alpha"] = alpha_eval.mean(dim=0).cpu()
            for expert_idx in range(Yhat.shape[1]):
                expert_pred = Yhat[:, expert_idx][split_idx].argmax(dim=1)
                expert_acc = (expert_pred == y_class[split_idx]).float().mean()
                res[f"{split}/acc/expert{expert_idx}"] = expert_acc.item()

            # Include MeanAgg, as a baseline
            Yhat_meanagg = Yhat[split_idx].mean(dim=1)
            meanagg_preds = Yhat_meanagg.argmax(dim=1)
            meanagg_acc = (meanagg_preds == y_class[split_idx]).float().mean().item()
            res[f"{split}/acc/meanagg"] = meanagg_acc

            # ----- Alpha entropy diagnostics -----
            eps = 1e-12
            alpha_safe = alpha_eval.clamp(min=eps)

            # Per-node entropy: (N_split,)
            node_entropy = -(alpha_safe * alpha_safe.log()).sum(dim=1)

            # Normalize by log(t) so values are in [0, 1]
            max_entropy = np.log(alpha_eval.shape[1])
            node_entropy_norm = node_entropy / max_entropy

            # Aggregate stats
            res[f"{split}/alpha_entropy_mean"] = node_entropy_norm.mean().item()
            res[f"{split}/alpha_entropy_std"] = node_entropy_norm.std().item()

            # Entropy of mean alpha (global preference)
            mean_alpha = alpha_eval.mean(dim=0)
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
    ):
        self.data = data
        self.dataset_name = dataset_name
        self.X = X
        self.all_pairs_dist = all_pairs_dist
        self.A = all_pairs_dist == 1
        self.N = X.shape[0]
        self.seed = seed
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

        self.bo_objective = operator_search_cfg.bo_objective
        self.basis_selection_rule = multi_search_cfg.basis_selection_rule

        # Cache for expensive-to-compute Yhats.
        # NOTE: only ever written to during operator search, never DeepSet training/eval.
        self._lin_heat_cache: dict[tuple[tuple[str, ...], float], torch.Tensor] = (
            {}
        )  # ((split_str,), tau) -> Yhat
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

        self.basis: list[OpId] | None = (
            None  # e.g. (|"gaussian", 2), ("fixed", "L1"), etc
        )

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
        fams["heat"] = OperatorFamily(
            name="heat",
            param_name="tau",
            grid=np.linspace(cfg.tau_min, cfg.tau_max, cfg.tau_num),
            eval_fn=lambda tau, standardized: self.eval_accuracy_lin_heat(
                tau, standardized
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
                cache_dir=Path(f"data_cache/lingauss/{self.dataset_name}"),
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

    def eval_accuracy(self, Yhat: torch.Tensor, standardized: bool = False) -> float:
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
        obj = self.bo_objective

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
        else:
            raise ValueError(
                f"Unknown basis selection rule: {self.basis_selection_rule}"
            )
        self.update_basis(basis, overwrite=True)
        return basis

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
                    cache_dir=Path(f"data_cache/lingauss/{self.dataset_name}"),
                )
            elif op.family == "heat":
                tau, key = op.param, (tuple(splits), op.param)
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

    # ---- DeepSet ----
    def init_deepset(self):
        self.get_Yhat_experts(splits=["train_fit"])

        feat_dim = ExpertsDeepSet.build_features(self.Yhat).shape[-1]
        self.deepset = ExpertsDeepSet(
            feat_dim=feat_dim, cfg=self.deepset_cfg, seed=self.seed
        )
        return self.deepset

    def train_deepset(self, verbose: bool = True) -> Dict[str, float]:
        self.get_Yhat_experts(splits=["train_fit"])
        if self.deepset is None:
            self.init_deepset()
        assert self.deepset is not None and self.Yhat is not None

        train_idx = torch.cat([self.splits[split] for split in ["train_eval"]])

        # Train on *train_eval* labels, with feats derived from Yhat fit to *train_fit* split
        self.deepset.train_model(
            Yhat=self.Yhat,
            y_class=self.y_class,
            train_idx=train_idx,
            verbose=verbose,
        )

        # Refit on full train set before eval
        self.get_Yhat_experts(splits=["train_fit", "train_eval"])
        metrics = self.deepset.evaluate(
            Yhat=self.Yhat,
            y_class=self.y_class,
            splits={"val": self.splits["val"], "train": train_idx},
        )
        return metrics

    def eval_deepset(self, splits: list[str]) -> Dict[str, float]:
        self.get_Yhat_experts(splits=["train_fit", "train_eval"])
        if self.deepset is None:
            raise ValueError(
                "DeepSet not initialised. Call init_deepset() or train_deepset() or load_deepset()."
            )
        eval_splits = {split: self.splits[split] for split in splits}
        eval_splits["+".join(splits)] = torch.cat(
            [self.splits[split] for split in splits]
        )
        return self.deepset.evaluate(
            Yhat=self.Yhat,
            y_class=self.y_class,
            splits=eval_splits,
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
