from functools import partial
import dgl
from dgl import DGLGraph
import torch


def dgl_graph_to_adj_mat(g: dgl.DGLGraph) -> torch.Tensor:
    num_nodes = g.num_nodes()
    edge_index = torch.stack(g.adj().coo())

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0

    return adj


def dgl_to_sparse_rw_adj(g: dgl.DGLGraph) -> torch.sparse.FloatTensor:
    """Convert DGL graph to torch sparse random-walk normalized adjacency matrix."""
    src, dst = g.edges()
    N = g.num_nodes()

    deg = torch.bincount(dst, minlength=N).float()
    deg_inv = torch.zeros_like(deg)
    deg_inv[deg > 0] = 1.0 / deg[deg > 0]

    values = deg_inv[dst]
    indices = torch.stack([dst, src])

    A_rw = torch.sparse_coo_tensor(
        indices,
        values,
        size=(N, N),
    )
    return A_rw.coalesce()


def sampled_jacobian_wrapper(
    base_fn,
    *,
    N: int,
    d: int,
    C: int,
    out_nodes: torch.Tensor | None = None,  # indices into N
    out_ch: (
        torch.Tensor | None
    ) = None,  # indices into C, can be vector or [N, ?] matrix if indexing per-node (e.g. for correct class only)
    in_nodes: torch.Tensor | None = None,  # indices into N
    in_ch: torch.Tensor | None = None,  # indices into d
    device: torch.device,
):
    """
    base_fn: function X_full -> Y_hat, Y_hat shape [N, C]

    Returns a function suitable for torch.autograd.functional.jacobian
    that takes X_sub of shape [len(in_nodes), len(in_ch)].
    """

    # ---- normalize indices (None -> full range) ----
    in_nodes = in_nodes if in_nodes is not None else torch.arange(N, device=device)
    in_ch = in_ch if in_ch is not None else torch.arange(d, device=device)
    out_nodes = out_nodes if out_nodes is not None else torch.arange(N, device=device)
    out_ch = out_ch if out_ch is not None else torch.arange(C, device=device)

    def wrapped(X_sub: torch.Tensor):
        """
        X_sub: [len(in_nodes), len(in_ch)]
        """
        # ---- reconstruct full X ----
        X_full = torch.zeros((N, d), device=device, dtype=X_sub.dtype)
        X_full[in_nodes[:, None], in_ch] = X_sub

        # ---- forward ----
        Y = base_fn(X_full)  # [N, C]

        # ---- restrict output ----
        Y = Y[out_nodes, :]
        if len(out_ch.shape) == 2:
            assert (
                out_nodes.shape[0] == out_ch.shape[0]
            ), "If out_ch is per-node, its first dimension must match out_nodes."
            # per-node channel indexing
            Y = Y[torch.arange(Y.shape[0], device=device), out_ch.reshape(-1)]
        else:
            # global channel indexing
            Y = Y[:, out_ch]

        return Y

    return wrapped


def generate_adj_power_F(
    X: torch.Tensor,
    A_rw: torch.sparse.FloatTensor,
    n_hops: int,
    low_pass: bool = True,
) -> tuple[torch.Tensor, torch.sparse.FloatTensor]:
    """Generates features F according to standard GraphAny feats, A^kX or (I-A)^kX according to n_hops and low pass.
    Also returns the sparse matrix S such that F = S X.
    """
    N, device = X.shape[0], X.device
    # Compute F from SX, where S is given by n_hops and low_pass and is a power of A_rw or I - A_rw

    # Initialize S as sparse identity and F as X
    idxs = torch.arange(N, device=device)
    S: torch.sparse.FloatTensor = torch.sparse_coo_tensor(
        indices=torch.stack([idxs, idxs]),  # shape [2, N]
        values=torch.ones(N, device=device),
        size=(N, N),
    ).coalesce()
    F: torch.Tensor = X

    for _ in range(n_hops):
        if low_pass:
            F = torch.sparse.mm(A_rw, F)
            S = torch.sparse.mm(A_rw, S)  # stays sparse
        else:
            F = F - torch.sparse.mm(A_rw, F)
            S = S - torch.sparse.mm(A_rw, S)

    return F, S


def solve_linear_gnn(
    F: torch.Tensor,  # [N, d]
    Y: torch.Tensor,
    label_idxs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute \hat{Y} = F . F_L^+ / Y_L"""
    F_L = F[label_idxs, :]  # [N_L, d]
    Y_L = Y[label_idxs, :]  # [N_L, C]
    W = torch.linalg.lstsq(F_L, Y_L, driver="gelss").solution  # [d, C], from Y_hat = FW
    Y_hat = F @ W  # [N, C]

    return W, Y_hat


def get_linear_gnn_jacobians(
    X: torch.Tensor,
    Y: torch.Tensor,
    A_rw: torch.sparse.FloatTensor,
    label_idxs: torch.Tensor,
    n_hops: int,
    low_pass: bool = True,
    out_nodes: torch.Tensor | None = None,
    out_ch: torch.Tensor | None = None,
    in_nodes: torch.Tensor | None = None,
    in_ch: torch.Tensor | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given graph structure, node features, known labels, plus specifics of the Linear GNN,
    compute the linear GNN pred features and their Jacobian (the pred nodes/channels wrt. the input nodes/feats).
    Returns:
        y_feat: torch.Tensor, shape [N, C], the linear GNN predicted features
        jac: torch.Tensor, shape [len(out_nodes), len(out_ch), N, d],
                the Jacobian of y_feat wrt X, sparse in the input dimensions if in_nodes/in_ch are not full
    """

    # Wrapper for gnn solver that bakes in all args except node feats and only outputs Y_hat
    def differentiable_get_linear_gnn_feats(X: torch.Tensor) -> torch.Tensor:
        """Full Linear GNN feat computation with embedded args so that we can differentiate output wrt X."""
        # Generate standard GraphAny feats
        F, _ = generate_adj_power_F(
            X=X,
            A_rw=A_rw.to(device),
            n_hops=n_hops,
            low_pass=low_pass,
        )  # F: [N, d], S: sparse [N, N]

        # Analytically solve
        _, Y_hat = solve_linear_gnn(
            F=F,
            Y=Y,
            label_idxs=label_idxs,
        )
        return Y_hat

    # Get actual feats
    y_feat = differentiable_get_linear_gnn_feats(X.to(device))  # [N, C]
    C = y_feat.shape[1]
    N, d = X.shape

    if out_ch is not None and len(out_ch.shape) == 2:
        if (out_nodes is None and out_ch.shape[0] != N) or (
            out_nodes is not None and out_ch.shape[0] != out_nodes.shape[0]
        ):
            raise ValueError(
                "out_ch should be of shape [len(out_nodes),] or [N, out_nodes] if specifying per-node channels."
            )

    # Normalize input indices exactly the same way as the wrapper
    in_nodes = in_nodes if in_nodes is not None else torch.arange(N, device=device)
    in_ch = in_ch if in_ch is not None else torch.arange(d, device=device)

    X_sub = X.to(device)[in_nodes][:, in_ch].detach().requires_grad_(True)

    # Get Jacobian of feats wrt input
    sampled_differentiable_get_linear_gnn_feats = sampled_jacobian_wrapper(
        differentiable_get_linear_gnn_feats,
        N=N,
        d=d,
        C=C,
        out_nodes=out_nodes,
        out_ch=out_ch,
        in_nodes=in_nodes,
        in_ch=in_ch,
        device=device,
    )

    # Compute Jacobian (sampled or otherwise)
    jac = torch.autograd.functional.jacobian(
        sampled_differentiable_get_linear_gnn_feats,
        X_sub,
    )  # [N, C, N, d] (len(out_nodes), len(out_ch), N, d) (sampling -> sparse in N, d)

    return y_feat, jac
