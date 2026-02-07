import os
import dgl
import networkx as nx
from sympy import deg
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch_geometric.utils import to_torch_coo_tensor
from time import time
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.datasets import (
    Planetoid,
    CitationFull,
    Coauthor,
    Amazon,
    WikipediaNetwork,
    Actor,
    AttributedGraphDataset,
    WikiCS,
    Reddit,
    Airports,
    HeterophilousGraphDataset,
    WebKB,
)
from torch_geometric.data import Data as PygGraph

from ogb.nodeproppred import PygNodePropPredDataset


def apspd_to_tensor(G: PygGraph | nx.Graph) -> torch.Tensor:
    """Convert all pairs shortest path distances from networkx dict to tensor."""
    if isinstance(G, PygGraph):
        G = to_networkx(G, to_undirected=True)
    else:
        G = G
    N = G.number_of_nodes()
    all_pairs_dist = dict(nx.all_pairs_shortest_path_length(G))
    dist_tensor = torch.full((N, N), -1)  # -1 means unreachable
    for u in range(N):
        for v, dist in all_pairs_dist[u].items():
            dist_tensor[u, v] = dist
    return dist_tensor


def get_L_sym_eigvals_eigvecs(
    data: PygGraph,  # PyG Data object
    N_exact: int,
):
    """
    Compute symmetric normalized Laplacian L_sym and (optionally) its eigendecomposition.
    """
    N = data.num_nodes

    # Sparse adjacency (COO)
    A = to_torch_coo_tensor(
        data.edge_index,
        size=(N, N),
    ).float()

    # Degree vector
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg[deg == 0] = 1.0  # avoid divide-by-zero

    # D^{-1/2}
    deg_inv_sqrt = deg.pow(-0.5)

    # Normalized adjacency: D^{-1/2} A D^{-1/2}
    D_inv_sqrt_A = A * deg_inv_sqrt.view(-1, 1)
    A_norm = D_inv_sqrt_A * deg_inv_sqrt.view(1, -1)

    # L_sym = I - A_norm
    if N <= N_exact:
        L_sym = torch.eye(N) - A_norm.to_dense()
        eigvals, eigvecs = torch.linalg.eigh(L_sym)
        return L_sym, eigvals, eigvecs

    # Large graph: keep sparse
    L_sym = torch.eye(N) - A_norm
    return L_sym, None, None


def build_and_cache_distance_operators(
    apspd: torch.Tensor,  # (N, N) int, CPU
    max_dist: int,
    cache_dir: str,
):
    """
    Build sparse matrices M_d where (i,j) ∈ M_d iff apspd[i,j] == d.
    Saves one file per distance.
    """
    assert apspd.device.type == "cpu"
    os.makedirs(cache_dir, exist_ok=True)

    N = apspd.shape[0]

    for d in range(max_dist + 1):
        path = os.path.join(cache_dir, f"M_dist_{d}.pt")
        if os.path.exists(path):
            print(f"Distance operator for dist={d} already cached at {path}. Skipping.")
            continue

        idx = (apspd == d).nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue

        values = torch.ones(idx.shape[0], dtype=torch.float32)
        M_d = torch.sparse_coo_tensor(
            idx.t(),
            values,
            size=(N, N),
        ).coalesce()

        torch.save(M_d, path)

        print(f"Saved dist={d}, nnz={M_d._nnz()}")

    print("Done.")


def get_fixed_operator(
    data: PygGraph,
    X: torch.Tensor,
    operator_name: str,
) -> torch.Tensor:
    """
    Compute fixed operator * data F = S X,
    where S is one of a few predefined operators.

    Supported operators:
    - "L1": 1-hop adjacency
    - "L2": 2-hop adjacency
    - "H1": 1-hop high-pass (I - A)
    - "H2": 2-hop high-pass (I - A)^2
    """
    N = data.num_nodes
    edge_index = data.edge_index
    A = to_torch_coo_tensor(edge_index, size=(N, N)).float()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg[deg == 0] = 1.0
    row = A.indices()[0]
    values = A.values() / deg[row]
    A_rw = torch.sparse_coo_tensor(A.indices(), values, A.size())

    if operator_name == "X":
        F = X
    elif operator_name == "L1":
        F = A_rw @ X
    elif operator_name == "L2":
        F = A_rw @ (A_rw @ X)
    elif operator_name == "H1":  # (I-A)
        F = X - (A_rw @ X)
    elif operator_name == "H2":  # (I-A)^2
        F = (X - A_rw @ X) - (A_rw @ (X - A_rw @ X))
    else:
        raise ValueError(f"Unknown fixed operator: {operator_name}")

    return F


def apply_lin_gaussian_operator_slow(
    all_pairs_dist: torch.Tensor,
    X: torch.Tensor,
    mu: float,
    sigma: float,
) -> torch.Tensor:
    """
    Compute F = S X for the linear Gaussian operator S defined by weights w(d) = exp(-((mu - d)^2) / (2 sigma^2)),
    where d is the shortest path distance between nodes.

    Slow version that does not use cached sparse matrices.
    """
    N, d = X.shape
    device = X.device

    F = torch.zeros(N, d, device=device)

    max_d = int(torch.max(all_pairs_dist).item())
    for dist in range(max_d + 1):
        w = torch.exp(-((torch.tensor([mu]) - dist) ** 2) / (2 * sigma**2))
        if w < 1e-2:
            continue

        mask = all_pairs_dist == dist  # (N, N)
        F += w * (mask.float() @ X)  # (N, d)

    return F


def apply_lin_gaussian_operator(
    X: torch.Tensor,
    mu: float,
    sigma: float,
    cache_dir: Path,
    weight_cutoff: float = 1e-2,
) -> torch.Tensor:
    """
    Compute F = S X for the linear Gaussian operator S defined by weights w(d) = exp(-((mu - d)^2) / (2 sigma^2)),
    where d is the shortest path distance between nodes.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    X = X.to(device)

    N, d = X.shape

    F = torch.zeros_like(X, device=device)

    # Precompute Gaussian weights (cheap)
    # We only load matrices that exist on disk
    for fname in os.listdir(cache_dir):
        if not fname.startswith("M_dist_"):
            continue

        dist = int(fname.split("_")[-1].split(".")[0])
        w = np.exp(-((mu - dist) ** 2) / (2 * sigma**2))

        if w < weight_cutoff:
            continue

        M_d = torch.load(
            os.path.join(cache_dir, fname),
            map_location=device,
        )

        # Sparse @ dense
        F += w * (M_d @ X)

    return F.cpu()


def apply_lin_heat_operator(
    tau: float,
    L_sym: torch.Tensor,
    X: torch.Tensor,
    eigvals: torch.Tensor | None = None,
    eigvecs: torch.Tensor | None = None,
    *,
    taylor_k: int = 5,
) -> torch.Tensor:
    """
    Compute heat kernel operator * data F = SX = exp(-tau * L_rw).X,
    where L_rw = I - D^{-1}A is the random-walk Laplacian.

    Strategy:
    - Exact eigendecomposition if N <= N_exact
    - Truncated Taylor expansion otherwise

    Stick to F to avoid NxN blowup.

    Returns:
        F : (N, d) torch.Tensor
    """
    N = L_sym.shape[0]
    X = X.to(L_sym.device)
    if torch.cuda.is_available() and not L_sym.is_cuda:
        raise ValueError("LinHeat on CUDA not working properly")

    # Exact heat kernel for small graphs
    if (eigvals is not None) and (eigvecs is not None):

        if tau == 0.0:
            return X.cpu()

        exp_eigvals = torch.exp(-tau * eigvals)

        # F = V exp(-tau Λ) V^{-1} X
        F = eigvecs @ (exp_eigvals[:, None] * (eigvecs.T @ X))
        return F.cpu()

    # Approximate heat kernel for large graphs
    # using truncated Taylor expansion of exp(-tau L):
    # exp(-tau L) ≈ sum_{k=0}^K (-tau L)^k / k!
    F = X.clone()
    term = X.clone()  # (-tau L)^k X

    for k in range(1, taylor_k + 1):
        term = (-tau / k) * (L_sym @ term)
        F = F + term
    return F.cpu()


def build_softhopsign_dataset(
    N: int = 1000,
    radius: float = 0.1,
    k: float = 0.0,
    label_noise: float = 0.5,
    cache_dir: Path = Path("data_cache/goblin_khopsign"),
    topology_seed: int = 0,
):
    """Build a soft k-HopSign dataset on a random geometric graph."""
    saved_dataset_path = (
        cache_dir
        / f"{k}HopSign_N={N}_r={radius}_ln={label_noise}_seed={topology_seed}.pt"
    )

    if saved_dataset_path.exists() and "G" in torch.load(saved_dataset_path):
        print(f"Loading cached dataset from {saved_dataset_path}...")
        dataset = torch.load(saved_dataset_path)
        print("Loaded cached dataset.")
        return dataset

    # Generate random geometric graph
    G = nx.random_geometric_graph(N, radius, seed=topology_seed)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    N = G.number_of_nodes()
    G = nx.convert_node_labels_to_integers(G)

    # Use k as seed for label features
    torch.manual_seed(int(k * 1000))
    X = torch.randn(N, 1)
    all_pairs_dist: torch.Tensor = apspd_to_tensor(G)

    F = apply_lin_gaussian_operator_slow(all_pairs_dist, X, mu=k, sigma=label_noise)

    y = torch.sign((F)).squeeze()
    y_class = (y > 0).long()  # maps -1 -> 0, +1 -> 1, [N]
    y_onehot = torch.nn.functional.one_hot(y_class, num_classes=2)  # [N, 2]

    # train_fit : train_eval : val : test = 1:1:1:1 balanced splits
    torch.manual_seed(topology_seed)
    perm = torch.randperm(N)
    q = N // 4
    splits = {
        "train_fit": perm[:q],
        "train_eval": perm[q : 2 * q],
        "val": perm[2 * q : 3 * q],
        "test": perm[3 * q :],
    }

    dataset = {
        "data": from_networkx(G),
        "X": X,
        "all_pairs_dist": all_pairs_dist,
        "y_class": y_class,
        "y_onehot": y_onehot,
        "splits": splits,
        "G": G,
    }

    saved_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, saved_dataset_path)
    print(f"Saved dataset to {saved_dataset_path}.")

    return dataset


def build_softhopsign_dataset_wrapper(
    N: int = 1000,
    radius: float = 0.1,
    k: float = 0.0,
    label_noise: float = 0.5,
    cache_dir: Path = Path("data_cache/goblin_khopsign"),
    topology_seed: int = 0,
):

    dataset = build_softhopsign_dataset(
        N=N,
        radius=radius,
        k=k,
        label_noise=label_noise,
        cache_dir=cache_dir,
        topology_seed=topology_seed,
    )

    N = dataset["data"].num_nodes
    graph = dgl.from_networkx(dataset["G"])
    graph = dgl.to_simple(graph)
    graph = dgl.to_bidirected(graph)

    labels = dataset["y_class"]
    num_classes = 2
    node_features = dataset["X"]
    train_idxs = torch.cat(
        (dataset["splits"]["train_fit"], dataset["splits"]["train_eval"])
    )
    val_idxs = dataset["splits"]["val"]
    test_idxs = dataset["splits"]["test"]
    train_mask = torch.zeros(N).long()
    val_mask = torch.zeros(N).long()
    test_mask = torch.zeros(N).long()
    train_mask[train_idxs] = 1
    val_mask[val_idxs] = 1
    test_mask[test_idxs] = 1

    return (
        graph,
        labels,
        num_classes,
        node_features,
        train_mask,
        val_mask,
        test_mask,
    )


# -----------------------------
# Loading PyG benchmarks and prepping them for GOBLIN
# -----------------------------


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(
        1, labels.view(-1, 1), 1
    )


from sklearn.model_selection import train_test_split


def graphany_style_split(
    y_class: torch.Tensor,
    *,
    train_nodes_per_class: int = 20,
    seed: int = 0,
):
    """
    GraphAny-style semi-supervised split:
      - total training nodes = train_nodes_per_class * C
      - stratified by class proportions
      - no per-class minimum enforced
    """
    y_np = y_class.cpu().numpy()
    N = len(y_np)
    C = int(y_np.max() + 1)

    num_train_nodes = train_nodes_per_class * C
    num_train_nodes = min(num_train_nodes, N)

    label_idx = np.arange(N)
    test_rate = (N - num_train_nodes) / N

    train_idx, test_and_val_idx = train_test_split(
        label_idx,
        test_size=test_rate,
        random_state=seed,
        shuffle=True,
        stratify=y_np,
    )

    val_idx, test_idx = train_test_split(
        test_and_val_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=y_np[test_and_val_idx],
    )

    # Split train into train_fit / train_eval
    half = len(train_idx) // 2

    return {
        "train_fit": torch.tensor(train_idx[:half], dtype=torch.long),
        "train_eval": torch.tensor(train_idx[half:], dtype=torch.long),
        "val": torch.tensor(val_idx, dtype=torch.long),
        "test": torch.tensor(test_idx, dtype=torch.long),
    }


def extract_mask(mask: torch.Tensor, split_idx: int = 0) -> torch.Tensor:
    """
    Handles both 1D masks [N] and 2D masks [N, K] (e.g. WikiCS).
    """
    if mask.dim() == 2:
        mask = mask[:, split_idx]
    return mask.nonzero(as_tuple=False).reshape(-1)


# --------- Main loader ---------------


def load_graph_dataset(
    name: str,
    root: Path,
    seed: int = 0,
    compute_all_pairs_dist: bool = False,
) -> tuple[
    PygGraph,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    int,
]:
    """
    Universal dataset loader for GOBLIN.

    Returns:
        X              : (N, d) float tensor
        all_pairs_dist : (N, N) tensor (or None)
        y_class        : (N,) long tensor
        y_onehot       : (N, C) float tensor
        splits         : dict[str, LongTensor]
        C              : number of classes
        data           : PyG Data object
    """

    # Dataset dispatch (PyG only for now)
    if name in {"Cora", "Citeseer", "Pubmed"}:
        dataset = Planetoid(root=str(root), name=name)
        data = dataset[0]

    elif name in {"FCora", "DBLP"}:
        dataset = CitationFull(root=str(root), name=name.replace("F", ""))
        data = dataset[0]

    elif name in {"CoCS", "CoPhysics"}:
        dataset = Coauthor(root=str(root), name=name.replace("Co", ""))
        data = dataset[0]

    elif name in {"AmzComp", "AmzPhoto"}:
        dataset = Amazon(
            root=str(root), name="Computers" if "Comp" in name else "Photo"
        )
        data = dataset[0]

    elif name in {"Chameleon", "Squirrel"}:
        dataset = WikipediaNetwork(root=str(root), name=name.lower())
        data = dataset[0]

    elif name == "Actor":
        dataset = Actor(root=f"{root}/actor")
        data = dataset[0]

    elif name in {"Roman", "AmzRatings", "Minesweeper", "Tolokers", "Questions"}:
        mapping = {
            "Roman": "Roman-empire",
            "AmzRatings": "Amazon-ratings",
        }
        dataset = HeterophilousGraphDataset(
            root=str(root), name=mapping.get(name, name)
        )
        data = dataset[0]

    elif name in {"Wiki", "BlogCatalog"}:
        dataset = AttributedGraphDataset(root=str(root), name=name)
        data = dataset[0]

    elif name == "WkCS":
        dataset = WikiCS(root=str(root))
        data = dataset[0]

    elif name == "Reddit":
        dataset = Reddit(root=str(root))
        data = dataset[0]

    elif name in {"AirBrazil", "AirUS", "AirEU"}:
        region = (
            name.replace("Air", "").replace("US", "USA").replace("EU", "Europe").lower()
        )
        dataset = Airports(root=str(root), name=region)
        data = dataset[0]

    elif name in {"Wisconsin", "Texas", "Cornell"}:
        dataset = WebKB(root=str(root), name=name)
        data = dataset[0]

    # # # Broken; 404 error
    # elif name == "Deezer":
    #     dataset = DeezerEurope(root=str(root))
    #     data = dataset[0]

    # # Broken; 404 error
    # elif name == "LastFMAsia":
    #     dataset = LastFMAsia(root=str(root))
    #     data = dataset[0]

    elif name == "Arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root / "Arxiv")
        data = dataset[0]
        compute_all_pairs_dist = False  # too big
        print("Warning: all_pairs_dist computation disabled for Arxiv due to size.")

    else:
        raise ValueError(f"Unknown or unsupported dataset: {name}")

    # -----------------------------
    # Graph + features
    # -----------------------------
    X = data.x.float()

    # -----------------------------
    # Labels
    # -----------------------------
    y_class = data.y.long()
    C = int(y_class.max().item() + 1)
    y_onehot = one_hot(y_class, C)

    # -----------------------------
    # Splits
    # -----------------------------
    if hasattr(data, "train_mask"):
        split_idx = 0  # GraphAny default
        train_idx = extract_mask(data.train_mask, split_idx)
        val_idx = extract_mask(data.val_mask, split_idx)
        test_idx = extract_mask(data.test_mask, split_idx)

        # Split train into train_fit / train_eval
        n_train = train_idx.numel()
        train_fit = train_idx[: n_train // 2]
        train_eval = train_idx[n_train // 2 :]

        splits = {
            "train_fit": train_fit,
            "train_eval": train_eval,
            "val": val_idx,
            "test": test_idx,
        }
    else:
        splits = graphany_style_split(y_class, seed=seed)

    # -----------------------------
    # APSP distances
    # -----------------------------
    if compute_all_pairs_dist:
        N = X.shape[0]
        apspd_filepath = Path(f"data_cache/{name}_apspd.pt")
        if apspd_filepath.exists():
            print(f"Loading cached all pairs shortest path distances for {name}...")
            all_pairs_dist = torch.load(str(apspd_filepath))["spd"]
            if all_pairs_dist.shape[0] == N**2:
                all_pairs_dist = all_pairs_dist.reshape(N, N)
            assert all_pairs_dist.shape == (N, N)
            print(f"Loaded cached all pairs shortest path distances for {name}.")
        else:
            print(f"Computing all pairs shortest path distances for {name}...")
            all_pairs_dist = apspd_to_tensor(data)
    else:
        all_pairs_dist = None

    return data, X, all_pairs_dist, y_class, y_onehot, splits, C
