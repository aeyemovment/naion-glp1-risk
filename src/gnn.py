"""Graph neural network (GraphSAGE-mean, 2 layers) on a patient-pattern similarity graph.

Nodes  : unique covariate patterns (train + test + twin patterns), features = same encoding as the other models
Edges  : k-nearest neighbours (k=10, standardized features, symmetric) + same-study arm links (exposed <-> comparator)
Labels : count-weighted binomial loss on TRAINING counts only: -(E log p + (N-E) log(1-p)); test counts never touch training
Output : node risk p; scored on held-out real counts. Transductive in structure (features only), never in labels.
Attribution: integrated gradients on node features (graph analogue of Grad-CAM), count-weighted, grouped by feature.

research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
import numpy as np, pandas as pd, torch
from sklearn.neighbors import NearestNeighbors

torch.set_num_threads(1)


class SAGE(torch.nn.Module):
    def __init__(self, d, h=32, p_init=1e-3):
        super().__init__()
        self.l1s, self.l1n = torch.nn.Linear(d, h), torch.nn.Linear(d, h, bias=False)
        self.l2s, self.l2n = torch.nn.Linear(h, 1), torch.nn.Linear(h, 1, bias=False)
        self.drop = torch.nn.Dropout(0.1)
        with torch.no_grad(): self.l2s.bias.fill_(float(np.log(p_init / (1 - p_init))))

    def forward(self, x, A):
        h = torch.relu(self.l1s(x) + self.l1n(torch.sparse.mm(A, x)))
        h = self.drop(h)
        return (self.l2s(h) + self.l2n(torch.sparse.mm(A, h))).squeeze(-1)


def build_graph(F, study=None, k=10):
    """F: (n, d) standardized features. Returns row-normalized sparse adjacency (mean aggregation)."""
    n = len(F); k = min(k, n - 1)
    rows, cols = [], []
    if k > 0:
        nn_ = NearestNeighbors(n_neighbors=k + 1).fit(F); _, idx = nn_.kneighbors(F)
        for i in range(n):
            for j in idx[i, 1:]: rows += [i, j]; cols += [j, i]
    if study is not None:
        s = pd.Series(study)
        for _, g in s.groupby(s):
            ii = g.index.to_numpy()
            for a in ii:
                for b in ii:
                    if a != b: rows.append(a); cols.append(b)
    if not rows: rows, cols = list(range(n)), list(range(n))
    e = pd.DataFrame({"r": rows, "c": cols}).drop_duplicates()
    deg = e.groupby("r").size()
    v = 1.0 / deg.loc[e.r].to_numpy()
    return torch.sparse_coo_tensor(np.vstack([e.r, e.c]), torch.tensor(v, dtype=torch.float32), (n, n)).coalesce()


def collapse(df, feat_cols, X):
    """Collapse aggregated rows (features, y, n) to unique patterns with N and E."""
    key = pd.Series([tuple(r) for r in X.round(6).to_numpy()], index=X.index)
    g = pd.DataFrame({"key": key, "y": df.y.to_numpy(), "n": df.n.to_numpy()}, index=X.index)
    N = g.groupby("key").n.sum(); E = g[g.y == 1].groupby("key").n.sum().reindex(N.index, fill_value=0)
    return N, E


def fit_predict_gnn(Xtr, ytr, wtr, Xte, study_tr=None, study_te=None, epochs=300, seed=0, return_model=False):
    """Xtr/Xte: encoded feature DataFrames (aggregated rows); ytr/wtr training labels/counts.
    Returns predicted risk for each Xte row (and optionally model + graph pieces for attribution)."""
    torch.manual_seed(seed); np.random.seed(seed)
    cols = Xtr.columns
    allX = pd.concat([Xtr, Xte[cols]], ignore_index=True)
    keys = [tuple(r) for r in allX.round(6).to_numpy()]
    uniq = pd.Index(pd.unique(pd.Series(keys)))
    node = uniq.get_indexer(keys)
    U = np.array([list(k) for k in uniq], dtype=float)
    mu, sd = U.mean(0), U.std(0) + 1e-9; Z = (U - mu) / sd
    st = None
    if study_tr is not None:
        s_all = np.r_[np.asarray(study_tr, dtype=object), np.asarray(study_te, dtype=object)]
        st = pd.Series(s_all).groupby(node).first().reindex(range(len(U))).to_numpy()
    A = build_graph(Z, st)
    ntr = len(Xtr)
    N = np.zeros(len(U)); E = np.zeros(len(U))
    np.add.at(N, node[:ntr], wtr); np.add.at(E, node[:ntr], wtr * ytr)
    prev = max(E.sum() / max(N.sum(), 1), 1e-6)
    x = torch.tensor(Z, dtype=torch.float32); Nt = torch.tensor(N, dtype=torch.float32); Et = torch.tensor(E, dtype=torch.float32)
    scale = 1.0 / max(E.sum(), 1.0)                 # loss per event (keeps optimisation well-conditioned)
    m = SAGE(Z.shape[1], 32, prev); opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        z = m(x, A)
        loss = -(Et * torch.nn.functional.logsigmoid(z) + (Nt - Et) * torch.nn.functional.logsigmoid(-z)).sum() * scale
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): p_node = torch.sigmoid(m(x, A)).numpy()
    p_te = p_node[node[ntr:]]
    if return_model:
        return p_te, {"model": m, "x": x, "A": A, "node": node, "ntr": ntr, "cols": cols, "mu": mu, "sd": sd}
    return p_te


def integrated_gradients(g, w_te, steps=32):
    """Count-weighted |IG| per input feature for the test rows' nodes (baseline = feature mean = 0 in z-space)."""
    m, x, A, node, ntr = g["model"], g["x"], g["A"], g["node"], g["ntr"]
    te_nodes = node[ntr:]
    wn = np.zeros(len(x)); np.add.at(wn, te_nodes, w_te)
    tot = torch.zeros_like(x)
    for a in np.linspace(1 / steps, 1, steps):
        xa = (a * x).clone().requires_grad_(True)
        out = m(xa, A)
        out.sum().backward()
        tot += xa.grad
    ig = (x * tot / steps).detach().numpy()
    return pd.Series(np.average(np.abs(ig), axis=0, weights=np.maximum(wn, 0) + 1e-12), index=g["cols"])


# ---------------------------------------------------------------- tunable variants (Stage 6c)
def build_graph_w(F, study=None, k=10, edge_mode="uniform", study_w=1.0):
    """Weighted adjacency. edge_mode: 'uniform' | 'gauss0.5' | 'gauss1' | 'gauss2' (bandwidth = factor x median kNN distance).
    study_w: weight of same-study arm links relative to a unit kNN edge (0 = none). Row-normalised (weighted mean)."""
    n = len(F); k = min(k, n - 1); W = {}
    if k > 0:
        nn_ = NearestNeighbors(n_neighbors=k + 1).fit(F); dist, idx = nn_.kneighbors(F)
        med = np.median(dist[:, 1:]) + 1e-9
        for i in range(n):
            for d_, j in zip(dist[i, 1:], idx[i, 1:]):
                if edge_mode == "uniform": w = 1.0
                else: bw = float(edge_mode.replace("gauss", "")) * med; w = float(np.exp(-(d_ / bw) ** 2))
                for a, b in ((i, j), (j, i)): W[(a, b)] = max(W.get((a, b), 0.0), w)
    if study is not None and study_w > 0:
        s = pd.Series(study)
        for _, g in s.groupby(s):
            ii = g.index.to_numpy()
            for a in ii:
                for b in ii:
                    if a != b: W[(a, b)] = W.get((a, b), 0.0) + study_w
    if not W: W = {(i, i): 1.0 for i in range(n)}
    e = pd.DataFrame([(a, b, w) for (a, b), w in W.items()], columns=["r", "c", "w"])
    e = e[e.w > 1e-8]
    rs = e.groupby("r").w.sum(); e["v"] = e.w / rs.loc[e.r].to_numpy()
    return torch.sparse_coo_tensor(np.vstack([e.r, e.c]), torch.tensor(e.v.to_numpy(), dtype=torch.float32), (n, n)).coalesce()


class DeepSAGE(torch.nn.Module):
    """L-layer GraphSAGE-mean with residual connections (depth 1..L)."""
    def __init__(self, d, L=2, h=32, p_init=1e-3):
        super().__init__()
        self.inp = torch.nn.Linear(d, h)
        self.selfs = torch.nn.ModuleList([torch.nn.Linear(h, h) for _ in range(L)])
        self.neigh = torch.nn.ModuleList([torch.nn.Linear(h, h, bias=False) for _ in range(L)])
        self.out = torch.nn.Linear(h, 1); self.drop = torch.nn.Dropout(0.1)
        with torch.no_grad(): self.out.bias.fill_(float(np.log(p_init / (1 - p_init))))

    def forward(self, x, A):
        h = torch.relu(self.inp(x))
        for s, n in zip(self.selfs, self.neigh):
            h = h + self.drop(torch.relu(s(h) + n(torch.sparse.mm(A, h))))
        return self.out(h).squeeze(-1)


class GAT(torch.nn.Module):
    """L-layer single-head graph attention: edge weights are LEARNED (attention), structure from A's edges."""
    def __init__(self, d, L=2, h=32, p_init=1e-3):
        super().__init__()
        self.inp = torch.nn.Linear(d, h)
        self.W = torch.nn.ModuleList([torch.nn.Linear(h, h, bias=False) for _ in range(L)])
        self.a_src = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(h) * 0.1) for _ in range(L)])
        self.a_dst = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(h) * 0.1) for _ in range(L)])
        self.out = torch.nn.Linear(h, 1)
        with torch.no_grad(): self.out.bias.fill_(float(np.log(p_init / (1 - p_init))))
        self.last_attention = None

    def forward(self, x, A):
        r, c = A.indices(); n = x.shape[0]
        h = torch.relu(self.inp(x))
        for W, asrc, adst in zip(self.W, self.a_src, self.a_dst):
            z = W(h)
            e = torch.nn.functional.leaky_relu((z[r] * adst).sum(-1) + (z[c] * asrc).sum(-1), 0.2)
            emax = torch.full((n,), -1e9).scatter_reduce(0, r, e, reduce="amax", include_self=True)
            ex = torch.exp(e - emax[r]); den = torch.zeros(n).index_add(0, r, ex) + 1e-12
            alpha = ex / den[r]
            agg = torch.zeros_like(z).index_add(0, r, alpha.unsqueeze(-1) * z[c])
            h = h + torch.relu(agg)
            self.last_attention = alpha.detach()
        return self.out(h).squeeze(-1)


def fit_predict_gnn_cfg(Xtr, ytr, wtr, Xte, study_tr, study_te, arch="sage", L=2, k=10, edge_mode="uniform", study_w=1.0,
                        epochs=300, seed=0, return_train=False):
    torch.manual_seed(seed); np.random.seed(seed)
    cols = Xtr.columns
    allX = pd.concat([Xtr, Xte[cols]], ignore_index=True)
    keys = [tuple(r) for r in allX.round(6).to_numpy()]
    uniq = pd.Index(pd.unique(pd.Series(keys))); node = uniq.get_indexer(keys)
    U = np.array([list(kk) for kk in uniq], dtype=float); Z = (U - U.mean(0)) / (U.std(0) + 1e-9)
    s_all = np.r_[np.asarray(study_tr, dtype=object), np.asarray(study_te, dtype=object)]
    st = pd.Series(s_all).groupby(node).first().reindex(range(len(U))).to_numpy()
    A = build_graph_w(Z, st, k=k, edge_mode=edge_mode, study_w=study_w)
    ntr = len(Xtr); N = np.zeros(len(U)); E = np.zeros(len(U))
    np.add.at(N, node[:ntr], wtr); np.add.at(E, node[:ntr], wtr * ytr)
    prev = max(E.sum() / max(N.sum(), 1), 1e-6)
    x = torch.tensor(Z, dtype=torch.float32); Nt = torch.tensor(N, dtype=torch.float32); Et = torch.tensor(E, dtype=torch.float32)
    m = (GAT if arch == "gat" else DeepSAGE)(Z.shape[1], L, 32, prev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4); scale = 1.0 / max(E.sum(), 1.0)
    for _ in range(epochs):
        m.train(); opt.zero_grad(); z = m(x, A)
        loss = -(Et * torch.nn.functional.logsigmoid(z) + (Nt - Et) * torch.nn.functional.logsigmoid(-z)).sum() * scale
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): p = torch.sigmoid(m(x, A)).numpy()
    return (p[node[ntr:]], p[node[:ntr]]) if return_train else p[node[ntr:]]
