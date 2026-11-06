"""
Method A -- Boolean weighted zone scoring with weights learned by grid search.

For a query q and document d, the Boolean match score of zone i is
        s_i(q,d) = |{t in q : t occurs in zone i of d}| / |q|          (in [0,1])
(for a single-term query this is exactly the 0/1 Boolean match; for longer queries it is the
fraction of query terms matched).  The document score is  score(q,d) = sum_i g_i * s_i(q,d)
with g_i >= 0, sum_i g_i = 1 (Manning et al., sec. 6.1.2 / 15.1.3).  The g_i are learned by an
exhaustive grid search over the weight simplex (coarse step, followed by a fine local refinement),
using the human relevance judgements as the target.
"""
import itertools
import numpy as np
from ir_lib import FIELDS, FEATURE_NAMES, rank_docs

_QTR = [FEATURE_NAMES.index(f"{f}:query_term_ratio") for f in FIELDS]


def zone_matrix(X):
    """(nq, N, 4) Boolean zone-match scores taken from the feature tensor."""
    return X[:, :, _QTR]


def simplex_grid(step, n=4):
    m = int(round(1 / step))
    out = [c for c in itertools.product(range(m + 1), repeat=n - 1) if sum(c) <= m]
    return np.array([list(c) + [m - sum(c)] for c in out], float) / m


def _refine_grid(w, delta=0.05, step=0.01):
    k = int(round(delta / step)); base = np.round(w / step).astype(int); m = int(round(1 / step))
    out = []
    for d in itertools.product(range(-k, k + 1), repeat=3):
        c = base[:3] + np.array(d); last = m - c.sum()
        if (c >= 0).all() and last >= 0: out.append(list(c) + [last])
    return np.array(out, float) / m


def _ap_mean(scores, rel):
    order = rank_docs(scores)
    r = np.take_along_axis(rel, order, 1).astype(np.float64)
    cum = np.cumsum(r, 1); rk = np.arange(1, r.shape[1] + 1)
    return float((((cum / rk) * r).sum(1) / np.maximum(rel.sum(1), 1)).mean())


def objective_value(S, rel, w, objective):
    sc = S @ w
    if objective == "map":
        return _ap_mean(sc, rel)
    if objective == "sse":                      # textbook squared-error criterion (negated)
        return -float(((sc - rel) ** 2).mean())
    if objective == "f1@10":
        order = rank_docs(sc)
        hit = np.take_along_axis(rel, order, 1)[:, :10].sum(1)
        P, R = hit / 10, hit / np.maximum(rel.sum(1), 1)
        return float(np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0).mean())
    raise ValueError(objective)


def learn_weights(S, rel, objective="map", step=0.05, refine=True, log=None):
    """Grid search. Returns (best_weights, best_value, trace list of (w, value))."""
    grid = simplex_grid(step)
    vals = np.array([objective_value(S, rel, w, objective) for w in grid])
    trace = list(zip(grid.tolist(), vals.tolist()))
    best = grid[int(np.argmax(vals))]; bv = vals.max()
    if refine:
        g2 = _refine_grid(best)
        v2 = np.array([objective_value(S, rel, w, objective) for w in g2])
        trace += list(zip(g2.tolist(), v2.tolist()))
        if v2.max() > bv: best, bv = g2[int(np.argmax(v2))], v2.max()
    return best, float(bv), trace


def score(S, w):
    return S @ np.asarray(w, float)
