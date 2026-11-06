"""
Method B -- real-valued weights learned with a linear SVM.

Input  : 22 features x 5 streams (title, author, bib, text, whole document) = 110 features per
         (query, document) pair, see ir_lib.FEATURE_NAMES.
Output : relevance judgement (1 = judged relevant, 0 otherwise).
Learner: sklearn LinearSVC (liblinear, squared-hinge, L2) -- the learned weight vector w gives the
         final score  w . x(q,d) + b, which is used to rank the documents of each query.

Two variants are implemented:
  * pointwise SVM  : classic binary linear SVM on (q,d) pairs, class_weight='balanced'
  * RankSVM        : pairwise SVM on difference vectors x(q,d+) - x(q,d-) (Joachims 2002)
Training pairs are restricted to *candidate* documents (documents sharing at least one query term
with the query in the whole-document stream); non-candidates are always ranked last.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.svm import LinearSVC
warnings.filterwarnings('ignore')   # liblinear convergence warnings (iteration cap is deliberate)
from ir_lib import FEATURE_NAMES, STREAMS, NF, metrics

_CAND = FEATURE_NAMES.index("all:query_term_number")
NEG = -1e9


def candidate_mask(X):
    return X[:, :, _CAND] > 0


def feature_subset(kind):
    if kind == "all": return np.arange(len(FEATURE_NAMES))
    if kind == "text": return np.array([i for i, n in enumerate(FEATURE_NAMES) if n.startswith("text:")])
    if kind == "no_author_bib":
        return np.array([i for i, n in enumerate(FEATURE_NAMES) if n.split(":")[0] in ("title", "text", "all")])
    raise ValueError(kind)


class SVMRanker:
    def __init__(self, mode="pointwise", C=0.1, feats="all", n_neg=150, seed=0):
        self.mode, self.C, self.feat_kind, self.n_neg, self.seed = mode, C, feats, n_neg, seed
        self.fi = feature_subset(feats)

    def fit(self, X, rel):
        cand = candidate_mask(X)
        Xf = X[:, :, self.fi]
        flat = Xf[cand]
        self.scaler = StandardScaler().fit(flat)
        if self.mode == "pointwise":
            
            rng = np.random.RandomState(self.seed); Z, y = [], []
            for q in range(X.shape[0]):
                pos = np.where(rel[q] & cand[q])[0]; neg = np.where(~rel[q] & cand[q])[0]
                if len(neg) > self.n_neg: neg = rng.choice(neg, self.n_neg, replace=False)
                idx = np.concatenate([pos, neg])
                Z.append(Xf[q, idx]); y.append(np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)])
            Z, y = self.scaler.transform(np.vstack(Z)), np.concatenate(y)
            clf = LinearSVC(C=self.C, class_weight="balanced", loss="hinge", dual=True, max_iter=300)
            clf.fit(Z, y)
            self.w, self.b = clf.coef_.ravel(), float(clf.intercept_[0])

        elif self.mode == "ranksvm":
            rng = np.random.RandomState(self.seed); D, Y = [], []

            for q in range(X.shape[0]):
                pos = np.where(rel[q] & cand[q])[0]; neg = np.where(~rel[q] & cand[q])[0]
                if len(pos) == 0 or len(neg) == 0: continue
                neg = rng.choice(neg, min(self.n_neg, len(neg)), replace=False)
                Zp = self.scaler.transform(Xf[q, pos]); Zn = self.scaler.transform(Xf[q, neg])
                diff = (Zp[:, None, :] - Zn[None, :, :]).reshape(-1, Zp.shape[1])
                sign = rng.choice([-1, 1], len(diff))
                D.append(diff * sign[:, None]); Y.append(sign)

            D, Y = np.vstack(D), np.concatenate(Y)
            clf = LinearSVC(C=self.C, fit_intercept=False, loss="hinge", dual=True, max_iter=300).fit(D, Y)
            self.w, self.b = clf.coef_.ravel(), 0.0

        else:
            raise ValueError(self.mode)
        
        return self

    def score(self, X):
        nq, N, _ = X.shape
        Z = self.scaler.transform(X[:, :, self.fi].reshape(nq * N, -1)).reshape(nq, N, -1)
        s = Z @ self.w + self.b
        s[~candidate_mask(X)] = NEG
        return s

    def weights_table(self):
        names = [FEATURE_NAMES[i] for i in self.fi]
        return sorted(zip(names, self.w), key=lambda t: -abs(t[1]))


def cv_select_C(X, rel, Cs, mode="pointwise", feats="all", folds=5, seed=0):
    """Query-grouped K-fold CV on the *training* queries; criterion = MAP."""
    rng = np.random.RandomState(seed); perm = rng.permutation(X.shape[0])
    parts = np.array_split(perm, folds); res = {}
    for C in Cs:
        aps = []
        for k in range(folds):
            te = parts[k]; tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            m = SVMRanker(mode, C, feats).fit(X[tr], rel[tr])
            aps.append(metrics(m.score(X[te]), rel[te])["AP"].mean())
        res[C] = float(np.mean(aps))
    best = max(res, key=res.get)
    return best, res
