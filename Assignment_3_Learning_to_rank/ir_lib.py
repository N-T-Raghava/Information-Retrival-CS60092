import io, os, re, math, tarfile
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

FIELDS = ["title", "author", "bib", "text"]          # the four zones of a document
STREAMS = FIELDS + ["all"]                           # + whole document
FEATURE_BASE = [
    "query_term_number", "query_term_ratio", "stream_length", "idf",
    "tf_sum", "tf_min", "tf_max", "tf_mean", "tf_var",
    "ntf_sum", "ntf_min", "ntf_max", "ntf_mean", "ntf_var",
    "tfidf_sum", "tfidf_min", "tfidf_max", "tfidf_mean", "tfidf_var",
    "boolean", "vsm", "bm25",
]
FEATURE_NAMES = [f"{s}:{f}" for s in STREAMS for f in FEATURE_BASE]
NF = len(FEATURE_BASE)

_MARK = re.compile(r"^\.([ITABWXK])(?:\s+(.*))?$")


def _parse(text):
    recs, cur, field = [], None, None
    for line in text.splitlines():
        m = _MARK.match(line.rstrip("\r\n"))
        if m:
            tag, rest = m.group(1), (m.group(2) or "").strip()
            if tag == "I":
                cur = {"id": rest, "T": [], "A": [], "B": [], "W": []}
                recs.append(cur); field = None
            elif cur is not None:
                field = tag
                cur.setdefault(field, [])
                if rest: cur[field].append(rest)
        elif cur is not None and field:
            cur[field].append(line.rstrip("\r\n"))
    return recs


def _read_files(path):
    """Return {basename: text} for cran.all.1400, cran.qry, cranqrel from dir or tar.gz."""
    want = ("cran.all.1400", "cran.qry", "cranqrel")
    out = {}
    if os.path.isdir(path):
        for w in want:
            p = os.path.join(path, w)
            if os.path.exists(p):
                out[w] = open(p, encoding="latin-1").read()
    else:
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                b = os.path.basename(m.name)
                if b in want and m.isfile():
                    out[b] = tf.extractfile(m).read().decode("latin-1")
    return out


def load_docs(path):
    files = _read_files(path)
    docs = {}
    for r in _parse(files["cran.all.1400"]):
        docs[int(r["id"])] = {"title": " ".join(r["T"]), "author": " ".join(r["A"]),
                              "bib": " ".join(r["B"]), "text": " ".join(r["W"])}
    return docs


def load_queries(source_text):
    """Return list of (orig_id:str, text). Works on any file in cran.qry format."""
    return [(r["id"], " ".join(r["W"])) for r in _parse(source_text)]


def load_collection(path, relevant_levels=(1, 2, 3, 4)):
    """docs, queries, qrels. Queries are numbered 1..n in file order -- this is the
    numbering used by cranqrel (verified empirically, see report)."""
    files = _read_files(path)
    docs = load_docs(path)
    queries = load_queries(files["cran.qry"])
    qrels = {}
    for line in files["cranqrel"].splitlines():
        p = line.split()
        if len(p) < 3: continue
        q, d, r = int(p[0]), int(p[1]), int(p[2])
        if r in relevant_levels:
            qrels.setdefault(q, set()).add(d)
    return docs, queries, qrels


_STOP = set(ENGLISH_STOP_WORDS)
_ps = PorterStemmer()
_cache = {}


def _stem(t):
    s = _cache.get(t)
    if s is None:
        s = _cache[t] = _ps.stem(t)
    return s


def tokenize(text):
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOP or (len(t) == 1 and t.isalpha()):
            continue
        out.append(_stem(t))
    return out


class Index:
    K1, B = 1.2, 0.75

    def __init__(self, docs):
        self.doc_ids = sorted(docs)
        self.N = N = len(self.doc_ids)
        toks = {f: [tokenize(docs[d][f]) for d in self.doc_ids] for f in FIELDS}
        toks["all"] = [sum((toks[f][i] for f in FIELDS), []) for i in range(N)]
        vocab = {}
        for ts in toks["all"]:
            for t in ts:
                if t not in vocab: vocab[t] = len(vocab)
        self.vocab, V = vocab, len(vocab)
        self.tf, self.len, self.avglen = {}, {}, {}
        self.idf, self.bidf, self.dnorm = {}, {}, {}
        for s in STREAMS:
            rows, cols, vals = [], [], []
            for i, ts in enumerate(toks[s]):
                for t, c in Counter(ts).items():
                    rows.append(i); cols.append(vocab[t]); vals.append(c)
            M = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(N, V))
            self.tf[s] = M.tocsc()
            self.len[s] = np.array([len(t) for t in toks[s]], np.float32)
            self.avglen[s] = max(float(self.len[s].mean()), 1e-9)
            df = np.asarray((M > 0).sum(0)).ravel().astype(np.float64)
            self.idf[s] = np.log((N + 1.0) / (df + 0.5)).astype(np.float32)
            self.bidf[s] = np.log(1.0 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
            W = M.copy(); W.data = (1.0 + np.log(W.data)) * self.idf[s][W.indices]
            self.dnorm[s] = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel()).astype(np.float32)
        self.oov_idf = float(np.log((N + 1.0) / 0.5))
        self.oov_bidf = float(np.log(1.0 + (N + 0.5) / 0.5))

    # ---- features -----------------------------------------------------------------
    def features(self, query_text):
        """(N, 110) float32 feature matrix for one query against all documents."""
        qtoks = tokenize(query_text)
        qtf = Counter(qtoks)
        terms = list(qtf)
        m, N = len(terms), self.N
        X = np.zeros((N, len(STREAMS) * NF), np.float32)
        if m == 0:
            return X
        cols = [self.vocab.get(t, -1) for t in terms]
        valid = [j for j, c in enumerate(cols) if c >= 0]
        qc = np.array([qtf[t] for t in terms], np.float32)
        for si, s in enumerate(STREAMS):
            M = np.zeros((N, m), np.float32)
            if valid:
                M[:, valid] = self.tf[s][:, [cols[j] for j in valid]].toarray()
            idf = np.full(m, self.oov_idf, np.float32); bidf = np.full(m, self.oov_bidf, np.float32)
            for j in valid:
                idf[j] = self.idf[s][cols[j]]; bidf[j] = self.bidf[s][cols[j]]
            L = self.len[s]
            cov = M > 0
            qtn = cov.sum(1).astype(np.float32)
            ntf = M / np.maximum(L, 1.0)[:, None]
            tfidf = M * idf
            wq = (1.0 + np.log(qc)) * idf
            wd = np.where(cov, (1.0 + np.log(np.maximum(M, 1.0))) * idf, 0.0)
            vsm = (wd @ wq) / (self.dnorm[s] * np.linalg.norm(wq) + 1e-9)
            norm = (self.K1 * (1 - self.B + self.B * L / self.avglen[s]))[:, None]
            tfpart = M * (self.K1 + 1) / (M + norm)
            bm25 = (tfpart * bidf).sum(1)
            cols_out = [qtn, qtn / m, L, (cov * idf).sum(1)]
            for A in (M, ntf, tfidf):
                cols_out += [A.sum(1), A.min(1), A.max(1), A.mean(1), A.var(1)]
            cols_out += [(qtn == m).astype(np.float32), vsm, bm25]
            X[:, si * NF:(si + 1) * NF] = np.stack(cols_out, 1)
        return X


def build_feature_tensor(index, query_texts):
    return np.stack([index.features(q) for q in query_texts]).astype(np.float32)   # (nq, N, 110)


def relevance_matrix(index, qrels, qids):
    pos = {d: i for i, d in enumerate(index.doc_ids)}
    R = np.zeros((len(qids), index.N), bool)
    for i, q in enumerate(qids):
        for d in qrels.get(q, ()):
            if d in pos: R[i, pos[d]] = True
    return R


def rank_docs(scores):
    """Descending sort, ties broken by document order (deterministic)."""
    return np.argsort(-scores, axis=1, kind="stable")


def metrics(scores, rel, ks=(1, 3, 5, 10, 20)):
    """Per-query metrics. scores/rel: (nq, N). Returns dict of arrays."""
    nq, N = scores.shape
    order = rank_docs(scores)
    r = np.take_along_axis(rel, order, 1).astype(np.float64)
    cum = np.cumsum(r, 1)
    nrel = np.maximum(rel.sum(1), 1)
    rk = np.arange(1, N + 1)
    out = {"AP": ((cum / rk) * r).sum(1) / nrel}
    disc = 1.0 / np.log2(rk + 1)
    dcg = (r[:, :10] * disc[:10]).sum(1)
    idcg = np.array([disc[:min(int(n), 10)].sum() for n in rel.sum(1)])
    out["nDCG@10"] = dcg / np.maximum(idcg, 1e-12)
    for k in ks:
        P = cum[:, k - 1] / k; Rc = cum[:, k - 1] / nrel
        out[f"P@{k}"], out[f"R@{k}"] = P, Rc
        out[f"F1@{k}"] = np.where(P + Rc > 0, 2 * P * Rc / np.maximum(P + Rc, 1e-12), 0.0)
    return out


def f1_curve(scores, rel, kmax=100):
    """Macro-F1 / P / R for every cutoff k = 1..kmax."""
    order = rank_docs(scores)
    r = np.take_along_axis(rel, order, 1)[:, :kmax].astype(float)
    cum = np.cumsum(r, 1); k = np.arange(1, kmax + 1)
    nrel = np.maximum(rel.sum(1), 1)[:, None]
    P, R = cum / k, cum / nrel
    F = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)
    return P.mean(0), R.mean(0), F.mean(0)


def best_k(scores, rel, kmax=100):
    _, _, F = f1_curve(scores, rel, kmax)
    return int(np.argmax(F)) + 1


def at_k(scores, rel, k):
    P, R, F = f1_curve(scores, rel, max(k, 1))
    return float(P[k - 1]), float(R[k - 1]), float(F[k - 1])


def interpolated_pr(scores, rel):
    """11-point interpolated precision averaged over queries."""
    order = rank_docs(scores)
    r = np.take_along_axis(rel, order, 1).astype(float)
    cum = np.cumsum(r, 1); rk = np.arange(1, r.shape[1] + 1)
    P = cum / rk; R = cum / np.maximum(rel.sum(1), 1)[:, None]
    pts = np.linspace(0, 1, 11); res = np.zeros((r.shape[0], 11))
    for i in range(r.shape[0]):
        for j, lv in enumerate(pts):
            m = R[i] >= lv - 1e-12
            res[i, j] = P[i][m].max() if m.any() else 0.0
    return pts, res.mean(0)
