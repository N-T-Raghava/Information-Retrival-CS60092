"""
Runs every experiment of the report and writes results/, models/, figures/.
    python run_experiments.py --data data            (or --data cran.tar.gz)
    python run_experiments.py --data data --cv       (adds 5-fold CV over all 225 queries)
"""
import argparse, json, os, pickle, time
import numpy as np
from scipy.stats import wilcoxon
from ir_lib import *
import method_a as MA
import method_b as MB

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--cv", action="store_true")
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--test_frac", type=float, default=0.2)
args = ap.parse_args()
for d in ("results", "models", "figures"): os.makedirs(d, exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ data + features
docs, queries, qrels = load_collection(args.data)
index = Index(docs)
qids = np.arange(1, len(queries) + 1)                       # cranqrel numbering
X = build_feature_tensor(index, [q for _, q in queries])
rel = relevance_matrix(index, qrels, qids)
S = MA.zone_matrix(X)
log("collection:", index.N, "docs,", len(qids), "queries,", int(rel.sum()), "relevant pairs, vocab", len(index.vocab))

rng = np.random.RandomState(args.seed); perm = rng.permutation(len(qids))
n_test = int(round(args.test_frac * len(qids)))
te, tr = np.sort(perm[:n_test]), np.sort(perm[n_test:])
log(f"split: {len(tr)} train / {len(te)} test queries")
R = {"n_docs": index.N, "n_queries": len(qids), "n_rel": int(rel.sum()), "vocab": len(index.vocab),
     "n_train": len(tr), "n_test": len(te), "test_qids": (te + 1).tolist(),
     "mean_rel_per_query": float(rel.sum(1).mean())}
cand = MB.candidate_mask(X)
R["candidate_fraction"] = float(cand.mean()); R["candidate_recall_ceiling"] = float((cand & rel).sum() / rel.sum())

# zone statistics (for discussion)
zs = {}
for i, f in enumerate(FIELDS):
    m = S[:, :, i] > 0
    zs[f] = {"match_rate_relevant": float(m[rel].mean()), "match_rate_nonrelevant": float(m[~rel].mean()),
             "mean_score_relevant": float(S[:, :, i][rel].mean()), "mean_score_nonrelevant": float(S[:, :, i][~rel].mean())}
R["zone_stats"] = zs

# ------------------------------------------------------------------ helpers
import hashlib
os.makedirs("results/cache", exist_ok=True)
def cached(key, fn):
    """Resumable runs: expensive steps are stored on disk (delete results/cache to recompute)."""
    p = f"results/cache/{hashlib.md5(key.encode()).hexdigest()}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    v = fn(); pickle.dump(v, open(p, "wb")); return v

def summarize(scores, rel_, kstar):
    m = metrics(scores, rel_)
    d = {k: float(v.mean()) for k, v in m.items()}
    P, Rc, F = at_k(scores, rel_, kstar)
    d.update({"k*": kstar, "P@k*": P, "R@k*": Rc, "F1@k*": F})
    return d, m["AP"]

def col(name): return X[:, :, FEATURE_NAMES.index(name)]
def oof_scores(Xs, rels, mode, C, feats, folds=5, seed=0):
    out = np.zeros(rels.shape); parts = np.array_split(np.random.RandomState(seed).permutation(len(Xs)), folds)
    for k in range(folds):
        t_ = parts[k]; r_ = np.concatenate([parts[j] for j in range(folds) if j != k])
        out[t_] = MB.SVMRanker(mode, C, feats).fit(Xs[r_], rels[r_]).score(Xs[t_])
    return out
def write_run(path, qs, sc, tag, topn=100):
    order = rank_docs(sc)
    with open(path, "w") as f:
        for i, q in enumerate(qs):
            for r, j in enumerate(order[i, :topn]):
                f.write(f"{q} Q0 {index.doc_ids[j]} {r+1} {sc[i,j]:.6f} {tag}\n")
def write_relevant(path, qs, sc, k):
    order = rank_docs(sc)
    with open(path, "w") as f:
        for i, q in enumerate(qs):
            for j in order[i, :k]: f.write(f"{q} {index.doc_ids[j]}\n")

SYS, AP_Q, TEST_SCORES = {}, {}, {}
def evaluate_system(name, test_scores, kstar):
    d, apq = summarize(test_scores, rel[te], kstar)
    SYS[name] = d; AP_Q[name] = apq.tolist(); TEST_SCORES[name] = test_scores
    log(f"{name:34s} MAP={d['AP']:.4f} nDCG@10={d['nDCG@10']:.4f} P@10={d['P@10']:.3f} R@10={d['R@10']:.3f} "
        f"F1@10={d['F1@10']:.3f} | k*={kstar} F1@k*={d['F1@k*']:.3f}")

# ------------------------------------------------------------------ baselines (no learning)
for name, feat in [("BM25 (text zone)", "text:bm25"), ("BM25 (whole doc)", "all:bm25"), ("TF-IDF cosine (whole doc)", "all:vsm")]:
    evaluate_system(name, col(feat)[te], best_k(col(feat)[tr], rel[tr]))
evaluate_system("Zones, equal weights (0.25 each)", MA.score(S[te], [.25] * 4), best_k(MA.score(S[tr], [.25] * 4), rel[tr]))
for i, f in enumerate(FIELDS):                       # single zone only
    w = np.eye(4)[i]; evaluate_system(f"Single zone: {f}", MA.score(S[te], w), best_k(MA.score(S[tr], w), rel[tr]))

# ------------------------------------------------------------------ Method A
R["method_A"] = {}
for obj in ("map", "sse", "f1@10"):
    t = time.time()
    w, v, trace = cached(f'A{obj}{args.seed}', lambda: MA.learn_weights(S[tr], rel[tr], obj, step=0.05, refine=True))
    top = sorted(trace, key=lambda z: -z[1])[:5]
    R["method_A"][obj] = {"weights": dict(zip(FIELDS, np.round(w, 3).tolist())), "train_value": v,
                          "n_grid_points": len(trace), "seconds": round(time.time() - t, 1),
                          "top5": [[np.round(a, 3).tolist(), b] for a, b in top]}
    name = f"Method A (grid, obj={obj})"
    evaluate_system(name, MA.score(S[te], w), best_k(MA.score(S[tr], w), rel[tr]))
    if obj == "map": wA, trace_map = w, trace
A_NAME = "Method A (grid, obj=map)"

# ------------------------------------------------------------------ Method B
R["method_B"] = {}
Cs = [0.001, 0.01, 0.1, 1.0]
cfgs = [("Method B (SVM, all 110 feat.)", "pointwise", "all"),
        ("Method B (SVM, text-zone feat. only)", "pointwise", "text"),
        ("Method B (SVM, no author/bib feat.)", "pointwise", "no_author_bib"),
        ("Method B+ (RankSVM, all 110 feat.)", "ranksvm", "all")]
chosenC = {}
models = {}
for name, mode, feats in cfgs:
    t = time.time()
    if mode == "pointwise" and feats != "all": C, cvres = chosenC["pointwise"], None
    else: C, cvres = cached(f'C{mode}{feats}{args.seed}', lambda: MB.cv_select_C(X[tr], rel[tr], Cs, mode, feats))
    chosenC.setdefault(mode, C)
    m = MB.SVMRanker(mode, C, feats).fit(X[tr], rel[tr]); models[name] = m
    oof = cached(f'oof{mode}{feats}{C}{args.seed}', lambda: oof_scores(X[tr], rel[tr], mode, C, feats))
    kstar = best_k(oof, rel[tr])
    R["method_B"][name] = {"mode": mode, "features": feats, "C": C, "cv_map": cvres, "seconds": round(time.time() - t, 1),
                           "top_weights": [[a, float(b)] for a, b in m.weights_table()[:20]]}
    evaluate_system(name, m.score(X[te]), kstar)
B_NAME, BR_NAME = cfgs[0][0], cfgs[3][0]

# ------------------------------------------------------------------ significance (paired Wilcoxon on per-query AP)
def wil(a, b):
    a, b = np.array(AP_Q[a]), np.array(AP_Q[b])
    try: return float(wilcoxon(a, b).pvalue)
    except Exception: return None
R["wilcoxon_AP"] = {f"{x} vs {y}": wil(x, y) for x, y in [
    (B_NAME, A_NAME), (BR_NAME, A_NAME), (BR_NAME, B_NAME), (B_NAME, "BM25 (whole doc)"),
    (BR_NAME, "BM25 (whole doc)"), (A_NAME, "BM25 (whole doc)"), (A_NAME, "Single zone: text")]}

# ------------------------------------------------------------------ curves for figures
curves = {}
for name in (A_NAME, B_NAME, BR_NAME, "BM25 (whole doc)", "TF-IDF cosine (whole doc)"):
    pts, p11 = interpolated_pr(TEST_SCORES[name], rel[te])
    P, Rc, F = f1_curve(TEST_SCORES[name], rel[te], 40)
    curves[name] = {"recall_pts": pts.tolist(), "p11": p11.tolist(), "P_at_k": P.tolist(), "R_at_k": Rc.tolist(), "F_at_k": F.tolist()}
R["curves"] = curves
R["systems"] = SYS; R["per_query_AP"] = AP_Q

# ------------------------------------------------------------------ test-query result files
tq = (te + 1).tolist()
write_run("results/methodA_test_results.txt", tq, TEST_SCORES[A_NAME], "methodA_zone")
write_run("results/methodB_test_results.txt", tq, TEST_SCORES[B_NAME], "methodB_svm")
write_run("results/methodB_ranksvm_test_results.txt", tq, TEST_SCORES[BR_NAME], "methodB_ranksvm")
write_relevant("results/methodA_test_predicted_relevant.txt", tq, TEST_SCORES[A_NAME], SYS[A_NAME]["k*"])
write_relevant("results/methodB_test_predicted_relevant.txt", tq, TEST_SCORES[B_NAME], SYS[B_NAME]["k*"])
write_relevant("results/methodB_ranksvm_test_predicted_relevant.txt", tq, TEST_SCORES[BR_NAME], SYS[BR_NAME]["k*"])
json.dump(R, open("results/results.json", "w"), indent=1)
log("holdout stage done, results.json written")

# ------------------------------------------------------------------ 5-fold CV over all 225 queries
if args.cv:
    folds = np.array_split(np.random.RandomState(1).permutation(len(qids)), 5)
    cv = {n: [] for n in ("BM25 (whole doc)", "Method A", "Method B", "Method B+ (RankSVM)")}
    for k, f in enumerate(folds):
        trn = np.sort(np.concatenate([folds[j] for j in range(5) if j != k])); f = np.sort(f)
        bm = col("all:bm25"); ks = best_k(bm[trn], rel[trn])
        cv["BM25 (whole doc)"].append(summarize(bm[f], rel[f], ks)[0])
        w, _, _ = cached(f'cvA{k}', lambda: MA.learn_weights(S[trn], rel[trn], "map", 0.05, True))
        cv["Method A"].append(summarize(MA.score(S[f], w), rel[f], best_k(MA.score(S[trn], w), rel[trn]))[0] | {"weights": w.tolist()})
        for nm, mode in (("Method B", "pointwise"), ("Method B+ (RankSVM)", "ranksvm")):
            C = chosenC[mode]; m = MB.SVMRanker(mode, C, "all").fit(X[trn], rel[trn])
            ks = best_k(cached(f'cvoof{k}{mode}{C}', lambda: oof_scores(X[trn], rel[trn], mode, C, "all")), rel[trn])
            cv[nm].append(summarize(m.score(X[f]), rel[f], ks)[0])
        log(f"CV fold {k+1}/5 done")
    R["cv5"] = {n: {met: [float(np.mean([d[met] for d in v])), float(np.std([d[met] for d in v]))]
                    for met in ("AP", "nDCG@10", "P@10", "R@10", "F1@10", "P@k*", "R@k*", "F1@k*")} for n, v in cv.items()}
    R["cv5_weightsA"] = [d["weights"] for d in cv["Method A"]]
    json.dump(R, open("results/results.json", "w"), indent=1)

# ------------------------------------------------------------------ final models on ALL queries (used by predict.py)
allq = np.arange(len(qids))
wA_all, _, _ = cached('Aall', lambda: MA.learn_weights(S, rel, "map", 0.05, True))
kA = best_k(MA.score(S, wA_all), rel)
json.dump({"weights": dict(zip(FIELDS, wA_all.tolist())), "k_star": kA}, open("models/method_A.json", "w"), indent=1)
final = {}
for key, mode in (("method_B", "pointwise"), ("method_B_ranksvm", "ranksvm")):
    C = chosenC[mode]; m = MB.SVMRanker(mode, C, "all").fit(X, rel)
    ks = best_k(cached(f'oofall{mode}{C}', lambda: oof_scores(X, rel, mode, C, "all")), rel)
    pickle.dump({"model": m, "k_star": ks}, open(f"models/{key}.pkl", "wb")); final[key] = {"C": C, "k_star": ks}
R["final_models"] = {"method_A": {"weights": dict(zip(FIELDS, np.round(wA_all, 3).tolist())), "k_star": kA}, **final}
json.dump(R, open("results/results.json", "w"), indent=1)
log("final models saved; all done")
