"""
Rank the Cranfield documents for NEW queries with a trained model.
    python predict.py --data data --queries new_queries.qry --method B --out results/new
    python predict.py --data cran.tar.gz --queries new_queries.qry --method A --id-mode orig

--data     directory (or cran.tar.gz) holding cran.all.1400 -- the documents to rank
--queries  file in cran.qry format (.I id / .W text)
--method   A       Boolean weighted zone scoring   (models/method_A.json)
           B       pointwise linear SVM            (models/method_B.pkl)
           B-rank  RankSVM                         (models/method_B_ranksvm.pkl)
--id-mode  seq  (default) queries are numbered 1..n in file order -- the convention of cranqrel
           orig use the .I identifiers found in the file (leading zeros removed)
Writes  <out>_results.txt            top-N ranking, TREC format: qid Q0 docid rank score tag
        <out>_predicted_relevant.txt "qid docid" for the top k* documents (k* tuned on training data)
"""
import argparse, json, os, pickle
import numpy as np
import ir_lib as L
import method_a as MA
import method_b as MB

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--queries", required=True)
ap.add_argument("--method", choices=["A", "B", "B-rank"], default="B")
ap.add_argument("--models", default="models"); ap.add_argument("--out", default="results/predictions")
ap.add_argument("--topn", type=int, default=100); ap.add_argument("--id-mode", choices=["seq", "orig"], default="seq")
ap.add_argument("--k", type=int, default=None, help="override k* for the predicted-relevant file")
a = ap.parse_args()

docs = L.load_docs(a.data)
index = L.Index(docs)
queries = L.load_queries(open(a.queries, encoding="latin-1").read())
ids = [str(i + 1) if a.id_mode == "seq" else str(int(orig)) if orig.strip().isdigit() else orig
       for i, (orig, _) in enumerate(queries)]
X = L.build_feature_tensor(index, [q for _, q in queries])

if a.method == "A":
    m = json.load(open(os.path.join(a.models, "method_A.json")))
    w = np.array([m["weights"][f] for f in L.FIELDS]); scores = MA.score(MA.zone_matrix(X), w); kstar, tag = m["k_star"], "methodA_zone"
else:
    key, tag = ("method_B", "methodB_svm") if a.method == "B" else ("method_B_ranksvm", "methodB_ranksvm")
    m = pickle.load(open(os.path.join(a.models, key + ".pkl"), "rb"))
    scores, kstar = m["model"].score(X), m["k_star"]
kstar = a.k or kstar

order = L.rank_docs(scores)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out + "_results.txt", "w") as f:
    for i, q in enumerate(ids):
        for r, j in enumerate(order[i, :a.topn]):
            f.write(f"{q} Q0 {index.doc_ids[j]} {r+1} {scores[i,j]:.6f} {tag}\n")
with open(a.out + "_predicted_relevant.txt", "w") as f:
    for i, q in enumerate(ids):
        for j in order[i, :kstar]: f.write(f"{q} {index.doc_ids[j]}\n")
print(f"{len(ids)} queries ranked with method {a.method}; wrote {a.out}_results.txt and {a.out}_predicted_relevant.txt (k*={kstar})")
