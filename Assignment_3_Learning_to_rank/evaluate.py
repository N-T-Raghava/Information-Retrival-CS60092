"""
Evaluate a result file against cranqrel.

    python evaluate.py --run results/methodB_test_results.txt --qrels data/cranqrel --k 10
    python evaluate.py --relevant results/methodB_test_predicted_relevant.txt --qrels data/cranqrel

--run       TREC run file  (qid Q0 docid rank score tag): ranked evaluation -> MAP, P/R/F1 @ k
--relevant  plain "qid docid" pairs (the set a system declares relevant): set-based P/R/F1
Relevance: cranqrel codes 1-4 are relevant, -1 is not (use --levels to change).
Scores are macro-averaged over the queries that appear in the run / relevant file.
"""

import argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--run"); ap.add_argument("--relevant"); ap.add_argument("--qrels", required=True)
ap.add_argument("--k", type=int, nargs="*", default=[1, 5, 10, 20])
ap.add_argument("--levels", type=int, nargs="*", default=[1, 2, 3, 4])
a = ap.parse_args()

qrel = defaultdict(set)
for line in open(a.qrels):
    p = line.split()
    if len(p) >= 3 and int(p[2]) in a.levels: qrel[int(p[0])].add(int(p[1]))

def f1(p, r): return 2 * p * r / (p + r) if p + r > 0 else 0.0

if a.run:
    run = defaultdict(list)
    for line in open(a.run):
        p = line.split()
        if len(p) >= 5: run[int(p[0])].append((int(p[3]), int(p[2])))
    qs = sorted(run); aps = []; tot = {k: [0, 0, 0] for k in a.k}
    for q in qs:
        docs = [d for _, d in sorted(run[q])]; rel = qrel.get(q, set()); hit = 0; ap_ = 0.0
        for i, d in enumerate(docs, 1):
            if d in rel: hit += 1; ap_ += hit / i
        aps.append(ap_ / max(len(rel), 1))
        for k in a.k:
            h = len(set(docs[:k]) & rel); P, R = h / k, h / max(len(rel), 1)
            for j, v in enumerate((P, R, f1(P, R))): tot[k][j] += v
    print(f"{len(qs)} queries | MAP = {sum(aps)/len(qs):.4f}  (ranking truncated at {max(len(v) for v in run.values())} docs/query)")
    for k in a.k: print(f"  @{k:<3d} P={tot[k][0]/len(qs):.4f}  R={tot[k][1]/len(qs):.4f}  F1={tot[k][2]/len(qs):.4f}")

if a.relevant:
    ret = defaultdict(set)
    for line in open(a.relevant):
        p = line.split()
        if len(p) >= 2: ret[int(p[0])].add(int(p[1]))
    P = R = F = 0.0
    for q, s in ret.items():
        rel = qrel.get(q, set()); h = len(s & rel); p_, r_ = h / len(s), h / max(len(rel), 1)
        P += p_; R += r_; F += f1(p_, r_)
    n = len(ret)
    print(f"{n} queries | predicted-relevant sets: P={P/n:.4f}  R={R/n:.4f}  F1={F/n:.4f}  (avg size {sum(map(len,ret.values()))/n:.1f})")
