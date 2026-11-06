"""
plot_all.py -- draws EVERY plot of the project from results/results.json (12 figures, PNG).

    python3 plot_all.py                                   # results/results.json -> plots/*.png
    python3 plot_all.py --results results/results.json --out plots --formats png --dpi 200   (add pdf/svg if wanted)

Only needs numpy and matplotlib (no data files, no model training). File names:
    fig01_overview  fig02_zones   fig03_weightsA  fig04_C       fig05_ablation  fig06_weightsB
    fig07_pr        fig08_fk      fig09_fixedk    fig10_relative fig11_scatter   fig12_cv
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default="results/results.json"); _ap.add_argument("--out", default="plots")
_ap.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"]); _ap.add_argument("--dpi", type=int, default=200)
_args = _ap.parse_args()
R = json.load(open(_args.results)); S = R["systems"]; CV = R["cv5"]
os.makedirs(_args.out, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True,
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
                     "axes.axisbelow": True, "axes.unicode_minus": False})
NAVY, BLUE, TEAL, ORANGE, GREY, LGREY, RED = "#1F3864", "#2E75B6", "#2A9D8F", "#E07B39", "#7F7F7F", "#BFBFBF", "#C0504D"
A, B, BR = "Method A (grid, obj=map)", "Method B (SVM, all 110 feat.)", "Method B+ (RankSVM, all 110 feat.)"
BM, BMT, TF = "BM25 (whole doc)", "BM25 (text zone)", "TF-IDF cosine (whole doc)"
COL = {A: ORANGE, B: NAVY, BR: TEAL, BM: GREY, TF: LGREY}
LAB = {A: "Method A (zones)", B: "Method B (SVM)", BR: "Method B+ (RankSVM)", BM: "BM25", TF: "TF-IDF cosine"}
def save(name):
    plt.tight_layout()
    for ext in _args.formats: plt.savefig(os.path.join(_args.out, f"{name}.{ext}"), dpi=_args.dpi)
    plt.close()

# 1 -- overview of all systems --------------------------------------------------------------
items = [("Single zone: author", "Single zone: author", RED), ("Single zone: bib", "Single zone: bibliography", RED),
         ("Single zone: text", "Single zone: text", RED), ("Single zone: title", "Single zone: title", RED),
         ("Zones, equal weights (0.25 each)", "Zones, equal weights", ORANGE), ("Method A (grid, obj=sse)", "Method A, squared-error obj.", ORANGE),
         ("Method A (grid, obj=f1@10)", "Method A, F1@10 obj.", ORANGE), (A, "Method A, MAP obj.", ORANGE),
         (TF, "TF-IDF cosine (whole doc)", GREY), ("Method B (SVM, text-zone feat. only)", "Method B, text-zone features", NAVY),
         (BMT, "BM25 (text zone)", GREY), (BM, "BM25 (whole doc)", GREY), (B, "Method B, all features", NAVY),
         ("Method B (SVM, no author/bib feat.)", "Method B, no author/bib feat.", NAVY), (BR, "Method B+ (RankSVM)", TEAL)]
items = sorted(items, key=lambda t: S[t[0]]["AP"])
fig, ax = plt.subplots(figsize=(6.3, 4.1))
y = np.arange(len(items)); v = [S[k]["AP"] for k, _, _ in items]
ax.barh(y, v, color=[c for *_, c in items], height=.68)
for i, x in enumerate(v): ax.text(x + .004, i, f"{x:.3f}", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([l for _, l, _ in items]); ax.set_xlabel("MAP (45 held-out queries)"); ax.set_xlim(0, .39)
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=RED, label="single zone"), Patch(color=ORANGE, label="Method A family"), Patch(color=GREY, label="untrained reference"),
                   Patch(color=NAVY, label="Method B family"), Patch(color=TEAL, label="RankSVM")], loc="lower right")
save("fig01_overview")

# 2 -- zone evidence ---------------------------------------------------------------------
zs = R["zone_stats"]; Z = ["title", "author", "bib", "text"]
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
x = np.arange(4); w = .38
ax[0].bar(x - w / 2, [100 * zs[z]["match_rate_relevant"] for z in Z], w, color=NAVY, label="relevant pairs")
ax[0].bar(x + w / 2, [100 * zs[z]["match_rate_nonrelevant"] for z in Z], w, color=LGREY, label="non-relevant pairs")
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(Z); ax[0].set_ylabel("query-term match rate (%)"); ax[0].set_title("(a) How often a zone matches"); ax[0].set_ylim(.01, 4000); ax[0].legend(loc="upper center", ncol=2, fontsize=7.5)
sz = [S[f"Single zone: {z}"]["AP"] for z in Z]
ax[1].bar(x, sz, .55, color=RED)
for i, t in enumerate(sz): ax[1].text(i, t + .005, f"{t:.3f}", ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(Z); ax[1].set_ylabel("MAP"); ax[1].set_title("(b) Ranking by one zone alone"); ax[1].set_ylim(0, .24)
save("fig02_zones")

# 3 -- learned zone weights ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8), gridspec_kw={"width_ratios": [1, 1.5]})
zc = [NAVY, ORANGE, RED, TEAL]
def stack(a, W, labels, title):
    bottom = np.zeros(len(W))
    for j, z in enumerate(Z):
        vals = np.array([w_[j] for w_ in W]); a.bar(range(len(W)), vals, .62, bottom=bottom, color=zc[j], label=z); bottom += vals
    a.set_xticks(range(len(W))); a.set_xticklabels(labels); a.set_ylim(0, 1); a.set_ylabel("weight"); a.set_title(title)
objs = [("map", "MAP"), ("sse", "squared\nerror"), ("f1@10", "F1@10")]
stack(ax[0], [[R["method_A"][o]["weights"][z] for z in Z] for o, _ in objs], [l for _, l in objs], "(a) Training objective")
stack(ax[1], R["cv5_weightsA"], [f"fold {i+1}" for i in range(5)], "(b) MAP-objective weights per CV fold")
ax[1].legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=4)
save("fig03_weightsA")

# 4 -- choice of C ---------------------------------------------------------------------------
Cs = ["0.001", "0.01", "0.1", "1.0"]
fig, ax = plt.subplots(figsize=(4.4, 2.7))
for n, c, l in ((B, NAVY, "pointwise SVM"), (BR, TEAL, "RankSVM")):
    ax.plot([float(c_) for c_ in Cs], [R["method_B"][n]["cv_map"][c_] for c_ in Cs], "-o", color=c, label=l, ms=4)
ax.set_xscale("log"); ax.set_xlabel("regularisation constant C"); ax.set_ylabel("5-fold CV MAP (training queries)"); ax.legend()
save("fig04_C")

# 5 -- ablation --------------------------------------------------------------------------------
abl = [("Method B (SVM, text-zone feat. only)", "text-zone\nfeatures"), ("Method B (SVM, no author/bib feat.)", "no author/bib\nfeatures"),
       (B, "all 110\nfeatures"), (BR, "RankSVM\nall features")]
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
for a, met, t in ((ax[0], "AP", "MAP"), (ax[1], "F1@k*", "F1 at tuned cut-off k*")):
    vals = [S[k][met] for k, _ in abl]; a.bar(range(4), vals, .6, color=[BLUE, BLUE, NAVY, TEAL])
    a.axhline(S[BM][met], color=GREY, ls="--", lw=1, label="BM25 (untrained)")
    for i, t_ in enumerate(vals): a.text(i, t_ + .003, f"{t_:.3f}", ha="center", fontsize=8)
    a.set_xticks(range(4)); a.set_xticklabels([l for _, l in abl], fontsize=8); a.set_title(t); a.set_ylim(min(vals) - .05, max(vals) + .03); a.legend(loc="upper left", fontsize=7.5)
save("fig05_ablation")

# 6 -- SVM weights -------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.3))
for a, n, t in ((ax[0], B, "(a) Method B (pointwise SVM)"), (ax[1], BR, "(b) Method B+ (RankSVM)")):
    tw = R["method_B"][n]["top_weights"][:12][::-1]
    a.barh([k.replace("_", " ") for k, _ in tw], [w_ for _, w_ in tw], color=[TEAL if w_ > 0 else ORANGE for _, w_ in tw], height=.7)
    a.set_title(t); a.set_xlabel("weight (standardised features)"); a.tick_params(axis="y", labelsize=7); a.axvline(0, color="k", lw=.6); a.grid(axis="y", visible=False)
save("fig06_weightsB")

# 7 -- PR curves -----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for n in (A, B, BR, BM, TF):
    c = R["curves"][n]; ax.plot(c["recall_pts"], c["p11"], "-o", ms=3, lw=1.4, color=COL[n], label=LAB[n])
ax.set_xlabel("recall"); ax.set_ylabel("interpolated precision"); ax.legend(); ax.set_ylim(0, .68)
save("fig07_pr")

# 8 -- F1/P/R versus k -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.9)); ks = np.arange(1, 41)
for n in (A, B, BR, BM, TF): ax[0].plot(ks, R["curves"][n]["F_at_k"], color=COL[n], lw=1.5, label=LAB[n])
ax[0].axvline(R["mean_rel_per_query"], color="k", ls=":", lw=.8); ax[0].text(R["mean_rel_per_query"] + .5, .12, "mean #relevant", fontsize=7.5)
ax[0].set_xlabel("cut-off k"); ax[0].set_ylabel("macro F1@k"); ax[0].set_title("(a) F1 versus cut-off"); ax[0].legend(loc="upper right", fontsize=7.5); ax[0].set_ylim(.1, .33)
for n, ls in ((B, "-"), (A, "--")):
    ax[1].plot(ks, R["curves"][n]["P_at_k"], ls, color=BLUE, lw=1.5, label=f"precision, {LAB[n]}"); ax[1].plot(ks, R["curves"][n]["R_at_k"], ls, color=ORANGE, lw=1.5, label=f"recall, {LAB[n]}")
ax[1].set_xlabel("cut-off k"); ax[1].set_title("(b) Precision/recall trade-off"); ax[1].legend(fontsize=7)
save("fig08_fk")

# 9 -- fixed cut-offs ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.7), sharey=True); sysl = [A, B, BR, BM]; w = .2
for a, k in zip(ax, (5, 10, 20)):
    for j, (m, c) in enumerate((("P", BLUE), ("R", ORANGE), ("F1", NAVY))):
        a.bar(np.arange(4) + (j - 1) * w * 1.15, [S[n][f"{m}@{k}"] for n in sysl], w, color=c, label={"P": "precision", "R": "recall", "F1": "F1"}[m])
    a.set_xticks(range(4)); a.set_xticklabels(["A", "B", "B+", "BM25"]); a.set_title(f"k = {k}")
ax[0].legend(fontsize=7); ax[0].set_ylabel("score (macro-average)")
save("fig09_fixedk")

# 10 -- relative improvement ------------------------------------------------------------------------
mets = ["AP", "nDCG@10", "P@5", "P@10", "R@10", "F1@10", "P@20", "R@20", "F1@k*"]; ml = ["MAP", "nDCG@10", "P@5", "P@10", "R@10", "F1@10", "P@20", "R@20", "F1@k*"]
fig, ax = plt.subplots(figsize=(6.3, 2.9)); x = np.arange(len(mets)); w = .38
ra = [100 * (S[B][m] / S[A][m] - 1) for m in mets]; rb = [100 * (S[B][m] / S[BM][m] - 1) for m in mets]
ax.bar(x - w / 2, ra, w, color=NAVY, label="Method B over Method A"); ax.bar(x + w / 2, rb, w, color=GREY, label="Method B over BM25")
for i, (p, q) in enumerate(zip(ra, rb)):
    ax.text(i - w / 2, p + .6, f"{p:.0f}", ha="center", fontsize=7); ax.text(i + w / 2, q + (.6 if q >= 0 else -2.2), f"{q:.0f}", ha="center", fontsize=7)
ax.axhline(0, color="k", lw=.6); ax.set_xticks(x); ax.set_xticklabels(ml, fontsize=8); ax.set_ylabel("relative change (%)"); ax.legend(loc="upper right")
save("fig10_relative")

# 11 -- per-query scatter ---------------------------------------------------------------------------
Q = R["per_query_AP"]; fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.1))
for a, ref, t in ((ax[0], A, "(a) Method B versus Method A"), (ax[1], BM, "(b) Method B versus BM25")):
    a.scatter(Q[ref], Q[B], s=14, color=NAVY, alpha=.75, edgecolor="white", linewidth=.4); a.plot([0, 1], [0, 1], "k--", lw=.8)
    a.set_xlabel(f"AP, {LAB[ref]}"); a.set_ylabel("AP, Method B"); a.set_title(t); a.set_xlim(0, 1.02); a.set_ylim(0, 1.02); a.set_aspect("equal")
save("fig11_scatter")

# 12 -- cross-validation ----------------------------------------------------------------------------
cvs = [("Method A", "Method A", ORANGE), ("Method B", "Method B", NAVY), ("Method B+ (RankSVM)", "Method B+", TEAL), (BM, "BM25", GREY)]
fig, ax = plt.subplots(figsize=(6.3, 2.9)); mm = [("AP", "MAP"), ("nDCG@10", "nDCG@10"), ("F1@10", "F1@10"), ("F1@k*", "F1@k*")]; w = .2
for j, (k, l, c) in enumerate(cvs):
    ax.bar(np.arange(4) + (j - 1.5) * w, [CV[k][m][0] for m, _ in mm], w, yerr=[CV[k][m][1] for m, _ in mm], color=c, label=l, capsize=2, error_kw={"lw": .8})
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in mm]); ax.set_ylabel("5-fold CV mean $\\pm$ std"); ax.legend(ncol=4, loc="upper right"); ax.set_ylim(0, .55)
save("fig12_cv")
print(f"{len(os.listdir(_args.out))} files written to {_args.out}/:", ", ".join(sorted(os.listdir(_args.out))))
