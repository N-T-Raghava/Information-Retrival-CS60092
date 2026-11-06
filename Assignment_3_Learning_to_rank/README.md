# Learning to Rank on the Cranfield Collection
**CS60092 Information Retrieval - Programming Assignment III - Team MaamTeam**

| Roll No. | Name |
|---|---|
| 23CS30035 | Tanmai Raghava |
| 23CS10013 | Yesheeth Sree Narayana |
| 23CS10055 | Lokeswara Reddy |
| 23CS30016 | Manijith Dameruppala |
| 23CS10011 | Vivek Vardhan Varma |

**The report:** `MaamTeam_IR_Assignment_III_Report.pdf`.

---

## 1. What this project does
* Collection: Cranfield - 1,400 documents, 225 queries, 1,612 relevant (query, document) pairs (relevance codes 1-4 = relevant, -1 / unjudged = not relevant).
* **Method A** - *Boolean weighted zone scoring*: score = sum of 4 weights x Boolean match of the query in the **title / author / bibliography / text** zones; the 4 weights are learned by **grid search**.
* **Method B** - *linear SVM*: 22 features x 5 streams = **110 features** per (query, document); the SVM's learned weights give the ranking score.
* **Method B+** (our extra) - *RankSVM*: same features, trained on (relevant, non-relevant) document pairs.
* **References** (not learned): BM25, TF-IDF cosine, equal zone weights, each zone alone.
* Evaluation: precision / recall / F1 (at k = 1, 5, 10, 20 and at a tuned cut-off k\*), MAP, nDCG@10, 11-point PR curve, Wilcoxon + bootstrap tests, 5-fold cross-validation.

## 2. Headline results

**Hold-out: 45 test queries** (180 train / 45 test, random split, seed 42; all tuning on the 180 training queries only)

| System | MAP | nDCG@10 | P@10 | R@10 | F1@10 | F1 @ k\* (k\*) |
|---|---|---|---|---|---|---|
| BM25 (whole document, untrained) | 0.310 | 0.385 | 0.224 | 0.391 | 0.261 | 0.273 (6) |
| TF-IDF cosine (untrained) | 0.298 | 0.370 | 0.220 | 0.372 | 0.252 | 0.267 (8) |
| Zones, equal weights (untrained) | 0.246 | 0.328 | 0.176 | 0.335 | 0.211 | 0.225 (8) |
| **Method A** (grid-searched zone weights) | **0.252** | 0.330 | 0.176 | 0.338 | 0.211 | 0.234 (7) |
| **Method B** (linear SVM, 110 features) | **0.313** | 0.403 | 0.236 | 0.422 | 0.275 | 0.283 (8) |
| Method B+ (RankSVM) | 0.325 | 0.406 | 0.244 | 0.418 | 0.282 | 0.282 (8) |

**5-fold cross-validation over all 225 queries** (mean +- std of 5 folds): Method A 0.270 +- 0.032 | **Method B 0.324 +- 0.026** | Method B+ 0.325 +- 0.029 | BM25 0.321 +- 0.023 (MAP).

**How to read this**
* **Method B is clearly better than Method A**: MAP 0.313 vs 0.252 (+24 % relative), Wilcoxon p = 0.023, bootstrap 95 % CI of the difference [+0.013, +0.116]; it wins on 27 of 45 queries (2 ties, 16 losses). Cross-validation agrees (0.324 vs 0.270).
* **Method B is NOT significantly better than plain BM25** (0.313 vs 0.310; difference +0.003, CI [-0.035, +0.040], p = 0.33; CV 0.324 vs 0.321). It is somewhat better at the top of the ranking (P@1 0.42 vs 0.31) but with 45 queries that is not statistically established. RankSVM is nominally best (0.325) but not different from the pointwise SVM (p = 0.51).
* **Method A**: combining zones helps a lot (title alone 0.192, text alone 0.176, combined 0.252) but *tuning* the weights adds little over equal weights (0.246). The learned weights on author (0.27) / bibliography (0.13) do **not** mean those zones are important - they almost never match a query (author zone matches 0.25 % of relevant pairs), so they only break ties. Only the title/text balance is really learned. The squared-error objective is degenerate (puts 0.78 on the author zone, MAP 0.216).
* Recall ceiling: only 95.4 % of relevant pairs share even one query term with the query, so no lexical method can exceed that recall.

## 3. Folder map - what every file does

```
MaamTeam_Assignment_3/
|- MaamTeam_IR_Assignment_III_Report.pdf  <- THE REPORT (25 pages)
|- README.md                              <- this file
|- requirements.txt                       numpy, scipy, scikit-learn, nltk, matplotlib, reportlab
|
|- ir_lib.py            shared library: load Cranfield (dir or cran.tar.gz), text analysis (lower-case, stop words,
|                       Porter stemmer), per-stream index, the 110 features, metrics (P/R/F1/AP/nDCG/PR curve)
|- method_a.py          Method A: zone-match matrix, weight-simplex grid search (coarse 0.05 + fine 0.01), 3 objectives
|- method_b.py          Method B / B+: SVMRanker (pointwise SVM or RankSVM), candidate filter, CV choice of C
|- run_experiments.py   RUNS EVERYTHING (baselines, A, B, B+, tests, CV, result files, final models)
|- predict.py           ranks documents for NEW queries with a saved model  (the "prediction" program)
|- evaluate.py          scores any result file against cranqrel            (the "evaluation" program)
|
|- data/                cran.all.1400 (documents), cran.qry (queries), cranqrel (relevance judgements)
|- results/             outputs of the final run (explained in section 7)
|   |- results.json                              every metric, per-query AP, curves, weights, CV, significance
|   |- methodA_test_results.txt                  Method A ranking of the 45 test queries (top 100 each)
|   |- methodB_test_results.txt                  Method B ranking of the 45 test queries
|   |- methodB_ranksvm_test_results.txt          Method B+ ranking of the 45 test queries
|   |- method*_test_predicted_relevant.txt       the top-k* documents per query that each method declares relevant  
|
|- models/              final models trained on all 225 queries (see models/README.txt - copy YOUR models here)
|- plots/               all the 12 plots and the plotng script         
```

## 4. Setup
```bash
pip3 install -r requirements.txt          # macOS / Linux   (Windows: pip install -r requirements.txt)
```
If macOS says "externally-managed-environment": `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
Python 3.9+ ; the report build additionally needs `pdflatex` (MacTeX / BasicTeX / TeX Live). On Windows use `python` instead of `python3`.

## 5. How to run

| Goal | Command | Time |
|---|---|---|
| Run **all experiments** (hold-out + 5-fold CV + final models) | `python3 run_experiments.py --data data --cv` | about 20 min |
| Hold-out experiments only (no CV) | `python3 run_experiments.py --data data` | about 11 min |
| **Predict** (rank) new queries | `python3 predict.py --data data --queries NEW.qry --method B --out results/new` | seconds |
| **Evaluate** a result file | `python3 evaluate.py --run results/methodB_test_results.txt --qrels data/cranqrel --k 5 10` | instant |
| Draw **all plots** | `python3 plot_all.py` | seconds |
| Rebuild the **report** | `cd report && python3 make_figures.py && python3 build_report.py` | about 20 s |

`--data` accepts the folder `data/` **or** the archive `cran.tar.gz`. `run_experiments.py` also accepts `--seed` (split seed, default 42) and `--test_frac` (default 0.2).

## 6. How to test that everything works

**(a) Evaluation check - needs no models and no training.** These commands must print exactly:
```bash
python3 evaluate.py --run results/methodA_test_results.txt        --qrels data/cranqrel --k 5 10
#  45 queries | MAP = 0.2458        @5  P=0.2800 R=0.2902 F1=0.2572    @10 P=0.1756 R=0.3375 F1=0.2115
python3 evaluate.py --run results/methodB_test_results.txt        --qrels data/cranqrel --k 5 10
#  45 queries | MAP = 0.3075        @5  P=0.3111 R=0.2941 F1=0.2714    @10 P=0.2356 R=0.4217 F1=0.2754
python3 evaluate.py --run results/methodB_ranksvm_test_results.txt --qrels data/cranqrel --k 5 10
#  45 queries | MAP = 0.3191        @5  P=0.3333 R=0.3156 F1=0.2910    @10 P=0.2444 R=0.4181 F1=0.2815
python3 evaluate.py --relevant results/methodB_test_predicted_relevant.txt --qrels data/cranqrel
#  45 queries | predicted-relevant sets: P=0.2667  R=0.3784  F1=0.2835  (avg size 8.0)
```
(The same check for Method A / B+ sets gives P=0.2222 R=0.3097 F1=0.2345 and P=0.2667 R=0.3755 F1=0.2823.) The MAP printed here is slightly lower than in `results.json` (0.2458 vs 0.2521 for A, 0.3075 vs 0.3134 for B) **only because the result files keep the top 100 documents per query**; P@5, P@10, R@10 and the set-based numbers match `results.json` exactly.

**(b) Prediction check** (needs `models/`):
```bash
python3 predict.py --data data --queries data/cran.qry --method B --out results/sanity
python3 evaluate.py --run results/sanity_results.txt --qrels data/cranqrel --k 10
```
Expect MAP about 0.32 - but this is **in-sample** (the final models were trained on all 225 queries), so it is only a "pipeline works" check, never a test score.

**(c) Full reproduction:** `python3 run_experiments.py --data data --cv`, then compare `results/results.json` with the one shipped. The SVM solver is run with an iteration cap, so SVM numbers can differ in the third decimal between machines (an independent run gave Method B MAP 0.316 instead of 0.313); baselines and Method A reproduce exactly.

## 7. Result files - what each one is

All ranking files are in **TREC format**, one line per (query, document), best first, top 100 per query (45 queries x 100 = 4,500 lines):
```
qid Q0 docid rank score tag
10 Q0 302 1 3.716256 methodB_svm
```
`qid` = query number 1..225 in `cran.qry` order (the numbering `cranqrel` uses - **not** the `.I` ids); `docid` = Cranfield document id; `score` = model score; `tag` = run name.

| File | Method | Contents |
|---|---|---|
| `results/methodA_test_results.txt` | A | ranking of the 45 test queries by the grid-searched zone weights (title 0.35, author 0.27, bib 0.13, text 0.25) |
| `results/methodB_test_results.txt` | B | ranking by the linear SVM (C = 0.001) |
| `results/methodB_ranksvm_test_results.txt` | B+ | ranking by the RankSVM (C = 0.001) |
| `results/methodA_test_predicted_relevant.txt` | A | `qid docid` pairs: the **top k\* = 7** documents per query (315 lines) |
| `results/methodB_test_predicted_relevant.txt` | B | top **k\* = 8** per query (360 lines) |
| `results/methodB_ranksvm_test_predicted_relevant.txt` | B+ | top **k\* = 8** per query (360 lines) |
| `results/results.json` | all | all hold-out metrics for 15 systems, per-query AP, PR / F1 curves, learned weights, CV results, Wilcoxon p-values, runtimes, final-model settings |

The 45 test queries are 10, 16, 17, 19, 26, 31, 46, 61, 69, 76, 80, 83, 87, 97, 102, 105, 112, 115, 118, 119, 121, 142, 143, 144, 145, 149, 153, 166, 168, 173, 176, 177, 179, 183, 185, 194, 196, 199, 201, 202, 208, 214, 215, 221, 223.
(The files use Windows line endings because they were produced on Windows; all tools here read them fine.)

## 8. Prediction vs. evaluation - what they mean

* **Prediction** = `predict.py`: takes documents + a query file in `cran.qry` format (`.I` / `.W`), loads a saved model from `models/`, and writes (i) the **ranked list** `<out>_results.txt` (TREC format, top `--topn`, default 100) and (ii) `<out>_predicted_relevant.txt`, the top-k\* documents it declares relevant. Options: `--method A | B | B-rank`, `--id-mode seq` (default, queries numbered 1..n like `cranqrel`) or `orig` (keep the `.I` ids), `--k` (override k\*).
* **Evaluation** = `evaluate.py`: compares a ranking (`--run`) or a set (`--relevant`) with `cranqrel` and prints macro-averaged P / R / F1 (at the cut-offs given by `--k`) and MAP. Relevance: codes 1-4 relevant (`--levels` to change).
* **Cut-off k\*** = the k in 1..100 that maximises mean F1 on the *training* queries (out-of-fold scores for the SVMs), so the "predicted relevant" sets are not tuned on test data.
* **Three kinds of numbers**: *hold-out* (45 test queries, models trained on 180) = the headline table; *5-fold CV* (all 225 queries, re-trained in each fold) = variance estimate; *final models* (trained on all 225) = what `predict.py` uses on unseen queries - these cannot be scored on Cranfield itself.

## 9. Which runs were done

1. **Development runs** (not part of the deliverable): data checks (verified that `cranqrel` numbers queries 1..225 in file order: BM25 MAP 0.32 with that numbering vs 0.01 with raw `.I` ids), feature sanity checks, and solver timing tests that led to the dual-hinge `LinearSVC` setting.
2. **THE final run** (on the team machine): `python run_experiments.py --data data --cv` - stages and wall-clock from its console log:
   * collection + 110 features for 225 x 1,400 pairs: ~2 s; 8 reference systems: ~3 s;
   * **Method A** grid search, 3 objectives (MAP / squared error / F1@10): 81 s / 32 s / 79 s;
   * **Method B** pointwise SVM (all features), text-zone-only and no-author/bib ablations: ~72 s / 2 s / 4 s (incl. 5-fold CV for C);
   * **Method B+** RankSVM: ~276 s; hold-out stage finished at 549 s;
   * **5-fold CV over all 225 queries** (A, B, B+, BM25): 488 s; **final models** retrained on all queries: 125 s; total 1,162 s.
3. `results/results.json` and the three result files in `results/` are the artifacts of run 2; the report and plots are generated only from them.

## 10. Plots (`plots/`, PNG; all drawn by `python3 plot_all.py`; the numbers in brackets are the figure numbers in the report)

| File | Report fig. | What it shows |
|---|---|---|
| `fig01_overview.png` | 6 | MAP of all 15 systems on the 45 test queries, colour-coded by family |
| `fig02_zones.png` | 1 | (a) how often each zone matches a query term, relevant vs non-relevant pairs; (b) MAP of each zone alone |
| `fig03_weightsA.png` | 2 | Method A zone weights: (a) per training objective, (b) per cross-validation fold |
| `fig04_C.png` | 3 | 5-fold CV MAP vs SVM regularisation constant C (pointwise SVM and RankSVM) |
| `fig05_ablation.png` | 4 | Method B feature-set ablation (text only / no author-bib / all) + RankSVM, MAP and F1@k\* |
| `fig06_weightsB.png` | 5 | 12 largest learned weights of Method B and of the RankSVM |
| `fig07_pr.png` | 8 | 11-point interpolated precision-recall curves (A, B, B+, BM25, TF-IDF) |
| `fig08_fk.png` | 9 | (a) macro-F1 vs cut-off k; (b) precision and recall vs k (Method A vs B) |
| `fig09_fixedk.png` | 7 | precision / recall / F1 at k = 5, 10, 20 for A, B, B+, BM25 |
| `fig10_relative.png` | 10 | relative improvement (%) of Method B over Method A and over BM25, per metric |
| `fig11_scatter.png` | 12 | per-query AP: Method B vs Method A, and Method B vs BM25 |
| `fig12_cv.png` | 11 | 5-fold CV mean +- std of MAP, nDCG@10, F1@10, F1@k\* for A, B, B+, BM25 |

`plot_all.py` needs only `results/results.json`, numpy and matplotlib: `python3 plot_all.py --results results/results.json --out plots --formats png --dpi 200`.

## 11. Method details (short)
* **Pre-processing**: lower-case -> alphanumeric tokens -> English stop words and single letters removed -> Porter stemming (6,490 stems). Five streams: title, author, bib, text, all (concatenation).
* **Method A**: `s_i = (#query terms found in zone i) / |q|`, `score = sum_i g_i s_i`, `g_i >= 0`, `sum g_i = 1`. Grid: step 0.05 over the simplex (1,771 points) + step 0.01 refinement around the best (1,331 points). Main objective = MAP on training queries; ties are ranked in collection order.
* **Method B features** (per stream, 22 each): covered-term count and ratio, stream length, idf; sum / min / max / mean / variance of tf, of length-normalised tf, and of tf*idf; Boolean (AND) match; vector-space cosine; BM25 (k1 = 1.2, b = 0.75).
* **SVM**: scikit-learn `LinearSVC` (hinge loss, dual, <= 300 iterations), standardised features, balanced class weights, all positives + 150 sampled negatives per query, only *candidate* documents (>= 1 shared query term; 63.9 % of pairs, 95.4 % of relevant pairs) are scored. C in {0.001, 0.01, 0.1, 1} chosen by query-grouped 5-fold CV (best = 0.001 for both SVMs, the edge of the grid).
* **RankSVM**: same features; linear SVM without intercept on differences x(d+) - x(d-) with random sign flips.

## 12. Known limitations (also in the report, section 9.6)
Only 45 held-out queries (CV std about 0.025 MAP, so small gaps are noise); relevance codes 1-4 collapsed to binary; SVM uses negative sub-sampling and a solver iteration cap; C is at the grid edge; Boolean zone scores create many ties; result files are truncated at 100 documents (MAP from them is slightly lower); the final models cannot be evaluated on Cranfield itself.

## 13. Troubleshooting
* `ModuleNotFoundError` -> `pip3 install -r requirements.txt`.
* `predict.py` says `models/method_B.pkl` not found -> copy your `models/` folder from the full run (or run `run_experiments.py` once).
* Report build fails -> install `pdflatex`; the PDF in this folder is already built.
* Wrong MAP with your own `.qry` file -> check you did not pass `--id-mode orig` when evaluating against `cranqrel` (which numbers queries 1..n).
