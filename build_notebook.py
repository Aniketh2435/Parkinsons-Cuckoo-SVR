import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Parkinson's Disease Severity Prediction from Voice Recordings using SVR Optimised with Cuckoo Search

**Bio-Inspired Algorithms AAT (23DS7BSBIO)**, Group 15, Department of CSE (Data Science), B.M.S. College of Engineering

**Goal:** predict a patient's clinician-rated **motor UPDRS** score (Parkinson's symptom severity) from biomedical
voice measurements, so symptom progression can be monitored remotely from a phone recording.

**Method:** a Support Vector Regressor (RBF kernel) whose hyperparameters **and** input-feature subset are chosen jointly by
**Cuckoo Search**, implemented from scratch.

This notebook is the complete, executed record of the experiment:
1. Dataset, cleaning and preprocessing
2. Baselines without optimisation: mean predictor, linear regression, default SVR
3. Cuckoo Search (joint hyperparameter tuning + feature selection)
4. Random Search with the same budget (non-bio-inspired reference)
5. Baseline vs optimised comparison on the untouched test set
6. Learning curves of the default and optimised SVR
7. Robustness: 10 independent runs of both optimisers, with statistical tests

Before the SVR experiment, the Cuckoo Search operators are checked on four standard optimisation test functions.

All figures and numbers are also saved to `outputs/`.
""")

md("## 0. Run information")
code("""
import datetime, platform, subprocess, sys
import numpy as np, pandas as pd, scipy, sklearn
print("Run started :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Machine     :", platform.platform())
print("CPU         :", subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor())
print("Python      :", sys.version.split()[0])
print("Libraries   : NumPy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__, "| SciPy", scipy.__version__)
""")

md("""
## 1. Configuration

**Search budget.** Each optimiser run gets about 820 fitness evaluations (20 nests x 20 generations; Random Search
gets exactly the same number). An earlier run with only 170 evaluations (10 nests x 8 generations) left Cuckoo
Search and Random Search tied: 8 generations are too few for a guided search to pay off in a 21-dimensional space.
A budget study on three seeds showed Cuckoo Search pulling clearly ahead after about 500 evaluations, so the
budget was raised for **both** methods. The earlier runs are kept in `archive/`.
""")
code("""
import json, time, inspect, warnings
from pathlib import Path
import joblib
from IPython.display import Image, Markdown, display
from scipy.stats import wilcoxon

from src import plots
from src.cuckoo import CuckooSearch
from src.data import TARGET, prepare_data
from src.metrics import compare, regression_metrics
from src.model import DEFAULT_PARAMS, baselines, build_svr
from src.random_search import random_search
from src.search_space import HYPERPARAMS, Objective, decode, default_position, dim

warnings.filterwarnings("ignore")
CONFIG = {
    "data": "data/parkinsons_updrs.data",
    "split_seed": 42,          # train/test split and CV folds
    "main_seed": 42,           # seed of the detailed run (sections 5-8)
    "nests": 20,               # Cuckoo Search population size
    "generations": 20,         # about 820 fitness evaluations per run (Random Search gets the same number)
    "pa": 0.25,                # probability a nest is discovered and abandoned
    "alpha": 1.0,              # Levy step-size scale
    "beta": 1.5,               # Levy exponent
    "cv_folds": 3,             # fitness = 3-fold CV RMSE on the training set
    "robustness_seeds": list(range(1, 11)),
}
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
timings, T0 = {}, time.time()
print(json.dumps(CONFIG, indent=2))
""")

md("## 2. Dataset\nOxford Parkinson's Disease Telemonitoring dataset (UCI, Tsanas et al., 2009): 5,875 voice recordings from 42 early-stage Parkinson's patients, about 200 recordings per patient over six months.")
code("""
from src.data import load_raw
raw = load_raw(CONFIG["data"])
print("Shape:", raw.shape)
display(raw.head())
display(raw.describe().T.round(4))
""")

md("""
## 3. Preprocessing
- Checked for missing values and duplicate rows
- Removed non-voice columns: patient ID, time since recruitment, and `total_UPDRS` (a second score that contains the target)
- **Inputs (18):** age, sex and 16 voice measures (jitter, shimmer, NHR, HNR, RPDE, DFA, PPE). **Target:** `motor_UPDRS`
- `log1p` transform of the strongly right-skewed jitter, shimmer and NHR measures
- 80/20 train/test split (seed 42); the test set is used **only once**, for the final evaluation
- Standardisation of features and target happens **inside** each model pipeline, fitted on training data only (no leakage)
""")
code("""
data = prepare_data(CONFIG["data"], seed=CONFIG["split_seed"])
print(json.dumps(data.report, indent=2))
skew = pd.DataFrame({"before": raw[data.report["log_transformed"]].skew(),
                     "after log1p": np.log1p(raw[data.report["log_transformed"]]).skew()}).round(2)
display(Markdown("**Skewness of the log-transformed features**")); display(skew)
plots.target_distribution(raw[TARGET], OUT / "target_distribution.png")
plots.correlation_heatmap(raw.drop(columns=["subject#"]), OUT / "correlation_heatmap.png")
display(Image(OUT / "target_distribution.png")); display(Image(OUT / "correlation_heatmap.png"))
""")

md("## 4. Baselines (no optimisation)\nAll trained on all 18 features. The **default SVR** uses scikit-learn's defaults (C = 1, gamma = 'scale', epsilon = 0.1) and is the model that Cuckoo Search optimises.")
code("""
results, preds = {}, {}
for name, model in baselines().items():
    model.fit(data.X_train, data.y_train)
    preds[name] = model.predict(data.X_test)
    results[name] = regression_metrics(data.y_test, preds[name])
pd.DataFrame(results).T.round(4)
""")

md("""
## 5. Cuckoo Search: joint hyperparameter tuning and feature selection

**Encoding.** Each nest is a position in [0, 1]^21:

| Dimensions | Meaning | Decoding |
|---|---|---|
| 1-3 | C, gamma, epsilon | log-scale between the bounds below |
| 4-21 | one per feature | feature used if value > 0.5 (threshold transfer to binary) |

**Fitness** = 3-fold cross-validated RMSE on the training set (lower is better). The test set is never seen during the search.

**Operators** (`src/cuckoo.py`):
1. **Levy flight**: new egg = nest + alpha * Levy(beta) * (nest - best) * N(0, 1), with Levy steps drawn by Mantegna's algorithm (beta = 1.5). Most steps are small (exploitation), occasional steps are long (exploration).
2. **Nest replacement**: the new egg replaces the nest it was laid from if it is fitter (greedy selection, as in Yang & Deb's reference code).
3. **Nest abandonment**: each component is discovered with probability pa = 0.25, and the nest is rebuilt by a random walk between two other nests (keeps diversity).
4. **Elitism**: the best nest is always kept.
5. **Warm start**: one initial nest is the default SVR with all features, so the search starts from the baseline.
""")
code("""
display(pd.DataFrame(HYPERPARAMS, columns=["Hyperparameter", "Lower bound", "Upper bound"]).assign(Scale="log10"))
import src.cuckoo
display(Markdown("**Cuckoo Search implementation (`src/cuckoo.py`):**\\n```python\\n" + inspect.getsource(src.cuckoo) + "\\n```"))
""")

md("""
### 5.1 Operator check on standard test functions

Before trusting Cuckoo Search on the SVR, its operators are checked on four standard benchmark functions in 10
dimensions, each with known global minimum f = 0 (`src/benchmarks.py`):

| Function | Domain | Difficulty |
|---|---|---|
| Sphere | [-100, 100] | smooth, one minimum: tests basic convergence |
| Rastrigin | [-5.12, 5.12] | about 10^10 local minima: tests escaping local optima |
| Rosenbrock | [-5, 10] | narrow curved valley: tests fine convergence |
| Ackley | [-32.768, 32.768] | flat outer region with many small dips |

Each method gets 10 seeds and the same budget of evaluations. Three methods are compared:
* **Cuckoo Search**: the implementation used in this project.
* **Cuckoo Search (random-nest replacement)**: an ablation in which a new egg replaces a *randomly chosen* nest
  instead of its own (a reading of the original paper's pseudocode, and our first version). One good egg can then
  overwrite the whole population, so diversity collapses and the search stalls.
* **Random Search**: uniform sampling, the non-bio-inspired reference.
""")
code("""
from src.benchmarks import FUNCTIONS, BenchmarkObjective
from src.cuckoo import levy_step

class CuckooRandomReplacement(CuckooSearch):
    \"\"\"Ablation: the Levy-flight egg replaces a randomly chosen nest j instead of its own nest i.\"\"\"
    def _lay_eggs_by_levy_flight(self, nests, fitness, best):
        for i in range(self.n_nests):
            step = self.alpha * levy_step(self.rng, self.dim, self.beta) * (nests[i] - best)
            egg = np.clip(nests[i] + step * self.rng.standard_normal(self.dim), 0, 1)
            f_egg = self.objective(egg)
            j = self.rng.integers(self.n_nests)
            if f_egg < fitness[j]:
                nests[j], fitness[j] = egg, f_egg

BENCH = {"dim": 10, "nests": 25, "generations": 200, "budget": 9000, "seeds": list(range(1, 11))}
budget = BENCH["budget"]   # every method is compared over exactly the same number of evaluations
t = time.time()
bench_rows, bench_curves = [], {}
for fname in FUNCTIONS:
    bench_curves[fname] = {}
    for method in ["Cuckoo Search", "Cuckoo Search (random-nest replacement)", "Random Search"]:
        finals, runs = [], []
        for s in BENCH["seeds"]:
            obj = BenchmarkObjective(fname)
            if method == "Random Search":
                random_search(obj, BENCH["dim"], budget, seed=s + 1000, verbose=False)
            else:
                cls = CuckooSearch if method == "Cuckoo Search" else CuckooRandomReplacement
                cls(obj, BENCH["dim"], n_nests=BENCH["nests"], n_generations=BENCH["generations"],
                    pa=CONFIG["pa"], alpha=CONFIG["alpha"], beta=CONFIG["beta"], seed=s, verbose=False).run()
            assert obj.n_evals >= budget
            curve = np.array([e["best_so_far"] for e in obj.log])[:budget]
            runs.append(curve); finals.append(curve[-1])
        bench_curves[fname][method] = np.array(runs)
        bench_rows.append({"Function": fname, "Method": method, "mean best f": np.mean(finals),
                           "std": np.std(finals), "median": np.median(finals), "best run": np.min(finals)})
timings["benchmarks_s"] = time.time() - t
bench_df = pd.DataFrame(bench_rows)
bench_df.to_csv(OUT / "benchmark_functions.csv", index=False)
print(f"{len(BENCH['seeds'])} seeds x {budget} evaluations per run, dimension {BENCH['dim']}, {timings['benchmarks_s']:.0f} s")
display(bench_df.pivot(index="Function", columns="Method", values="mean best f")
        [["Cuckoo Search", "Cuckoo Search (random-nest replacement)", "Random Search"]].map(lambda v: f"{v:.3g}"))
plots.benchmark_convergence(bench_curves, OUT / "benchmark_convergence.png")
display(Image(OUT / "benchmark_convergence.png"))
""")

md("### 5.2 Main Cuckoo Search run on the SVR")
code("""
D = dim(data.n_features)
warm = [default_position(data.n_features)]

def run_cuckoo(seed, verbose):
    obj = Objective(data.X_train, data.y_train, data.feature_names, folds=CONFIG["cv_folds"], seed=CONFIG["split_seed"])
    cs = CuckooSearch(obj, D, n_nests=CONFIG["nests"], n_generations=CONFIG["generations"], pa=CONFIG["pa"],
                      alpha=CONFIG["alpha"], beta=CONFIG["beta"], seed=seed, warm_start=warm, verbose=verbose)
    x, f = cs.run()
    return cs, obj, x, f

def run_random(seed, n_evals, verbose):
    obj = Objective(data.X_train, data.y_train, data.feature_names, folds=CONFIG["cv_folds"], seed=CONFIG["split_seed"])
    x, f = random_search(obj, D, n_evals, seed=seed + 1000, verbose=verbose)
    return obj, x, f

def fit_and_test(x):
    params, mask = decode(x)
    model = build_svr(**params).fit(data.X_train[:, mask], data.y_train)
    return model, params, mask, model.predict(data.X_test[:, mask])

t = time.time()
cs, cs_obj, cs_x, cs_cv = run_cuckoo(CONFIG["main_seed"], verbose=True)
timings["cuckoo_search_s"] = time.time() - t
cs_params, cs_mask = decode(cs_x)
print(f"\\nFinished in {timings['cuckoo_search_s']/60:.1f} min, {cs_obj.n_evals} fitness evaluations")
print(f"Default SVR CV RMSE (warm-start nest): {cs_obj.log[0]['fitness']:.4f}")
print(f"Best CV RMSE: {cs_cv:.4f}")
print("Best hyperparameters:", {k: float(f"{v:.4g}") for k, v in cs_params.items()})
print(f"Selected {cs_mask.sum()} of {data.n_features} features:", np.array(data.feature_names)[cs_mask].tolist())
""")

md("## 6. Random Search with the same budget")
code("""
t = time.time()
rs_obj, rs_x, rs_cv = run_random(CONFIG["main_seed"], cs_obj.n_evals, verbose=True)
timings["random_search_s"] = time.time() - t
rs_params, rs_mask = decode(rs_x)
print(f"\\nBest CV RMSE: {rs_cv:.4f} (Cuckoo Search: {cs_cv:.4f})")
print("Best hyperparameters:", {k: float(f"{v:.4g}") for k, v in rs_params.items()}, f"| {rs_mask.sum()} features")
plots.convergence(cs_obj.log, rs_obj.log, cs.history, cs.mean_history, OUT / "convergence.png")
display(Image(OUT / "convergence.png"))
pd.DataFrame(cs_obj.log).to_csv(OUT / "search_log_cuckoo.csv", index=False)
pd.DataFrame(rs_obj.log).to_csv(OUT / "search_log_random.csv", index=False)
print("Best CV RMSE per generation:", [round(float(v), 4) for v in cs.history])
""")

md("## 7. Test-set evaluation: baseline vs optimised")
code("""
cs_model, _, _, preds["Cuckoo-SVR"] = fit_and_test(cs_x)
_, _, _, preds["Random Search SVR"] = fit_and_test(rs_x)
results["Random Search SVR"] = regression_metrics(data.y_test, preds["Random Search SVR"])
results["Cuckoo-SVR"] = regression_metrics(data.y_test, preds["Cuckoo-SVR"])
joblib.dump({"model": cs_model, "params": cs_params, "features": np.array(data.feature_names)[cs_mask].tolist()},
            OUT / "cuckoo_svr.joblib")
display(Markdown("### All models, test set")); display(pd.DataFrame(results).T.round(4))

cmp_default = compare(results["Default SVR"], results["Cuckoo-SVR"])
display(Markdown("### Default SVR (no optimisation) vs Cuckoo-optimised SVR")); display(cmp_default.round(4))
cmp_rs = compare(results["Random Search SVR"], results["Cuckoo-SVR"], names=("Random Search SVR", "Cuckoo-SVR"))
display(Markdown("### Random Search vs Cuckoo Search (same budget)")); display(cmp_rs.round(4))

display(Markdown("### Configurations"))
display(pd.DataFrame({
    "Default SVR": {**{k: str(v) for k, v in DEFAULT_PARAMS.items()}, "features used": data.n_features},
    "Random Search SVR": {**{k: f"{v:.4g}" for k, v in rs_params.items()}, "features used": int(rs_mask.sum())},
    "Cuckoo-SVR": {**{k: f"{v:.4g}" for k, v in cs_params.items()}, "features used": int(cs_mask.sum())},
}))
""")
code("""
plots.metric_bars({k: results[k] for k in ["Linear Regression", "Default SVR", "Random Search SVR", "Cuckoo-SVR"]},
                  OUT / "metric_comparison.png")
plots.predicted_vs_actual(data.y_test, {k: preds[k] for k in ["Default SVR", "Cuckoo-SVR"]}, OUT / "predicted_vs_actual.png")
plots.residuals(data.y_test, {k: preds[k] for k in ["Default SVR", "Cuckoo-SVR"]}, OUT / "residuals.png")
for f in ["metric_comparison", "predicted_vs_actual", "residuals"]:
    display(Image(OUT / f"{f}.png"))
""")

md("""
## 8. Learning curves

RMSE on the training folds and on the held-out folds as the training set grows (5-fold CV on the training set,
the test set is not used). A gap between the two curves shows overfitting, a high pair of curves shows underfitting.
""")
code("""
from sklearn.model_selection import KFold, learning_curve
t = time.time()
lc = {}
for name, model, cols in [("Default SVR", build_svr(**DEFAULT_PARAMS), np.ones(data.n_features, bool)),
                          ("Cuckoo-SVR", build_svr(**cs_params), cs_mask)]:
    sizes, tr, va = learning_curve(model, data.X_train[:, cols], data.y_train,
                                   train_sizes=np.linspace(0.1, 1.0, 8),
                                   cv=KFold(5, shuffle=True, random_state=CONFIG["split_seed"]),
                                   scoring="neg_root_mean_squared_error", n_jobs=-1)
    lc[name] = (sizes, -tr, -va)
timings["learning_curves_s"] = time.time() - t
lc_df = pd.concat({name: pd.DataFrame({"train_size": s, "train_RMSE": tr.mean(1), "validation_RMSE": va.mean(1),
                                       "validation_std": va.std(1)}) for name, (s, tr, va) in lc.items()})
lc_df.to_csv(OUT / "learning_curves.csv")
display(lc_df.round(4))
plots.learning_curves(lc, OUT / "learning_curves.png")
display(Image(OUT / "learning_curves.png"))
""")

md("## 9. Robustness: 10 independent runs\nMetaheuristics are stochastic, so a single run can be lucky. Both optimisers are re-run with 10 different seeds (same budget each time) and every resulting model is scored on the test set.")
code("""
t = time.time()
per_seed, feature_counts = [], np.zeros(data.n_features)
for s in CONFIG["robustness_seeds"]:
    cs_s, obj_s, x_s, f_s = run_cuckoo(s, verbose=False)
    _, xr_s, fr_s = run_random(s, obj_s.n_evals, verbose=False)
    _, _, mask_s, p_cs = fit_and_test(x_s)
    _, _, _, p_rs = fit_and_test(xr_s)
    feature_counts += mask_s
    row = {"seed": s, "Cuckoo Search": regression_metrics(data.y_test, p_cs)["RMSE"],
           "Random Search": regression_metrics(data.y_test, p_rs)["RMSE"],
           "CS_cv": f_s, "RS_cv": fr_s, "CS_R2": regression_metrics(data.y_test, p_cs)["R2"],
           "CS_features": int(mask_s.sum())}
    per_seed.append(row)
    print(f"seed {s:2d}: Cuckoo {row['Cuckoo Search']:.4f} | Random {row['Random Search']:.4f} | "
          f"{'Cuckoo' if row['Cuckoo Search'] < row['Random Search'] else 'Random'} better", flush=True)
timings["robustness_s"] = time.time() - t
seeds_df = pd.DataFrame(per_seed).set_index("seed")
seeds_df.to_csv(OUT / "robustness_seeds.csv")
""")
code("""
d_rmse = results["Default SVR"]["RMSE"]
summary = pd.DataFrame({
    "mean test RMSE": seeds_df[["Cuckoo Search", "Random Search"]].mean(),
    "std": seeds_df[["Cuckoo Search", "Random Search"]].std(),
    "best": seeds_df[["Cuckoo Search", "Random Search"]].min(),
    "worst": seeds_df[["Cuckoo Search", "Random Search"]].max(),
    "runs beating default SVR": (seeds_df[["Cuckoo Search", "Random Search"]] < d_rmse).sum().astype(str) + f"/{len(seeds_df)}",
})
summary["mean improvement over default %"] = 100 * (d_rmse - summary["mean test RMSE"]) / d_rmse
display(summary.round(4))
wins = int((seeds_df["Cuckoo Search"] < seeds_df["Random Search"]).sum())
stat_rs = wilcoxon(seeds_df["Cuckoo Search"], seeds_df["Random Search"], alternative="less")
stat_def = wilcoxon(seeds_df["Cuckoo Search"] - d_rmse, alternative="less")
print(f"Cuckoo Search beat Random Search in {wins}/{len(seeds_df)} runs")
print(f"Wilcoxon signed-rank, Cuckoo < Random Search : p = {stat_rs.pvalue:.4f}")
print(f"Wilcoxon signed-rank, Cuckoo < Default SVR   : p = {stat_def.pvalue:.4f}")
plots.seed_boxplot(seeds_df[["Cuckoo Search", "Random Search"]].to_dict("list"), d_rmse, OUT / "robustness_boxplot.png")
plots.selected_features(data.feature_names, feature_counts, len(seeds_df), OUT / "feature_selection_frequency.png")
display(Image(OUT / "robustness_boxplot.png")); display(Image(OUT / "feature_selection_frequency.png"))
""")

md("## 10. Save all results")
code("""
timings["total_s"] = time.time() - T0
record = {
    "run_finished": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": CONFIG, "preprocessing": data.report,
    "cuckoo_evaluations": cs_obj.n_evals,
    "default_params": DEFAULT_PARAMS,
    "cuckoo": {"params": cs_params, "features": np.array(data.feature_names)[cs_mask].tolist(), "cv_rmse": cs_cv},
    "random_search": {"params": rs_params, "features": np.array(data.feature_names)[rs_mask].tolist(), "cv_rmse": rs_cv},
    "test_metrics": results,
    "improvement_over_default_%": cmp_default["Improvement_%"].to_dict(),
    "improvement_over_random_search_%": cmp_rs["Improvement_%"].to_dict(),
    "cuckoo_history_best": [float(v) for v in cs.history],
    "cuckoo_history_mean": [float(v) for v in cs.mean_history],
    "robustness": {"summary": summary.to_dict(), "cuckoo_wins": wins,
                   "wilcoxon_p_vs_random": float(stat_rs.pvalue), "wilcoxon_p_vs_default": float(stat_def.pvalue)},
    "benchmark_functions": bench_df.to_dict("records"),
    "learning_curves": {name: {"train_sizes": s.tolist(), "train_rmse": tr.mean(1).tolist(),
                               "validation_rmse": va.mean(1).tolist()} for name, (s, tr, va) in lc.items()},
    "timings_s": timings,
}
(OUT / "results.json").write_text(json.dumps(record, indent=2, default=str))
pd.DataFrame(results).T.to_csv(OUT / "test_metrics.csv")
cmp_default.to_csv(OUT / "comparison_default_vs_cuckoo.csv")
cmp_rs.to_csv(OUT / "comparison_random_vs_cuckoo.csv")
summary.to_csv(OUT / "robustness_summary.csv")
print(f"Total run time: {timings['total_s']/60:.1f} min | finished {record['run_finished']}")
print("\\nSaved files:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:38s} {f.stat().st_size/1024:8.1f} KB")
""")

md("## 11. Summary")
code("""
r = cmp_default.loc
print(f"Test RMSE : default SVR {r['RMSE','Default SVR']:.3f} -> Cuckoo-SVR {r['RMSE','Cuckoo-SVR']:.3f} ({r['RMSE','Improvement_%']:.1f}% better)")
print(f"Test MAE  : default SVR {r['MAE','Default SVR']:.3f} -> Cuckoo-SVR {r['MAE','Cuckoo-SVR']:.3f} ({r['MAE','Improvement_%']:.1f}% better)")
print(f"Test R2   : default SVR {r['R2','Default SVR']:.3f} -> Cuckoo-SVR {r['R2','Cuckoo-SVR']:.3f}")
print(f"Features  : {data.n_features} -> {int(cs_mask.sum())}")
print(f"Over {len(seeds_df)} runs: Cuckoo mean test RMSE {summary.loc['Cuckoo Search','mean test RMSE']:.3f} "
      f"+/- {summary.loc['Cuckoo Search','std']:.3f} vs Random Search {summary.loc['Random Search','mean test RMSE']:.3f} "
      f"+/- {summary.loc['Random Search','std']:.3f}; Cuckoo won {wins}/{len(seeds_df)} runs")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, sys.argv[1])
print("written", len(cells), "cells")
