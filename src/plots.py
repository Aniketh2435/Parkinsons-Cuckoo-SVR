"""All figures for the report. Each function saves one PNG into the output folder."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COLORS = {"Mean predictor": "#c4c7c5", "Linear Regression": "#9aa0a6", "Default SVR": "#e8710a",
          "Random Search": "#9334e6", "Cuckoo Search": "#1a73e8", "Cuckoo-SVR": "#1a73e8",
          "Random Search SVR": "#9334e6"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def target_distribution(y, path, name="motor UPDRS"):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(y, bins=40, color="#1a73e8", alpha=0.85)
    ax.set(xlabel=name, ylabel="Recordings", title=f"Distribution of {name}")
    _save(fig, path)


def correlation_heatmap(frame, path):
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(frame.corr(), cmap="coolwarm", center=0, ax=ax, annot=True, fmt=".1f",
                annot_kws={"size": 6})
    ax.set_title("Feature correlation matrix")
    _save(fig, path)


def convergence(cs_log, rs_log, cs_best, cs_mean, path):
    """Left: best CV RMSE so far per evaluation (same budget). Right: CS population per generation."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, log in [("Cuckoo Search", cs_log), ("Random Search", rs_log)]:
        a.plot([e["eval"] for e in log], [e["best_so_far"] for e in log], label=name, color=COLORS[name], lw=2)
    a.set(xlabel="Fitness evaluations", ylabel="Best CV RMSE so far",
          title="Convergence: Cuckoo Search vs Random Search")
    a.legend()
    gens = range(len(cs_best))
    b.plot(gens, cs_best, marker="o", color=COLORS["Cuckoo Search"], lw=2, label="Best nest")
    b.plot(gens, cs_mean, marker="s", color=COLORS["Cuckoo Search"], lw=1.2, ls="--", alpha=0.6,
           label="Population mean")
    b.set(xlabel="Generation", ylabel="CV RMSE", title="Cuckoo Search: fitness per generation")
    b.legend()
    _save(fig, path)


def metric_bars(results, path, metrics=("RMSE", "MAE", "R2")):
    names = list(results)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        bars = ax.bar(names, [results[n][m] for n in names], color=[COLORS[n] for n in names])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_title(f"Test {m}" + (" (higher is better)" if m == "R2" else " (lower is better)"))
        ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def predicted_vs_actual(y_true, preds, path):
    """One scatter panel per model, with the ideal y = x line."""
    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.6), sharey=True)
    lo, hi = y_true.min() - 2, y_true.max() + 2
    for ax, (name, p) in zip(np.atleast_1d(axes), preds.items()):
        ax.scatter(y_true, p, s=5, alpha=0.35, color=COLORS[name])
        ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual motor UPDRS", title=name)
    np.atleast_1d(axes)[0].set_ylabel("Predicted motor UPDRS")
    _save(fig, path)


def residuals(y_true, preds, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, p in preds.items():
        ax.hist(y_true - p, bins=50, alpha=0.55, color=COLORS[name], label=name)
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="Residual (actual - predicted)", ylabel="Recordings", title="Test-set residuals")
    ax.legend()
    _save(fig, path)


def selected_features(feature_names, counts, n_runs, path):
    """How often Cuckoo Search kept each feature across independent runs."""
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(np.array(feature_names)[order], np.array(counts)[order] / n_runs * 100, color="#1a73e8")
    ax.set(xlabel="Selected in % of runs", title=f"Features chosen by Cuckoo Search ({n_runs} runs)", xlim=(0, 105))
    _save(fig, path)


def benchmark_convergence(curves, path):
    """One panel per test function: median best-so-far over seeds, shaded between min and max.
    `curves[function][method]` is an array of shape (seeds, evaluations)."""
    styles = {"Cuckoo Search": (COLORS["Cuckoo Search"], "-"),
              "Cuckoo Search (random-nest replacement)": ("#5f6368", ":"),
              "Random Search": (COLORS["Random Search"], "--")}
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.8))
    for ax, (fname, methods) in zip(axes, curves.items()):
        for method, runs in methods.items():
            color, ls = styles[method]
            x = np.arange(1, runs.shape[1] + 1)
            ax.plot(x, np.median(runs, axis=0), color=color, ls=ls, lw=2, label=method)
            ax.fill_between(x, runs.min(axis=0), runs.max(axis=0), color=color, alpha=0.12)
        ax.set(yscale="log", xlabel="Fitness evaluations", title=fname)
    axes[0].set_ylabel("Best f(x) so far (log scale)")
    axes[0].legend(fontsize=8)
    _save(fig, path)


def learning_curves(curves, path):
    """Train and cross-validation RMSE against training-set size, one line pair per model.
    `curves[model] = (train_sizes, train_rmse, val_rmse)` with RMSE arrays of shape (sizes, folds)."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, (sizes, tr, va) in curves.items():
        c = COLORS[name]
        ax.plot(sizes, tr.mean(axis=1), color=c, ls="--", marker="o", ms=4, label=f"{name}: training")
        ax.plot(sizes, va.mean(axis=1), color=c, marker="s", ms=4, lw=2, label=f"{name}: validation")
        ax.fill_between(sizes, va.mean(axis=1) - va.std(axis=1), va.mean(axis=1) + va.std(axis=1), color=c, alpha=0.15)
    ax.set(xlabel="Training recordings", ylabel="RMSE (motor UPDRS)", title="Learning curves (5-fold CV)")
    ax.legend(fontsize=8)
    _save(fig, path)


def seed_boxplot(per_seed, default_rmse, path):
    """Test RMSE of each optimiser over independent seeds, against the default SVR."""
    df = pd.DataFrame(per_seed)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names = ["Cuckoo Search", "Random Search"]
    ax.boxplot([df[n] for n in names], tick_labels=names, widths=0.5, patch_artist=True,
               boxprops={"facecolor": "#e8f0fe"}, medianprops={"color": "black"})
    for k, n in enumerate(names, start=1):
        ax.scatter(np.full(len(df), k) + np.random.default_rng(0).uniform(-0.08, 0.08, len(df)),
                   df[n], color=COLORS[n], s=22, zorder=3)
    ax.axhline(default_rmse, color=COLORS["Default SVR"], ls="--", lw=1.5, label=f"Default SVR ({default_rmse:.3f})")
    ax.set(ylabel="Test RMSE", title=f"Robustness over {len(df)} independent runs")
    ax.legend()
    _save(fig, path)
