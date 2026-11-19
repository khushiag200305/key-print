"""
plots.py - every graph in the project.

Each function draws one figure, saves it as a PNG in results/figures/,
and returns the matplotlib Figure so the notebook can show it.
"""

import sys

import matplotlib

# Outside Jupyter, draw without needing a screen (works on any laptop/server).
# Inside Jupyter, keep the notebook's normal inline plotting.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, FIGURES_DIR, RANDOM_STATE

# One consistent look for every figure.
INK = "#1d2430"
MUTED = "#6b7686"
GRID = "#e6e9ee"
COLORS = {"H": "#e08a00", "DD": "#0f8fb3", "UD": "#7a5cc7"}
SERIES = ["#0f8fb3", "#e08a00", "#7a5cc7", "#2e9e5b", "#d1495b", "#3b5bdb", "#8a6d3b", "#c2255c"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False,
})


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    return fig


def _kind(col):
    return col.split(".")[0]


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def eda_samples_per_subject(df):
    counts = df.groupby("subject").size().sort_index()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.bar(range(len(counts)), counts.values, color=SERIES[0], width=0.75)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90, fontsize=7)
    ax.set_ylabel("attempts")
    ax.set_title("EDA 1 · Attempts per person (class balance)")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "eda1_samples_per_subject")


def eda_feature_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    names = {"H": "Hold time (H)", "DD": "Press → press (DD)", "UD": "Release → press (UD)"}
    for ax, kind in zip(axes, ["H", "DD", "UD"]):
        cols = [c for c in FEATURE_COLUMNS if _kind(c) == kind]
        vals = df[cols].to_numpy().ravel() * 1000
        lo, hi = np.percentile(vals, [0.5, 99.5])
        ax.hist(vals[(vals >= lo) & (vals <= hi)], bins=60, color=COLORS[kind], alpha=0.9)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(names[kind])
        ax.set_xlabel("milliseconds")
    axes[0].set_ylabel("count")
    fig.suptitle("EDA 2 · How the three kinds of timing are distributed", fontweight="bold", y=1.04)
    return _save(fig, "eda2_feature_distributions")


def eda_correlation(df):
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(9, 7.8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=6.5)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="correlation")
    ax.set_title("EDA 3 · Correlation between the 31 features")
    return _save(fig, "eda3_correlation")


def eda_subject_rhythms(df, subjects=None):
    if subjects is None:
        subjects = sorted(df["subject"].unique())[:4]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(FEATURE_COLUMNS))
    for i, s in enumerate(subjects):
        part = df.loc[df["subject"] == s, FEATURE_COLUMNS] * 1000
        m, sd = part.mean().to_numpy(), part.std().to_numpy()
        ax.plot(x, m, marker="o", ms=3, lw=1.6, color=SERIES[i], label=s)
        ax.fill_between(x, m - sd, m + sd, color=SERIES[i], alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=7)
    ax.set_ylabel("milliseconds (mean ± 1 std)")
    ax.set_title("EDA 4 · Each person has their own typing rhythm")
    ax.legend(ncol=len(subjects), loc="upper right")
    return _save(fig, "eda4_subject_rhythms")


def eda_pca_preview(df, n_subjects=8):
    subjects = sorted(df["subject"].unique())[:n_subjects]
    part = df[df["subject"].isin(subjects)]
    Xs = StandardScaler().fit_transform(part[FEATURE_COLUMNS])
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs)
    P = pca.transform(Xs)
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for i, s in enumerate(subjects):
        m = (part["subject"] == s).to_numpy()
        ax.scatter(P[m, 0], P[m, 1], s=7, alpha=0.55, color=SERIES[i % len(SERIES)], label=s)
    v = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({v[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({v[1]:.0f}% of variance)")
    ax.set_title(f"EDA 5 · PCA map of {n_subjects} people (each dot = one attempt)")
    ax.legend(markerscale=2.5, ncol=2, fontsize=8)
    return _save(fig, "eda5_pca_preview")


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def plot_k_scaling(k_table, chosen_k):
    fig, ax = plt.subplots(figsize=(7, 4))
    for scaled, color, label in ((False, MUTED, "without scaling"), (True, SERIES[0], "with scaling (z-score)")):
        t = k_table[k_table["scaled"] == scaled].sort_values("k")
        ax.plot(t["k"], t["accuracy"] * 100, marker="o", color=color, lw=2, label=label)
    ax.axvline(chosen_k, color=SERIES[1], ls="--", lw=1)
    ax.text(chosen_k, ax.get_ylim()[0], f"  chosen K={chosen_k}", color=SERIES[1], va="bottom", fontsize=9)
    ax.set_xlabel("K (number of neighbours)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Experiment 1 · KNN accuracy vs K, with and without scaling")
    ax.legend()
    return _save(fig, "exp1_k_and_scaling")


def plot_pca_experiment(pca_table, evr, chosen):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    c = np.arange(1, len(evr) + 1)
    axes[0].plot(c, np.cumsum(evr) * 100, marker="o", ms=3, color=SERIES[2])
    axes[0].axhline(90, color=MUTED, ls=":", lw=1)
    axes[0].set_xlabel("number of components")
    axes[0].set_ylabel("cumulative variance kept (%)")
    axes[0].set_title("Information kept")
    axes[1].plot(pca_table["components"], pca_table["accuracy"] * 100, marker="o", color=SERIES[0])
    axes[1].set_xlabel("number of components")
    axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[2].plot(pca_table["components"], pca_table["predict_seconds"] * 1000, marker="o", color=SERIES[1])
    axes[2].set_xlabel("number of components")
    axes[2].set_ylabel("prediction time (ms)")
    axes[2].set_title("Speed")
    for ax in axes[1:]:
        ax.axvline(chosen, color=SERIES[1], ls="--", lw=1)
    fig.suptitle(f"Experiment 2 · PCA before KNN (chosen: {chosen} components)", fontweight="bold", y=1.04)
    return _save(fig, "exp2_pca")


def plot_training_size(size_table):
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, t) in enumerate(size_table.groupby("model")):
        ax.plot(t["per_person"], t["accuracy"] * 100, marker="o", lw=2, color=SERIES[i], label=name)
    ax.set_xscale("log")
    ax.set_xticks(sorted(size_table["per_person"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("training attempts per person (log scale)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Experiment 3 · How much enrollment data is needed?")
    ax.legend()
    return _save(fig, "exp3_training_size")


def plot_model_comparison(final):
    t = final.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(t))
    colors = [MUTED if n.startswith("Baseline") else SERIES[0] for n in t["model"]]
    ax.barh(y, t["macro_f1"] * 100, color=colors)
    for yi, v in zip(y, t["macro_f1"] * 100):
        ax.text(v + 0.5, yi, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(t["model"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("macro-F1 on the TEST set (%)")
    ax.set_title("Experiment 4 · Final comparison (grey = baseline)")
    return _save(fig, "exp4_model_comparison")


def plot_confusion(y_true, y_pred, title):
    labels = sorted(pd.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.grid(False)
    ax.set_xlabel("predicted person")
    ax.set_ylabel("true person")
    fig.colorbar(im, ax=ax, fraction=0.04, label="share of that person's attempts")
    ax.set_title(f"Confusion matrix · {title}")
    return _save(fig, "exp4_confusion_matrix")


def plot_eer_by_detector(per_subject):
    order = per_subject.groupby("detector")["eer"].mean().sort_values().index.tolist()
    data = [per_subject.loc[per_subject["detector"] == d, "eer"] * 100 for d in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, tick_labels=[d.replace("_", " ") for d in order], patch_artist=True, widths=0.5)
    for patch, col in zip(bp["boxes"], SERIES):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
    for i, d in enumerate(data, start=1):
        ax.text(i, d.mean(), f" mean {d.mean():.1f}%", fontsize=9, color=INK, va="center")
    ax.set_ylabel("equal error rate per person (%)")
    ax.set_title("Experiment 5 · Login detectors compared (lower is better)")
    return _save(fig, "exp5_eer_by_detector")


def plot_far_frr(genuine, impostor):
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    t = np.linspace(0, np.percentile(impostor, 99), 300)
    far = np.array([(impostor <= x).mean() for x in t]) * 100
    frr = np.array([(genuine > x).mean() for x in t]) * 100
    i = int(np.argmin(np.abs(far - frr)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(genuine, bins=60, alpha=0.6, color=SERIES[3], label="genuine (real person)", density=True)
    axes[0].hist(impostor, bins=60, alpha=0.6, color=SERIES[4], label="impostor (someone else)", density=True,
                 range=(0, np.percentile(impostor, 99)))
    axes[0].axvline(1.0, color=INK, ls="--", lw=1)
    axes[0].set_xlabel("distance ÷ that person's EER threshold")
    axes[0].set_ylabel("density")
    axes[0].set_title("Distance scores")
    axes[0].legend()
    axes[1].plot(t, far, color=SERIES[4], lw=2, label="FAR: impostors let in")
    axes[1].plot(t, frr, color=SERIES[3], lw=2, label="FRR: real users blocked")
    axes[1].scatter([t[i]], [far[i]], color=INK, zorder=5)
    axes[1].annotate(f"EER ≈ {far[i]:.1f}%", (t[i], far[i]), xytext=(10, 10), textcoords="offset points")
    axes[1].set_xlabel("threshold (normalised)")
    axes[1].set_ylabel("error rate (%)")
    axes[1].set_title("The FAR / FRR trade-off")
    axes[1].legend()
    fig.suptitle("Experiment 5 · Scaled Manhattan login detector, all people pooled", fontweight="bold", y=1.03)
    return _save(fig, "exp5_far_frr")
