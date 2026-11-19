"""
experiments.py - every experiment in the project, as small functions.

You can run everything in one go from the project folder:

    python -m src.experiments

That reads data/DSL-StrongPasswordData.csv and writes
    results/tables/*.csv   (the numbers)
    results/figures/*.png  (the graphs)

The notebook (notebooks/KeyPrint_Analysis.ipynb) calls the same functions
step by step with explanations.

Beginner glossary for this file:
    Pipeline        : a chain of steps, e.g. "scale the numbers, then run KNN".
                      Using a Pipeline guarantees the scaler only learns from
                      TRAINING data (this prevents "data leakage").
    StandardScaler  : z-score scaling: (value - mean) / std, per feature.
    KNeighborsClassifier : KNN. Finds the K most similar stored attempts and votes.
    LogisticRegression   : learns a weight per feature, outputs probabilities.
    PCA             : squeezes the 31 features into fewer "summary" features.
"""

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import plots
from .config import CMU_CSV, FEATURE_COLUMNS, RANDOM_STATE, RESULTS_DIR, TABLES_DIR
from .data import X_y, load_cmu, split_by_session
from .evaluate import equal_error_rate, identification_scores
from .verifier import DistanceVerifier

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def make_knn(k=1, scaled=True, n_components=None):
    """KNN, optionally with scaling and PCA in front of it."""
    steps = []
    if scaled:
        steps.append(("scale", StandardScaler()))
    if n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=k)))
    return Pipeline(steps)


def make_logreg(scaled=True):
    """Logistic regression. With 51 people it learns 51 sets of weights (multinomial)."""
    steps = []
    if scaled:
        steps.append(("scale", StandardScaler()))
    steps.append(("logreg", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)))
    return Pipeline(steps)


def fit_and_score(model, X_train, y_train, X_eval, y_eval, timing_repeats=3):
    """
    Train a model, predict, and return scores plus how long it took.

    Prediction time is measured `timing_repeats` times and we keep the median,
    because a single timing on a laptop can jump around (other apps running).
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    times = []
    for _ in range(timing_repeats):
        t1 = time.perf_counter()
        y_pred = model.predict(X_eval)
        times.append(time.perf_counter() - t1)
    out = identification_scores(y_eval, y_pred)
    out["fit_seconds"] = fit_seconds
    out["predict_seconds"] = float(np.median(times))
    return out, y_pred


# ---------------------------------------------------------------------------
# Experiment 1: K values, with and without scaling (on the validation set)
# ---------------------------------------------------------------------------

def exp_k_and_scaling(train, val, ks=(1, 3, 5, 7, 9, 15, 25)):
    Xtr, ytr = X_y(train)
    Xva, yva = X_y(val)
    rows = []
    for scaled in (False, True):
        for k in ks:
            res, _ = fit_and_score(make_knn(k, scaled), Xtr, ytr, Xva, yva)
            rows.append({"k": k, "scaled": scaled, **res})
    return pd.DataFrame(rows)


def best_k(k_table):
    """Choose K with the best VALIDATION macro-F1 (scaled). Ties -> smaller K."""
    t = k_table[k_table["scaled"]].sort_values(["macro_f1", "k"], ascending=[False, True])
    return int(t.iloc[0]["k"])


# ---------------------------------------------------------------------------
# Experiment 2: PCA - how many components do we need?
# ---------------------------------------------------------------------------

def exp_pca(train, val, k, components=(2, 3, 5, 8, 10, 15, 20, 25, 31)):
    Xtr, ytr = X_y(train)
    Xva, yva = X_y(val)
    # Fit PCA once on scaled training data just to read the explained variance.
    scaler = StandardScaler().fit(Xtr)
    full = PCA(random_state=RANDOM_STATE).fit(scaler.transform(Xtr))
    cumvar = np.cumsum(full.explained_variance_ratio_)
    rows = []
    for c in components:
        res, _ = fit_and_score(make_knn(k, True, n_components=c), Xtr, ytr, Xva, yva)
        rows.append({"components": c, "cumulative_variance": float(cumvar[c - 1]), **res})
    return pd.DataFrame(rows), full.explained_variance_ratio_


def best_pca(pca_table, tolerance=0.01):
    """Smallest number of components whose macro-F1 is within `tolerance` of the best."""
    top = pca_table["macro_f1"].max()
    ok = pca_table[pca_table["macro_f1"] >= top - tolerance]
    return int(ok["components"].min())


# ---------------------------------------------------------------------------
# Experiment 3: how many training attempts per person do we need?
# ---------------------------------------------------------------------------

def exp_training_size(train, val, k, sizes=(5, 10, 20, 50, 100, 250)):
    Xva, yva = X_y(val)
    train = train.sort_values(["subject", "sessionIndex", "rep"])
    rows = []
    for n in sizes:
        subset = train.groupby("subject", group_keys=False).head(n)
        Xs, ys = X_y(subset)
        for name, model in (("KNN (scaled)", make_knn(min(k, n), True)),
                            ("Logistic regression", make_logreg(True))):
            res, _ = fit_and_score(model, Xs, ys, Xva, yva)
            rows.append({"per_person": n, "model": name, **res})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiment 4: final comparison on the TEST set (used once)
# ---------------------------------------------------------------------------

def final_test(trainval, test, k, n_components):
    Xtr, ytr = X_y(trainval)
    Xte, yte = X_y(test)
    configs = [
        ("Baseline: KNN K=1, no scaling", make_knn(1, scaled=False)),
        (f"KNN K={k}, scaled", make_knn(k, scaled=True)),
        (f"PCA ({n_components}) + KNN K={k}", make_knn(k, scaled=True, n_components=n_components)),
        ("Logistic regression, no scaling", make_logreg(scaled=False)),
        ("Logistic regression, scaled", make_logreg(scaled=True)),
    ]
    rows, preds = [], {}
    for name, model in configs:
        res, y_pred = fit_and_score(model, Xtr, ytr, Xte, yte)
        rows.append({"model": name, **res})
        preds[name] = y_pred
    return pd.DataFrame(rows), yte, preds


def most_confused_pairs(y_true, y_pred, top=10):
    """Which people get mistaken for each other most often?"""
    wrong = pd.DataFrame({"true": y_true, "predicted": y_pred})
    wrong = wrong[wrong["true"] != wrong["predicted"]]
    counts = wrong.value_counts().reset_index(name="count")
    return counts.head(top)


# ---------------------------------------------------------------------------
# Experiment 5: verification (login) with three distance detectors
# ---------------------------------------------------------------------------

def exp_verification(trainval, test, kinds=("euclidean", "manhattan", "scaled_manhattan"),
                     impostor_reps=5):
    """
    For every person:
      - build their profile from sessions 1-6
      - GENUINE tries  = their own attempts from sessions 7-8
      - IMPOSTOR tries = the first `impostor_reps` test attempts of every OTHER person
    Then compute the EER for that person. Repeat for each distance type.
    """
    rows = []
    pooled = {kind: {"genuine": [], "impostor": []} for kind in kinds}
    subjects = sorted(trainval["subject"].unique())
    test_sorted = test.sort_values(["subject", "sessionIndex", "rep"])
    first_reps = test_sorted.groupby("subject", group_keys=False).head(impostor_reps)
    for s in subjects:
        X_enroll = trainval.loc[trainval["subject"] == s, FEATURE_COLUMNS].to_numpy()
        X_gen = test.loc[test["subject"] == s, FEATURE_COLUMNS].to_numpy()
        X_imp = first_reps.loc[first_reps["subject"] != s, FEATURE_COLUMNS].to_numpy()
        for kind in kinds:
            v = DistanceVerifier(kind).fit(X_enroll)
            g, imp = v.score(X_gen), v.score(X_imp)
            e = equal_error_rate(g, imp)
            rows.append({"subject": s, "detector": kind, "eer": e["eer"]})
            # Normalise by the threshold so different people can be pooled in one plot.
            pooled[kind]["genuine"].extend(g / e["threshold"])
            pooled[kind]["impostor"].extend(imp / e["threshold"])
    per_subject = pd.DataFrame(rows)
    summary = (per_subject.groupby("detector")["eer"]
               .agg(mean_eer="mean", std_eer="std", worst_eer="max").reset_index()
               .sort_values("mean_eer"))
    return per_subject, summary, pooled


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def save_environment():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "random_state": RANDOM_STATE,
    }
    (RESULTS_DIR / "environment.json").write_text(json.dumps(info, indent=2))
    return info


def run_all(csv_path=CMU_CSV, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    plots.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_cmu(csv_path)
    train, val, test = split_by_session(df)
    trainval = pd.concat([train, val])
    say(f"Loaded {len(df)} rows, {df['subject'].nunique()} people. "
        f"Train {len(train)} / Val {len(val)} / Test {len(test)}")

    say("EDA figures ...")
    plots.eda_samples_per_subject(df)
    plots.eda_feature_distributions(df)
    plots.eda_correlation(df)
    plots.eda_subject_rhythms(df)
    plots.eda_pca_preview(df)

    say("Experiment 1: K and scaling ...")
    k_table = exp_k_and_scaling(train, val)
    k_table.to_csv(TABLES_DIR / "exp1_k_and_scaling.csv", index=False)
    k = best_k(k_table)
    plots.plot_k_scaling(k_table, k)
    say(f"  best K on validation = {k}")

    say("Experiment 2: PCA components ...")
    pca_table, evr = exp_pca(train, val, k)
    pca_table.to_csv(TABLES_DIR / "exp2_pca.csv", index=False)
    n_comp = best_pca(pca_table)
    plots.plot_pca_experiment(pca_table, evr, n_comp)
    say(f"  chosen PCA components = {n_comp}")

    say("Experiment 3: training size ...")
    size_table = exp_training_size(train, val, k)
    size_table.to_csv(TABLES_DIR / "exp3_training_size.csv", index=False)
    plots.plot_training_size(size_table)

    say("Experiment 4: final test ...")
    final, y_test, preds = final_test(trainval, test, k, n_comp)
    final.to_csv(TABLES_DIR / "exp4_final_test.csv", index=False)
    plots.plot_model_comparison(final)
    best_name = final.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    plots.plot_confusion(y_test, preds[best_name], best_name)
    confused = most_confused_pairs(y_test, preds[best_name])
    confused.to_csv(TABLES_DIR / "exp4_most_confused_pairs.csv", index=False)

    say("Experiment 5: verification (FAR / FRR / EER) ...")
    per_subject, ver_summary, pooled = exp_verification(trainval, test)
    per_subject.to_csv(TABLES_DIR / "exp5_verification_per_subject.csv", index=False)
    ver_summary.to_csv(TABLES_DIR / "exp5_verification_summary.csv", index=False)
    plots.plot_eer_by_detector(per_subject)
    plots.plot_far_frr(pooled["scaled_manhattan"]["genuine"], pooled["scaled_manhattan"]["impostor"])

    env = save_environment()
    summary = {"best_k": k, "pca_components": n_comp, "best_model": best_name,
               "final_test": final.to_dict(orient="records"),
               "verification": ver_summary.to_dict(orient="records"), "environment": env}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    say("\nFinal test results:")
    say(final[["model", "accuracy", "macro_f1", "fit_seconds", "predict_seconds"]].to_string(index=False))
    say("\nVerification (lower EER is better):")
    say(ver_summary.to_string(index=False))
    say(f"\nSaved tables to {TABLES_DIR} and figures to {plots.FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run every KeyPrint experiment.")
    ap.add_argument("--data", default=str(CMU_CSV), help="path to DSL-StrongPasswordData.csv")
    args = ap.parse_args()
    run_all(args.data)
