"""
server.py - the backend of the KeyPrint web app (Flask).

Run it from the project folder:

    python app/server.py

then open http://127.0.0.1:5000 in your browser.

Beginner note - who does what:
    FRONTEND (app/static/index.html, runs in the browser)
        - records WHEN each key goes down and up
        - draws the screens, charts and the PCA map
    BACKEND (this file, runs in Python)
        - turns key timings into the 31 features      (src/features.py)
        - stores enrollment attempts in data/app_users.csv
        - runs the ML: scaling, KNN, logistic regression, PCA (scikit-learn)
          and the login check                         (src/verifier.py)

    The browser and Python talk through small "API" addresses such as
    /api/enroll and /api/login, sending JSON (text that looks like a Python dict).

Who are the "other people"?
    If data/DSL-StrongPasswordData.csv exists, the app loads 10 CMU volunteers
    (50 attempts each) as background typists. They give KNN and logistic
    regression other people to compare you against, and let us measure FAR
    (how often someone else would get in as you). Without the file, the app
    still works with your group members only.
"""

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import (  # noqa: E402
    APP_USERS_CSV, BACKGROUND_REPS, BACKGROUND_SUBJECTS, CMU_CSV, DATA_DIR,
    FEATURE_COLUMNS, FIGURES_DIR, MIN_ENROLL_SAMPLES, RANDOM_STATE,
)
from src.data import load_app_users, load_cmu  # noqa: E402
from src.features import TypingError, events_to_features  # noqa: E402
from src.verifier import DistanceVerifier, leave_one_out_scores, threshold_from_scores  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
REGISTRY = DATA_DIR / "app_users.json"   # list of typist names (even ones with 0 attempts)
app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
LOCK = threading.Lock()

FIGURE_TITLES = {
    "eda1_samples_per_subject": "EDA 1 · Attempts per person",
    "eda2_feature_distributions": "EDA 2 · Timing distributions",
    "eda3_correlation": "EDA 3 · Feature correlations",
    "eda4_subject_rhythms": "EDA 4 · Everyone's rhythm differs",
    "eda5_pca_preview": "EDA 5 · PCA map of 8 people",
    "exp1_k_and_scaling": "Exp 1 · K and scaling",
    "exp2_pca": "Exp 2 · PCA components",
    "exp3_training_size": "Exp 3 · Training size",
    "exp4_model_comparison": "Exp 4 · Final test comparison",
    "exp4_confusion_matrix": "Exp 4 · Confusion matrix",
    "exp5_eer_by_detector": "Exp 5 · Login detectors (EER)",
    "exp5_far_frr": "Exp 5 · FAR / FRR trade-off",
}

# ---------------------------------------------------------------------------
# Data store
# ---------------------------------------------------------------------------

class Store:
    def __init__(self):
        self.samples = load_app_users()
        self.names = self._load_registry()
        for n in self.samples["subject"].unique():
            if n not in self.names:
                self.names.append(n)
        self.background = self._load_background()
        self.version = 0
        self._model = None

    def _load_registry(self):
        try:
            return list(json.loads(REGISTRY.read_text())["users"])
        except (FileNotFoundError, ValueError, KeyError):
            return []

    def _load_background(self):
        try:
            cmu = load_cmu(CMU_CSV)
        except (FileNotFoundError, ValueError):
            return pd.DataFrame(columns=["subject", "sessionIndex", "rep", *FEATURE_COLUMNS])
        chosen = sorted(cmu["subject"].unique())[:BACKGROUND_SUBJECTS]
        bg = (cmu[cmu["subject"].isin(chosen)]
              .sort_values(["subject", "sessionIndex", "rep"])
              .groupby("subject", group_keys=False).head(BACKGROUND_REPS).copy())
        bg["subject"] = "CMU " + bg["subject"]
        return bg

    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.samples.to_csv(APP_USERS_CSV, index=False)
        REGISTRY.write_text(json.dumps({"users": self.names}, indent=2))
        self.version += 1
        self._model = None

    def user_X(self, name):
        return self.samples.loc[self.samples["subject"] == name, FEATURE_COLUMNS].to_numpy(dtype=float)

    def everyone(self):
        """All samples: your group + CMU background, with a 'source' column."""
        a = self.samples.assign(source="you")
        b = self.background.assign(source="cmu")
        return pd.concat([a, b], ignore_index=True)

    def model(self):
        """
        Fit the shared models on everyone's attempts. Rebuilt only when data changes.

            StandardScaler     -> make all 31 features comparable (z-scores)
            NearestNeighbors   -> used for the KNN vote
            LogisticRegression -> probability for each person (needs 2+ people)
            PCA(2)             -> 2D coordinates for the typist map
        """
        if self._model is not None:
            return self._model
        df = self.everyone()
        if len(df) < 3:
            self._model = {"empty": True}
            return self._model
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = df["subject"].to_numpy()
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        nn = NearestNeighbors().fit(Xs)
        lr = None
        if len(set(y)) >= 2:
            lr = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE).fit(Xs, y)
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs)
        self._model = {"empty": False, "df": df, "y": y, "scaler": scaler, "Xs": Xs,
                       "nn": nn, "lr": lr, "pca": pca, "P": pca.transform(Xs)}
        return self._model


STORE = Store()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bad(msg, code=400):
    return jsonify({"error": msg}), code


def profile_info(name, strictness):
    """Profile + threshold + error rates for one typist (used by several endpoints)."""
    X = STORE.user_X(name)
    out = {"name": name, "n": int(len(X))}
    if len(X) >= 2:
        v = DistanceVerifier("scaled_manhattan").fit(X)
        out["mean"] = v.mean_.round(4).tolist()
        out["spread"] = v.spread_.round(4).tolist()
    if len(X) >= MIN_ENROLL_SAMPLES:
        loo = leave_one_out_scores(X)
        thr = threshold_from_scores(loo, strictness)
        v = DistanceVerifier("scaled_manhattan").fit(X)
        others = STORE.everyone()
        others = others[others["subject"] != name][FEATURE_COLUMNS].to_numpy(dtype=float)
        out["threshold"] = thr
        out["consistency"] = float(np.mean(loo))
        out["frr"] = float((loo > thr).mean())
        out["far"] = float((v.score(others) <= thr).mean()) if len(others) else None
        out["n_others"] = int(len(others))
    return out


def match_percent(score, thr):
    """Turn distance/threshold into a friendly 0-100% (50% exactly at the threshold)."""
    return float(100.0 / (1.0 + np.exp(5.0 * (score / thr - 1.0))))


def read_json():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise TypingError("Request body must be JSON.")
    return data

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/figures/<path:name>")
def figure(name):
    return send_from_directory(FIGURES_DIR, name)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/state")
def state():
    with LOCK:
        counts = STORE.samples.groupby("subject").size().to_dict()
        users = [{"name": n, "n": int(counts.get(n, 0)),
                  "ready": int(counts.get(n, 0)) >= MIN_ENROLL_SAMPLES} for n in STORE.names]
        figs = []
        if FIGURES_DIR.exists():
            for p in sorted(FIGURES_DIR.glob("*.png")):
                figs.append({"file": p.name, "title": FIGURE_TITLES.get(p.stem, p.stem)})
        return jsonify({
            "users": users,
            "min_samples": MIN_ENROLL_SAMPLES,
            "features": FEATURE_COLUMNS,
            "background": {"loaded": len(STORE.background) > 0,
                           "people": int(STORE.background["subject"].nunique()),
                           "samples": int(len(STORE.background))},
            "figures": figs,
        })


@app.post("/api/users")
def add_user():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()[:24]
    if not name:
        return bad("Type a name for the new typist.")
    if name.lower().startswith("cmu "):
        return bad("Names starting with 'CMU' are reserved for the dataset volunteers.")
    with LOCK:
        if name.lower() in (n.lower() for n in STORE.names):
            return bad("That name is already taken.")
        STORE.names.append(name)
        STORE.save()
    return jsonify({"ok": True, "name": name})


@app.delete("/api/users/<name>")
def delete_user(name):
    with LOCK:
        if name not in STORE.names:
            return bad("No typist with that name.", 404)
        STORE.names.remove(name)
        STORE.samples = STORE.samples[STORE.samples["subject"] != name].reset_index(drop=True)
        STORE.save()
    return jsonify({"ok": True})


@app.get("/api/profile")
def profile():
    name = request.args.get("name", "")
    z = float(request.args.get("z", 2.0))
    with LOCK:
        if name not in STORE.names:
            return bad("No typist with that name.", 404)
        return jsonify(profile_info(name, z))


@app.post("/api/enroll")
def enroll():
    try:
        data = read_json()
        name = data.get("name")
        feats = events_to_features(data.get("events"))
    except TypingError as e:
        return bad(str(e))
    with LOCK:
        if name not in STORE.names:
            return bad("Pick or add a typist first.")
        n = int((STORE.samples["subject"] == name).sum())
        row = {"subject": name, "sessionIndex": 0, "rep": n + 1, **feats}
        STORE.samples = pd.concat([STORE.samples, pd.DataFrame([row])], ignore_index=True)
        STORE.save()
        info = profile_info(name, float(data.get("z", 2.0)))
    return jsonify({"ok": True, "features": [feats[c] for c in FEATURE_COLUMNS], "profile": info})


@app.post("/api/login")
def login():
    """
    The login check. Nothing is saved: the attempt is only compared.

    1. Distance verifier: distance from the claimed person's profile vs their threshold.
    2. KNN: who owns the K most similar stored attempts?
    3. Logistic regression: probability that it is the claimed person.
    4. PCA: where does this attempt land on the 2D map?
    """
    try:
        data = read_json()
        claim = data.get("claim")
        feats = events_to_features(data.get("events"))
        k = int(data.get("k", 5))
        z = float(data.get("z", 2.0))
    except (TypingError, ValueError) as e:
        return bad(str(e))
    x = np.array([feats[c] for c in FEATURE_COLUMNS], dtype=float)
    with LOCK:
        if claim not in STORE.names:
            return bad("Pick who you are logging in as.")
        info = profile_info(claim, z)
        if "threshold" not in info:
            return bad(f"{claim} needs at least {MIN_ENROLL_SAMPLES} enrolled attempts first.")
        v = DistanceVerifier("scaled_manhattan").fit(STORE.user_X(claim))
        score = float(v.score(x)[0])
        thr = info["threshold"]
        result = {
            "claim": claim, "features": x.round(4).tolist(), "profile": info,
            "score": score, "threshold": thr, "match": match_percent(score, thr),
            "granted": bool(score <= thr),
        }
        m = STORE.model()
        if not m["empty"]:
            xs = m["scaler"].transform(x.reshape(1, -1))
            k = max(1, min(k, len(m["Xs"])))
            dist, idx = m["nn"].kneighbors(xs, n_neighbors=k)
            names = m["y"][idx[0]]
            votes = pd.Series(names).value_counts()
            pred = votes.index[0]
            result["knn"] = {
                "k": k, "prediction": pred, "votes": int(votes.iloc[0]),
                "neighbours": [{"user": str(names[i]), "distance": float(dist[0][i]),
                                "x": float(m["P"][idx[0][i]][0]), "y": float(m["P"][idx[0][i]][1])}
                               for i in range(k)],
            }
            if m["lr"] is not None:
                probs = m["lr"].predict_proba(xs)[0]
                classes = list(m["lr"].classes_)
                order = np.argsort(probs)[::-1][:5]
                result["logreg"] = {
                    "p_claim": float(probs[classes.index(claim)]) if claim in classes else None,
                    "top": [{"user": classes[i], "p": float(probs[i])} for i in order],
                }
            p = m["pca"].transform(xs)[0]
            result["pca"] = {"x": float(p[0]), "y": float(p[1])}
    return jsonify(result)


@app.get("/api/pca")
def pca_points():
    with LOCK:
        m = STORE.model()
        if m["empty"]:
            return jsonify({"points": [], "explained": [0, 0]})
        src = m["df"]["source"].to_numpy()
        pts = [{"x": float(p[0]), "y": float(p[1]), "user": str(u), "source": str(s)}
               for p, u, s in zip(m["P"], m["y"], src)]
        return jsonify({"points": pts, "explained": m["pca"].explained_variance_ratio_.round(4).tolist()})


@app.get("/api/export.csv")
def export_csv():
    with LOCK:
        csv = STORE.samples[["subject", "sessionIndex", "rep", *FEATURE_COLUMNS]].to_csv(index=False)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=keyprint_group_data.csv"})


if __name__ == "__main__":
    # On your laptop this is http://127.0.0.1:5000 .
    # A hosting site (like Render) sets PORT itself; there we start with gunicorn instead.
    port = int(os.environ.get("PORT", 5000))
    bg = STORE.background
    print(f"\n  KeyPrint is running.  Open  http://127.0.0.1:{port}  in your browser.")
    if len(bg):
        print(f"  Loaded {bg['subject'].nunique()} CMU volunteers as background typists.")
    else:
        print("  CMU dataset not found in data/, so only your group's typists will be used.")
    print("  Press Ctrl+C to stop.\n")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)
