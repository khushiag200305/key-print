# How the Code Works

*A tour of every file, in the order the data flows through them.*

---

## The big picture

```
                 ┌──────────── THE STUDY ────────────┐
 data/DSL-...csv │  src/data.py      load + split    │
        │        │  src/experiments.py  KNN, LR, PCA │──▶ results/tables/*.csv
        └──────▶ │  src/verifier.py  login detector  │──▶ results/figures/*.png
                 │  src/evaluate.py  F1, FAR, EER    │
                 └───────────────────────────────────┘
                          ▲ notebook calls the same functions

                 ┌──────────── THE APP ──────────────┐
 browser         │  app/static/index.html            │
 (you type)  ──▶ │   records key down/up times       │
                 │   sends them to Python  ──────────┼──▶ app/server.py
                 │   draws the result   ◀────────────┼──   src/features.py  (31 numbers)
                 └───────────────────────────────────┘     src/verifier.py  (login check)
                                                            scikit-learn     (KNN, LR, PCA)
```

---

## `src/` · the machine learning code

### `config.py` · settings

* **Paths** to the data, results and figures folders.
* `RANDOM_STATE = 42`, the seed that makes results repeatable.
* `FEATURE_COLUMNS`, the 31 feature names built in the exact CMU order: `H.period, DD.period.t, UD.period.t, H.t, …, H.Return`.
* The **session split**: train 1–5, validation 6, test 7–8.
* App settings: 5 attempts needed to log in, and 10 CMU volunteers × 50 attempts as background typists.

### `features.py` · key times → 31 numbers

`events_to_features(events)` receives 11 dicts like `{"key": "t", "down": 1234.5, "up": 1320.1}` (milliseconds) and:
1. checks each key is the right one (`.`, `t`, `i`, … `Enter`)
2. converts milliseconds to seconds
3. computes H = up − down, DD = next down − down, UD = next down − up
4. rejects silly attempts (a key held over 1.5 s, or a pause over 4 s)

This is the **same calculation** the CMU researchers did, so app data and dataset data are directly comparable.

### `data.py` · loading and splitting

* `load_cmu()` reads the CSV and checks every expected column is there.
* `split_by_session(df)` returns train / validation / test using the session rule.
* `X_y(df)` separates the numbers (**X**, 31 columns) from the answers (**y**, the person).

### `verifier.py` · the login detector

* `DistanceVerifier(kind)`:
  * `.fit(X_user)` computes the person's **mean** of each feature and their **spread** (mean absolute deviation).
  * `.score(X)` gives the distance of each attempt from the profile: `euclidean`, `manhattan` or `scaled_manhattan`.
* `leave_one_out_scores(X_user)` measures how far each of a person's own attempts is from a profile built without it.
* `threshold_from_scores(scores, strictness)` returns mean + strictness × std.

### `evaluate.py` · scoring

* `identification_scores()` → accuracy and macro-F1 (from scikit-learn).
* `far_frr()` → FAR and FRR at many thresholds.
* `equal_error_rate()` → the threshold where FAR ≈ FRR, and that error rate.

### `experiments.py` · all experiments

Model builders use scikit-learn **Pipelines** (chains of steps):

```python
make_knn(k=5, scaled=True, n_components=10)
#   StandardScaler  →  PCA(10)  →  KNeighborsClassifier(5)
make_logreg(scaled=True)
#   StandardScaler  →  LogisticRegression
```

A Pipeline learns the scaler's mean and std **only during `.fit()` on training data**, which prevents leakage.

| Function | Experiment |
|---|---|
| `exp_k_and_scaling` | 1 · K values × scaled / unscaled |
| `exp_pca` | 2 · number of PCA components, accuracy and speed |
| `exp_training_size` | 3 · 5 to 250 attempts per person |
| `final_test` | 4 · all models on the test set, once |
| `most_confused_pairs` | failure cases |
| `exp_verification` | 5 · EER of three login detectors |
| `run_all` | runs everything and saves tables, figures, `summary.json` and `environment.json` |

`fit_and_score` also times training and prediction. Prediction is timed 3 times and the median is kept, because laptop timings are noisy.

### `plots.py` · graphs

One function per figure; each saves a PNG to `results/figures/`. EDA figures are named `eda1…eda5` and experiment figures `exp1…exp5`.

---

## `app/` · the web app

### `app/server.py` · the Python backend (Flask)

**Flask** is a small Python library that answers web requests. When the page asks for `/api/login`, Flask runs the matching Python function and sends the answer back as **JSON** (text that looks like a Python dict).

**What it keeps:**
* your group's attempts in `data/app_users.csv` (same 31 columns as CMU)
* typist names in `data/app_users.json`
* 10 CMU volunteers × 50 attempts as **background typists**, if the dataset is in `data/`

**Models** (rebuilt automatically when someone enrolls):
`StandardScaler` → `NearestNeighbors` (for the KNN vote) + `LogisticRegression` + `PCA(2)` (for the map).

| Address | What it does |
|---|---|
| `GET /` | sends the web page |
| `GET /api/state` | list of typists, attempt counts, available figures |
| `POST /api/users` | add a typist |
| `DELETE /api/users/<name>` | delete a typist and their attempts |
| `POST /api/enroll` | key times → 31 features → saved as a new row |
| `POST /api/login` | compares an attempt (saves nothing): distance vs threshold, KNN vote, logistic regression probability, PCA position |
| `GET /api/profile` | one typist's mean rhythm, spread, threshold, FAR and FRR |
| `GET /api/pca` | 2D coordinates of every stored attempt |
| `GET /api/export.csv` | download your group's attempts |
| `GET /figures/<file>` | experiment graphs for the results gallery |

**The login, step by step** (`login()` in the code):
1. `events_to_features` turns the key times into 31 numbers.
2. `DistanceVerifier.fit` on the claimed person's attempts, then `.score` on the new attempt.
3. The threshold comes from leave-one-out scores and the strictness slider.
4. Granted if distance ≤ threshold. The match % is a smooth curve that equals 50% exactly at the threshold.
5. KNN: scale, then find the K nearest stored attempts and count votes.
6. Logistic regression: the probability for the claimed person.
7. PCA: the attempt's position on the map.

### `app/static/index.html` · the frontend

Plain HTML, CSS and JavaScript (no framework).

* **Capture:** listens to `keydown` / `keyup` inside the typing box and stores `performance.now()` timestamps (precise to fractions of a millisecond). It checks each key as you go, Backspace restarts, and Caps Lock is detected.
* **Keystroke trace** (canvas): orange bars are your keys (length = hold time); cyan outlines are the stored average rhythm.
* **Result panel:** verdict, match gauge, distance/threshold, KNN vote, logistic regression probability, FRR/FAR, and a 31-feature chart (your attempt vs the stored mean ± spread).
* **Typist map:** an SVG scatter of the PCA coordinates; your attempt is a star with lines to its K nearest neighbours.
* **Settings:** K and strictness sliders re-run the last login instantly.
* **Results gallery:** shows `results/figures/*.png` after you run the experiments.

All ML happens in Python; the page only records and draws.

---

## `notebooks/KeyPrint_Analysis.ipynb`

The full study with explanations. It calls the same functions as `python -m src.experiments`, but one step at a time, with "what to notice" notes after each graph and a conclusion template at the end.
