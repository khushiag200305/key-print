# KeyPrint · Recognising People by How They Type

> **Your password can leak. Your rhythm can't.**

KeyPrint is a machine learning project for the **UML501 Machine Learning Lab**. It checks *who* is typing a password by looking at **how** they type it: how long each key is held down, and the small pauses between keys.

It has three parts:

1. **A study** (Jupyter notebook) on a real research dataset of 51 people, comparing KNN, logistic regression and PCA with proper experiments.
2. **A web app** with a live demo. You enroll by typing the password a few times; then a friend types the *same* password and gets rejected.
3. **Beginner documentation** that explains every idea assuming you have never studied machine learning.

---

## Contents

1. [The idea in 60 seconds](#1-the-idea-in-60-seconds)
2. [Quick start](#2-quick-start)
3. [Folder structure](#3-folder-structure)
4. [Machine learning explained from zero](#4-machine-learning-explained-from-zero)
5. [What the experiments test](#5-what-the-experiments-test)
6. [How this project meets every lab requirement](#6-how-this-project-meets-every-lab-requirement)
7. [Troubleshooting](#7-troubleshooting)
8. [Credits and citation](#8-credits-and-citation)

More detail lives in `docs/`:

| File | What it is for |
|---|---|
| [`docs/ML_EXPLAINED.md`](docs/ML_EXPLAINED.md) | The full beginner guide: every concept with examples |
| [`docs/HOW_THE_CODE_WORKS.md`](docs/HOW_THE_CODE_WORKS.md) | A tour of every file and how the app talks to Python |
| [`docs/VIVA_QUESTIONS.md`](docs/VIVA_QUESTIONS.md) | Likely viva questions with simple answers |
| [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md) | The structure for your final report |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | A 5-minute demo plan, split between group members |

---

## 1. The idea in 60 seconds

Everyone types the same password: **`.tie5Roanl`** then **Enter**.

The computer records two moments for each key: when it goes **down** and when it comes **up**. From those it calculates **31 numbers** per attempt:

| Kind | Meaning | How many |
|---|---|---|
| **H** · hold time | how long one key stays pressed | 11 (one per key) |
| **DD** · press → press | from pressing one key to pressing the next | 10 (gaps between 11 keys) |
| **UD** · release → press | from releasing one key to pressing the next | 10 |
| | **Total** | **31** |

Your 31 numbers look similar every time *you* type, and different when *someone else* types. Machine learning learns those patterns.

```
Type the password
      ↓
31 timing numbers           ← features
      ↓
Scale them fairly           ← standardisation (z-score)
      ↓
Compare with stored attempts
   ├─ KNN: who owns the most similar past attempts?
   ├─ Logistic regression: what's the probability it's you?
   └─ Distance + threshold: close enough to your profile?
      ↓
✅ Access granted   /   ❌ Access denied
      ↓
PCA map shows it visually  ·  FAR / FRR measure the mistakes
```

---

## 2. Quick start

You need **Python 3.10 or newer** and a computer with a **physical keyboard** (the app does not work on phones).

### Step 1 · Install the libraries

Open a terminal **inside this project folder** and run:

```bash
pip install -r requirements.txt
```

> On some computers the command is `pip3` or `python -m pip`. Using a virtual environment is optional but tidy:
> `python -m venv .venv` then `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows).

### Step 2 · Download the dataset

1. Go to **https://www.cs.cmu.edu/~keystroke/**
2. Download **`DSL-StrongPasswordData.csv`**
3. Put it in the **`data/`** folder, so the path is `data/DSL-StrongPasswordData.csv`

See [`data/README.md`](data/README.md) for details. Everything except the web app needs this file. The web app runs without it, but it's better with it.

### Step 3 · Run all experiments (about 1 minute)

```bash
python -m src.experiments
```

This creates:
* `results/tables/*.csv` with all the numbers
* `results/figures/*.png` with all the graphs (they also appear in the web app)
* `results/summary.json` and `results/environment.json` (versions, for reproducibility)

### Step 4 · Start the web app

```bash
python app/server.py
```

Open **http://127.0.0.1:5000** in your browser. Stop the app with `Ctrl + C`.

**Trying it:**
1. Under **Enroll**, add your name and click the typing area.
2. Type `.tie5Roanl` then `Enter` at your normal speed. Repeat **5 to 10 times**.
3. Switch to **Log in**, pick your name, and type it again. You should get **Access granted**.
4. Now let a friend type your password while your name is selected. They should get **Access denied**.

### Step 5 · Open the notebook (the full study with explanations)

```bash
jupyter notebook notebooks/KeyPrint_Analysis.ipynb
```

Run the cells top to bottom with `Shift + Enter`. VS Code can also open `.ipynb` files directly.

---

## 3. Folder structure

```
keyprint-project/
├── README.md                     ← you are here
├── requirements.txt              ← Python libraries to install
│
├── data/
│   ├── README.md                 ← how to get the dataset
│   ├── DSL-StrongPasswordData.csv  (you download this)
│   ├── app_users.csv               (created by the app: your group's attempts)
│   └── app_users.json              (created by the app: list of typist names)
│
├── src/                          ← all the machine learning code
│   ├── config.py                 ← settings: paths, seed, the 31 feature names, split
│   ├── features.py               ← key press/release times → 31 numbers
│   ├── data.py                   ← load the CSV, split train / validation / test
│   ├── verifier.py               ← login check: distance to your profile + threshold
│   ├── evaluate.py               ← accuracy, F1, FAR, FRR, EER
│   ├── experiments.py            ← every experiment (run: python -m src.experiments)
│   └── plots.py                  ← every graph
│
├── app/                          ← the web app
│   ├── server.py                 ← Python backend (Flask): runs the ML for the app
│   └── static/index.html         ← frontend: records typing, draws everything
│
├── notebooks/
│   └── KeyPrint_Analysis.ipynb   ← the full study, explained step by step
│
├── results/                      ← created when you run the experiments
│   ├── figures/                  ← PNG graphs
│   └── tables/                   ← CSV results
│
└── docs/                         ← beginner explanations, viva prep, report outline
```

---

## 4. Machine learning explained from zero

*This is the short version. The full guide with more examples is [`docs/ML_EXPLAINED.md`](docs/ML_EXPLAINED.md).*

### 4.1 What is machine learning?

Normally a programmer writes the rules: *"if the password is correct, let them in."*

In machine learning, **we don't write the rules. We show the computer many examples and it finds the pattern itself.** It's like recognising your mother's voice on the phone: nobody gave you a rule, you just heard it thousands of times.

### 4.2 Data: features and labels

One typing attempt becomes **one row of 31 numbers**:

| Person | H.period | DD.period.t | H.t | … 31 columns |
|---|---|---|---|---|
| Rahul | 0.10 | 0.30 | 0.09 | … |
| Rahul | 0.11 | 0.28 | 0.08 | … |
| Priya | 0.07 | 0.15 | 0.06 | … |

* **Features** are the 31 measured numbers (the columns).
* The **label** is the person's name (the answer we want to predict).
* A **dataset** is many rows like these.

### 4.3 Where the 31 numbers come from (tiny example)

Password `cat` + Enter, times in milliseconds:

| key | pressed | released |
|---|---|---|
| c | 0 | 100 |
| a | 250 | 330 |
| t | 500 | 590 |
| Enter | 800 | 880 |

* **Hold** = released − pressed. For `c`: 100 − 0 = **100**
* **Press→press** = next pressed − this pressed. For `c→a`: 250 − 0 = **250**
* **Release→press** = next pressed − this released. For `c→a`: 250 − 100 = **150**

4 keys give 4 holds + 3 + 3 gaps = **10 numbers**. The real password has 11 keys, so it gives 11 + 10 + 10 = **31**.

### 4.4 Training vs testing

We **hide** some data from the model and use it only at the end, like an exam with new questions. If we tested on the same attempts the model learned from, it could just memorise them.

We split by **typing session** (each person typed on 8 different days):

| Sessions | Role |
|---|---|
| 1–5 | **Train**: the model learns |
| 6 | **Validation**: "practice exam" to pick settings such as K |
| 7–8 | **Test**: "final exam", used once |

### 4.5 Scaling

Gaps (around 0.30 s) are bigger numbers than hold times (around 0.09 s), so they would dominate every comparison. **Standardisation** fixes that:

> new value = (value − average) ÷ standard deviation

Afterwards every feature means "how unusual is this?" and counts equally. It's like converting marks out of 100 and out of 10 into percentages before comparing them.

### 4.6 Distance: how similar are two attempts?

Picture each attempt as a dot. Similar typing means the dots are close together.

```
gap ↑
    |          P P
    |         P P        ← Priya's attempts
    |   R R
    |  R R R             ← Rahul's attempts
    +------------------→ hold time
```

With 31 features we can't draw it, but the maths is the same: compare all 31 numbers and add up the differences.

### 4.7 KNN (K-Nearest Neighbours): "who is this?"

*"Tell me who your neighbours are, and I'll tell you who you are."*

1. Measure the distance from the new attempt to **every stored attempt**.
2. Pick the **K closest** (for example K = 5).
3. **Majority vote**: Rahul, Rahul, Rahul, Priya, Rahul → it's **Rahul**.

Small K trusts very few neighbours (noisy); large K can pull in other people (blurry). Experiment 1 finds the best K.

### 4.8 Logistic regression: "how sure are we?"

1. It learns a **weight** for every feature for every person.
2. It multiplies the features by the weights and adds them into a score.
3. It turns the scores into **probabilities** (sigmoid / softmax), for example "Rahul 91%, Priya 6%".

**Training** means adjusting the weights again and again until the guesses on training data are right, like tuning a guitar string by ear.

### 4.9 PCA: squeezing 31 numbers into a few

PCA finds the directions where attempts differ the most and keeps only those. It's like photographing a teapot from the angle where you can see both the spout and the handle.

* We use **2 components** to draw the **typist map**, where each person forms a cluster.
* Experiment 2 tests whether KNN still works with fewer components, and whether it gets faster.

### 4.10 The login check: distance + threshold

For a login we ask *"is this REALLY Rahul?"*:

1. From Rahul's attempts, compute his **average** for each feature and his **spread** (how much he usually varies).
2. Measure how far the new attempt is from his average, dividing each difference by his spread. This is the **scaled Manhattan distance**.
3. If the distance is **below a threshold** → ✅ granted; above it → ❌ denied.

### 4.11 Two kinds of mistakes: FAR and FRR

| Mistake | Name | Example |
|---|---|---|
| Let a stranger in | **FAR**, False Accept Rate | Priya logs in as Rahul (security failure) |
| Block the real person | **FRR**, False Reject Rate | Rahul gets rejected (annoying) |

A stricter threshold means lower FAR but higher FRR. The point where they are equal is the **EER** (Equal Error Rate), one number to compare methods; lower is better. Try the **Strictness** slider in the app.

### 4.12 How we measure the models

* **Accuracy**: the percentage of attempts where we named the right person.
* **Macro-F1**: the F1 score for each person, averaged, so every person counts equally.
* **Confusion matrix**: a grid showing who gets mistaken for whom.
* **EER**: for the login check (above).

---

## 5. What the experiments test

All experiments live in `src/experiments.py` and run in the notebook.

| # | Question | What we vary | Scored on |
|---|---|---|---|
| Baseline | How good is the simplest model? | KNN, K = 1, **no scaling** | validation |
| 1 | Does scaling matter? What is the best K? | K ∈ {1, 3, 5, 7, 9, 15, 25} × scaled / unscaled | validation |
| 2 | How many PCA components do we need? | 2 to 31 components before KNN | validation (+ speed) |
| 3 | How much enrollment data is needed? | 5 to 250 attempts per person, KNN vs logistic regression | validation |
| 4 | Final comparison | baseline, KNN, PCA + KNN, logistic regression (raw / scaled) | **test, once** |
| 5 | Which login detector is best? | Euclidean vs Manhattan vs scaled Manhattan distance | **EER on test** |

Settings (best K, number of PCA components) are chosen on **validation** only. The test set is used once at the end, so the final numbers are honest.

---

## 6. How this project meets every lab requirement

| # | Deliverable in the lab PDF | Where it is in this project |
|---|---|---|
| 1 | Problem definition and background | Notebook Part 1; `docs/REPORT_OUTLINE.md` |
| 2 | Dataset documentation | Notebook Part 2 (source, terms, counts, limitations); `data/README.md` |
| 3 | EDA (at least 3, with interpretation) | Notebook Part 3: class balance, distributions, correlations, per-person rhythm, PCA preview |
| 4 | Preprocessing pipeline | Notebook Part 4: session split, z-score fitted on train only (`Pipeline`), and why missing-value, encoding and imbalance steps are not needed |
| 5 | Baseline and proposed method | Baseline KNN K=1 raw; proposed scaled KNN, logistic regression, PCA + KNN, scaled Manhattan verifier |
| 6 | At least 3 experiments | 5 experiments (table above) |
| 7 | Results table + at least 3 result plots | `results/tables/`, 7 result figures, not accuracy alone (macro-F1, timing, EER, FAR/FRR) |
| 8 | Analysis and conclusion | Notebook Part 14 template; confused pairs and hardest-person failure cases |
| 9 | Reproducible code | `requirements.txt`, fixed seed (`RANDOM_STATE = 42`), `results/environment.json`, this README |
| 10 | Report and presentation | `docs/REPORT_OUTLINE.md`, `docs/DEMO_SCRIPT.md`, the live web app |

General rules from the PDF:
* **Python + pandas, NumPy, Matplotlib, scikit-learn:** yes, all of them.
* **Cite sources:** the dataset paper is cited in the notebook, the app footer and below.
* **Fixed seed, no leakage:** the split is by session, and the scaler is fitted on train only.
* **CPU-friendly:** the whole study runs in about a minute on a normal laptop.
* **Never fabricate results:** every number in your report must come from your own run.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError ... DSL-StrongPasswordData.csv` | Download the dataset into `data/` (Step 2). |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the **project folder**, not from inside `src/`. |
| `pip` not found | Try `pip3 install -r requirements.txt` or `python -m pip install -r requirements.txt`. |
| The app says it can't reach the server | Keep the terminal with `python app/server.py` open while you use the page. |
| Port 5000 is busy (common on Mac) | Edit the last line of `app/server.py` to use `port=5001`, then open http://127.0.0.1:5001 |
| "Wrong key" while typing | Type exactly `.tie5Roanl` (capital **R** with Shift), then Enter. Caps Lock must be off. Backspace restarts. |
| Experiment graphs don't show in the app | Run `python -m src.experiments` first, then refresh the page. |
| Want to start the app fresh | Stop the server and delete `data/app_users.csv` and `data/app_users.json`. |

---

## 8. Credits and citation

**Dataset:** K. S. Killourhy and R. A. Maxion, "Comparing Anomaly-Detection Algorithms for Keystroke Dynamics," *IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)*, 2009. Data: https://www.cs.cmu.edu/~keystroke/

**Libraries:** scikit-learn, pandas, NumPy, Matplotlib, Flask. Fonts in the app come from Google Fonts.

**Course:** UML501 Machine Learning Lab, Thapar Institute of Engineering and Technology, Patiala.
