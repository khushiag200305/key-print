"""
verifier.py - answers the login question "is this REALLY Rahul?"

Beginner note:
    KNN and logistic regression answer "WHO is typing?" (pick one name from many).
    A login needs a different question: "is this the person they claim to be?"

    The idea is simple:
      1. From Rahul's enrolled attempts, compute his AVERAGE time for each of
         the 31 features, and how much he normally VARIES (his spread).
      2. For a new attempt, measure how far each of its 31 numbers is from
         Rahul's average.
      3. Add those differences up into ONE number: the distance.
         Small distance = typed like Rahul. Big distance = typed differently.
      4. If the distance is below a THRESHOLD, let them in.

    Three ways to measure the distance (we compare them in the experiments):
      - "euclidean"        : straight-line distance (like a ruler)
      - "manhattan"        : add up the absolute differences (like walking city blocks)
      - "scaled_manhattan" : like manhattan, but each difference is divided by
                             Rahul's own spread for that feature. A key Rahul is
                             very consistent on counts more; a key he is sloppy
                             on counts less. This is the method from the
                             Killourhy & Maxion (2009) paper that made the dataset.
"""

import numpy as np


class DistanceVerifier:
    def __init__(self, kind="scaled_manhattan"):
        if kind not in ("euclidean", "manhattan", "scaled_manhattan"):
            raise ValueError("kind must be euclidean, manhattan or scaled_manhattan")
        self.kind = kind

    def fit(self, X_user):
        """Learn one person's profile from their attempts (rows of 31 numbers)."""
        X_user = np.asarray(X_user, dtype=float)
        self.mean_ = X_user.mean(axis=0)
        # Mean absolute deviation: on average, how far is each attempt from the mean?
        spread = np.abs(X_user - self.mean_).mean(axis=0)
        # Avoid dividing by zero for a feature that never changed.
        self.spread_ = np.maximum(spread, 0.004)
        return self

    def score(self, X):
        """Distance of each row in X from the profile. Lower = more similar."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        diff = X - self.mean_
        if self.kind == "euclidean":
            return np.sqrt((diff ** 2).sum(axis=1))
        if self.kind == "manhattan":
            return np.abs(diff).sum(axis=1)
        return (np.abs(diff) / self.spread_).sum(axis=1) / X.shape[1]


def leave_one_out_scores(X_user, kind="scaled_manhattan"):
    """
    How far is each enrolled attempt from a profile built WITHOUT it?

    This tells us how much a person naturally varies, which we use to pick a
    fair threshold for them when we only have a few attempts (as in the web app).
    """
    X_user = np.asarray(X_user, dtype=float)
    scores = []
    for i in range(len(X_user)):
        rest = np.delete(X_user, i, axis=0)
        scores.append(DistanceVerifier(kind).fit(rest).score(X_user[i])[0])
    return np.array(scores)


def threshold_from_scores(loo_scores, strictness=2.0):
    """
    threshold = average LOO score + strictness x standard deviation.

    Lower strictness -> stricter (more people rejected, fewer impostors get in).
    """
    m = float(np.mean(loo_scores))
    sd = float(np.std(loo_scores, ddof=1)) if len(loo_scores) > 1 else 0.0
    return max(m * 1.05, m + strictness * sd)
