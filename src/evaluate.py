"""
evaluate.py - measure how good the models are.

Beginner note:
    Identification ("who is typing?"):
        accuracy  = % of attempts where we named the right person
        macro-F1  = F1 score computed for each person, then averaged, so every
                    person counts equally (good when classes are many)

    Verification ("is this really Rahul?") makes two kinds of mistakes:
        FAR (False Accept Rate)  = % of IMPOSTOR attempts we wrongly let in
        FRR (False Reject Rate)  = % of GENUINE attempts we wrongly blocked

        Moving the threshold trades one for the other. The point where
        FAR == FRR is called the EER (Equal Error Rate). Lower EER = better.
        It lets us compare methods with ONE number.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def identification_scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def far_frr(genuine_scores, impostor_scores, thresholds):
    """FAR and FRR at each threshold. Access is granted when score <= threshold."""
    g = np.asarray(genuine_scores)
    imp = np.asarray(impostor_scores)
    far = np.array([(imp <= t).mean() for t in thresholds])
    frr = np.array([(g > t).mean() for t in thresholds])
    return far, frr


def equal_error_rate(genuine_scores, impostor_scores, n_steps=400):
    """Find the threshold where FAR and FRR are (almost) equal."""
    allv = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(allv.min(), allv.max(), n_steps)
    far, frr = far_frr(genuine_scores, impostor_scores, thresholds)
    i = int(np.argmin(np.abs(far - frr)))
    return {"eer": float((far[i] + frr[i]) / 2), "threshold": float(thresholds[i])}
