"""
data.py - load the CMU dataset and split it into train / validation / test.

Beginner note on splitting:
    We never test a model on the same attempts it learned from. That would be
    like giving students the exact exam questions to study: high marks, but
    no proof they learned anything.

    Every CMU volunteer typed in 8 separate sessions (different days),
    50 attempts per session. We split BY SESSION:

        sessions 1-5  -> TRAIN       the model learns from these
        session  6    -> VALIDATION  "practice exam": we use it to choose settings
        sessions 7-8  -> TEST        "final exam": used only once at the end

    Why by session and not randomly? Attempts typed in the same sitting are
    very similar to each other. If attempt #12 of a session is in training and
    attempt #13 is in testing, the test becomes too easy. Splitting by session
    tests the honest question: "can we recognise you on a LATER day?"
"""

import pandas as pd

from .config import (
    APP_USERS_CSV,
    CMU_CSV,
    FEATURE_COLUMNS,
    TEST_SESSIONS,
    TRAIN_SESSIONS,
    VAL_SESSIONS,
)


def load_cmu(path=CMU_CSV):
    """Read the CMU CSV and check that it has the columns we expect."""
    path = str(path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Download DSL-StrongPasswordData.csv from https://www.cs.cmu.edu/~keystroke/ "
            "and put it in the data/ folder. See data/README.md."
        )
    missing = [c for c in ["subject", "sessionIndex", "rep", *FEATURE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"The file is missing these columns: {missing}")
    df["subject"] = df["subject"].astype(str)
    return df


def split_by_session(df):
    """Return (train, val, test) DataFrames using the session rule above."""
    train = df[df["sessionIndex"].isin(TRAIN_SESSIONS)].copy()
    val = df[df["sessionIndex"].isin(VAL_SESSIONS)].copy()
    test = df[df["sessionIndex"].isin(TEST_SESSIONS)].copy()
    return train, val, test


def X_y(df):
    """Split a DataFrame into the numbers (X) and the answers (y)."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["subject"].to_numpy()
    return X, y


def load_app_users(path=APP_USERS_CSV):
    """Typing samples recorded in the web app (same columns as CMU)."""
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["subject", "sessionIndex", "rep", *FEATURE_COLUMNS])
    df["subject"] = df["subject"].astype(str)
    return df
