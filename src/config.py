"""
config.py - one place for every setting the project uses.

Beginner note:
    Instead of typing the same file paths and numbers in many files,
    we keep them here. If you want to change something (for example the
    random seed), you change it once in this file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains src/, app/, data/ ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# The CMU dataset file. Download it from https://www.cs.cmu.edu/~keystroke/
# and put it inside the data/ folder with exactly this name.
CMU_CSV = DATA_DIR / "DSL-StrongPasswordData.csv"

# Typing samples recorded by the web app (your group's own data).
APP_USERS_CSV = DATA_DIR / "app_users.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# A "random seed" makes random steps (like shuffling) give the SAME result
# every time you run the code, so another student can reproduce your numbers.
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# The password and the 31 features
# ---------------------------------------------------------------------------
# Everyone types the same password:  .tie5Roanl  then Enter
# These are the 11 keys as the CMU dataset names them.
KEY_NAMES = ["period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return"]

# The characters the browser reports for those keys (used by the web app).
PASSWORD_KEYS = [".", "t", "i", "e", "5", "R", "o", "a", "n", "l", "Enter"]


def build_feature_columns():
    """
    Build the 31 column names in the SAME order as the CMU dataset.

    For every key we have:
        H.<key>            hold time      (release - press of the same key)
    and for every pair of neighbouring keys:
        DD.<key1>.<key2>   press -> press gap   (key2 press - key1 press)
        UD.<key1>.<key2>   release -> press gap (key2 press - key1 release)

    11 hold times + 10 DD gaps + 10 UD gaps = 31 features.
    """
    cols = []
    for i, key in enumerate(KEY_NAMES):
        cols.append(f"H.{key}")
        if i < len(KEY_NAMES) - 1:
            nxt = KEY_NAMES[i + 1]
            cols.append(f"DD.{key}.{nxt}")
            cols.append(f"UD.{key}.{nxt}")
    return cols


FEATURE_COLUMNS = build_feature_columns()
assert len(FEATURE_COLUMNS) == 31, "There must be exactly 31 features"

# ---------------------------------------------------------------------------
# How we split the data (by typing session, see src/data.py for why)
# ---------------------------------------------------------------------------
TRAIN_SESSIONS = [1, 2, 3, 4, 5]   # model learns from these
VAL_SESSIONS = [6]                 # "practice exam" used to choose settings (K, PCA size)
TEST_SESSIONS = [7, 8]             # "final exam", used only once at the very end

# ---------------------------------------------------------------------------
# Web app settings
# ---------------------------------------------------------------------------
MIN_ENROLL_SAMPLES = 5        # a person needs at least this many attempts to log in
BACKGROUND_SUBJECTS = 10      # how many CMU volunteers the app uses as "other people"
BACKGROUND_REPS = 50          # how many attempts per CMU volunteer the app loads
