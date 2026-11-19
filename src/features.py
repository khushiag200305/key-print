"""
features.py - turn raw key presses into the 31 timing numbers.

Beginner note:
    When you type, the browser only tells us two moments for every key:
        - when it went DOWN (pressed)
        - when it came UP   (released)
    This file turns those moments into the 31 numbers the models use.

    Example with the tiny password "cat" + Enter (times in milliseconds):

        key    down   up
        c        0    100
        a      250    330
        t      500    590
        Enter  800    880

        hold time of c          = 100 - 0   = 100 ms
        press->press  c->a      = 250 - 0   = 250 ms
        release->press c->a     = 250 - 100 = 150 ms

    The real password has 11 keys, so we get 11 + 10 + 10 = 31 numbers.
    We store them in SECONDS (0.100 instead of 100) to match the CMU dataset.
"""

from .config import FEATURE_COLUMNS, PASSWORD_KEYS


class TypingError(ValueError):
    """Raised when a typing attempt cannot be turned into features."""


def events_to_features(events):
    """
    Convert one typing attempt into a dict of 31 features (in seconds).

    Parameters
    ----------
    events : list of dict
        One dict per key, in typing order, for example
        {"key": "t", "down": 1234.5, "up": 1320.1}  (times in milliseconds)

    Returns
    -------
    dict  {feature_name: value_in_seconds}  in the CMU column order.
    """
    if not isinstance(events, list) or len(events) != len(PASSWORD_KEYS):
        raise TypingError(f"Expected {len(PASSWORD_KEYS)} keys, got {len(events) if isinstance(events, list) else 'none'}.")

    downs, ups = [], []
    for i, (ev, expected) in enumerate(zip(events, PASSWORD_KEYS)):
        if ev.get("key") != expected:
            raise TypingError(f"Key {i + 1} should be '{expected}' but was '{ev.get('key')}'.")
        try:
            down = float(ev["down"])
            up = float(ev["up"])
        except (KeyError, TypeError, ValueError):
            raise TypingError(f"Key {i + 1} is missing its press or release time.")
        if up < down:
            raise TypingError(f"Key {i + 1} was released before it was pressed.")
        downs.append(down / 1000.0)  # milliseconds -> seconds
        ups.append(up / 1000.0)

    values = []
    for i in range(len(PASSWORD_KEYS)):
        values.append(ups[i] - downs[i])                   # H  : hold time
        if i < len(PASSWORD_KEYS) - 1:
            values.append(downs[i + 1] - downs[i])         # DD : press -> press
            values.append(downs[i + 1] - ups[i])           # UD : release -> press

    # Simple sanity checks so one weird attempt doesn't pollute the data.
    for name, v in zip(FEATURE_COLUMNS, values):
        if name.startswith("H.") and v > 1.5:
            raise TypingError("A key was held for more than 1.5 seconds. Please type normally.")
        if name.startswith("DD.") and v > 4.0:
            raise TypingError("There was a pause of more than 4 seconds. Type the password in one go.")

    return {name: round(v, 4) for name, v in zip(FEATURE_COLUMNS, values)}
