"""Utility helpers for the EV model."""

def fc(flag):
    """Return a numeric weight for topological flags.

    Simple helper used in the notebook. Accepts numeric-like input and returns
    a float. Non-convertible values result in 0.0.
    """
    try:
        return float(flag)
    except Exception:
        return 0.0
