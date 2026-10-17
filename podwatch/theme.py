"""Colours. Explicit hex values, so the look does not depend on the terminal's own palette."""
from __future__ import annotations

TEXT = "#c8ccd4"
MUTED = "#6b7280"
FAINT = "#2f3441"  # empty part of a bar
ACCENT = "#5fd7ff"
GOOD = "#5fd787"
WARN = "#ffd75f"
BAD = "#ff6b6b"
VIOLET = "#c792ea"

# Panel borders are a step quieter than the text colours so a wall of pods does not shout.
BORDER = {"running": "#3f8f66", "idle": "#9a8230", "connecting": "#9a8230", "unreachable": "#a64545"}
BORDER_DEFAULT = "#4b5263"

STATE_COLOR = {"running": GOOD, "idle": WARN, "connecting": WARN, "unreachable": BAD}


def load_color(percent: float) -> str:
    """GPU utilisation: low means the card is waiting for something."""
    return BAD if percent < 40 else WARN if percent < 75 else GOOD


def memory_color(percent: float) -> str:
    return BAD if percent >= 95 else WARN if percent >= 85 else ACCENT
