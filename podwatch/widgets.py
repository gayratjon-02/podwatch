"""Small drawing helpers: bars, the loss chart, progress parsing and log colouring."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from podwatch import theme

BLOCKS = "▁▂▃▄▅▆▇█"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Palette(NamedTuple):
    """Chart colours from the old side (left) to the newest values (right): a bright top edge over a dark fill."""
    edge_from: str
    edge_to: str
    fill_from: str
    fill_to: str


LOSS_PALETTE = Palette("#3f8fa8", theme.ACCENT, "#173844", "#1f6a82")
GPU_PALETTE = Palette("#3f9f6f", theme.GOOD, "#17382a", "#1f6a45")

SEPARATOR = (" │ ", theme.FAINT)
HIDDEN_KEYS = {"step", "epoch", "time", "timestamp", "ts"}
RATE_KEYS = {"lr", "learning_rate"}
SHOWN_METRICS = 5


def spinner_char(now: float) -> str:
    return SPINNER[int(now * 10) % len(SPINNER)]


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def bar(value: float, total: float, color: str, width: int | None = None) -> ProgressBar:
    return ProgressBar(total=total, completed=min(value, total), width=width, style=theme.FAINT,
                       complete_style=color, finished_style=color)


def gpu_rows(gpus: list[list[float]], cell_width: int) -> Table:
    """One line per GPU: utilisation bar and memory bar side by side."""
    grid = Table.grid(padding=(0, 1))
    bar_width = max(8, min(22, (cell_width - 46) // 2))
    for index, (util, used, total) in enumerate(gpus):
        label = "GPU" if len(gpus) == 1 else f"GPU{index}"
        memory = used / total * 100 if total else 0.0
        util_color, mem_color = theme.load_color(util), theme.memory_color(memory)
        grid.add_row(Text(label, style=f"bold {theme.TEXT}"), bar(util, 100, util_color, bar_width),
                     Text(f"{util:3.0f}%", style=f"bold {util_color}"),
                     Text(" VRAM", style=f"bold {theme.TEXT}"), bar(memory, 100, mem_color, bar_width),
                     Text(f"{used / 1024:4.1f}/{total / 1024:.0f} GB", style=mem_color))
    return grid


def metrics_text(record: dict[str, object]) -> Text:
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append(f"step {record.get('step', '?')}", style=f"bold {theme.ACCENT}")
    shown = 0
    for key, value in record.items():
        if key in HIDDEN_KEYS or not is_number(value):
            continue
        text.append(*SEPARATOR)
        text.append(f"{key} ", style=theme.MUTED)
        text.append(f"{value:.1e}" if key in RATE_KEYS else f"{value:.3f}", style=f"bold {theme.TEXT}")
        shown += 1
        if shown >= SHOWN_METRICS:
            break
    return text


def resample(values: list[float], width: int) -> list[float]:
    """Exactly `width` points: bucket means when there are more values, a step plot when there are fewer."""
    count = len(values)
    if count == 0 or width <= 0:
        return []
    if count >= width:
        out = []
        for i in range(width):
            start = i * count // width
            chunk = values[start:max(start + 1, (i + 1) * count // width)]
            out.append(sum(chunk) / len(chunk))
        return out
    return [values[min(count - 1, i * count // width)] for i in range(width)]


def smooth(values: list[float], window: int = 7) -> list[float]:
    """Centred moving average, shorter at the ends. Takes the jitter out of a noisy loss without lagging behind it."""
    half = window // 2
    return [sum(values[max(0, i - half):i + half + 1]) / len(values[max(0, i - half):i + half + 1])
            for i in range(len(values))]


def chart(values: list[float], width: int, height: int, low: float | None = None,
          high: float | None = None) -> list[str]:
    """`height` rows of block characters, top row first.

    The scale is the min and max of the values shown, unless `low` and `high` fix it (GPU load is always 0 to 100).
    """
    if len(values) < 2 or width < 1:
        return []
    points = resample(values, width)
    low = min(points) if low is None else low
    high = max(points) if high is None else high
    span = (high - low) or 1.0
    levels = height * 8
    heights = [1 + round((min(max(p, low), high) - low) / span * (levels - 1)) for p in points]
    rows = []
    for row in range(height - 1, -1, -1):
        cells = [min(8, h - row * 8) for h in heights]
        rows.append("".join(BLOCKS[c - 1] if c > 0 else " " for c in cells))
    return rows


def lerp_color(start: str, end: str, t: float) -> str:
    a, b = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (start, end))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def chart_rows(rows: list[str], palette: Palette = LOSS_PALETTE) -> list[Text]:
    """Colour the chart: the top cell of every column bright, the fill below it dark, both fading in from the old side."""
    if not rows:
        return []
    width = len(rows[0])
    texts = [Text(row, no_wrap=True) for row in rows]
    for x in range(width):
        t = x / max(1, width - 1)
        edge, fill = lerp_color(palette.edge_from, palette.edge_to, t), lerp_color(palette.fill_from, palette.fill_to, t)
        edge_done = False
        for text, row in zip(texts, rows):
            if row[x] == " ":
                continue
            text.stylize(fill if edge_done else edge, x, x + 1)
            edge_done = True
    return texts


def trend(values: list[float]) -> float | None:
    """Relative change from the first third of the values to the last third."""
    count = len(values)
    if count < 6:
        return None
    third = max(2, count // 3)
    early, late = sum(values[:third]) / third, sum(values[-third:]) / third
    return (late - early) / abs(early) if early else None


def trend_text(change: float | None, key: str, window: int) -> Text:
    if change is None:
        return Text("")
    flat = abs(change) < 0.005
    arrow = "→" if flat else "↓" if change < 0 else "↑"
    if flat:
        color = theme.MUTED
    elif "loss" in key:
        color = theme.GOOD if change < 0 else theme.BAD
    else:
        color = theme.ACCENT
    return Text(f"{arrow} {abs(change) * 100:.0f}% over last {window}", style=color)


TQDM = re.compile(r"(?P<label>[^|%]*?)\s*(?P<pct>\d{1,3})%\|[^|]*\|\s*(?P<done>\d+)/(?P<total>\d+)"
                  r"(?:\s*\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+),\s*(?P<rate>[^\]]*)\])?")
STAMP = re.compile(r"^\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[.,]\d+)?\s*")


@dataclass(frozen=True)
class Progress:
    label: str
    percent: int
    done: int
    total: int
    eta: str
    rate: str


def parse_progress(line: str) -> Progress | None:
    """A tqdm style line such as `Running requests:  76%|███▏| 1009/1319 [18:23<05:03, 1.02it/s]`."""
    match = TQDM.search(line)
    if not match:
        return None
    label = STAMP.sub("", match["label"]).strip().rstrip(":").strip()
    return Progress(label, min(100, int(match["pct"])), int(match["done"]), int(match["total"]),
                    match["eta"] or "?", (match["rate"] or "").strip())


def progress_row(progress: Progress) -> Table:
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(no_wrap=True, max_width=32, overflow="ellipsis")
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    stats = Text(no_wrap=True)
    stats.append(f"{progress.percent:3d}%", style=f"bold {theme.ACCENT}")
    stats.append(f"  {progress.done}/{progress.total}", style=theme.TEXT)
    stats.append(f"  eta {progress.eta}", style=theme.MUTED)
    if progress.rate:
        stats.append(f"  {progress.rate}", style=theme.MUTED)
    grid.add_row(Text(progress.label or "progress", style=f"bold {theme.TEXT}"),
                 bar(progress.percent, 100, theme.ACCENT), stats)
    return grid


LEVEL = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b")
LEVEL_COLOR = {"DEBUG": theme.MUTED, "INFO": theme.ACCENT, "WARNING": theme.WARN, "WARN": theme.WARN,
               "ERROR": theme.BAD, "CRITICAL": theme.BAD}
ALERT = re.compile(r"Traceback|Exception|\bERROR\b|out of memory|\bKilled\b|CUDA error", re.IGNORECASE)
GOOD_WORDS = re.compile(r"checkpoint saved|\bsaved\b|finished|\bdone\b", re.IGNORECASE)


def style_log_line(line: str) -> Text:
    text = Text(line[:400], style=theme.TEXT, no_wrap=True, overflow="ellipsis")
    stamp = STAMP.match(line)
    if stamp:
        text.stylize(theme.MUTED, 0, stamp.end())
    if ALERT.search(line):
        text.stylize(f"bold {theme.BAD}")
        return text
    level = LEVEL.search(line)
    if level:
        text.stylize(f"bold {LEVEL_COLOR[level.group(1)]}", level.start(), level.end())
    if "warn" in line.lower() and not level:
        text.stylize(theme.WARN)
    good = GOOD_WORDS.search(line)
    if good:
        text.stylize(theme.GOOD, good.start(), good.end())
    return text
