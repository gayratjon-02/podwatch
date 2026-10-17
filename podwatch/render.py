"""Turns the newest snapshot of every pod into a screen-filling layout."""
from __future__ import annotations

import time
from datetime import datetime

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from podwatch import theme, widgets
from podwatch.snapshot import Snapshot
from podwatch.streams import Stream

ACTIVE_WITHIN = 60.0  # a log or metrics file written this recently means the pod is working
MIN_CELL_HEIGHT = 8
MIN_LOG_LINES = 3
TWO_COLUMNS_FROM = 110  # terminal width


def pod_state(data: Snapshot) -> tuple[str, str]:
    """(label, colour) of a pod."""
    if "error" in data:
        label = "connecting" if data["error"].startswith("connecting") else "unreachable"
    else:
        ages = [item["age"] for item in (data.get("log"), data.get("train")) if item]
        label = "no recent logs" if not ages else "running" if min(ages) < ACTIVE_WITHIN else "idle"
    return label, theme.STATE_COLOR.get(label, theme.MUTED)


def chart_height(room: int) -> int:
    """Rows for the loss chart, given the rows left after the fixed lines; the log keeps at least MIN_LOG_LINES."""
    for rows in (3, 2, 1):
        if room - (rows + 1) >= MIN_LOG_LINES:
            return rows
    return 0


def metric_block(train: dict, chart_key: str, width: int, rows: int) -> list[Text]:
    """Metrics line, then (when there is room) the chart of one metric with its latest value, trend and range."""
    records = train["records"]
    out: list[Text] = [widgets.metrics_text(records[-1])]
    series = [r[chart_key] for r in records if widgets.is_number(r.get(chart_key))]
    if rows and len(series) >= 2:
        head = Text(no_wrap=True, overflow="ellipsis")
        head.append(f"{chart_key} ", style=f"bold {theme.TEXT}")
        head.append(f"{series[-1]:.3f}  ", style=f"bold {theme.ACCENT}")
        head.append_text(widgets.trend_text(widgets.trend(series), chart_key, len(series)))
        head.append(f"   min {min(series):.3f}  max {max(series):.3f}", style=theme.MUTED)
        out.append(head)
        out += widgets.chart_rows(widgets.chart(widgets.smooth(series), width, rows))
    return out


def gpu_history_block(history: tuple[float, ...], width: int, rows: int) -> list[Text]:
    """Utilisation over the last couple of minutes on a fixed 0 to 100 scale, so a stalled card is plain to see."""
    if not rows or len(history) < 2:
        return []
    series = list(history)
    head = Text(no_wrap=True, overflow="ellipsis")
    head.append("GPU util ", style=f"bold {theme.TEXT}")
    head.append(f"{series[-1]:.0f}%  ", style=f"bold {theme.load_color(series[-1])}")
    head.append(f"avg {sum(series) / len(series):.0f}%  min {min(series):.0f}%   last {len(series)}s", style=theme.MUTED)
    return [head] + widgets.chart_rows(widgets.chart(series, width, rows, low=0, high=100), widgets.GPU_PALETTE)


def panel(stream: Stream, chart_key: str, cell_width: int, cell_height: int, now: float) -> Panel:
    data = stream.data
    state, color = pod_state(data)
    inner_width, inner_height = cell_width - 4, cell_height - 2
    mark = widgets.spinner_char(now) if state == "running" else "●"
    title = Text.assemble((f" {stream.name} ", f"bold {theme.TEXT}"), (f"{mark} {state} ", color))
    fresh = f"updated {now - stream.seen:.1f}s ago" if stream.seen else ""
    body: list = []
    subtitle = fresh
    if "error" in data:
        body.append(Text(data["error"], style=theme.MUTED))
        if stream.seen:
            body.append(Text(f"last data {now - stream.seen:.0f}s ago", style=theme.MUTED))
    else:
        gpus, train, log = data.get("gpus"), data.get("train"), data.get("log")
        lines = list(log["lines"]) if log else []
        progress = next((p for p in map(widgets.parse_progress, reversed(lines)) if p), None)
        lines = [line for line in lines if widgets.parse_progress(line) is None]
        used = 0  # rows already in the body, so the log gets exactly the rows that are left
        if gpus:
            body.append(widgets.gpu_rows(gpus, cell_width))
            used += len(gpus)
        if progress:
            body.append(widgets.progress_row(progress))
            used += 1
        if train and train["records"]:
            rows = chart_height(inner_height - used - 2)  # 2: the metrics line and the log header
            block = metric_block(train, chart_key, inner_width, rows)
            body += block
            used += len(block)
            subtitle = f"{train['run']} · {fresh}"
        elif len(stream.gpu_history) >= 2:  # nothing to chart from metrics: chart the GPU instead
            block = gpu_history_block(stream.gpu_history, inner_width, chart_height(inner_height - used - 1))
            body += block
            used += len(block)
        if log:
            body.append(Text.assemble((f"── {log['name']} ", theme.VIOLET), (f"· {log['age']:.0f}s ago", theme.MUTED)))
            room = max(0, inner_height - used - 1)
            body += [widgets.style_log_line(line) for line in lines[-room:]] if room else []
        if not log and not train:
            body.append(Text("no log written in the last 30 min", style=theme.MUTED))
    return Panel(Group(*body), title=title, title_align="left", subtitle=Text(subtitle, style=theme.MUTED),
                 subtitle_align="right", border_style=theme.BORDER.get(state, theme.BORDER_DEFAULT))


def header(streams: list[Stream], mock: bool, width: int) -> Table:
    states = [pod_state(s.data)[0] for s in streams]
    running, idle = states.count("running"), states.count("idle")
    down = sum(1 for s in states if s in ("unreachable", "connecting"))
    gpus = [g for s in streams for g in s.data.get("gpus", [])]
    left = Text(no_wrap=True, overflow="ellipsis")
    left.append(" podwatch ", style=f"bold #0b0e14 on {theme.ACCENT}")
    left.append(f"  ● {running} running", style=theme.GOOD)
    if idle:
        left.append(f"  ● {idle} idle", style=theme.WARN)
    if down:
        left.append(f"  ● {down} down", style=theme.BAD)
    if gpus:
        left.append(f"   GPU avg {sum(g[0] for g in gpus) / len(gpus):.0f}%", style=theme.TEXT)
        left.append(f"   VRAM {sum(g[1] for g in gpus) / 1024:.0f}/{sum(g[2] for g in gpus) / 1024:.0f} GB", style=theme.TEXT)
    right = Text(no_wrap=True, justify="right")
    if mock:
        right.append(" MOCK DATA ", style=f"bold #0b0e14 on {theme.WARN}")
    right.append(f"  {datetime.now():%H:%M:%S} ", style=theme.MUTED)
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(left, right)
    return grid


def footer(count: int) -> Text:
    return Text(f" read-only · {count} pod{'s' if count != 1 else ''} · Ctrl+C quits", style=theme.MUTED)


def frame(streams: list[Stream], chart_key: str, mock: bool, size: tuple[int, int]) -> Layout:
    """One screen of `size` columns by rows: header, an even grid of panels, footer."""
    width, height = size
    now = time.time()
    columns = 2 if width >= TWO_COLUMNS_FROM and len(streams) > 1 else 1
    rows = -(-len(streams) // columns)
    cell_width = width // columns
    cell_height = max(MIN_CELL_HEIGHT, (height - 2) // rows)
    grid = []
    for start in range(0, len(streams), columns):
        cells = [Layout(panel(s, chart_key, cell_width, cell_height, now)) for s in streams[start:start + columns]]
        cells += [Layout(Text(" "))] * (columns - len(cells))  # an empty renderable would draw a placeholder box
        row = Layout()
        row.split_row(*cells)
        grid.append(row)
    root = Layout()
    root.split_column(Layout(header(streams, mock, width), size=1), Layout(name="body"),
                      Layout(footer(len(streams)), size=1))
    root["body"].split_column(*grid)
    return root
