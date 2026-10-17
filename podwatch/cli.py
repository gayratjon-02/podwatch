"""Command line entry: pick the pods (config file, this machine or mock data) and draw the dashboard."""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from podwatch.agent import build_agent
from podwatch.config import ConfigError, PodConfig, load_config
from podwatch.mock import MAX_PODS, mock_streams
from podwatch.render import TWO_COLUMNS_FROM, frame
from podwatch.streams import LOCAL_COMMAND, ProcessStream, Stream, ssh_command

ONCE_WAIT_SECONDS = 25
ONCE_CELL_HEIGHT = 18


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="podwatch", description="Live terminal dashboard for GPU pods. Read-only.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="TOML file with the pods to watch (see pods.example.toml)")
    source.add_argument("--local", action="store_true", help="watch this machine instead of ssh pods")
    source.add_argument("--mock", action="store_true", help="fake pods with moving data, no network needed")
    parser.add_argument("--root", default=".", help="with --local: directory scanned for *.log and the metrics file")
    parser.add_argument("--train-log", default="trainer_log.jsonl", help="with --local: metrics file name (jsonl)")
    parser.add_argument("--pods", type=int, default=4, help=f"with --mock: how many fake pods (1-{MAX_PODS})")
    parser.add_argument("--seed", type=int, default=7, help="with --mock: random seed")
    parser.add_argument("--lines", type=int, default=30,
                        help="most log lines asked from each pod; the screen shows as many as fit")
    parser.add_argument("--chart-key", default="loss", help="metric drawn as the line chart")
    parser.add_argument("--once", action="store_true", help="print one frame and exit")
    parser.add_argument("--width", type=int, default=200, help="with --once: frame width in columns")
    return parser.parse_args(argv)


def build_streams(args: argparse.Namespace, halt: threading.Event) -> list[Stream]:
    if args.mock:
        return list(mock_streams(args.pods, args.lines, args.seed, halt))
    if args.local:
        root = str(Path(args.root).expanduser().resolve())
        agent = build_agent(root, args.lines, args.train_log)
        return [ProcessStream("local", LOCAL_COMMAND, agent, halt)]
    pods: list[PodConfig] = load_config(args.config)
    return [ProcessStream(p.name, ssh_command(p), build_agent(p.root, args.lines, p.train_log), halt) for p in pods]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    halt = threading.Event()
    try:
        streams = build_streams(args, halt)
    except ConfigError as exc:
        raise SystemExit(f"podwatch: {exc}")
    for stream in streams:
        stream.start()
    console = Console()
    try:
        if args.once:
            deadline = time.time() + ONCE_WAIT_SECONDS
            while time.time() < deadline and any(s.data.get("error") == "connecting…" for s in streams):
                time.sleep(0.2)
            time.sleep(1.2)  # let the first real snapshots arrive
            columns = 2 if args.width >= TWO_COLUMNS_FROM and len(streams) > 1 else 1
            height = -(-len(streams) // columns) * ONCE_CELL_HEIGHT + 2
            Console(width=args.width, height=height).print(frame(streams, args.chart_key, args.mock, (args.width, height)))
            return
        with Live(frame(streams, args.chart_key, args.mock, console.size), console=console, screen=True,
                  refresh_per_second=4) as live:
            while True:
                time.sleep(0.25)
                live.update(frame(streams, args.chart_key, args.mock, console.size))
    except KeyboardInterrupt:
        pass
    finally:
        halt.set()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
