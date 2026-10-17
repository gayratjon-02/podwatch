"""Shape of one snapshot: what the agent prints once a second and what the mock produces."""
from __future__ import annotations

from typing import TypedDict


class LogInfo(TypedDict):
    name: str
    age: float  # seconds since the file was last written
    lines: list[str]


class TrainInfo(TypedDict):
    run: str  # path of the metrics file, relative to the watched root
    age: float
    records: list[dict[str, object]]  # newest last, one dict per jsonl line


class Snapshot(TypedDict, total=False):
    gpus: list[list[float]]  # per GPU: [utilization %, memory used MiB, memory total MiB]
    log: LogInfo
    train: TrainInfo
    error: str  # set instead of the fields above while the pod cannot be read
