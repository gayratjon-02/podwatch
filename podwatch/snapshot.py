"""Shape of one snapshot: what the agent prints once a second and what the mock produces."""
from __future__ import annotations

from typing import NotRequired, TypedDict


class LogInfo(TypedDict):
    name: str
    age: float  # seconds since the file was last written
    lines: NotRequired[list[str]]  # left out when `same` is set
    same: NotRequired[bool]  # on the wire only: the lines have not changed since the last full snapshot


class TrainInfo(TypedDict):
    run: str  # path of the metrics file, relative to the watched root
    age: float
    records: NotRequired[list[dict[str, object]]]  # newest last, one dict per jsonl line; left out when `same`
    same: NotRequired[bool]


class Snapshot(TypedDict, total=False):
    gpus: list[list[float]]  # per GPU: [utilization %, memory used MiB, memory total MiB]
    log: LogInfo
    train: TrainInfo
    error: str  # set instead of the fields above while the pod cannot be read
