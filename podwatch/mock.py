"""Fake pods, so the dashboard can be tried without a GPU, a server or an ssh key. Nothing here touches the network."""
from __future__ import annotations

import math
import random
import threading
import time
from collections import deque

from podwatch.snapshot import Snapshot
from podwatch.streams import Stream

# Pod 1 trains, pod 2 evaluates, pod 3 trains, pod 4 drops off and comes back; later pods repeat with an idle one.
KINDS = ("train", "eval", "train", "flaky", "idle", "train", "eval", "flaky")
MAX_PODS = len(KINDS)
EVAL_TOTAL = 1319
FLAKY_PERIOD = 40.0  # seconds: 28 up, 12 down
FLAKY_UP = 28.0
GPU_TOTAL_MIB = 48 * 1024
HISTORY = 120  # metric records kept, and pre-filled so the chart is full from the first frame
GPU_TARGET = {"train": 94.0, "flaky": 94.0, "eval": 78.0, "idle": 0.0}


class MockStream(Stream):
    def __init__(self, name: str, kind: str, lines: int, seed: int, halt: threading.Event) -> None:
        super().__init__(name, halt)
        self.kind, self.lines = kind, lines
        self.rng = random.Random(seed)
        self.started = time.time()
        self.step = self.rng.randint(80, 600)
        self.done = 0
        self.progress = ""
        self.util = GPU_TARGET[kind]
        self.records: deque[dict[str, object]] = deque(maxlen=HISTORY)
        self.log: deque[str] = deque(maxlen=lines)
        self.log_written = self.started
        self.run_name = f"exp-{self.rng.randint(1, 20):02d}/train/trainer_log.jsonl"
        self.ticks = 0
        self.prefill()

    def prefill(self) -> None:
        """Start with some history, like a pod that has already been running for a while."""
        if self.kind == "eval":
            self.write("INFO eval: model loaded, 1319 requests queued", self.started - HISTORY - 2)
        for back in range(HISTORY, 0, -1):
            self.ticks = back
            if self.kind == "idle":
                self.remember(0.0)
                continue
            if self.kind == "eval":
                self.eval_step(self.started - back)
            else:
                self.train_step(self.started - back)
            self.step_gpu()
            self.remember(self.util)
        self.ticks = 0

    def step_gpu(self) -> None:
        target = GPU_TARGET[self.kind]
        if self.kind == "eval" and self.ticks % 45 in (20, 21, 22):
            target = 12.0  # the card waits for the next batch
        self.util = min(100.0, max(3.0, self.util + (target - self.util) * 0.3 + self.rng.gauss(0, 2.5)))

    def run(self) -> None:
        while not self.halt.is_set():
            self.receive(self.tick())
            self.halt.wait(1.0)

    def tick(self) -> Snapshot:
        now = time.time()
        self.ticks += 1
        if self.kind == "flaky" and (now - self.started) % FLAKY_PERIOD > FLAKY_UP:
            return {"error": "not reachable (stopped or being recreated), retrying"}
        if self.kind == "idle":
            return self.idle_snapshot(now)
        if self.kind == "eval":
            self.eval_step(now)
        else:
            self.train_step(now)
        self.step_gpu()
        memory = (14.2 if self.kind == "eval" else 20.1 + self.rng.uniform(-0.3, 0.3)) * 1024
        snapshot: Snapshot = {
            "gpus": [[self.util, memory, GPU_TOTAL_MIB]],
            "log": {"name": "run.log" if self.kind == "eval" else "driver.log", "age": now - self.log_written,
                    "lines": list(self.log) + ([self.progress] if self.kind == "eval" else [])},
        }
        if self.kind != "eval":
            snapshot["train"] = {"run": self.run_name, "age": self.rng.uniform(0.1, 1.5), "records": list(self.records)}
        return snapshot

    def write(self, text: str, now: float) -> None:
        self.log.append(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))},{int(now % 1 * 1000):03d} {text}")
        self.log_written = now

    def train_step(self, now: float) -> None:
        self.step += 1
        loss = 0.55 + 2.1 * math.exp(-self.step / 450) + self.rng.gauss(0, 0.02)
        lr = 1e-4 * min(1.0, self.step / 60) * (0.5 + 0.5 * math.cos(min(self.step, 2000) / 2000 * math.pi))
        self.records.append({"step": self.step, "loss": round(loss, 4), "ce": round(loss * 0.97, 4),
                             "grad_norm": round(self.rng.uniform(0.4, 1.3), 3), "lr": lr})
        if self.ticks % 3 == 0:
            self.write(f"INFO train: step {self.step} loss {loss:.3f} lr {lr:.1e}", now)
        if self.step % 50 == 0:
            self.write(f"INFO train: checkpoint saved to checkpoints/step-{self.step:06d}", now)
        if self.step % 137 == 0:
            self.write("WARNING train: gradient norm spike, clipped to 1.0", now)

    def eval_step(self, now: float) -> None:
        self.done += self.rng.randint(1, 3)
        if self.done >= EVAL_TOTAL:
            self.done = 0
            self.write("INFO eval: pass finished, restarting", now)
        if self.ticks % 20 == 0:
            self.write(f"INFO eval: {self.done} requests scored, {self.rng.randint(58, 66)} per minute", now)
        pct = self.done * 100 // EVAL_TOTAL
        bar = "█" * (pct // 10) + "▒" * (1 if pct % 10 >= 5 else 0)
        rate = self.rng.uniform(0.95, 1.1)
        eta = (EVAL_TOTAL - self.done) / rate
        self.progress = (f"Running generate_until requests: {pct:3d}%|{bar:<10}| {self.done}/{EVAL_TOTAL} "
                         f"[{int(self.done / rate) // 60:02d}:{int(self.done / rate) % 60:02d}<"
                         f"{int(eta) // 60:02d}:{int(eta) % 60:02d}, {rate:.2f}it/s]")
        self.log_written = now

    def idle_snapshot(self, now: float) -> Snapshot:
        if not self.log:
            self.write("INFO train: finished, 2712 rows, final loss 0.612", self.started - 600)
            self.write("INFO stage: done in 1042.7 s", self.started - 599)
        return {"gpus": [[0.0, 0.4 * 1024, GPU_TOTAL_MIB]],
                "log": {"name": "driver.log", "age": now - self.log_written, "lines": list(self.log)}}


def mock_streams(count: int, lines: int, seed: int, halt: threading.Event) -> list[MockStream]:
    count = max(1, min(count, MAX_PODS))
    return [MockStream(f"pod{i + 1}", KINDS[i], lines, seed + i, halt) for i in range(count)]
