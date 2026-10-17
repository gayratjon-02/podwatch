"""One background thread per pod. Each keeps the newest snapshot in `data` and the time it arrived in `seen`."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

from podwatch.config import PodConfig
from podwatch.snapshot import Snapshot

RETRY_SECONDS = 15
GPU_HISTORY = 120  # snapshots kept per pod, about two minutes


class Stream(threading.Thread):
    def __init__(self, name: str, halt: threading.Event) -> None:
        super().__init__(daemon=True)
        self.name, self.halt = name, halt
        self.data: Snapshot = {"error": "connecting…"}
        self.seen = 0.0
        # Mean utilisation over the GPUs, newest last. A tuple that is replaced, never changed in place, so the
        # drawing thread can read it while this thread appends.
        self.gpu_history: tuple[float, ...] = ()

    def remember(self, util: float) -> None:
        self.gpu_history = (*self.gpu_history, util)[-GPU_HISTORY:]

    def receive(self, data: Snapshot) -> None:
        for key, field in (("log", "lines"), ("train", "records")):
            item = data.get(key)
            if item and item.pop("same", False):  # the pod says this file is unchanged: keep what we have
                item[field] = (self.data.get(key) or {}).get(field, [])
        self.data, self.seen = data, time.time()
        if data.get("gpus"):
            self.remember(sum(gpu[0] for gpu in data["gpus"]) / len(data["gpus"]))

    def close(self) -> None:
        """Release whatever the stream holds (a child process); nothing by default."""


class ProcessStream(Stream):
    """Runs the agent through `command` (ssh or a local bash) and reconnects after a drop."""

    def __init__(self, name: str, command: list[str], agent: str, halt: threading.Event) -> None:
        super().__init__(name, halt)
        self.command, self.agent = command, agent
        self.proc: subprocess.Popen | None = None

    def run(self) -> None:
        while not self.halt.is_set():
            error = "not reachable (stopped or being recreated), retrying"
            try:
                self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL, text=True)
                self.proc.stdin.write(self.agent)
                self.proc.stdin.close()
                for line in self.proc.stdout:
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue  # ssh banners and other noise
                    if isinstance(data, dict):
                        self.receive(data)
                    if self.halt.is_set():
                        break
            except OSError as exc:
                error = f"cannot start {self.command[0]}: {exc}"
            except ValueError:
                pass  # close() ran from the main thread while this one was reading
            finally:
                self.close()
            if not self.halt.is_set():
                self.data = {"error": error}
                self.halt.wait(RETRY_SECONDS)

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for pipe in (proc.stdin, proc.stdout):
            if pipe and not pipe.closed:
                pipe.close()


def ssh_command(pod: PodConfig) -> list[str]:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=accept-new"]
    if pod.key:
        command += ["-i", str(Path(pod.key).expanduser())]
    if pod.jump:
        command += ["-J", pod.jump]
    if pod.port:
        command += ["-p", str(pod.port)]
    target = f"{pod.user}@{pod.host}" if pod.user else pod.host
    return command + [target, "bash", "-s"]


LOCAL_COMMAND = ["bash", "-s"]
