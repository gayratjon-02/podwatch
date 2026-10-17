"""The small program that runs on the pod (or on this machine with --local).

It is sent over the ssh connection and started with `bash -s`, so nothing is installed on the pod. It needs python3;
nvidia-smi is optional. Once a second it prints one JSON line: GPU load, the newest log file's last lines and the newest
metrics file's last records. It only reads files; it never writes or stops anything.
"""
from __future__ import annotations

import base64
import json

# The config travels as base64 inside a quoted heredoc, so no pod path or file name is ever interpreted by a shell.
AGENT_TEMPLATE = r"""python3 -u - <<'PY'
import base64, json, os, subprocess, time

CFG = json.loads(base64.b64decode("__CFG__").decode())
ROOT, LINES, TRAIN = CFG["root"], CFG["lines"], CFG["train_log"]
RECENT = 1800  # ignore files that were not written for 30 minutes
SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", "site-packages"}


def scan():
    now, logs, trains = time.time(), [], []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            is_train = name == TRAIN
            if not (is_train or name.endswith(".log")):
                continue
            path = os.path.join(base, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if now - mtime < RECENT:
                (trains if is_train else logs).append((mtime, path))
    newest = lambda items: max(items)[1] if items else None
    return newest(logs), newest(trains)


def tail(path, n):
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 60000))
        text = fh.read().decode("utf-8", "replace")
    lines = []
    for physical in text.split("\n"):
        parts = [s for s in physical.split("\r") if s.strip()]  # a carriage return overwrites the line
        if parts:
            lines.append(parts[-1])
    return lines[-n:]


def log_lines(path, n):
    # the last n ordinary lines, then the newest progress bar line (tqdm style) if there is one
    lines = tail(path, 400)
    bars = [s for s in lines if "%|" in s]
    return [s for s in lines if "%|" not in s][-n:] + bars[-1:]


def gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        rows = [[float(x) for x in line.split(",")] for line in out.splitlines() if line.strip()]
        return rows or None
    except Exception:
        return None


log = train = None
scanned = 0.0
while True:
    now = time.time()
    if now - scanned > 10:
        log, train = scan()
        scanned = now
    out = {}
    rows = gpus()
    if rows:
        out["gpus"] = rows
    try:
        if log:
            out["log"] = {"name": os.path.basename(log), "age": now - os.path.getmtime(log), "lines": log_lines(log, LINES)}
        if train:
            records = []
            for line in tail(train, 120):
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
            out["train"] = {"age": now - os.path.getmtime(train), "records": records,
                            "run": os.path.relpath(train, ROOT)}
    except OSError:
        scanned = 0.0  # a file vanished: scan again on the next turn
    print(json.dumps(out), flush=True)
    time.sleep(1)
PY
"""


def build_agent(root: str, lines: int, train_log: str) -> str:
    cfg = json.dumps({"root": root, "lines": lines, "train_log": train_log})
    return AGENT_TEMPLATE.replace("__CFG__", base64.b64encode(cfg.encode()).decode())
