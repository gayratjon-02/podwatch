# podwatch

A live terminal dashboard for GPU pods. One panel per machine shows the GPU load, the newest training step with a loss
line chart, and the last lines of the newest log, and it all keeps moving while a run is going. It was written for keeping
an eye on several rented GPUs during fine-tuning runs, from one terminal window.

podwatch only reads. It never starts, stops or changes anything on a pod.

```
 podwatch   ● 1 running   GPU avg 91%   VRAM 20/48 GB                           MOCK DATA   17:07:21
╭─  pod1 ⠋ running  ───────────────────────────────────────────────────────────────────────────────╮
│ GPU ━━━━━━━━━━━━━━━━━━━╸    91%  VRAM ━━━━━━━━━              20.3/48 GB                          │
│ step 533 │ loss 1.202 │ ce 1.166 │ grad_norm 1.144 │ lr 8.3e-05                                  │
│ loss 1.202  ↓ 9% over last 120   min 1.163  max 1.400                                            │
│ █▇▇███▇▇▇▆▅▅▅▅▅▆▅▅▅▅▅▅▅▅▄▄▃▂▂▁ ▁▁                                                                │
│ ██████████████████████████████████▇▇▇▇▆▇▇▆▆▆▅▅▅▅▅▅▅▃▃▃▂▂▂▃▂▁▁▁▁                                  │
│ ███████████████████████████████████████████████████████████████████▇▆▆▇▆▆▅▅▅▅▅▅▆▅▅▅▄▃▂▂▂▂▃▄▃▃▂▂▁ │
│ ── driver.log · 4s ago                                                                           │
│ 2026-10-17 17:06:52,802 INFO train: step 505 loss 1.204 lr 8.5e-05                               │
│ 2026-10-17 17:06:55,802 INFO train: step 508 loss 1.171 lr 8.5e-05                               │
│ 2026-10-17 17:06:58,802 INFO train: step 511 loss 1.215 lr 8.5e-05                               │
│ 2026-10-17 17:07:01,802 INFO train: step 514 loss 1.236 lr 8.5e-05                               │
│ 2026-10-17 17:07:04,802 INFO train: step 517 loss 1.205 lr 8.4e-05                               │
│ 2026-10-17 17:07:07,802 INFO train: step 520 loss 1.192 lr 8.4e-05                               │
│ 2026-10-17 17:07:10,802 INFO train: step 523 loss 1.194 lr 8.4e-05                               │
│ 2026-10-17 17:07:13,802 INFO train: step 526 loss 1.229 lr 8.4e-05                               │
│ 2026-10-17 17:07:16,802 INFO train: step 529 loss 1.236 lr 8.4e-05                               │
╰────────────────────────────────────────────── exp-05/train/trainer_log.jsonl · updated 0.2s ago ─╯
 read-only · 1 pod · Ctrl+C quits
```

## Try it without any server

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
podwatch --mock
```

`--mock` draws four fake pods with moving data: one training, one running an evaluation, one training, and one that
drops off the network for a few seconds and comes back. `--pods 5` adds an idle pod. Nothing touches the network.
Without installing, `pip install rich` and `python -m podwatch --mock` work from the repository folder.

## Watch real pods

1. Copy `pods.example.toml` to `pods.toml` and put in your hosts. `pods.toml` is git-ignored.
2. Make sure `ssh -i <key> -p <port> user@host` works without a password prompt. podwatch runs ssh with `BatchMode=yes`,
   so use a key without a passphrase or load it into ssh-agent.
3. Run `podwatch --config pods.toml`.

Each pod keeps one ssh connection open. podwatch sends a small python program through it (nothing is installed on the
pod), and the program prints one JSON snapshot per second. If the connection drops, podwatch retries every 15 seconds.

What the pod needs: `python3` and a readable log directory. `nvidia-smi` is optional; without it the GPU row is left out.

What it looks at, inside the `root` directory you configure (default `/workspace`):

- the newest `*.log` file written in the last 30 minutes: its last lines are shown;
- the newest metrics file written in the last 30 minutes (default name `trainer_log.jsonl`, one JSON object per line):
  the last record is shown as `step N  key value ...` and one key (default `loss`) is drawn as a line chart with its trend.

Point `root` at the folder that holds your runs, not at a whole disk: the folder is walked every 10 seconds.
Folders named `.git`, `venv`, `.venv`, `site-packages`, `node_modules` and `__pycache__` are skipped.

## Watch this machine

```
podwatch --local --root ./runs
```

Runs the same program locally, no ssh. Useful for a training run on your own computer.

## Options

| Option | Meaning |
|---|---|
| `--config FILE` | watch the pods listed in a TOML file |
| `--local` | watch this machine; `--root DIR` and `--train-log NAME` choose where to look |
| `--mock` | fake pods; `--pods N` (1-8) and `--seed N` |
| `--lines N` | log lines per pod, default 8 |
| `--chart-key NAME` | metric drawn as the line chart, default `loss` |
| `--once` | print one frame and exit |

A panel with no metrics to chart (an evaluation, an idle pod) shows the GPU utilisation of the last two
minutes instead, on a fixed 0 to 100 scale, so a card that stalls between batches is easy to spot.

The states are: `running` (something was written in the last 60 seconds), `idle`, `connecting`, `unreachable`
and `no recent logs`.

## Troubleshooting

- `unreachable` right away: run the same ssh command by hand and read its error. A changed host key, a missing key file
  or a wrong port are the usual causes. Rented pods often get a new host key on the same address; remove the old line
  with `ssh-keygen -R "[host]:port"`.
- A pod shows `running` but no metrics line: the metrics file name differs from `train_log`, or its lines are not JSON.
- The screen looks broken: use a terminal that supports 256 colours and Unicode, and make it wide enough for two panels.

## Development

```
pip install -e .
python -m unittest discover -s tests -t .
```

The tests run the pod program through a real local `bash -s`, so the same path that runs over ssh is exercised.

Layout: `podwatch/agent.py` is the program sent to the pod, `streams.py` runs it and keeps the newest snapshot,
`render.py` turns snapshots into the screen, `mock.py` makes fake ones, `cli.py` wires it together.
