"""Pod list from a TOML file. See pods.example.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PodConfig:
    name: str
    host: str
    port: int = 22
    user: str = "root"
    key: str | None = None
    root: str = "/workspace"  # remote directory scanned for logs
    train_log: str = "trainer_log.jsonl"  # file name of the per-step metrics, one JSON object per line


DEFAULT_KEYS = ("port", "user", "key", "root", "train_log")


def load_config(path: Path) -> list[PodConfig]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    defaults = {k: data[k] for k in DEFAULT_KEYS if k in data}
    pods: list[PodConfig] = []
    for index, raw in enumerate(data.get("pod", []), start=1):
        merged = {**defaults, **raw}
        unknown = set(merged) - set(PodConfig.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"pod #{index}: unknown keys {sorted(unknown)}")
        if "name" not in merged or "host" not in merged:
            raise ConfigError(f"pod #{index}: 'name' and 'host' are required")
        pod = PodConfig(**merged)
        if pod.host.startswith("-") or pod.user.startswith("-"):
            raise ConfigError(f"pod {pod.name!r}: host and user must not start with '-'")
        if not isinstance(pod.port, int) or not 0 < pod.port < 65536:
            raise ConfigError(f"pod {pod.name!r}: port must be 1-65535")
        pods.append(pod)
    if not pods:
        raise ConfigError(f"{path} has no [[pod]] tables")
    names = [p.name for p in pods]
    if len(set(names)) != len(names):
        raise ConfigError("pod names must be unique")
    return pods
