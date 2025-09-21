from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0 or subprocess.call(["git", "diff", "--cached", "--quiet"]) != 0
        info.update({"commit": commit, "branch": branch, "dirty": dirty})
    except Exception:
        pass
    return info


def get_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        import torch  # type: ignore

        versions["torch"] = getattr(torch, "__version__", "unknown")
    except Exception:
        versions["torch"] = "missing"
    try:
        import gymnasium as gym  # type: ignore

        versions["gymnasium"] = getattr(gym, "__version__", "unknown")
    except Exception:
        versions["gymnasium"] = "missing"
    try:
        import numpy as np  # type: ignore

        versions["numpy"] = getattr(np, "__version__", "unknown")
    except Exception:
        versions["numpy"] = "missing"
    try:
        import tensorboard as tb  # type: ignore

        versions["tensorboard"] = getattr(tb, "__version__", "unknown")
    except Exception:
        versions["tensorboard"] = "missing"
    return versions


def write_json(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def default(o: Any):
        if is_dataclass(o):
            return asdict(o)
        return str(o)
    with p.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=default)


def append_jsonl(path: Path | str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        json.dump(data, f)
        f.write("\n")

