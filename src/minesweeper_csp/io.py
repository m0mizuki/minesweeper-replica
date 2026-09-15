"""Reproducible, atomic experiment input/output helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def to_jsonable(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def write_json_atomic(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write strict UTF-8 JSON and atomically replace the destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        to_jsonable(data), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(serialized)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_npz_atomic(path: str | Path, arrays: Mapping[str, NDArray]) -> None:
    """Write a compressed NumPy archive and atomically replace its destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(workdir: str | Path | None = None) -> str | None:
    """Return the current Git commit without failing outside a repository."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def git_is_dirty(workdir: str | Path | None = None) -> bool | None:
    """Return whether tracked or untracked worktree changes are present."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip())
