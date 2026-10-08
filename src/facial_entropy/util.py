"""Small shared helpers: stable seeds, hashing, atomic writes, JSON encoding."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_SEED = 20261008


def stable_seed(*parts: object, base: int = PRIMARY_SEED) -> int:
    """Derive a 32-bit seed from a base seed and stable string parts via SHA-256.

    Python's ``hash()`` is randomised per process, so it is never used here.
    """
    msg = ":".join([str(base), *[str(p) for p in parts]]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(msg).digest()[:4], "little")


def rng_for(*parts: object, base: int = PRIMARY_SEED) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts, base=base))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_savez(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def git_info(repo: str | Path) -> dict[str, Any]:
    """Commit and dirty-tree status of a git checkout (never raises)."""
    repo = Path(repo)
    out: dict[str, Any] = {"path": str(repo), "commit": None, "dirty": None, "dirty_files": []}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        files = [ln for ln in status.splitlines() if ln.strip()]
        out["dirty"] = bool(files)
        out["dirty_files"] = files
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        out["error"] = repr(exc)
    return out


def environment_info() -> dict[str, Any]:
    import importlib.metadata as md

    pkgs = {}
    for name in [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "scikit-image",
        "torch",
        "face-alignment",
        "forge-detect",
        "Pillow",
    ]:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": pkgs,
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        d = now - self.t0
        self.t0 = now
        return d


def dir_size_bytes(path: str | Path) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total
