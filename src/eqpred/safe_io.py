"""Write JSON stores so a crash or a second run cannot leave them half-written."""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

STALE_LOCK_SECONDS = 120.0


def write_json(path: Path, payload: Any, *, indent: int | None = 2, backup: bool = False, default=None) -> None:
    """Write a temporary file, then swap it in. A reader sees the old file or the new one, never half."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=indent, default=default))
        handle.flush()
        os.fsync(handle.fileno())
    if backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    replace_file(tmp, path)


def replace_file(src: Path, dest: Path, tries: int = 40) -> None:
    for attempt in range(tries):
        try:
            os.replace(src, dest)
            return
        except PermissionError:
            # Windows refuses while a reader holds the file open for a moment.
            if attempt == tries - 1:
                raise
            time.sleep(0.05)


def read_json(path: Path) -> Any:
    """Read a JSON store, naming the backup if the file is damaged."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        backup = path.with_name(path.name + ".bak")
        hint = f" The previous copy is {backup.name}." if backup.exists() else ""
        raise ValueError(f"{path.name} is damaged ({exc}).{hint}") from exc


@contextmanager
def file_lock(target: Path, timeout: float = 180.0) -> Iterator[None]:
    """One writer at a time across processes. A lock left by a killed run clears after two minutes."""
    lock = Path(target).with_name(Path(target).name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > STALE_LOCK_SECONDS:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"{lock.name} is held by another run")
            time.sleep(0.1)
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        break
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
