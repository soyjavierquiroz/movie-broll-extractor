"""Canonical, title-aware production-owner detection.

A PID in supervisor_state is evidence to inspect, never proof of ownership:
PIDs can be stale or reused.  A process owns a title only when its live command
is a movie-broll production command addressed to that exact canonical input.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable


_ACTIONS = {"run", "supervise", "process"}


def _command_target(values: list[str], cwd: Path) -> Path | None:
    """Return the title target for a movie-broll production command."""
    for index, value in enumerate(values[:-1]):
        if value not in _ACTIONS:
            continue
        launcher = values[:index]
        is_movie_broll = ("movie_broll.cli" in launcher
                          or any(Path(item).name == "movie-broll" for item in launcher))
        if not is_movie_broll:
            continue
        target = Path(values[index + 1])
        try:
            return target.resolve() if target.is_absolute() else (cwd / target).resolve()
        except OSError:
            return None
    return None


def _process_command(pid: int) -> tuple[list[str], Path] | None:
    proc = Path("/proc") / str(pid)
    try:
        values = [value.decode("utf-8", "surrogateescape")
                  for value in (proc / "cmdline").read_bytes().split(b"\0") if value]
        return values, (proc / "cwd").resolve()
    except OSError:
        return None


def _candidate_pids(state: dict[str, Any] | None) -> Iterable[int]:
    """Yield the recorded PID first, then all live PIDs, without duplicates."""
    seen: set[int] = set()
    owner_pid = state.get("owner_pid") if isinstance(state, dict) else None
    if isinstance(owner_pid, int) and owner_pid > 0:
        seen.add(owner_pid)
        yield owner_pid
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid not in seen:
            seen.add(pid)
            yield pid


def active_title_owner(input_dir: Path, state: dict[str, Any] | None = None,
                       *, exclude_pid: int | None = None) -> str | None:
    """Return a live, matching movie-broll owner description, if any.

    ``exclude_pid`` lets a run perform read-only preflight while holding its
    own supervisor lock; the caller cannot become its own conflicting owner.
    """
    expected = input_dir.resolve()
    for pid in _candidate_pids(state):
        if pid == exclude_pid:
            continue
        command = _process_command(pid)
        if command is None:
            continue
        values, cwd = command
        if _command_target(values, cwd) == expected:
            return f"pid {pid}: {' '.join(values)}"
    return None


def running_state(value: dict[str, Any] | None) -> bool:
    return bool(value and (value.get("status") == "RUNNING" or value.get("run_state") == "RUNNING"))

