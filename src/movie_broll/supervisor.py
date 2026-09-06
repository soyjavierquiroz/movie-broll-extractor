"""Durable single-child supervisor for resumable production runs."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

STALE_TIMEOUT_SECONDS = 1800
BACKOFF_SECONDS = (30, 60, 120, 300)


def _root(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1]


class Supervisor:
    def __init__(self, input_dir: Path, *, stale_timeout_seconds: float = STALE_TIMEOUT_SECONDS,
                 grace_period_seconds: float = 30, poll_interval_seconds: float = 1,
                 clock: Callable[[], float] = time.monotonic, sleeper: Callable[[float], None] = time.sleep,
                 process_factory: Callable[..., Any] = subprocess.Popen,
                 progress_reader: Callable[[Path], dict[str, Any] | None] | None = None,
                 killer: Callable[[int, int], None] = os.killpg) -> None:
        self.input_dir=input_dir; self.run_dir=_root(input_dir)/"runs"/input_dir.name
        self.stale_timeout_seconds=stale_timeout_seconds; self.grace_period_seconds=grace_period_seconds
        self.poll_interval_seconds=poll_interval_seconds; self.clock=clock; self.sleeper=sleeper
        self.process_factory=process_factory; self.progress_reader=progress_reader or self._read_summary; self.killer=killer
        self.child=None; self.shutdown_requested=False; self._log_handle=None

    def _log(self, message: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "supervisor.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[supervisor] {message}\n"); handle.flush(); os.fsync(handle.fileno())

    @staticmethod
    def _read_summary(path: Path) -> dict[str, Any] | None:
        try: return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): return None

    def _summary(self) -> dict[str, Any] | None:
        return self.progress_reader(self.run_dir / "progress_summary.json")

    @staticmethod
    def _complete(summary: dict[str, Any] | None) -> bool:
        if not summary or summary.get("status") != "COMPLETE": return False
        complete,total=summary.get("segments_complete"),summary.get("segments_total")
        return isinstance(complete, int) and isinstance(total, int) and complete == total

    @staticmethod
    def _fingerprint(summary: dict[str, Any] | None) -> tuple[Any, Any, Any]:
        if not summary: return (None, None, None)
        return (summary.get("segments_complete"), summary.get("segments_total"), summary.get("status"))

    def _spawn(self) -> Any:
        child_log=(self.run_dir / "process.log").open("ab")
        try:
            return self.process_factory([sys.executable, "-m", "movie_broll.cli", "process", str(self.input_dir)],
                stdout=child_log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            child_log.close()

    def _terminate_child(self) -> None:
        child=self.child
        if child is None or child.poll() is not None: return
        pgid=child.pid
        self.killer(pgid, signal.SIGTERM)
        deadline=self.clock()+self.grace_period_seconds
        while child.poll() is None and self.clock()<deadline: self.sleeper(self.poll_interval_seconds)
        if child.poll() is None: self.killer(pgid, signal.SIGKILL)

    def request_shutdown(self, signum: int) -> None:
        self.shutdown_requested=True; self._log(f"SIGNAL signal={signal.Signals(signum).name}")
        self._terminate_child()

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        self.request_shutdown(signum)

    def _wait_for_child(self) -> tuple[int, bool]:
        progress=self.run_dir / "progress.jsonl"
        changed_at=self.clock()
        try: observed=progress.stat().st_mtime_ns
        except OSError: observed=None
        while self.child.poll() is None and not self.shutdown_requested:
            try: current=progress.stat().st_mtime_ns
            except OSError: current=None
            if current != observed: observed=current; changed_at=self.clock()
            age=self.clock()-changed_at
            if age >= self.stale_timeout_seconds:
                self._log(f"WATCHDOG_STALE age_seconds={age:.3f}"); self._terminate_child(); return self.child.poll() or -signal.SIGTERM, True
            self.sleeper(self.poll_interval_seconds)
        return self.child.poll() if self.child.poll() is not None else -signal.SIGTERM, False

    def run(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        lock=(self.run_dir / "supervisor.lock").open("a+")
        try:
            try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error: raise RuntimeError(f"another supervisor is active for {self.input_dir.name}") from error
            old_int,old_term=signal.signal(signal.SIGINT,self._signal_handler),signal.signal(signal.SIGTERM,self._signal_handler)
            try:
                generation=1; restart_streak=0; previous=self._fingerprint(self._summary())
                while not self.shutdown_requested:
                    self._log(f"{'START' if generation == 1 else 'RESTART'} generation={generation}")
                    self.child=self._spawn()
                    rc,stale=self._wait_for_child(); self._log(f"EXIT generation={generation} rc={rc}")
                    if self.shutdown_requested: return 128
                    summary=self._summary()
                    if self._complete(summary):
                        self._log(f"COMPLETE segments={summary['segments_complete']}/{summary['segments_total']}"); return 0
                    current=self._fingerprint(summary)
                    restart_streak=0 if current != previous else restart_streak+1; previous=current
                    delay=BACKOFF_SECONDS[min(generation-1, len(BACKOFF_SECONDS)-1)]
                    self._log(f"BACKOFF seconds={delay} consecutive_restarts_without_progress={restart_streak}")
                    self.sleeper(delay); generation+=1; self.child=None
                return 128
            finally:
                signal.signal(signal.SIGINT,old_int); signal.signal(signal.SIGTERM,old_term)
        finally:
            lock.close()


def supervise(input_dir: Path, **kwargs: Any) -> int:
    return Supervisor(input_dir, **kwargs).run()
