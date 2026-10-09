"""Durable single-child supervisor for resumable production runs."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .production_ownership import active_title_owner, running_state

STALE_TIMEOUT_SECONDS = 1800
BACKOFF_SECONDS = (30, 60, 120, 300)
MAX_IDENTICAL_UNCLASSIFIED_FAILURES = 5


def _root(input_dir: Path) -> Path:
    return input_dir.resolve().parents[1]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _active_production_owner(input_dir: Path, *, exclude_pid: int | None = None) -> str | None:
    """Return an orphaned production child's PID and command, if one exists.

    The flock is the canonical supervisor ownership mechanism.  This extra
    check closes the small window where a supervisor was killed but its child
    survived outside the supervisor process group.
    """
    return active_title_owner(input_dir, exclude_pid=exclude_pid)


def _active_production_process(input_dir: Path) -> bool:
    return _active_production_owner(input_dir, exclude_pid=os.getpid()) is not None


class Supervisor:
    def __init__(self, input_dir: Path, *, stale_timeout_seconds: float = STALE_TIMEOUT_SECONDS,
                 grace_period_seconds: float = 30, poll_interval_seconds: float = 1,
                 clock: Callable[[], float] = time.monotonic, sleeper: Callable[[float], None] = time.sleep,
                 process_factory: Callable[..., Any] = subprocess.Popen,
                 progress_reader: Callable[[Path], dict[str, Any] | None] | None = None,
                 killer: Callable[[int, int], None] = os.killpg,
                 active_process_checker: Callable[[Path], bool] = _active_production_process,
                 before_start: Callable[[], None] | None = None,
                 stream_child_output: bool = False) -> None:
        self.input_dir=input_dir; self.run_dir=_root(input_dir)/"runs"/input_dir.name
        self.stale_timeout_seconds=stale_timeout_seconds; self.grace_period_seconds=grace_period_seconds
        self.poll_interval_seconds=poll_interval_seconds; self.clock=clock; self.sleeper=sleeper
        self.process_factory=process_factory; self.progress_reader=progress_reader or self._read_summary; self.killer=killer
        self.active_process_checker=active_process_checker
        self.before_start=before_start; self.stream_child_output=stream_child_output
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

    def _outcome(self) -> dict[str, Any] | None:
        try:
            value = json.loads((self.run_dir / "process_outcome.json").read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _state(self, **values: Any) -> None:
        path = self.run_dir / "supervisor_state.json"
        existing = self._read_summary(path) or {}
        existing.update(values)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _running(value: dict[str, Any] | None) -> bool:
        return running_state(value)

    def _update_json(self, path: Path, values: dict[str, Any]) -> None:
        """Atomically update operational fields without disturbing progress data."""
        existing = self._read_summary(path) or {}
        existing.update(values)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def _recover_stale_runtime_state(self) -> bool:
        """Mark a lockless prior RUNNING execution as safely interrupted.

        This runs only after this supervisor owns the canonical lock and after
        orphan-child detection.  It deliberately touches no caches, assets, or
        production checkpoints.
        """
        summary_path = self.run_dir / "progress_summary.json"
        state_path = self.run_dir / "supervisor_state.json"
        summary, state = self._read_summary(summary_path), self._read_summary(state_path)
        if not self._running(summary) and not self._running(state):
            return False
        recovered_at = _utc()
        for path, value in ((summary_path, summary), (state_path, state)):
            if not self._running(value):
                continue
            update: dict[str, Any] = {
                "status": "INTERRUPTED",
                "run_state": "INTERRUPTED",
                "resume_safe": True,
                "interruption_reason": "stale_runtime_state_recovered",
                "recovered_at": recovered_at,
            }
            if value and value.get("stage") is not None:
                update["previous_stage"] = value["stage"]
            self._update_json(path, update)
        self._log("RECOVERED_STALE_RUNTIME_STATE status=INTERRUPTED resume_safe=true")
        return True

    def _terminal_summary(self, classification: str, reason: str) -> None:
        """Mirror terminal supervision into the user-facing progress summary."""
        path=self.run_dir / "progress_summary.json"
        summary=self._read_summary(path) or {}
        summary.update(status="FAILED",run_state="FAILED",failure_classification=classification,
                       failure_reason=reason)
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('w',encoding='utf-8') as handle:
            json.dump(summary,handle,sort_keys=True)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        temporary.replace(path)

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
        (self.run_dir / "process_outcome.json").unlink(missing_ok=True)
        if self.stream_child_output:
            return self.process_factory([sys.executable, "-m", "movie_broll.cli", "process", str(self.input_dir)],
                start_new_session=True)
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
            except BlockingIOError as error:
                state=self._read_summary(self.run_dir / "supervisor_state.json") or {}
                owner=state.get("owner_pid")
                command=state.get("owner_command")
                detail=f" (owner pid {owner}{'; ' + command if command else ''})" if owner else ""
                raise RuntimeError(f"another supervisor is active for {self.input_dir.name}{detail}") from error
            if self.active_process_checker(self.input_dir):
                owner=_active_production_owner(self.input_dir, exclude_pid=os.getpid())
                detail=f" ({owner})" if owner else ""
                raise RuntimeError(f"an active production process is already running for {self.input_dir.name}{detail}")
            self._recover_stale_runtime_state()
            if self.before_start is not None:
                self.before_start()
            old_int,old_term=signal.signal(signal.SIGINT,self._signal_handler),signal.signal(signal.SIGTERM,self._signal_handler)
            try:
                self._state(status="RUNNING", run_state="RUNNING", started_at=_utc(), owner_pid=os.getpid(),
                            owner_command="movie-broll run" if self.stream_child_output else "movie-broll supervise")
                generation=1; restart_streak=0; previous=self._fingerprint(self._summary()); identical_failures=0; failure_key=None
                while not self.shutdown_requested:
                    self._log(f"{'START' if generation == 1 else 'RESTART'} generation={generation}")
                    self.child=self._spawn()
                    rc,stale=self._wait_for_child(); self._log(f"EXIT generation={generation} rc={rc}")
                    if self.shutdown_requested: return 128
                    summary=self._summary()
                    if self._complete(summary):
                        self._state(status="COMPLETE", run_state="COMPLETE", generation=generation, completed_at=_utc())
                        self._log(f"COMPLETE segments={summary['segments_complete']}/{summary['segments_total']}"); return 0
                    outcome = self._outcome()
                    classification = outcome.get("classification") if outcome else None
                    reason = str(outcome.get("reason", "child exited without a structured outcome")) if outcome else "child exited without a structured outcome"
                    if classification == "DETERMINISTIC":
                        self._terminal_summary(classification,reason)
                        self._state(status="TERMINAL", generation=generation, terminal_reason=reason,
                                    classification=classification, exit_code=rc)
                        self._log(f"TERMINAL generation={generation} classification=DETERMINISTIC reason={reason}")
                        return rc if rc else 2
                    current=self._fingerprint(summary)
                    restart_streak=0 if current != previous else restart_streak+1; previous=current
                    transient = classification == "TRANSIENT" or stale
                    key = (rc, reason) if not transient and rc != 0 else None
                    identical_failures = identical_failures + 1 if key == failure_key else (1 if key else 0)
                    failure_key = key
                    if identical_failures >= MAX_IDENTICAL_UNCLASSIFIED_FAILURES:
                        self._terminal_summary("UNCLASSIFIED_NO_PROGRESS",reason)
                        self._state(status="TERMINAL", generation=generation, terminal_reason=reason,
                                    classification="UNCLASSIFIED_NO_PROGRESS", exit_code=rc,
                                    identical_failure_count=identical_failures)
                        self._log(f"TERMINAL generation={generation} classification=UNCLASSIFIED_NO_PROGRESS identical_failures={identical_failures}")
                        return rc if rc else 2
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
