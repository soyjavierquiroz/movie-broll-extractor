"""Canonical CLI environment bootstrap; never logs credential values."""
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parents[2]


def load_environment() -> None:
    """Load MBE's own .env before dispatch, preserving process overrides.

    Resolve from the installed source location, never the caller's directory.
    Each CLI invocation calls this once; providers only consume configuration.
    """
    load_dotenv(APP_ROOT / ".env", override=False)
