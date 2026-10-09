"""Safe, shared Gemini credential discovery and .env refresh support."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


_INDEXED_KEY = re.compile(r"^GEMINI_API_KEY_([1-9][0-9]*)$")


@dataclass(frozen=True)
class GeminiCredential:
    """A credential paired only with its safe provider identifier."""

    identifier: str
    key: str


@dataclass(frozen=True)
class GeminiCredentials:
    """Resolved primary and backup credentials, with no secret-derived IDs."""

    primaries: tuple[GeminiCredential, ...]
    backup: GeminiCredential | None
    legacy_only: bool = False

    def all_credentials(self) -> tuple[GeminiCredential, ...]:
        return self.primaries + ((self.backup,) if self.backup else ())


def _usable(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def discover_gemini_credentials(environ: Mapping[str, object]) -> GeminiCredentials:
    """Discover indexed Gemini credentials in numeric order without a limit.

    Only exact positive-integer suffixes are indexed.  Deduplication is by the
    actual credential value, never a logged prefix or a persisted fingerprint.
    """
    indexed: list[tuple[int, str, str]] = []
    for name, raw_value in environ.items():
        match = _INDEXED_KEY.fullmatch(name)
        value = _usable(raw_value)
        if match and value is not None:
            indexed.append((int(match.group(1)), name, value))
    indexed.sort(key=lambda item: (item[0], item[1]))

    seen: set[str] = set()
    primaries: list[GeminiCredential] = []
    for index, _name, key in indexed:
        if key not in seen:
            primaries.append(GeminiCredential(f"gemini-primary-{index}", key))
            seen.add(key)

    backup_key = _usable(environ.get("GEMINI_API_KEY_BACKUP"))
    legacy_key = _usable(environ.get("GEMINI_API_KEY"))

    # Preserve the existing semantic contract: legacy is a primary fallback
    # only when indexed primaries are absent.  It is still deduplicated if it
    # has the same value as the backup.
    if not primaries and legacy_key is not None and legacy_key not in seen:
        primaries.append(GeminiCredential("gemini-primary-1", legacy_key))
        seen.add(legacy_key)

    backup = None
    if backup_key is not None and backup_key not in seen:
        backup = GeminiCredential("gemini-backup", backup_key)

    return GeminiCredentials(
        tuple(primaries),
        backup,
        legacy_only=bool(legacy_key and not indexed and not backup_key),
    )


def gemini_secret_values(environ: Mapping[str, object]) -> tuple[str, ...]:
    """Return all configured Gemini secrets for redaction, without logging them."""
    values: list[str] = []
    for name, raw_value in environ.items():
        if _INDEXED_KEY.fullmatch(name) or name in {
            "GEMINI_API_KEY",
            "GEMINI_API_KEY_BACKUP",
        }:
            value = _usable(raw_value)
            if value is not None and value not in values:
                values.append(value)
    return tuple(values)


class GeminiCredentialSource:
    """Read the authoritative .env between requests without mutating os.environ."""

    def __init__(
        self,
        env_file: Path | None = None,
        environ: Mapping[str, object] | None = None,
    ) -> None:
        self.env_file = env_file
        self.environ = os.environ if environ is None else environ

    def discover(self) -> GeminiCredentials:
        # The project .env is intentionally authoritative for Gemini entries:
        # this lets an operator add *and remove* a key in a running process.
        values: dict[str, object] = dict(self.environ)
        if self.env_file is not None:
            # load_dotenv may have populated os.environ at startup.  Excluding
            # those names here makes later removals from the authoritative file
            # take effect instead of silently retaining a stale credential.
            values = {
                name: value for name, value in values.items()
                if not (_INDEXED_KEY.fullmatch(name)
                        or name in {"GEMINI_API_KEY", "GEMINI_API_KEY_BACKUP"})
            }
            if self.env_file.is_file():
                values.update(dotenv_values(self.env_file))
        return discover_gemini_credentials(values)

    def secret_values(self) -> tuple[str, ...]:
        values: dict[str, object] = dict(self.environ)
        if self.env_file is not None:
            values = {
                name: value for name, value in values.items()
                if not (_INDEXED_KEY.fullmatch(name)
                        or name in {"GEMINI_API_KEY", "GEMINI_API_KEY_BACKUP"})
            }
            if self.env_file.is_file():
                values.update(dotenv_values(self.env_file))
        return gemini_secret_values(values)
