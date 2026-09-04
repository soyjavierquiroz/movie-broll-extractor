"""Safe reset of derived full-movie production state."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def _path_size(path: Path) -> int:
    """Return disk bytes without following directory symlinks."""
    try:
        if path.is_symlink() or path.is_file():
            return path.lstat().st_size
        if not path.is_dir():
            return 0

        total = 0
        for root, dirs, files in os.walk(path, followlinks=False):
            root_path = Path(root)

            for name in files:
                file = root_path / name
                try:
                    total += file.lstat().st_size
                except OSError:
                    pass

            # os.walk does not descend through symlink dirs with
            # followlinks=False, but count the link itself.
            for name in list(dirs):
                item = root_path / name
                if item.is_symlink():
                    try:
                        total += item.lstat().st_size
                    except OSError:
                        pass

        return total

    except OSError:
        return 0


def _resolve_reset_paths(input_dir: Path) -> tuple[Path, Path, Path, Path]:
    input_dir = input_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f'input directory does not exist: {input_dir}'
        )

    if input_dir.parent.name != 'input':
        raise ValueError(
            'reset requires canonical input/<movie-id> directory'
        )

    movie_id = input_dir.name
    root = input_dir.parent.parent.resolve()
    runs_root = (root / 'runs').resolve()
    run = (runs_root / movie_id).resolve()

    # Strong ownership boundary: reset may operate only on the matching
    # direct child of this repository's runs/ directory.
    if run.parent != runs_root or run == runs_root:
        raise ValueError(
            'refusing reset outside canonical runs/<movie-id> directory'
        )

    if not run.is_dir():
        raise FileNotFoundError(
            f'run directory does not exist: {run}'
        )

    preserve = run / 'narrative-v2' / 'narrative_map.json'

    if preserve.is_symlink() or not preserve.is_file():
        raise RuntimeError(
            'refusing reset: canonical '
            'narrative-v2/narrative_map.json is missing'
        )

    return input_dir, root, run, preserve


def _deletion_targets(run: Path, preserve: Path) -> list[Path]:
    """Everything in run except the exact preservation allowlist."""
    targets: list[Path] = []

    for child in sorted(run.iterdir(), key=lambda p: p.name):
        if child.name != 'narrative-v2':
            targets.append(child)
            continue

        if not child.is_dir() or child.is_symlink():
            raise RuntimeError(
                'refusing reset: narrative-v2 is not a normal directory'
            )

        for nested in sorted(child.iterdir(), key=lambda p: p.name):
            if nested != preserve:
                targets.append(nested)

    return targets


def _delete_owned(path: Path, run: Path) -> None:
    # Deliberately do not resolve path: resolving a symlink would inspect
    # its target.  We only need lexical ownership because unlinking the
    # symlink itself is safe.
    absolute = path.absolute()
    run_absolute = run.absolute()

    if not absolute.is_relative_to(run_absolute) or absolute == run_absolute:
        raise RuntimeError(
            f'refusing to delete path outside owned run: {path}'
        )

    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def reset_run(input_dir: Path, *, execute: bool = False) -> dict[str, Any]:
    """Plan or execute a clean rebuild while preserving the narrative map."""
    input_dir, root, run, preserve = _resolve_reset_paths(input_dir)
    targets = _deletion_targets(run, preserve)

    delete_entries = [
        {
            'path': target.relative_to(run).as_posix(),
            'size_bytes': _path_size(target),
        }
        for target in targets
    ]

    report: dict[str, Any] = {
        'status': 'PLANNED',
        'mode': 'EXECUTE' if execute else 'DRY_RUN',
        'movie_id': input_dir.name,
        'root': str(root),
        'run': str(run),
        'preserve': {
            'path': preserve.relative_to(run).as_posix(),
            'size_bytes': _path_size(preserve),
        },
        'delete': delete_entries,
        'delete_bytes': sum(x['size_bytes'] for x in delete_entries),
    }

    if not execute:
        return report

    # Re-check preservation immediately before the destructive phase.
    if preserve.is_symlink() or not preserve.is_file():
        raise RuntimeError(
            'refusing reset: preservation file disappeared before execution'
        )

    for target in targets:
        _delete_owned(target, run)

    # Strong postcondition: exactly one top-level directory and one file.
    top_level = sorted(
        x.relative_to(run).as_posix()
        for x in run.iterdir()
    )
    narrative_entries = sorted(
        x.relative_to(run).as_posix()
        for x in (run / 'narrative-v2').iterdir()
    )

    if top_level != ['narrative-v2']:
        raise RuntimeError(
            f'reset postcondition failed at run root: {top_level}'
        )

    if narrative_entries != ['narrative-v2/narrative_map.json']:
        raise RuntimeError(
            'reset postcondition failed inside narrative-v2: '
            f'{narrative_entries}'
        )

    if not preserve.is_file() or preserve.is_symlink():
        raise RuntimeError(
            'reset postcondition failed: narrative map was not preserved'
        )

    report['status'] = 'COMPLETE'
    return report
