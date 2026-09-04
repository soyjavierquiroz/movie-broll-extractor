from pathlib import Path

import pytest

from movie_broll.cli import main
from movie_broll.reset import reset_run


def _fixture(tmp_path: Path):
    input_dir = tmp_path / 'input' / 'movie-a'
    run = tmp_path / 'runs' / 'movie-a'
    narrative = run / 'narrative-v2'

    input_dir.mkdir(parents=True)
    narrative.mkdir(parents=True)

    preserved = narrative / 'narrative_map.json'
    preserved.write_text('{"schema_version":"narrative_map_v2"}')

    (run / 'assets').mkdir()
    (run / 'assets' / 'rc001.mp4').write_bytes(b'asset')

    (run / 'semantic_checkpoints').mkdir()
    (run / 'semantic_checkpoints' / 'x.json').write_bytes(b'checkpoint')

    (run / '.work').mkdir()
    (run / '.work' / 'temp.bin').write_bytes(b'temp')

    (run / 'processing_ledger.json').write_bytes(b'ledger')
    (run / 'progress.jsonl').write_bytes(b'progress')

    (narrative / 'chunks').mkdir()
    (narrative / 'chunks' / 'chunk.json').write_bytes(b'chunk')
    (narrative / 'narrative_run.json').write_bytes(b'run')

    return input_dir, run, preserved


def test_reset_dry_run_changes_nothing(tmp_path):
    input_dir, run, preserved = _fixture(tmp_path)

    before = sorted(
        x.relative_to(run).as_posix()
        for x in run.rglob('*')
    )

    report = reset_run(input_dir, execute=False)

    after = sorted(
        x.relative_to(run).as_posix()
        for x in run.rglob('*')
    )

    assert report['status'] == 'PLANNED'
    assert report['mode'] == 'DRY_RUN'
    assert report['preserve']['path'] == 'narrative-v2/narrative_map.json'
    assert report['delete']
    assert before == after
    assert preserved.is_file()


def test_reset_execute_leaves_only_narrative_map(tmp_path):
    input_dir, run, preserved = _fixture(tmp_path)

    report = reset_run(input_dir, execute=True)

    assert report['status'] == 'COMPLETE'
    assert preserved.is_file()

    assert sorted(
        x.relative_to(run).as_posix()
        for x in run.rglob('*')
    ) == [
        'narrative-v2',
        'narrative-v2/narrative_map.json',
    ]


def test_reset_refuses_when_narrative_map_missing(tmp_path):
    input_dir, run, preserved = _fixture(tmp_path)
    preserved.unlink()

    with pytest.raises(RuntimeError, match='narrative_map.json is missing'):
        reset_run(input_dir, execute=True)

    assert (run / 'assets' / 'rc001.mp4').is_file()
    assert (run / 'processing_ledger.json').is_file()


def test_reset_refuses_noncanonical_input_directory(tmp_path):
    input_dir = tmp_path / 'somewhere' / 'movie-a'
    input_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match='canonical input'):
        reset_run(input_dir, execute=False)


def test_reset_cli_dry_run_reports_preserve_and_delete(tmp_path,capsys):
    input_dir, _, _ = _fixture(tmp_path)

    rc = main([
        'reset',
        str(input_dir),
        '--dry-run',
    ])

    output = capsys.readouterr().out

    assert rc == 0
    assert '[reset] mode: DRY_RUN' in output
    assert '[reset] PRESERVE narrative-v2/narrative_map.json' in output
    assert '[reset] DELETE assets' in output
    assert '[reset] status: PLANNED' in output
