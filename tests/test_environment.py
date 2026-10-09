"""Credential bootstrap checks use isolated files and mocked provider clients."""
import os
from unittest.mock import Mock

import pytest

from movie_broll import cli, environment, production_run
from movie_broll import broll_semantics as semantics
from movie_broll.narrative_provider import OpenAINarrativeProvider
from movie_broll.production_preflight import preflight, print_preflight
from movie_broll.gemini_credentials import discover_gemini_credentials


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(environment, "APP_ROOT", tmp_path)
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "NARRATIVE_PROVIDER_MODE", "NARRATIVE_PROVIDER", "SEMANTIC_PROVIDER", "GEMINI_API_KEY_42"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path.parent)
    return tmp_path


def test_root_is_source_relative():
    assert environment.APP_ROOT / "src" / "movie_broll" == __import__('pathlib').Path(environment.__file__).resolve().parent


@pytest.mark.parametrize("override", [None, "fake-shell-key", ""])
def test_cli_loads_root_once_and_process_wins(root, monkeypatch, capsys, override):
    (root / ".env").write_text("OPENAI_API_KEY=fake-file-key\nGEMINI_API_KEY_42=fake-gemini-key\n")
    if override is not None:
        monkeypatch.setenv("OPENAI_API_KEY", override)
    loader = Mock(wraps=environment.load_environment)
    monkeypatch.setattr(environment, "load_environment", loader)
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0
    loader.assert_called_once_with()
    assert os.environ["OPENAI_API_KEY"] == (override if override is not None else "fake-file-key")
    assert any(c.key == "fake-gemini-key" for c in discover_gemini_credentials(os.environ).all_credentials())
    assert "fake-" not in capsys.readouterr().out


def test_both_providers_and_preflight_share_loaded_key(root, monkeypatch, capsys):
    (root / ".env").write_text("OPENAI_API_KEY=fake-file-key\nSEMANTIC_PROVIDER=openai\n")
    environment.load_environment()
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr(semantics, "OpenAISemanticProvider", constructor)
    semantics.build_semantic_provider_from_env()
    OpenAINarrativeProvider("gpt-6-luna")
    assert constructor.call_count == 2
    assert all(call.args[0] == "fake-file-key" for call in constructor.call_args_list)
    source = root / "input" / "film"
    source.mkdir(parents=True)
    report = preflight(source)
    print_preflight(report)
    production_run.print_status(production_run.read_status(source))
    output = capsys.readouterr().out
    assert "OPENAI_API_KEY: SET" in output
    assert "fake-file-key" not in output and "fake-file-key" not in str(report)
    constructor.return_value.generate.assert_not_called()


@pytest.mark.parametrize("contents", ["", "OPENAI_API_KEY=\n"])
def test_missing_key_waits_without_calls_or_reset(root, monkeypatch, capsys, contents):
    (root / ".env").write_text(contents)
    environment.load_environment()
    monkeypatch.setenv("NARRATIVE_PROVIDER_MODE", "api")
    constructor = Mock(side_effect=AssertionError("provider must not be constructed"))
    monkeypatch.setattr(semantics, "OpenAISemanticProvider", constructor)
    source = root / "input" / "film"
    source.mkdir(parents=True)
    chunk = root / "runs" / "film" / "narrative-v2" / "chunks" / "existing.input.json"
    chunk.parent.mkdir(parents=True)
    chunk.write_text("persisted work")
    for _ in range(2):
        assert not production_run.ensure_narrative(source, print)
    assert chunk.read_text() == "persisted work"
    assert "WAITING_PROVIDER" in capsys.readouterr().out
    constructor.assert_not_called()
    assert semantics.build_semantic_provider_from_env() is None
    report = preflight(source)
    print_preflight(report)
    assert "OPENAI_API_KEY: MISSING" in capsys.readouterr().out
    # A fresh invocation after editing the same file resumes persisted chunks.
    (root / ".env").write_text("OPENAI_API_KEY=fake-replacement-key\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    environment.load_environment()
    adapter = Mock()
    constructor.side_effect = None
    constructor.return_value = adapter
    runner = Mock(return_value={"status": "PARTIAL"})
    monkeypatch.setattr("movie_broll.narrative_runner.run_narrative", runner)
    assert not production_run.ensure_narrative(source, print)
    runner.assert_called_once()
    assert runner.call_args.args[0] == source
    assert chunk.read_text() == "persisted work"
    adapter.generate.assert_not_called()
    assert "fake-replacement-key" not in capsys.readouterr().out
