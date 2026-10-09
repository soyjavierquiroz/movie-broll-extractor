import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_narrative_runner import prepared, FakeProvider, _response
from movie_broll.narrative_runner import run_narrative
from movie_broll.narrative_provider import ProviderResponse, OpenAINarrativeProvider
from movie_broll.production_run import ensure_narrative


def test_openai_structured_transport_reuses_sdk(monkeypatch):
    response = SimpleNamespace(output_text='{"result": true}', usage=None)
    create = Mock(return_value=response)
    adapter = SimpleNamespace(client=SimpleNamespace(responses=SimpleNamespace(create=create)), reasoning_effort='low', _usage=lambda _: {'total_tokens': 12})
    monkeypatch.setattr('movie_broll.broll_semantics.build_openai_provider_from_env', lambda model: adapter)
    provider = OpenAINarrativeProvider('configured-model')
    result = provider.generate('existing V3 prompt', {'cues': []})
    assert result.usage['total_tokens'] == 12
    request = create.call_args.kwargs
    assert request['text']['format']['strict'] is True
    assert request['model'] == 'configured-model'
    assert request['text']['format']['schema']['additionalProperties'] is False


def test_api_happy_path_consolidates_only_complete(prepared, monkeypatch):
    provider = FakeProvider()
    source = Path('runs/pilot/source-v1/srt_cues.jsonl')
    source.parent.mkdir(parents=True)
    source.write_bytes(Path('cues.jsonl').read_bytes())
    monkeypatch.setenv('NARRATIVE_PROVIDER_MODE', 'api')
    monkeypatch.setenv('NARRATIVE_PROVIDER', 'openai')
    monkeypatch.setattr('movie_broll.narrative_provider.OpenAINarrativeProvider', lambda _: provider)
    assert ensure_narrative(prepared, lambda _: None)
    assert (Path('runs/pilot/narrative-v2/narrative_map.json')).is_file()
    assert provider.calls == 2
    assert ensure_narrative(prepared, lambda _: None)
    assert provider.calls == 2


def test_outage_no_consolidation_no_gemini_secret_leak(prepared, monkeypatch):
    monkeypatch.setenv('NARRATIVE_PROVIDER_MODE', 'api')
    monkeypatch.setenv('NARRATIVE_PROVIDER', 'openai')
    class Outage(FakeProvider):
        def generate(self, *args):
            self.calls += 1
            raise TimeoutError('sk-super-secret and other secrets')
    provider = Outage()
    monkeypatch.setattr('movie_broll.narrative_provider.OpenAINarrativeProvider', lambda _: provider)
    monkeypatch.setattr('movie_broll.narrative_runner.time.sleep', lambda _: None)
    # run_narrative binds sleep as a default; patch at the coordinator boundary.
    import movie_broll.narrative_runner as runner
    actual = runner.run_narrative
    monkeypatch.setattr(runner, 'run_narrative', lambda *a, **kw: actual(*a, **kw, sleep=lambda _: None))
    forbidden = Mock(side_effect=AssertionError('Gemini fallback'))
    monkeypatch.setattr(runner, 'GeminiNarrativeProviderPool', forbidden)
    lines = []
    assert not ensure_narrative(prepared, lines.append)
    state = Path('runs/pilot/narrative-v2/narrative_run.json').read_text()
    assert json.loads(state)['status'] == 'PARTIAL'
    assert provider.calls == 3
    assert 'sk-super-secret' not in state + '\n'.join(lines)
    assert not Path('runs/pilot/narrative-v2/narrative_map.json').exists()
    forbidden.assert_not_called()


def test_strict_schema_extra_fields_retry_bounded(prepared):
    class Invalid(FakeProvider):
        def generate(self, prompt, chunk):
            self.calls += 1
            value = _response(chunk)
            value['extra'] = True
            return ProviderResponse(value, {'total_tokens': 1})
    provider = Invalid()
    result = run_narrative(prepared, provider=provider, sleep=lambda _: None, output=lambda _: None)
    assert result['status'] == 'BLOCKED_NARRATIVE_VALIDATION' and provider.calls == 2
    assert result['valid_chunks'] == 0


def test_five_chunks_interrupted_after_three_resume_at_four(prepared, monkeypatch):
    from movie_broll.srt import Cue
    source = Path('five-cues.jsonl')
    source.write_text(''.join(json.dumps(Cue(f'SRT_{i+1:06d}', i+1, i*1200+1, i*1200+2, 'texto').as_dict()) + '\n' for i in range(5)))
    monkeypatch.setattr('movie_broll.narrative_runner._ensure_source', lambda *_: source)
    class Interrupted(FakeProvider):
        def generate(self, prompt, chunk):
            if self.calls == 3: raise KeyboardInterrupt()
            return super().generate(prompt, chunk)
    with pytest.raises(KeyboardInterrupt):
        run_narrative(prepared, provider=Interrupted(), output=lambda _: None)
    resumed = FakeProvider()
    result = run_narrative(prepared, provider=resumed, output=lambda _: None)
    assert result['reused_chunks'] == 3 and resumed.calls == 2 and result['status'] == 'COMPLETE'


def test_invalid_json_from_transport_is_bounded_and_usage_persisted(prepared):
    from movie_broll.narrative_provider import NarrativeResponseError
    class InvalidJSON(FakeProvider):
        def generate(self, *args):
            self.calls += 1
            raise NarrativeResponseError({'total_tokens': 5})
    provider = InvalidJSON()
    result = run_narrative(prepared, provider=provider, output=lambda _: None)
    assert provider.calls == 2 and result['usage']['total_tokens'] == 10
    assert result['status'] == 'BLOCKED_NARRATIVE_VALIDATION'


def test_missing_credentials_after_interrupt_preserves_checkpoints(prepared, monkeypatch):
    class Interrupt(FakeProvider):
        def generate(self, prompt, chunk):
            if self.calls == 1: raise KeyboardInterrupt()
            return super().generate(prompt, chunk)
    with pytest.raises(KeyboardInterrupt):
        run_narrative(prepared, provider=Interrupt(), output=lambda _: None)
    monkeypatch.setenv('NARRATIVE_PROVIDER_MODE', 'api')
    monkeypatch.setenv('NARRATIVE_PROVIDER', 'openai')
    def unavailable(_): raise RuntimeError('missing credential')
    monkeypatch.setattr('movie_broll.narrative_provider.OpenAINarrativeProvider', unavailable)
    assert not ensure_narrative(prepared, lambda _: None)
    resumed = FakeProvider()
    result = run_narrative(prepared, provider=resumed, output=lambda _: None)
    assert result['reused_chunks'] == 1 and resumed.calls == 1


def _completed_api(prepared, monkeypatch):
    from movie_broll.narrative_consolidate import consolidate_narrative
    from movie_broll.utils import write_json
    provider = FakeProvider()
    provider.identifier = 'openai'
    provider.model = 'configured-api-model'
    source = prepared.parents[1] / 'runs/pilot/source-v1'
    source.mkdir(parents=True)
    (source / 'srt_cues.jsonl').write_bytes(Path('cues.jsonl').read_bytes())
    write_json(source / 'source_manifest.json', {'source': {'movie': {'filename': 'movie.mp4'}}})
    result = run_narrative(prepared, model=provider.model, provider=provider, output=lambda _: None)
    assert result['status'] == 'COMPLETE'
    assert consolidate_narrative(prepared, output=lambda _: None)['status'] == 'PASS'
    monkeypatch.setenv('SEMANTIC_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-test-key')
    return prepared.parents[1] / 'runs/pilot/narrative-v2', provider


def test_api_provenance_preflight_and_completed_cache_reuse(prepared, monkeypatch):
    from movie_broll.production_preflight import preflight
    run, provider = _completed_api(prepared, monkeypatch)
    before = {p: p.read_bytes() for p in (run / 'maps').glob('*')}
    # Environment changes must not relabel or invalidate a completed producer.
    monkeypatch.setenv('NARRATIVE_PROVIDER_MODE', 'external')
    monkeypatch.setenv('NARRATIVE_MODEL', 'another-model')
    assert preflight(prepared)['ready']
    result = run_narrative(prepared, model=provider.model, provider=provider, output=lambda _: None)
    assert result['reused_chunks'] == 2 and provider.calls == 2
    assert before == {p: p.read_bytes() for p in (run / 'maps').glob('*')}


@pytest.mark.parametrize('artifact,key,value', [
    ('narrative_map.json', 'provider', 'other-provider'),
    ('narrative_map.json', 'model', 'other-model'),
    ('narrative_run.json', 'provider', 'other-provider'),
    ('narrative_run.json', 'model', 'other-model'),
    ('narrative_run.json', 'provider', None),
    ('narrative_run.json', 'model', []),
    ('narrative_run.json', 'prompt_version', 'invented-prompt'),
    ('maps/NCHUNK_0001.checkpoint.json', 'provider', 'other-provider'),
    ('maps/NCHUNK_0001.checkpoint.json', 'model', 'other-model'),
])
def test_api_provenance_mismatch_blocks(prepared, monkeypatch, artifact, key, value):
    from movie_broll.production_preflight import preflight
    run, _ = _completed_api(prepared, monkeypatch)
    path = run / artifact
    data = json.loads(path.read_text())
    target = data['analysis'] if artifact == 'narrative_map.json' else data
    target[key] = value
    path.write_text(json.dumps(data))
    report = preflight(prepared)
    assert not report['ready'] and report['narrative']['status'] == 'FAIL'
    assert any('provenance mismatch' in reason for reason in report['blockers'])


@pytest.mark.parametrize('damage', ['missing', 'malformed', 'nonobject', 'bad-fingerprint', 'bad-analysis'])
def test_api_missing_or_corrupt_provenance_blocks(prepared, monkeypatch, damage):
    from movie_broll.production_preflight import preflight
    run, _ = _completed_api(prepared, monkeypatch)
    path = run / 'maps/NCHUNK_0001.checkpoint.json'
    if damage == 'missing': path.unlink()
    elif damage == 'malformed': path.write_text('{')
    elif damage == 'nonobject': path.write_text('[]')
    elif damage == 'bad-analysis':
        path = run / 'narrative_map.json'
        data = json.loads(path.read_text()); data['analysis'] = []
        path.write_text(json.dumps(data))
    else:
        data = json.loads(path.read_text()); data['content_fingerprint'] = 'fabricated'
        path.write_text(json.dumps(data))
    assert not preflight(prepared)['ready']


def test_real_preflight_failure_stops_before_downstream(prepared, monkeypatch):
    from movie_broll import production_run
    run, _ = _completed_api(prepared, monkeypatch)
    (run / 'maps/NCHUNK_0001.checkpoint.json').unlink()
    monkeypatch.setattr(production_run, 'ensure_source', lambda _: True)
    monkeypatch.setattr(production_run, 'ensure_narrative', lambda *_: True)
    downstream = Mock(side_effect=AssertionError('downstream started'))
    class SupervisorBoundary:
        def __init__(self, _, **kwargs): self.before_start = kwargs['before_start']
        def run(self):
            self.before_start()
            downstream()
    with pytest.raises(RuntimeError, match='preflight blocked:.*checkpoint'):
        production_run.run(prepared, output=lambda _: None, supervisor_factory=SupervisorBoundary)
    downstream.assert_not_called()
