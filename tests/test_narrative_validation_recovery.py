"""Offline synthetic regression for the saved E03 missing long-reason shape."""
import json
from pathlib import Path

import pytest

from test_narrative_runner import prepared, FakeProvider, _response
from movie_broll import narrative_runner as runner
from movie_broll.narrative import ENUMS, normalize_external_v3_response
from movie_broll.narrative_provider import ProviderResponse
from movie_broll.srt import Cue


def long_input(tmp_path):
    value = {"schema_version": "srt_narrative_input_v1", "movie_id": "fixture",
             "chunk": {"chunk_id": "NCHUNK_0001", "start_seconds": 0, "end_seconds": 132},
             "cues": [Cue("SRT_000001", 1, 0, 1, "Hola").as_dict(),
                      Cue("SRT_000002", 2, 131, 132, "Adiós").as_dict()]}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value))
    return path, value


def test_exact_failure_and_repaired_canonical_and_manual_path(tmp_path):
    path, chunk = long_input(tmp_path)
    response = _response(chunk)
    with pytest.raises(runner.V3ValidationError) as failure:
        runner.validate_v3_candidate(path, response, tmp_path / "candidate.json")
    validation = failure.value.validation
    assert validation['stage'] == 'canonical_semantic_validation'
    assert validation['errors'] == [{"code": "LONG_SEGMENT_REASON_REQUIRED",
        "path": "segments[0].long_segment_reason",
        "message": "segment 1 exceeds 120 seconds (132.000s, SRT_000001–SRT_000002) and requires long_segment_reason"}]
    feedback = runner._retry_prompt('base', validation)
    assert '132.000s' in feedback and 'segment 1' in feedback
    response['segments'][0]['long_segment_reason'] = sorted(ENUMS['long_segment_reason'])[0]
    canonical = runner.validate_v3_candidate(path, response, tmp_path / 'candidate.json')
    assert canonical['segments'][0]['end_seconds'] == 132
    # Existing external canonical importer accepts the same repaired map.
    assert normalize_external_v3_response(chunk, response)['segments'][0]['cue_ids'] == ['SRT_000001', 'SRT_000002']


def test_persisted_diagnostics_feedback_and_repair(prepared, monkeypatch):
    source = Path('long-cues.jsonl')
    source.write_text(''.join(json.dumps(c.as_dict()) + '\n' for c in [
        Cue('SRT_000001', 1, 0, 1, 'Hola'), Cue('SRT_000002', 2, 131, 132, 'Adiós')]))
    monkeypatch.setattr(runner, '_ensure_source', lambda *_: source)
    class Repair(FakeProvider):
        def generate(self, prompt, chunk):
            self.calls += 1
            response = _response(chunk)
            if self.calls == 2:
                execution = json.loads(Path('runs/pilot/narrative-v2/responses/NCHUNK_0001.execution.json').read_text())
                assert execution['validation']['errors'][0]['code'] == 'LONG_SEGMENT_REASON_REQUIRED'
                assert execution['validation_failures'] == 1
                assert 'segment 1 exceeds 120 seconds (132.000s' in prompt
                response['segments'][0]['long_segment_reason'] = sorted(ENUMS['long_segment_reason'])[0]
            return ProviderResponse(response, {})
    provider = Repair()
    assert runner.run_narrative(prepared, provider=provider, output=lambda _: None)['status'] == 'COMPLETE'
    assert provider.calls == 2


def test_durable_block_reuses_first_two_and_stops_before_later_chunks(prepared, monkeypatch):
    source = Path('five-cues.jsonl')
    source.write_text(''.join(json.dumps(Cue(f'SRT_{i+1:06d}', i+1, i*1200+1, i*1200+2, 'texto').as_dict()) + '\n' for i in range(5)))
    monkeypatch.setattr(runner, '_ensure_source', lambda *_: source)
    class InvalidThird(FakeProvider):
        def generate(self, prompt, chunk):
            self.calls += 1
            value = _response(chunk)
            if chunk['chunk']['chunk_id'] == 'NCHUNK_0003':
                value['segments'][0]['first_cue_id'] = 'SRT_999999'
            return ProviderResponse(value, {})
    provider = InvalidThird()
    first = runner.run_narrative(prepared, provider=provider, output=lambda _: None)
    assert first['status'] == 'BLOCKED_NARRATIVE_VALIDATION' and provider.calls == 4
    root = Path('runs/pilot/narrative-v2')
    preserved = {p: p.read_bytes() for p in (root / 'maps').glob('*')}
    restarted = runner.run_narrative(prepared, provider=provider, force=False, output=lambda _: None)
    assert restarted['requests'] == 0 and restarted['reused_chunks'] == 2 and provider.calls == 4
    assert restarted['status'] == 'BLOCKED_NARRATIVE_VALIDATION'
    assert all(p.read_bytes() == data for p, data in preserved.items())
    execution = json.loads((root / 'responses/NCHUNK_0003.execution.json').read_text())
    assert execution['validation_failures'] == 2 and execution['failure_fingerprint']
    assert 'unknown cue SRT_999999' in execution['validation']['errors'][0]['message']
    assert not (root / 'narrative_map.json').exists()
    assert not list((root / 'responses').glob('NCHUNK_000[45]*'))


def test_transient_exhaustion_resumes(prepared):
    provider = FakeProvider(failures=3)
    first = runner.run_narrative(prepared, provider=provider, sleep=lambda _: None, output=lambda _: None)
    assert first['status'] == 'PARTIAL' and first['requests'] == 3
    second = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert second['valid_chunks'] == 1 and second['requests'] == 1


def test_diagnostics_never_echo_schema_values_or_configured_secrets(prepared, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-private-secret')
    class SecretResponse(FakeProvider):
        def generate(self, prompt, chunk):
            self.calls += 1
            value = _response(chunk)
            value['segments'][0]['first_cue_id'] = 'fixture-private-secret'
            return ProviderResponse(value, {})
    lines = []
    runner.run_narrative(prepared, provider=SecretResponse(), output=lines.append)
    root = Path('runs/pilot/narrative-v2')
    diagnostic = (root / 'responses/NCHUNK_0001.execution.json').read_text()
    assert 'fixture-private-secret' not in diagnostic + '\n'.join(lines) + (root / 'narrative_run.json').read_text()
    assert 'SCHEMA_PATTERN' in diagnostic


def test_legacy_exhausted_execution_blocks_even_force(prepared):
    runner.run_narrative(prepared, provider=FakeProvider(invalid=True), max_chunks=1, output=lambda _: None)
    path = Path('runs/pilot/narrative-v2/responses/NCHUNK_0001.execution.json')
    execution = json.loads(path.read_text())
    execution.pop('validation_failures')
    execution['status'] = 'PARTIAL'
    path.write_text(json.dumps(execution))
    provider = FakeProvider()
    result = runner.run_narrative(prepared, provider=provider, force=True, output=lambda _: None)
    assert result['status'] == 'BLOCKED_NARRATIVE_VALIDATION' and provider.calls == 0


def test_provider_model_switch_does_not_authorize_paid_retries(prepared):
    provider = FakeProvider(invalid=True)
    first = runner.run_narrative(prepared, model='model-a', provider=provider, max_chunks=1, output=lambda _: None)
    assert first['requests'] == 2
    for model, identifier in [('model-b', 'gemini'), ('model-b', 'other-provider'), ('model-a', 'gemini')]:
        provider.identifier = identifier
        result = runner.run_narrative(prepared, model=model, provider=provider, max_chunks=1, output=lambda _: None)
        assert result['status'] == 'BLOCKED_NARRATIVE_VALIDATION'
        assert result['requests'] == 0 and provider.calls == 2


def test_blocked_coordinator_does_not_consolidate_or_start_downstream(prepared, monkeypatch):
    from unittest.mock import Mock
    from movie_broll.production_run import ensure_narrative
    monkeypatch.setenv('NARRATIVE_PROVIDER_MODE', 'api')
    monkeypatch.setenv('NARRATIVE_PROVIDER', 'openai')
    provider = FakeProvider(invalid=True)
    monkeypatch.setattr('movie_broll.narrative_provider.OpenAINarrativeProvider', lambda _: provider)
    consolidate = Mock(side_effect=AssertionError('Incomplete narrative cannot consolidate'))
    monkeypatch.setattr('movie_broll.narrative_consolidate.consolidate_narrative', consolidate)
    lines = []
    assert ensure_narrative(prepared, lines.append) is False
    assert ensure_narrative(prepared, lines.append) is False
    assert provider.calls == 2
    assert 'STATUS: BLOCKED_NARRATIVE_VALIDATION' in lines
    assert not Path('runs/pilot/narrative-v2/narrative_map.json').exists()
    consolidate.assert_not_called()


def test_retry_feedback_survives_budget_stop(prepared, monkeypatch):
    monkeypatch.setattr(runner, 'FREE_TIER_REQUEST_BUDGET', 1)
    first = runner.run_narrative(prepared, provider=FakeProvider(invalid=True), max_chunks=1, output=lambda _: None)
    assert first['status'] == 'REQUEST_BUDGET_EXHAUSTED'
    class Repair(FakeProvider):
        def generate(self, prompt, chunk):
            assert 'segment 1 first_cue_id references unknown cue SRT_999999' in prompt
            return super().generate(prompt, chunk)
    provider = Repair()
    result = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert result['valid_chunks'] == 1 and provider.calls == 1


@pytest.mark.parametrize('validator_name', ['validate_llm_v3_response', 'validate_narrative_map'])
def test_validator_fix_recovery_revision_preserves_valid_chunks(prepared, monkeypatch, validator_name):
    source = Path('three-cues.jsonl')
    source.write_text(''.join(json.dumps(Cue(f'SRT_{i+1:06d}', i+1, i*1200+1, i*1200+2, 'texto').as_dict()) + '\n' for i in range(3)))
    monkeypatch.setattr(runner, '_ensure_source', lambda *_: source)
    original = getattr(runner, validator_name)
    def buggy_validator(chunk, data):
        chunk_id = chunk['chunk']['chunk_id'] if isinstance(chunk, dict) else chunk.name.removesuffix('.input.json')
        if chunk_id == 'NCHUNK_0003':
            return ['segment 1 incorrectly rejected by historical validator']
        return original(chunk, data)
    monkeypatch.setattr(runner, validator_name, buggy_validator)
    provider = FakeProvider()
    first = runner.run_narrative(prepared, provider=provider, output=lambda _: None)
    assert first['status'] == 'BLOCKED_NARRATIVE_VALIDATION' and provider.calls == 4
    root = Path('runs/pilot/narrative-v2')
    checkpoints = {p: p.read_bytes() for p in (root / 'maps').glob('*')}
    archives = {p: p.read_bytes() for p in (root / 'responses').glob('*')}
    # Fixing validation recovers the saved response without replenishing the paid budget.
    monkeypatch.setattr(runner, validator_name, original)
    unchanged = runner.run_narrative(prepared, provider=provider, output=lambda _: None)
    assert unchanged['status'] == 'COMPLETE'
    assert unchanged['requests'] == 0 and unchanged['reused_chunks'] == 3
    assert provider.calls == 4
    assert all(p.read_bytes() == data for p, data in archives.items())
    execution = root / 'responses/NCHUNK_0003.execution.json'
    assert json.loads(execution.read_text())['validation_failures'] == 2
    checkpoint = json.loads((root / 'maps/NCHUNK_0003.checkpoint.json').read_text())
    assert checkpoint['provider'] == 'gemini' and checkpoint['model'] == provider.model
    assert checkpoint['recovered_from'].endswith('.llm-v3.json')
    # A deliberate recovery revision is separate from the semantic cache key.
    monkeypatch.setattr(runner, 'VALIDATION_RECOVERY_REVISION', 'narrative_validation_recovery_v2', raising=False)
    recovered = runner.run_narrative(prepared, provider=provider, output=lambda _: None)
    assert recovered['status'] == 'COMPLETE' and recovered['requests'] == 0
    assert recovered['reused_chunks'] == 3 and provider.calls == 4
    assert all(p.read_bytes() == data for p, data in checkpoints.items())


def test_recovery_revision_is_bounded_and_switching_back_stays_blocked(prepared, monkeypatch):
    provider = FakeProvider(invalid=True)
    revisions = ('narrative_validation_recovery_v1', 'narrative_validation_recovery_v2')
    for revision in revisions:
        monkeypatch.setattr(runner, 'VALIDATION_RECOVERY_REVISION', revision)
        result = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
        assert result['status'] == 'BLOCKED_NARRATIVE_VALIDATION' and result['requests'] == 2
        repeat = runner.run_narrative(prepared, provider=provider, force=True, max_chunks=1, output=lambda _: None)
        assert repeat['requests'] == 0
    monkeypatch.setattr(runner, 'VALIDATION_RECOVERY_REVISION', revisions[0])
    result = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert result['requests'] == 0 and provider.calls == 4


@pytest.mark.parametrize('via_history', [False, True])
def test_unversioned_guard_keeps_exhaustion(prepared, monkeypatch, via_history):
    monkeypatch.setattr(runner, 'VALIDATION_RECOVERY_REVISION', runner._LEGACY_RECOVERY_REVISION)
    provider = FakeProvider(invalid=True)
    runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    path = Path('runs/pilot/narrative-v2/responses/NCHUNK_0001.execution.json')
    state = json.loads(path.read_text())
    state.pop('retry_history')
    state.pop('validation_recovery_revision')
    if via_history:
        legacy_identity = {key: state[key] for key in ('content_fingerprint', 'provider', 'model')}
        key = runner.sha256_text(json.dumps(legacy_identity, sort_keys=True))
        state = {"content_fingerprint": 'other-content', 'provider': provider.identifier,
                 'model': 'other-model', 'retry_history': {key: state}}
    path.write_text(json.dumps(state))
    result = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert result['requests'] == 0 and provider.calls == 2


def test_semantic_schema_revision_changes_retry_and_cache_identity(prepared, monkeypatch):
    import copy
    from movie_broll import narrative_provider
    first = runner.run_narrative(prepared, provider=FakeProvider(invalid=True), max_chunks=1, output=lambda _: None)
    assert first['status'] == 'BLOCKED_NARRATIVE_VALIDATION'
    schema = copy.deepcopy(narrative_provider.v3_schema())
    schema['properties']['segments']['items']['properties']['narrative_tone']['enum'].append('synthetic_new_tone')
    actual_hash = runner.sha256_file
    def revised_hash(path):
        if path.name == 'narrative_mapper_llm_v3.schema.json':
            return runner.sha256_text(json.dumps(schema, sort_keys=True))
        return actual_hash(path)
    monkeypatch.setattr(runner, 'sha256_file', revised_hash)
    monkeypatch.setattr(narrative_provider, 'v3_schema', lambda: schema)
    provider = FakeProvider()
    result = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert result['requests'] == 1 and result['valid_chunks'] == 1
    repeated = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert repeated['requests'] == 0 and repeated['reused_chunks'] == 1


def test_semantic_prompt_revision_archives_old_contract_and_recovers(prepared, monkeypatch):
    runner.run_narrative(prepared, provider=FakeProvider(invalid=True), max_chunks=1, output=lambda _: None)
    actual_read = Path.read_text
    def revised_read(path, *args, **kwargs):
        value = actual_read(path, *args, **kwargs)
        if path.name == 'srt_narrative_mapper_v3.md':
            return value + '\nSynthetic semantic prompt revision for this offline fixture.\n'
        return value
    monkeypatch.setattr(Path, 'read_text', revised_read)
    provider = FakeProvider()
    result = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert result['semantics_replaced'] is True and result['requests'] == 1
    archive = Path('runs/pilot/narrative-v2/superseded/superseded-narrative-semantics')
    assert list(archive.glob('*/responses/NCHUNK_0001.execution.json'))


def test_feedback_has_semantic_long_segment_choice_without_auto_reason(tmp_path):
    path, chunk = long_input(tmp_path)
    response = _response(chunk)
    with pytest.raises(runner.V3ValidationError) as failure:
        runner.validate_v3_candidate(path, response, tmp_path / 'candidate.json')
    prompt = runner._retry_prompt('base', failure.value.validation)
    assert 'Split distinct situations' in prompt
    assert 'justify a long segment only if it is one continuous interaction/action' in prompt
    assert response['segments'][0]['long_segment_reason'] is None
    assert not (tmp_path / 'candidate.json').exists()


@pytest.mark.parametrize('legacy', [False, True])
def test_offline_recovery_preserves_evidence_and_original_producer(prepared, monkeypatch, legacy):
    original = runner.validate_llm_v3_response
    monkeypatch.setattr(runner, 'validate_llm_v3_response', lambda *_: ['historical validator defect'])
    producer = FakeProvider()
    producer.identifier = 'original-producer'
    runner.run_narrative(prepared, provider=producer, max_chunks=1, output=lambda _: None)
    root = Path('runs/pilot/narrative-v2')
    if legacy:
        # Synthetic fixture only: emulate responses saved before per-attempt provenance.
        for path in (root / 'responses').glob('*.provenance.json'):
            path.unlink()
    evidence = {p: p.read_bytes() for p in (root / 'responses').glob('*')}
    monkeypatch.setattr(runner, 'validate_llm_v3_response', original)
    replacement = FakeProvider()
    replacement.identifier = 'replacement-producer'
    monkeypatch.setattr(runner, 'VALIDATION_RECOVERY_REVISION', 'reviewed-fixture-revision')
    result = runner.run_narrative(prepared, provider=replacement, max_chunks=1, output=lambda _: None)
    assert result['requests'] == 0 and result['reused_chunks'] == 1 and replacement.calls == 0
    checkpoint = json.loads((root / 'maps/NCHUNK_0001.checkpoint.json').read_text())
    assert checkpoint['provider'] == producer.identifier and checkpoint['model'] == producer.model
    assert all(path.read_bytes() == data for path, data in evidence.items())
    cached = {p: p.read_bytes() for p in (root / 'maps').glob('*')}
    repeated = runner.run_narrative(prepared, provider=replacement, force=True, max_chunks=1, output=lambda _: None)
    assert repeated['requests'] == 0 and repeated['reused_chunks'] == 1
    assert all(path.read_bytes() == data for path, data in cached.items())


def test_offline_recovery_tries_older_response_without_rewriting_invalid_archive(prepared, monkeypatch):
    original = runner.validate_llm_v3_response
    monkeypatch.setattr(runner, 'validate_llm_v3_response', lambda *_: ['historical validator defect'])
    class Mixed(FakeProvider):
        def generate(self, prompt, chunk):
            response = super().generate(prompt, chunk)
            if self.calls == 2:
                response.data['segments'][0]['first_cue_id'] = 'SRT_999999'
            return response
    runner.run_narrative(prepared, provider=Mixed(), max_chunks=1, output=lambda _: None)
    monkeypatch.setattr(runner, 'validate_llm_v3_response', original)
    provider = FakeProvider()
    result = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert result['requests'] == 0 and provider.calls == 0
    root = Path('runs/pilot/narrative-v2')
    checkpoint = json.loads((root / 'maps/NCHUNK_0001.checkpoint.json').read_text())
    assert 'attempt-1.' in checkpoint['recovered_from']
    invalid = json.loads((root / 'responses/NCHUNK_0001.attempt-2.llm-v3.json').read_text())
    assert invalid['segments'][0]['first_cue_id'] == 'SRT_999999'


@pytest.mark.parametrize('unversioned', [False, True])
def test_actionable_feedback_revision_reopens_once_without_changing_contract(prepared, monkeypatch, unversioned):
    current_revision = runner.VALIDATION_RECOVERY_REVISION
    assert current_revision == 'narrative_validation_recovery_v2'
    monkeypatch.setattr(runner, 'VALIDATION_RECOVERY_REVISION', runner._LEGACY_RECOVERY_REVISION)
    class InvalidWithFeedback(FakeProvider):
        def generate(self, prompt, chunk):
            if self.calls % runner.MAX_SEMANTIC_ATTEMPTS:
                assert 'segment 1 first_cue_id references unknown cue SRT_999999' in prompt
                assert 'Split distinct situations' in prompt
            return super().generate(prompt, chunk)
    provider = InvalidWithFeedback(invalid=True)
    first = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert first['requests'] == runner.MAX_SEMANTIC_ATTEMPTS
    root = Path('runs/pilot/narrative-v2')
    execution_path = root / 'responses/NCHUNK_0001.execution.json'
    if unversioned:
        state = json.loads(execution_path.read_text())
        state.pop('validation_recovery_revision')
        state.pop('retry_history')
        execution_path.write_text(json.dumps(state))
    input_path = root / 'chunks/NCHUNK_0001.input.json'
    prompt_hash = runner.sha256_text(Path(runner.__file__).resolve().parents[2].joinpath(
        'config/prompts/srt_narrative_mapper_v3.md').read_text())
    contract = runner._narrative_content_inputs(input_path, provider.model, prompt_hash)
    fingerprint = runner._content_fingerprint(contract)
    blocked = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert blocked['requests'] == 0
    monkeypatch.setattr(runner, 'VALIDATION_RECOVERY_REVISION', current_revision)
    recovered = runner.run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert recovered['requests'] == runner.MAX_SEMANTIC_ATTEMPTS
    assert recovered['status'] == 'BLOCKED_NARRATIVE_VALIDATION'
    assert runner._narrative_content_inputs(input_path, provider.model, prompt_hash) == contract
    state = json.loads(execution_path.read_text())
    assert state['content_fingerprint'] == fingerprint
    assert state['validation_recovery_revision'] == current_revision
    assert state['validation_failures'] == runner.MAX_SEMANTIC_ATTEMPTS
    repeated = runner.run_narrative(prepared, provider=provider, force=True, max_chunks=1, output=lambda _: None)
    assert repeated['requests'] == 0
    assert provider.calls == 2 * runner.MAX_SEMANTIC_ATTEMPTS
