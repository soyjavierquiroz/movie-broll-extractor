import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from test_publication import _review_package
from movie_broll.closure import close, package_names
from movie_broll.publication import apply_publication_projection, decide_review_vertical
from movie_broll.utils import write_json


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run, base, data = _review_package(tmp_path, asset_id='m007')
    target = run.with_name('series-s03e03')
    run.rename(target)
    for filename in ('processing_ledger.json', 'asset_registry.json'):
        path = target / filename
        state = json.loads(path.read_text())
        state['movie_id'] = target.name
        write_json(path, state)
    return target, base, data


def auto(package):
    run, base, data = package
    data['visual']['final_vertical']['validation_status'] = 'PASS'
    write_json(run / 'review' / f'{base}.json', apply_publication_projection(data))
    for name in package_names(base):
        (run / 'review' / name).rename(run / 'assets' / name)


def execute(package):
    return close(Path('input') / package[0].name, output=lambda _: None)


def test_auto_pass_exact_sparse_manifest_idempotent_zero_calls(package, monkeypatch):
    auto(package)
    forbidden = Mock(side_effect=AssertionError('provider called during close'))
    monkeypatch.setattr('movie_broll.narrative_provider.OpenAINarrativeProvider.generate', forbidden)
    monkeypatch.setattr('movie_broll.narrative_provider.GeminiNarrativeProvider.generate', forbidden)
    monkeypatch.setattr('movie_broll.broll_semantics.OpenAISemanticProvider.generate', forbidden)
    first = execute(package)
    assert first['verdict'] == 'HANDOFF_COMPLETE' and first['semantic_calls'] == 0
    manifest = first['manifest']
    assert manifest['approved_producer_asset_ids'] == ['m007']
    assert manifest['auto_pass_total'] == 1
    export = package[0] / 'final-export'
    assert {p.name for p in export.iterdir()} == set(package_names(package[1])) | {'E03_FINAL_APPROVAL_MANIFEST.json'}
    assert json.loads((export / 'E03_FINAL_APPROVAL_MANIFEST.json').read_text()) == manifest
    assert execute(package)['idempotent']
    forbidden.assert_not_called()


def test_review_blocks_then_human_approval_included(package):
    assert execute(package)['verdict'] == 'BLOCKED'
    decide_review_vertical(package[0], 'm007', 'APPROVE', reason='Visually inspected', reviewer='operator')
    result = execute(package)
    assert result['verdict'] == 'HANDOFF_COMPLETE'
    assert result['manifest']['human_approved_total'] == 1
    provenance = result['manifest']['assets'][0]['approval_provenance']
    assert provenance['vertical_validation']['validation_status'] == 'REVIEW'
    assert provenance['human_review']['status'] == 'APPROVED'


def test_explicit_hard_override_preserves_qa_idempotent(package):
    run, base, data = package
    data['visual']['final_vertical'].update(hard_failures=['editorial_crop'], soft_warnings=['warning'], review_reason='editorial_crop')
    write_json(run / 'review' / f'{base}.json', data)
    with pytest.raises(ValueError, match='override-qa'):
        decide_review_vertical(run, 'm007', 'APPROVE')
    with pytest.raises(ValueError, match='reason'):
        decide_review_vertical(run, 'm007', 'APPROVE', override_qa=True)
    decide_review_vertical(run, 'm007', 'APPROVE', override_qa=True, reason='Inspected crop', reviewer='editor')
    result = decide_review_vertical(run, 'm007', 'APPROVE')
    assert result['idempotent']
    data = json.loads((run / 'assets' / f'{base}.json').read_text())
    assert data['visual']['final_vertical']['hard_failures'] == ['editorial_crop']
    assert data['publication']['human_review']['automatic_qa_overridden'] is True


@pytest.mark.parametrize('member', ['.mp4', '.jpg'])
def test_invalid_media_blocks_even_with_matching_hash(package, member):
    from movie_broll.utils import sha256_file
    auto(package)
    run, base, _ = package
    path = run / 'assets' / f'{base}{member}'
    path.write_bytes(b'corrupt')
    data = json.loads((run / 'assets' / f'{base}.json').read_text())
    record = data['media']['horizontal'] if member == '.mp4' else data['media']['horizontal']['thumbnail']
    record.update(sha256=sha256_file(path), size_bytes=path.stat().st_size)
    write_json(run / 'assets' / f'{base}.json', data)
    assert execute(package)['verdict'] == 'BLOCKED'
    assert not (run / 'final-export').exists()


@pytest.mark.parametrize('failure', ['missing', 'hash', 'identity', 'event', 'corrupt'])
def test_structural_blockers_cannot_be_overridden(package, failure):
    run, base, data = package
    path = run / 'review' / f'{base}.mp4'
    if failure == 'missing': path.rename(path.with_suffix('.absent'))
    if failure == 'hash': data['media']['horizontal']['sha256'] = 'bad'
    if failure == 'identity': data['asset']['id'] = 'm008'
    if failure == 'event': data['source_timeline']['visual_event_id'] = 'other'
    if failure == 'corrupt':
        from movie_broll.utils import sha256_file
        path.write_bytes(b'bad')
        data['media']['horizontal'].update(sha256=sha256_file(path), size_bytes=3)
    write_json(run / 'review' / f'{base}.json', data)
    with pytest.raises(ValueError):
        decide_review_vertical(run, 'm007', 'APPROVE', override_qa=True, reason='Inspected')


@pytest.mark.parametrize('failure', ['missing', 'hash', 'duplicate', 'stale', 'manifest'])
def test_closure_blockers(package, failure):
    auto(package)
    run, base, _ = package
    if failure in {'stale', 'manifest'}:
        assert execute(package)['verdict'] == 'HANDOFF_COMPLETE'
        if failure == 'stale': (run / 'final-export' / 'log.txt').write_text('stale')
        else: (run / 'final-export' / 'E03_FINAL_APPROVAL_MANIFEST.json').write_text('{}')
    elif failure == 'missing': (run / 'assets' / f'v{base}.jpg').rename(run / 'assets' / 'missing.fixture')
    elif failure == 'hash': (run / 'assets' / f'{base}.mp4').write_bytes(b'changed')
    else: (run / 'review' / f'{base}.json').write_bytes((run / 'assets' / f'{base}.json').read_bytes())
    assert execute(package)['verdict'] == 'BLOCKED'


def test_rejected_excluded(package):
    decide_review_vertical(package[0], 'm007', 'REJECT')
    result = execute(package)
    assert result['verdict'] == 'HANDOFF_COMPLETE'
    assert result['manifest']['approved_producer_asset_ids'] == []


def test_missing_entire_approved_package_blocks(package):
    auto(package)
    run, base, _ = package
    ledger = json.loads((run / 'processing_ledger.json').read_text())
    ledger['events']['VE_000001']['stages']['finalization'].update(decision='PASS', publish_ready=True)
    write_json(run / 'processing_ledger.json', ledger)
    (run / 'assets' / f'{base}.json').rename(run / 'metadata.fixture')
    result = execute(package)
    assert result['verdict'] == 'BLOCKED'
    assert any('incomplete approved package' in error for error in result['errors'])
