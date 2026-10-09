"""Read-only legacy audits and guarded replacement of vertical renditions only."""
from __future__ import annotations
import copy
import json
import os
import shutil
import uuid
from pathlib import Path
import cv2
from . import finalization as f
from .face_safe import VERSION, audit_video, capability, preflight
from .publication import apply_publication_projection
from .utils import sha256_file, write_json


def context(location):
    location = Path(location).resolve()
    if location.name == 'assets':
        run = location.parent
    elif (location/'assets').is_dir():
        run = location
    else:
        run = location.parents[1]/'runs'/location.name
    movie = run.parents[1]/'input'/run.name/'movie.mp4'
    if not (run/'assets').is_dir() or not movie.is_file():
        raise ValueError('Expected input/<movie>, runs/<movie>, or runs/<movie>/assets with canonical source')
    events = {r['visual_event_id']: r for r in json.loads((run/'visual_events.json').read_text())['events']}
    shots = {r['shot_id']: r for r in json.loads((run/'technical_shots.json').read_text())['shots']}
    return run, movie, events, shots


def package_path(assets, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('Package member must be a flat filename')
    path = assets/name
    if path.is_symlink():
        raise ValueError('Symlink package members are unsupported')
    return path


def source_plan(movie, event, shots, strategy='subject_focus', active_picture=None):
    cap = cv2.VideoCapture(str(movie))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    active=active_picture or {'x':0,'y':0,'width':width,'height':height,
                              'source_width':width,'source_height':height,
                              'structural_bars':False,'detection_profile':'legacy_full_frame'}
    return f.build_shot_crop_plan(movie, event, shots, int(active['width']), int(active['height']),
                                  strategy=strategy, active_picture=active)


def audit_asset(metadata_path, movie, events, shots):
    data = json.loads(metadata_path.read_text())
    assets = metadata_path.parent
    vertical = package_path(assets, data['media']['vertical']['file'])
    event = events[data['source_timeline']['visual_event_id']]
    plan = source_plan(movie, event, shots, active_picture=data.get('source', {}).get('active_picture'))
    face = audit_video(vertical, movie, event, plan)
    # The established person audit remains an additional gate, never proof of a face.
    person = f._post_render_person_audit(vertical, plan)
    decision = face['decision']
    if decision == 'PASS' and person['decision'] != 'PASS':
        decision = 'AMBIGUOUS'
    return {'producer_asset_id': data['asset']['id'], 'metadata_file': metadata_path.name,
            'vertical_path': str(vertical), 'decision': decision, 'reason_codes': face['reason_codes'] +
            ([person['reason']] if person['decision'] != 'PASS' else []),
            'repairable': decision == 'REPAIR', 'source_limited': face['source_limited'],
            'confidence': 'medium' if decision in {'PASS', 'REPAIR', 'SOURCE_LIMIT'} else 'low',
            'vertical_sha256': sha256_file(vertical), 'metadata_sha256': sha256_file(metadata_path),
            'evidence': face, 'person_audit_decision': person['decision']}


def audit_verticals(location, output=None, asset_id=None, reporter=None):
    run, movie, events, shots = context(location)
    assets = run/'assets'
    output = Path(output).resolve() if output else run/'vertical_repair_audit.json'
    if output == assets or assets in output.parents:
        raise ValueError('Audit output must be outside assets')
    cv2.setNumThreads(2)
    report = {'schema_version': 'vertical_repair_audit_v1', 'version': VERSION,
              'run': str(run), 'source_movie': str(movie), 'face_capability': preflight(provision=False),
              'assets': [], 'scope': asset_id or 'all'}
    for metadata in sorted(assets.glob('*.json')):
        row = {'metadata_file': metadata.name, 'producer_asset_id': None, 'vertical_path': None}
        try:
            data = json.loads(metadata.read_text())
            row['producer_asset_id'] = data['asset']['id']
            if asset_id and row['producer_asset_id'] != asset_id:
                continue
            row = audit_asset(metadata, movie, events, shots)
        except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as error:
            row.update(decision='ERROR', reason_codes=['asset_audit_error'], repairable=False,
                       source_limited=False, confidence='low', error=str(error))
        report['assets'].append(row)
        if reporter:
            reporter(f"{row['producer_asset_id']}: {row['decision']}")
    if asset_id and not report['assets']:
        raise ValueError('Asset ID not found: ' + asset_id)
    report['counts'] = {d: sum(r['decision'] == d for r in report['assets'])
                        for d in ('PASS', 'REPAIR', 'SOURCE_LIMIT', 'AMBIGUOUS', 'ERROR')}
    write_json(output, report)
    return report


def replacement_members(paths):
    if len(paths)!=3:
        raise ValueError('Replacement requires vertical MP4, JPG, metadata')
    metadata=paths[-1]
    expected=['v'+metadata.stem+'.mp4','v'+metadata.stem+'.jpg',metadata.name]
    if metadata.suffix!='.json' or [p.name for p in paths]!=expected or any(p.parent!=metadata.parent for p in paths):
        raise ValueError('Replacement can only address canonical vertical members and metadata')


def sync_file(path):
    with path.open('rb') as handle:
        os.fsync(handle.fileno())


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def restore_transaction(backup, assets):
    journal=backup/'transaction.json'
    record=json.loads(journal.read_text())
    if record['state']!='PREPARED':
        return
    destinations=[Path(p) for p in record['destinations']]
    replacement_members(destinations)
    if len(destinations)!=3 or any(p.parent.resolve()!=assets.resolve() for p in destinations):
        raise RuntimeError('Invalid replacement recovery destinations')
    for destination in destinations:
        if sha256_file(backup/destination.name)!=record['hashes'][destination.name]:
            raise RuntimeError('Replacement backup checksum mismatch')
    for destination in destinations:
        recovery=destination.with_name('.'+destination.name+'.restore')
        shutil.copy2(backup/destination.name,recovery)
        sync_file(recovery)
        os.replace(recovery,destination)
    sync_directory(assets)
    write_json(journal,{**record,'state':'ROLLED_BACK'})
    sync_directory(backup)


def replace_vertical(staged, final, backup, expected_hashes):
    """Extend package promotion semantics for existing packages, with durable backup.

    Three os.replace operations cannot be atomic together. A journal and copies
    survive SIGKILL; catchable failures roll back, and repair refuses unfinished
    transactions until their originals are recovered. Horizontal paths never
    enter this replacement set. Metadata is last, as in package promotion.
    """
    replacement_members(staged)
    replacement_members(final)
    if any(sha256_file(p) != expected_hashes[p.name] for p in final):
        raise RuntimeError('Package changed since audit')
    if not all(p.is_file() for p in staged):
        raise RuntimeError('Incomplete vertical staging')
    replacement_hashes={p.name:sha256_file(p) for p in staged}
    backup.mkdir(parents=True, exist_ok=False)
    for path in final:
        shutil.copy2(path, backup/path.name)
        sync_file(backup/path.name)
        if sha256_file(backup/path.name)!=expected_hashes[path.name]:
            raise RuntimeError('Replacement backup verification failed')
    journal = backup/'transaction.json'
    write_json(journal, {'state': 'PREPARED', 'destinations': [str(p) for p in final], 'hashes': expected_hashes})
    sync_directory(backup)
    try:
        for source, destination in zip(staged, final):
            sync_file(source)
            os.replace(source, destination)
            sync_directory(destination.parent)
        if any(sha256_file(p)!=replacement_hashes[p.name] for p in final):
            raise RuntimeError('Replacement verification failed')
        write_json(journal, {'state': 'COMPLETE', 'destinations': [str(p) for p in final], 'hashes': expected_hashes})
    except BaseException:
        restore_transaction(backup,final[0].parent)
        raise


def _repair_verticals(location, audit, asset_id=None, reporter=None):
    run, movie, events, shots = context(location)
    for journal in (run/'vertical_repairs').glob('*/*/backup/transaction.json'):
        restore_transaction(journal.parent,run/'assets')
    report = json.loads(Path(audit).read_text())
    if report.get('schema_version') != 'vertical_repair_audit_v1' or Path(report['run']).resolve() != run:
        raise ValueError('Audit does not belong to this run')
    cv2.setNumThreads(2)
    face_capability = preflight(provision=True)
    if not face_capability['available']:
        raise RuntimeError('Face detector unavailable: ' + str(face_capability))
    results = []
    assets = run/'assets'
    for row in report['assets']:
        if row['decision'] != 'REPAIR' or (asset_id and row['producer_asset_id'] != asset_id):
            continue
        metadata = package_path(assets, row['metadata_file'])
        data = json.loads(metadata.read_text())
        aid = data['asset']['id']
        if aid != row['producer_asset_id']:
            raise ValueError('Audit asset identity mismatch')
        vertical = package_path(assets, data['media']['vertical']['file'])
        if sha256_file(vertical) != row['vertical_sha256'] or sha256_file(metadata) != row['metadata_sha256']:
            raise RuntimeError('Stale audit for ' + aid)
        if not f._asset_metadata_contract_valid(data, assets):
            raise RuntimeError('Existing package failed hash/metadata validation')
        history = run/'vertical_repairs'/aid
        for journal in history.glob('*/backup/transaction.json'):
            if json.loads(journal.read_text())['state'] == 'PREPARED':
                raise RuntimeError('Unfinished replacement; recover from ' + str(journal.parent))
        job = history/uuid.uuid4().hex
        stage = job/'staged'
        stage.mkdir(parents=True)
        event = events[data['source_timeline']['visual_event_id']]
        old = audit_asset(metadata, movie, events, shots)
        if old['decision'] != 'REPAIR':
            results.append({'producer_asset_id': aid, 'status': 'SKIPPED', 'audit': old})
            continue
        active_picture=data.get('source',{}).get('active_picture')
        plan = source_plan(movie, event, shots, 'face_priority', active_picture)
        v = stage/vertical.name
        vt = stage/package_path(assets,data['media']['vertical']['thumbnail']['file']).name
        replacement_members([v,vt,stage/metadata.name])
        f.render_vertical(movie, v, event, plan, active_picture)
        expected = data['media']['horizontal']['duration_seconds']
        validation = f.validate_vertical(v, expected, plan, event)
        vtime = f.thumbnail(v, vt)
        result = {'producer_asset_id': aid, 'old_audit': old, 'validation': validation,
                  'strategy': 'face_priority', 'attempts': 1, 'face_detector': face_capability, 'plan': plan, 'stage': str(stage)}
        if validation['status'] != 'PASS':
            result['status'] = 'REVIEW_VERTICAL'
            write_json(job/'result.json', result)
            results.append(result)
            continue
        new = copy.deepcopy(data)
        new['media']['vertical'] = {**data['media']['vertical'],
            **f._rendition(v, vt, vtime, f._technical_properties(v, validation), 'portrait')}
        new['visual']['final_vertical'].update(reframe_algorithm_version=f.REFRAME_ALGORITHM_VERSION,
            vertical_validation_version=f.VERTICAL_VALIDATION_VERSION, validation_status='PASS',
            reframe_fingerprint=f.reframe_fingerprint(event, shots, data['media']['horizontal']['width'], data['media']['horizontal']['height'], active_picture=active_picture))
        new['visual']['final_vertical']['face_safe_validation']={'version':VERSION, 'decision':'PASS', 'model':face_capability.get('model'), 'model_sha256':face_capability.get('sha256'), 'strategy':'face_priority'}
        new=apply_publication_projection(new)
        md = stage/metadata.name
        write_json(md, new)
        horizontal = data['media']['horizontal']
        for name in (horizontal['file'], horizontal['thumbnail']['file']):
            shutil.copy2(package_path(assets, name), stage/name)
        if not f._asset_metadata_contract_valid(new, stage):
            raise RuntimeError('Replacement package failed 5/5 validation')
        final = [vertical, package_path(assets, data['media']['vertical']['thumbnail']['file']), metadata]
        hashes = {p.name: sha256_file(p) for p in final}
        # Audit SHA values are rechecked immediately before promotion.
        if hashes[vertical.name] != row['vertical_sha256'] or hashes[metadata.name] != row['metadata_sha256']:
            raise RuntimeError('Package changed while rendering')
        replace_vertical([v, vt, md], final, job/'backup', hashes)
        result.update(status='PASS', horizontal_unchanged=new['media']['horizontal']==data['media']['horizontal'] and
                      all(sha256_file(assets/n)==sha256_file(stage/n) for n in (horizontal['file'], horizontal['thumbnail']['file'])),
                      package_valid=f._asset_metadata_contract_valid(new, assets), same_asset_id=new['asset']==data['asset'],
                      new_vertical_sha256=sha256_file(vertical))
        write_json(job/'result.json', result)
        results.append(result)
        if reporter:
            reporter(f'{aid}: {result["status"]}')
    result = {'schema_version': 'vertical_repair_result_v1', 'results': results}
    write_json(run/'vertical_repair_result.json', result)
    return result


def repair_verticals(location, audit, asset_id=None, reporter=None):
    import fcntl
    run,_,_,_=context(location)
    # Share the supervisor lock so production supervision and repairs cannot race.
    with (run/'supervisor.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another supervisor or vertical repair is active') from error
        return _repair_verticals(location,audit,asset_id,reporter)
