"""Local producer handoff. This module has no provider or semantic execution path."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from .publication import human_review_status, is_publish_ready
from .utils import sha256_file, write_json

MEMBERS = (('', '.json'), ('', '.mp4'), ('v', '.mp4'), ('', '.jpg'), ('v', '.jpg'))


def package_names(base):
    return [f'{prefix}{base}{suffix}' for prefix, suffix in MEMBERS]


def validate_media(path: Path, vertical: bool, thumbnail: bool = False):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], capture_output=True, check=True, timeout=60)
        probe = json.loads(result.stdout)
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width, height = int(video['width']), int(video['height'])
        if width <= 0 or height <= 0 or (height > width) != vertical:
            raise ValueError('invalid rendition orientation/dimensions')
        if not thumbnail:
            duration = float(probe['format']['duration'])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('invalid duration')
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-map', '0:v:0', '-f', 'null', '-'], capture_output=True, check=True, timeout=120)
    except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as error:
        raise ValueError(f'invalid media: {path.name}') from error


def validate_package(directory: Path, base: str, data: dict):
    from .finalization import _asset_metadata_contract_valid
    names = package_names(base)
    if not all((directory / name).is_file() for name in names):
        raise ValueError(f'incomplete package: {base}')
    if not _asset_metadata_contract_valid(data):
        raise ValueError(f'invalid metadata contract: {base}')
    if not isinstance(data.get('asset', {}).get('id'), str) or not data['asset']['id']:
        raise ValueError(f'invalid producer identity: {base}')
    if base != f"{data['asset']['id']}-{data['asset'].get('slug', '')}":
        raise ValueError(f'producer identity mismatch: {base}')
    for role, vertical in (('horizontal', False), ('vertical', True)):
        record = data['media'][role]
        expected = f"{'v' if vertical else ''}{base}"
        if record.get('orientation') not in ({'portrait', 'vertical'} if vertical else {'landscape', 'horizontal'}):
            raise ValueError(f'invalid metadata rendition role: {base}')
        for member, suffix in ((record, '.mp4'), (record['thumbnail'], '.jpg')):
            if member.get('file') != expected + suffix:
                raise ValueError(f'invalid metadata filename: {base}')
            path = directory / member['file']
            if member.get('sha256') is not None and member['sha256'] != sha256_file(path):
                raise ValueError(f'hash mismatch: {path.name}')
            if member.get('size_bytes') is not None and member['size_bytes'] != path.stat().st_size:
                raise ValueError(f'hash mismatch (size): {path.name}')
            validate_media(path, vertical, suffix == '.jpg')


def _close(input_dir: Path, output=print):
    episode = input_dir.name
    run = input_dir.resolve().parents[1] / 'runs' / episode
    destination = run / 'final-export'
    counts = dict(auto=0, human=0, unresolved=0, duplicates=0, incomplete=0, invalid=0, hashes=0, mismatch=0)
    approved, seen, errors = [], set(), []
    registry_path = run / 'asset_registry.json'
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    for directory in (run / 'assets', run / 'review'):
        for metadata in sorted(directory.glob('*.json')):
            try:
                data = json.loads(metadata.read_text())
                asset_id = data.get('asset', {}).get('id')
                if not asset_id or asset_id in seen:
                    counts['duplicates'] += 1
                    raise ValueError(f'duplicate/invalid producer ID: {asset_id}')
                seen.add(asset_id)
                vertical = data.get('visual', {}).get('final_vertical', {}).get('validation_status')
                human = human_review_status(data)
                if human == 'REJECTED' or data.get('editorial', {}).get('decision') == 'REJECT':
                    continue
                if vertical == 'REVIEW' and human in {'PENDING', 'NOT_REQUIRED'}:
                    counts['unresolved'] += 1
                    continue
                if vertical not in {'PASS', 'REVIEW'} or human not in {'PENDING', 'NOT_REQUIRED', 'APPROVED'}:
                    raise ValueError(f'ambiguous approval state: {asset_id}')
                if not is_publish_ready(data):
                    raise ValueError(f'ambiguous or broken approved package: {asset_id}')
                kind = 'AUTO_PASS' if vertical == 'PASS' else 'HUMAN_APPROVED'
                counts['auto' if kind == 'AUTO_PASS' else 'human'] += 1
                from .asset_window_handoff import registry_matches
                if not registry_matches(registry, asset_id, data.get('source_timeline', {}).get('visual_event_id'), data['asset'].get('slug'), data.get('source_timeline', {}).get('producer_window_id')):
                    raise ValueError(f'producer/source event mismatch: {asset_id}')
                validate_package(directory, metadata.stem, data)
                names = package_names(metadata.stem)
                approved.append((directory, names, {
                    'producer_asset_id': asset_id,
                    **dict(zip(('metadata_filename', 'horizontal_filename', 'vertical_filename', 'horizontal_thumbnail_filename', 'vertical_thumbnail_filename'), names)),
                    'approval_provenance': {'kind': kind, 'vertical_validation': data['visual']['final_vertical'], 'human_review': data.get('publication', {}).get('human_review', {})},
                    'checksums': {name: sha256_file(directory / name) for name in names},
                    'source_timeline': data.get('source_timeline', {}),
                }))
            except (ValueError, OSError, KeyError, TypeError) as error:
                message = str(error)
                errors.append(message)
                for fragment, key in (('incomplete', 'incomplete'), ('invalid media', 'invalid'), ('hash mismatch', 'hashes')):
                    if fragment in message: counts[key] += 1
    ledger_path = run / 'processing_ledger.json'
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text())
        for event_id, entry in registry.get('events', {}).items():
            if entry.get('asset_id') in seen: continue
            stages = ledger.get('events', {}).get(event_id, {}).get('stages', {})
            final = stages.get('finalization', {})
            decision = final.get('decision')
            if decision == 'REVIEW_VERTICAL':
                counts['unresolved'] += 1
                errors.append(f'missing REVIEW package: {entry.get("asset_id")}')
            elif decision != 'HUMAN_REJECTED':
                counts['incomplete'] += 1
                errors.append(f'incomplete approved package: {entry.get("asset_id")}')
    if not registry_path.is_file(): errors.append('missing producer registry')
    if counts['unresolved']: errors.append('unresolved REVIEW assets')
    match = re.search(r'(?:^|[-_])s\d+e(\d+)(?:$|[-_])', episode, re.I)
    if not match: errors.append('episode identity has no season/episode number')
    manifest_name = f'E{int(match.group(1)):02d}_FINAL_APPROVAL_MANIFEST.json' if match else 'FINAL_APPROVAL_MANIFEST.json'
    records = sorted((item[2] for item in approved), key=lambda item: item['producer_asset_id'])
    manifest = {'schema_version': 'producer_final_approval_manifest_v1', 'episode_key': episode,
                'total_approved': len(records), 'auto_pass_total': counts['auto'], 'human_approved_total': counts['human'],
                'approved_producer_asset_ids': [item['producer_asset_id'] for item in records], 'assets': records}
    expected = {manifest_name} | {name for _, names, _ in approved for name in names}
    idempotent = destination.exists()
    if not errors:
        try:
            if destination.exists():
                if {path.name for path in destination.iterdir()} != expected or json.loads((destination / manifest_name).read_text()) != manifest:
                    raise ValueError('manifest/package mismatch')
                for _, names, record in approved:
                    data = json.loads((destination / names[0]).read_text())
                    validate_package(destination, Path(names[0]).stem, data)
                    if any(sha256_file(destination / name) != record['checksums'][name] for name in names):
                        raise ValueError('hash mismatch in final-export')
            else:
                # Durable staging outside the handoff; interruption is safe to resume.
                stage = run / '.work' / 'final-export'
                stage.mkdir(parents=True, exist_ok=True)
                if any(path.name not in expected for path in stage.iterdir()):
                    raise ValueError('manifest/package mismatch in staging')
                for directory, names, record in approved:
                    for name in names:
                        target = stage / name
                        if target.exists():
                            if sha256_file(target) != record['checksums'][name]: raise ValueError('hash mismatch in staging')
                            continue
                        # Metadata is copied to isolate publication from workspace edits.
                        if name.endswith('.json'): shutil.copy2(directory / name, target)
                        else:
                            try: os.link(directory / name, target)
                            except OSError: shutil.copy2(directory / name, target)
                    validate_package(stage, Path(names[0]).stem, json.loads((stage / names[0]).read_text()))
                write_json(stage / manifest_name, manifest)
                if {path.name for path in stage.iterdir()} != expected: raise ValueError('manifest/package mismatch')
                stage.rename(destination)
            write_json(run / 'handoff_state.json', {'status': 'HANDOFF_COMPLETE', 'episode_key': episode, 'manifest_sha256': sha256_file(destination / manifest_name), 'semantic_calls': 0})
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(str(error)); counts['mismatch'] += 1
    verdict = 'BLOCKED' if errors else 'HANDOFF_COMPLETE'
    output(f'VERDICT: {verdict}')
    output(f'EPISODE: {episode}')
    output(f"EDITORIAL:\n    AUTO PASS = {counts['auto']}\n    HUMAN APPROVED = {counts['human']}\n    TOTAL APPROVED = {counts['auto'] + counts['human']}\n    UNRESOLVED REVIEW = {counts['unresolved']}")
    files = list(destination.iterdir()) if destination.is_dir() else []
    output('FINAL EXPORT:')
    output(f'    packages = {len(approved) if not errors else 0}')
    for label, suffix, vertical in (('asset JSON', '.json', False), ('horizontal MP4', '.mp4', False), ('vertical MP4', '.mp4', True), ('horizontal JPG', '.jpg', False), ('vertical JPG', '.jpg', True)):
        output(f"    {label} = {sum(p.suffix == suffix and p.name != manifest_name and p.name.startswith('v') == vertical for p in files)}")
    output(f'    manifest present = {(destination / manifest_name).is_file()}')
    output(f"IDENTITY:\n    unique producer_asset_id = {len(seen)}\n    sparse IDs preserved = true\n    duplicates = {counts['duplicates']}")
    output(f"VALIDATION:\n    incomplete packages = {counts['incomplete']}\n    invalid media = {counts['invalid']}\n    hash mismatches = {counts['hashes']}\n    manifest/package mismatch = {counts['mismatch']}")
    output('SEMANTIC CALLS DURING CLOSURE: 0')
    output(f'HANDOFF: {destination}')
    if not errors: output(f'IDEMPOTENT COMPLETION: {idempotent}')
    for error in errors: output(f'BLOCKER: {error}')
    return {'verdict': verdict, 'errors': errors, 'manifest': manifest, 'semantic_calls': 0, 'idempotent': idempotent}


def close(input_dir: Path, output=print):
    """Serialize closure with the canonical production owner lock."""
    import fcntl
    run = input_dir.resolve().parents[1] / 'runs' / input_dir.name
    if not run.is_dir():
        output('VERDICT: BLOCKED\nBLOCKER: missing producer run\nSEMANTIC CALLS DURING CLOSURE: 0')
        return {'verdict': 'BLOCKED', 'semantic_calls': 0}
    with (run / 'supervisor.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            output('VERDICT: BLOCKED\nBLOCKER: production owner is active\nSEMANTIC CALLS DURING CLOSURE: 0')
            return {'verdict': 'BLOCKED', 'semantic_calls': 0}
        return _close(input_dir, output)
