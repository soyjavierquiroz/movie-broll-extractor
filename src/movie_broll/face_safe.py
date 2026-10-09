"""Bounded, local face/head evidence. Identity is supplied by the shot tracker.

No recognition, gender inference or negative detection claims.
A missing detector or an unmatched face is UNKNOWN, never evidence of safety.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .active_picture import crop_frame

# This is planning policy, not detector identity.  It is deliberately
# versioned separately so reframe cache keys cannot reuse pixels made before a
# face/head constraint change.
VERSION = 'face_safe_v2_planner_constraints'
FACE_HEAD_CONSTRAINT_VERSION = 'face_head_constraint_v1_margin_12pct_min8'
FACE_HEAD_MARGIN_RATIO = .12
FACE_HEAD_MARGIN_MIN_PX = 8.
_RUNTIME = None
_VERIFIED = None
_PROVISION_RESULT = None
MODEL_NAME = 'face_detection_yunet_2026may.onnx'
MODEL_SHA256 = 'ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0'
MODEL_REVISION = '47534e27c9851bb1128ccc0102f1145e27f23f98'
MODEL_URL = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_REVISION}/models/face_detection_yunet/{MODEL_NAME}'


def preflight(provision=True):
    """Use the production model cache; one bounded acquisition per process."""
    import urllib.request
    import tempfile
    from .utils import sha256_file
    global _PROVISION_RESULT
    info = capability()
    path = Path(info['yunet_model'])
    if not path.exists() and provision and _PROVISION_RESULT is None:
        temp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.download.tmp', delete=False) as out:
                temp = Path(out.name)
                with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
                    data = response.read(1024*1024)
                    if len(data) >= 1024*1024:
                        raise RuntimeError('YuNet model exceeds bounded download size')
                    out.write(data)
            if sha256_file(temp) != MODEL_SHA256:
                raise RuntimeError('YuNet download checksum mismatch')
            temp.replace(path)
        except (OSError, RuntimeError) as error:
            _PROVISION_RESULT = {'available': False, 'failure_reason': str(error)}
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    try:
        if not hasattr(cv2, 'FaceDetectorYN'):
            raise RuntimeError('OpenCV FaceDetectorYN is unavailable')
        verified_model(path)
        # Smoke execution, including the actual OpenCV backend compatibility.
        detect_faces(np.zeros((320,320,3), dtype=np.uint8))
        _PROVISION_RESULT = {'available': True, 'model': MODEL_NAME, 'sha256': MODEL_SHA256,
                             'revision': MODEL_REVISION, 'backend': 'opencv_cpu'}
    except (OSError, RuntimeError, cv2.error) as error:
        _PROVISION_RESULT = {'available': False, 'failure_reason': str(error)}
    return dict(_PROVISION_RESULT)


def verified_model(path):
    from .utils import sha256_file
    global _VERIFIED
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key != _VERIFIED:
        if sha256_file(path) != MODEL_SHA256:
            raise RuntimeError('YuNet cached model checksum mismatch: ' + str(path))
        _VERIFIED = key



def capability():
    from .finalization import _model_path
    model = _model_path().with_name(MODEL_NAME)
    cascade = Path(getattr(getattr(cv2, 'data', None), 'haarcascades', '/unavailable')) / 'haarcascade_frontalface_default.xml'
    return {'yunet_model': str(model), 'yunet_available': hasattr(cv2, 'FaceDetectorYN') and model.is_file(),
            'haar_model': str(cascade), 'haar_available': hasattr(cv2, 'CascadeClassifier') and cascade.is_file(),
            'eyes': 'UNKNOWN', 'fallback': 'tracked_person_upper_region_proxy', 'downloads': 'preflight_only'}


def detect_faces(frame):
    global _RUNTIME
    info = capability()
    h, w = frame.shape[:2]
    scale = min(1., 640 / max(h, w))
    small = cv2.resize(frame, (round(w * scale), round(h * scale)))
    if info['yunet_available']:
        verified_model(Path(info['yunet_model']))
        key = ('yunet', info['yunet_model'])
        if _RUNTIME is None or _RUNTIME[0] != key:
            _RUNTIME = (key, cv2.FaceDetectorYN.create(key[1], '', (320, 320), .85, .3, 1000))
        net = _RUNTIME[1]
        net.setInputSize((small.shape[1], small.shape[0]))
        _, rows = net.detect(small)
        return [{'x': float(r[0]/scale), 'y': float(r[1]/scale), 'width': float(r[2]/scale),
                 'height': float(r[3]/scale), 'confidence': float(r[-1]), 'kind': 'face', 'detector': 'yunet',
                 'landmarks': {name: [float(r[j]/scale), float(r[j+1]/scale)] for name,j in
                               [('right_eye',4),('left_eye',6),('nose',8),('right_mouth',10),('left_mouth',12)]}}
                for r in ([] if rows is None else rows)]
    if info['haar_available']:
        key = ('haar', info['haar_model'])
        if _RUNTIME is None or _RUNTIME[0] != key:
            _RUNTIME = (key, cv2.CascadeClassifier(key[1]))
        return [{'x': float(x/scale), 'y': float(y/scale), 'width': float(bw/scale), 'height': float(bh/scale),
                 'kind': 'face', 'detector': 'haar', 'confidence': .7}
                for x, y, bw, bh in _RUNTIME[1].detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 5)]
    return []


def applies(rule):
    # All single-person editorial targets receive protection, including unfamiliar
    # wording of close-up/reaction/dialogue. No semantic contract additions.
    return rule.get('focus_subject') in {'woman', 'man'} and rule.get('focus_role', 'primary') == 'primary'


def bind_face(faces, target, competitors=()):
    if not target:
        return None
    matches = []
    for face in faces:
        cx = face['x'] + face['width']/2
        cy = face['y'] + face['height']/2
        if (target['x'] <= cx <= target['x'] + target['width'] and
                target['y'] <= cy <= target['y'] + target['height']*.8 and
                face['width'] <= target['width'] * 1.1):
            def containment(person):
                left=max(face['x'],person['x']); right=min(face['x']+face['width'],person['x']+person['width'])
                top=max(face['y'],person['y']); bottom=min(face['y']+face['height'],person['y']+person['height'])
                return max(0.,right-left)*max(0.,bottom-top)/max(1.,face['width']*face['height'])
            own=containment(target)
            others=[containment(p) for p in competitors if
                    abs(p['x']-target['x'])+abs(p['width']-target['width'])>2]
            if own>=.8 and (not others or own-max(others)>=.15):
                matches.append(face)
    # Never select another/easier face or arbitrate overlapping interlocutors.
    return matches[0] if len(matches) == 1 else None


def head_proxy(target):
    if not target:
        return None
    return {'x': target['x'], 'y': target['y'], 'width': target['width'],
            'height': target['height']*.45, 'kind': 'head_proxy', 'confidence': 'low'}


def required_margin(region):
    """The compact editorial buffer used by both planning and verification."""
    return max(FACE_HEAD_MARGIN_MIN_PX, float(region['width']) * FACE_HEAD_MARGIN_RATIO)


def crop_interval(region, source_width, crop_width):
    """Return the horizontal crop interval which keeps an associated region safe.

    The caller has already bound ``region`` to the semantic target person.
    This helper intentionally contains no identity selection logic.
    """
    if not region:
        return None
    margin = required_margin(region)
    maximum = max(0., float(source_width) - float(crop_width))
    raw_lower = float(region['x']) + float(region['width']) + margin - float(crop_width)
    raw_upper = float(region['x']) - margin
    lower = max(0., raw_lower)
    upper = min(maximum, raw_upper)
    source_limited = (
        float(region['x']) <= 1
        or float(region['x']) + float(region['width']) >= float(source_width) - 1
        or raw_lower > raw_upper
    )
    return {
        'min_x': lower,
        'max_x': upper,
        'margin': margin,
        'feasible': lower <= upper,
        'source_limited': source_limited,
        'version': FACE_HEAD_CONSTRAINT_VERSION,
    }


def assess(region, source_width, source_height, crop_x, crop_width, binding=True):
    result = {k: 'UNKNOWN' for k in ('primary_target_present', 'face_detected', 'face_fully_inside_frame',
              'head_safe', 'left_eye_visible', 'right_eye_visible', 'both_eyes_visible',
              'source_face_available', 'source_allows_better_crop', 'repairable_face_crop')}
    result.update(decision='AMBIGUOUS', reason_codes=['face_detection_inconclusive'], repairable=False,
                  source_limited=False, confidence='low', face_edge_margin=None)
    if not binding:
        result['reason_codes'] = ['semantic_target_binding_unresolved']
        return result
    if not region or region.get('kind') != 'face':
        result['evidence_origin'] = 'head_proxy' if region else 'unavailable'
        result['head_proxy'] = region
        return result
    result['evidence_origin'] = region.get('detector', 'injected_face') + '_face'
    x, y, w, h = (region[k] for k in ('x', 'y', 'width', 'height'))
    left, right = x-crop_x, crop_x+crop_width-x-w
    margin = min(left, right)
    result.update(face_detected='MATCH', confidence='medium', face_edge_margin=round(margin/crop_width, 4),
                  face_fully_inside_frame='MATCH' if margin >= 0 and y >= 0 and y+h <= source_height else 'MISMATCH')
    # Body truncation does not establish a source face limit. Only face evidence does.
    source_limited = x <= 1 or x+w >= source_width-1 or y <= 1 or y+h >= source_height-1
    needed = required_margin(region)
    can_fit = w + 2*needed <= crop_width and x >= needed and x+w <= source_width-needed
    result['source_face_available'] = 'MISMATCH' if source_limited else 'MATCH'
    result['source_allows_better_crop'] = 'MATCH' if can_fit and not source_limited else 'MISMATCH'
    if source_limited or not can_fit:
        result.update(decision='SOURCE_LIMIT', source_limited=True,
                      reason_codes=['source_face_already_clipped' if source_limited else 'face_exceeds_vertical_crop'])
    elif margin < needed:
        reason = 'face_partially_clipped_but_source_has_room' if margin < 0 else (
            'primary_face_too_close_to_left_edge' if left < right else 'primary_face_too_close_to_right_edge')
        result.update(decision='REPAIR', repairable=True, repairable_face_crop='MATCH', head_safe='MISMATCH', reason_codes=[reason])
    else:
        result.update(decision='PASS', primary_target_present='MATCH', head_safe='MATCH',
                      repairable_face_crop='MISMATCH', reason_codes=['primary_face_safe'])
    landmarks = region.get('landmarks', {})
    eyes = [landmarks.get(name) for name in ('left_eye', 'right_eye')]
    reliable = (region.get('confidence', 0) >= .9 and all(eyes) and
                .15*w <= abs(eyes[0][0]-eyes[1][0]) <= .8*w and
                abs(eyes[0][1]-eyes[1][1]) <= .25*h and
                all(x <= eye[0] <= x+w and y <= eye[1] <= y+.65*h for eye in eyes))
    result['landmark_evidence'] = 'geometric_consistency' if reliable else 'UNKNOWN'
    if reliable and not source_limited:
        for name in ('left_eye', 'right_eye'):
            ex, ey = landmarks[name]
            result[name+'_visible'] = 'MATCH' if crop_x <= ex <= crop_x+crop_width and 0 <= ey <= source_height else 'MISMATCH'
        result['both_eyes_visible'] = 'MATCH' if all(result[n+'_visible']=='MATCH' for n in ('left_eye','right_eye')) else 'MISMATCH'
        if result['both_eyes_visible']=='MISMATCH' and can_fit:
            result.update(decision='REPAIR', repairable=True, repairable_face_crop='MATCH')
            result['reason_codes'].append('both_eyes_not_visible_when_source_allows')
    return result


def aggregate(samples):
    if not samples:
        return {'decision': 'AMBIGUOUS', 'reason_codes': ['no_face_safe_samples'], 'repairable': False, 'source_limited': False}
    decisions = [s['decision'] for s in samples]
    # A reliable clipping observation cannot be averaged away by good frames.
    decision = next((d for d in ('ERROR', 'REPAIR', 'AMBIGUOUS', 'SOURCE_LIMIT') if d in decisions), 'PASS')
    return {'decision': decision, 'reason_codes': sorted({r for s in samples for r in s['reason_codes']}),
            'repairable': decision == 'REPAIR', 'source_limited': 'SOURCE_LIMIT' in decisions, 'samples': samples}


def register_crop(source, rendered):
    """Measure the crop in real encoded pixels; never trust stored x for QA."""
    sh, sw = source.shape[:2]
    vh, vw = rendered.shape[:2]
    if sh != vh or vw > sw:
        return None, 0.
    scale = min(1., 480/sw)
    src = cv2.resize(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), (round(sw*scale), round(sh*scale)))
    dst = cv2.resize(cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY), (round(vw*scale), src.shape[0]))
    if float(dst.std()) < 5:
        return None, 0.
    scores = cv2.matchTemplate(src, dst, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    return (location[0]/scale if score >= .90 else None), float(score)


def audit_video(path, source, event, plan, face_detector=detect_faces):
    from .finalization import _post_render_sample_indices
    src, dst = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(path))
    samples = []
    try:
        count = int(dst.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = src.get(cv2.CAP_PROP_FPS) or 24.
        for rule in plan:
            if not applies(rule):
                continue
            for index in _post_render_sample_indices(rule, count):
                src.set(cv2.CAP_PROP_POS_FRAMES, int(event['start_frame'])+index)
                dst.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, sf = src.read()
                vok, vf = dst.read()
                base = {'shot_id': rule.get('shot_id'), 'frame': index}
                if not ok or not vok:
                    samples.append({**base, 'decision': 'ERROR', 'reason_codes': ['paired_frame_decode_failed']})
                    continue
                # Plans are active-picture relative, exactly like the vertical
                # render.  Do not bind their boxes against container pixels.
                sf = crop_frame(sf, rule.get('active_picture'))
                time = event['start_seconds'] + index/fps
                evidence = rule.get('face_safe_samples', [])
                nearest = min(evidence, key=lambda r: abs(r['time']-time)) if evidence else {}
                target = nearest.get('target')
                binding = rule.get('target_binding_resolved') is True and len(rule.get('target_person_ids', [])) == 1
                face = bind_face(face_detector(sf), target, nearest.get('competitors',[])) if binding else None
                x, score = register_crop(sf, vf)
                result = assess(face or head_proxy(target), sf.shape[1], sf.shape[0], x or 0, vf.shape[1], binding)
                if x is None:
                    result.update(decision='AMBIGUOUS', repairable=False, face_edge_margin=None,
                                  reason_codes=['source_render_registration_inconclusive'])
                    for signal in ('primary_target_present','face_fully_inside_frame','head_safe',
                                   'left_eye_visible','right_eye_visible','both_eyes_visible',
                                   'source_allows_better_crop','repairable_face_crop'):
                        result[signal]='UNKNOWN'
                # Detector on the real vertical is diagnostic; registered source face
                # geometry remains usable even when clipping defeats that detector.
                result['rendered_face_count'] = len(face_detector(vf))
                samples.append({**base, **result, 'source_region': face, 'observed_crop_x': x,
                                'registration_score': round(score, 5), 'target_person_ids': rule.get('target_person_ids', [])})
    finally:
        src.release()
        dst.release()
    if not any(applies(r) for r in plan):
        return {'version': VERSION, 'decision': 'PASS', 'reason_codes': ['face_safe_not_applicable'], 'repairable': False, 'source_limited': False, 'samples': []}
    return {'version': VERSION, **aggregate(samples)}
