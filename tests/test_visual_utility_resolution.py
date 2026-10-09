"""Synthetic batch execution only; no clients, credentials or live providers."""
import copy
import json
import socket
from pathlib import Path

import pytest

from movie_broll import visual_utility_resolution as r
from movie_broll import broll_policy_v2 as policy
from movie_broll.broll_semantics import SemanticResponse
from movie_broll.temporal_evidence import sample_plan
from movie_broll.utils import write_json
from test_broll_policy_v2 import fixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *args: pytest.fail('real provider/network forbidden'))


def items(n):
    events, bodies = [], {}
    for index in range(n):
        event, record = fixture()
        offset = index*240
        event.update(visual_event_id=f'VE_{index:03d}',candidate_id=f'C_{index}',timeline_ordinal=index+1,
                     start_frame=offset,end_frame_exclusive=offset+240,start_seconds=offset/24,end_seconds=(offset+240)/24,
                     visual={'shot_focus_plan':record['observation']['shot_focus_plan']})
        for shot in event['technical_shots']:
            shot['start_frame']+=offset; shot['end_frame_exclusive']+=offset
        b=record['observation']; b['event_id']=event['visual_event_id']
        m=b['visual_moments'][0]; m.update(start_frame=offset,end_frame_exclusive=offset+240,reuse_key=f'need_{index}')
        b.update(visually_reusable='true',visual_moment_type='sustained_state',sustained_state='true',confidence=.9,
                 evidence_sample_ids=m['sample_ids'],utility_assessments=[{'utility_class':k,
                     'assessment':'useful' if k=='useful_state' else 'absent',
                     'sample_ids':m['sample_ids'] if k=='useful_state' else []} for k in policy.UTILITY_CLASSES])
        events.append(event); bodies[event['visual_event_id']]=b
    return events,bodies


def setup(tmp_path,n=8):
    source=tmp_path/'input'/'synthetic'; source.mkdir(parents=True)
    run=tmp_path/'runs'/source.name
    events,bodies=items(n)
    write_json(run/'source_fingerprint.json',{'movie_sha256':'source'})
    write_json(run/'visual_event_segments_v1.json',{'events':events})
    write_json(run/'source-v1/source_manifest.json',{'metadata':{'video':{'fps':24}}})
    return source,run,events,bodies


def sheet(event,out):
    out.update(evidence_profile=r.PROFILE,fps=24,samples=sample_plan(event,24),technical_shots=[])
    return event['visual_event_id'].encode()


class Fake:
    identifier='synthetic'; model='fixture'
    def __init__(self,bodies,transform=None):
        self.bodies=bodies; self.calls=[]; self.transform=transform
    def generate(self,prompt,context,images):
        assert len(context['events'])==len(images)
        assert 'People speaking does not make' in prompt
        ids=[e['event_id'] for e in context['events']]
        self.calls.append(ids)
        result=[copy.deepcopy(self.bodies[eid]) for eid in ids]
        if self.transform:
            result=self.transform(result,len(self.calls))
        return SemanticResponse({'events':[json.dumps(b) for b in result]}, {'total_tokens':10},self.identifier,self.model)


def resolve(source,events,fake,**kwargs):
    return r.resolve(source,events,movie_sha256='source',provider=fake,make_contact_sheet=sheet,fps=24,**kwargs)


def test_eight_independent_events_one_request_and_policy_windows(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    fake=Fake(bodies)
    result=resolve(source,events,fake)
    assert result['status']=='COMPLETE' and result['requests']==1
    assert len(fake.calls[0])==8 and len(result['asset_windows'])==8
    assert all(d['policy_decision']=='KEEP' for d in result['decisions'].values())
    assert all(r.latest_record(run,e)['status']=='VALID' for e in events)
    assert policy.offline_projection(source)['counts']['KEEP']==8


def test_malformed_member_preserves_siblings_and_retries_only_failure(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    def damage(rows,call):
        if call==1: rows[3].pop('confidence')
        return rows
    fake=Fake(bodies,damage)
    result=resolve(source,events,fake)
    assert result['status']=='COMPLETE' and fake.calls==[[e['visual_event_id'] for e in events],[events[3]['visual_event_id']]]
    assert len(list((run/r.ROOT/'events').rglob('*.json')))==8


def test_missing_member_has_independent_durable_state(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    fake=Fake(bodies,lambda rows,call:rows[:-1])
    result=resolve(source,events,fake)
    missing=events[-1]['visual_event_id']
    assert fake.calls[1]==[missing]
    assert result['event_states'][missing]=='MISSING_FROM_BATCH_RESPONSE'
    assert missing not in result['decisions']
    assert result['status']=='PARTIAL' and len(result['asset_windows'])==7
    calls=len(fake.calls)
    again=resolve(source,events,fake)
    assert len(fake.calls)==calls and again['reused']==7


class Rejected(ValueError):
    code='content_policy_violation'
    status_code=400


def test_rejection_bisects_unchanged_and_quarantines_single_member(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    bad=events[5]['visual_event_id']
    class RejectOne(Fake):
        def generate(self,prompt,context,images):
            ids=[e['event_id'] for e in context['events']]
            if bad in ids:
                self.calls.append(ids)
                assert images==[eid.encode() for eid in ids]
                raise Rejected('content policy violation')
            return super().generate(prompt,context,images)
    fake=RejectOne(bodies)
    result=resolve(source,events,fake)
    assert result['event_states'][bad]=='PROVIDER_BLOCKED'
    assert len(result['asset_windows'])==7
    assert len(fake.calls)==7  # binary isolation path plus all clean siblings
    successful={e for c in fake.calls if bad not in c for e in c}
    assert successful==set(bodies)-{bad}
    count=len(fake.calls)
    again=resolve(source,events,fake)
    assert len(fake.calls)==count and again['reused']==7
    assert r.offline_plan(source)['quarantined_events_excluded']==1


def test_event_fingerprints_stable_across_regrouping(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    fake=Fake(bodies)
    resolve(source,events,fake,target=6)
    first={e['visual_event_id']:r.latest_record(run,e)['semantic_fingerprint'] for e in events}
    again=resolve(source,list(reversed(events)),fake,target=8)
    assert again['requests']==0 and again['reused']==8
    assert first=={e['visual_event_id']:r.latest_record(run,e)['semantic_fingerprint'] for e in events}


def test_resume_recovers_raw_batch_before_any_repayment(tmp_path,monkeypatch):
    source,run,events,bodies=setup(tmp_path)
    fake=Fake(bodies)
    original=r._commit
    monkeypatch.setattr(r,'_commit',lambda *a:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):resolve(source,events,fake)
    assert len(fake.calls)==1 and not list((run/r.ROOT/'events').rglob('*.json'))
    monkeypatch.setattr(r,'_commit',original)
    again=resolve(source,events,fake)
    assert again['requests']==0 and again['reused']==8 and len(fake.calls)==1


def test_resume_after_first_batch_skips_completed_events(tmp_path):
    source,run,events,bodies=setup(tmp_path,16)
    class Interrupt(Fake):
        def generate(self,prompt,context,images):
            if len(self.calls)==1: raise KeyboardInterrupt()
            return super().generate(prompt,context,images)
    fake=Interrupt(bodies)
    with pytest.raises(KeyboardInterrupt):resolve(source,events,fake)
    resumed=Fake(bodies)
    result=resolve(source,events,resumed)
    assert result['reused']==8 and resumed.calls==[[e['visual_event_id'] for e in events[8:]]]


def test_inconclusive_is_cached_editorial_review_not_retry(tmp_path):
    source,run,events,bodies=setup(tmp_path,1)
    b=bodies[events[0]['visual_event_id']]
    b.update(visually_reusable='unclear',visual_moment_type='unclear',visual_moments=[],sustained_state='unclear')
    for judgment in b['utility_assessments']:judgment.update(assessment='unclear',sample_ids=[])
    fake=Fake(bodies)
    result=resolve(source,events,fake)
    assert result['status']=='COMPLETE' and result['event_states'][events[0]['visual_event_id']]=='VALID_INCONCLUSIVE'
    assert result['decisions'][events[0]['visual_event_id']]['policy_decision']=='REVIEW'
    assert not result['decisions'][events[0]['visual_event_id']]['targeted_reobservation']
    assert resolve(source,events,fake)['requests']==0
    assert policy.offline_projection(source)['additional_provider_calls_expected']==0


def test_bounded_validation_failure_is_pipeline_state_not_editorial_review(tmp_path):
    source,run,events,bodies=setup(tmp_path,1)
    fake=Fake(bodies,lambda rows,call:[{'event_id':events[0]['visual_event_id']}])
    result=resolve(source,events,fake)
    assert result['requests']==2 and not result['decisions']
    assert result['event_states'][events[0]['visual_event_id']]=='VALIDATION_BLOCKED'
    assert resolve(source,events,fake)['requests']==0


def test_offline_plan_is_readonly_and_generic(tmp_path):
    source,run,events,bodies=setup(tmp_path,17)
    before={str(p):p.read_bytes() for p in run.rglob('*') if p.is_file()}
    plan=r.offline_plan(source)
    assert plan['unresolved_logical_observations']==17 and plan['planned_batches']==3
    assert plan['batch_size_distribution']=={1:1,8:2}
    assert plan['maximum_provider_requests']==62
    assert before=={str(p):p.read_bytes() for p in run.rglob('*') if p.is_file()}
    assert 'mi-otra' not in Path(r.__file__).read_text() and 's03e03' not in Path(r.__file__).read_text()


def test_batch_limit_splits_on_samples_and_rejects_invalid_targets():
    events,_=items(10)
    members=[{'event':e,'recipe':{'samples':[{}]*16},'context':{}} for e in events]
    assert list(map(len,r.partition(members,target=10)))==[8,2]
    with pytest.raises(ValueError):r.partition(members,target=0)


def test_wrong_event_citations_do_not_contaminate_siblings(tmp_path):
    source,run,events,bodies=setup(tmp_path,2)
    bodies[events[0]['visual_event_id']]['evidence_sample_ids']=['INVENTED']
    fake=Fake(bodies)
    result=resolve(source,events,fake)
    assert result['event_states'][events[0]['visual_event_id']]=='VALIDATION_BLOCKED'
    assert result['event_states'][events[1]['visual_event_id']]=='VALID'
    assert fake.calls[1]==[events[0]['visual_event_id']]


def production_setup(tmp_path,monkeypatch,n=8):
    from movie_broll import production as p
    source,run,events,bodies=setup(tmp_path,n)
    (run/'visual_event_segments_v1.json').unlink()  # synthetic first run, no historical policy
    (source/'movie.mp4').write_bytes(b'synthetic')
    (source/'subtitles.srt').write_text('1\n00:00:00,000 --> 00:00:10,000\nDialogue context\n')
    write_json(run/'narrative-v2/narrative_map.json',{'segments':[]})
    info={'movie':source/'movie.mp4','srt':source/'subtitles.srt','run':run,
          'narrative':run/'narrative-v2/narrative_map.json','movie_sha256':'source','srt_sha256':'srt',
          'metadata':{'video':{'fps':24,'width':160,'height':120}},'active_picture':{}}
    monkeypatch.setattr(p,'preflight',lambda _:info)
    monkeypatch.setattr(p,'_active_picture',lambda _: {})
    monkeypatch.setattr(p,'_technical_shots',lambda _: [s for e in events for s in e['technical_shots']])
    monkeypatch.setattr(p,'_visual_events',lambda *_:copy.deepcopy(events))
    monkeypatch.setattr(p,'_event_store',lambda *_:(run/'visual_event_segments_v1.json',
        json.loads((run/'visual_event_segments_v1.json').read_text()) if (run/'visual_event_segments_v1.json').is_file() else {}))
    monkeypatch.setattr(p,'_bounded_operation',lambda _label,action,*_:action())
    monkeypatch.setattr(p,'candidate_contact_sheet',lambda _movie,event,_fps,out,*args,**kwargs:sheet(event,out))
    monkeypatch.setattr(p,'person_detector_preflight',lambda:{})
    finalized=[]
    def finalize(*args,**kwargs):
        finalized.extend(copy.deepcopy(kwargs['candidates']))
        return {'status':'COMPLETE','completed':len(kwargs['candidates'])}
    monkeypatch.setattr(p,'finalize_pilot',finalize)
    monkeypatch.setattr(p,'semantic_validate',lambda *args,**kwargs:pytest.fail('no manual/legacy semantic stage'))
    return source,run,events,bodies,finalized


def test_new_run_automatically_resolves_batches_plans_and_hands_off_windows(tmp_path,monkeypatch):
    from movie_broll import production as p
    source,run,events,bodies,finalized=production_setup(tmp_path,monkeypatch)
    fake=Fake(bodies)
    result=p.process(source,provider=fake)
    assert result['status']=='COMPLETE' and len(fake.calls)==1 and len(fake.calls[0])==8
    assert len(finalized)==8 and all(c['producer_window_id'] for c in finalized)
    assert all(c['source_visual_event_id'] in bodies for c in finalized)
    assert json.loads((run/'visual_event_segments_v1.json').read_text())['policy_version']==policy.POLICY_VERSION
    assert p.process(source,provider=fake)['status']=='COMPLETE' and len(fake.calls)==1


def test_new_run_renders_valid_siblings_and_retains_pipeline_failure(tmp_path,monkeypatch):
    from movie_broll import production as p
    source,run,events,bodies,finalized=production_setup(tmp_path,monkeypatch)
    fake=Fake(bodies,lambda rows,call:rows[:-1])
    result=p.process(source,provider=fake)
    assert result['status']=='PARTIAL' and len(finalized)==7
    stored=json.loads((run/'visual_event_segments_v1.json').read_text())['events']
    failed=stored[-1]['editorial']
    assert failed['status']=='MISSING_FROM_BATCH_RESPONSE' and 'decision' not in failed
    assert result['summary']['editorial']['REVIEW']==0


def test_lazy_batch_provider_constructs_only_on_work_and_uses_batch_schema(tmp_path,monkeypatch):
    from movie_broll import production as p
    source,run,events,bodies,finalized=production_setup(tmp_path,monkeypatch,2)
    fake=Fake(bodies); configurations=[]
    def factory(*args,**kwargs):
        configurations.append(kwargs)
        return fake
    monkeypatch.setattr(p,'build_semantic_provider_from_env',factory)
    assert p.process(source)['status']=='COMPLETE'
    assert configurations[0]['response_model'] is r.BatchResponse
    assert configurations[0]['environ']['OPENAI_MAX_RETRIES']=='1'
    monkeypatch.setattr(p,'build_semantic_provider_from_env',lambda *args,**kwargs:pytest.fail('cached run must not construct provider'))
    assert p.process(source)['status']=='COMPLETE'


def test_provider_auth_defer_is_pipeline_pending_and_does_not_bisect(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    class Auth(Fake):
        def generate(self,*args):
            self.calls.append('auth')
            error=RuntimeError('auth failed'); error.status_code=401
            raise error
    fake=Auth(bodies)
    result=resolve(source,events,fake)
    assert len(fake.calls)==1 and result['status']=='PARTIAL'
    assert all(s=='PENDING_OBSERVATION' for s in result['event_states'].values())
    assert not result['decisions'] and not r.quarantine_ids(run,events)


def test_adapters_send_separate_labeled_batch_images_without_live_clients():
    from types import SimpleNamespace
    from movie_broll.broll_semantics import OpenAISemanticProvider, GeminiBrollSemanticProvider
    calls=[]
    result=SimpleNamespace(output_parsed=r.BatchResponse(events=['{"event_id":"E1"}','{"event_id":"E2"}']),model='fixture',usage=None)
    client=SimpleNamespace(responses=SimpleNamespace(parse=lambda **kwargs:calls.append(kwargs) or result))
    provider=OpenAISemanticProvider('synthetic',client=client,response_model=r.BatchResponse,max_retries=1)
    context={'events':[{'event_id':'E1'},{'event_id':'E2'}]}
    provider.generate(r.PROMPT,context,[b'one',b'two'])
    content=calls[0]['input'][0]['content']
    assert len([p for p in content if p['type']=='input_image'])==2
    assert [p['text'] for p in content if p['type']=='input_text'][1:]==['Event image 1: E1','Event image 2: E2']
    gemini=GeminiBrollSemanticProvider.__new__(GeminiBrollSemanticProvider)
    gemini.model='fixture'; gemini.response_schema=r.RESPONSE_SCHEMA
    gemini_calls=[]
    gemini.client=SimpleNamespace(interactions=SimpleNamespace(create=lambda **kwargs:gemini_calls.append(kwargs) or SimpleNamespace(output_text='{"events":[]}',usage=None)))
    gemini.generate(r.PROMPT,context,[b'one',b'two'])
    assert len([p for p in gemini_calls[0]['input'][0]['content'] if p['type']=='image'])==2


def test_batch_identity_captures_provider_versions_and_event_order():
    members=[{'semantic_fingerprint':'one'},{'semantic_fingerprint':'two'}]
    identity=r.batch_identity(members,'p','m')
    assert identity['ordered_event_input_fingerprints']==['one','two']
    assert identity['schema_version']==r.SCHEMA and identity['prompt_version']==r.BATCH_PROMPT_VERSION
    assert identity!=r.batch_identity(list(reversed(members)),'p','m')
    assert identity!=r.batch_identity(members,'p','another-model')


def test_request_budget_persists_across_regrouping(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    class Down(Fake):
        def generate(self,*args):
            self.calls.append('failed')
            raise TimeoutError('timeout')
    fake=Down(bodies)
    resolve(source,events,fake)
    path=next((run/r.ROOT/'executions').glob('*.json'))
    budget=json.loads(path.read_text()); budget['initial_plan']['maximum_provider_requests']=1
    write_json(path,budget)
    again=resolve(source,events,fake,target=6)
    assert again['requests']==0 and len(fake.calls)==1
    assert all(s=='REQUEST_BUDGET_DEFERRED' for s in again['event_states'].values())


def test_status_counts_batch_requests_once_and_exposes_pipeline_states(tmp_path,monkeypatch):
    from movie_broll import production as p
    from movie_broll.production_run import read_status
    source,run,events,bodies,finalized=production_setup(tmp_path,monkeypatch)
    fake=Fake(bodies,lambda rows,call:rows[:-1])
    p.process(source,provider=fake)
    status=read_status(source)
    assert status['api_usage']['requests']==2 and status['api_usage']['usage']['total_tokens']==20
    assert status['active_semantic_contract']==r.SCHEMA and status['review']==0
    assert status['visual_utility_pipeline_states']['MISSING_FROM_BATCH_RESPONSE']==1
    assert status['semantic_observations']['policy_version']==policy.POLICY_VERSION


def test_fully_assessed_weak_dialogue_rejects_without_completion_gate(tmp_path):
    source,run,events,bodies=setup(tmp_path,1)
    body=bodies[events[0]['visual_event_id']]
    body.update(visually_reusable='false',visual_moments=[],visual_moment_type='unclear',sustained_state='false',
                moment_status='unclear',visual_utility_kind='generic_dialogue_only',conversation_present='true',
                action_or_moment_complete='false')
    for j in body['utility_assessments']:
        j.update(assessment='weak' if j['utility_class']=='conversation_scene' else 'absent',
                 sample_ids=body['evidence_sample_ids'] if j['utility_class']=='conversation_scene' else [])
    fake=Fake(bodies)
    result=resolve(source,events,fake)
    assert result['status']=='COMPLETE' and next(iter(result['decisions'].values()))['policy_decision']=='REJECT'
    assert len(fake.calls)==1


def test_bad_batch_envelope_never_consumes_event_semantic_attempts(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    class BadEnvelope(Fake):
        def generate(self,prompt,context,images):
            self.calls.append([e['event_id'] for e in context['events']])
            return SemanticResponse({'event_id':events[0]['visual_event_id']},{},self.identifier,self.model)
    fake=BadEnvelope(bodies)
    result=resolve(source,events,fake)
    assert len(fake.calls)==2
    assert set(result['event_states'].values())=={'BATCH_CONTRACT_BLOCKED'}
    assert not list((run/r.ROOT/'attempts').rglob('*.json'))
    assert resolve(source,events,fake)['requests']==0


def test_sdk_batch_parse_error_does_not_poison_event_fingerprints(tmp_path):
    from movie_broll.broll_semantics import OpenAIStructuredOutputError
    source,run,events,bodies=setup(tmp_path)
    class BadParse(Fake):
        def generate(self,*args):
            self.calls.append('parse')
            raise OpenAIStructuredOutputError(ValueError('invalid envelope'))
    fake=BadParse(bodies)
    result=resolve(source,events,fake)
    assert len(fake.calls)==2 and set(result['event_states'].values())=={'BATCH_CONTRACT_BLOCKED'}
    assert not list((run/r.ROOT/'attempts').rglob('*.json'))


def test_timeout_then_invalid_batch_still_has_semantic_retry(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    class FirstTimeout(Fake):
        timeout_config={'read':300}
        def generate(self,*args):
            if not self.calls:
                saved = [json.loads(p.read_text()) for p in (run/r.ROOT/'requests').glob('*.json')]
                assert len(saved) == 1 and saved[0]['status'] == 'IN_FLIGHT'
                assert saved[0]['request_id'] == args[1]['batch_request_id']
                self.calls.append('timeout')
                raise TimeoutError('timed out')
            return super().generate(*args)
    def damage(rows,call):
        if call==2:
            for row in rows:row['shot_focus_plan']=[]
        return rows
    fake=FirstTimeout(bodies,damage)
    first=resolve(source,events,fake)
    assert first['requests']==1 and not list((run/r.ROOT/'attempts').rglob('*.json'))
    request=next((run/r.ROOT/'requests').glob('*.json')); saved=json.loads(request.read_text())
    assert saved['status']=='TRANSPORT_DEFERRED' and saved['identity']['ordered_event_input_fingerprints']
    assert saved['transport_config']=={'read':300} and saved['failure']['exception_class']=='TimeoutError'
    again=resolve(source,events,fake)
    assert again['status']=='COMPLETE' and len(fake.calls)==3
    assert json.loads(request.read_text())==saved
    assert resolve(source,events,fake)['requests']==0


def test_repeated_timeouts_stay_transport_and_never_exhaust_semantics(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    class Timeout(Fake):
        def generate(self,*args):
            self.calls.append('timeout');raise TimeoutError('timed out')
    fake=Timeout(bodies)
    resolve(source,events,fake);resolve(source,events,fake)
    result=resolve(source,events,fake)
    assert len(fake.calls)==2 and set(result['event_states'].values())=={'TRANSPORT_DEFERRED'}
    assert not list((run/r.ROOT/'attempts').rglob('*.json'))


def test_offline_focus_recovery_preserves_raw_and_rejects_semantic_conflict(tmp_path,monkeypatch):
    source,run,events,bodies=setup(tmp_path,2)
    bodies[events[1]['visual_event_id']]['visual_utility_kind']='generic_dialogue_only'
    fake=Fake(bodies,lambda rows,call:[{**row,'shot_focus_plan':[]} for row in rows])
    resolve(source,events,fake)
    for f in (run/r.ROOT/'requests').glob('*.json'):
        row=json.loads(f.read_text())
        for m in row['members']:
            m['context']['existing_observation']={'shot_focus_plan':events[0]['visual']['shot_focus_plan']}
        write_json(f,row)
    raw={f:f.read_bytes() for f in (run/r.ROOT/'requests').glob('*.json')}
    attempts={f:f.read_bytes() for f in (run/r.ROOT/'attempts').rglob('*.json')}
    monkeypatch.setattr(fake,'generate',lambda *a:pytest.fail('offline recovery called provider'))
    r.recover(run,reuse_existing_focus=True)
    record=r.latest_record(run,events[0])
    assert record and record['derived_fields']['shot_focus_plan']['derived_from']=='existing_observation'
    assert r.latest_record(run,events[1]) is None
    assert all(f.read_bytes()==b for f,b in {**raw,**attempts}.items())
    assert resolve(source,events,fake)['reused']==1


def test_batch_parser_preserves_ids_with_reordered_members(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    fake=Fake(bodies,lambda rows,call:list(reversed(rows)))
    result=resolve(source,events,fake)
    assert result['status']=='COMPLETE'
    assert [r.latest_record(run,e)['event_id'] for e in events]==[e['visual_event_id'] for e in events]


def test_offline_plan_uses_geometry_inside_active_picture_artifact(tmp_path):
    source,run,events,bodies=setup(tmp_path)
    geometry={'x':0,'y':0,'width':160,'height':120}
    write_json(run/'active_picture.json',{'active_picture':geometry,'status':'VALID'})
    fake=Fake(bodies)
    resolve(source,events,fake,active_picture=geometry)
    plan=r.offline_plan(source)
    assert plan['completed_observations_reused']==8 and plan['unresolved_logical_observations']==0


def test_resume_feedback_is_loaded_from_saved_failed_member(tmp_path,monkeypatch):
    source,run,events,bodies=setup(tmp_path,1)
    def invalid(rows,call):
        rows[0]['shot_focus_plan']=[]
        return rows
    fake=Fake(bodies,invalid)
    # Interrupt between the first durable member attempt and the next dispatch.
    original=r._validation_feedback
    def interrupt(*args):raise KeyboardInterrupt()
    monkeypatch.setattr(r,'_validation_feedback',interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):resolve(source,events,fake)
    finally:monkeypatch.setattr(r,'_validation_feedback',original)
    class Inspect(Fake):
        def generate(self,prompt,context,images):
            feedback=context['events'][0]['validation_feedback']
            assert feedback['event_id']==events[0]['visual_event_id']
            assert 'one_focus_directive_per_technical_shot' in feedback['errors']
            return super().generate(prompt,context,images)
    resumed=Inspect(bodies)
    assert resolve(source,events,resumed)['status']=='COMPLETE'
