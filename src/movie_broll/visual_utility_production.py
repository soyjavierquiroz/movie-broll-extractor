"""Automatic Policy V2 lifecycle. Historical jobs keep their recorded policy."""
from __future__ import annotations
import os
import time
from . import visual_utility_resolution as resolver
from . import broll_policy_v2 as policy
from .asset_window_handoff import producer_candidates
from .processing_ledger import ProcessingLedger
from .utils import write_json


class LazyBatchProvider:
    """No client or credentials required to finish a fully cached invocation."""
    def __init__(self, factory, model=None, env_file=None):
        self.factory = factory
        self.identifier = os.getenv('SEMANTIC_PROVIDER', 'openai')
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-6-luna') if self.identifier == 'openai' else model or 'gemini-3.6-flash'
        self.provider = None
        self.transport_attempts = 1
        if self.identifier == 'gemini':
            from .gemini_credentials import GeminiCredentialSource
            configured = GeminiCredentialSource(env_file, os.environ).discover()
            self.transport_attempts = max(1, len(configured.primaries) + bool(configured.backup))

    def generate(self, prompt, context, images):
        if self.provider is None:
            env = dict(os.environ)
            env['OPENAI_MAX_RETRIES'] = '1'
            env['OPENAI_IMAGE_DETAIL'] = 'high'
            self.provider = self.factory(self.model, response_model=resolver.BatchResponse,
                                         response_schema=resolver.RESPONSE_SCHEMA, environ=env)
            if self.provider is None:
                raise RuntimeError('semantic provider configuration missing')
        if hasattr(self.provider, 'credential_source'):
            self.provider.credential_source = None
        if resolver.transport_bound(self.provider) > self.transport_attempts:
            raise RuntimeError('provider transport roster exceeds planned bound')
        return self.provider.generate(prompt, context, images)


def process(input_dir, info, provider, model, report, started):
    # Import orchestrator seams here so production tests can inject detectors,
    # source discovery and finalization without constructing any provider.
    from . import production as p
    run = info['run']; video = info['metadata']['video']
    source = run/'source_fingerprint.json'
    old = p._read_json(source).get('movie_sha256') if source.is_file() else None
    p._invalidate_source_media(run, old, info['movie_sha256'])
    write_json(source, {'movie_sha256':info['movie_sha256'],'srt_sha256':info['srt_sha256']})
    ledger = ProcessingLedger(run,input_dir.name,{'movie_sha256':info['movie_sha256'],
                             'srt_sha256':info['srt_sha256'],'orchestrator_version':'production_policy_v2'})
    info['_ledger']=ledger; info['_report']=report
    ledger.log('PRODUCTION_STARTED',mode='production',policy_version=policy.POLICY_VERSION)
    shots=p._bounded_operation('technical-shots',lambda:p._technical_shots(info),report,ledger)
    store_path,store=p._event_store(info,shots)
    if store.get('events') and store.get('event_discovery_status')=='COMPLETE':
        events=store['events']
    else:
        events=p._bounded_operation('visual-events full timeline',lambda:p._visual_events(info,shots),report,ledger)
    p._assign_timeline_ordinals(events)
    store.update(events=events,event_discovery_status='COMPLETE',policy_version=policy.POLICY_VERSION,
                 production_status='RUNNING',status='RUNNING')
    write_json(store_path,store)
    active=p._active_picture(info)
    def sheet(event,evidence):
        return p.candidate_contact_sheet(info['movie'],event,float(video['fps']),evidence,active,evidence_profile=resolver.PROFILE)
    batch_provider=provider or LazyBatchProvider(
        lambda *args,**kwargs:p.build_semantic_provider_from_env(*args,reporter=report,env_file=p._root(input_dir)/'.env',**kwargs),model,p._root(input_dir)/'.env')
    resolution=p._bounded_operation('batched visual utility resolution',lambda:resolver.resolve(input_dir,events,
        movie_sha256=info['movie_sha256'],provider=batch_provider,make_contact_sheet=sheet,
        fps=float(video['fps']),active_picture=active,reporter=report),report,ledger)
    write_json(run/'visual_utility_resolution.json',resolution)
    candidates=[]
    for event in events:
        eid=event['visual_event_id']; decision=resolution['decisions'].get(eid)
        state=resolution['event_states'].get(eid,'PENDING_OBSERVATION')
        if decision is None:
            # Missing requests/formats/provider failures are pipeline states,
            # never a human editorial REVIEW or a validated rejection.
            event['editorial']={'status':state,'policy_version':policy.POLICY_VERSION}
            continue
        event['editorial']={**event.get('editorial',{}),'status':'VALIDATED','decision':decision['policy_decision'],
                            'policy_version':policy.POLICY_VERSION,'policy_reasons':decision['policy_reasons']}
        # Focus identity comes from the same saved visual observation used by
        # policy, not an old event-projection or a second semantic provider call.
        saved=resolver.latest_record(run,event)
        native=policy.latest_visual_reuse_record(run,event)
        record=saved or native or resolver.source_observation(run,event)
        if record:
            event.setdefault('visual',{})['shot_focus_plan']=record['observation']['shot_focus_plan']
        candidates.extend(producer_candidates(event,decision))
    store['events']=events
    store['producer_windows']=[{'producer_window_id':c['producer_window_id'],'visual_event_id':c['visual_event_id'],
        'source_visual_event_id':c['source_visual_event_id'],'start_frame':c['start_frame'],
        'end_frame_exclusive':c['end_frame_exclusive']} for c in candidates]
    write_json(store_path,store)
    write_json(run/'visual_events.json',{'schema_version':'visual_events_v1','events':events,'policy_version':policy.POLICY_VERSION})
    finals=[]; render_complete=True
    # Bounded finalization groups; windows use their own durable ledger IDs.
    for index in range(0,len(candidates),resolver.TARGET_EVENTS):
        group=candidates[index:index+resolver.TARGET_EVENTS]
        detector=p._bounded_operation('person-detector-preflight',p.person_detector_preflight,report,ledger) if index==0 else detector
        final=p._bounded_operation('finalize visual utility windows',lambda:p.finalize_pilot(input_dir,
            f'UTILITY_{index//resolver.TARGET_EVENTS+1:04d}',candidates=group,shots={s['shot_id']:s for s in shots},
            detector_preflight=detector),report,ledger)
        finals.append(final)
        render_complete &= final.get('status','COMPLETE')=='COMPLETE' and p._batch_finalization_complete(group,ledger)
    status='COMPLETE' if resolution['status']=='COMPLETE' and render_complete else 'PARTIAL'
    store.update(production_status=status,status=status,completed_at=p._utc() if status=='COMPLETE' else None)
    write_json(store_path,store)
    summary=p._summary(info,ledger,events,status,stage='complete' if status=='COMPLETE' else 'visual_utility_resolution',
        technical_shot_count=len(shots),timings={'total_seconds':time.monotonic()-started})
    summary['visual_utility']={k:resolution[k] for k in ('event_states','pipeline_pending_event_ids','requests','http_requests','reused')}
    summary['planned_asset_windows']=len(candidates)
    ledger.summary(**summary)
    return {'status':status,'summary':summary,'semantic':[resolution],'finalization':finals}
