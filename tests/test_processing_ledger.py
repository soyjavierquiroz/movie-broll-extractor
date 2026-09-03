import json
from pathlib import Path

from movie_broll.processing_ledger import ProcessingLedger, fingerprint


def item(identifier='VE_000001'):
    return {'visual_event_id':identifier,'start_frame':10,'end_frame_exclusive':40,'source_shot_ids':['S1','S2']}


def test_ledger_atomic_resume_and_stale_event(tmp_path):
    ledger=ProcessingLedger(tmp_path,'film',{'movie_sha256':'old'})
    ledger.register(item(), 'first')
    ledger.stage('VE_000001','semantic','RUNNING')
    # A fresh process turns an interrupted RUNNING request into safe retry work.
    resumed=ProcessingLedger(tmp_path,'film',{'movie_sha256':'old'})
    assert resumed.data['events']['VE_000001']['stages']['semantic']['status']=='FAILED_RETRYABLE'
    resumed.stage('VE_000001','semantic','COMPLETE',checkpoint='checkpoint.json')
    resumed.register(item(), 'changed')
    assert resumed.data['events']['VE_000001']['stages']['semantic']['status']=='STALE'
    assert json.loads((tmp_path/'processing_ledger.json').read_text())['movie_id']=='film'


def test_ledger_summary_and_append_only_log(tmp_path):
    ledger=ProcessingLedger(tmp_path,'film',{})
    ledger.register(item(), fingerprint({'a':1}))
    ledger.stage('VE_000001','semantic','COMPLETE')
    summary=ledger.summary(status='COMPLETE',visual_events_total=1,semantic_complete=1,semantic_pending=0)
    assert summary['resume_safe'] is True
    assert json.loads((tmp_path/'progress_summary.json').read_text())['semantic_complete']==1
    assert 'SEMANTIC_COMPLETE' in (tmp_path/'progress.jsonl').read_text()
