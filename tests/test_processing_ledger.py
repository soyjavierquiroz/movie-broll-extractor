import json
from pathlib import Path
import pytest

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

def test_review_is_a_decision_not_a_generic_status(tmp_path):
    ledger=ProcessingLedger(tmp_path,'film',{}); ledger.register(item(), 'x')
    with pytest.raises(ValueError, match='invalid ledger status REVIEW_VERTICAL'):
        ledger.stage('VE_000001','vertical_validation','REVIEW_VERTICAL')
    ledger.stage('VE_000001','vertical_validation','COMPLETE',decision='REVIEW_VERTICAL')
    assert ledger.data['events']['VE_000001']['stages']['vertical_validation']['status'] == 'COMPLETE'
    assert ledger.data['events']['VE_000001']['stages']['vertical_validation']['decision'] == 'REVIEW_VERTICAL'

def test_legacy_review_status_migrates_without_losing_completed_work(tmp_path):
    raw={'schema_version':'processing_ledger_v1','movie_id':'film','events':{'VE_000001':{'visual_event_id':'VE_000001','stages':{
        'semantic':{'status':'COMPLETE'},'horizontal_export':{'status':'COMPLETE'},
        'vertical_validation':{'status':'REVIEW_VERTICAL'},'finalization':{'status':'REVIEW_VERTICAL'}}}}}
    (tmp_path/'processing_ledger.json').write_text(json.dumps(raw))
    ledger=ProcessingLedger(tmp_path,'film',{})
    stages=ledger.data['events']['VE_000001']['stages']
    assert stages['semantic']['status'] == stages['horizontal_export']['status'] == 'COMPLETE'
    assert stages['vertical_validation']['status'] == stages['finalization']['status'] == 'COMPLETE'
    assert stages['vertical_validation']['decision'] == stages['finalization']['decision'] == 'REVIEW_VERTICAL'
