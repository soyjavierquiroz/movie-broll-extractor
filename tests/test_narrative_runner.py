import json
from pathlib import Path

import pytest

from movie_broll.narrative_provider import ProviderResponse
from movie_broll.narrative_runner import run_narrative
from movie_broll.srt import Cue


def _map(data):
    first, last = data["cues"][0], data["cues"][-1]
    assertion = lambda value: {"value": value, "source": "srt_llm", "confidence": .5}
    suffix = data["chunk"]["chunk_id"].removeprefix("NCHUNK_")
    return {"schema_version": "narrative_map_chunk_v1", "movie_id": data["movie_id"], "chunk": {key: data["chunk"][key] for key in ("chunk_id", "start_seconds", "end_seconds")}, "source": {"type": "external_srt", "literal_transcription": False}, "chunk_summary": assertion("Resumen."), "segments": [{"segment_id": f"NARR_{suffix}_001", "start_seconds": first["start_seconds"], "end_seconds": last["end_seconds"], "cue_ids": [cue["cue_id"] for cue in data["cues"]], "segment_type": assertion("conversation"), "narrative_summary": assertion("Resumen."), "dialogue_density": assertion("medium"), "narrative_tone": assertion("neutral"), "narrative_function": assertion("conversation"), "continuity": {"previous": "unknown", "next": "outside_chunk"}, "possible_visual_opportunities": [], "context_dependency": assertion("medium"), "boundary": {"start_confidence": .5, "end_confidence": .5}}]}


class FakeProvider:
    identifier = "gemini"
    model = "gemini-2.5-flash"
    def __init__(self, failures=0, invalid=False): self.calls = 0; self.failures = failures; self.invalid = invalid
    def generate(self, prompt, chunk_input):
        self.calls += 1
        if self.calls <= self.failures: raise TimeoutError("temporary")
        value = _map(chunk_input)
        if self.invalid: value["movie_id"] = "wrong"
        return ProviderResponse(value, {"prompt_tokens": 3, "response_tokens": 4, "thinking_tokens": 0, "cached_tokens": 0, "total_tokens": 7})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "cues.jsonl"
    cues = [Cue("SRT_000001", 1, 1, 2, "uno"), Cue("SRT_000002", 2, 601, 602, "dos")]
    source.write_text("".join(json.dumps(cue.as_dict()) + "\n" for cue in cues), encoding="utf-8")
    movie_dir = tmp_path / "input" / "pilot"; movie_dir.mkdir(parents=True)
    (movie_dir / "movie.mp4").touch(); (movie_dir / "subtitles.srt").write_text("x", encoding="utf-8")
    monkeypatch.setattr("movie_broll.narrative_runner._ensure_source", lambda *args: source)
    return movie_dir


def test_missing_key_fails_before_requests(prepared, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("movie_broll.narrative_runner.load_dotenv", lambda *_: False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not configured"):
        run_narrative(prepared)


def test_success_reuse_and_usage(prepared):
    provider = FakeProvider()
    first = run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    second = run_narrative(prepared, provider=provider, max_chunks=1, output=lambda _: None)
    assert (first["requests"], first["usage"]["total_tokens"], provider.calls) == (1, 7, 1)
    assert second["reused_chunks"] == 1 and second["requests"] == 0


def test_invalid_response_retries_and_preserves_resumability(prepared):
    provider = FakeProvider(invalid=True)
    manifest = run_narrative(prepared, provider=provider, max_chunks=1, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "PARTIAL" and manifest["requests"] == 2 and manifest["retries"] == 1
    assert not list((Path("runs") / "pilot" / "narrative-v1" / "maps").glob("*.narrative_map.json"))


def test_transient_retry_and_checkpoint_invalidation(prepared):
    transient = FakeProvider(failures=1)
    manifest = run_narrative(prepared, provider=transient, max_chunks=1, sleep=lambda _: None, output=lambda _: None)
    assert manifest["status"] == "COMPLETE" and manifest["requests"] == 2 and manifest["retries"] == 1
    changed_model = FakeProvider()
    run_narrative(prepared, provider=changed_model, model="other", max_chunks=1, output=lambda _: None)
    assert changed_model.calls == 1
    checkpoint = Path("runs/pilot/narrative-v1/maps/NCHUNK_0001.checkpoint.json")
    record = json.loads(checkpoint.read_text()); record["prompt_sha256"] = "changed"; checkpoint.write_text(json.dumps(record))
    regenerated = FakeProvider(); run_narrative(prepared, provider=regenerated, model="other", max_chunks=1, output=lambda _: None)
    assert regenerated.calls == 1
