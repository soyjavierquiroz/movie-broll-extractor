import json

from movie_broll.cli import main
from movie_broll.pilot_selector import mark_attempted, profile_for_range, registry_path, select_next


def segment(identifier, start, end, function, tone, kind="conversation", opportunities=None, summary=""):
    assertion = lambda value: {"value": value, "source": "srt_llm", "confidence": .5}
    return {"segment_id": identifier, "start_seconds": start, "end_seconds": end,
            "segment_type": assertion(kind), "narrative_function": assertion(function),
            "narrative_tone": assertion(tone), "narrative_summary": assertion(summary),
            "possible_visual_opportunities": [{"value": x, "source": "srt_llm_hint", "confidence": .5} for x in (opportunities or [])]}


def fixture(tmp_path):
    input_dir = tmp_path / "input" / "film"; input_dir.mkdir(parents=True)
    run = tmp_path / "runs" / "film"; (run / "narrative-v2").mkdir(parents=True); (run / "source-inspect-v1").mkdir()
    data = {"segments": [
        segment("N1", 100, 180, "conversation", "mixed", opportunities=["conversation"]),
        segment("N2", 195, 275, "conversation", "mixed", opportunities=["conversation"]),
        segment("N3", 900, 990, "conflict", "tense", "monologue", ["reaction", "gesture"]),
        segment("N4", 1600, 1680, "transition", "sad", "sparse_dialogue", ["movement"]),
    ]}
    (run / "narrative-v2" / "narrative_map.json").write_text(json.dumps(data))
    (run / "source-inspect-v1" / "source_manifest.json").write_text(json.dumps({"source": {"movie": {"duration_seconds": 1800}}}))
    return input_dir, data


def test_diverse_selection_is_idempotent_and_preserves_existing_window(tmp_path):
    input_dir, data = fixture(tmp_path)
    registry_path(input_dir).parent.mkdir(parents=True, exist_ok=True)
    registry_path(input_dir).write_text(json.dumps({"schema_version": "pilot_windows_v1", "movie_id": "film", "windows": [{"window_id": "SW_01", "start_seconds": 100, "end_seconds": 160, "profile": profile_for_range(data, 100, 160), "status": "TESTED"}]}))
    first = select_next(input_dir)
    second = select_next(input_dir)
    assert first["window_id"] == "SW_02" == second["window_id"]
    assert first["narrative_segment_ids"] == ["N3"]  # nearby identical N2 loses to conflict.
    assert "different_interaction_context" in first["selection_reason"]
    assert "different_tone" in first["selection_reason"]
    assert "different_movie_region" in first["selection_reason"]
    stored = json.loads(registry_path(input_dir).read_text())["windows"]
    assert [x["window_id"] for x in stored] == ["SW_01", "SW_02"]


def test_partial_quota_marks_same_window_tested_then_allows_next(tmp_path):
    input_dir, data = fixture(tmp_path)
    registry_path(input_dir).parent.mkdir(parents=True, exist_ok=True)
    registry_path(input_dir).write_text(json.dumps({"schema_version": "pilot_windows_v1", "movie_id": "film", "windows": [{"window_id": "SW_01", "start_seconds": 100, "end_seconds": 160, "profile": profile_for_range(data, 100, 160), "status": "TESTED"}]}))
    selected = select_next(input_dir)
    assert select_next(input_dir)["window_id"] == selected["window_id"]
    mark_attempted(input_dir, selected["window_id"], "PARTIAL_QUOTA")
    later = select_next(input_dir)
    assert later["window_id"] == "SW_03"
    rows = json.loads(registry_path(input_dir).read_text())["windows"]
    old = next(x for x in rows if x["window_id"] == selected["window_id"])
    assert old["status"] == "TESTED" and old["pilot_status"] == "PARTIAL_QUOTA"


def test_relationships_need_explicit_narrative_support(tmp_path):
    _, data = fixture(tmp_path)
    data["segments"][0]["narrative_summary"]["value"] = "Dos mujeres hablan"
    assert "relationship_hints" not in profile_for_range(data, 100, 160)
    data["segments"][0]["narrative_summary"]["value"] = "Una madre e hija discuten"
    assert profile_for_range(data, 100, 160)["relationship_hints"] == ["mother_daughter"]


def test_cli_select_next_prints_manager_output(tmp_path, capsys):
    input_dir, _ = fixture(tmp_path)
    assert main(["pilot", "select-next", str(input_dir)]) == 0
    output = capsys.readouterr().out
    assert "[pilot-selector] selected: SW_01" in output and "status: COMPLETE" in output
