from movie_broll.gemini_credentials import (
    GeminiCredentialSource,
    discover_gemini_credentials,
)


def _names(credentials):
    return [item.identifier for item in credentials.primaries], (
        credentials.backup.identifier if credentials.backup else None
    )


def test_indexed_keys_are_unbounded_and_numerically_sorted():
    credentials = discover_gemini_credentials({
        "GEMINI_API_KEY_10": "ten",
        "GEMINI_API_KEY_2": "two",
        "GEMINI_API_KEY_1": "one",
        "GEMINI_API_KEY_37": "thirty-seven",
    })
    assert _names(credentials) == (
        ["gemini-primary-1", "gemini-primary-2", "gemini-primary-10", "gemini-primary-37"],
        None,
    )


def test_six_key_pool_and_new_keys_are_discovered():
    env = {f"GEMINI_API_KEY_{index}": f"key-{index}" for index in range(1, 7)}
    env.update({
        "GEMINI_API_KEY_7": "key-7",
        "GEMINI_API_KEY_8": "key-8",
        "GEMINI_API_KEY_12": "key-12",
    })
    credentials = discover_gemini_credentials(env)
    assert [item.identifier for item in credentials.primaries] == [
        *(f"gemini-primary-{index}" for index in range(1, 9)),
        "gemini-primary-12",
    ]


def test_malformed_empty_and_duplicate_values_are_ignored():
    credentials = discover_gemini_credentials({
        "GEMINI_API_KEY_0": "zero",
        "GEMINI_API_KEY_X": "x",
        "GEMINI_API_KEY_TEST": "test",
        "GEMINI_API_KEY_1": " shared ",
        "GEMINI_API_KEY_2": "shared",
        "GEMINI_API_KEY_3": "  ",
        "GEMINI_API_KEY_BACKUP": "shared",
        "GEMINI_API_KEY": "legacy",
    })
    assert _names(credentials) == (["gemini-primary-1"], None)


def test_backup_and_legacy_compatibility():
    backup = discover_gemini_credentials({"GEMINI_API_KEY_BACKUP": "backup"})
    legacy = discover_gemini_credentials({"GEMINI_API_KEY": "legacy"})
    assert _names(backup) == ([], "gemini-backup")
    assert _names(legacy) == (["gemini-primary-1"], None)
    assert legacy.legacy_only is True


def test_env_file_changes_are_seen_without_process_restart(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY_1=one\n", encoding="utf-8")
    source = GeminiCredentialSource(env_file, {})
    assert _names(source.discover()) == (["gemini-primary-1"], None)
    env_file.write_text("GEMINI_API_KEY_1=one\nGEMINI_API_KEY_7=seven\n", encoding="utf-8")
    assert _names(source.discover()) == (
        ["gemini-primary-1", "gemini-primary-7"], None
    )


def test_narrative_and_semantic_import_the_same_discovery_function():
    from movie_broll import broll_semantics, narrative_runner

    assert broll_semantics.GeminiCredentialSource is narrative_runner.GeminiCredentialSource
