from movie_broll.broll_semantics import (
    GeminiProviderPool,
    GeminiProviderPoolError,
    SemanticResponse,
)


class FakeProvider:
    def __init__(self, identifier, outcomes):
        self.identifier = identifier
        self.model = "gemini-3.6-flash"
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate(self, *args):
        self.calls += 1
        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return SemanticResponse(
            outcome,
            {
                "prompt_tokens": 1,
                "response_tokens": 1,
                "thinking_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 2,
            },
        )


def ok():
    return {"ok": True}


def quota_error(seconds=17.25):
    return RuntimeError(
        "Error code: 429 "
        "generate_content_free_tier_requests quota exceeded. "
        f"Please retry in: {seconds}s"
    )


def test_round_robin_primaries():
    providers = [
        FakeProvider("gemini-primary-1", [ok(), ok()]),
        FakeProvider("gemini-primary-2", [ok()]),
        FakeProvider("gemini-primary-3", [ok()]),
    ]

    pool = GeminiProviderPool(providers)

    resolved = [
        pool.generate("p", {"candidate_id": str(i)}, b"x").provider
        for i in range(4)
    ]

    assert resolved == [
        "gemini-primary-1",
        "gemini-primary-2",
        "gemini-primary-3",
        "gemini-primary-1",
    ]


def test_429_retries_same_event_on_next_primary():
    now = [100.0]

    p1 = FakeProvider("gemini-primary-1", [quota_error()])
    p2 = FakeProvider("gemini-primary-2", [ok()])
    p3 = FakeProvider("gemini-primary-3", [ok()])

    pool = GeminiProviderPool(
        [p1, p2, p3],
        clock=lambda: now[0],
    )

    response = pool.generate(
        "p",
        {"candidate_id": "BRC_0009"},
        b"x",
    )

    assert response.provider == "gemini-primary-2"
    assert response.attempts == 2
    assert p1.calls == 1
    assert p2.calls == 1
    assert [x["provider"] for x in response.provider_trace] == [
        "gemini-primary-1",
        "gemini-primary-2",
    ]


def test_backup_after_all_primaries_fail():
    primaries = [
        FakeProvider(
            f"gemini-primary-{i}",
            [RuntimeError("Error code: 503 service unavailable")],
        )
        for i in (1, 2, 3)
    ]

    backup = FakeProvider("gemini-backup", [ok()])

    pool = GeminiProviderPool(primaries, backup)

    response = pool.generate(
        "p",
        {"candidate_id": "BRC_0010"},
        b"x",
    )

    assert response.provider == "gemini-backup"
    assert response.attempts == 4


def test_cooldown_skips_provider_until_expired():
    now = [100.0]

    p1 = FakeProvider(
        "gemini-primary-1",
        [quota_error(10), ok()],
    )
    p2 = FakeProvider(
        "gemini-primary-2",
        [ok(), ok()],
    )

    pool = GeminiProviderPool(
        [p1, p2],
        clock=lambda: now[0],
    )

    first = pool.generate("p", {"candidate_id": "A"}, b"x")
    assert first.provider == "gemini-primary-2"
    assert p1.calls == 1

    # Round-robin would come back to p1, but it is still cooling down.
    second = pool.generate("p", {"candidate_id": "B"}, b"x")
    assert second.provider == "gemini-primary-2"
    assert p1.calls == 1

    now[0] = 111.0

    third = pool.generate("p", {"candidate_id": "C"}, b"x")
    assert third.provider == "gemini-primary-1"
    assert p1.calls == 2


def test_all_quota_failures_raise_precise_pool_error():
    primaries = [
        FakeProvider(f"gemini-primary-{i}", [quota_error()])
        for i in (1, 2, 3)
    ]

    backup = FakeProvider("gemini-backup", [quota_error()])

    pool = GeminiProviderPool(primaries, backup)

    try:
        pool.generate(
            "p",
            {"candidate_id": "BRC_0011"},
            b"x",
        )
    except GeminiProviderPoolError as error:
        assert error.quota_exhausted is True
        assert error.reason == "quota_exceeded"
        assert error.http_status == 429
        assert error.retryable is True
        assert error.retry_after_seconds == 17.25
        assert [x["provider"] for x in error.failures] == [
            "gemini-primary-1",
            "gemini-primary-2",
            "gemini-primary-3",
            "gemini-backup",
        ]
    else:
        raise AssertionError("expected GeminiProviderPoolError")



def test_cooldown_skip_counts_zero_actual_requests():
    from movie_broll.broll_semantics import (
        GeminiProviderPool,
        GeminiProviderPoolError,
    )

    now = [100.0]
    lines = []

    class QuotaProvider:
        model = "gemini-3.6-flash"

        def __init__(self, identifier):
            self.identifier = identifier
            self.calls = 0

        def generate(self, *args):
            self.calls += 1
            raise RuntimeError(
                "Error code: 429 "
                "generate_content_free_tier_requests "
                "quota exceeded. Please retry in: 20s"
            )

    p1 = QuotaProvider("gemini-primary-1")
    p2 = QuotaProvider("gemini-primary-2")

    pool = GeminiProviderPool(
        [p1, p2],
        clock=lambda: now[0],
        reporter=lines.append,
    )

    # Primera llamada: ambos providers son realmente consultados.
    try:
        pool.generate(
            "prompt",
            {
                "candidate_id": "BRC_0012",
                "visual_event_id": "VE_REAL_0012",
            },
            b"jpeg",
        )
    except GeminiProviderPoolError as error:
        assert error.attempts == 2
        assert error.providers_attempted == [
            "gemini-primary-1",
            "gemini-primary-2",
        ]
    else:
        raise AssertionError("expected provider pool exhaustion")

    assert p1.calls == 1
    assert p2.calls == 1

    # Segunda llamada inmediata: ambos siguen en cooldown.
    # Debe realizar CERO requests reales.
    try:
        pool.generate(
            "prompt",
            {
                "candidate_id": "BRC_0012",
                "visual_event_id": "VE_REAL_0012",
            },
            b"jpeg",
        )
    except GeminiProviderPoolError as error:
        assert error.attempts == 0
        assert error.providers_attempted == []
        assert error.quota_exhausted is True
    else:
        raise AssertionError("expected cooldown-only pool exhaustion")

    assert p1.calls == 1
    assert p2.calls == 1

    # Observabilidad usa visual_event_id, no solo candidate_id.
    assert any(
        "event=VE_REAL_0012" in line
        for line in lines
    )



def test_auth_error_disables_member_and_fails_over_same_request():
    from movie_broll.broll_semantics import SemanticResponse

    class Provider:
        model = "gemini-3.6-flash"

        def __init__(self, identifier, auth_fail=False):
            self.identifier = identifier
            self.auth_fail = auth_fail
            self.calls = 0

        def generate(self, *args):
            self.calls += 1

            if self.auth_fail:
                raise RuntimeError("Error code: 401 invalid API key")

            return SemanticResponse(
                {"ok": True},
                {},
            )

    bad = Provider("gemini-primary-1", auth_fail=True)
    good = Provider("gemini-primary-2")

    lines = []

    pool = GeminiProviderPool(
        [bad, good],
        reporter=lines.append,
    )

    response = pool.generate(
        "prompt",
        {
            "candidate_id": "BRC_AUTH",
            "visual_event_id": "VE_AUTH",
        },
        b"jpg",
    )

    assert response.provider == "gemini-primary-2"
    assert response.attempts == 2
    assert bad.calls == 1
    assert good.calls == 1

    assert any(
        "provider=gemini-primary-1" in line
        and "reason=auth_error" in line
        and "action=DISABLE" in line
        for line in lines
    )

    # Next logical request must skip bad credential completely.
    response2 = pool.generate(
        "prompt",
        {
            "candidate_id": "BRC_NEXT",
            "visual_event_id": "VE_NEXT",
        },
        b"jpg",
    )

    assert response2.provider == "gemini-primary-2"
    assert bad.calls == 1
    assert good.calls == 2


def test_all_auth_failures_are_resume_safe_pool_failure():
    class BadProvider:
        model = "gemini-3.6-flash"

        def __init__(self, identifier):
            self.identifier = identifier

        def generate(self, *args):
            raise RuntimeError("Error code: 401 invalid API key")

    pool = GeminiProviderPool(
        [
            BadProvider("gemini-primary-1"),
            BadProvider("gemini-primary-2"),
        ]
    )

    try:
        pool.generate(
            "prompt",
            {"visual_event_id": "VE_AUTH_ALL"},
            b"jpg",
        )
    except GeminiProviderPoolError as error:
        assert error.reason == "auth_error"
        assert error.retryable is True
        assert error.quota_exhausted is False
        assert error.attempts == 2
    else:
        raise AssertionError("expected GeminiProviderPoolError")



def test_env_builder_uses_six_primaries_before_backup(monkeypatch):
    import movie_broll.broll_semantics as semantics

    calls=[]

    class EnvProvider:
        def __init__(
            self,
            api_key,
            model="gemini-3.6-flash",
            identifier="gemini",
        ):
            self.api_key=api_key
            self.model=model
            self.identifier=identifier

        def generate(self,*args):
            calls.append(self.identifier)

            if self.identifier!="gemini-backup":
                raise RuntimeError(
                    "Error code: 503 service unavailable"
                )

            return SemanticResponse(
                {"ok":True},
                {
                    "prompt_tokens":1,
                    "response_tokens":1,
                    "thinking_tokens":0,
                    "cached_tokens":0,
                    "total_tokens":2,
                },
            )

    monkeypatch.setattr(
        semantics,
        "GeminiBrollSemanticProvider",
        EnvProvider,
    )

    env={
        f"GEMINI_API_KEY_{i}":f"fake-key-{i}"
        for i in range(1,7)
    }
    env["GEMINI_API_KEY_BACKUP"]="fake-backup-key"

    pool=semantics.build_gemini_provider_from_env(
        environ=env,
    )

    response=pool.generate(
        "prompt",
        {"candidate_id":"BRC_TEST"},
        b"image",
    )

    assert response.provider=="gemini-backup"
    assert response.attempts==7
    assert calls == [
        "gemini-primary-1",
        "gemini-primary-2",
        "gemini-primary-3",
        "gemini-primary-4",
        "gemini-primary-5",
        "gemini-primary-6",
        "gemini-backup",
    ]
