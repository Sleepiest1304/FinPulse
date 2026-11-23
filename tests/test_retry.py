import pytest

from engine.retry import retry


def test_retry_backoff_delays_grow():
    delays = []
    calls = {"n": 0}

    @retry(attempts=4, base_delay=1.0, sleep=delays.append)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise OSError("boom")
        return "ok"

    assert flaky() == "ok"
    assert len(delays) == 3
    for d, base in zip(delays, [1, 2, 4]):
        assert base <= d <= base * 1.1


def test_retry_only_catches_listed_exceptions():
    @retry(attempts=3, exceptions=(OSError,), sleep=lambda s: None)
    def bad():
        raise KeyError("not retried")

    with pytest.raises(KeyError):
        bad()


def test_retry_validates_attempts():
    with pytest.raises(ValueError):
        retry(attempts=0)
