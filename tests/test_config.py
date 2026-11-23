import pytest

from finpulse.config import validate_ticker


@pytest.mark.parametrize("raw,expected", [
    ("reliance", "RELIANCE.NS"), (" TCS.NS ", "TCS.NS"), ("m&m", "M&M.NS")])
def test_validate_ticker_ok(raw, expected):
    assert validate_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "HDFC BANK.NS", "AAPL", None, 5, "TCS.NS; DROP TABLE"])
def test_validate_ticker_rejects(raw):
    with pytest.raises(ValueError):
        validate_ticker(raw)
