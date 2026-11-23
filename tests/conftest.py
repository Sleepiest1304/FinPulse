import sys
import types

import pytest


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retries never actually wait in tests."""
    monkeypatch.setattr("finpulse.retry.time.sleep", lambda s: None)


@pytest.fixture
def fake_yfinance(monkeypatch):
    """Install a stand-in ``yfinance`` module so no test touches the network."""
    mod = types.ModuleType("yfinance")
    state = {"news": [], "errors": 0, "calls": 0, "download": None}

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def news(self):
            state["calls"] += 1
            if state["errors"] > 0:
                state["errors"] -= 1
                raise ConnectionError("simulated outage")
            return state["news"]

    def download(*args, **kwargs):
        state["calls"] += 1
        return state["download"]

    mod.Ticker = Ticker
    mod.download = download
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    return state
