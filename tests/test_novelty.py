import pandas as pd
import pytest

from finpulse.novelty import jaccard, mark_novelty, normalise_title, tokens


def frame(rows):
    cols = ["ticker", "title", "published_at", "fetched_at"]
    return pd.DataFrame(rows, columns=cols)


def test_normalise_strips_publisher_suffix_punctuation_and_case():
    assert normalise_title("Wipro Q2 Profit Slips, Revenue Up 9% - Reuters") == \
        "wipro q2 profit slips revenue up 9"
    assert normalise_title("Infosys wins deal | Mint") == "infosys wins deal"


def test_normalise_keeps_long_trailing_segment():
    # A long tail after " - " is part of the headline, not a publisher name.
    t = "HDFC Bank shares fall - analysts flag slower deposit growth this quarter"
    assert "deposit growth" in normalise_title(t)


def test_normalise_handles_unicode_and_missing():
    assert normalise_title("ＴＣＳ “hits” record") == "tcs hits record"
    assert normalise_title(None) == ""


def test_tokens_drop_stopwords():
    assert tokens("The rupee falls to a record low") == {"rupee", "falls", "record", "low"}


def test_jaccard():
    a, b = frozenset({"x", "y"}), frozenset({"y", "z"})
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert jaccard(a, a) == 1.0
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_repeat_within_window_same_ticker():
    df = mark_novelty(frame([
        ["TCS.NS", "TCS Q2 profit rises 5% on strong deal wins", "2026-10-10T04:00:00Z", None],
        ["TCS.NS", "TCS Q2 profit rises 5%, strong deal wins - Reuters", "2026-10-11T04:00:00Z", None],
    ]))
    assert df["novel"].tolist() == [True, False]


def test_not_repeat_outside_window_or_other_ticker_or_dissimilar():
    df = mark_novelty(frame([
        ["TCS.NS", "TCS Q2 profit rises 5% on strong deal wins", "2026-10-01T04:00:00Z", None],
        ["TCS.NS", "TCS Q2 profit rises 5% on strong deal wins", "2026-10-06T04:00:00Z", None],
        ["INFY.NS", "TCS Q2 profit rises 5% on strong deal wins", "2026-10-06T05:00:00Z", None],
        ["TCS.NS", "TCS names new chief financial officer", "2026-10-07T04:00:00Z", None],
    ]))
    assert df["novel"].tolist() == [True, True, True, True]


def test_order_is_by_time_not_row_order_and_fetched_at_fallback():
    df = mark_novelty(frame([
        ["SBIN.NS", "SBI raises lending rates by 10 bps", "2026-10-12T04:00:00Z", None],
        ["SBIN.NS", "SBI raises lending rates by 10 bps", None, "2026-10-11T04:00:00Z"],
    ]))
    # The second row is earlier (via fetched_at), so it is the novel one.
    assert df["novel"].tolist() == [False, True]


def test_row_without_any_timestamp_is_novel():
    df = mark_novelty(frame([
        ["SBIN.NS", "SBI raises lending rates", "2026-10-12T04:00:00Z", None],
        ["SBIN.NS", "SBI raises lending rates", None, None],
    ]))
    assert df["novel"].tolist() == [True, True]


def test_multi_ticker_flag_uses_ist_day():
    title = "Stocks to watch: TCS, Infosys, Wipro"
    df = mark_novelty(frame([
        ["TCS.NS", title, "2026-10-12T20:00:00Z", None],   # 01:30 IST on 13 Oct
        ["INFY.NS", title, "2026-10-13T03:00:00Z", None],  # 08:30 IST on 13 Oct
        ["WIPRO.NS", title, "2026-10-13T04:00:00Z", None],
        ["HCLTECH.NS", title, "2026-10-12T03:00:00Z", None],  # a different IST day
    ]))
    assert df["multi_ticker"].tolist() == [True, True, True, False]


def test_input_not_modified():
    src = frame([["TCS.NS", "x", "2026-10-12T04:00:00Z", None]])
    mark_novelty(src)
    assert list(src.columns) == ["ticker", "title", "published_at", "fetched_at"]
