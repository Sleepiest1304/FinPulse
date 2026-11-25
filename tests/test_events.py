import pytest

from finpulse.events import TYPE_ORDER, event_type, tag_headline


@pytest.mark.parametrize("title, expected", [
    ("Wipro Q2 profit slips slightly YoY to Rs 3,201 crore", "earnings"),
    ("Infosys raises FY27 revenue guidance", "earnings"),
    ("HDFC Bank misses estimates", "earnings"),
    ("SEBI issues show-cause notice to Adani Enterprises", "regulatory"),
    ("RBI fines Kotak Mahindra Bank for lapses", "regulatory"),
    ("NCLT admits insolvency plea", "regulatory"),
    ("Reliance reportedly in talks to buy a stake in a solar firm", "rumour"),
    ("Tata Steel mulls sale of Dutch unit", "rumour"),
    ("Bajaj Finance CEO steps down", "management"),
    ("Titan appoints new chairman", "management"),
    ("Rupee hits record low as crude climbs", "macro"),
    ("Sensex, Nifty end higher", "macro"),
    ("Maruti launches a new SUV", "other"),
    ("", "other"),
    (None, "other"),
])
def test_tag_headline(title, expected):
    assert tag_headline(title) == expected


def test_priority_order():
    # Earnings wins over regulatory, regulatory over management, and so on.
    assert tag_headline("SEBI probe weighs on Q3 profit") == "earnings"
    assert tag_headline("Court blocks board appointment") == "regulatory"
    assert tag_headline("CEO said to be in talks to leave") == "rumour"
    assert tag_headline("Board meets as Nifty slides") == "management"


def test_whole_word_and_case_insensitive():
    assert tag_headline("Fedex expands in India") == "other"   # not "fed"
    assert tag_headline("Quarterly cadence") == "earnings"
    assert tag_headline("MANAGEMENT SHAKE-UP: CFO QUITS") == "management"
    assert tag_headline("Mdx launch") == "other"               # not "md"


def test_multi_ticker_falls_back_to_macro_only_when_nothing_earlier_matches():
    assert tag_headline("Five stocks in focus today", multi_ticker=True) == "macro"
    assert tag_headline("Five stocks in focus today") == "other"
    assert tag_headline("Q2 results today: TCS, Wipro", multi_ticker=True) == "earnings"


def test_event_type_takes_earliest():
    assert event_type(["macro", "management"]) == "management"
    assert event_type(["other", "earnings", "rumour"]) == "earnings"
    assert event_type([]) == "other"
    assert TYPE_ORDER[0] == "earnings" and TYPE_ORDER[-1] == "other"
