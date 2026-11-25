"""Rule-based event-type tagging, as defined in docs/v3-preregistration.md.

Each headline gets one primary type. Rules are checked in ``TYPE_ORDER`` and
the first match wins; ``multi_ticker`` headlines are ``macro`` unless an
earlier rule matches. Matching is case-insensitive on whole words.
"""
from __future__ import annotations

import re

TYPE_ORDER = ("earnings", "regulatory", "rumour", "management", "macro", "other")

_KEYWORDS = {
    "earnings": [
        r"results?", r"earnings?", r"quarterly", r"q[1-4]", r"profits?", r"net income",
        r"revenues?", r"eps", r"ebitda", r"margins?", r"guidance",
        r"beats?", r"beat estimates", r"miss(?:es|ed)?",
    ],
    "regulatory": [
        r"sebi", r"rbi", r"cci", r"nclt", r"courts?", r"tribunal", r"regulators?",
        r"regulatory", r"probes?", r"investigations?", r"penalty", r"penalties",
        r"fined?", r"fines", r"bans?", r"banned", r"licen[cs]es?", r"lawsuits?",
        r"tax demand", r"gst notice", r"enforcement directorate",
    ],
    "rumour": [
        r"reportedly", r"sources", r"people familiar", r"said to", r"in talks",
        r"mulls?", r"weighs?", r"considering", r"rumou?rs?", r"speculation",
    ],
    "management": [
        r"ceo", r"cfo", r"md", r"chairman", r"chairperson", r"chairwoman", r"board",
        r"directors?", r"resigns?", r"resignation", r"resigned", r"appoints?",
        r"appointed", r"appointment", r"steps down", r"stepping down", r"succession",
        r"promoters?", r"founders?",
    ],
    "macro": [
        r"nifty", r"sensex", r"rupee", r"inflation", r"gdp", r"repo rate",
        r"interest rates?", r"fed", r"tariffs?", r"crude", r"oil prices?",
        r"monetary policy", r"budget", r"economy", r"stocks to watch",
        r"top gainers", r"top losers",
    ],
}

PATTERNS = {
    kind: re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)
    for kind, words in _KEYWORDS.items()
}


def tag_headline(title: str, multi_ticker: bool = False) -> str:
    """Primary event type of one headline."""
    text = str(title or "")
    for kind in TYPE_ORDER[:-1]:
        if PATTERNS[kind].search(text):
            return kind
        if kind == "macro" and multi_ticker:
            return "macro"
    return "other"


def event_type(types) -> str:
    """Type of an event made of several headlines: the earliest in TYPE_ORDER."""
    present = set(types)
    for kind in TYPE_ORDER:
        if kind in present:
            return kind
    return "other"
