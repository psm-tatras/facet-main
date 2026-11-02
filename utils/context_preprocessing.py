"""Citation-context text preprocessing shared with the context-classifier's
training data build -- must stay identical to what the adapter was trained on."""

import re

MARKER_RE = re.compile(r"\{\{(?:cite|figure|table):[^}]*\}\}")
WS_RE = re.compile(r"\s+")


def collapse_ws(text):
    """Collapse any whitespace run (incl. the "\\n  " arXiv titles/sentences
    contain) into a single space."""
    return WS_RE.sub(" ", text or "").strip()


def clean_section(section):
    """Strip leftover {{cite:...}}/{{figure:...}}/{{table:...}} markers and
    whitespace from a section heading; "" becomes "(none)"."""
    s = MARKER_RE.sub("", section or "")
    s = collapse_ws(s)
    return s if s else "(none)"


def clean_narrative(narrative):
    """Replace each empty/missing narrative field with "(not stated)"."""
    narrative = narrative or {}
    return {k: (narrative.get(k) or "(not stated)") for k in ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT")}


def cap_side(words, side, max_words):
    """Keep the LAST max_words of `words` for side="before", or the FIRST
    max_words for side="after"; mark truncation with a leading/trailing "…"."""
    if len(words) <= max_words:
        return " ".join(words)
    if side == "before":
        return "… " + " ".join(words[-max_words:])
    return " ".join(words[:max_words]) + " …"


def build_context_strings(before_sentences, after_sentences, max_side_words):
    """Join before/after sentence lists with " ", then cap each side at
    max_side_words. Returns (before_text, after_text, was_truncated)."""
    before_words = " ".join(before_sentences or []).split()
    after_words = " ".join(after_sentences or []).split()
    truncated = len(before_words) > max_side_words or len(after_words) > max_side_words
    return (cap_side(before_words, "before", max_side_words),
            cap_side(after_words, "after", max_side_words),
            truncated)
