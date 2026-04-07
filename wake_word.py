"""
Wake word detection for Cloud — the Weather Agent.

Supports multiple activation variants. After Groq Whisper transcribes the
audio, this module checks whether the user said a wake phrase and extracts
the actual query that follows it.

Variants (case-insensitive, punctuation-tolerant):
    "Hey Cloud, ..."
    "OK Cloud, ..."
    "Okay Cloud, ..."
    "Hi Cloud, ..."
    "Yo Cloud, ..."
    "Cloud, ..."
    "Hello Cloud, ..."

Usage:
    from wake_word import detect_wake_word

    result = detect_wake_word("Hey Cloud, what's the weather in Berlin?")
    # WakeResult(detected=True, variant="Hey Cloud", query="what's the weather in Berlin?")
"""

import re
from dataclasses import dataclass

WAKE_VARIANTS = [
    "hey cloud",
    "ok cloud",
    "okay cloud",
    "hi cloud",
    "yo cloud",
    "hello cloud",
    "cloud",
]

_PATTERN = re.compile(
    r"^[\s,.\-!?]*"
    r"(?P<wake>"
    + "|".join(re.escape(v) for v in WAKE_VARIANTS)
    + r")"
    r"[\s,.\-!?]*"
    r"(?P<query>.*)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class WakeResult:
    detected: bool
    variant: str
    query: str
    raw_text: str


def detect_wake_word(transcription: str) -> WakeResult:
    """Check if the transcription starts with a Cloud wake phrase.

    Returns a WakeResult with the detected variant and the extracted query.
    The matching is case-insensitive and tolerates punctuation between the
    wake word and the query.
    """
    text = transcription.strip()
    if not text:
        return WakeResult(detected=False, variant="", query="", raw_text=text)

    match = _PATTERN.match(text)
    if match:
        variant = match.group("wake").strip()
        query = match.group("query").strip()
        query = query.lstrip(",").lstrip(".").lstrip("!").lstrip("?").strip()

        if query:
            return WakeResult(
                detected=True,
                variant=variant.title(),
                query=query,
                raw_text=text,
            )

    return WakeResult(detected=False, variant="", query=text, raw_text=text)


def get_variants_display() -> list[str]:
    """Return the wake variants formatted for display."""
    return [v.title() for v in WAKE_VARIANTS]
