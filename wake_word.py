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
    "\u064a\u0627 \u0643\u0644\u0627\u0648\u062f",
    "\u0647\u0627\u064a \u0643\u0644\u0627\u0648\u062f",
    "\u0645\u0631\u062d\u0628\u0627 \u0643\u0644\u0627\u0648\u062f",
    "\u0627\u0648\u0643\u064a \u0643\u0644\u0627\u0648\u062f",
    "\u0643\u0644\u0627\u0648\u062f",
]

_PATTERN = re.compile(
    r"^[\s,.\-!?\u060C\u061B\u061F]*"
    r"(?P<wake>"
    + "|".join(re.escape(v) for v in WAKE_VARIANTS)
    + r")"
    r"[\s,.\-!?\u060C\u061B\u061F]*"
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
    text = transcription.strip()
    if not text:
        return WakeResult(detected=False, variant="", query="", raw_text=text)

    match = _PATTERN.match(text)
    if match:
        variant = match.group("wake").strip()
        query = match.group("query").strip()
        query = query.lstrip(",.\u060C!?\u061F").strip()

        if query:
            return WakeResult(
                detected=True,
                variant=variant.title(),
                query=query,
                raw_text=text,
            )

    return WakeResult(detected=False, variant="", query=text, raw_text=text)


def get_variants_display() -> list[str]:
    return [v.title() for v in WAKE_VARIANTS]
