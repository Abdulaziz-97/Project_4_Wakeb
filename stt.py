"""
Speech-to-Text via Groq Whisper Large V3 Turbo.

Model: whisper-large-v3-turbo (809M params)
Speed: 216x real-time (10 min audio in ~3 seconds)
Cost:  $0.00067/minute
WER:   7.75% (multi-language, 99+ languages)

Usage:
    from stt import transcribe_audio
    text = transcribe_audio(wav_bytes)
"""

import io
import logging
from groq import Groq
from config.settings import GROQ_API_KEY

logger = logging.getLogger("weather_agent")

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com/keys"
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def transcribe_audio(
    audio_bytes: bytes,
    language: str | None = None,
    filename: str = "recording.wav",
) -> str:
    """Transcribe audio bytes to text using Groq Whisper Large V3 Turbo.

    Args:
        audio_bytes: Raw audio data (WAV format from st.audio_input).
        language: Optional ISO 639-1 language code (e.g. "en", "de").
                  None = auto-detect.
        filename: Filename hint for the API.

    Returns:
        Transcribed text string.
    """
    client = _get_client()

    kwargs = {
        "file": (filename, io.BytesIO(audio_bytes)),
        "model": "whisper-large-v3-turbo",
        "response_format": "text",
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language

    logger.info(f"[STT] Sending {len(audio_bytes)} bytes to Groq Whisper...")

    transcription = client.audio.transcriptions.create(**kwargs)

    text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
    logger.info(f"[STT] Transcribed: '{text[:100]}...' ({len(text)} chars)")

    return text
