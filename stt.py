import io
import json
import logging
from dataclasses import dataclass
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


@dataclass
class TranscriptionResult:
    text: str
    language: str


def _lang_from_text(text: str) -> str | None:
\
\
\
\
\
       
    arabic = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    alpha = sum(1 for c in text if c.isalpha())
    if alpha > 0 and arabic / alpha > 0.20:
        return "ar"
    return None


def transcribe_audio(
    audio_bytes: bytes,
    language: str | None = None,
    filename: str = "recording.wav",
) -> TranscriptionResult:
    client = _get_client()

    kwargs = {
        "file": (filename, io.BytesIO(audio_bytes)),
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language

    logger.info(f"[STT] Sending {len(audio_bytes)} bytes to Groq Whisper...")
    transcription = client.audio.transcriptions.create(**kwargs)

    if hasattr(transcription, "text"):
        text = transcription.text.strip()
        detected_lang = getattr(transcription, "language", None) or ""
    elif isinstance(transcription, str):
        try:
            data = json.loads(transcription)
            text = data.get("text", "").strip()
            detected_lang = data.get("language", "") or ""
        except (json.JSONDecodeError, TypeError):
            text = transcription.strip()
            detected_lang = ""
    else:
        text = str(transcription).strip()
        detected_lang = ""

                                                                             
                                                                            
                                                                                          
    whisper_lang = "ar" if detected_lang.startswith("ar") else ("en" if detected_lang else None)
    text_lang = _lang_from_text(text)

    if text_lang == "ar":
                                                                           
        lang_code = "ar"
        if whisper_lang != "ar":
            logger.info(f"[STT] Language override: Whisper said '{detected_lang}', text analysis says Arabic")
    elif whisper_lang:
        lang_code = whisper_lang
    else:
        lang_code = "en"

    logger.info(f"[STT] Transcribed ({lang_code}): '{text[:100]}' ({len(text)} chars)")

    return TranscriptionResult(text=text, language=lang_code)
