import io
import logging
from groq import Groq
from config.settings import GROQ_API_KEY

logger = logging.getLogger("weather_agent")

_client = None

_MODELS = {
    "en": "canopylabs/orpheus-v1-english",
    "ar": "canopylabs/orpheus-arabic-saudi",
}

_DEFAULT_VOICES = {
    "en": "troy",
    "ar": "fahad",
}

_AVAILABLE_VOICES = {
    "en": ["autumn", "diana", "hannah", "austin", "daniel", "troy"],
    "ar": ["abdullah", "fahad", "sultan", "lulwa", "noura", "aisha"],
}

MAX_CHUNK_CHARS = 195


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


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    separators = [". ", "، ", "! ", "? ", "; ", ", ", " "]

    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break

        split_at = -1
        for sep in separators:
            idx = text.rfind(sep, 0, max_chars)
            if idx > 0:
                split_at = idx + len(sep)
                break

        if split_at <= 0:
            split_at = max_chars

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    return [c for c in chunks if c]


def synthesize_speech(
    text: str,
    language: str = "en",
    voice: str | None = None,
) -> bytes:
    client = _get_client()

    lang = "ar" if language.startswith("ar") else "en"
    model = _MODELS[lang]
    voice = voice or _DEFAULT_VOICES[lang]

    if voice not in _AVAILABLE_VOICES[lang]:
        voice = _DEFAULT_VOICES[lang]

    chunks = _chunk_text(text)
    logger.info(f"[TTS] Generating speech: lang={lang}, voice={voice}, chunks={len(chunks)}, chars={len(text)}")

    audio_parts = []
    for i, chunk in enumerate(chunks):
        logger.debug(f"[TTS] Chunk {i+1}/{len(chunks)}: '{chunk[:50]}...' ({len(chunk)} chars)")
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=chunk,
            response_format="wav",
        )

        audio_parts.append(response.read())

    if len(audio_parts) == 1:
        return audio_parts[0]

    return _concat_wav(audio_parts)


def _concat_wav(wav_parts: list[bytes]) -> bytes:
    import struct

    all_pcm = []
    sample_rate = 24000
    num_channels = 1
    bits_per_sample = 16

    for wav_data in wav_parts:
        if len(wav_data) < 44:
            continue

        data_offset = 44
        header = wav_data[:44]

        if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            sr = struct.unpack_from("<I", header, 24)[0]
            nc = struct.unpack_from("<H", header, 22)[0]
            bps = struct.unpack_from("<H", header, 34)[0]
            if sr:
                sample_rate = sr
            if nc:
                num_channels = nc
            if bps:
                bits_per_sample = bps

            pos = 12
            while pos < len(wav_data) - 8:
                chunk_id = wav_data[pos:pos+4]
                chunk_size = struct.unpack_from("<I", wav_data, pos+4)[0]
                if chunk_id == b"data":
                    data_offset = pos + 8
                    break
                pos += 8 + chunk_size

        all_pcm.append(wav_data[data_offset:])

    pcm = b"".join(all_pcm)
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)

    out = io.BytesIO()
    out.write(b"RIFF")
    out.write(struct.pack("<I", 36 + len(pcm)))
    out.write(b"WAVE")
    out.write(b"fmt ")
    out.write(struct.pack("<I", 16))
    out.write(struct.pack("<H", 1))
    out.write(struct.pack("<H", num_channels))
    out.write(struct.pack("<I", sample_rate))
    out.write(struct.pack("<I", byte_rate))
    out.write(struct.pack("<H", block_align))
    out.write(struct.pack("<H", bits_per_sample))
    out.write(b"data")
    out.write(struct.pack("<I", len(pcm)))
    out.write(pcm)

    return out.getvalue()


def get_available_voices(language: str = "en") -> list[str]:
    lang = "ar" if language.startswith("ar") else "en"
    return _AVAILABLE_VOICES[lang]
