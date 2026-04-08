import sys
import os
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import GROQ_API_KEY

PASS = "PASS"
FAIL = "FAIL"


def test_stt_whisper():
    print("\n" + "=" * 60)
    print("TEST 1: STT - Groq Whisper Large V3 Turbo")
    print("=" * 60)

    from stt import transcribe_audio
    import struct

    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    silent_pcm = b"\x00\x00" * num_samples

    import io
    buf = io.BytesIO()
    buf.write(b"RIFF")
    data_size = num_samples * 2
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<H", 2))
    buf.write(struct.pack("<H", 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(silent_pcm)
    wav_bytes = buf.getvalue()

    t0 = time.time()
    try:
        result = transcribe_audio(wav_bytes)
        elapsed = time.time() - t0
        print(f"  Text: '{result.text}'")
        print(f"  Language: {result.language}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {PASS}")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Error: {e}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {FAIL}")
        return False


def test_tts_english():
    print("\n" + "=" * 60)
    print("TEST 2: TTS - Orpheus English (canopylabs/orpheus-v1-english)")
    print("=" * 60)

    from tts import synthesize_speech, get_available_voices

    voices = get_available_voices("en")
    print(f"  Available voices: {voices}")

    test_text = "The weather in Riyadh today is sunny with a high of 38 degrees Celsius."
    print(f"  Text: '{test_text}'")

    t0 = time.time()
    try:
        audio = synthesize_speech(test_text, language="en", voice="troy")
        elapsed = time.time() - t0
        print(f"  Audio size: {len(audio)} bytes")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Has WAV header: {audio[:4] == b'RIFF'}")

        out_path = "test_output_english.wav"
        with open(out_path, "wb") as f:
            f.write(audio)
        print(f"  Saved to: {out_path}")
        print(f"  Status: {PASS}")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Error: {e}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {FAIL}")
        return False


def test_tts_arabic():
    print("\n" + "=" * 60)
    print("TEST 3: TTS - Orpheus Arabic Saudi (canopylabs/orpheus-arabic-saudi)")
    print("=" * 60)

    from tts import synthesize_speech, get_available_voices

    voices = get_available_voices("ar")
    print(f"  Available voices: {voices}")

    test_text = "\u0627\u0644\u062c\u0648 \u0627\u0644\u064a\u0648\u0645 \u0641\u064a \u0627\u0644\u0631\u064a\u0627\u0636 \u062d\u0627\u0631 \u0648\u0635\u062d\u0648\u060c \u0627\u0644\u062d\u0631\u0627\u0631\u0629 38 \u062f\u0631\u062c\u0629"
    print(f"  Text: '{test_text}'")

    t0 = time.time()
    try:
        audio = synthesize_speech(test_text, language="ar", voice="fahad")
        elapsed = time.time() - t0
        print(f"  Audio size: {len(audio)} bytes")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Has WAV header: {audio[:4] == b'RIFF'}")

        out_path = "test_output_arabic.wav"
        with open(out_path, "wb") as f:
            f.write(audio)
        print(f"  Saved to: {out_path}")
        print(f"  Status: {PASS}")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Error: {e}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {FAIL}")
        return False


def test_tts_chunking():
    print("\n" + "=" * 60)
    print("TEST 4: TTS - Long text chunking (>200 chars)")
    print("=" * 60)

    from tts import synthesize_speech, _chunk_text

    long_text = (
        "The weather in Riyadh this week shows temperatures ranging from "
        "35 to 42 degrees Celsius. Monday will be sunny and hot. "
        "Tuesday brings light winds from the northwest. "
        "Wednesday and Thursday remain clear with low humidity. "
        "Friday may see some dust in the air."
    )
    print(f"  Text length: {len(long_text)} chars")

    chunks = _chunk_text(long_text)
    print(f"  Chunks created: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"    Chunk {i+1}: {len(c)} chars - '{c[:60]}...'")

    t0 = time.time()
    try:
        audio = synthesize_speech(long_text, language="en")
        elapsed = time.time() - t0
        print(f"  Audio size: {len(audio)} bytes")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {PASS}")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Error: {e}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {FAIL}")
        return False


def test_all_english_voices():
    print("\n" + "=" * 60)
    print("TEST 5: TTS - All English voices")
    print("=" * 60)

    from tts import synthesize_speech, get_available_voices

    voices = get_available_voices("en")
    text = "Hello, I am Cloud, your weather assistant."
    results = []

    for voice in voices:
        t0 = time.time()
        try:
            audio = synthesize_speech(text, language="en", voice=voice)
            elapsed = time.time() - t0
            print(f"  {voice}: {PASS} ({len(audio)} bytes, {elapsed:.2f}s)")
            results.append(True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {voice}: {FAIL} - {e} ({elapsed:.2f}s)")
            results.append(False)

    passed = all(results)
    print(f"  Status: {PASS if passed else FAIL} ({sum(results)}/{len(results)})")
    return passed


def test_all_arabic_voices():
    print("\n" + "=" * 60)
    print("TEST 6: TTS - All Arabic Saudi voices")
    print("=" * 60)

    from tts import synthesize_speech, get_available_voices

    voices = get_available_voices("ar")
    text = "\u0645\u0631\u062d\u0628\u0627\u060c \u0627\u0646\u0627 \u0643\u0644\u0627\u0648\u062f\u060c \u0645\u0633\u0627\u0639\u062f\u0643 \u0644\u0644\u0637\u0642\u0633"
    results = []

    for voice in voices:
        t0 = time.time()
        try:
            audio = synthesize_speech(text, language="ar", voice=voice)
            elapsed = time.time() - t0
            print(f"  {voice}: {PASS} ({len(audio)} bytes, {elapsed:.2f}s)")
            results.append(True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {voice}: {FAIL} - {e} ({elapsed:.2f}s)")
            results.append(False)

    passed = all(results)
    print(f"  Status: {PASS if passed else FAIL} ({sum(results)}/{len(results)})")
    return passed


def main():
    print("=" * 60)
    print("  GROQ MODELS TEST SUITE")
    print(f"  API Key: {'SET' if GROQ_API_KEY else 'MISSING'}")
    print("=" * 60)

    if not GROQ_API_KEY:
        print("\nERROR: GROQ_API_KEY not set in .env")
        sys.exit(1)

    results = {}
    t_total = time.time()

    results["STT Whisper"] = test_stt_whisper()
    results["TTS English"] = test_tts_english()
    results["TTS Arabic Saudi"] = test_tts_arabic()
    results["TTS Chunking"] = test_tts_chunking()
    results["All EN Voices"] = test_all_english_voices()
    results["All AR Voices"] = test_all_arabic_voices()

    total_time = time.time() - t_total

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if ok:
            passed += 1

    total = len(results)
    print(f"\n  {passed}/{total} tests passed in {total_time:.1f}s")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
