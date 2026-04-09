import sys
import os
import re
import time
import logging
import warnings
import base64
import html as _html

sys.path.insert(0, os.path.dirname(__file__))

warnings.filterwarnings("ignore", message=".*torchvision.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st

st.set_page_config(
    page_title="Cloud - Weather Agent",
    page_icon="https://cdn-icons-png.flaticon.com/512/1163/1163661.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.station-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 1rem 1.5rem; border-radius: 14px; margin-bottom: 0.8rem;
    color: white; display: flex; align-items: center; gap: 12px;
}
.station-header h1 { font-size: 1.4rem; font-weight: 700; margin: 0; }
.station-header p { font-size: 0.75rem; opacity: 0.6; margin: 0; }
.station-header .live-dot {
    width: 8px; height: 8px; background: #10b981; border-radius: 50%;
    display: inline-block; margin-right: 4px; animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }
    50% { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}
.voice-answer {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 1px solid #10b981; border-radius: 12px; padding: 1.2rem 1.5rem;
    color: #d1fae5; font-size: 1.05rem; line-height: 1.6; margin: 0.8rem 0;
}
.voice-answer .voice-label {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;
    color: #6ee7b7; margin-bottom: 0.4rem; font-weight: 600;
}
.unit-pill {
    display: inline-flex; align-items: center; gap: 0;
    background: #1e293b; border: 1px solid #334155; border-radius: 999px;
    padding: 3px; user-select: none;
}
.unit-pill a {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.5px;
    text-decoration: none; transition: all 0.12s ease;
    color: #64748b;
}
.unit-pill a.active {
    background: #6366f1; color: #fff; box-shadow: 0 1px 4px #6366f166;
}
</style>
""", unsafe_allow_html=True)


import dspy
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_lm = dspy.LM(
    model=f"openai/{DEEPSEEK_MODEL}",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    temperature=0.0,
    max_tokens=2000,
)
try:
    # Streamlit reruns can execute in different threads. DSPy 3.x locks settings
    # to the thread that first configures them, so we must avoid re-configuring.
    if getattr(dspy.settings, "lm", None) is None:
        dspy.configure(lm=_lm)
except RuntimeError:
    # If another thread already configured DSPy, keep the existing settings.
    pass


@st.cache_resource
def _boot():
    from agent.graph import build_graph
    return build_graph()


_graph = _boot()

_CELSIUS_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*°\s*C\b')
_FAHRENHEIT_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*°\s*F\b')


def _convert_temps(text: str, to_unit: str) -> str:
    """Instant °C↔°F conversion using regex + arithmetic. No LLM, no network."""
    if to_unit == "F":
        def _c_to_f(m):
            v = float(m.group(1))
            r = (v * 9 / 5) + 32
            out = int(r) if r == int(r) else round(r, 1)
            return f"{out}°F"
        return _CELSIUS_RE.sub(_c_to_f, text)
    else:
        def _f_to_c(m):
            v = float(m.group(1))
            r = (v - 32) * 5 / 9
            out = int(r) if r == int(r) else round(r, 1)
            return f"{out}°C"
        return _FAHRENHEIT_RE.sub(_f_to_c, text)


def _translate_ar_to_en(text: str) -> str:
                                                                              
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL, temperature=0.0, timeout=15,
    )
    resp = llm.invoke([HumanMessage(
        content=f"Translate this Arabic weather query to English. "
                f"Return ONLY the English translation, nothing else.\n\n{text}"
    )])
    return resp.content.strip()


def _run(query, voice_mode, voice_language):
    from agent.state import WeatherAgentState

    pipeline_query = query
    if voice_mode and voice_language == "ar":
        try:
            pipeline_query = _translate_ar_to_en(query)
        except Exception:
            pipeline_query = query

    config = {"configurable": {"thread_id": f"ui_{int(time.time())}"}}
    initial = WeatherAgentState(
        user_query=pipeline_query, voice_mode=voice_mode, voice_language=voice_language,
    ).model_dump()
    t0 = time.time()
    result = _graph.invoke(initial, config=config)
    return result, time.time() - t0


def main():
    if "unit" not in st.session_state:
        st.session_state.unit = "C"
    if "last_output" not in st.session_state:
        st.session_state.last_output = ""

    hcol, ucol = st.columns([8, 1])
    with hcol:
        st.markdown("""
        <div class="station-header">
            <div><span class="live-dot"></span></div>
            <div>
                <h1>Cloud</h1>
                <p>Always listening. Say "Hey Cloud" for voice, or type below.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ucol:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        use_f = st.toggle("°F", value=(st.session_state.unit == "F"), key="unit_toggle",
                          help="Switch between Celsius and Fahrenheit")
        st.session_state.unit = "F" if use_f else "C"

    from components.audio_listener import audio_listener
    from wake_word import detect_wake_word
    from config.settings import GROQ_API_KEY as _groq_key

    audio_bytes = audio_listener(key="mic")

    if audio_bytes and _groq_key:
        from stt import transcribe_audio
        with st.spinner("Transcribing..."):
            try:
                stt_result = transcribe_audio(audio_bytes, filename="recording.webm")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                stt_result = None

        if stt_result and stt_result.text:
            voice_language = stt_result.language
            wake = detect_wake_word(stt_result.text)

            if wake.detected:
                lang_label = "AR" if voice_language == "ar" else "EN"
                st.caption(f"Cloud activated ({lang_label}): \"{wake.query}\"")

                with st.spinner("Cloud is thinking..."):
                    result, elapsed = _run(wake.query, voice_mode=True, voice_language=voice_language)

                output = result.get("final_latex_document", "")
                if output:
                    st.session_state.last_output = output
                    lang_display = "Arabic (Saudi)" if voice_language == "ar" else "English"
                    display = _convert_temps(output, st.session_state.unit)
                    safe_output = _html.escape(display)
                    st.markdown(
                        f"<div class='voice-answer'>"
                        f"<div class='voice-label'>Cloud ({lang_display}):</div>"
                        f"{safe_output}</div>",
                        unsafe_allow_html=True)
                    try:
                        from tts import synthesize_speech
                        wav = synthesize_speech(output, language=voice_language)
                        b64 = base64.b64encode(wav).decode()
                        st.markdown(
                            f'<audio autoplay style="display:none">'
                            f'<source src="data:audio/wav;base64,{b64}" type="audio/wav">'
                            f'</audio>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.warning(f"TTS failed: {e}")
                else:
                    st.warning("No response generated.")
            else:
                st.caption(f"No wake word detected. Heard: \"{stt_result.text}\"")

    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        text_query = st.text_input("Or type your question:", placeholder="What is the weather in Riyadh this week?", label_visibility="collapsed")
    with col2:
        run_btn = st.button("Run", type="primary", use_container_width=True)

    if run_btn and text_query:
        with st.spinner("Processing..."):
            result, elapsed = _run(text_query, voice_mode=False, voice_language="en")

        output = result.get("final_latex_document", "")
        if output:
            st.session_state.last_output = output

    if st.session_state.last_output:
        from latex_renderer import render as render_latex
        display = _convert_temps(st.session_state.last_output, st.session_state.unit)
        st.markdown(render_latex(display), unsafe_allow_html=True)
        with st.expander("Raw", expanded=False):
            st.code(display, language="latex")
    elif run_btn and text_query:
        st.warning("No output generated.")


if __name__ == "__main__":
    main()
