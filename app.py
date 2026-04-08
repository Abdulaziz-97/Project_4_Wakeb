import sys
import os
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def _boot():
    import dspy
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    lm = dspy.LM(
        model=f"openai/{DEEPSEEK_MODEL}",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=2000,
    )
    dspy.configure(lm=lm)
    from agent.graph import build_graph
    return build_graph()


_graph = _boot()



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
    st.markdown("""
    <div class="station-header">
        <div><span class="live-dot"></span></div>
        <div>
            <h1>Cloud</h1>
            <p>Always listening. Say "Hey Cloud" for voice, or type below.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
                    lang_display = "Arabic (Saudi)" if voice_language == "ar" else "English"
                    safe_output = _html.escape(output)
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
            from latex_renderer import render as render_latex
            st.markdown(render_latex(output), unsafe_allow_html=True)
            with st.expander("Raw LaTeX", expanded=False):
                st.code(output, language="latex")
        else:
            st.warning("No output generated.")


if __name__ == "__main__":
    main()
