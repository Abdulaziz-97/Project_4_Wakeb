import base64
import streamlit as st
import streamlit.components.v1 as components
import os

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component = components.declare_component("audio_listener", path=_FRONTEND_DIR)


def audio_listener(key=None):
    raw = _component(key=key, default=None)
    if raw and isinstance(raw, dict) and raw.get("audio"):
        ts = raw.get("timestamp", 0)
        last = st.session_state.get("_last_audio_ts", 0)
        if ts <= last:
            return None
        st.session_state["_last_audio_ts"] = ts
        audio_b64 = raw["audio"]
        if "," in audio_b64:
            audio_b64 = audio_b64.split(",", 1)[1]
        return base64.b64decode(audio_b64)
    return None
