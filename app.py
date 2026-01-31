import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
from src.rag_pipeline import run_voice_rag

st.title("Voice-Enabled Knowledge Agent")

# Bonus Task requirement: Display the answer text in a "Chat" format
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Recording logic
audio = mic_recorder(start_prompt="Record Question ⏺️", stop_prompt="Stop ⏹️", key='recorder')

if audio:
    audio_path = "query.wav"
    with open(audio_path, "wb") as f:
        f.write(audio['bytes'])
    
    with st.spinner("Thinking..."):
        answer = run_voice_rag(audio_path)
        
        # Add to chat history
        st.session_state.messages.append({"role": "user", "content": "Voice Question Received"})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()