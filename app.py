import streamlit as st
import os
import google.generativeai as genai
from time import sleep as sleep


# App title
st.set_page_config(page_title="🦙💬 Chatbot")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


with st.sidebar:
    st.title("🦙💬 Tagglabs Chatbot")
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        [
            "gemini-pro",
            "gemini-1.0-pro",
            "gemini-1.0-pro-001",
            "gemini-1.0-pro-latest",
            "gemini-pro-vision",
            "gemini-1.0-pro-vision-latest",
        ],
        key="selected_model",
        index=0,
    )
model = genai.GenerativeModel(selected_model)


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initialize an empty list to store all conversations
if "history" not in st.session_state.keys():
    st.session_state.history = []


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]
    st.session_state.history = []
    


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_response(prompt_input):
    # Add new conversation to history
    st.session_state.history.append({"role": "user", "content": f"{prompt_input}"})

    # Generate response
    output = model.generate(prompt=prompt_input)
    
    # Append response to history
    st.session_state.history.append({"role": "assistant", "content": output.text})

    return output.text


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # with st.spinner("Thinking..."):
        responses = generate_response(prompt)
        full_response = ""
        message_placeholder = st.empty()
        for response in responses:
            full_response += response  # .choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
            sleep(0.1)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
