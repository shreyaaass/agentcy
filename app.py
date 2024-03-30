import streamlit as st
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from accelerate import infer_auto_device_map, init_empty_weights
import os


# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

token = os.environ["HF_TOKEN"]


def clear_model():
    try:
        torch.cuda.empty_cache()
        global tokenizer
        global model
        del tokenizer
        del model
        torch.cuda.empty_cache()
        print("memory freed")
    except Exception as e:
        print(e)


@st.cache_resource
def load_model(model_id):
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=True,
            load_in_8bit_fp32_cpu_offload=True,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    config = AutoConfig.from_pretrained(
        model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
    )
    config.use_cache = False
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    device_map = infer_auto_device_map(model=model)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map=device_map,
        attn_implementation="flash_attention_2"
        if torch.cuda.is_available()
        else "eager",
    )

    return model


@st.cache_resource
def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, token=token, legacy=False, padding_side="left"
    )
    return tokenizer


torch.cuda.empty_cache()

with st.sidebar:
    st.title("🦙💬 Tagglabs Chatbot")

    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    # st.subheader("Models and parameters")

# model_id = "HuggingFaceH4/zephyr-7b-beta"
model_id = "google/gemma-7b"

model = load_model(model_id)
tokenizer = load_tokenizer(model_id)

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


def generate_llama2_response(prompt_input):
    # Add new conversation to history
    st.session_state.history.append({"role": "user", "content": f"{prompt_input}"})

    messages = [
        {
            "role": "system",
            "content": "You are a friendly anti-nsfw chatbot who always responds in short answer and crisp assistant voice. Only provide what is asked and nothing extra.",
        },
    ] + st.session_state.history

    prompt_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, return_tensors="pt"
    )

    inputs = tokenizer(
        prompt_input,
        return_tensors="pt",
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    outputs = model.generate(
        **inputs,
        streamer=streamer,
        use_cache=True,
        max_new_tokens=10000,
        temperature=0.1,
        top_p=0.5,
        repetition_penalty=1.15,
        do_sample=True,
        # early_stopping = True,
        # num_beams = 1,
        # top_k = 50,
    )

    output = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Append response from Llama2 to history
    st.session_state.history.append({"role": "assistant", "content": output})

    return output


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # with st.spinner("Thinking..."):
        responses = generate_llama2_response(prompt)
        full_response = ""
        message_placeholder = st.empty()
        for response in responses:
            full_response += response  # .choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
