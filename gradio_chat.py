import os
import json

import gradio as gr

from llama_cpp import Llama


PROMPT = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

"""

llm = Llama(model_path="../chinese-alpaca-2-13b-hf/ggml-model-q4_0.bin")

PARAMETERS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_k": 50,
    "truncate": 1000,
    "max_new_tokens": 1024,
    "seed": 42,
    "stop_sequences": ["</s>"],
}


def format_message(message, history, memory_limit=5):
    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return PROMPT + f"{message} [/INST]"

    formatted_message = PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    # Handle the current message
    formatted_message += f"<s>[INST] {message} [/INST]"

    return formatted_message


def predict(message, history):
    query = format_message(message, history)
    text = ""
    stream = llm(query, max_tokens=1024, stop=["</s>"], stream=True)

    for output in stream:
        text += output["choices"][0]["text"]
        yield text
    
demo = gr.ChatInterface(
    predict,
    chatbot=gr.Chatbot(height=800),
    textbox=gr.Textbox(placeholder="Can I help you?", container=False, scale=7),
    title="Chinese Alpaca2 Base On LlaMA 13B Chat",
).queue()

demo.launch()
