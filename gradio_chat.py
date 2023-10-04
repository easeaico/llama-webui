import os
import json

import gradio as gr

from llama_cpp import Llama


PROMPT = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

"""

model_path = "../Nous-Hermes-Llama2-70b/nous-hermes-llama2-70b.Q5_K_M.gguf"
gui_title = "Nous Hermes LlaMA-2 70B"
llm = Llama(model_path=model_path, n_ctx=4096)

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
    stream = llm(query, max_tokens=2048, stop=["</s>"], stream=True)

    for output in stream:
        text += output["choices"][0]["text"]
        yield text
    
demo = gr.ChatInterface(
    predict,
    chatbot=gr.Chatbot(height=900),
    textbox=gr.Textbox(placeholder="Can I help you?", container=False, scale=7),
    title=gui_title,
).queue()

demo.launch(share=True)
