from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from config import LLM, LLM_TEMPERATURE, MultiRolePrompt, Prompt
import openai
import os


def prompt_llm(prompt: Prompt) -> str:
    llm = OpenAI(model=LLM.ChatGPT35.value, temperature=LLM_TEMPERATURE)

    if isinstance(prompt, MultiRolePrompt):
        chat_messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt.system_prompt),
            ChatMessage(role=MessageRole.USER, content=prompt.user_prompt),
        ]
    else:
        chat_messages = [
            ChatMessage(role=MessageRole.USER, content=prompt.user_prompt)
        ]

    chat_response = llm.chat(chat_messages)
    return chat_response.message.content


def run_tot_llm(prompt, n_sample=1, stop='\n', temperature=LLM_TEMPERATURE, max_tokens=1024):
    client = openai.OpenAI()

    outputs = []
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": prompt.user_prompt}
        ],
        n = n_sample, 
        stop = stop, 
        temperature= temperature,
        max_tokens = max_tokens,
    )


    outputs.extend([choice.message.content for choice in completion.choices])
    return outputs