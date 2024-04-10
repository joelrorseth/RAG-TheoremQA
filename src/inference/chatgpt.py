import logging
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from config import LLM, LLM_TEMPERATURE, MultiRolePrompt, Prompt
import openai


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

    response_str = "N/A"
    try:
        chat_response = llm.chat(chat_messages)
        response_str = chat_response.message.content
    except openai.APIError as e:
        logging.error(
            "ChatGPT threw the following exception for the following prompt:"
        )
        logging.error(prompt)
        logging.error(e.message)

    return response_str


def run_tot_llm(prompt, n_sample=1, stop='\n', temperature=LLM_TEMPERATURE, max_tokens=1024):
    client = openai.OpenAI()

    outputs = []
    completion = client.chat.completions.create(
        model=LLM.ChatGPT35.value,
        messages=[
            {"role": "user", "content": prompt.user_prompt}
        ],
        n=n_sample,
        stop=stop,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    outputs.extend([choice.message.content for choice in completion.choices])
    return outputs
