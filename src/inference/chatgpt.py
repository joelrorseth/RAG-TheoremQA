from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from config import LLM, LLM_TEMPERATURE, MultiRolePrompt, Prompt


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
