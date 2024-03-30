from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel

LLM_MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.1


class ChatGPTPrompt(BaseModel):
    system_prompt: str
    user_prompt: str


def prompt_llm(prompt: ChatGPTPrompt) -> str:
    llm = OpenAI(model=LLM_MODEL_NAME, temperature=TEMPERATURE)
    chat_messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt.system_prompt),
        ChatMessage(role=MessageRole.USER, content=prompt.user_prompt),
    ]
    chat_response = llm.chat(chat_messages)
    return chat_response.message.content
