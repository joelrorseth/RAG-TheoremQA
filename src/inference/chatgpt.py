from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole

LLM_MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.1


def prompt_llm(prompt: str) -> str:
    llm = OpenAI(model=LLM_MODEL_NAME, temperature=TEMPERATURE)

    chat_messages = [
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]
    chat_response = llm.chat(chat_messages)
    return chat_response.message.content
