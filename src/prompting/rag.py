from typing import List
from src.inference.chatgpt import MultiRolePrompt


RAG_SYSTEM_PROMPT = """
You must determine the correct answer to a given question.
You will be provided with a number of documents, which may be helpful.
Respond with only the value of the answer, and absolutely nothing else.
"""


def build_rag_prompt(question: str, sources: List[str]) -> MultiRolePrompt:
    # TODO: Add parameter for answer_type?
    prompt_sources = "\n".join(
        f"SOURCE {i+1}:\n----\n{source.strip()}\n"
        for i, source in enumerate(sources)
    )
    user_prompt = (
        f"QUESTION:\n----\n{question}\n\n"
        f"{prompt_sources}\n"
        "----\nANSWER: "
    )
    return MultiRolePrompt(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )
