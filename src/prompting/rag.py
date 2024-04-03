from typing import List
from src.inference.chatgpt import MultiRolePrompt


RAG_SYSTEM_PROMPT = """\
You are a mathematician, you are supposed to answer the given question. \
You will be provided with a number of documents, which may be helpful. \
You need to output the answer in your final sentence like "Therefore, the answer is ...". \
The answer can only be one of the following forms:
1. a numerical value like 0.1, no symbol and no unit at all.
2. a list of number like [2, 3, 4].
3. True/False.
4. an option like (a), (b), (c), (d)
"""


def build_rag_prompt(question: str, documents: List[str]) -> MultiRolePrompt:
    # TODO: Add parameter for answer_type?
    prompt_documents = "\n".join(
        f"Document {i+1}:\n----\n{document.strip()}\n"
        for i, document in enumerate(documents)
    )
    user_prompt = (
        f"Question:\n----\n{question}\n\n"
        f"{prompt_documents}\n"
        "\n"
    )
    return MultiRolePrompt(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )
