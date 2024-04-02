from config import MultiRolePrompt


BASIC_SYSTEM_PROMPT = """\
You are a mathematician, you are supposed to answer the given question. \
You need to output the answer in your final sentence like "Therefore, the answer is ...". \
The answer can only be one of the following forms:
1. a numerical value like 0.1, no symbol and no unit at all.
2. a list of number like [2, 3, 4].
3. True/False.
4. an option like (a), (b), (c), (d)
"""


def build_basic_prompt(question: str, answer_type: str) -> MultiRolePrompt:
    return MultiRolePrompt(
        system_prompt=BASIC_SYSTEM_PROMPT,
        user_prompt=f"Question: {question}"
    )
