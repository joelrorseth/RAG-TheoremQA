from config import MultiRolePrompt
from src.prompting.basic import BASIC_SYSTEM_PROMPT


def build_cot_prompt(question: str, answer_type: str) -> MultiRolePrompt:
    return MultiRolePrompt(
        system_prompt=BASIC_SYSTEM_PROMPT,
        user_prompt=f"Question: {question}\nLet\'s think step by step."
    )
