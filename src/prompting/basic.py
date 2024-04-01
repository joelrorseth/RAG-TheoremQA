def build_basic_prompt(question: str, answer_type: str) -> str:
    prompt = '''Please solve the following math question with'''

    if answer_type == 'bool':
        prompt += "a True or False answer."
    elif answer_type == 'option':
        prompt += "a multiple choice answer, in the form of (a), (b), (c) or (d)."
    else:
        prompt += "a numerical answer."

    prompt += f"\nQuestion: {question}\n ###Answer:"
    return prompt
