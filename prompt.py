def get_cot_prompt(question: str, answer_type: str):
    prompt = '''Please read a math question, and then think step by step to derive the answer. You need to output your final answer in the form of "Therefore, the answer is", followed by '''

    if answer_type == 'bool':
        prompt += "a True or False answer."
    elif answer_type == 'option':
        prompt += "a multiple choice answer, in the form of (a), (b), (c) or (d)."
    else:
        prompt += "a numerical answer."

    prompt += f"\nQuestion: {question}\n ###Answer:"
    return prompt 
