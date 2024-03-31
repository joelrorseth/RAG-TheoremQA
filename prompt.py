def get_standard_prompt(question: str, answer_type: str):
    prompt = '''Please solve the following math question with'''

    if answer_type == 'bool':
        prompt += "a True or False answer."
    elif answer_type == 'option':
        prompt += "a multiple choice answer, in the form of (a), (b), (c) or (d)."
    else:
        prompt += "a numerical answer."

    prompt += f"\nQuestion: {question}\n ###Answer:"
    return prompt 



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


# Functions for Tree of Thought Prompting
def get_tot_sample_prompt(question: str, rationale: str, answer_type: str):
    prompt = f"Please read a math question, and then think step by step to derive the answer. Question: {question}\n"
    prompt += f"Consider the reasoning steps: \n {rationale}\n"
    prompt += "Write the continuation steps line by line. Output your final answer in the form of \"Therefore, the answer is \", followed by "
    if answer_type == 'bool':
        prompt += "a True or False answer."
    elif answer_type == 'option':
        prompt += "a multiple choice answer, in the form of (a), (b), (c) or (d)."
    else:
        prompt += "a numerical answer."

    return prompt


def get_tot_propose_prompt(question: str, rationale: str, n_propose: int):
    prompt = f"Please read a math question: {question}\n"
    prompt += f"Consider the reasoning steps: \n {rationale}\n"
    prompt += f"Propose {n_propose} different possible next steps to solve the question."
    prompt += "Write every proposal in their own line."
    return prompt


def get_tot_vote_prompt(question: str, nodes: list):
    prompt = f"Consider the math question: {question}\n"
    prompt += "Given several approaches to solve this question, decide which one is most promising. Analyze each choice in detail. "
    for i, node in enumerate(nodes):
        prompt += f"Choice {i}. \n{node}\n"

    prompt += "Conclude in the last line \"The best choice is {s}\", where s the integer id of the choice."
    return prompt

