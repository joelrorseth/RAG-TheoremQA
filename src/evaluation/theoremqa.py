# NOTE: Parts of this file are adapted from the following repository:
# https://github.com/wenhuchen/TheoremQA

from typing import Dict, Any, List, Optional
import logging
import json
import re
import ast
from sympy import Rational
import numpy as np
from config import (
    EVAL_DATA_PATH,
    OUTPUTS_DATA_PATH,
    RESULTS_DATA_PATH,
    Experiment,
    MultiRolePrompt
)
from src.inference.chatgpt import prompt_llm


THEOREMQA_PATH = EVAL_DATA_PATH / "theoremqa/test.json"

THEOREMQA_EXTRACT_ANSWER_SYSTEM_PROMPT = """\
You are supposed to extract the numeric answer \
(answer or Python formula or latex form) from a given string. \
If there is a unit in the input, try to remove that and only keep the number. \
If you think there is no numerical number within the input, just return 0."""

THEOREMQA_EXTRACT_ANSWER_PARTIAL_USER_PROMPT = """\
Input: 1/8 for both starting positions
Output: 1/8

Input: 0 an Euler homogeneous equation?
Output: 0

Input: based on the evaluation, the answer is [2, 3, 4].
Output: [2, 3, 4]

Input: Therefore, it will take 330 ms for client A to receive the whole file from the server after sending a request
Output: 330

Input: 3.02xmath.pow(10, 16) V
Output: 3.02*math.pow(10, 16)

Input: 4kHz
Output: 4

Input: individual will work 4,800 hours
Output: 4800

Input: the overall margin exceeds $13,133.4
Output: 13133.4

Input: x^y - 2e(x)
Output: 0

Input: 0.3465735 (approximate value)
Output: 0.3465735

Input: 3 and 4
Output: [3, 4]

Input: 3.57 * 10^(-29)
Output: 3.57 * math.pow(10, -29)

Input: """


def _extract_answer(query: str, use_azure: bool = False):
    system_prompt = THEOREMQA_EXTRACT_ANSWER_SYSTEM_PROMPT
    user_prompt = f"{THEOREMQA_EXTRACT_ANSWER_PARTIAL_USER_PROMPT}{query}\nOutput:"
    return prompt_llm(MultiRolePrompt(
        user_prompt=user_prompt, system_prompt=system_prompt
    )).strip()

# NOTE: We are not using Wolfram
# def get_decimal_with_wolfram(string: str) -> float:
#     for ex in wolfram_client.query(f'compute {string}').pods:
#         if ex['@title'] in ['Decimal approximation', 'Decimal form']:
#             for sub in ex.subpods:
#                 try:
#                     return float(sub['plaintext'][:20])
#                 except Exception:
#                     pass

#     for ex in wolfram_client.query(f'compute {string}').pods:
#         if ex['@title'] in ['Result']:
#             for sub in ex.subpods:
#                 try:
#                     return float(sub['plaintext'][:8])
#                 except Exception:
#                     pass

#     return None


def _find_numbers_in_string(s: str):
    pattern = r'[-+]?(?:\d*\.*\d+)'
    numbers = re.findall(pattern, s)
    tmp = [float(x) for x in numbers]
    if len(tmp) == 0:
        return None
    elif len(tmp) == 1:
        return tmp[0]
    else:
        return tmp


def _within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def _parse_number_list(s: str):
    # Check if the string is a valid list by trying to parse it
    parsed_list = ast.literal_eval(s)
    return parsed_list


def _compare_two_numbers(p, gt):
    if isinstance(p, int) or isinstance(p, float):
        pass
    elif isinstance(p, list) or isinstance(p, bool) or isinstance(p, str):
        return False
    elif isinstance(p, tuple) or isinstance(p, complex) or isinstance(p, dict):
        return False
    else:
        raise ValueError(p)

    if isinstance(gt, float):
        return _within_eps(pred=p, gt=gt)
    else:
        return round(p) == gt


def _compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    elif len(pred) != len(gt):
        return False
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([_compare_two_numbers(p, g) for p, g in zip(pred, gt)])


def _is_number(string):
    pattern = r'^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$'
    match = re.match(pattern, string)
    return bool(match)


def _is_scientific_number(string):
    pattern = r'^[-+]?\d+(\.\d+)?e[-]?\d+$'
    match = re.match(pattern, string)
    return bool(match)


def _contain_num_and_str(string):
    pattern_str = r'[a-zA-Z]'
    pattern_num = r'[0-9]'
    return bool(re.search(pattern_str, string) and re.search(pattern_num, string))


def _normalize(prediction: str):
    # Preprocessing the string [Stage 1]
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else '0'

    # Replace special tokens
    if '=' in prediction:
        prediction = prediction.split('=')[-1].strip()
    if '≈' in prediction:
        prediction = prediction.split('≈')[-1].strip()
    if '`' in prediction:
        prediction = prediction.replace('`', '')
    if '$' in prediction:
        prediction = prediction.replace('$', '')
    if '°' in prediction:
        prediction = prediction.replace('°', '')

    # Detect the boolean keyword in the generation
    if prediction in ['true', 'yes', 'false', 'no']:
        if prediction == 'true' or prediction == 'yes':
            prediction = 'True'
        else:
            prediction = 'False'
    if 'True' in prediction or 'False' in prediction:
        prediction = 'True' if 'True' in prediction else 'False'

    # Detect the approximation keyword
    if 'approximately' in prediction:
        prediction = prediction.replace('approximately', '').strip()
    if ' or ' in prediction:
        prediction = prediction.split(' or ')[0]

    # Drop the units before and after the number
    if re.match(r'[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$', prediction):
        prediction = re.search(
            r'([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$', prediction).group(1)
    if re.match(r'[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(
            r'[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$', prediction).group(1)
    if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
        prediction = re.search(
            r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction).group(1)
    if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
        prediction = re.search(
            r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction).group(1)

    # Preprocessing the number [Stage 1]
    if '10^' in prediction:
        prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)
    if ' x ' in prediction:
        prediction = prediction.replace(' x ', '*')
    if ' × ' in prediction:
        prediction = prediction.replace(' × ', '*')
    if _is_number(prediction):
        prediction = prediction.replace(',', '')

    # Preprocessing the option [Stage 3]
    if '(a)' in prediction or '(b)' in prediction or '(c)' in prediction or '(d)' in prediction:
        prediction = '"' + re.search(r'\([a-d]\)', prediction).group(0) + '"'

    # If the prediction is empty, use dummy '0'
    if not prediction:
        prediction = '0'

    # Converting the string answer to a number/list/bool/option
    try:
        prediction = eval(prediction)
    except Exception:
        # extracting the answer with ChatGPT and try again
        prediction = _extract_answer(prediction, False)
        try:
            prediction = eval(prediction)
        except Exception:
            logging.error(f"Failed to normalize prediction: {prediction}")
            prediction = 0
            # NOTE: We are not using Wolfram
            # print(entry['id'])
            # print('Wolfram: ------------------', prediction)
            # tmp = get_decimal_with_wolfram(prediction)
            # if tmp is None:
            #     print('Wolfram Fail: ------------------', prediction)
            #     prediction = _find_numbers_in_string(prediction)
            #     # If it does not find any number; boil down to base case.
            #     if prediction is None:
            #         prediction = 0
            # else:
            #     prediction = tmp
            #     print('Wolfram Success: ------------------', prediction)

    # Performing common type conversion
    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if isinstance(prediction[0], complex):
            prediction = [tmp.real for tmp in prediction]
        elif isinstance(prediction[0], Rational):
            prediction = [float(tmp) for tmp in prediction]
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real
        elif isinstance(prediction, Rational):
            prediction = float(prediction)

    return prediction


def evaluate_majority_predictions(answers, answer_type, subfield=""):
    # Given a list of CoT generation answers
    predictions = []
    similar_vote = [0 for _ in range(len(answers))]
    for answer in answers:
        prediction = extract_predicted_answer(answer)
        predictions.append(_normalize(prediction))

    for i, prediction in enumerate(predictions):
        for j in range(i + 1, len(answers)):
            tmp_prediction = predictions[j]
            if (isinstance(prediction, (str, int, float)) or isinstance(prediction, list)) \
                    and (isinstance(tmp_prediction, (str, int, float)) or isinstance(tmp_prediction, list)):
                # Comparing prediction against each others
                if answer_type in ['bool', 'option', 'Option']:
                    cur_correct = int(prediction == tmp_prediction)
                elif answer_type == 'integer':
                    cur_correct = int(_compare_two_numbers(prediction, tmp_prediction))
                elif answer_type == 'float':
                    cur_correct = int(_compare_two_numbers(prediction, tmp_prediction))
                elif answer_type in ['list of integer', 'list of float']:
                    cur_correct = int(_compare_two_list(prediction, tmp_prediction))
                similar_vote[i] += cur_correct
                similar_vote[j] += cur_correct

    max_value = max(similar_vote)
    index = similar_vote.index(max_value)
    return index, answers[index]


def evaluate_theoremqa_predictions(
    experiment: Experiment, subfield: str, overwrite: bool = False
):
    """
    Load and evaluate the predictions made for a given experiment.
    """

    results_file_path = RESULTS_DATA_PATH / \
        f"{experiment.to_string()}_{subfield.lower()}.json"

    # Handle existing results file
    if results_file_path.exists():
        if not overwrite:
            logging.info("Already generated these results, skipping")
            return
        else:
            logging.info("Removing old results")
            results_file_path.unlink()

    # Load predictions
    predictions_file_path = OUTPUTS_DATA_PATH / \
        f"{experiment.to_string()}_{subfield.lower()}.json"
    if not predictions_file_path.exists():
        raise ValueError(
            f"Cannot find TheoremQA predictions for experiment: {experiment.to_string()}"
        )

    with open(predictions_file_path, 'r') as file:
        theoremqa_prediction_objects: List[Dict[str, Any]] = json.load(file)

    all_entries = theoremqa_prediction_objects
    new_entries = []
    correct = 0
    logging.info(
        f"Evaluating TheoremQA predictions for experiment {experiment.to_string()}"
    )

    for entry_idx, entry in enumerate(all_entries):
        logging.info(
            f"Evaluating TheoremQA prediction for question {entry_idx+1}/{len(all_entries)}"
        )

        need_further_parsing = False
        prediction = entry['prediction']
        if 'answer' in entry:
            gt = entry['answer']
        else:
            gt = entry['Answer']
        if 'answer_type' in entry:
            answer_type = entry['answer_type']
        else:
            answer_type = entry['Answer_type']

        prediction = _normalize(prediction)
        if isinstance(prediction, (str, int, float)) or isinstance(prediction, list):
            # Comparing prediction against the reference
            if answer_type in ['bool', 'option', 'Option']:
                cur_correct = int(prediction == gt)
            elif answer_type == 'integer':
                cur_correct = int(_compare_two_numbers(prediction, gt))
            elif answer_type == 'float':
                cur_correct = int(_compare_two_numbers(prediction, gt))
            elif answer_type in ['list of integer', 'list of float']:
                cur_correct = int(_compare_two_list(prediction, gt))
            entry['prediction'] = prediction
        else:
            entry['prediction'] = str(prediction)
            cur_correct = 0
            logging.error(f"Failed to parse prediction: {prediction}")

        entry['correct'] = bool(cur_correct)
        new_entries.append(entry)

        correct += cur_correct

    logging.info(f"Experiment accuracy = {correct / len(all_entries)}")

    # Write results to file
    with open(results_file_path, 'w') as f:
        json.dump(new_entries, f, ensure_ascii=False, indent=4)


def load_theoremqa() -> List[Dict[str, Any]]:
    with open(THEOREMQA_PATH, "r") as file:
        return json.load(file)


def get_theoremqa_questions_for_subfield(subfield: str) -> List[Dict[str, Any]]:
    all_theoremqa_questions = load_theoremqa()
    # TODO: Double check that all subfield names match index names
    return [
        question
        for question in all_theoremqa_questions
        if question["subfield"].lower() == subfield
    ]


def extract_predicted_answer(result: str) -> str:
    prediction = result.strip().strip('\n').split('\n')[-1]
    tmp = ''
    for entry in prediction.split(' ')[::-1]:
        if entry == 'is' or entry == 'be' or entry == 'are' or entry.endswith(':'):
            break
        tmp = entry + ' ' + tmp
    prediction = tmp.strip().strip('.')
    return prediction


def build_theoremqa_prediction_obj(
    theoremqa_question_obj: Dict[str, Any], llm_response: str,
    llm_system_prompt: Optional[str] = None, llm_user_prompt: Optional[str] = None
):
    prediction = extract_predicted_answer(result=llm_response)
    return {
        'id': theoremqa_question_obj['id'],
        'question': theoremqa_question_obj['Question'],
        'prediction': prediction,
        'llm_system_prompt': llm_system_prompt,
        'llm_user_prompt': llm_user_prompt,
        'answer': theoremqa_question_obj['Answer'],
        'rationale': theoremqa_question_obj,
        'answer_type': theoremqa_question_obj['Answer_type'],
    }


def write_theoremqa_predictions(
    theoremqa_prediction_objects: List[Dict[str, Any]], experiment: Experiment, subfield: str
):
    predictions_file_path = OUTPUTS_DATA_PATH / \
        f"{experiment.to_string()}_{subfield.lower()}.json"
    with open(predictions_file_path, 'w', encoding='utf-8') as f:
        json.dump(
            theoremqa_prediction_objects, f, ensure_ascii=False, indent=4
        )
