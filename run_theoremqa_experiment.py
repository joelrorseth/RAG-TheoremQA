import json
import logging
from typing import Any, Dict, List
from config import EVAL_DATA_PATH
from src.indexing.strategy import IndexingStrategy
from src.indexing.vector import load_index
from src.inference.chatgpt import prompt_llm
from src.prompting.rag import build_rag_chatgpt_prompt
from src.retrieval.vector import retrieve_top_k_documents


def load_theoremqa() -> List[Dict[str, Any]]:
    dataset_path = EVAL_DATA_PATH / "theoremqa/test.json"
    with open(dataset_path, "r") as file:
        return json.load(file)


def get_theoremqa_questions_for_subfield(
    all_theoremqa_questions: List[Dict[str, Any]], subfield: str
) -> List[Dict[str, Any]]:
    # TODO: Double check that all subfield names match index names
    return [
        question
        for question in all_theoremqa_questions
        if question["subfield"].lower() == subfield
    ]


def run_experiment_for_one_question(
    indexing_strategy: IndexingStrategy, subfield: str, k: int = 2
):
    all_theoremqa_questions = load_theoremqa()
    subfield_theoremqa_questions = get_theoremqa_questions_for_subfield(
        all_theoremqa_questions, subfield
    )

    theorem_question_obj = subfield_theoremqa_questions[0]
    question: str = theorem_question_obj["Question"]
    actual_answer: Any = theorem_question_obj["Answer"]

    logging.info(f"Question: {question}")
    logging.info(f"Actual Answer: {actual_answer}")

    index = load_index(subfield, indexing_strategy)
    documents = retrieve_top_k_documents(index, question, k)
    prompt = build_rag_chatgpt_prompt(question, documents)

    llm_answer = prompt_llm(prompt)
    logging.info(f"LLM Answer: {llm_answer}")


# Example
indexing_strategy = IndexingStrategy.Subject
subfield = "algebra"
run_experiment_for_one_question(indexing_strategy, subfield)
