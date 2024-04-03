import logging
from config import (
    LLM,
    OUTPUTS_DATA_PATH,
    Experiment,
    MultiRolePrompt,
    Prompt,
    PromptingStrategy,
    RetrievalStrategy
)
from src.evaluation.theoremqa import (
    build_theoremqa_prediction_obj,
    evaluate_theoremqa_predictions,
    get_theoremqa_questions_for_subfield,
    write_theoremqa_predictions
)
from src.indexing.vector import load_index
from src.inference.chatgpt import prompt_llm
from src.prompting.basic import build_basic_prompt
from src.prompting.cot import build_cot_prompt
from src.prompting.rag import build_rag_prompt
from src.retrieval.vector import retrieve_top_k_documents


def _get_experiment_prompt(
    experiment: Experiment, question: str, answer_type: str
) -> Prompt:
    """
    Create an LLM prompt according to experiment parameters.
    """
    def _get_experiment_rag_prompt(
        k: int, retrieval_strategy: RetrievalStrategy, num_words_per_doc: int = 4000
    ):
        if experiment.index_config is not None:
            index = load_index(
                experiment.index_config.index_name,
                experiment.index_config.indexing_strategy
            )
            documents = retrieve_top_k_documents(
                index, question, k, retrieval_strategy, num_words_per_doc
            )
            return build_rag_prompt(question, documents)
        else:
            raise ValueError(
                "RAG prompting strategy requires nonempty index config"
            )

    if experiment.prompting_strategy == PromptingStrategy.Basic:
        return build_basic_prompt(question, answer_type)
    elif experiment.prompting_strategy == PromptingStrategy.COT:
        return build_cot_prompt(question, answer_type)
    elif experiment.prompting_strategy == PromptingStrategy.RAG_TOP5_NEARBY500:
        return _get_experiment_rag_prompt(5, RetrievalStrategy.Nearby, 500)
    elif experiment.prompting_strategy == PromptingStrategy.RAG_TOP2_NEARBY500:
        return _get_experiment_rag_prompt(2, RetrievalStrategy.Nearby, 500)
    elif experiment.prompting_strategy == PromptingStrategy.RAG_TOP2_NEARBY200:
        return _get_experiment_rag_prompt(2, RetrievalStrategy.Nearby, 200)
    elif experiment.prompting_strategy == PromptingStrategy.RAG_TOP1_NEARBY500:
        return _get_experiment_rag_prompt(1, RetrievalStrategy.Nearby, 500)
    elif experiment.prompting_strategy == PromptingStrategy.RAG_TOP1_NEARBY200:
        return _get_experiment_rag_prompt(1, RetrievalStrategy.Nearby, 200)
    elif experiment.prompting_strategy == PromptingStrategy.RAG_TOP1_SECTION:
        return _get_experiment_rag_prompt(1, RetrievalStrategy.Section)
    else:
        raise ValueError(
            f"Unexpected prompting strategy {experiment.prompting_strategy.value}"
        )


def _prompt_experiment_llm(experiment: Experiment, prompt: Prompt) -> str:
    """
    Pose a prompt to the LLM specified by the experiment parameters.
    """
    if experiment.llm == LLM.ChatGPT35:
        return prompt_llm(prompt)
    else:
        raise ValueError(f"Unexpected LLM type {experiment.llm.value}")


def _generate_theoremqa_predictions(
    experiment: Experiment, subfield: str, overwrite: bool = False
):
    """
    Generate LLM predictions for all TheoremQA questions in a given experiment.
    """
    question_objects = get_theoremqa_questions_for_subfield(subfield)
    prediction_objects = []
    predictions_file_path = OUTPUTS_DATA_PATH / \
        f"{experiment.to_string()}_{subfield.lower()}.json"

    # Handle existing predictions file
    if predictions_file_path.exists():
        if not overwrite:
            logging.info("Already generated these predictions, skipping")
            return
        else:
            logging.info("Removing old predictions")
            predictions_file_path.unlink()

    for question_idx, question_obj in enumerate(question_objects):
        logging.info(
            f"Predicting TheoremQA question {question_idx+1}/{len(question_objects)}"
        )

        # Get prediction
        question: str = question_obj["Question"]
        actual_answer_type: str = question_obj["Answer_type"]
        prompt = _get_experiment_prompt(
            experiment, question, actual_answer_type
        )
        llm_answer = _prompt_experiment_llm(experiment, prompt)
        prediction_objects.append(
            build_theoremqa_prediction_obj(
                question_obj,
                llm_answer,
                prompt.system_prompt if isinstance(
                    prompt, MultiRolePrompt
                ) else None,
                prompt.user_prompt
            )
        )

    # Write all question predictions to file
    write_theoremqa_predictions(prediction_objects, experiment, subfield)


def run_theoremqa_experiment(
    experiment: Experiment, subfield: str, overwrite: bool = False
):
    """
    Generate and evaluate LLM predictions for all TheoremQA questions in a
    given experiment.
    """
    _generate_theoremqa_predictions(experiment, subfield, overwrite)
    evaluate_theoremqa_predictions(experiment, subfield, overwrite)
