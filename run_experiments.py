import logging
from config import LLM, EvaluationDataset, Experiment, IndexConfig, IndexingStrategy, PromptingStrategy
from src.evaluation.experiment import run_theoremqa_experiment

# SUBFIELD = "calculus"
SUBFIELD = "combinatorics"


experiments = [
    Experiment(
        llm=LLM.ChatGPT35,
        prompting_strategy=PromptingStrategy.COT_SC,
        index_config=None,
        prompting_strategy=PromptingStrategy.Basic,
        index_config=IndexConfig(
            indexing_strategy=IndexingStrategy.Subject,
            index_name=SUBFIELD
        ),
        evaluation_dataset=EvaluationDataset.TheoremQA
    ),
    Experiment(
        llm=LLM.ChatGPT35,
        prompting_strategy=PromptingStrategy.TOT,
        index_config=None,
        prompting_strategy=PromptingStrategy.COT,
        index_config=IndexConfig(
            indexing_strategy=IndexingStrategy.Subject,
            index_name=SUBFIELD
        ),
        evaluation_dataset=EvaluationDataset.TheoremQA
    ),
    Experiment(
        llm=LLM.ChatGPT35,
        prompting_strategy=PromptingStrategy.RAG_TOP1_NEARBY200,
        index_config=IndexConfig(
            indexing_strategy=IndexingStrategy.Subject,
            index_name=SUBFIELD
        ),
        evaluation_dataset=EvaluationDataset.TheoremQA
    ),
    Experiment(
        llm=LLM.ChatGPT35,
        prompting_strategy=PromptingStrategy.RAG_TOP2_NEARBY200,
        index_config=IndexConfig(
            indexing_strategy=IndexingStrategy.Subject,
            index_name=SUBFIELD
        ),
        evaluation_dataset=EvaluationDataset.TheoremQA
    ),
    Experiment(
        llm=LLM.ChatGPT35,
        prompting_strategy=PromptingStrategy.RAG_TOP2_NEARBY500,
        index_config=IndexConfig(
            indexing_strategy=IndexingStrategy.Subject,
            index_name=SUBFIELD
        ),
        evaluation_dataset=EvaluationDataset.TheoremQA
    ),
    # NOTE: Sections are too large for chatgpt3.5 context
    # Experiment(
    #     llm=LLM.ChatGPT35,
    #     prompting_strategy=PromptingStrategy.RAG_TOP1_SECTION,
    #     index_config=IndexConfig(
    #         indexing_strategy=IndexingStrategy.Subject,
    #         index_name=SUBFIELD
    #     ),
    #     evaluation_dataset=EvaluationDataset.TheoremQA
    # )
    # NOTE: Getting vague errors from chatgpt3.5
    # Experiment(
    #     llm=LLM.ChatGPT35,
    #     prompting_strategy=PromptingStrategy.RAG_TOP5_NEARBY500,
    #     index_config=IndexConfig(
    #         indexing_strategy=IndexingStrategy.Subject,
    #         index_name=SUBFIELD
    #     ),
    #     evaluation_dataset=EvaluationDataset.TheoremQA
    # )
]

for experiment in experiments:
    logging.info(f"Running experiment: {experiment.to_string()}")
    run_theoremqa_experiment(experiment, SUBFIELD, False)
