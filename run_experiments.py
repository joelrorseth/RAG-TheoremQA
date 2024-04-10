import logging
from config import (
    EMBEDDING_MODELS,
    LLM,
    EvaluationDataset,
    Experiment,
    IndexConfig,
    IndexingStrategy,
    PromptingStrategy,
    TextbookSubfield
)
from src.evaluation.experiment import run_theoremqa_experiment
from src.downloading.download_pdf_textbooks import download_all_pdf_textbooks
from src.indexing.vector import build_all_vector_indexes
from src.parsing.parse_pdf_textbooks import parse_all_pdf_textbooks
from src.preprocessing.preprocess_pdf_textbooks import preprocess_all_pdf_textbooks


def build_all_indexes():
    logging.info("Downloading all textbooks")
    download_all_pdf_textbooks()
    logging.info("Successfully downloaded all textbooks")

    logging.info("Parsing all textbooks")
    parse_all_pdf_textbooks()
    logging.info("Successfully parsed all textbooks")

    logging.info("Preprocessing all textbooks")
    all_textbooks = preprocess_all_pdf_textbooks()
    logging.info("Successfully preprocessed all textbooks")

    logging.info("Building all indexes")
    build_all_vector_indexes(all_textbooks)
    logging.info("Successfully built all indexes")


def run_advanced_prompting_experiments():
    ADVANCED_PROMPTING_STRATEGIES = [
        PromptingStrategy.Basic,
        PromptingStrategy.COT,
        PromptingStrategy.COT_SC,
        PromptingStrategy.TOT
    ]
    for adv_prompting_strategy in ADVANCED_PROMPTING_STRATEGIES:
        for textbook_subfield in TextbookSubfield:
            subfield = textbook_subfield.value
            experiment = Experiment(
                llm=LLM.ChatGPT35,
                embedding_model=None,
                prompting_strategy=adv_prompting_strategy,
                index_config=None,
                evaluation_dataset=EvaluationDataset.TheoremQA
            )
            logging.info(f"Running experiment: {experiment.to_string()}")
            run_theoremqa_experiment(experiment, subfield, False)


def run_rag_experiments():
    RAG_PROMPTING_STRATEGIES = [
        PromptingStrategy.RAG_TOP5_NEARBY200,
        PromptingStrategy.RAG_TOP2_NEARBY500,
        PromptingStrategy.RAG_TOP2_NEARBY200,
        PromptingStrategy.RAG_TOP1_NEARBY500,
        PromptingStrategy.RAG_TOP1_NEARBY200
    ]
    for embedding_model in EMBEDDING_MODELS:
        for rag_prompting_strategy in RAG_PROMPTING_STRATEGIES:
            for textbook_subfield in TextbookSubfield:
                subfield = textbook_subfield.value
                experiment = Experiment(
                    llm=LLM.ChatGPT35,
                    embedding_model=embedding_model,
                    prompting_strategy=rag_prompting_strategy,
                    index_config=IndexConfig(
                        indexing_strategy=IndexingStrategy.Subfield,
                        index_name=subfield
                    ),
                    evaluation_dataset=EvaluationDataset.TheoremQA
                )
                logging.info(f"Running experiment: {experiment.to_string()}")
                run_theoremqa_experiment(experiment, subfield, False)


build_all_indexes()
run_advanced_prompting_experiments()
run_rag_experiments()
