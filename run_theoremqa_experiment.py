from src.indexing.strategy import IndexingStrategy
from src.indexing.vector import load_index
from src.inference.chatgpt import prompt_llm
from src.prompting.rag import build_rag_prompt
from src.retrieval.vector import retrieve_top_k_documents


def run_experiment_for_one_question(
    indexing_strategy: IndexingStrategy, index_name: str, question: str, k: int = 2
):
    index = load_index(index_name, indexing_strategy)
    documents = retrieve_top_k_documents(index, question, k)
    prompt = build_rag_prompt(question, documents)
    llm_response = prompt_llm(prompt)

    # TODO: Check answer


# Example question
indexing_strategy = IndexingStrategy.Subject
index_name = "algebra"
question = "formula to find the slope of a line when we have two points"
run_experiment_for_one_question()
