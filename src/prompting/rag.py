from typing import List


def build_rag_prompt(question: str, sources: List[str]) -> str:
    prompt_sources = "\n".join(
        f"SOURCE {i+1}:\n----\n{source.strip()}\n"
        for i, source in enumerate(sources)
    )
    return (
        "Answer the following question, and use the following sources if helpful.\n\n"
        f"QUESTION:\n----\n{question}\n\n"
        f"{prompt_sources}\n"
        "----\nReturn only the answer below, without any terminating punctuation or newline\nANSWER: "
    )
