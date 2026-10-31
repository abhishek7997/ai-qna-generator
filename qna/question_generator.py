import re

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from . import config
from .prompts import QUESTION_PROMPT, REFINE_PROMPT


def _extract_questions(text: str) -> list[str]:
    """Keep only lines that are questions, minus list numbering and bullets."""
    questions = []
    for line in text.splitlines():
        line = re.sub(r'^\s*(\d+[.)]|[-*])\s*', '', line).strip()
        if line.endswith("?"):
            questions.append(line)
    return questions


def generate_questions(question_chunks: list[list[Document]], verbose: bool = True) -> list[str]:
    """Run a refine chain over each PDF's chunks and collect the questions."""
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=config.QUESTION_TEMPERATURE)
    chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        verbose=verbose,
        question_prompt=QUESTION_PROMPT,
        refine_prompt=REFINE_PROMPT,
    )
    questions = []
    for doc_chunks in question_chunks:
        output = chain.invoke({"input_documents": doc_chunks})["output_text"]
        questions.extend(_extract_questions(output))
    return questions
