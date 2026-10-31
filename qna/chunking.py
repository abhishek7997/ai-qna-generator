from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

from . import config


def _splitter(chunk_size: int, chunk_overlap: int) -> TokenTextSplitter:
    return TokenTextSplitter(
        encoding_name=config.TOKEN_ENCODING, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def chunk_for_questions(pdfs: list[list[Document]]) -> list[list[Document]]:
    """Large chunks, one list per PDF, used for question generation."""
    splitter = _splitter(config.QUESTION_CHUNK_TOKENS, config.QUESTION_CHUNK_OVERLAP)
    # merge pages first, otherwise every page becomes its own chunk (one LLM call per page)
    merged = [
        Document(page_content="\n\n".join(p.page_content for p in pages), metadata={"source": pages[0].metadata["source"]})
        for pages in pdfs
    ]
    return [splitter.split_documents([doc]) for doc in merged]


def chunk_for_answers(question_chunks: list[list[Document]]) -> list[Document]:
    """Small chunks, flattened across all PDFs, used for retrieval."""
    splitter = _splitter(config.ANSWER_CHUNK_TOKENS, config.ANSWER_CHUNK_OVERLAP)
    return [chunk for doc_chunks in question_chunks for chunk in splitter.split_documents(doc_chunks)]
