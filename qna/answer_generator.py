from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from . import config
from .prompts import ANSWER_TEMPLATE


def build_retriever(answer_chunks: list[Document]):
    # check_embedding_ctx_length=False sends raw text, which OpenAI-compatible local servers (LM Studio, Ollama) require
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, check_embedding_ctx_length=False)
    vec_store = FAISS.from_documents(answer_chunks, embeddings)
    return vec_store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})


def answer_questions(questions: list[str], answer_chunks: list[Document]) -> list[dict]:
    """Answer each question from its retrieved chunks; returns [{"question", "answer"}, ...]."""
    retriever = build_retriever(answer_chunks)
    llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=config.ANSWER_TEMPERATURE)

    qa_pairs = []
    for question in questions:
        context = "\n\n".join(doc.page_content for doc in retriever.invoke(question))
        answer = llm.invoke(ANSWER_TEMPLATE.format(context=context, question=question)).content
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs
