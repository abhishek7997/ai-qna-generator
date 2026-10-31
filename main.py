import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from pypdf import PdfReader

load_dotenv()

OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5.5')

# 1. Data loading

DATA_DIR = Path.cwd() / 'data'
loaded_pdf = []

for file_path in DATA_DIR.glob("*.pdf"):
    reader = PdfReader(file_path)
    loaded_pdf.append([
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": str(file_path), "page": i},
        )
        for i, page in enumerate(reader.pages)
    ])

# print(loaded_pdf)

# 2. Chunking
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(encoding_name='o200k_base', chunk_size = int(os.getenv('QA_CHUNK_TOKENS', '10000')), chunk_overlap = 200)

chunked_texts = []

for pdf_texts in loaded_pdf:
    chunked_texts.append(text_splitter.split_documents(pdf_texts))

# print(chunked_texts[0])
print(type(chunked_texts[0][0]))

# smaller chunks for retrieval, flattened into a single list of Documents
text_splitter_ans = TokenTextSplitter(encoding_name='o200k_base', chunk_size = 1000, chunk_overlap = 100)
doc_ans_gen = []
for chunked_text in chunked_texts:
    doc_ans_gen.extend(text_splitter_ans.split_documents(chunked_text))

# LLM integration
from langchain_openai import ChatOpenAI

llm_pipeline = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)

PROMPT_TEMPLATE = """
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:

----------
{text}
----------

Create questions that will prepare the coders or programmers for the coding tests.
Make sure not to lost any important information.
QUESTIONS:
"""

from langchain_core.prompts import PromptTemplate

PROMPT_QUESTIONS = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=['text'])

REFINE_TEMPLATE = """
You are an expert at creating practice questions based on coding materials and documentation.
Your goal is to help a coder or programmer prepare for their exam and coding tests.
We have received some practice questions to a certain extent:
{existing_answer}

We have the option to refine the existing questions or add new ones (only if necessary)
with some more context below:
----------
{text}
----------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    template=REFINE_TEMPLATE,
    input_variables=['existing_answer', 'text'],
)

from langchain_classic.chains.summarize import load_summarize_chain

ques_gen_chain = load_summarize_chain(llm = llm_pipeline, chain_type='refine', verbose=True, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)

if not chunked_texts:
    raise SystemExit(f"No PDFs found in {DATA_DIR}")

questions = []
for doc_chunks in chunked_texts:
    ques = ques_gen_chain.invoke({"input_documents": doc_chunks})["output_text"]
    for line in ques.splitlines():
        line = re.sub(r'^\s*(\d+[.)]|[-*])\s*', '', line).strip()
        if line.endswith("?"):
            questions.append(line)

# Embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'), check_embedding_ctx_length=False)

from langchain_community.vectorstores import FAISS

vec_store = FAISS.from_documents(doc_ans_gen, embeddings)
llm_answer_gen = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)

retriever = vec_store.as_retriever(search_kwargs={"k": 4})

qa_pairs = []
for q in questions:
    context = "\n\n".join(d.page_content for d in retriever.invoke(q))
    answer = llm_answer_gen.invoke(
        f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {q}"
    ).content
    qa_pairs.append({"question": q, "answer": answer})

for pair in qa_pairs:
    print(f"Q: {pair['question']}\nA: {pair['answer']}\n")
