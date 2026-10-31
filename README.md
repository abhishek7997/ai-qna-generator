# AI Q&A Generator

Reads PDFs from the `data/` folder, generates questions from them with an LLM, and answers each question using the PDFs' own content (retrieval with FAISS).

## How it works

1. Extract text from every PDF in `data/`.
2. Split it into chunks.
3. Generate questions chunk by chunk (LangChain refine chain).
4. Embed the chunks into a FAISS index.
5. For each question, retrieve the 4 closest chunks and have the LLM answer from them.
6. Print the question and answer pairs.

## Setup

```
uv sync
```

Create a `.env` file:

```
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-5.5
```

## Run

Put your PDFs in `data/`, then:

```
uv run main.py
```

## Using a local model (LM Studio, no OpenAI key)

Load a chat model and an embedding model in LM Studio, start its server, then set:

```
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL=openai/gpt-oss-20b
OPENAI_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
QA_CHUNK_TOKENS=3000
```
