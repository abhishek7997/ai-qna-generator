from qna import config
from qna.answer_generator import answer_questions
from qna.chunking import chunk_for_answers, chunk_for_questions
from qna.pdf_loader import load_pdfs
from qna.question_generator import generate_questions


def main():
    pdfs = load_pdfs(config.DATA_DIR)
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.DATA_DIR}")

    question_chunks = chunk_for_questions(pdfs)
    answer_chunks = chunk_for_answers(question_chunks)

    questions = generate_questions(question_chunks)
    qa_pairs = answer_questions(questions, answer_chunks)

    for pair in qa_pairs:
        print(f"Q: {pair['question']}\nA: {pair['answer']}\n")


if __name__ == "__main__":
    main()
