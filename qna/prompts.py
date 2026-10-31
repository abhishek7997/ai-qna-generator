from langchain_core.prompts import PromptTemplate

QUESTION_TEMPLATE = """
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

ANSWER_TEMPLATE = """Answer the question using only the context below.

Context:
{context}

Question: {question}"""

QUESTION_PROMPT = PromptTemplate(template=QUESTION_TEMPLATE, input_variables=['text'])
REFINE_PROMPT = PromptTemplate(template=REFINE_TEMPLATE, input_variables=['existing_answer', 'text'])
