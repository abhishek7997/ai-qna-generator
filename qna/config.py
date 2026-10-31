import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5.5')
EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

TOKEN_ENCODING = 'o200k_base'
QUESTION_CHUNK_TOKENS = int(os.getenv('QA_CHUNK_TOKENS', '10000'))
QUESTION_CHUNK_OVERLAP = 200
ANSWER_CHUNK_TOKENS = 1000
ANSWER_CHUNK_OVERLAP = 100

RETRIEVAL_K = 4
QUESTION_TEMPERATURE = 0.3
ANSWER_TEMPERATURE = 0.1
