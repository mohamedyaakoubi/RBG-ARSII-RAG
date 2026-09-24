import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")
    TOP_K = 3
    PDF_FOLDER = os.getenv("PDF_FOLDER")
    EMBEDDING_DIMENSION = 384
    # "transparent": French questions are also embedded in English, questions
    # naming several products are split per product, and a question naming a
    # product by its code is answered from its sheet (formulation and
    # restriction are shown). "strict": the question is embedded exactly as typed.
    SEARCH_MODE = os.getenv("SEARCH_MODE", "transparent")
    # rows fetched from pgvector before duplicate contents are collapsed
    CANDIDATES = int(os.getenv("CANDIDATES", "60"))
    TRANSLATION_CACHE = os.getenv("TRANSLATION_CACHE", "cache/translations.json")

config = Config()