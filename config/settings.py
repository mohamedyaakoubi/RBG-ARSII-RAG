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
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 80
    EMBEDDING_DIMENSION = 384

config = Config()