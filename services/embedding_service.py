from sentence_transformers import SentenceTransformer
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

logger.info("Initialisation du modèle d'embedding...")
model= SentenceTransformer('all-MiniLM-L6-v2')
def create_embedding(text):
    try:
        embedding = model.encode(text)
        logger.info("Embedding créé avec succès.")
        return embedding
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'embedding: {e}")
        return None
    