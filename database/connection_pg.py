import psycopg2
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def connect_to_db():
    try:
        connection = psycopg2.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            dbname=config.DB_NAME
        )
        logger.info("Connexion à la base de données réussie.")
        return connection
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à la base de données: {e}")
        return None
def close_db_connection(connection):
    if connection:
        connection.close()
        logger.info("Connexion à la base de données fermée.")
        