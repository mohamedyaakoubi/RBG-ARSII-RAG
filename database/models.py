from database.connection_pg import connect_to_db , close_db_connection
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def create_embeddings_table():
    connection = connect_to_db()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    id_document INT,
                    texte_fragment TEXT,
                    vecteur vector({config.EMBEDDING_DIMENSION})
                )
            """)
        connection.commit()
        cursor.close()
        close_db_connection(connection)
        logger.info("Table embeddings créée avec succès.")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la création de la table embeddings: {e}")
        close_db_connection(connection)
        return False

def clear_embeddings_table():
    """Vider la table embeddings avant réingestion"""
    connection = connect_to_db()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("TRUNCATE TABLE embeddings RESTART IDENTITY")
        connection.commit()
        cursor.close()
        close_db_connection(connection)
        logger.info("Table embeddings vidée.")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du vidage de la table: {e}")
        close_db_connection(connection)
        return False
    
def insert_embedding(id_document, texte_fragment, vecteur):
    connection = connect_to_db()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO embeddings (id_document, texte_fragment, vecteur)
            VALUES (%s, %s, %s)
        """, (id_document, texte_fragment, vecteur.tolist()))
        connection.commit()
        cursor.close()
        close_db_connection(connection)
        logger.info(f"Embedding inséré pour le document {id_document}.")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion de l'embedding: {e}")
        close_db_connection(connection)
        return False

def search_cosine_similarity(query_vector, top_k=config.TOP_K):
    connection = connect_to_db()
    if not connection:
        return []

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT 
                id_document,
                texte_fragment,
                1 - (vecteur <=> %s::vector) AS score
            FROM embeddings
            ORDER BY vecteur <=> %s::vector
            LIMIT %s
        """, (query_vector.tolist(), query_vector.tolist(), top_k))

        results = cursor.fetchall()

        cursor.close()
        close_db_connection(connection)

        logger.info("Recherche de similarité cosinus effectuée avec succès.")

        return results

    except Exception as e:
        logger.error(f"Erreur lors de la recherche de similarité: {e}")
        close_db_connection(connection)
        return []
