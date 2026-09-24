from database.connection_pg import connect_to_db, close_db_connection
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# The challenge's table (id, id_document, texte_fragment, vecteur) plus
# display / de-duplication columns. vecteur is always the embedding of
# texte_fragment.
_SCHEMA = f"""
    CREATE TABLE IF NOT EXISTS documents (
        id_document INT PRIMARY KEY,
        fichier TEXT
    );
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        id_document INT,
        texte_fragment TEXT,
        vecteur vector({config.EMBEDDING_DIMENSION}),
        texte_original TEXT,
        langue TEXT,
        produit TEXT,
        cle TEXT,
        couvre TEXT[]
    );
"""


def reset_tables():
    """Drop and recreate the tables (ingestion always rebuilds everything)."""
    connection = connect_to_db()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS embeddings; DROP TABLE IF EXISTS documents;")
        cursor.execute(_SCHEMA)
        connection.commit()
        cursor.close()
        logger.info("Tables embeddings et documents recréées.")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la création des tables: {e}")
        return False
    finally:
        close_db_connection(connection)


def insert_fragments(documents, fragments, vectors):
    connection = connect_to_db()
    if not connection:
        return False
    try:
        cursor = connection.cursor()
        cursor.executemany("INSERT INTO documents (id_document, fichier) VALUES (%s, %s)", documents)
        cursor.executemany("""
            INSERT INTO embeddings (id_document, texte_fragment, vecteur, texte_original,
                                    langue, produit, cle, couvre)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, [(f['id_document'], f['texte'], v.tolist(), f['texte_original'], f['langue'],
               f['produit'], f['cle'], f['couvre']) for f, v in zip(fragments, vectors)])
        connection.commit()
        cursor.close()
        logger.info(f"{len(fragments)} fragments insérés.")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion des fragments: {e}")
        return False
    finally:
        close_db_connection(connection)


def search_cosine_similarity(query_vector, top_k=config.TOP_K):
    """Rows ranked by cosine similarity with query_vector (pgvector <=> is the
    cosine distance, so similarity = 1 - distance)."""
    connection = connect_to_db()
    if not connection:
        return []
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT e.id, e.id_document, d.fichier, e.texte_fragment, e.texte_original,
                   e.langue, e.produit, e.cle, e.couvre,
                   1 - (e.vecteur <=> %s::vector) AS score
            FROM embeddings e JOIN documents d USING (id_document)
            ORDER BY e.vecteur <=> %s::vector
            LIMIT %s
        """, (query_vector.tolist(), query_vector.tolist(), top_k))
        columns = [c.name for c in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
        cursor.close()
        return rows
    except Exception as e:
        logger.error(f"Erreur lors de la recherche de similarité: {e}")
        return []
    finally:
        close_db_connection(connection)
