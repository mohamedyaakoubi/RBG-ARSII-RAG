"""
PDF ingestion pipeline: PDFs → fragments → embeddings → PostgreSQL.

- Fragments are text from the PDFs (see services/pdf_processor.py).
- French documents are also indexed in English, the embedding model's
  language (guarded translation, see services/translation.py).
- Each row stores one fragment and the embedding of exactly that text.
"""
from services.embedding_service import embed_texts, model
from services.pdf_processor import build_fragments
from services.translation import translate_fragments
from database.models import reset_tables, insert_fragments
from utils.logger import setup_logger

logger = setup_logger(__name__)


def ingest_pdfs(pdf_folder):
    documents, fragments = build_fragments(pdf_folder, model.tokenizer.vocab)
    if not documents:
        logger.warning(f"Aucun fichier PDF trouvé dans {pdf_folder}")
        return False
    fragments += translate_fragments([f for f in fragments if f['langue'] == 'fr'])
    if not reset_tables():
        return False
    vectors = embed_texts([f['texte'] for f in fragments])
    if not insert_fragments(documents, fragments, vectors):
        return False
    print(f"\n  Total: {len(fragments)} fragments indexés depuis {len(documents)} PDFs")
    return True
