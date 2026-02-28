"""
PDF ingestion pipeline.
Extracts, chunks, embeds and stores PDF content into the vector database.

Enriched embedding strategy:
  - The stored text (texte_fragment) is always the original chunk from the PDF.
  - Each chunk gets TWO embeddings stored as separate rows:
    1. The chunk text as-is (matches direct queries)
    2. The chunk text + retrieval keywords derived from its content category
       (matches indirect/differently-phrased queries)
  - The user always sees the original factual text. The enrichment only
    affects the embedding vectors used for cosine similarity retrieval.
"""
import re
from pathlib import Path
from services.pdf_processor import process_pdf
from services.embedding_service import create_embedding
from database.models import create_embeddings_table, insert_embedding, clear_embeddings_table
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVAL KEYWORDS — derived from chunk content, not fabricated
# ═══════════════════════════════════════════════════════════════════════════
# Each pattern detects a content category from words ALREADY IN the chunk.
# The suffix is appended to the embedding input only (never stored/returned).

_ENRICHMENT_RULES = [
    # (detection pattern on chunk text, keywords to append for embedding)
    (r'\bdosage\b.*\bppm\b|\bppm\b.*\bdosage\b|\bdosage\s+for\b',
     'dosage quantity ppm recommended amount how much'),
    (r'\bstorage\b.*\bshelf\s*life\b|\bdurability\b|\bstore\s+in\b',
     'storage shelf life conditions temperature durability how long'),
    (r'\bpackaging\b|\bpackage\b|\bcarton\s+box\b',
     'packaging package carton box container'),
    (r'\ballergen\b',
     'allergen gluten contains food safety'),
    (r'\bfunction\b.*\bbakery\b|\bfunction\b.*\bbread\b|\bimprove\b.*\bvolume\b',
     'function effect purpose role improve bread'),
    (r'\bactivity\b.*\b[A-Z]{2,}/g\b|\bactivity\b.*\bU/g\b',
     'activity enzyme unit measurement'),
    (r'\bproduct\b.*\bdescription\b|\bsource\b.*\bbacterial\b|\bsource\b.*\bfungal\b',
     'source origin description what is'),
    (r'\bphysical\s+properties\b|\baspect\b.*\bpowder\b|\bmoisture\b',
     'physical properties aspect color powder moisture'),
    (r'\bGMO\b|\bionization\b|\birradiation\b',
     'regulatory GMO ionization status compliance'),
    (r'\bmicrobiology\b|\bsalmonella\b|\bheavy\s+metals\b',
     'food safety microbiology heavy metals specifications'),
    (r'\bboulangerie\b|\bpanification\b',
     'boulangerie panification dosage bakery'),
]


def _get_enrichment_suffix(chunk_text):
    """Detect what category this chunk belongs to and return retrieval keywords.

    Returns keywords derived from the chunk's own content — no fabrication.
    """
    suffixes = []
    for pattern, keywords in _ENRICHMENT_RULES:
        if re.search(pattern, chunk_text, re.IGNORECASE):
            suffixes.append(keywords)
    return ' '.join(suffixes) if suffixes else ''


def ingest_pdfs(pdf_folder):

    if not create_embeddings_table():
        return False

    clear_embeddings_table()

    pdf_files = list(Path(pdf_folder).glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"Aucun fichier PDF trouvé dans {pdf_folder}")
        return False

    total_chunks = 0
    enriched_count = 0

    for pdf_idx, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"Traitement du PDF {pdf_idx}/{len(pdf_files)}: {pdf_path.name}")
        print(f"  [{pdf_idx}/{len(pdf_files)}] {pdf_path.name}")

        chunks = process_pdf(pdf_path)

        if not chunks:
            logger.warning(f"Aucun chunk extrait de {pdf_path.name}")
            continue

        logger.info(f"{len(chunks)} chunks créés pour {pdf_path.name}")

        for chunk_idx, chunk in enumerate(chunks):
            # ── Embedding #1: original chunk text ────────────────
            embedding = create_embedding(chunk)
            if embedding is not None:
                insert_embedding(id_document=pdf_idx, texte_fragment=chunk, vecteur=embedding)
                total_chunks += 1
            else:
                logger.error(f"Échec embedding section {chunk_idx} de {pdf_path.name}")
                continue

            # ── Embedding #2: enriched (if category detected) ────
            suffix = _get_enrichment_suffix(chunk)
            if suffix:
                enriched_text = f"{chunk} {suffix}"
                enriched_emb = create_embedding(enriched_text)
                if enriched_emb is not None:
                    # Same original text stored, different embedding vector
                    insert_embedding(id_document=pdf_idx, texte_fragment=chunk, vecteur=enriched_emb)
                    total_chunks += 1
                    enriched_count += 1

    logger.info(f"Ingestion: {total_chunks} embeddings ({enriched_count} enriched) from {len(pdf_files)} PDFs.")
    print(f"\n  Total: {total_chunks} embeddings ({enriched_count} enriched) from {len(pdf_files)} PDFs")
    return True
