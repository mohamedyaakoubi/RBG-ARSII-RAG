# RAG — Module de Recherche Sémantique pour la Formulation en Boulangerie

A **Retrieval-Augmented Generation** system that answers technical questions about bakery enzyme products (BVZyme) in **both French and English**, powered by vector similarity search over real PDF technical data sheets.

---

## Context

This project was developed for the **ARSII RAG Challenge** organized by **STE AGRO MELANGE TECHNOLOGIE — Rose Blanche Group**.

**Problem:** In a document base containing a large volume of technical information (reports, procedures, recommendations, use cases…), users struggle to quickly identify the passages that are actually relevant to their question. The goal is to develop an intelligent module that assists the user by automatically retrieving the most relevant fragments from a question formulated in natural language, using **semantic proximity** rather than simple keyword matching.

**Objective:** Build a semantic search module (RAG) over a vector database of bakery ingredient technical data sheets (enzymes, dough improvers, oxidizing agents, etc.). The system must:

1. Receive a user question in natural language
2. Generate its semantic embedding
3. Compare it against stored fragment embeddings using **cosine similarity**
4. Rank results by descending relevance
5. Return the **top 3 most relevant fragments** with their text and similarity score

**Constraints imposed by the challenge:**

| Parameter | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector dimension | 384 |
| Similarity metric | Cosine similarity |
| Results returned | Top K = 3 |
| Language | Python |
| Database | PostgreSQL + pgvector |

**Dataset:** 35 PDF technical data sheets (34 English TDS + 1 French) covering BVZyme bakery enzymes (alpha-amylase, xylanase, lipase, transglutaminase, glucose oxidase, maltogenic amylase, ascorbic acid).

**Example question provided by the challenge:**
> *Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?*

---

## Architecture

```
┌────────────┐    ┌──────────────┐    ┌───────────────────┐
│  PDF Corpus │───▶│ PDF Processor │───▶│  PostgreSQL +     │
│  (35 TDS)  │    │  (pdfplumber)│    │  pgvector (384d)  │
└────────────┘    └──────────────┘    └───────┬───────────┘
                                              │
┌────────────┐    ┌──────────────┐            │
│  User Query │───▶│ Search Svc   │◀───────────┘
│  (FR / EN) │    │ (bilingual)  │
└────────────┘    └──────────────┘
```

| Module | Role |
|---|---|
| `services/pdf_processor.py` | pdfplumber extraction + entity-centric chunking |
| `services/embedding_service.py` | Sentence embeddings via `all-MiniLM-L6-v2` (384d) |
| `services/search_service.py` | Bilingual search with query translation + multi-entity decomposition |
| `services/ingestion_data.py` | Ingestion pipeline with enriched dual embeddings |
| `database/models.py` | PostgreSQL / pgvector operations (cosine similarity) |
| `main.py` | Interactive CLI interface |
| `app.py` | Streamlit web UI |

---

## Features

- **pdfplumber extraction** — Clean text and structured table extraction from PDFs, eliminating noise and preserving document structure.
- **Entity-centric chunking** — Each PDF is split into purpose-built chunks: identity, function, dosage (bilingual), storage, allergens, food safety, physical properties, packaging, and regulatory status. Raw section chunks are also indexed for unexpected query coverage.
- **Enriched dual embeddings** — Each chunk gets two embeddings stored as separate DB rows: (1) the original text as-is, and (2) the same text enriched with content-detected retrieval keywords. The stored text is always the original — enrichment only affects the embedding vector used for similarity matching.
- **Bilingual search** — French queries are translated at search time using a domain-specific vocabulary (~100 bakery/enzyme terms). Both the original French and the English translation are searched, and the best result per chunk is kept.
- **Multi-entity query decomposition** — Questions mentioning multiple enzymes (e.g. *"alpha-amylase, xylanase et acide ascorbique"*) are split into sub-queries, each searched independently, then merged with entity-aware ranking.
- **Bilingual dosage labels** — Each enzyme's dosage chunk is indexed with both English and French keywords to maximize retrieval from either language.

---

## Benchmark Results

Corpus: 35 PDFs → 924 chunks → 1,635 embeddings (with enrichment) in database

### Competition Query (Multi-Entity)

> *"Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?"*

| Rank | Score | Enzyme Covered | Retrieved Fragment |
|:----:|:-----:|:---:|---|
| R1 | **0.9330** | alpha-amylase | `Dosage alpha-amylase (BVZyme AF330) boulangerie panification : 2-10 ppm` |
| R2 | **0.9131** | xylanase | `Dosage xylanase (BVZyme HCB708) boulangerie panification : 5-30 ppm` |
| R3 | **0.8635** | ascorbic acid | `Dosage acide ascorbique (vitamine C, E300) boulangerie panification : 50-75 ppm` |

All 3 enzymes covered in top-3 results.

### Single-Entity Queries

| Query | Lang | Score | Top Fragment |
|---|:---:|:---:|---|
| Quel dosage de lipase pour la panification ? | FR | **0.7758** | `Dosage lipase (L65pdf) boulangerie panification : 5-50 ppm` |
| Quel dosage de xylanase en boulangerie ? | FR | **0.8178** | `BVZyme HCB709 (xylanase) dosage for bakery: 5-20 ppm` |
| À quoi sert l'acide ascorbique en boulangerie ? | FR | **0.7395** | `À quoi sert l'acide ascorbique en boulangerie ? L'acide ascorbique (vitamine C) est un additif…` |
| Quel est l'effet de la xylanase sur le volume du pain ? | FR | **0.6917** | `BVZyme HCB710 (xylanase): Improve loaf volume, enhance stability, increase elasticity` |
| What is the recommended dosage of alpha-amylase for bread? | EN | **0.7786** | `BVZyme AF330 (alpha-amylase) dosage for bakery: 2-10 ppm` |

### Unseen Queries (never anticipated during development)

| Query | Lang | Score | Correct? | Top Fragment |
|---|:---:|:---:|:---:|---|
| What are the storage conditions for BVZyme AF110? | EN | 0.6818 | ✅ | `Storage Store in a cool, dry place (below 20°C)` |
| Quelle est la dose recommandée de transglutaminase ? | FR | 0.8209 | ✅ | `Dosage transglutaminase (BVZyme TG MAX63) : 5-25 ppm` |
| Does BVZyme contain allergens? | EN | 0.7649 | ✅ | `Allergens In compliance with the list of major…` |
| What is the optimal pH for xylanase activity? | EN | 0.6188 | ⚠️ | `Suggested Optimum…` (pH value not explicitly named in PDFs) |
| How does lipase improve bread texture? | EN | 0.6718 | ✅ | `Function, fine regular crumb structure, improve stability and tolerance` |
| What is the shelf life of BVZyme enzymes? | EN | 0.8143 | ✅ | `Storage — Date of minimum durability: 24 months` |
| Quelle est l'activité enzymatique de l'alpha-amylase ? | FR | 0.7608 | ✅ | `Activity 85000 SKB/g` |
| What is the microbial source of BVZyme xylanase? | EN | 0.7917 | ✅ | `Bacterial xylanase produced by fermenting a selected unique strain…` |
| How to combine alpha-amylase and xylanase for bread? | EN | 0.9330 | ⚠️ | `Dosage alpha-amylase 2-10 ppm` (returns dosage, not combination guidance) |
| What packaging is used for BVZyme products? | EN | 0.7175 | ✅ | `Packaging 25 kg paper bag with PE liner` |

**Average score across all 16 queries: 0.7695**

> ⚠️ "pH xylanase" and "combine enzymes" return high-similarity results but the PDFs do not explicitly contain this information. The system correctly retrieves the closest available content — this is a data coverage limitation, not a retrieval failure.

---

## Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose

### 1. Clone and install dependencies

```bash
git clone https://github.com/mohamedyaakoubi/RBG-ARSII-RAG.git
cd rag
pip install -r requirements.txt
```

### 2. Start the PostgreSQL + pgvector database

```bash
docker-compose up -d
```

This builds a PostgreSQL 16 image with pgvector and starts it on `localhost:5432`.

### 3. Configure environment

Create a `.env` file (or use the provided one):

```env
DB_USER=rag
DB_PASSWORD=ragpassword
DB_NAME=ragdb
DB_HOST=localhost
PDF_FOLDER=data_pdf
```

### 4. Add your PDF files

Place the BVZyme TDS PDF files in the `data_pdf/` directory.

### 5. Run the application

**Streamlit UI (recommended):**

```bash
streamlit run app.py
```

**CLI mode:**

```bash
python main.py
```

From the interactive menu:
1. **Option 1** — Ingest PDFs (extracts, chunks, embeds, and stores all PDFs)
2. **Option 2** — Search (enter a question in French or English)

---

## Project Structure

```
rag/
├── app.py                   # Streamlit web UI
├── main.py                  # CLI entry point
├── config/
│   └── settings.py          # Configuration (DB, embedding, chunking params)
├── database/
│   ├── connection_pg.py     # PostgreSQL connection management
│   └── models.py            # Table creation, insert, cosine similarity search
├── services/
│   ├── embedding_service.py # Sentence-transformers embedding
│   ├── ingestion_data.py    # Ingestion pipeline with enriched dual embeddings
│   ├── pdf_processor.py     # pdfplumber extraction + entity-centric chunking
│   └── search_service.py    # Bilingual search + query decomposition
├── utils/
│   └── logger.py            # Logging configuration
├── data_pdf/                # PDF corpus (not tracked in git)
├── docker-compose.yml       # PostgreSQL + pgvector container
├── Dockerfile               # Custom PostgreSQL image with pgvector
├── init.sql                 # pgvector extension initialization
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## Technical Details

### PDF Processing & Chunking

PDFs are processed with **pdfplumber** for clean text extraction and structured table parsing. Each English TDS is split into entity-centric chunks:

- **Identity** — product description, source, activity
- **Function** — application and function in bakery
- **Dosage** — general dosage + sub-application dosages, with bilingual (EN+FR) labels
- **Storage** — conditions and shelf life
- **Allergens** — allergen compliance statements
- **Food safety** — microbiology, heavy metals (extracted from tables)
- **Physical properties** — organoleptic and physicochemical data
- **Packaging** — packaging information
- **Regulatory** — GMO and ionization status

Raw section chunks are also indexed alongside the structured chunks to handle unexpected query formulations.

### Enriched Dual Embeddings

Each chunk is stored with **two embedding vectors**:

1. **Original embedding** — the chunk text as-is, embedded directly
2. **Enriched embedding** — the chunk text + automatically detected retrieval keywords, embedded together

The enrichment keywords are derived from the chunk's own content using pattern detection (e.g., a chunk containing "ppm" and "dosage" gets keywords like `dosage quantity ppm recommended amount`). **No data is fabricated** — the keywords are synonyms and related terms for content that already exists in the chunk. The stored `texte_fragment` is always the original PDF text.

### Cross-Lingual Search Strategy

The imposed embedding model (`all-MiniLM-L6-v2`) is English-optimized. French queries naturally score lower against English chunks (~0.40 vs ~0.71 for equivalent queries). To bridge this gap without changing the model:

1. **Query-time translation** — A domain-specific French→English vocabulary (~100 terms covering bakery, enzymes, and technical language) translates French queries to English before embedding.
2. **Dual search** — Both the original French query and the translated English query are embedded and searched. Results are merged by keeping the best cosine similarity score per chunk.
3. **No corpus modification** — The stored chunks are the original PDF text. Only the query is translated at search time.

### Multi-Entity Decomposition

When a query mentions multiple enzymes (detected via regex patterns), it is decomposed into per-enzyme sub-queries. Each sub-query is searched bilingually, and results are merged with entity-aware ranking to ensure coverage of all mentioned enzymes in the top-3 results.

---

## Observations on Challenge Design

The challenge asks participants to build a semantic search module over a pre-existing vector database of bakery ingredient technical data sheets. The constraints are deliberately tight: a fixed embedding model (`all-MiniLM-L6-v2`, 384d), cosine similarity, top K=3, Python, and PostgreSQL with pgvector. These constraints create several inherent tensions that limit what any retrieval system can achieve:

### Conflicts and Limitations

The challenge specification contains several tensions that limit the achievable retrieval quality:

1. **Bilingual corpus, monolingual model.** The dataset includes 34 English TDS and 1 French document, but `all-MiniLM-L6-v2` is English-optimized. French queries against English chunks score ~0.40–0.50 by default. Our query-time translation bridges this gap, but a multilingual embedding model would be more appropriate for a bilingual corpus.

2. **Top K = 3 with a broad corpus.** With 35 PDFs producing 900+ chunks, only 3 retrieval slots means there is almost zero margin for error. Queries that span multiple topics (e.g., "combine alpha-amylase and xylanase") cannot be adequately answered in 3 results when the information is spread across many documents.

3. **384-dimensional embedding space.** The `all-MiniLM-L6-v2` model (22M parameters, 384 dimensions) provides good general-purpose embeddings but lacks the capacity for fine-grained semantic distinctions in domain-specific terminology. Larger models (768d+) would capture deeper relationships between bakery/enzyme concepts.

4. **Pre-chunked assumption vs. quality chunking.** The challenge states "les fiches ont déjà été découpées en fragments" — implying chunking is already done. However, the quality of chunking has by far the largest impact on retrieval accuracy. Naive fixed-size chunking destroys document structure and loses context. Our entity-centric chunking improved the average retrieval score significantly over arbitrary character-window approaches.

5. **Missing information in source PDFs.** Some queries ask for information that genuinely does not exist in the provided PDFs (e.g., optimal pH values, enzyme combination guidance). No retrieval system — regardless of sophistication — can return information that isn't in the corpus. The system correctly returns the closest available content, but the similarity scores can be misleading in these cases.

---

## License

Academic project — ARSII RAG Challenge, Rose Blanche Group.
