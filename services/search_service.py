import re
from database.models import search_cosine_similarity
from services.embedding_service import create_embedding
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# FRENCH → ENGLISH QUERY TRANSLATION (keyword-based, no external API)
# ═══════════════════════════════════════════════════════════════════════════
# The embedding model (all-MiniLM-L6-v2) is English-optimized. French queries
# underperform against English chunks (~0.40 vs ~0.71 for the same query in
# English). Query-time translation bridges this gap — a standard cross-lingual
# information retrieval technique.

_FR_TO_EN = [
    # ── Multi-word phrases FIRST (matched before single words) ───────
    (r"\bacide\s+ascorbique\b",        "ascorbic acid"),
    (r"\bvitamine\s*C\b",             "vitamin C ascorbic acid"),
    (r"\bamylase\s+maltog[ée]nique\b", "maltogenic amylase"),
    (r"\bglucose\s+oxy?dase\b",       "glucose oxidase"),
    (r"\bdur[ée]e\s+de\s+conservation\b", "shelf life"),
    (r"\bs[ée]curit[ée]\s+alimentaire\b", "food safety"),
    (r"\bconditions?\s+de\s+stockage\b", "storage conditions"),
    (r"\bactivit[ée]\s+enzymatique\b", "enzyme activity"),
    (r"\bquantit[ée]s?\s+recommand[ée]e?s?\b", "recommended dosage"),
    (r"\ba\s+quoi\s+sert\b",          "what is the purpose of"),
    (r"\bà\s+quoi\s+sert\b",          "what is the purpose of"),
    (r"\bquel(?:le)?s?\s+(?:est|sont)\b", "what is"),
    (r"\butilis[ée]e?\s+(?:en|pour)\b", "used for"),
    (r"\bam[ée]liorant\s+de\b",       "improver for"),
    (r"\bde\s+la\b",                   "of"),
    (r"\bsur\s+(?:la\s+)?(?:le\s+)?", "on "),
    (r"\bpour\s+(?:la\s+)?(?:le\s+)?", "for "),
    # ── Bakery domain ────────────────────────────────────────────────
    (r"\bpanification\b",              "bread making"),
    (r"\bboulangerie\b",               "bakery"),
    (r"\bpain\b",                      "bread"),
    (r"\bp[âa]te\b",                   "dough"),
    (r"\bfarine\b",                    "flour"),
    (r"\blevure\b",                    "yeast"),
    (r"\blevain\b",                    "sourdough"),
    (r"\bmie\b",                       "crumb"),
    (r"\bcro[ûu]te\b",                "crust"),
    (r"\bcuisson\b",                   "baking"),
    (r"\bfermentation\b",             "fermentation"),
    (r"\bp[ée]trissage\b",            "kneading"),
    (r"\bp[ée]tr[iy]\w*\b",           "kneading"),
    (r"\bgluten\b",                    "gluten"),
    (r"\bsouplesse\b",                 "softness"),
    (r"\bmoelleux\b",                  "softness"),
    (r"\btexture\b",                   "texture"),
    (r"\b[ée]lasticit[ée]\b",         "elasticity"),
    (r"\bextensibilit[ée]\b",         "extensibility"),
    (r"\btol[ée]rance\b",             "tolerance"),
    (r"\bmachinabilit[ée]\b",         "dough handling"),
    (r"\bstabilit[ée]\b",             "stability"),
    (r"\bvolume\b",                    "volume"),
    (r"\bforce\b",                     "strength"),
    (r"\bgaz\b",                       "gas"),
    (r"\br[ée]tention\b",             "retention"),
    (r"\bfra[îi]cheur\b",             "freshness"),
    (r"\bconservation\b",             "shelf life"),
    # ── Query verbs / question words ─────────────────────────────────
    (r"\bcomment\b",                   "how"),
    (r"\bcombien\b",                   "how much"),
    (r"\bpourquoi\b",                  "why"),
    (r"\butilis[ée]r?\b",             "use"),
    (r"\bfonction\b",                  "function"),
    (r"\br[ôo]le\b",                   "role function"),
    (r"\beffet\b",                     "effect"),
    (r"\bam[ée]lio\w+\b",             "improve"),
    (r"\brenforc\w+\b",               "strengthen"),
    (r"\baugment\w+\b",               "increase"),
    (r"\boptimi[sz]\w+\b",            "optimize"),
    (r"\brecommand[ée]e?s?\b",        "recommended"),
    # ── Technical / safety ───────────────────────────────────────────
    (r"\bdosage\b",                    "dosage"),
    (r"\bdose\b",                      "dosage"),
    (r"\bstockage\b",                  "storage"),
    (r"\btemp[ée]rature\b",           "temperature"),
    (r"\ballerg[èe]ne\w*\b",          "allergen"),
    (r"\bemballage\b",                 "packaging"),
    (r"\bactivit[ée]\b",              "activity"),
    (r"\bsp[ée]cification\w*\b",      "specification"),
    (r"\bmicrobiolog\w+\b",           "microbiology"),
    (r"\bOGM\b",                       "GMO"),
    # ── Connectors (last — clean up remaining French articles) ───────
    (r"\bdu\b",                        "of"),
    (r"\bdes\b",                       "of"),
    (r"\bde\b",                        "of"),
    (r"\bet\b",                        "and"),
    (r"\ble\b",                        "the"),
    (r"\bla\b",                        "the"),
    (r"\bun\b",                        "a"),
    (r"\bune\b",                       "a"),
    (r"\bquel\w*\b",                   "what"),
    (r"\ben\b",                        "in"),
    (r"\best\b",                       "is"),
    (r"\bles\b",                       "the"),
    (r"\bdans\b",                      "in"),
    (r"\bm[êe]me\b",                  "same"),
    (r"\bcombiner\b",                  "combine"),
    (r"\bpeut-on\b",                   "can we"),
    (r"\bformulation\b",               "formulation"),
    (r"\bsont\b",                      "are"),
]


def _is_french(text):
    """Quick heuristic: does the text contain distinctly French words?
    Uses words that are exclusively French — avoids 'dosage', 'acide' etc.
    that might appear in English technical text too.
    """
    # Words that are unambiguously French (never appear in English TDS queries)
    strong_markers = [
        r'\bquel(?:le)?s?\b', r'\bquoi\b', r'\bpanification\b',
        r'\bboulangerie\b', r'\bp[âa]te\b', r'\bfarine\b',
        r'\butiliser\b', r'\bam[ée]lio\w+\b', r'\brenforc\w+\b',
        r'\bstockage\b', r'\brecommand[ée]\w*\b', r'\benzymatique\b',
        r'\bpourquoi\b', r'\bcomment\b', r'\bcombien\b',
        r'\bà quoi sert\b', r'\ba quoi sert\b',
        r'\bsont\b', r'\bdans\b', r'\bavec\b',
        r'\beffet\b', r'\br[ôo]le\b', r'\bdans\b',
        r'\bpeut\b', r'\bm[êe]me\b', r'\bformulation\b',
    ]
    score = sum(1 for p in strong_markers if re.search(p, text, re.I))
    # Need at least 2 strong markers to confirm French
    return score >= 2


def _strip_french_elisions(text):
    """Split French elided articles: l'effet → l effet, d'alpha → d alpha, etc."""
    # Handle l', d', n', s', qu', j', c' — common French elisions
    # Include ASCII apostrophe, right single quote (\u2019), backtick-style quote
    return re.sub(r"\b([lLdDnNsS]|[qQ]u|[jJ]|[cC])[\u0027\u2019\u2018`\u00B4]([\w])", r"\1 \2", text)


def _translate_fr_to_en(question):
    """Translate a French query to English using keyword substitution.

    Not a full MT system — just maps French bakery/enzyme vocabulary to English
    so the embedding model can match against English PDF chunks.
    """
    # Step 1: Split elisions so "l'effet" → "l effet", "d'alpha" → "d alpha"
    result = _strip_french_elisions(question)
    # Step 2: Apply vocabulary mappings
    for pattern, replacement in _FR_TO_EN:
        result = re.sub(pattern, replacement, result, flags=re.I)
    # Step 3: Clean up artifacts
    result = re.sub(r'\s+', ' ', result).strip()
    # Remove orphan single-letter leftovers from elisions (l, d, n, s, j, c)
    result = re.sub(r"\b[ldnsjcLD]\b\s*", '', result)
    # Remove orphan apostrophes/quotes left after elision split
    result = re.sub(r"[\u0027\u2019\u2018]", '', result)
    # Remove duplicate articles: "the the" → "the"
    result = re.sub(r'\b(the|of|a|for|in|on)\s+\1\b', r'\1', result, flags=re.I)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# QUERY DECOMPOSITION — split multi-entity queries into sub-queries
# ═══════════════════════════════════════════════════════════════════════════

_ENZYME_PATTERNS = [
    (r"acide\s+ascorbique|ascorbic\s+acid|vitamine\s*C\b|E300",
     "acide ascorbique", "ascorbic acid"),
    (r"alpha[\s-]*amylase|α[\s-]*amylase",
     "alpha-amylase", "alpha-amylase"),
    (r"xylanase",
     "xylanase", "xylanase"),
    (r"(?<!\w)lipase",
     "lipase", "lipase"),
    (r"transglutaminase",
     "transglutaminase", "transglutaminase"),
    (r"amyloglucosidase|glucoamylase",
     "amyloglucosidase", "amyloglucosidase"),
    (r"glucose\s*oxy?dase",
     "glucose oxydase", "glucose oxidase"),
    (r"amylase\s+maltogénique|maltogenic\s+amylase",
     "amylase maltogénique", "maltogenic amylase"),
]


def _decompose_query(question):
    """Detect multiple enzyme entities in a query and split into sub-queries."""
    found = []
    for pattern, name_fr, name_en in _ENZYME_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            found.append((name_fr, name_en))

    if len(found) <= 1:
        return [(question, _translate_fr_to_en(question) if _is_french(question) else question)]

    # Detect intent
    intent_fr = "dosage recommandé"
    intent_en = "recommended dosage"
    if re.search(r"à\s+quoi\s+sert|fonction|rôle|purpose|function", question, re.I):
        intent_fr = "fonction"
        intent_en = "function"
    elif re.search(r"quantit|dosage|dose|ppm|combien", question, re.I):
        intent_fr = "dosage recommandé"
        intent_en = "recommended dosage"
    elif re.search(r"stock|conserv|storage|température", question, re.I):
        intent_fr = "conditions de stockage"
        intent_en = "storage conditions"
    elif re.search(r"effet|volume|force|texture|amélio", question, re.I):
        intent_fr = "effet"
        intent_en = "effect function"

    sub_queries = []
    for name_fr, name_en in found:
        sq_fr = f"{intent_fr} {name_fr} panification boulangerie"
        sq_en = f"{intent_en} {name_en} bakery bread"
        sub_queries.append((sq_fr, sq_en, name_fr, name_en))

    logger.info(f"Query decomposed into {len(sub_queries)} sub-queries: "
                f"{[sq[0] for sq in sub_queries]}")
    return sub_queries


# ═══════════════════════════════════════════════════════════════════════════
# BILINGUAL SEARCH — search both FR and EN, merge best results
# ═══════════════════════════════════════════════════════════════════════════

def _bilingual_search(query_fr, query_en, top_k):
    """Search with both French and English query variants, return merged top-k."""
    candidates = {}  # key → (id_doc, text, best_score)

    for query in [query_fr, query_en]:
        if not query or not query.strip():
            continue
        emb = create_embedding(query)
        if emb is None:
            continue
        for id_doc, text, score in search_cosine_similarity(emb, top_k=top_k):
            key = text[:120].lower().strip()
            s = float(score)
            if key not in candidates or s > candidates[key][2]:
                candidates[key] = (id_doc, text, s)

    # Return sorted by score
    return sorted(candidates.values(), key=lambda x: x[2], reverse=True)[:top_k]


def _result_mentions_entity(text, name_fr, name_en):
    """Check if a result text is relevant to the target entity."""
    t = text.lower()
    return (name_fr.lower() in t) or (name_en.lower() in t)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SEARCH ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def search(question):
    """Search for similar fragments with bilingual support and query decomposition.

    1. If the query is French, translate it to English
    2. Search with BOTH the original French AND English translation
    3. For multi-entity queries, decompose and search each entity separately
    4. Merge results keeping the best score per chunk
    """
    sub_queries = _decompose_query(question)

    if len(sub_queries) == 1:
        # Single entity or generic query — bilingual search
        query_fr, query_en = sub_queries[0]
        results = _bilingual_search(query_fr, query_en, config.TOP_K)
        if not results:
            logger.error("Pas de résultats trouvés.")
            return []
        logger.info(f"✓ Bilingual search: FR='{query_fr[:50]}' EN='{query_en[:50]}'")
        return _format_results(results)

    # ── Multi-entity: search each sub-query bilingualy ───────────────
    per_entity_best = []
    all_candidates = {}

    for sq_fr, sq_en, name_fr, name_en in sub_queries:
        entity_results = _bilingual_search(sq_fr, sq_en, config.TOP_K)

        # Track in global pool
        for id_doc, text, score in entity_results:
            key = text[:120].lower().strip()
            if key not in all_candidates or score > all_candidates[key][2]:
                all_candidates[key] = (id_doc, text, score)

        # Best result that actually mentions this entity
        entity_match = [r for r in entity_results
                        if _result_mentions_entity(r[1], name_fr, name_en)]
        if entity_match:
            per_entity_best.append(entity_match[0])
        elif entity_results:
            per_entity_best.append(entity_results[0])

    # Merge: 1 per entity first, then fill from global pool
    final = []
    used_keys = set()

    for id_doc, text, score in per_entity_best:
        key = text[:120].lower().strip()
        if key not in used_keys:
            used_keys.add(key)
            final.append((id_doc, text, score))
        if len(final) >= config.TOP_K:
            break

    if len(final) < config.TOP_K:
        remaining = sorted(all_candidates.values(), key=lambda x: x[2], reverse=True)
        for id_doc, text, score in remaining:
            key = text[:120].lower().strip()
            if key not in used_keys:
                used_keys.add(key)
                final.append((id_doc, text, score))
            if len(final) >= config.TOP_K:
                break

    final.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"✓ Multi-entity search: {len(sub_queries)} sub-queries → {len(final)} results")
    return _format_results(final)


def _format_results(results):
    """Format raw DB results into dicts."""
    formatted = []
    for rank, (id_document, texte, score) in enumerate(results, 1):
        formatted.append({
            'rank': rank,
            'id_document': id_document,
            'texte': texte,
            'score': round(float(score), 4)
        })
    logger.info(f"✓ Recherche effectuée: {len(formatted)} résultats")
    return formatted

def display_results(question, results):
    """Afficher les résultats formatés"""
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}\n")
    
    for res in results:
        print(f"Résultat {res['rank']}")
        print(f"Score: {res['score']}")
        print(f"Texte: {res['texte']}...")
        print(f"{'-'*80}\n")
