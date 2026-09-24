"""
Semantic search, as specified by the challenge: the question is embedded with
all-MiniLM-L6-v2, compared with the stored vectors by cosine similarity, and
the 3 best fragments are returned with their text and score.

Two modes (config.SEARCH_MODE):
  strict       the question is embedded exactly as typed.
  transparent  a French question is also embedded in English (the model's
               language) and the better of the two is kept; a question naming
               several product families is split into one sub-question per
               family, keeping the user's wording. Every result shows the
               formulation that produced its score.

In both modes a fragment whose content is already shown (its translation, or
a section contained in a fragment already shown) is skipped.
"""
import re

from config.settings import config
from database.models import search_cosine_similarity
from services.embedding_service import embed_texts
from services.translation import translate_question
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Product families of the corpus and how people name them (FR / EN).
ENTITIES = [
    ('alpha-amylase', r"alpha[\s-]*amylases?|α[\s-]*amylases?|amylases?\s+fongiques?|fungal\s+(?:alpha[\s-]*)?amylases?"),
    ('maltogenic amylase', r"amylases?\s+maltog[ée]niques?|maltogenic\s+amylases?"),
    ('amyloglucosidase', r"amyloglucosidases?|glucoamylases?"),
    ('xylanase', r"xylanases?"),
    ('lipase', r"(?:phospho)?lipases?"),
    ('glucose oxidase', r"glucose[\s-]*ox[yi]dases?"),
    ('transglutaminase', r"transglutaminases?"),
    ('ascorbic acid', r"acides?\s+ascorbiques?|ascorbic\s+acid|vitamine?\s*C\b|E300"),
]
_GLUE = r"(?:\s|,|;|/|&|\+|\bet\b|\band\b|\bou\b|\bor\b|\bde\b|\bd['’]|\bdu\b|\bdes\b|\bla\b|\ble\b|\bl['’]|\bles\b|\bof\b|\bthe\b)*"


def _ranked(formulations):
    """Rows by decreasing cosine similarity, best formulation kept per row."""
    best = {}
    for text, vec in zip(formulations, embed_texts(formulations)):
        for row in search_cosine_similarity(vec, top_k=config.CANDIDATES):
            score = float(row['score'])
            if row['id'] not in best or score > best[row['id']]['score']:
                best[row['id']] = {**row, 'score': score, 'formulation': text}
    return sorted(best.values(), key=lambda r: r['score'], reverse=True)


def _distinct(rows, k):
    """First k rows whose content is not already shown."""
    shown, keys, out = {}, set(), []
    for r in rows:
        seen = shown.get(r['id_document'], set())
        if set(r['couvre']) <= seen or r['cle'] in keys:
            continue
        shown[r['id_document']] = seen | set(r['couvre'])
        keys.add(r['cle'])
        out.append(r)
        if len(out) == k:
            break
    return out


def _formulations(question, mode):
    if mode == 'transparent':
        english = translate_question(question)
        if english:
            return [question, english]
    return [question]


def _entities(question):
    found = [(name, m) for name, p in ENTITIES for m in [re.search(p, question, re.I)] if m]
    return sorted(found, key=lambda x: x[1].start())


def split_by_entity(question):
    """One sub-question per product family named in the question, keeping the
    user's wording (a coordinated list is reduced to one of its members)."""
    spans = _entities(question)
    if len(spans) < 2:
        return []
    listed = all(re.fullmatch(_GLUE, question[a.end():b.start()], re.I)
                 for (_, a), (_, b) in zip(spans, spans[1:]))
    subs = []
    for keep, kept in spans:
        if listed:
            s = question[:spans[0][1].start()] + kept.group(0) + question[spans[-1][1].end():]
        else:
            s = question
            for name, m in reversed(spans):
                if name != keep:
                    s = s[:m.start()] + s[m.end():]
        subs.append((keep, re.sub(r'\s+', ' ', s).strip()))
    return subs


def search(question, mode=None, k=config.TOP_K):
    mode = mode or config.SEARCH_MODE
    subs = split_by_entity(question) if mode == 'transparent' else []
    if not subs:
        results = _distinct(_ranked(_formulations(question, mode)), k)
    else:
        patterns, picked, keys = dict(ENTITIES), [], set()
        for name, sub in subs:                  # best fragment about each product family
            hits = _distinct(_ranked(_formulations(sub, mode)), 10)
            about = [r for r in hits if re.search(patterns[name], r['produit'], re.I)]
            for r in about or hits:
                if r['cle'] not in keys:
                    picked.append(r)
                    keys.add(r['cle'])
                    break
        for r in _distinct(_ranked(_formulations(question, mode)), 2 * k):   # fill with the whole question
            if len(picked) >= k:
                break
            if r['cle'] not in keys:
                picked.append(r)
                keys.add(r['cle'])
        results = sorted(picked[:k], key=lambda r: r['score'], reverse=True)
    logger.info(f"✓ Recherche ({mode}): {len(results)} résultats")
    return [{'rank': i, 'id_document': r['id_document'], 'fichier': r['fichier'], 'produit': r['produit'],
             'texte': r['texte_fragment'], 'original': r['texte_original'] if r['langue'] == 'en'
             and r['texte_original'] != r['texte_fragment'] else None,
             'score': round(r['score'], 4), 'formulation': r['formulation']}
            for i, r in enumerate(results, 1)]


def display_results(question, results):
    """Afficher les résultats formatés"""
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"{'='*80}\n")
    for res in results:
        print(f"Résultat {res['rank']}")
        print(f"Score: {res['score']}   (similarité cosinus avec : « {res['formulation']} »)")
        print(f"Texte: {res['texte']}")
        if res['original']:
            print(f"Texte original: {res['original']}")
        print(f"Source: {res['fichier']}")
        print(f"{'-'*80}\n")
