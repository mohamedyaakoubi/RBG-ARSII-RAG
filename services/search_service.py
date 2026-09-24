"""
Semantic search, as specified by the challenge: the question is embedded with
all-MiniLM-L6-v2, compared with the stored vectors by cosine similarity, and
the 3 best fragments are returned with their text and score.

Two modes (config.SEARCH_MODE):
  strict       the question is embedded exactly as typed.
  transparent  a French question is also embedded in English (the model's
               language) and the better of the two is kept; a question naming
               several product families is split into one sub-question per
               family, keeping the user's wording; a question naming a product
               by its code ("L MAX64") is answered from that product's sheet.
               Every result shows the formulation that produced its score, and
               a restriction to named sheets is shown with the results.

In both modes a fragment whose content is already shown (its translation, or
a section contained in a fragment already shown) is skipped.
"""
import re

from config.settings import config
from database.models import document_headers, search_cosine_similarity
from services.embedding_service import embed_texts
from services.pdf_processor import sheet_product
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


def _ranked(formulations, documents=None):
    """Rows by decreasing cosine similarity, best formulation kept per row.
    documents: only rows of these documents."""
    best = {}
    for text, vec in zip(formulations, embed_texts(formulations)):
        for row in search_cosine_similarity(vec, top_k=config.CANDIDATES, documents=documents):
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


def _split(question, spans):
    """One sub-question per named span [(name, match)] in order, keeping the
    user's wording (a coordinated list is reduced to one of its members)."""
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
        subs.append(re.sub(r'\s+', ' ', s).strip())
    return subs


def split_by_entity(question):
    """[(family, sub-question)] when the question names several product
    families, else []."""
    spans = _entities(question)
    return list(zip([name for name, _ in spans], _split(question, spans))) if len(spans) >= 2 else []


def code_pattern(product):
    """'BVZyme L MAX64' -> regex for its code however it is cased, spaced or
    hyphenated ('L MAX64', 'lmax64', 'L-MAX 64'), not glued to other letters
    or digits ('AF110' does not match 'AF1100')."""
    code = re.sub(r'^BVZyme\s+', '', product, flags=re.I).replace(' ', '')
    return (r'(?:(?<![A-Za-z0-9])|(?<=bvzyme))' + r'[\s_‐‑–-]*'.join(map(re.escape, code))
            + r'(?![A-Za-z0-9])')


def named_products(question):
    """Products the question names by their code, read from the stored sheet
    headers: [(id_document, header, product, match)] in order."""
    found = []
    for doc, header in document_headers():
        product = sheet_product(header)
        m = product and re.search(code_pattern(product), question, re.I)
        if m:
            found.append((doc, header, product, m))
    return sorted(found, key=lambda x: x[3].start())


def _search_products(question, products, mode, k):
    """Product filter. One product, and no other product family (its own may be
    named): the k best fragments of its sheet. Otherwise one sub-question per
    product and per other family named, a product's answered from its sheet, a
    family's from fragments about it; the remaining slots are filled from all
    of these. Ranking is by cosine throughout; content already shown is skipped."""
    patterns = dict(ENTITIES)
    docs = [doc for doc, _, _, _ in products]
    own = {n for n, p in ENTITIES for _, header, _, _ in products if re.search(p, header, re.I)}
    families = [(n, m) for n, m in _entities(question) if n not in own]
    names = [product for _, _, product, _ in products] + [m.group(0) for _, m in families]
    if len(names) == 1:
        rows = _distinct(_ranked(_formulations(question, mode), docs), k)
        return rows, f"Recherche limitée à la fiche {names[0]}, nommée dans la question."
    restriction = f"Recherche limitée à ce que nomme la question : {', '.join(names)}."
    about = {doc: (lambda r, doc=doc: r['id_document'] == doc) for doc in docs}
    about.update({n: (lambda r, p=patterns[n]: re.search(p, r['produit'], re.I) is not None) for n, _ in families})
    spans = sorted([(doc, m) for doc, _, _, m in products] + families, key=lambda x: x[1].start())
    picked, keys, shown = [], set(), {}

    def take(r):
        seen = shown.get(r['id_document'], set())
        if r['cle'] in keys or set(r['couvre']) <= seen:
            return False
        picked.append(r)
        keys.add(r['cle'])
        shown[r['id_document']] = seen | set(r['couvre'])
        return True

    for (name, _), sub in zip(spans, _split(question, spans)):
        formulations = _formulations(sub, mode)
        rows = _ranked(formulations, [name]) if name in docs else _ranked(formulations)
        for r in _distinct([r for r in rows if about[name](r)], 10):
            if take(r):
                break
    formulations = _formulations(question, mode)          # fill with the whole question
    rows = {r['id']: r for r in _ranked(formulations)}
    rows.update({r['id']: r for r in _ranked(formulations, docs)})
    for r in _distinct(sorted((r for r in rows.values() if any(a(r) for a in about.values())),
                              key=lambda r: r['score'], reverse=True), 3 * k):
        if len(picked) >= k:
            break
        take(r)
    return sorted(picked[:k], key=lambda r: r['score'], reverse=True), restriction


def search(question, mode=None, k=config.TOP_K):
    mode = mode or config.SEARCH_MODE
    products = named_products(question) if mode == 'transparent' else []
    subs = split_by_entity(question) if mode == 'transparent' and not products else []
    restriction = None
    if products:
        results, restriction = _search_products(question, products, mode, k)
    elif not subs:
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
             'score': round(r['score'], 4), 'formulation': r['formulation'], 'restriction': restriction}
            for i, r in enumerate(results, 1)]


def display_results(question, results):
    """Afficher les résultats formatés"""
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    if results and results[0]['restriction']:
        print(results[0]['restriction'])
    print(f"{'='*80}\n")
    for res in results:
        print(f"Résultat {res['rank']}")
        print(f"Score: {res['score']}   (similarité cosinus avec : « {res['formulation']} »)")
        print(f"Texte: {res['texte']}")
        if res['original']:
            print(f"Texte original: {res['original']}")
        print(f"Source: {res['fichier']}")
        print(f"{'-'*80}\n")
