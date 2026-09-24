"""
French → English translation (OPUS-MT, Helsinki-NLP/opus-mt-fr-en).

Used for two things:
  - indexing French documents also in English, the embedding model's
    language (the translation is shown next to the original);
  - in "transparent" search mode, embedding a French question also in English.

Translation must not change facts: fragments are translated sentence by
sentence (MT tends to drop clauses of long inputs), section labels come from
a fixed glossary, and a translation is rejected if a sentence changes a number
or has an implausible length. Results are cached in config.TRANSLATION_CACHE.
"""
import json
import re
from pathlib import Path

from config.settings import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

_MODEL = {}
_CACHE_PATH = Path(config.TRANSLATION_CACHE)
_CACHE = json.loads(_CACHE_PATH.read_text()) if _CACHE_PATH.exists() else {}

LABELS_FR_EN = {
    'Résumé Général': 'General summary', 'Propriétés Principales': 'Main properties',
    'Points Importants': 'Key points', 'Dosages Recommandés (ppm*)': 'Recommended dosages (ppm*)',
    'Dosages Recommandés (ppm = g/tonne de farine)': 'Recommended dosages (ppm = g/tonne of flour)',
    'Table de Conversion Rapide (en grammes)': 'Quick conversion table (in grams)',
    'Spécifications Techniques > Caractéristiques du Produit': 'Technical specifications > Product characteristics',
    'Conditionnement Recommandé': 'Recommended packaging', "Mode d'Emploi en Production": 'Directions for use in production',
    'Points de Contrôle': 'Control points', 'Avantages et Limitations > Avantages': 'Advantages and limitations > Advantages',
    'Limitations': 'Limitations', 'Alternatives et Complémentarité': 'Alternatives and complementarity',
    'Réglementation > Statut Légal': 'Regulation > Legal status', 'Dosage Maximum Autorisé': 'Maximum authorised dosage',
    'Recommandations pour ta Production > Test et Validation': 'Recommendations for your production > Testing and validation',
    'Stockage et Sécurité': 'Storage and safety', 'Références': 'References',
}
HEADERS_FR_EN = {'Acide Ascorbique (E300)': 'Ascorbic Acid (E300)'}


def is_french(text):
    import langid
    langid.set_languages(['fr', 'en'])
    return langid.classify(text)[0] == 'fr'


def translate(texts, direction='fr-en'):
    """Machine translation with a persistent cache."""
    todo = [t for t in dict.fromkeys(texts) if f'{direction}|{t}' not in _CACHE]
    if todo:
        if direction not in _MODEL:
            from transformers import MarianMTModel, MarianTokenizer
            name = f'Helsinki-NLP/opus-mt-{direction}'
            logger.info(f'Chargement du modèle de traduction {name}...')
            _MODEL[direction] = (MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name))
        tok, mt = _MODEL[direction]
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
            out = mt.generate(**enc, num_beams=4, max_new_tokens=256)
            for src, o in zip(batch, out):
                _CACHE[f'{direction}|{src}'] = tok.decode(o, skip_special_tokens=True)
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(_CACHE, ensure_ascii=False, indent=0))
    return [_CACHE[f'{direction}|{t}'] for t in texts]


def translate_question(question):
    """English version of a French question (None if it is not French)."""
    return translate([question])[0] if is_french(question) else None


def numbers(s):
    """Multiset of numbers, ignoring separators ('0,5' == '0.5', '1 000' == '1,000')."""
    return sorted(re.sub(r'\D', '', n) for n in re.findall(r'\d+(?:[ ., ]\d+)*', s))


_SEP = r'((?<=[.;])\s+(?=[A-Z0-9*(\-])|\s+\|\s+|\s+(?=- ))'


def _segments(content):
    """Sentences, table cells and bullets, translated one by one; separators kept."""
    pieces = re.split(_SEP, content)
    return [p for p in pieces[0::2] if p.strip()], [s if '|' in s else ' ' for s in pieces[1::2]]


def _join(segments, separators):
    out = segments[0]
    for sep, seg in zip(separators, segments[1:]):
        out += sep + seg
    return out


def _faithful(src, dst):
    return numbers(src) == numbers(dst) and (len(src) <= 15 or 0.5 <= len(dst) / len(src) <= 2.5)


def translate_fragments(frags):
    """English versions of French fragments (same content key, so a fragment
    and its translation are never shown together). Rejected translations are
    logged and skipped: the original stays indexed."""
    from services.pdf_processor import fragment
    plans = []
    for f in frags:
        parts = []
        for label, content in f['parts']:
            segs, seps = _segments(content)
            parts.append((label, segs, seps))
        plans.append((f, parts))
    translate([s for _, parts in plans for _, segs, _ in parts for s in segs])       # batch
    labels = {l for _, parts in plans for l, _, _ in parts if l and l not in LABELS_FR_EN}
    headers = {f['produit'] for f in frags if f['produit'] not in HEADERS_FR_EN}
    extra = dict(zip(sorted(labels | headers), translate(sorted(labels | headers))))
    out, rejected = [], 0
    for f, parts in plans:
        new_parts, ok = [], True
        for label, segs, seps in parts:
            tsegs = translate(segs)
            ok = ok and all(_faithful(s, t) for s, t in zip(segs, tsegs))
            new_label = LABELS_FR_EN.get(label) or extra.get(label) if label else None
            new_parts.append((new_label, _join(tsegs, seps)))
        if not ok:
            rejected += 1
            continue
        header = HEADERS_FR_EN.get(f['produit']) or extra.get(f['produit'], f['produit'])
        if numbers(header) != numbers(f['produit']):
            header = f['produit']
        out.append(fragment(f['id_document'], f['fichier'], header, new_parts, 'en', f['cle'],
                            set(f['couvre']), original=f['texte']))
    logger.info(f'{len(out)} fragments traduits en anglais, {rejected} traductions rejetées (garde-fous)')
    return out
