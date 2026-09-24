"""
Check every row of the original README benchmark tables against what the
committed code actually returns.

    python -m audit.reproduce_readme

For each README cell: does the score reproduce, does the quoted fragment match
the fragment that produced that score, and does the quoted text exist in the
PDFs at all? Also reports the honest score, cosine(question as typed, returned
fragment), which is what the challenge asks to display.
Writes audit/results/readme_check.md. Needs the `legacy_embeddings` table built by
audit/legacy/ingestion_data.py, the original pipeline (python -m audit.evaluate builds it).
"""
import logging
import re
import sys
from pathlib import Path

logging.disable(logging.CRITICAL)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pdfplumber

from services.embedding_service import model
from audit.legacy.search_service import search

COMPETITION = "Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?"

# (question, rank, README score, fragment as printed in the README, regex the
#  README fragment asserts (applied to squashed text), explanation)
README = [
    (COMPETITION, 1, 0.9330, 'Dosage alpha-amylase (BVZyme AF330) boulangerie panification : 2-10 ppm', r'af330.*2-10ppm',
     "README adds 'ppm'; the real output has no unit ('2-10.'). Score is the rewritten sub-query's, not the question's"),
    (COMPETITION, 2, 0.9131, 'Dosage xylanase (BVZyme HCB708) boulangerie panification : 5-30 ppm', r'hcb708.*5-30ppm',
     "different product, dosage and score: the real rank 2 is HCF400 '15-35.' (0.9185)"),
    (COMPETITION, 3, 0.8635, 'Dosage acide ascorbique (vitamine C, E300) boulangerie panification : 50-75 ppm', r'acideascorbique.*50-75ppm',
     "matches; score is the rewritten sub-query's (honest 0.339)"),
    ("Quel dosage de lipase pour la panification ?", 1, 0.7758, 'Dosage lipase (L65pdf) boulangerie panification : 5-50 ppm', r'l65pdf.*5-50ppm',
     "README adds 'ppm'; the real output has no unit ('5-50.')"),
    ("Quel dosage de xylanase en boulangerie ?", 1, 0.8178, 'BVZyme HCB709 (xylanase) dosage for bakery: 5-20 ppm', r'hcb709.*5-20ppm',
     'README shows HCB709 (really rank 2, 0.8153); rank 1 is HCF500'),
    ("À quoi sert l'acide ascorbique en boulangerie ?", 1, 0.7395, "À quoi sert l'acide ascorbique en boulangerie ? L'acide ascorbique (vitamine C) est un additif…", r"^àquoisertl'acideascorbiqueenboulangerie\?l'acideascorbique\(vitaminec\)estunadditif",
     'matches, but the fragment starts with this exact benchmark question (synthetic chunk)'),
    ("Quel est l'effet de la xylanase sur le volume du pain ?", 1, 0.6917, 'BVZyme HCB710 (xylanase): Improve loaf volume, enhance stability, increase elasticity', r'hcb710.*improveloafvolume',
     "README shows HCB710's function text (really rank 2, 0.6880); rank 1 is a dosage line that does not answer the question"),
    ("What is the recommended dosage of alpha-amylase for bread?", 1, 0.7786, 'BVZyme AF330 (alpha-amylase) dosage for bakery: 2-10 ppm', r'af330.*2-10ppm',
     'README shows AF330 (really rank 2, 0.7775); rank 1 is AF220'),
    ("What are the storage conditions for BVZyme AF110?", 1, 0.6818, 'Storage Store in a cool, dry place (below 20°C)', r'af110.*storeinacool,dryplace\(below20°c\)',
     'matches (abbreviated)'),
    ("Quelle est la dose recommandée de transglutaminase ?", 1, 0.8209, 'Dosage transglutaminase (BVZyme TG MAX63) : 5-25 ppm', r'tgmax63.*5-25ppm',
     "matches (reworded); score is the dictionary-translated query's (honest 0.558)"),
    ("Does BVZyme contain allergens?", 1, 0.7649, 'Allergens In compliance with the list of major…', r'allergensincompliancewiththelistofmajor',
     "matches, but README marks it correct: the fragment is cut before the answer ('...contains the following allergen: gluten')"),
    ("What is the optimal pH for xylanase activity?", 1, 0.6188, 'Suggested Optimum…', r'suggestedoptimum',
     "no indexed fragment contains 'Suggested Optimum' (in the PDFs it is a dosage line, not a pH); rank 1 is a product description"),
    ("How does lipase improve bread texture?", 1, 0.6718, 'Function, fine regular crumb structure, improve stability and tolerance', r'fineregularcrumbstructure,improvestabilityandtolerance',
     'matches (abbreviated)'),
    ("What is the shelf life of BVZyme enzymes?", 1, 0.8143, 'Storage — Date of minimum durability: 24 months', r'dateofminimumdurability:24months',
     'matches (abbreviated)'),
    ("Quelle est l'activité enzymatique de l'alpha-amylase ?", 1, 0.7608, 'Activity 85000 SKB/g', r'85000skb/g',
     "85000 SKB/g is AF SX's activity; rank 1 is AF330 (11900 FAU/g)"),
    ("What is the microbial source of BVZyme xylanase?", 1, 0.7917, 'Bacterial xylanase produced by fermenting a selected unique strain…', r'bacterialxylanaseproducedbyfermentingaselecteduniquestrain',
     'matches (abbreviated)'),
    ("How to combine alpha-amylase and xylanase for bread?", 1, 0.9330, 'Dosage alpha-amylase 2-10 ppm', r'dosagealpha-amylase.*2-10ppm',
     "question rewritten to 'recommended dosage alpha-amylase…': 'combine' is dropped; 0.9330 is the sub-query score (honest 0.528)"),
    ("What packaging is used for BVZyme products?", 1, 0.7175, 'Packaging 25 kg paper bag with PE liner', r'paperbag|peliner',
     "'paper bag with PE liner' is in none of the 35 PDFs; every product ships in a 'Carton box of 25 kg'"),
]


def squash(s):
    return re.sub(r'\s+', '', s.lower())


def embed(t):
    return model.encode(t, normalize_embeddings=True)


def main():
    corpus = ''
    for p in sorted((ROOT / 'data_pdf').glob('*.pdf')):
        with pdfplumber.open(p) as pdf:
            corpus += squash(p.name + ' ' + ' '.join(pg.extract_text() or '' for pg in pdf.pages))
    out = ['# README benchmark check', '',
           '| # | question | README score | reproduced | honest cos(question, fragment) | README fragment matches actual output? | README text exists in PDFs? | actual fragment returned | note |',
           '|---:|---|---:|---:|---:|---|---|---|---|']
    cache = {}
    for i, (q, rank, score, frag, claim, note) in enumerate(README, 1):
        res = cache.setdefault(q, search(q))
        actual = res[rank - 1]
        honest = float(np.dot(embed(q), embed(actual['texte'])))
        matches = bool(re.search(claim, squash(actual['texte'])))
        exists = bool(re.search(claim.lstrip('^'), corpus)) or matches
        out.append(f'| {i} | {q[:60]}{"…" if len(q) > 60 else ""} (R{rank}) | {score:.4f} | {actual["score"]:.4f} | '
                   f'{honest:.4f} | {"yes" if matches else "**no**"} | {"yes" if exists else "**no**"} | '
                   f'`{actual["texte"][:90]}` | {note} |')
    text = '\n'.join(out) + '\n'
    (ROOT / 'audit' / 'results').mkdir(parents=True, exist_ok=True)
    (ROOT / 'audit' / 'results' / 'readme_check.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
