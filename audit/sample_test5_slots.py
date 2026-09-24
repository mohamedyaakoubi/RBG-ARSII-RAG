"""Draw the question slots for TEST-5 at random (fixed seed). TEST-5 is frozen
with ROUTES_PLAN.md, before any of the extraction or chunking routes it tests
is run. It covers the whole corpus like TEST-3 (product, family, corpus-wide,
ascorbic-acid and multi-product questions), at about twice TEST-3's size, and
a product is written in one of TEST-4's code spellings."""
import random
import re

from audit.sample_test4_slots import CODE, FAMILIES, FORMS, NO_ACTIVITY

rng = random.Random(20260927)
PRODUCTS = [p for ps in FAMILIES.values() for p in ps]

PRODUCT_ATTRS = {'dosage': 16, 'function': 9, 'application': 3, 'source': 5, 'activity': 4, 'enzyme_type': 3,
                 'storage': 4}
CORPUS_ATTRS = {'allergens': 2, 'gmo': 1, 'irradiation': 1, 'packaging': 2, 'microbiology': 3, 'heavy_metals': 2,
                'moisture': 1, 'aspect': 1, 'manufacturer': 1, 'last_update': 2}
AA_ATTRS = {'aa_dosage_type': 7, 'aa_max': 2, 'aa_conversion': 2, 'aa_mode': 5, 'aa_specs': 4, 'aa_storage': 3,
            'aa_role': 3, 'aa_limits': 1, 'aa_alternatives': 1}
SUB = {
    'microbiology': ['salmonella', 'total plate count', 'coliforms', 'staphylococcus', 'ASR'],
    'heavy_metals': ['lead', 'cadmium', 'mercury', 'arsenic'],
    'aa_dosage_type': ['directe standard', 'pousse lente', 'blocage froid', 'surgélation', 'pain de mie CBP',
                       'viennoiserie', 'biscuits'],
    'aa_conversion': [f'{w} kg @ {p} ppm' for w in (10, 50, 100, 500, 1000) for p in (50, 75, 100, 150)],
    'aa_mode': ['incorporation', 'dilution', 'action time', 'optimal temperature', 'weighing'],
    'aa_specs': ['formula', 'purity', 'solubility', 'pH', 'density', 'aspect', 'additive code'],
    'aa_storage': ['formats', 'temperature/humidity', 'shelf life'],
    'aa_role': ['oxidizing/gluten', 'volume/gas', 'fermentation', 'structure', 'crust colour', 'proofing tolerance'],
    'aa_limits': ['limitation', 'advantage'],
}
MULTI_INTENTS = ['dosage'] * 5 + ['function'] * 3 + ['source', 'storage', 'storage', 'activity']


def lang():
    return 'fr' if rng.random() < 0.6 else 'en'


def render(d):
    """How the question writes the product (TEST-4's spellings)."""
    code, form = CODE[d], rng.choices(list(FORMS), weights=list(FORMS.values()))[0]
    brand = 'BVZyme ' if form == 'printed+brand' or (form != 'printed' and rng.random() < 0.5) else ''
    if form == 'lower':
        return (brand + code).lower()
    if form == 'respaced':
        return brand + (code.replace(' ', '') if ' ' in code else re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', code, count=1))
    if form == 'hyphenated':
        return brand + (code.replace(' ', '-') if ' ' in code else re.sub(r'(?<=[A-Za-z])(?=\d)', '-', code, count=1))
    return brand + code


slots = []
for attr, n in PRODUCT_ATTRS.items():
    for _ in range(n):
        if rng.random() < 0.4:
            d = rng.choice([p for p in PRODUCTS if not (attr == 'activity' and p in NO_ACTIVITY)])
            slots.append(('product', attr, d, render(d), lang()))
        else:
            slots.append(('family', attr, rng.choice(list(FAMILIES)), '', lang()))
for attr, n in CORPUS_ATTRS.items():
    for _ in range(n):
        sub = rng.choice(SUB[attr]) if attr in SUB else ''
        slots.append(('corpus', attr + (f':{sub}' if sub else ''), 'any TDS', '', lang()))
for attr, n in AA_ATTRS.items():
    for _ in range(n):
        sub = rng.choice(SUB[attr]) if attr in SUB else ''
        slots.append(('aa', attr + (f':{sub}' if sub else ''), 'ascorbic acid', '', lang()))
ENTITIES = list(FAMILIES) + ['ascorbic acid']
for intent in MULTI_INTENTS:
    k = 3 if rng.random() < 0.3 else 2
    ents = rng.sample([e for e in ENTITIES if not (intent == 'activity' and e == 'ascorbic acid')], k)
    slots.append(('multi', intent, ' + '.join(ents), '', lang()))

if __name__ == '__main__':
    for i, s in enumerate(slots, 1):
        print(f'{i:3d}', s)
    print(len(slots), 'slots;', sum(s[-1] == 'fr' for s in slots), 'fr')
