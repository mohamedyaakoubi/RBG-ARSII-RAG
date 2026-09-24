"""Draw the question slots for TEST-2 at random (fixed seed) so that the choice
of what to ask is not hand-picked. Questions are then written for each slot."""
import random
rng = random.Random(20260924)

FAMILIES = {
    'alpha-amylase': ['af110', 'af220', 'af330', 'afsx'],
    'maltogenic amylase': ['afresh101', 'afresh202', 'afresh303', 'asoft205', 'asoft305', 'asoft405'],
    'amyloglucosidase': ['amg880', 'amg1400'],
    'glucose oxidase': ['gox110', 'gomax63', 'gomax65'],
    'xylanase': ['hcb708', 'hcb709', 'hcb710', 'hcf400', 'hcf500', 'hcf600', 'hcfmax63', 'hcfmax64', 'hcfmaxx'],
    'lipase': ['l55', 'l65', 'lmax63', 'lmax64', 'lmax65', 'lmaxx'],
    'transglutaminase': ['tg881', 'tg883', 'tgmax63', 'tgmax64'],
}
PRODUCTS = [p for ps in FAMILIES.values() for p in ps]
PRODUCT_ATTRS = {'dosage': 12, 'function': 6, 'application': 3, 'source': 4, 'activity': 3, 'enzyme_type': 2, 'storage': 2}
CORPUS_ATTRS = {'allergens': 2, 'gmo': 1, 'irradiation': 1, 'packaging': 2, 'microbiology': 2, 'heavy_metals': 1,
                'moisture': 1, 'aspect': 1, 'manufacturer': 1, 'last_update': 1}
AA_ATTRS = {'aa_dosage_type': 6, 'aa_max': 2, 'aa_conversion': 2, 'aa_mode': 4, 'aa_specs': 3, 'aa_storage': 3,
            'aa_role': 2, 'aa_limits': 2, 'aa_alternatives': 1}
SUB = {
    'microbiology': ['salmonella', 'total plate count', 'coliforms', 'staphylococcus', 'ASR'],
    'heavy_metals': ['lead', 'cadmium', 'mercury', 'arsenic'],
    'aa_dosage_type': ['directe standard', 'pousse lente', 'blocage froid', 'surgélation', 'pain de mie CBP', 'viennoiserie', 'biscuits'],
    'aa_conversion': [f'{w} kg @ {p} ppm' for w in (10, 50, 100, 500, 1000) for p in (50, 75, 100, 150)],
    'aa_mode': ['incorporation', 'dilution', 'action time', 'optimal temperature', 'weighing'],
    'aa_specs': ['formula', 'purity', 'solubility', 'pH', 'density', 'aspect', 'additive code'],
    'aa_storage': ['formats', 'temperature/humidity', 'shelf life'],
    'aa_role': ['oxidizing/gluten', 'volume/gas', 'fermentation', 'structure', 'crust colour', 'proofing tolerance'],
    'aa_limits': ['limitation', 'advantage'],
}

def lang():
    return 'fr' if rng.random() < 0.65 else 'en'

slots = []
for attr, n in PRODUCT_ATTRS.items():
    for _ in range(n):
        if rng.random() < 0.4:
            slots.append(('product', attr, rng.choice(PRODUCTS), lang()))
        else:
            slots.append(('family', attr, rng.choice(list(FAMILIES)), lang()))
for attr, n in CORPUS_ATTRS.items():
    for _ in range(n):
        sub = rng.choice(SUB[attr]) if attr in SUB else ''
        slots.append(('corpus', attr + (f':{sub}' if sub else ''), 'any TDS', lang()))
for attr, n in AA_ATTRS.items():
    for _ in range(n):
        sub = rng.choice(SUB[attr]) if attr in SUB else ''
        slots.append(('aa', attr + (f':{sub}' if sub else ''), 'ascorbic acid', lang()))
MULTI_INTENTS = ['dosage'] * 4 + ['function'] * 3 + ['source', 'storage', 'activity']
ENTITIES = list(FAMILIES) + ['ascorbic acid']
for intent in MULTI_INTENTS:
    k = 3 if rng.random() < 0.3 else 2
    ents = rng.sample(ENTITIES, k)
    slots.append(('multi', intent, ' + '.join(ents), lang()))

for i, s in enumerate(slots, 1):
    print(f'{i:3d}', s)
print(len(slots), 'slots;', sum(s[3] == 'fr' for s in slots), 'fr')
