"""Draw the question slots for TEST-4 at random (fixed seed). TEST-4 measures
the product-code filter, so every question names at least one product by its
code. Questions are written for each slot and the set is frozen in git before
the filter is run on it.

Filter rules, fixed before this draw:
  - a product is named when its code (from the sheet's file name, e.g.
    "L MAX64") appears in the question, ignoring case and any spaces, hyphens
    or underscores between its characters, and not glued to other letters or
    digits ("AF110" does not name AF1100);
  - transparent mode only; strict mode is unchanged;
  - one product named, and no other product family (its own family may be
    named): the 3 results come from that product's sheet, ranked by cosine;
  - products and other families named together: one sub-question per product
    or family, a product's sub-question answered from its sheet; the remaining
    slots are filled from the named products' sheets and the named families'
    fragments;
  - no code named: nothing changes.

Primary measure, fixed before the draw: right answer in the top 3 on the
single-product questions, transparent mode with vs. without the filter
(exact McNemar test on paired questions).
"""
import random
import re

rng = random.Random(20260926)

FAMILIES = {
    'alpha-amylase': ['af110', 'af220', 'af330', 'afsx'],
    'maltogenic amylase': ['afresh101', 'afresh202', 'afresh303', 'asoft205', 'asoft305', 'asoft405'],
    'amyloglucosidase': ['amg880', 'amg1400'],
    'glucose oxidase': ['gox110', 'gomax63', 'gomax65'],
    'xylanase': ['hcb708', 'hcb709', 'hcb710', 'hcf400', 'hcf500', 'hcf600', 'hcfmax63', 'hcfmax64', 'hcfmaxx'],
    'lipase': ['l55', 'l65', 'lmax63', 'lmax64', 'lmax65', 'lmaxx'],
    'transglutaminase': ['tg881', 'tg883', 'tgmax63', 'tgmax64'],
}
FAMILY_OF = {d: f for f, ds in FAMILIES.items() for d in ds}
PRODUCTS = [p for ps in FAMILIES.values() for p in ps]
CODE = {  # as printed on the sheets (file names)
    'af110': 'AF110', 'af220': 'AF220', 'af330': 'AF330', 'afsx': 'AF SX',
    'afresh101': 'A FRESH101', 'afresh202': 'A FRESH202', 'afresh303': 'A FRESH303',
    'asoft205': 'A SOFT205', 'asoft305': 'A SOFT305', 'asoft405': 'A SOFT405',
    'amg880': 'AMG880', 'amg1400': 'AMG1400', 'gox110': 'GOX 110', 'gomax63': 'GO MAX 63', 'gomax65': 'GO MAX 65',
    'hcb708': 'HCB708', 'hcb709': 'HCB709', 'hcb710': 'HCB710', 'hcf400': 'HCF400', 'hcf500': 'HCF500',
    'hcf600': 'HCF600', 'hcfmax63': 'HCF MAX63', 'hcfmax64': 'HCF MAX64', 'hcfmaxx': 'HCF MAX X',
    'l55': 'L55', 'l65': 'L65', 'lmax63': 'L MAX63', 'lmax64': 'L MAX64', 'lmax65': 'L MAX65', 'lmaxx': 'L MAX X',
    'tg881': 'TG881', 'tg883': 'TG883', 'tgmax63': 'TG MAX63', 'tgmax64': 'TG MAX64',
}
NO_ACTIVITY = {'tgmax63', 'tgmax64'}          # these two sheets have no Activity section

# weights favour what formulators ask about a given product
SINGLE_ATTRS = {'dosage': 25, 'function': 15, 'activity': 10, 'source': 8, 'enzyme_type': 7,
                'storage': 7, 'packaging': 5, 'allergens': 5, 'gmo_irradiation': 4, 'microbiology': 4,
                'heavy_metals': 3, 'aspect_moisture': 4, 'manufacturer': 2, 'last_update': 1}
SUB = {
    'gmo_irradiation': ['gmo', 'irradiation'],
    'microbiology': ['salmonella', 'total plate count', 'coliforms', 'staphylococcus', 'ASR'],
    'heavy_metals': ['lead', 'cadmium', 'mercury', 'arsenic'],
    'aspect_moisture': ['aspect', 'colour', 'moisture'],
}
TWO_PRODUCT_ATTRS = {'dosage': 5, 'function': 2, 'activity': 2, 'source': 1}
PRODUCT_FAMILY_ATTRS = {'dosage': 6, 'function': 3, 'storage': 1}
FORMS = {'printed+brand': 35, 'printed': 25, 'lower': 10, 'respaced': 15, 'hyphenated': 15}


def pick(weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def lang():
    return 'fr' if rng.random() < 0.6 else 'en'


def render(d):
    """How the question writes the product: drawn, not chosen."""
    code, form = CODE[d], pick(FORMS)
    brand = 'BVZyme ' if form == 'printed+brand' or (form != 'printed' and rng.random() < 0.5) else ''
    if form == 'lower':
        text = (brand + code).lower()
    elif form == 'respaced':
        text = brand + (code.replace(' ', '') if ' ' in code else re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', code, count=1))
    elif form == 'hyphenated':
        text = brand + (code.replace(' ', '-') if ' ' in code else re.sub(r'(?<=[A-Za-z])(?=\d)', '-', code, count=1))
    else:
        text = brand + code
    return form, text


slots = []
for _ in range(60):
    d = rng.choice(PRODUCTS)
    attr = pick(SINGLE_ATTRS)
    while attr == 'activity' and d in NO_ACTIVITY:
        attr = pick(SINGLE_ATTRS)
    sub = rng.choice(SUB[attr]) if attr in SUB else ''
    form, text = render(d)
    family_word = FAMILY_OF[d] if rng.random() < 0.3 else ''
    slots.append(('single', attr + (f':{sub}' if sub else ''), d, text, form,
                  f'also names its family ({family_word})' if family_word else '', lang()))
for _ in range(8):
    a, b = rng.sample(PRODUCTS, 2)
    attr = pick(TWO_PRODUCT_ATTRS)
    while attr == 'activity' and {a, b} & NO_ACTIVITY:
        attr = pick(TWO_PRODUCT_ATTRS)
    slots.append(('two products', attr, f'{a} + {b}', f'{render(a)[1]} | {render(b)[1]}', '', '', lang()))
for _ in range(6):
    d = rng.choice(PRODUCTS)
    family = rng.choice([f for f in list(FAMILIES) + ['ascorbic acid'] if f != FAMILY_OF[d]])
    slots.append(('product + family', pick(PRODUCT_FAMILY_ATTRS), f'{d} + {family}', render(d)[1], '', '', lang()))

if __name__ == '__main__':
    for i, s in enumerate(slots, 1):
        print(f'{i:3d}', s)
    print(len(slots), 'slots;', sum(s[-1] == 'fr' for s in slots), 'fr')
