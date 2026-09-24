"""
Ground-truth evaluation set for the BVZyme / ascorbic-acid corpus.

Relevance is defined independently of any chunking strategy: a retrieved
fragment is relevant to a target when
  1. it comes from one of the target's source PDFs, and
  2. its text contains the actual answer (checked with a regex on a
     "squashed" copy of the text: lower-case, all whitespace removed, so
     PDF spacing defects such as "10-10 0 pp m" do not matter).

Dosage answers are accepted with or without the "ppm" unit, which is
lenient towards fragments that drop the unit.

Two query sets:
  DEV  - the 16 queries published in the original README (the previous
         pipeline was developed against these; the challenge example is here).
  TEST - held-out queries written for this audit before any new pipeline
         was built. Nothing was tuned on them.
Queries whose answer is not in the corpus are marked unanswerable and are
only used to check whether similarity scores separate answerable from
unanswerable questions.
"""
import re

# ── Document keys (derived from PDF file names) ────────────────────────────

def doc_key(filename):
    """'BVZyme TDS AF330.pdf' -> 'af330', 'acide ascorbique.pdf' -> 'aa'."""
    k = re.sub(r'\s+', '', filename.lower())
    k = re.sub(r'\.pdf$', '', k)
    for junk in ('bvzyme', 'tds', 'pdf', '(1)'):
        k = k.replace(junk, '')
    return 'aa' if k == 'acideascorbique' else k


AF = ['af110', 'af220', 'af330', 'afsx']                        # fungal alpha-amylase
MALTO = ['afresh101', 'afresh202', 'afresh303',
         'asoft205', 'asoft305', 'asoft405']                     # maltogenic amylase
AMG = ['amg880', 'amg1400']                                     # amyloglucosidase
GO = ['gox110', 'gomax63', 'gomax65']                           # glucose oxidase
XYL = ['hcb708', 'hcb709', 'hcb710', 'hcf400', 'hcf500', 'hcf600',
       'hcfmax63', 'hcfmax64', 'hcfmaxx']                        # xylanase
LIP = ['l55', 'l65', 'lmax63', 'lmax64', 'lmax65', 'lmaxx']     # lipase
TG = ['tg881', 'tg883', 'tgmax63', 'tgmax64']                   # transglutaminase
AA = ['aa']                                                     # ascorbic acid (FR doc)
TDS = AF + MALTO + AMG + GO + XYL + LIP + TG                    # the 34 English TDS
ALL_DOCS = TDS + AA

# Dosage ranges exactly as printed in each PDF (every tier is accepted).
DOSAGE = {
    'af110': ['2-12'], 'af220': ['2-10'], 'af330': ['2-10'], 'afsx': ['5-25'],
    'afresh101': ['15-100'], 'afresh202': ['10-90'], 'afresh303': ['15-100'],
    'asoft205': ['15-100'], 'asoft305': ['15-50', '30ppm'], 'asoft405': ['15-90'],
    'amg880': ['10-100'], 'amg1400': ['10-100'],
    'gox110': ['5-40'], 'gomax63': ['5-50'], 'gomax65': ['5-40'],
    'hcb708': ['5-30'], 'hcb709': ['5-20'], 'hcb710': ['5-20'],
    'hcf400': ['5-20', '10-70', '15-35'], 'hcf500': ['5-20', '10-70', '15-35'],
    'hcfmax63': ['5-20', '10-70', '15-35'],
    'hcf600': ['1-5', '2-35', '2-10'], 'hcfmax64': ['1-5', '2-35', '2-10'],
    'hcfmaxx': ['0.5-2', '1-15'],
    'l55': ['5-50'], 'l65': ['5-50'], 'lmax63': ['5-30'], 'lmax64': ['5-60'],
    'lmax65': ['5-50'], 'lmaxx': ['2-20'],
    'tg881': ['10-40'], 'tg883': ['5-30'], 'tgmax63': ['5-25'], 'tgmax64': ['5-20'],
    'aa': ['20-60', '60-80', '80-100', '150-200', '50-75', '30-50'],
}

ACTIVITY = {'af110': '150000skb/g', 'af220': '11000fau/g', 'af330': '11900fau/g',
            'afsx': '85000skb/g', 'hcfmaxx': '23500xylh/g', 'hcb708': '568xylh/g'}


def _num(r):
    """Regex for a number/range not glued to other digits ('2-10' != '2-100')."""
    return r'(?<![\d.])' + re.escape(r) + r'(?!\d)'


def dosage_regex(key):
    pats = [_num(r) for r in DOSAGE[key]]
    if key == 'aa':
        pats.append(r'paindemie\(cbp\)75')
    return '|'.join(pats)


# A target = (label, docs, answer) where answer is a regex string applied to
# every doc, or a dict {doc_key: regex}.
def T(label, docs, answer):
    return {'label': label, 'docs': list(docs), 'answer': answer}


def dosage_target(label, docs):
    return T(label, docs, {d: dosage_regex(d) for d in docs})


def Q(qid, lang, q, targets, answerable=True):
    return {'id': qid, 'lang': lang, 'q': q, 'targets': targets, 'answerable': answerable}


# ── DEV: the 16 queries of the original README ─────────────────────────────
DEV = [
    Q('D01', 'fr', "Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?",
      [dosage_target('alpha-amylase', AF), dosage_target('xylanase', XYL), dosage_target('acide ascorbique', AA)]),
    Q('D02', 'fr', "Quel dosage de lipase pour la panification ?", [dosage_target('lipase', LIP)]),
    Q('D03', 'fr', "Quel dosage de xylanase en boulangerie ?", [dosage_target('xylanase', XYL)]),
    Q('D04', 'fr', "À quoi sert l'acide ascorbique en boulangerie ?",
      [T('role AA', AA, r"améliorantdepanification|oxydante|renforce|volume")]),
    Q('D05', 'fr', "Quel est l'effet de la xylanase sur le volume du pain ?",
      [T('xylanase volume', XYL, r"loafvolume|increase(overall)?volume|ovenspring")]),
    Q('D06', 'en', "What is the recommended dosage of alpha-amylase for bread?", [dosage_target('alpha-amylase', AF)]),
    Q('D07', 'en', "What are the storage conditions for BVZyme AF110?",
      [T('storage AF110', ['af110'], r"cool,?dry|coolanddry|20°c")]),
    Q('D08', 'fr', "Quelle est la dose recommandée de transglutaminase ?", [dosage_target('transglutaminase', TG)]),
    Q('D09', 'en', "Does BVZyme contain allergens?", [T('allergen gluten', TDS, r"allergens?:?gluten")]),
    Q('D10', 'en', "What is the optimal pH for xylanase activity?", [], answerable=False),
    Q('D11', 'en', "How does lipase improve bread texture?",
      [T('lipase texture', LIP, r"crumbstructure|softness|doughhandling")]),
    Q('D12', 'en', "What is the shelf life of BVZyme enzymes?", [T('shelf life', TDS, r"24months")]),
    Q('D13', 'fr', "Quelle est l'activité enzymatique de l'alpha-amylase ?",
      [T('AF activity', AF, {d: re.escape(ACTIVITY[d]) for d in AF})]),
    Q('D14', 'en', "What is the microbial source of BVZyme xylanase?",
      [T('xylanase source', XYL, {d: ('bacillussubtilis' if d.startswith('hcb') else 'aspergillus') for d in XYL})]),
    Q('D15', 'en', "How to combine alpha-amylase and xylanase for bread?", [], answerable=False),
    Q('D16', 'en', "What packaging is used for BVZyme products?", [T('packaging', TDS, r"cartonboxof25kg")]),
]

# ── TEST: held-out queries written for this audit ──────────────────────────
TEST = [
    # dosage, enzyme family level
    Q('T01', 'fr', "Quelle quantité d'alpha-amylase faut-il ajouter à la farine ?", [dosage_target('alpha-amylase', AF)]),
    Q('T02', 'en', "How much fungal alpha-amylase should be added to the flour?", [dosage_target('alpha-amylase', AF)]),
    Q('T03', 'fr', "Combien de xylanase utiliser dans une recette de pain ?", [dosage_target('xylanase', XYL)]),
    Q('T04', 'en', "Recommended xylanase dose for bread improvement", [dosage_target('xylanase', XYL)]),
    Q('T05', 'fr', "Dosage conseillé de la lipase en panification", [dosage_target('lipase', LIP)]),
    Q('T06', 'en', "What amount of glucose oxidase is typically used in bread dough?", [dosage_target('glucose oxidase', GO)]),
    Q('T07', 'fr', "Quelle est la dose d'emploi de la transglutaminase ?", [dosage_target('transglutaminase', TG)]),
    Q('T08', 'fr', "Quel dosage pour l'amylase maltogénique anti-rassissement ?", [dosage_target('maltogenic amylase', MALTO)]),
    Q('T09', 'en', "Amyloglucosidase dosage in baking", [dosage_target('amyloglucosidase', AMG)]),
    Q('T10', 'fr', "Quelle quantité d'acide ascorbique ajouter à la farine ?", [dosage_target('acide ascorbique', AA)]),
    Q('T11', 'en', "How much vitamin C (E300) should be used as a flour improver?", [dosage_target('acide ascorbique', AA)]),
    # dosage, product level
    Q('T12', 'fr', "Quel est le dosage du BVZyme TG881 ?", [dosage_target('TG881', ['tg881'])]),
    Q('T13', 'en', "Dosage of BVZyme HCF MAX X for bread improvement", [T('HCF MAX X bread', ['hcfmaxx'], _num('1-15'))]),
    Q('T14', 'fr', "BVZyme A SOFT305 : quel dosage est suggéré ?", [dosage_target('A SOFT305', ['asoft305'])]),
    Q('T15', 'en', "What is the recommended dosage for BVZyme GO MAX 63?", [dosage_target('GO MAX 63', ['gomax63'])]),
    Q('T16', 'fr', "Dose recommandée du BVZyme L MAX64", [dosage_target('L MAX64', ['lmax64'])]),
    # ascorbic acid document
    Q('T17', 'fr', "Quel dosage d'acide ascorbique pour une pâte surgelée ?", [T('AA frozen', AA, _num('150-200'))]),
    Q('T18', 'fr', "Quelle est la dose maximale autorisée d'acide ascorbique en France ?", [T('AA max', AA, r"maximum.{0,30}300|belgique:?300")]),
    Q('T19', 'en', "What is the maximum legal dose of ascorbic acid in bread?", [T('AA max', AA, r"maximum.{0,30}300|belgique:?300")]),
    Q('T20', 'fr', "Combien de grammes d'acide ascorbique pour 100 kg de farine à 50 ppm ?", [T('AA conversion', AA, r"100kg(?:\|50ppm:)?5g")]),
    Q('T21', 'fr', "À quel moment faut-il incorporer l'acide ascorbique lors du pétrissage ?", [T('AA incorporation', AA, r"avanthydratation|ingrédientssecs")]),
    Q('T22', 'en', "Is ascorbic acid destroyed during baking?", [T('AA baking loss', AA, r"détruitpendantlacuisson|pertetotalependantlacuisson")]),
    Q('T23', 'fr', "Quels sont les risques d'un surdosage en acide ascorbique ?", [T('AA overdose', AA, r"suroxyder")]),
    Q('T24', 'fr', "Peut-on associer l'acide ascorbique à des enzymes ?", [T('AA + enzymes', AA, r"acideascorbique\+enzymes|syn[ée]rgie")]),
    Q('T25', 'en', "What is the pH of a 1% ascorbic acid solution?", [T('AA pH', AA, r"2,0-2,5")]),
    Q('T26', 'fr', "Dosage d'acide ascorbique pour la viennoiserie", [T('AA viennoiserie', AA, r"viennoiserie.{0,30}50-75")]),
    Q('T27', 'fr', "Comment stocker l'acide ascorbique ?", [T('AA storage', AA, r"15-25°c|frais,sec|endroitfraisetsec|endroitsec")]),
    # function / effect
    Q('T28', 'fr', "Quel est le rôle de la xylanase dans la pâte ?", [T('xylanase role', XYL, r"extensibility|gasretention|ovenspring|loafvolume|elasticity|increase(overall)?volume")]),
    Q('T29', 'en', "What does glucose oxidase do in bread dough?", [T('GO role', GO, r"strength|tolerance|glutennetwork")]),
    Q('T30', 'fr', "Quelle enzyme améliore la couleur de la croûte ?", [T('crust colour', AMG + AA, r"crustcolor|couleurdecro[uû]te")]),
    Q('T31', 'en', "Which enzyme keeps bread soft and fresh for longer?", [T('anti-staling', MALTO, r"freshness|softness|shelflife")]),
    Q('T32', 'fr', "Pourquoi utiliser de la transglutaminase en boulangerie ?", [T('TG role', TG, r"cross-link|glutennetwork|glutenstrength|elasticity|increasestrength")]),
    Q('T33', 'en', "How does lipase improve dough handling and crumb?", [T('lipase role', LIP, r"doughhandling|crumbstructure|softness")]),
    Q('T34', 'fr', "À quoi sert l'amylase fongique dans la panification ?", [T('AF role', AF, r"damagedstarch|gassingpower|assistinfermentation|softness|increasevolume")]),
    Q('T35', 'en', "What is ascorbic acid used for in breadmaking?", [T('AA role', AA, r"améliorantdepanification|oxydante|renforce|volume")]),
    # source / activity
    Q('T36', 'fr', "Quelle est l'origine de la xylanase bactérienne BVZyme HCB ?", [T('HCB source', ['hcb708', 'hcb709', 'hcb710'], r"bacillussubtilis")]),
    Q('T37', 'en', "Which microorganism is used to produce the fungal alpha-amylase?", [T('AF organism', AF, r"aspergillus")]),
    Q('T38', 'fr', "Quelle est l'activité enzymatique du BVZyme HCF MAX X ?", [T('HCF MAX X activity', ['hcfmaxx'], re.escape(ACTIVITY['hcfmaxx']))]),
    Q('T39', 'en', "What is the activity of BVZyme AF110 in SKB units?", [T('AF110 activity', ['af110'], re.escape(ACTIVITY['af110']))]),
    Q('T40', 'fr', "Quelle souche produit l'amylase maltogénique ?", [T('malto organism', MALTO, r"bacillus")]),
    # safety / regulatory / logistics (any TDS answers)
    Q('T41', 'fr', "Les enzymes BVZyme contiennent-elles du gluten ?", [T('allergen gluten', TDS, r"allergens?:?gluten")]),
    Q('T42', 'en', "Are these enzyme preparations GMO?", [T('GMO', TDS, r"1829/2003|nospecificlabeling")]),
    Q('T43', 'fr', "Les produits sont-ils irradiés ?", [T('irradiation', TDS, r"withoutirradiation")]),
    Q('T44', 'en', "What are the microbiological limits for Salmonella?", [T('salmonella', TDS, r"salmonella:absentin25g")]),
    Q('T45', 'fr', "Quelles sont les teneurs maximales en métaux lourds comme le plomb ou l'arsenic ?", [T('heavy metals', TDS, r"lead:<5mg/kg|arsenic:<3mg/kg")]),
    Q('T46', 'en', "What is the maximum moisture content of the enzyme powder?", [T('moisture', TDS, r"moisture:<15%")]),
    Q('T47', 'fr', "Quelle est la date de durabilité minimale des enzymes ?", [T('durability', TDS, r"24months")]),
    Q('T48', 'en', "At what temperature should the enzymes be stored?", [T('storage temp', TDS, r"20°c")]),
    Q('T49', 'fr', "Sous quel conditionnement les enzymes sont-elles livrées ?", [T('packaging', TDS, r"cartonboxof25kg")]),
    Q('T50', 'en', "What does the enzyme powder look like?", [T('aspect', TDS, r"freeflowingpowder")]),
    # multi-entity
    Q('M02', 'en', "Bread improver: what are the recommended amounts of alpha-amylase, xylanase and ascorbic acid?",
      [dosage_target('alpha-amylase', AF), dosage_target('xylanase', XYL), dosage_target('acide ascorbique', AA)]),
    Q('M03', 'fr', "Quels dosages utiliser pour la lipase et la glucose oxydase dans un améliorant ?",
      [dosage_target('lipase', LIP), dosage_target('glucose oxidase', GO)]),
    Q('M04', 'en', "Recommended doses of transglutaminase and xylanase for bread",
      [dosage_target('transglutaminase', TG), dosage_target('xylanase', XYL)]),
    Q('M05', 'fr', "Pour un pain de mie moelleux, quelles quantités d'amylase maltogénique et de lipase ?",
      [dosage_target('maltogenic amylase', MALTO), dosage_target('lipase', LIP)]),
    # unanswerable from this corpus
    Q('U01', 'en', "What is the optimal temperature for transglutaminase activity?", [], answerable=False),
    Q('U02', 'fr', "Quel est le prix d'un sac de 25 kg de xylanase ?", [], answerable=False),
    Q('U03', 'fr', "Quelle est la durée de pétrissage recommandée avec la lipase ?", [], answerable=False),
]


# ── Relevance ──────────────────────────────────────────────────────────────

def squash(text):
    t = text.lower().replace('–', '-').replace('—', '-').replace('：', ':')
    return re.sub(r'\s+', '', t)


def relevant(target, doc, text):
    if doc not in target['docs']:
        return False
    ans = target['answer']
    pattern = ans.get(doc) if isinstance(ans, dict) else ans
    return bool(pattern) and re.search(pattern, squash(text)) is not None
