"""
TEST-5: a fifth held-out set, frozen with ROUTES_PLAN.md before any of the
extraction or chunking routes it tests is run.

Slots (scope, attribute, target, language, and how a product code is written)
were drawn at random with a fixed seed (audit/sample_test5_slots.py,
random.Random(20260927)); 100 slots + 4 unanswerable questions added by hand.
Relevance works as in eval_set.py: right source PDF + the actual answer.
"""
import re

from audit.eval_set import AA, T, Q
from audit.eval_set_v2 import AA_DOSAGE, AA_ROLE, FAMILY, FUNCTION, fam
from audit.eval_set_v4 import ACT, ENZYME, P, SHARED

TDS = [d for ds in FAMILY.values() for d in ds]
APPLICATION = {  # words of each family's Application line (squashed text)
    'alpha-amylase': r"damagedstarch", 'maltogenic amylase': r"freshnessofbread",
    'amyloglucosidase': r"glucosidiclinkages", 'glucose oxidase': r"glutennetworks|strengthensgluten",
    'xylanase': r"bakeryandbread", 'lipase': r"esterbonds", 'transglutaminase': r"cross-linking|proteincross",
}


def F(family, attr):
    """Target: attribute `attr` of a product family (any of its sheets)."""
    if family == 'ascorbic acid':
        return {'dosage': AA_DOSAGE, 'function': T('ascorbic acid role', AA, AA_ROLE)}[attr]
    docs = FAMILY[family]
    if attr in ('dosage', 'function', 'source', 'storage'):
        return fam(family, attr)
    if attr == 'activity':
        docs = [d for d in docs if d in ACT]
        return T(f'{family} activity', docs, {d: r'(?<![\d.])' + re.escape(ACT[d]) for d in docs})
    if attr == 'application':
        return T(f'{family} application', docs, APPLICATION[family] + '|' + FUNCTION[family])
    if attr == 'enzyme_type':
        return T(f'{family} enzyme', docs, ENZYME[family])
    raise ValueError(attr)


def C(attr):
    """Target: a specification common to all 34 data sheets."""
    return T(attr, TDS, SHARED.get(attr) or {'staphylococcus': r"staphylococcusaureus:absentin1g",
                                              'salmonella': r"salmonella:absentin25g",
                                              'cadmium': r"cadmium:<0,5mg/kg",
                                              'aspect': r"freeflowingpowder|color:white"}[attr])


def A(label, answer):
    return T(f'AA {label}', AA, answer)


TEST5 = [
    # product / family attributes (slots 1-44)
    Q('R001', 'fr', "Quelle dose de glucose oxydase recommandez-vous pour du pain ?", [F('glucose oxidase', 'dosage')]),
    Q('R002', 'en', "How many ppm of A FRESH101 should be used?", [P('afresh101', 'dosage')]),
    Q('R003', 'fr', "À quel dosage emploie-t-on l'amyloglucosidase ?", [F('amyloglucosidase', 'dosage')]),
    Q('R004', 'fr', "Glucose oxydase : quel taux d'utilisation conseillé ?", [F('glucose oxidase', 'dosage')]),
    Q('R005', 'en', "What is the recommended dosage of HCFMAX64 for bread improvement?", [P('hcfmax64', 'dosage')]),
    Q('R006', 'fr', "Quelle quantité de lipase ajouter à la farine ?", [F('lipase', 'dosage')]),
    Q('R007', 'en', "What xylanase dosage is suggested for bread?", [F('xylanase', 'dosage')]),
    Q('R008', 'fr', "Quel est le dosage du bvzyme go max 63 ?", [P('gomax63', 'dosage')]),
    Q('R009', 'fr', "Combien de ppm de bvzyme af330 conseillez-vous ?", [P('af330', 'dosage')]),
    Q('R010', 'fr', "Dosage recommandé des xylanases pour l'amélioration du pain ?", [F('xylanase', 'dosage')]),
    Q('R011', 'fr', "Quelle quantité de BVZyme TG883 mettre dans une pâte ?", [P('tg883', 'dosage')]),
    Q('R012', 'en', "Suggested addition rate for BVZyme AF330?", [P('af330', 'dosage')]),
    Q('R013', 'en', "How much BVZyme L MAX65 is recommended?", [P('lmax65', 'dosage')]),
    Q('R014', 'en', "What dosage range applies to BVZyme HCF500?", [P('hcf500', 'dosage')]),
    Q('R015', 'en', "How much xylanase should I add to standardize wheat flour?", [F('xylanase', 'dosage')]),
    Q('R016', 'fr', "Dose conseillée pour le BVZyme GOX-110 ?", [P('gox110', 'dosage')]),
    Q('R017', 'en', "What are the effects of BVZyme GOMAX63 on dough?", [P('gomax63', 'function')]),
    Q('R018', 'en', "What does maltogenic amylase do in bread?", [F('maltogenic amylase', 'function')]),
    Q('R019', 'fr', "Quel est le rôle de l'amyloglucosidase en panification ?", [F('amyloglucosidase', 'function')]),
    Q('R020', 'en', "Why use BVZyme L MAX X in a bread improver?", [P('lmaxx', 'function')]),
    Q('R021', 'en', "What are the functions of alpha-amylase in baking?", [F('alpha-amylase', 'function')]),
    Q('R022', 'fr', "Qu'apporte la lipase à la pâte ?", [F('lipase', 'function')]),
    Q('R023', 'fr', "À quoi sert la glucose oxydase dans le pain ?", [F('glucose oxidase', 'function')]),
    Q('R024', 'en', "How does GO-MAX-65 improve the dough?", [P('gomax65', 'function')]),
    Q('R025', 'fr', "Quels bénéfices apporte le HCF MAX X à la pâte ?", [P('hcfmaxx', 'function')]),
    Q('R026', 'en', "In which applications is xylanase used?", [F('xylanase', 'application')]),
    Q('R027', 'en', "What is BVZyme L55 used for in bakery?",
      [T('l55 application', ['l55'], APPLICATION['lipase'] + '|' + FUNCTION['lipase'])]),
    Q('R028', 'en', "What is glucose oxidase used for in bakery?", [F('glucose oxidase', 'application')]),
    Q('R029', 'fr', "De quel micro-organisme provient la glucose oxydase ?", [F('glucose oxidase', 'source')]),
    Q('R030', 'en', "Which strain produces BVZyme L 65?", [P('l65', 'source')]),
    Q('R031', 'fr', "Comment est fabriquée la glucose oxydase BVZyme ?", [F('glucose oxidase', 'source')]),
    Q('R032', 'fr', "Quelle est l'origine de l'amyloglucosidase ?", [F('amyloglucosidase', 'source')]),
    Q('R033', 'en', "What organisms are the xylanases produced from?", [F('xylanase', 'source')]),
    Q('R034', 'en', "What is the enzyme activity of A SOFT405?", [P('asoft405', 'activity')]),
    Q('R035', 'en', "What is the activity of BVZyme AMG880?", [P('amg880', 'activity')]),
    Q('R036', 'en', "What is the activity of the amyloglucosidase products?", [F('amyloglucosidase', 'activity')]),
    Q('R037', 'fr', "Quelle est l'activité enzymatique de la transglutaminase ?", [F('transglutaminase', 'activity')]),
    Q('R038', 'fr', "Quels produits BVZyme contiennent de l'alpha-amylase ?", [F('alpha-amylase', 'enzyme_type')]),
    Q('R039', 'fr', "Le HCF MAX X est à base de quelle enzyme ?", [P('hcfmaxx', 'enzyme_type')]),
    Q('R040', 'fr', "Quelles références BVZyme sont des amyloglucosidases ?", [F('amyloglucosidase', 'enzyme_type')]),
    Q('R041', 'en', "How should maltogenic amylase be stored?", [F('maltogenic amylase', 'storage')]),
    Q('R042', 'fr', "Quelle est la durée de conservation des xylanases ?", [F('xylanase', 'storage')]),
    Q('R043', 'fr', "Dans quelles conditions conserver la glucose oxydase ?", [F('glucose oxidase', 'storage')]),
    Q('R044', 'fr', "Combien de temps se conserve le BVZyme L-MAX64 ?", [P('lmax64', 'storage')]),
    # corpus-wide specifications (slots 45-60)
    Q('R045', 'en', "Do BVZyme enzymes contain any allergens?", [C('allergens')]),
    Q('R046', 'fr', "Quels allergènes sont présents dans les enzymes BVZyme ?", [C('allergens')]),
    Q('R047', 'fr', "Les enzymes BVZyme doivent-elles être étiquetées OGM ?", [C('gmo')]),
    Q('R048', 'fr', "Les préparations enzymatiques sont-elles traitées par ionisation ?", [C('irradiation')]),
    Q('R049', 'fr', "Comment sont emballées les enzymes BVZyme ?", [C('packaging')]),
    Q('R050', 'en', "What packaging are the BVZyme enzymes supplied in?", [C('packaging')]),
    Q('R051', 'fr', "Quelle est la norme pour les staphylocoques dans les enzymes ?", [C('staphylococcus')]),
    Q('R052', 'fr', "Les enzymes BVZyme sont-elles exemptes de salmonelles ?", [C('salmonella')]),
    Q('R053', 'fr', "Quelle est la flore totale maximale des enzymes BVZyme ?", [C('total plate count')]),
    Q('R054', 'fr', "Quelle est la limite de plomb dans les enzymes ?", [C('lead')]),
    Q('R055', 'en', "What is the cadmium limit for the enzyme preparations?", [C('cadmium')]),
    Q('R056', 'en', "How much moisture can the enzyme powder contain?", [C('moisture')]),
    Q('R057', 'en', "What does the enzyme powder look like?", [C('aspect')]),
    Q('R058', 'en', "Which company produces the BVZyme range?", [C('manufacturer')]),
    Q('R059', 'fr', "Quand les fiches techniques BVZyme ont-elles été mises à jour pour la dernière fois ?",
      [C('last update')]),
    Q('R060', 'fr', "Date de révision des fiches BVZyme ?", [C('last update')]),
    # ascorbic acid (slots 61-88)
    Q('R061', 'fr', "Quel dosage d'acide ascorbique pour un pain de mie CBP ?", [A('CBP', r"paindemie\(cbp\).{0,25}75")]),
    Q('R062', 'en', "What ascorbic acid dosage for slow-proof bread making?", [A('slow proof', r"avecpousse.{0,30}60-80")]),
    Q('R063', 'en', "How much ascorbic acid for dough kept in positive cold storage at 2°C?",
      [A('cold', r"blocagefroid.{0,40}80-100")]),
    Q('R064', 'en', "Ascorbic acid dose for sandwich bread made with the CBP process?",
      [A('CBP', r"paindemie\(cbp\).{0,25}75")]),
    Q('R065', 'en', "How many ppm of vitamin C in CBP pan bread?", [A('CBP', r"paindemie\(cbp\).{0,25}75")]),
    Q('R066', 'fr', "Combien d'acide ascorbique en panification directe ?", [A('direct', r"directestandard.{0,30}20-60")]),
    Q('R067', 'en', "Recommended ascorbic acid dosage for frozen dough?", [A('frozen', r"surgélation.{0,30}150-200")]),
    Q('R068', 'fr', "Quelle est la dose maximale d'acide ascorbique autorisée en France ?",
      [A('max', r"maximum.{0,30}300|belgique:?300")]),
    Q('R069', 'en', "Is there a legal maximum for ascorbic acid in flour?", [A('max', r"maximum.{0,30}300|belgique:?300")]),
    Q('R070', 'fr', "Combien de grammes d'acide ascorbique pour 100 kg de farine à 100 ppm ?",
      [A('conversion 100 kg', r"(?<!\d)100kg.{0,45}10g")]),
    Q('R071', 'fr', "Pour 50 kg de farine, combien de grammes d'acide ascorbique à 100 ppm ?",
      [A('conversion 50 kg', r"(?<!\d)50kg.{0,45}100ppm:5g|(?<!\d)50kg2,5g3,75g5g")]),
    Q('R072', 'fr', "Comment peser l'acide ascorbique ?", [A('weighing', r"±5%|balancedeprécision")]),
    Q('R073', 'fr', "Au bout de combien de temps l'acide ascorbique agit-il ?", [A('action time', r"5-15minutes")]),
    Q('R074', 'en', "When should ascorbic acid be added during mixing?",
      [A('incorporation', r"ingrédientssecs|avanthydratation|dryingredient|beforehydration")]),
    Q('R075', 'fr', "Peut-on dissoudre l'acide ascorbique dans l'eau ?", [A('dilution', r"solution1-2%|diluer")]),
    Q('R076', 'fr', "Faut-il diluer la vitamine C avant de l'incorporer ?", [A('dilution', r"solution1-2%|diluer")]),
    Q('R077', 'fr', "Quel pH a une solution d'acide ascorbique ?", [A('pH', r"2,0-2,5")]),
    Q('R078', 'en', "What is the chemical formula of ascorbic acid?", [A('formula', r"c[6₆]h[8₈]o[6₆]")]),
    Q('R079', 'fr', "L'acide ascorbique est-il soluble dans l'eau ?",
      [A('solubility', r"trèssolubledansl'eau|verysoluble")]),
    Q('R080', 'fr', "Formule moléculaire de l'acide L-ascorbique ?", [A('formula', r"c[6₆]h[8₈]o[6₆]")]),
    Q('R081', 'en', "At what temperature and humidity should ascorbic acid be stored?",
      [A('storage', r"15-25°c|humidité:?<60%")]),
    Q('R082', 'fr', "Quelle température de stockage pour l'acide ascorbique ?", [A('storage', r"15-25°c|humidité:?<60%")]),
    Q('R083', 'fr', "Quelle humidité maximale pour stocker l'acide ascorbique ?",
      [A('storage', r"15-25°c|humidité:?<60%")]),
    Q('R084', 'fr', "Quel est l'effet de l'acide ascorbique sur le gluten ?",
      [A('gluten', r"pontsdisulfur|oxydante|renforcelegluten")]),
    Q('R085', 'fr', "L'acide ascorbique aide-t-il la pâte à retenir le gaz ?", [A('volume', r"volume|rétentiondegaz")]),
    Q('R086', 'en', "Does ascorbic acid improve proofing tolerance?", [A('proofing', r"apprêt|fenêtredetravail")]),
    Q('R087', 'fr', "Quels avantages apporte l'acide ascorbique en boulangerie ?",
      [A('advantages', r"15-30%|standardisation|compatibleavectous|coûtmodéré|améliorationvisible")]),
    Q('R088', 'en', "What can replace ascorbic acid in a bread improver?",
      [A('alternatives', r"levainnaturel|malte|enzymes\(amylases\)")]),
    # several product families (slots 89-100)
    Q('R089', 'en', "Recommended doses of transglutaminase, lipase and maltogenic amylase?",
      [F('transglutaminase', 'dosage'), F('lipase', 'dosage'), F('maltogenic amylase', 'dosage')]),
    Q('R090', 'en', "What dosages for amyloglucosidase and alpha-amylase?",
      [F('amyloglucosidase', 'dosage'), F('alpha-amylase', 'dosage')]),
    Q('R091', 'fr', "Quelles quantités de lipase et d'amylase maltogénique utiliser ?",
      [F('lipase', 'dosage'), F('maltogenic amylase', 'dosage')]),
    Q('R092', 'fr', "Quel dosage pour la lipase et la xylanase ?", [F('lipase', 'dosage'), F('xylanase', 'dosage')]),
    Q('R093', 'en', "How much ascorbic acid and amyloglucosidase should I add to my flour?",
      [F('ascorbic acid', 'dosage'), F('amyloglucosidase', 'dosage')]),
    Q('R094', 'fr', "Quels sont les effets de la xylanase et de la lipase sur le pain ?",
      [F('xylanase', 'function'), F('lipase', 'function')]),
    Q('R095', 'en', "What do transglutaminase and maltogenic amylase do in bread?",
      [F('transglutaminase', 'function'), F('maltogenic amylase', 'function')]),
    Q('R096', 'en', "What are the roles of amyloglucosidase and ascorbic acid in dough?",
      [F('amyloglucosidase', 'function'), F('ascorbic acid', 'function')]),
    Q('R097', 'en', "Which microorganisms produce alpha-amylase and maltogenic amylase?",
      [F('alpha-amylase', 'source'), F('maltogenic amylase', 'source')]),
    Q('R098', 'en', "Storage conditions for xylanase and maltogenic amylase?",
      [F('xylanase', 'storage'), F('maltogenic amylase', 'storage')]),
    Q('R099', 'en', "How long do xylanase and alpha-amylase keep, and how should they be stored?",
      [F('xylanase', 'storage'), F('alpha-amylase', 'storage')]),
    Q('R100', 'fr', "Quelle est l'activité de l'amylase maltogénique et de la transglutaminase ?",
      [F('maltogenic amylase', 'activity'), F('transglutaminase', 'activity')]),
    # not answered by the documents (added by hand)
    Q('N01', 'en', "What is the optimal pH of BVZyme HCF400?", [], answerable=False),
    Q('N02', 'fr', "Quel est le prix de l'acide ascorbique au kilo ?", [], answerable=False),
    Q('N03', 'fr', "Les enzymes BVZyme sont-elles certifiées bio ?", [], answerable=False),
    Q('N04', 'en', "What is the minimum order quantity for BVZyme AF220?", [], answerable=False),
]
