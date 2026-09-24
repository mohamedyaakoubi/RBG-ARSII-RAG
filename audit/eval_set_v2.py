"""
TEST-2: a second held-out set, frozen before the in-rules optimisation.

DEV and TEST (eval_set.py) were used to study failures, so they are now the
development pool. TEST-2 is only run once, at the end.

What to ask was not hand-picked: 80 (scope, attribute, target, language)
slots were drawn at random with a fixed seed (random.Random(20260924);
65% French; weights favour the attributes formulators ask about most).
The draw is kept as is, duplicates included; repeated slots get different
wording. 5 unanswerable questions were added by hand. Relevance works as in
eval_set.py: right source PDF + fragment text contains the actual answer.
"""
import re

from audit.eval_set import (AF, MALTO, AMG, GO, XYL, LIP, TG, AA, TDS,
                            T, Q, dosage_target)

FAMILY = {'alpha-amylase': AF, 'maltogenic amylase': MALTO, 'amyloglucosidase': AMG,
          'glucose oxidase': GO, 'xylanase': XYL, 'lipase': LIP, 'transglutaminase': TG}

FUNCTION = {  # words of each family's Function / Application lines (squashed text)
    'alpha-amylase': r"crumbsoftness|improvetexture|increasevolume|gassingpower|enhancesoftness|assistinfermentation|damagedstarch",
    'maltogenic amylase': r"freshness|softness|shelflife|resilience|elasticity",
    'amyloglucosidase': r"crustcolor|golden|crumbcolor|ovenspring|glucosidic|hydrolyzes",
    'glucose oxidase': r"strength|tolerance|stability|glutennetwo?r?ks",
    'xylanase': r"extensibility|gasretention|ovenspring|loafvolume|increase(overall)?volume|increasetolerance|enhancetolerance|bakingperformance|stability|elasticity",
    'lipase': r"increasevolume|crumbstructure|stability|tolerance|doughhandling|softness|esterbonds",
    'transglutaminase': r"cross-link|glutennetwork|glutenstrength|elasticity|increasestrength|improvethevolume",
}
SOURCE = {'af110': 'aspergillus', 'af220': 'aspergillus', 'af330': 'aspergillus', 'afsx': 'aspergillus',
          'amg880': 'aspergillus', 'amg1400': 'aspergillus',
          'gox110': 'aspergillus', 'gomax63': 'aspergillus', 'gomax65': 'aspergillus',
          'hcb708': 'bacillussubtilis|bacterial', 'hcb709': 'bacillussubtilis|bacterial', 'hcb710': 'bacillussubtilis|bacterial',
          'hcf400': 'aspergillus', 'hcf500': 'aspergillus', 'hcf600': 'aspergillus',
          'hcfmax63': 'aspergillus', 'hcfmax64': 'aspergillus', 'hcfmaxx': 'aspergillus',
          'tg881': 'fermentions|fermentation|selectedstrain', 'tg883': 'fermentions|fermentation|selectedstrain',
          'tgmax63': 'fermentions|fermentation|selectedstrain', 'tgmax64': 'fermentions|fermentation|selectedstrain'}
SOURCE.update({d: 'bacillus' for d in MALTO})
SOURCE.update({d: 'aspergillus' for d in LIP})
ACTIVITY = {'af110': '150000skb/g', 'af220': '11000fau/g', 'af330': '11900fau/g', 'afsx': '85000skb/g',
            'hcf400': '2040xylh/g', 'l55': '1120u/g', 'l65': '1080u/g', 'lmax63': '1255u/g',
            'lmax64': '47000u/g', 'lmax65': '50000u/g', 'lmaxx': '3300u/g'}
STORAGE = r"24months|cool,?dry|coolanddry|20°c"
AA_ROLE = r"améliorantdepanification|oxydante|renforce|volume|pontsdisulfur"


def fam(name, attr, docs=None):
    docs = docs or FAMILY[name]
    if attr == 'dosage':
        return dosage_target(name, docs)
    if attr == 'function':
        return T(f'{name} function', docs, FUNCTION[name])
    if attr == 'source':
        return T(f'{name} source', docs, {d: SOURCE[d] for d in docs})
    if attr == 'activity':
        return T(f'{name} activity', docs, {d: re.escape(ACTIVITY[d]) for d in docs})
    if attr == 'storage':
        return T(f'{name} storage', docs, STORAGE)
    raise ValueError(attr)


AA_DOSAGE = T('ascorbic acid dosage', AA,
              r"(?<![\d.])(20-60|60-80|80-100|150-200|50-75|30-50)(?!\d)|paindemie\(cbp\).{0,20}75")

TEST2 = [
    # product / family attributes (random draw, slots 1-32)
    Q('V01', 'fr', "Quelle dose de BVZyme A SOFT405 faut-il utiliser ?", [fam('A SOFT405', 'dosage', ['asoft405'])]),
    Q('V02', 'fr', "Combien de transglutaminase ajouter pour renforcer une pâte ?", [fam('transglutaminase', 'dosage')]),
    Q('V03', 'en', "What is the usual dosage range for amyloglucosidase in bread?", [fam('amyloglucosidase', 'dosage')]),
    Q('V04', 'en', "How many ppm of xylanase are recommended?", [fam('xylanase', 'dosage')]),
    Q('V05', 'fr', "Quelle quantité de glucose oxydase recommandez-vous par rapport à la farine ?", [fam('glucose oxidase', 'dosage')]),
    Q('V06', 'fr', "Dosage de l'amyloglucosidase en panification ?", [fam('amyloglucosidase', 'dosage')]),
    Q('V07', 'fr', "Je voudrais savoir la dose d'amylase fongique à mettre dans ma farine.", [fam('alpha-amylase', 'dosage')]),
    Q('V08', 'fr', "Transglutaminase : quel taux d'incorporation ?", [fam('transglutaminase', 'dosage')]),
    Q('V09', 'fr', "Quel est le dosage recommandé pour le BVZyme GO MAX 65 ?", [fam('GO MAX 65', 'dosage', ['gomax65'])]),
    Q('V10', 'en', "Glucose oxidase: what dose should I use in my bread improver?", [fam('glucose oxidase', 'dosage')]),
    Q('V11', 'en', "How much glucose oxidase per tonne of flour?", [fam('glucose oxidase', 'dosage')]),
    Q('V12', 'en', "Typical alpha-amylase addition level for pan bread?", [fam('alpha-amylase', 'dosage')]),
    Q('V13', 'fr', "Qu'apporte la xylanase à la pâte à pain ?", [fam('xylanase', 'function')]),
    Q('V14', 'fr', "Quel est l'effet de la glucose oxydase sur la pâte ?", [fam('glucose oxidase', 'function')]),
    Q('V15', 'fr', "À quoi sert le BVZyme AMG1400 ?", [T('AMG1400 function', ['amg1400'], FUNCTION['amyloglucosidase'])]),
    Q('V16', 'fr', "Pourquoi ajouter de la glucose oxydase dans un améliorant ?", [fam('glucose oxidase', 'function')]),
    Q('V17', 'fr', "Quels bénéfices apporte la lipase en boulangerie ?", [fam('lipase', 'function')]),
    Q('V18', 'fr', "Quelles sont les fonctions du BVZyme HCF400 ?", [T('HCF400 function', ['hcf400'], r"increasevolume|increasetolerance|bakingperformance|stability")]),
    Q('V19', 'fr', "Dans quel but est conçu le BVZyme A SOFT305 ?", [T('A SOFT305 purpose', ['asoft305'], r"freshness|softness|shelflife")]),
    Q('V20', 'fr', "Comment agit le BVZyme GOX 110 sur le gluten ?", [T('GOX 110 action', ['gox110'], r"strengthens?glutennetworks|glutennetwork|strength")]),
    Q('V21', 'en', "How does amyloglucosidase work on starch?", [T('AMG mechanism', AMG, r"glucosidic|hydrolyzes|polysaccharides|non-reducing")]),
    Q('V22', 'fr', "Comment est produite la transglutaminase BVZyme ?", [fam('transglutaminase', 'source')]),
    Q('V23', 'en', "Which microorganism produces BVZyme AF SX?", [T('AF SX organism', ['afsx'], r"aspergillusoryzae")]),
    Q('V24', 'fr', "Quelle est l'origine microbienne de l'amyloglucosidase ?", [fam('amyloglucosidase', 'source')]),
    Q('V25', 'fr', "Le BVZyme HCB708 est-il d'origine bactérienne ou fongique ?", [T('HCB708 origin', ['hcb708'], r"bacterial|bacillussubtilis")]),
    Q('V26', 'en', "What is the enzymatic activity of the lipase products, in U/g?", [fam('lipase', 'activity')]),
    Q('V27', 'fr', "Quelle est l'activité du BVZyme AF110 ?", [T('AF110 activity', ['af110'], re.escape(ACTIVITY['af110']))]),
    Q('V28', 'fr', "Activité enzymatique du HCF400 en XylH/g ?", [T('HCF400 activity', ['hcf400'], re.escape(ACTIVITY['hcf400']))]),
    Q('V29', 'en', "Which BVZyme products are based on lipase?", [T('lipase products', LIP, r"lipase|lipolytic")]),
    Q('V30', 'fr', "Quels produits BVZyme sont à base d'amylase maltogénique ?", [T('maltogenic products', MALTO, r"maltogenicamylase")]),
    Q('V31', 'fr', "Comment conserver l'alpha-amylase, et combien de temps ?", [fam('alpha-amylase', 'storage')]),
    Q('V32', 'en', "What is the shelf life of BVZyme A FRESH101 and how should it be stored?", [fam('A FRESH101', 'storage', ['afresh101'])]),
    # corpus-level facts (slots 33-45)
    Q('V33', 'fr', "Y a-t-il des allergènes dans les enzymes BVZyme ?", [T('allergen gluten', TDS, r"allergens?:?gluten")]),
    Q('V34', 'fr', "Ces améliorants enzymatiques conviennent-ils à un régime sans gluten ?", [T('allergen gluten', TDS, r"allergens?:?gluten")]),
    Q('V35', 'en', "Do the BVZyme datasheets require GMO labelling?", [T('GMO', TDS, r"1829/2003|nospecificlabeling")]),
    Q('V36', 'fr', "Les enzymes ont-elles subi un traitement par ionisation ?", [T('irradiation', TDS, r"withoutirradiation")]),
    Q('V37', 'fr', "Quel est le conditionnement des enzymes BVZyme ?", [T('packaging', TDS, r"cartonboxof25kg")]),
    Q('V38', 'en', "In what package size are the BVZyme products sold?", [T('packaging', TDS, r"cartonboxof25kg")]),
    Q('V39', 'fr', "Quelle est la limite en ASR (anaérobies sulfito-réducteurs) ?", [T('ASR', TDS, r"asr:<30")]),
    Q('V40', 'fr', "Quelles sont les spécifications microbiologiques des enzymes ?", [T('microbiology', TDS, r"asr:<30")]),
    Q('V41', 'fr', "Quelle est la teneur maximale en plomb ?", [T('lead', TDS, r"lead:<5mg/kg")]),
    Q('V42', 'fr', "Quel est le taux d'humidité maximal de la poudre enzymatique ?", [T('moisture', TDS, r"moisture:<15%")]),
    Q('V43', 'fr', "Sous quelle forme se présentent les enzymes BVZyme (aspect, couleur) ?", [T('aspect', TDS, r"freeflowingpowder|color:white")]),
    Q('V44', 'en', "Who manufactures the BVZyme enzymes and where are they based?", [T('manufacturer', TDS, r"vtr&beyond|vtrbeyond|zhuhai|berlin")]),
    Q('V45', 'fr', "De quand date la dernière mise à jour des fiches techniques BVZyme ?", [T('last update', TDS, r"05/02/2024")]),
    # ascorbic-acid document (slots 46-70)
    Q('V46', 'fr', "Pour une production surgelée, combien d'acide ascorbique faut-il ?", [T('AA frozen', AA, r"surgélation.{0,30}150-200")]),
    Q('V47', 'fr', "Quelle quantité de vitamine C pour la surgélation des pâtons ?", [T('AA frozen', AA, r"surgélation.{0,30}150-200")]),
    Q('V48', 'fr', "Acide ascorbique en surgélation : quelle dose ?", [T('AA frozen', AA, r"surgélation.{0,30}150-200")]),
    Q('V49', 'fr', "Quelle dose d'acide ascorbique en blocage froid positif ?", [T('AA cold', AA, r"blocagefroid.{0,40}80-100")]),
    Q('V50', 'fr', "Pour une pâte stockée au froid à 2°C, quel dosage d'acide ascorbique ?", [T('AA cold', AA, r"blocagefroid.{0,40}80-100")]),
    Q('V51', 'fr', "Quel dosage d'acide ascorbique pour une panification directe classique ?", [T('AA direct', AA, r"directestandard.{0,30}20-60")]),
    Q('V52', 'fr', "Quelle est la limite réglementaire d'acide ascorbique dans la farine ?", [T('AA max', AA, r"maximum.{0,30}300|belgique:?300")]),
    Q('V53', 'en', "What is the maximum authorised ascorbic acid level in France and Belgium?", [T('AA max', AA, r"maximum.{0,30}300|belgique:?300")]),
    Q('V54', 'fr', "Combien de grammes d'acide ascorbique pour 50 kg de farine à 75 ppm ?", [T('AA conversion', AA, r"(?<!\d)50kg.{0,40}3,75g")]),
    Q('V55', 'fr', "Pour une tonne de farine, combien de grammes d'acide ascorbique à 100 ppm ?", [T('AA conversion', AA, r"1000kg.{0,40}100g|g/tonnedefarine")]),
    Q('V56', 'en', "How long after mixing does ascorbic acid start acting?", [T('AA action time', AA, r"5-15minutes")]),
    Q('V57', 'fr', "À quelle température l'acide ascorbique est-il le plus efficace ?", [T('AA optimum temp', AA, r"25-30°c")]),
    Q('V58', 'fr', "Quel est le temps d'action de l'acide ascorbique ?", [T('AA action time', AA, r"5-15minutes")]),
    Q('V59', 'fr', "L'acide ascorbique agit-il rapidement après incorporation ?", [T('AA action time', AA, r"5-15minutes|actionrapide")]),
    Q('V60', 'fr', "Quelle est la formule chimique de l'acide ascorbique ?", [T('AA formula', AA, r"c[6₆]h[8₈]o[6₆]")]),
    Q('V61', 'fr', "Acide L-ascorbique : formule brute ?", [T('AA formula', AA, r"c[6₆]h[8₈]o[6₆]")]),
    Q('V62', 'fr', "Quelle est la densité de l'acide ascorbique ?", [T('AA density', AA, r"1,65g/cm")]),
    Q('V63', 'fr', "En quels formats l'acide ascorbique est-il conditionné ?", [T('AA formats', AA, r"500g,1kg,5kg,25kg")]),
    Q('V64', 'fr', "Dans quelles conditions de température et d'humidité stocker l'acide ascorbique ?", [T('AA storage', AA, r"15-25°c|humidité:?<60%")]),
    Q('V65', 'fr', "Combien de temps se conserve l'acide ascorbique ?", [T('AA shelf life', AA, r"18-24mois")]),
    Q('V66', 'fr', "Comment l'acide ascorbique renforce-t-il le gluten ?", [T('AA gluten', AA, r"pontsdisulfur|oxydante|renforcelegluten")]),
    Q('V67', 'fr', "L'acide ascorbique améliore-t-il le volume du pain ?", [T('AA volume', AA, r"volume|rétentiondegaz")]),
    Q('V68', 'en', "What are the advantages of using ascorbic acid?", [T('AA advantages', AA, r"15-30%|standardisation|compatibleavectous|coûtmodéré")]),
    Q('V69', 'fr', "Quelles sont les limites de l'acide ascorbique ?", [T('AA limitations', AA, r"actionlimitée|inefficacesur|suroxyder|pertetotale")]),
    Q('V70', 'fr', "Par quoi peut-on remplacer l'acide ascorbique ?", [T('AA alternatives', AA, r"levainnaturel|malte|enzymes\(amylases\)")]),
    # multi-entity (slots 71-80)
    Q('W01', 'en', "What doses of lipase and maltogenic amylase should be used?",
      [fam('lipase', 'dosage'), fam('maltogenic amylase', 'dosage')]),
    Q('W02', 'fr', "Quels dosages pour l'amylase maltogénique et la xylanase ?",
      [fam('maltogenic amylase', 'dosage'), fam('xylanase', 'dosage')]),
    Q('W03', 'en', "Recommended amounts of transglutaminase and amyloglucosidase in bread?",
      [fam('transglutaminase', 'dosage'), fam('amyloglucosidase', 'dosage')]),
    Q('W04', 'fr', "Dans un améliorant anti-rassissement, combien mettre de glucose oxydase et d'amylase maltogénique ?",
      [fam('glucose oxidase', 'dosage'), fam('maltogenic amylase', 'dosage')]),
    Q('W05', 'fr', "Quel est le rôle respectif de l'amylase maltogénique et de la xylanase ?",
      [fam('maltogenic amylase', 'function'), fam('xylanase', 'function')]),
    Q('W06', 'en', "What do alpha-amylase and xylanase each do in bread dough?",
      [fam('alpha-amylase', 'function'), fam('xylanase', 'function')]),
    Q('W07', 'en', "What are the functions of ascorbic acid, transglutaminase and alpha-amylase in bread?",
      [T('ascorbic acid role', AA, AA_ROLE), fam('transglutaminase', 'function'), fam('alpha-amylase', 'function')]),
    Q('W08', 'fr', "Qu'est-ce que l'acide ascorbique exactement, et d'où provient la glucose oxydase ?",
      [T('ascorbic acid identity', AA, r"acidel-ascorbique|vitaminec|additifalimentaire"), fam('glucose oxidase', 'source')]),
    Q('W09', 'en', "How should maltogenic amylase and alpha-amylase be stored?",
      [fam('maltogenic amylase', 'storage'), fam('alpha-amylase', 'storage')]),
    Q('W10', 'en', "What is the activity of alpha-amylase, and how quickly does ascorbic acid act?",
      [fam('alpha-amylase', 'activity'), T('ascorbic acid action time', AA, r"5-15minutes|actionrapide")]),
    # unanswerable from this corpus (added by hand)
    Q('X01', 'fr', "Quelle est la température optimale d'activité de la xylanase HCF400 ?", [], answerable=False),
    Q('X02', 'en', "Is BVZyme AF330 certified halal or kosher?", [], answerable=False),
    Q('X03', 'fr', "Quel est le prix du BVZyme L MAX65 ?", [], answerable=False),
    Q('X04', 'fr', "Quelle est l'activité enzymatique du BVZyme TG MAX64 ?", [], answerable=False),
    Q('X05', 'en', "By how many days does BVZyme A SOFT405 extend the shelf life of bread?", [], answerable=False),
]
