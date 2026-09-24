"""
TEST-3: a third held-out set, frozen after TEST-2 was used and BEFORE the
post-hoc fixes suggested by TEST-2 failures were implemented.

Slots drawn at random with a new seed (audit/sample_test3_slots.py,
random.Random(20260925)); 59 slots + 3 unanswerable questions added by hand.
Relevance works as in eval_set.py.
"""
import re

from audit.eval_set import MALTO, XYL, AA, TDS, T, Q
from audit.eval_set_v2 import fam, FUNCTION, STORAGE

ACT = {'afresh101': '10000nmau/g', 'afresh202': '10950nmau/g', 'afresh303': '11000nmau/g',
       'asoft205': '11600nmau/g', 'asoft305': '10500nmau/g', 'asoft405': '11720nmau/g',
       'hcb708': '568xylh/g', 'hcb709': '577xylh/g', 'hcb710': '583xylh/g', 'hcf400': '2040xylh/g',
       'hcf500': '2110xylh/g', 'hcf600': '7750xylh/g', 'hcfmax63': '2205xylh/g', 'hcfmax64': '7850xylh/g',
       'hcfmaxx': '23500xylh/g'}


def activity(label, docs):
    return T(label, docs, {d: re.escape(ACT[d]) for d in docs})


TEST3 = [
    Q('Z01', 'en', "How much BVZyme A SOFT305 should I add to my dough?", [fam('A SOFT305', 'dosage', ['asoft305'])]),
    Q('Z02', 'fr', "Quel dosage conseillez-vous pour le BVZyme HCB708 ?", [fam('HCB708', 'dosage', ['hcb708'])]),
    Q('Z03', 'fr', "Combien de BVZyme TG MAX64 faut-il par quintal de farine ?", [fam('TG MAX64', 'dosage', ['tgmax64'])]),
    Q('Z04', 'fr', "Dose d'utilisation de l'AF110 ?", [fam('AF110', 'dosage', ['af110'])]),
    Q('Z05', 'en', "Lipase addition rate in ppm?", [fam('lipase', 'dosage')]),
    Q('Z06', 'fr', "À quelle dose utilise-t-on la transglutaminase en boulangerie ?", [fam('transglutaminase', 'dosage')]),
    Q('Z07', 'fr', "Quelle est la plage de dosage des xylanases BVZyme ?", [fam('xylanase', 'dosage')]),
    Q('Z08', 'fr', "Combien d'amyloglucosidase mettre dans la pâte ?", [fam('amyloglucosidase', 'dosage')]),
    Q('Z09', 'en', "Recommended dose for GO MAX 65?", [fam('GO MAX 65', 'dosage', ['gomax65'])]),
    Q('Z10', 'en', "What is glucose oxidase used for in baking?", [fam('glucose oxidase', 'function')]),
    Q('Z11', 'en', "What are the benefits of fungal alpha-amylase in bread?", [fam('alpha-amylase', 'function')]),
    Q('Z12', 'fr', "La glucose oxydase améliore-t-elle la tolérance de la pâte ?", [fam('glucose oxidase', 'function')]),
    Q('Z13', 'fr', "Quels effets apporte le BVZyme L MAX X ?", [T('L MAX X function', ['lmaxx'], FUNCTION['lipase'])]),
    Q('Z14', 'en', "What does BVZyme L MAX64 improve?", [T('L MAX64 function', ['lmax64'], FUNCTION['lipase'])]),
    Q('Z15', 'fr', "Dans quelles applications utilise-t-on la xylanase ?", [T('xylanase application', XYL, r"bakeryandbread")]),
    Q('Z16', 'fr', "La xylanase est-elle destinée à la panification ?", [T('xylanase application', XYL, r"bakeryandbread|" + FUNCTION['xylanase'])]),
    Q('Z17', 'fr', "D'où proviennent les xylanases BVZyme : origine bactérienne ou fongique ?", [fam('xylanase', 'source')]),
    Q('Z18', 'en', "How is transglutaminase obtained?", [fam('transglutaminase', 'source')]),
    Q('Z19', 'en', "What organism is BVZyme A SOFT205 made from?", [T('A SOFT205 organism', ['asoft205'], r"bacillus")]),
    Q('Z20', 'en', "What is the activity of the maltogenic amylase products in NMAU/g?", [activity('maltogenic activity', MALTO)]),
    Q('Z21', 'fr', "Quelle est l'activité enzymatique des xylanases (XylH/g) ?", [activity('xylanase activity', XYL)]),
    Q('Z22', 'en', "What kind of enzyme is BVZyme GO MAX 65?", [T('GO MAX 65 enzyme', ['gomax65'], r"glucoseoxidase")]),
    Q('Z23', 'fr', "Le BVZyme TG883 est à base de quelle enzyme ?", [T('TG883 enzyme', ['tg883'], r"transgluta")]),
    Q('Z24', 'en', "How long can BVZyme AMG1400 be stored, and under what conditions?", [T('AMG1400 storage', ['amg1400'], STORAGE)]),
    Q('Z25', 'en', "Storage recommendations for amyloglucosidase?", [fam('amyloglucosidase', 'storage')]),
    Q('Z26', 'en', "Which allergens are declared on the BVZyme datasheets?", [T('allergen gluten', TDS, r"allergens?:?gluten")]),
    Q('Z27', 'fr', "Faut-il un étiquetage OGM pour ces enzymes ?", [T('GMO', TDS, r"1829/2003|nospecificlabeling")]),
    Q('Z28', 'fr', "Y a-t-il eu une irradiation des produits BVZyme ?", [T('irradiation', TDS, r"withoutirradiation")]),
    Q('Z29', 'en', "How are BVZyme enzymes packed?", [T('packaging', TDS, r"cartonboxof25kg")]),
    Q('Z30', 'fr', "Quelle est la spécification pour Staphylococcus aureus ?", [T('staphylococcus', TDS, r"staphylococcusaureus:absentin1g")]),
    Q('Z31', 'fr', "Teneur en plomb tolérée dans les enzymes ?", [T('lead', TDS, r"lead:<5mg/kg")]),
    Q('Z32', 'en', "What is the moisture specification of the enzyme preparations?", [T('moisture', TDS, r"moisture:<15%")]),
    Q('Z33', 'en', "What colour is the enzyme powder?", [T('aspect', TDS, r"freeflowingpowder|color:white")]),
    Q('Z34', 'fr', "Quelle société fabrique les enzymes BVZyme ?", [T('manufacturer', TDS, r"vtr&beyond|vtrbeyond|zhuhai|berlin")]),
    Q('Z35', 'fr', "Quelle est la date de mise à jour des fiches BVZyme ?", [T('last update', TDS, r"05/02/2024")]),
    Q('Z36', 'fr', "Dosage de l'acide ascorbique pour des pâtons surgelés ?", [T('AA frozen', AA, r"surgélation.{0,30}150-200")]),
    Q('Z37', 'fr', "Combien d'acide ascorbique pour un pain de mie en procédé CBP ?", [T('AA CBP', AA, r"paindemie\(cbp\).{0,25}75")]),
    Q('Z38', 'en', "What ascorbic acid level is recommended for frozen dough?", [T('AA frozen', AA, r"surgélation.{0,30}150-200")]),
    Q('Z39', 'fr', "Quelle quantité d'acide ascorbique en panification directe standard ?", [T('AA direct', AA, r"directestandard.{0,30}20-60")]),
    Q('Z40', 'fr', "Quel est le dosage maximum d'acide ascorbique autorisé ?", [T('AA max', AA, r"maximum.{0,30}300|belgique:?300")]),
    Q('Z41', 'fr', "Combien de grammes d'acide ascorbique faut-il pour 10 kg de farine à 150 ppm ?", [T('AA conversion', AA, r"(?<!\d)10kg.{0,45}1,5g")]),
    Q('Z42', 'fr', "Avec quelle précision faut-il peser l'acide ascorbique ?", [T('AA weighing', AA, r"±5%|balancedeprécision")]),
    Q('Z43', 'fr', "Peut-on diluer l'acide ascorbique dans l'eau avant de l'ajouter ?", [T('AA dilution', AA, r"solution1-2%|diluer")]),
    Q('Z44', 'en', "Can ascorbic acid be dissolved in water before use?", [T('AA dilution', AA, r"solution1-2%|diluer")]),
    Q('Z45', 'fr', "Quel est le pH de l'acide ascorbique en solution ?", [T('AA pH', AA, r"2,0-2,5")]),
    Q('Z46', 'fr', "Quelle est la formule moléculaire de la vitamine C ?", [T('AA formula', AA, r"c[6₆]h[8₈]o[6₆]")]),
    Q('Z47', 'fr', "L'acide ascorbique est disponible en quels conditionnements ?", [T('AA formats', AA, r"500g,1kg,5kg,25kg")]),
    Q('Z48', 'fr', "Quelle est la durée de vie de l'acide ascorbique ?", [T('AA shelf life', AA, r"18-24mois")]),
    Q('Z49', 'fr', "L'acide ascorbique raccourcit-il la fermentation ?", [T('AA fermentation', AA, r"raccourcissementdelafermentation|réductiondutempsdefermentation|15-30%")]),
    Q('Z50', 'en', "Does ascorbic acid affect crust colour?", [T('AA crust', AA, r"couleurdecroûte|coloration")]),
    Q('Z51', 'fr', "Quels sont les avantages de l'acide ascorbique en production ?", [T('AA advantages', AA, r"15-30%|standardisation|compatibleavectous|coûtmodéré|améliorationvisible")]),
    Q('Z52', 'fr', "Quelles alternatives à l'acide ascorbique existent ?", [T('AA alternatives', AA, r"levainnaturel|malte|enzymes\(amylases\)")]),
    Q('Z53', 'fr', "Quels dosages pour l'amyloglucosidase et l'alpha-amylase ?",
      [fam('amyloglucosidase', 'dosage'), fam('alpha-amylase', 'dosage')]),
    Q('Z54', 'en', "Dose ranges for lipase and glucose oxidase?", [fam('lipase', 'dosage'), fam('glucose oxidase', 'dosage')]),
    Q('Z55', 'fr', "Combien d'amylase maltogénique et de glucose oxydase faut-il ajouter ?",
      [fam('maltogenic amylase', 'dosage'), fam('glucose oxidase', 'dosage')]),
    Q('Z56', 'fr', "Quels sont les effets de l'amylase maltogénique et de la transglutaminase sur le pain ?",
      [fam('maltogenic amylase', 'function'), fam('transglutaminase', 'function')]),
    Q('Z57', 'fr', "À quoi servent la xylanase et l'amyloglucosidase ?", [fam('xylanase', 'function'), fam('amyloglucosidase', 'function')]),
    Q('Z58', 'fr', "Quelles souches produisent l'alpha-amylase et la xylanase ?", [fam('alpha-amylase', 'source'), fam('xylanase', 'source')]),
    Q('Z59', 'en', "Storage conditions for lipase and glucose oxidase?", [fam('lipase', 'storage'), fam('glucose oxidase', 'storage')]),
    Q('Y01', 'en', "What is the optimal pH of BVZyme GOX 110?", [], answerable=False),
    Q('Y02', 'fr', "Quel est le fournisseur en Tunisie des enzymes BVZyme ?", [], answerable=False),
    Q('Y03', 'fr', "Combien de jours dure l'effet anti-rassissement de l'A FRESH303 ?", [], answerable=False),
]
