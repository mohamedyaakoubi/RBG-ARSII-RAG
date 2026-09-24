"""
TEST-4: a fourth held-out set, for the product-code filter. Frozen in git
before the filter was implemented or run on it.

TEST-2 and TEST-3 suggested the filter (their failures V18, V32, Z01, Z14), so
they cannot measure it. Every TEST-4 question names at least one product by
its code, the only questions the filter can change. Slots (product, attribute,
language, how the code is written, whether the question also names the
product's family) were drawn at random with a fixed seed
(audit/sample_test4_slots.py, random.Random(20260926)); the code is written
exactly as drawn. 74 slots + 3 unanswerable questions added by hand.
Relevance works as in eval_set.py: right source PDF + the actual answer.

Attribute groups, reported separately:
  specific  dosage, function, activity, source: the answer differs between
            products or is in a product-specific passage;
  header    enzyme type: every fragment of the right sheet names it in its
            header, so any result from that sheet answers;
  shared    storage, packaging, allergens, GMO / irradiation, microbiology,
            heavy metals, aspect / moisture, manufacturer, update date: the
            same text on every sheet. A result from another sheet has the
            right words but is not evidence about the product asked about.
"""
import re

from audit.eval_set import AA, T, Q
from audit.eval_set_v2 import fam, AA_DOSAGE, AA_ROLE, FUNCTION

ACT = {  # Activity section of each sheet, squashed (TG MAX63 and TG MAX64 have none)
    'af110': '150000skb/g', 'af220': '11000fau/g', 'af330': '11900fau/g', 'afsx': '85000skb/g',
    'afresh101': '10000nmau/g', 'afresh202': '10950nmau/g', 'afresh303': '11000nmau/g',
    'asoft205': '11600nmau/g', 'asoft305': '10500nmau/g', 'asoft405': '11720nmau/g',
    'amg880': '70000agi/g', 'amg1400': '80000agi/g',
    'gox110': '10000u/g', 'gomax63': '10000u/g', 'gomax65': '11000u/g',
    'hcb708': '568xylh/g', 'hcb709': '577xylh/g', 'hcb710': '583xylh/g', 'hcf400': '2040xylh/g',
    'hcf500': '2110xylh/g', 'hcf600': '7750xylh/g', 'hcfmax63': '2205xylh/g', 'hcfmax64': '7850xylh/g',
    'hcfmaxx': '23500xylh/g',
    'l55': '1120u/g', 'l65': '1080u/g', 'lmax63': '1255u/g', 'lmax64': '47000u/g', 'lmax65': '50000u/g',
    'lmaxx': '3300u/g', 'tg881': '200u/g', 'tg883': '400u/g',
}
ENZYME = {'alpha-amylase': r"alpha-amylase", 'maltogenic amylase': r"maltogenicamylase",
          'amyloglucosidase': r"amyloglu?cosidase|glucoamylase", 'glucose oxidase': r"glucoseoxidase",
          'xylanase': r"xylanase", 'lipase': r"lipase|lipolytic", 'transglutaminase': r"transgluta"}
SHARED = {
    'packaging': r"cartonboxof25kg", 'allergens': r"allergens?:?gluten",
    'gmo': r"1829/2003|nospecificlabeling", 'irradiation': r"withoutirradiation",
    'total plate count': r"totalplatecount:<50000ufc", 'lead': r"lead:<5mg/kg", 'mercury': r"mercury:<0,5mg/kg",
    'colour': r"color:white", 'moisture': r"moisture:<15%",
    'manufacturer': r"vtr&beyond|vtrbeyond|zhuhai|berlin", 'last update': r"05/02/2024",
}
FAMILY_OF = {'af110': 'alpha-amylase', 'af220': 'alpha-amylase', 'af330': 'alpha-amylase', 'afsx': 'alpha-amylase',
             'afresh101': 'maltogenic amylase', 'afresh202': 'maltogenic amylase', 'afresh303': 'maltogenic amylase',
             'asoft205': 'maltogenic amylase', 'asoft305': 'maltogenic amylase', 'asoft405': 'maltogenic amylase',
             'amg880': 'amyloglucosidase', 'amg1400': 'amyloglucosidase',
             'gox110': 'glucose oxidase', 'gomax63': 'glucose oxidase', 'gomax65': 'glucose oxidase',
             'hcb708': 'xylanase', 'hcb709': 'xylanase', 'hcb710': 'xylanase', 'hcf400': 'xylanase',
             'hcf500': 'xylanase', 'hcf600': 'xylanase', 'hcfmax63': 'xylanase', 'hcfmax64': 'xylanase',
             'hcfmaxx': 'xylanase', 'l55': 'lipase', 'l65': 'lipase', 'lmax63': 'lipase', 'lmax64': 'lipase',
             'lmax65': 'lipase', 'lmaxx': 'lipase', 'tg881': 'transglutaminase', 'tg883': 'transglutaminase',
             'tgmax63': 'transglutaminase', 'tgmax64': 'transglutaminase'}
GROUP = {'dosage': 'specific', 'function': 'specific', 'activity': 'specific', 'source': 'specific',
         'enzyme_type': 'header'}


def P(d, attr):
    """Target: attribute `attr` of product `d`, answered from its own sheet."""
    if attr in ('dosage', 'source', 'storage'):
        return {**fam(d, attr, [d]), 'group': GROUP.get(attr, 'shared')}
    if attr == 'function':
        return {**T(f'{d} function', [d], FUNCTION[FAMILY_OF[d]]), 'group': 'specific'}
    if attr == 'activity':
        return {**T(f'{d} activity', [d], r'(?<![\d.])' + re.escape(ACT[d])), 'group': 'specific'}
    if attr == 'enzyme_type':
        return {**T(f'{d} enzyme', [d], ENZYME[FAMILY_OF[d]]), 'group': 'header'}
    return {**T(f'{d} {attr}', [d], SHARED[attr]), 'group': 'shared'}


TEST4 = [
    # one product (slots 1-60)
    Q('P01', 'en', "What dosage of A FRESH101 do you recommend for bread?", [P('afresh101', 'dosage')]),
    Q('P02', 'fr', "Quelle dose de l'alpha-amylase AF330 faut-il ajouter à la farine ?", [P('af330', 'dosage')]),
    Q('P03', 'fr', "Comment stocker le LMAX63 et combien de temps se conserve-t-il ?", [P('lmax63', 'storage')]),
    Q('P04', 'fr', "Quelles sont les conditions de conservation du BVZyme L MAX65 ?", [P('lmax65', 'storage')]),
    Q('P05', 'en', "What does BVZyme HCF MAX64 do in bread dough?", [P('hcfmax64', 'function')]),
    Q('P06', 'fr', "Quelle est la teneur maximale en plomb du BVZyme HCB 709 ?", [P('hcb709', 'lead')]),
    Q('P07', 'fr', "Quelle est l'activité de l'amyloglucosidase BVZyme AMG1400 ?", [P('amg1400', 'activity')]),
    Q('P08', 'fr', "Quel est le conditionnement du BVZyme AMG 1400 ?", [P('amg1400', 'packaging')]),
    Q('P09', 'fr', "Quelle est l'activité enzymatique du BVZyme AF-SX ?", [P('afsx', 'activity')]),
    Q('P10', 'fr', "Combien de temps peut-on garder le BVZyme AF330, et dans quelles conditions ?", [P('af330', 'storage')]),
    Q('P11', 'en', "What type of enzyme is BVZyme AF SX?", [P('afsx', 'enzyme_type')]),
    Q('P12', 'fr', "Quelle est la limite en plomb pour la lipase BVZyme LMAXX ?", [P('lmaxx', 'lead')]),
    Q('P13', 'en', "What is the enzyme activity of TG-883?", [P('tg883', 'activity')]),
    Q('P14', 'fr', "Comment est produit le bvzyme tg max63 ?", [P('tgmax63', 'source')]),
    Q('P15', 'fr', "Quel dosage pour l'amyloglucosidase BVZyme AMG880 ?", [P('amg880', 'dosage')]),
    Q('P16', 'fr', "Quels sont les effets de l'alpha-amylase AF SX sur la pâte ?", [P('afsx', 'function')]),
    Q('P17', 'fr', "Combien de BVZyme AF220 dois-je mettre dans ma farine ?", [P('af220', 'dosage')]),
    Q('P18', 'fr', "Comment est obtenue la transglutaminase BVZyme TG883 ?", [P('tg883', 'source')]),
    Q('P19', 'fr', "Quelle est l'activité du BVZyme A FRESH303 ?", [P('afresh303', 'activity')]),
    Q('P20', 'en', "What is the shelf life of BVZyme AF220?", [P('af220', 'storage')]),
    Q('P21', 'fr', "À partir de quoi est produit le TG MAX63 ?", [P('tgmax63', 'source')]),
    Q('P22', 'en', "Why would I add HCF MAX64 to my bread improver?", [P('hcfmax64', 'function')]),
    Q('P23', 'fr', "Le BVZyme L55 contient-il des allergènes ?", [P('l55', 'allergens')]),
    Q('P24', 'fr', "Quelle est l'activité enzymatique du BVZyme HCF500 ?", [P('hcf500', 'activity')]),
    Q('P25', 'fr', "Quelle quantité de plomb est tolérée dans le BVZyme L MAX X ?", [P('lmaxx', 'lead')]),
    Q('P26', 'fr', "La lipase BVZyme L65 contient-elle du gluten ou d'autres allergènes ?", [P('l65', 'allergens')]),
    Q('P27', 'fr', "À quoi sert le BVZyme A FRESH101 en panification ?", [P('afresh101', 'function')]),
    Q('P28', 'fr', "Comment conserver l'alpha-amylase bvzyme af sx ?", [P('afsx', 'storage')]),
    Q('P29', 'en', "What is the activity of BVZyme HCF-MAX-X?", [P('hcfmaxx', 'activity')]),
    Q('P30', 'en', "What is the maximum moisture content of AF330?", [P('af330', 'moisture')]),
    Q('P31', 'en', "What is the recommended dose of the amyloglucosidase AMG880?", [P('amg880', 'dosage')]),
    Q('P32', 'en', "What benefits does the alpha-amylase AF220 bring to bread?", [P('af220', 'function')]),
    Q('P33', 'fr', "Quelle souche produit le bvzyme a fresh202 ?", [P('afresh202', 'source')]),
    Q('P34', 'en', "What is BVZyme AF220 used for?", [P('af220', 'function')]),
    Q('P35', 'en', "How many NMAU/g does BVZyme A FRESH303 have?", [P('afresh303', 'activity')]),
    Q('P36', 'en', "what's the activity of bvzyme af sx?", [P('afsx', 'activity')]),
    Q('P37', 'fr', "Quelle est la limite en mercure du GO-MAX-63 ?", [P('gomax63', 'mercury')]),
    Q('P38', 'en', "What is the enzymatic activity of BVZyme L65?", [P('l65', 'activity')]),
    Q('P39', 'fr', "Quel est le taux d'utilisation recommandé de l'A-SOFT405 ?", [P('asoft405', 'dosage')]),
    Q('P40', 'fr', "Le TG MAX64 a-t-il subi un traitement par irradiation ?", [P('tgmax64', 'irradiation')]),
    Q('P41', 'fr', "Qui fabrique le BVZyme AF110 ?", [P('af110', 'manufacturer')]),
    Q('P42', 'en', "What is the total plate count limit for BVZyme HCF MAX63?", [P('hcfmax63', 'total plate count')]),
    Q('P43', 'fr', "De quelle couleur est la poudre GO MAX 63 ?", [P('gomax63', 'colour')]),
    Q('P44', 'en', "How does BVZyme TG MAX64 improve dough?", [P('tgmax64', 'function')]),
    Q('P45', 'en', "What dosage of the alpha-amylase BVZyme AF110 should I use?", [P('af110', 'dosage')]),
    Q('P46', 'fr', "quelle quantité de l max x ajouter à la pâte ?", [P('lmaxx', 'dosage')]),
    Q('P47', 'fr', "Dosage conseillé du GOX110 ?", [P('gox110', 'dosage')]),
    Q('P48', 'en', "How should BVZyme GO MAX 65 be stored?", [P('gomax65', 'storage')]),
    Q('P49', 'en', "How is A SOFT205 packaged?", [P('asoft205', 'packaging')]),
    Q('P50', 'en', "What is the activity of the lipase L-55?", [P('l55', 'activity')]),
    Q('P51', 'fr', "Le L MAX65 est-il une lipase ?", [P('lmax65', 'enzyme_type')]),
    Q('P52', 'en', "Recommended dosage for BVZyme HCF MAX63?", [P('hcfmax63', 'dosage')]),
    Q('P53', 'en', "Which microorganism is BVZyme L MAX X produced from?", [P('lmaxx', 'source')]),
    Q('P54', 'en', "what does tg max63 do to the dough?", [P('tgmax63', 'function')]),
    Q('P55', 'fr', "Quelle entreprise produit la lipase BVZyme L-55 ?", [P('l55', 'manufacturer')]),
    Q('P56', 'en', "When was the data sheet of the maltogenic amylase BVZyme ASOFT305 last updated?",
      [P('asoft305', 'last update')]),
    Q('P57', 'en', "What are the storage conditions for BVZyme HCB709?", [P('hcb709', 'storage')]),
    Q('P58', 'en', "What does the maltogenic amylase BVZyme ASOFT305 do for bread?", [P('asoft305', 'function')]),
    Q('P59', 'fr', "Quelle est la durée de conservation de la lipase BVZyme L MAX65 ?", [P('lmax65', 'storage')]),
    Q('P60', 'fr', "En quel emballage est vendue l'amylase maltogénique BVZyme A FRESH202 ?", [P('afresh202', 'packaging')]),
    # two products (slots 61-68)
    Q('P61', 'fr', "Quels dosages pour le GO MAX 63 et le hcb708 ?", [P('gomax63', 'dosage'), P('hcb708', 'dosage')]),
    Q('P62', 'fr', "Quelles doses de BVZyme AF110 et de BVZyme A FRESH303 utiliser dans un pain de mie ?",
      [P('af110', 'dosage'), P('afresh303', 'dosage')]),
    Q('P63', 'en', "What are the recommended doses of HCB 709 and BVZyme L MAX64?",
      [P('hcb709', 'dosage'), P('lmax64', 'dosage')]),
    Q('P64', 'fr', "Combien de L65 et de tg max64 faut-il ajouter ?", [P('l65', 'dosage'), P('tgmax64', 'dosage')]),
    Q('P65', 'en', "What do HCF600 and HCF MAX X do in dough?", [P('hcf600', 'function'), P('hcfmaxx', 'function')]),
    Q('P66', 'fr', "Pour un améliorant, quels dosages de BVZyme AF220 et de BVZyme GO MAX 63 ?",
      [P('af220', 'dosage'), P('gomax63', 'dosage')]),
    Q('P67', 'fr', "Dosage du HCF-MAX-X et du TG 883 ?", [P('hcfmaxx', 'dosage'), P('tg883', 'dosage')]),
    Q('P68', 'en', "Compare the activity of LMAX65 and BVZyme GOMAX65.",
      [P('lmax65', 'activity'), P('gomax65', 'activity')]),
    # a product and another family (slots 69-74)
    Q('P69', 'en', "How should A-SOFT405 and transglutaminase be stored?",
      [P('asoft405', 'storage'), fam('transglutaminase', 'storage')]),
    Q('P70', 'fr', "Quel est le rôle du HCB710 et de l'acide ascorbique dans la pâte ?",
      [P('hcb710', 'function'), T('ascorbic acid role', AA, AA_ROLE)]),
    Q('P71', 'en', "What dosage of BVZyme HCF-400 and of maltogenic amylase should I use?",
      [P('hcf400', 'dosage'), fam('maltogenic amylase', 'dosage')]),
    Q('P72', 'fr', "Quelles quantités de BVZyme AF110 et d'amyloglucosidase recommandez-vous ?",
      [P('af110', 'dosage'), fam('amyloglucosidase', 'dosage')]),
    Q('P73', 'en', "Recommended doses of BVZyme HCB 708 and ascorbic acid for bread?",
      [P('hcb708', 'dosage'), AA_DOSAGE]),
    Q('P74', 'fr', "Quel dosage pour le BVZyme L55 et pour l'alpha-amylase ?",
      [P('l55', 'dosage'), fam('alpha-amylase', 'dosage')]),
    # not answered by the sheets (added by hand)
    Q('U01', 'en', "What is the price per kilo of BVZyme AF330?", [], answerable=False),
    Q('U02', 'fr', "Quelle est la température optimale d'action du BVZyme HCB710 ?", [], answerable=False),
    Q('U03', 'fr', "Le BVZyme TG881 est-il certifié halal ?", [], answerable=False),
]
