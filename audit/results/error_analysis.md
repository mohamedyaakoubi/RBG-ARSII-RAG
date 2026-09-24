# Error analysis of the final system (v2)

Every single-answer question of TEST-2 and TEST-3 that the final system misses.
"rank" = rank of the first correct fragment in the full cosine ranking; "gap" = its score minus the
3rd result's. "oracle" = the same question asked in the documents' own words.

| set | id | mode | question | rank | gap | oracle phrasing | oracle rank | product filter fixes it |
|---|---|---|---|---:|---:|---|---:|---|
| TEST-2 | V02 | S | Combien de transglutaminase ajouter pour renforcer une pâte ? | 7 | -0.020 | transglutaminase dosage | 1 |  |
| TEST-2 | V05 | S | Quelle quantité de glucose oxydase recommandez-vous par rapport à la farine ? | 9 | -0.039 | glucose oxidase dosage | 1 |  |
| TEST-2 | V05 | S+ | Quelle quantité de glucose oxydase recommandez-vous par rapport à la farine ? | 5 | -0.008 | glucose oxidase dosage | 1 |  |
| TEST-2 | V07 | S | Je voudrais savoir la dose d'amylase fongique à mettre dans ma farine. | 14 | -0.025 | fungal alpha-amylase dosage | 1 |  |
| TEST-2 | V08 | S | Transglutaminase : quel taux d'incorporation ? | 6 | -0.021 | transglutaminase dosage | 1 |  |
| TEST-2 | V11 | S | How much glucose oxidase per tonne of flour? | 14 | -0.066 | glucose oxidase dosage | 1 |  |
| TEST-2 | V11 | S+ | How much glucose oxidase per tonne of flour? | 14 | -0.066 | glucose oxidase dosage | 1 |  |
| TEST-2 | V13 | S | Qu'apporte la xylanase à la pâte à pain ? | 23 | -0.039 | xylanase function | 24 |  |
| TEST-2 | V14 | S | Quel est l'effet de la glucose oxydase sur la pâte ? | 4 | -0.018 | glucose oxidase function | 1 |  |
| TEST-2 | V16 | S | Pourquoi ajouter de la glucose oxydase dans un améliorant ? | 4 | -0.025 | glucose oxidase function | 1 |  |
| TEST-2 | V18 | S+ | Quelles sont les fonctions du BVZyme HCF400 ? | 5 | -0.007 | BVZyme HCF400 function | 3 | yes |
| TEST-2 | V27 | S | Quelle est l'activité du BVZyme AF110 ? | 4 | -0.001 | BVZyme AF110 activity | 1 | no |
| TEST-2 | V31 | S | Comment conserver l'alpha-amylase, et combien de temps ? | 4 | -0.012 | alpha-amylase storage | 1 |  |
| TEST-2 | V32 | S | What is the shelf life of BVZyme A FRESH101 and how should it be stored? | 5 | -0.003 | BVZyme A FRESH101 storage | 1 | yes |
| TEST-2 | V32 | S+ | What is the shelf life of BVZyme A FRESH101 and how should it be stored? | 5 | -0.003 | BVZyme A FRESH101 storage | 1 | yes |
| TEST-2 | V34 | S+ | Ces améliorants enzymatiques conviennent-ils à un régime sans gluten ? | 4 | -0.005 | allergens gluten | 1 |  |
| TEST-2 | V37 | S | Quel est le conditionnement des enzymes BVZyme ? | 173 | -0.150 | BVZyme packaging | 1 |  |
| TEST-2 | V37 | S+ | Quel est le conditionnement des enzymes BVZyme ? | 211 | -0.145 | BVZyme packaging | 1 |  |
| TEST-2 | V39 | S | Quelle est la limite en ASR (anaérobies sulfito-réducteurs) ? | 140 | -0.208 | microbiology ASR | 1 |  |
| TEST-2 | V39 | S+ | Quelle est la limite en ASR (anaérobies sulfito-réducteurs) ? | 24 | -0.115 | microbiology ASR | 1 |  |
| TEST-2 | V40 | S | Quelles sont les spécifications microbiologiques des enzymes ? | 28 | -0.034 | microbiology | 1 |  |
| TEST-2 | V41 | S | Quelle est la teneur maximale en plomb ? | 85 | -0.142 | heavy metals lead | 1 |  |
| TEST-2 | V42 | S | Quel est le taux d'humidité maximal de la poudre enzymatique ? | 6 | -0.067 | moisture | 1 |  |
| TEST-2 | V43 | S | Sous quelle forme se présentent les enzymes BVZyme (aspect, couleur) ? | 9 | -0.010 | aspect color powder | 1 |  |
| TEST-2 | V44 | S | Who manufactures the BVZyme enzymes and where are they based? | 300 | -0.323 | manufacturer company address | 1 |  |
| TEST-2 | V44 | S+ | Who manufactures the BVZyme enzymes and where are they based? | 300 | -0.323 | manufacturer company address | 1 |  |
| TEST-2 | V45 | S | De quand date la dernière mise à jour des fiches techniques BVZyme ? | 7 | -0.060 | last updating date | 1 |  |
| TEST-2 | V47 | S | Quelle quantité de vitamine C pour la surgélation des pâtons ? | 8 | -0.046 | ascorbic acid dosage freezing | 1 |  |
| TEST-2 | V52 | S | Quelle est la limite réglementaire d'acide ascorbique dans la farine ? | 6 | -0.038 | maximum authorised dosage ascorbic acid | 1 |  |
| TEST-2 | V54 | S+ | Combien de grammes d'acide ascorbique pour 50 kg de farine à 75 ppm ? | 4 | -0.001 | conversion table 50 kg 75 ppm grams | 1 |  |
| TEST-2 | V57 | S | À quelle température l'acide ascorbique est-il le plus efficace ? | 6 | -0.019 | ascorbic acid optimal temperature | 1 |  |
| TEST-2 | V60 | S | Quelle est la formule chimique de l'acide ascorbique ? | 18 | -0.068 | ascorbic acid chemical formula | 1 |  |
| TEST-2 | V61 | S | Acide L-ascorbique : formule brute ? | 17 | -0.046 | ascorbic acid chemical formula | 1 |  |
| TEST-2 | V62 | S | Quelle est la densité de l'acide ascorbique ? | 21 | -0.109 | ascorbic acid density | 3 |  |
| TEST-2 | V62 | S+ | Quelle est la densité de l'acide ascorbique ? | 15 | -0.053 | ascorbic acid density | 3 |  |
| TEST-2 | V63 | S+ | En quels formats l'acide ascorbique est-il conditionné ? | 4 | -0.002 | ascorbic acid packaging formats | 1 |  |
| TEST-2 | V67 | S | L'acide ascorbique améliore-t-il le volume du pain ? | 8 | -0.033 | ascorbic acid volume improvement | 2 |  |
| TEST-2 | V70 | S | Par quoi peut-on remplacer l'acide ascorbique ? | 5 | -0.036 | alternatives to ascorbic acid | 1 |  |
| TEST-2 | V70 | S+ | Par quoi peut-on remplacer l'acide ascorbique ? | 5 | -0.004 | alternatives to ascorbic acid | 1 |  |
| TEST-3 | Z01 | S | How much BVZyme A SOFT305 should I add to my dough? | 9 | -0.041 | BVZyme A SOFT305 dosage | 1 | yes |
| TEST-3 | Z01 | S+ | How much BVZyme A SOFT305 should I add to my dough? | 9 | -0.041 | BVZyme A SOFT305 dosage | 1 | yes |
| TEST-3 | Z08 | S | Combien d'amyloglucosidase mettre dans la pâte ? | 4 | -0.008 | amyloglucosidase dosage | 1 |  |
| TEST-3 | Z14 | S | What does BVZyme L MAX64 improve? | 8 | -0.017 | BVZyme L MAX64 function | 6 | yes |
| TEST-3 | Z14 | S+ | What does BVZyme L MAX64 improve? | 8 | -0.017 | BVZyme L MAX64 function | 6 | yes |
| TEST-3 | Z27 | S | Faut-il un étiquetage OGM pour ces enzymes ? | 24 | -0.056 | GMO status labeling | 1 |  |
| TEST-3 | Z29 | S | How are BVZyme enzymes packed? | 9 | -0.024 | BVZyme packaging | 1 |  |
| TEST-3 | Z29 | S+ | How are BVZyme enzymes packed? | 9 | -0.024 | BVZyme packaging | 1 |  |
| TEST-3 | Z31 | S | Teneur en plomb tolérée dans les enzymes ? | 137 | -0.153 | heavy metals lead | 1 |  |
| TEST-3 | Z34 | S | Quelle société fabrique les enzymes BVZyme ? | 325 | -0.339 | manufacturer company address | 1 |  |
| TEST-3 | Z34 | S+ | Quelle société fabrique les enzymes BVZyme ? | 302 | -0.390 | manufacturer company address | 1 |  |
| TEST-3 | Z43 | S | Peut-on diluer l'acide ascorbique dans l'eau avant de l'ajouter ? | 7 | -0.002 | ascorbic acid dilute in water | 1 |  |
| TEST-3 | Z45 | S | Quel est le pH de l'acide ascorbique en solution ? | 13 | -0.063 | ascorbic acid pH | 1 |  |
| TEST-3 | Z48 | S | Quelle est la durée de vie de l'acide ascorbique ? | 6 | -0.032 | ascorbic acid shelf life | 1 |  |

**S**: 37 failures; correct fragment at rank 4-5 (lost to the top-3 cut): 7; found in the top 3 by the oracle phrasing: 35; fixed by a product-code filter: 3.

**S+**: 16 failures; correct fragment at rank 4-5 (lost to the top-3 cut): 7; found in the top 3 by the oracle phrasing: 15; fixed by a product-code filter: 4.
