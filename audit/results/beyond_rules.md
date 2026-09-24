# Beyond the challenge rules: embedding model swap

Same faithful chunks, same plain cosine top-3; only the embedding model changes.
Swapping the model is NOT allowed by the challenge (all-MiniLM-L6-v2 is imposed);
this only measures how much of the remaining error is due to that model.

## DEV

| embedding model | Hit@1 | Hit@3 | MRR@3 | P@3 | Hit@3 FR | Hit@3 EN | Multi cov. |
|---|---:|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 (imposed) | 0.846 | 0.846 | 0.846 | 0.769 | 0.667 | 1.000 | 0.000 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.692 | 0.692 | 0.692 | 0.538 | 0.833 | 0.571 | 0.000 |
| multilingual-e5-small | 0.846 | 1.000 | 0.923 | 0.821 | 1.000 | 1.000 | 0.333 |

## TEST

| embedding model | Hit@1 | Hit@3 | MRR@3 | P@3 | Hit@3 FR | Hit@3 EN | Multi cov. |
|---|---:|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 (imposed) | 0.620 | 0.700 | 0.660 | 0.493 | 0.621 | 0.810 | 0.333 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.500 | 0.660 | 0.573 | 0.420 | 0.621 | 0.714 | 0.250 |
| multilingual-e5-small | 0.720 | 0.880 | 0.793 | 0.593 | 0.931 | 0.810 | 0.375 |

## Paired test vs the imposed model (TEST, single-answer, Hit@3)

| model | model-only hits | imposed-only hits | exact McNemar p |
|---|---:|---:|---:|
| paraphrase-multilingual-MiniLM-L12-v2 | 6 | 8 | 0.791 |
| multilingual-e5-small | 10 | 1 | 0.012 |
