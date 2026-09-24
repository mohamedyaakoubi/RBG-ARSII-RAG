# Extraction fidelity against the page itself

Truth: 54745 letters and digits in 9227 words, from the positions of the visible glyphs (space characters ignored; a gap of at least 0.1 em between two glyphs is a space).

| extractor | letters and digits found in reading order | false splits (space inside a word) | false joins (words run together) | errors per 1,000 words |
|---|---:|---:|---:|---:|
| pdftotext-layout | 100.0% | 4 | 3 | 0.8 |
| pymupdf-sort | 100.0% | 23 | 0 | 2.5 |
| pypdf-layout | 100.0% | 67 | 34 | 10.9 |
| pdfplumber-1.5 | 100.0% | 169 | 0 | 18.3 |
| pdfplumber-1 | 100.0% | 169 | 0 | 18.3 |
| pdfplumber-2 | 100.0% | 169 | 0 | 18.3 |
| pdfplumber-3 | 100.0% | 169 | 119 | 31.2 |
| pdftotext | 99.3% | 4 | 1 | 0.5 |
| pdfminer | 98.7% | 68 | 0 | 7.4 |
| docling | 80.6% | 41 | 49 | 9.8 |
| ocr | 69.5% | 28 | 33 | 6.6 |
| pdfium | 66.5% | 1 | 0 | 0.1 |
| pymupdf | 66.5% | 1 | 0 | 0.1 |
| pypdf | 66.5% | 6 | 46 | 5.6 |

## Most frequent errors per extractor

- **pdftotext-layout**: `split: minoacid|LlysineF` ×3; `split: minoacid|Llysinea` ×1; `join: 05g075g1+g15g50kg` ×1; `join: 25g375g5+g75g100k` ×1; `join: 5g100kg5+g75g10g1` ×1
- **pymupdf-sort**: `split: Dosage15|100ppmOr` ×3; `split: Dosage15|35ppmOrg` ×3; `split: strength|improvet` ×2; `split: lFungalα|amylasep` ×2; `split: comTECHN|ICALDATA` ×2; `split: asedonAm|ylogluco` ×2
- **pypdf-layout**: `split: rgensinA|nnexIIRe` ×27; `split: 68676888|49030520` ×20; `split: glucoseo|xidasewh` ×3; `split: minoacid|LlysineF` ×3; `split: TASHEETB|VZymeHCF` ×3; `split: TASHEETB|VZymeAMG` ×2
- **pdfplumber-1.5**: `split: echnolog|yIndustr` ×65; `split: 68884903|05201417` ×65; `split: ondsingl|ycerides` ×3; `split: ansgluta|minasewh` ×2; `split: inkingco|nnecting` ×2; `split: Amyloglu|cosidase` ×2
- **pdfplumber-1**: `split: echnolog|yIndustr` ×65; `split: 68884903|05201417` ×65; `split: ondsingl|ycerides` ×3; `split: ansgluta|minasewh` ×2; `split: inkingco|nnecting` ×2; `split: Amyloglu|cosidase` ×2
- **pdfplumber-2**: `split: echnolog|yIndustr` ×65; `split: 68884903|05201417` ×65; `split: ondsingl|ycerides` ×3; `split: ansgluta|minasewh` ×2; `split: inkingco|nnecting` ×2; `split: Amyloglu|cosidase` ×2
- **pdfplumber-3**: `split: echnolog|yIndustr` ×65; `split: 68884903|05201417` ×65; `split: ondsingl|ycerides` ×3; `split: ansgluta|minasewh` ×2; `split: inkingco|nnecting` ×2; `split: Amyloglu|cosidase` ×2
- **pdftotext**: `split: minoacid|LlysineF` ×3; `split: minoacid|Llysinea` ×1; `join: ngDosage+220ppmOr` ×1
- **pdfminer**: `split: lorwhite|creamPhy` ×15; `split: edonendo|xylanase` ×6; `split: comTECHN|ICALDATA` ×5; `split: strength|improves` ×4; `split: Dosage15|100ppmOr` ×3; `split: lFungalα|amylasep` ×3
- **docling**: `join: asewhich+produced` ×13; `join: eProduct+Descript` ×11; `split: comTECHN|ICALDATA` ×5; `split: lorwhite|creamPhy` ×5; `split: strength|improves` ×4; `join: selected+uniquest` ×3
- **ocr**: `split: rmsat30C|30UFCper` ×20; `join: bsentin1+gHeavyme` ×19; `split: minoacid|LlysineF` ×3; `split: sevolume|fineregu` ×3; `split: minoacid|Llysinea` ×1; `split: VTR|beyondNo` ×1
- **pdfium**: `split: ationbas|edonPhos` ×1
- **pymupdf**: `split: ationbas|edonPhos` ×1
- **pypdf**: `split: TGMAX64i|sbasedon` ×1; `split: minoacid|Llysinea` ×1; `split: minoacid|LlysineF` ×1; `split: pergillu|snigerAc` ×1; `split: pergillu|snigeran` ×1; `split: ationbas|edonPhos` ×1
