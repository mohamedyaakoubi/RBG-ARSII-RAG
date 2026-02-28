"""
PDF processor — pdfplumber extraction + entity-centric chunking.

Design principles:
  1. pdfplumber for clean text + structured table extraction
  2. Entity-centric chunks: one focused chunk per (enzyme × property)
  3. Each chunk is self-contained with rich metadata prefix
  4. Bilingual indexing for key retrieval targets (dosage, function)
  5. Table data converted to natural-language text for embedding
"""
import pdfplumber
import re
from pathlib import Path
from utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# ENZYME TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

_PREFIX_TO_ENZYME = [
    ('A FRESH', 'maltogenic amylase'),
    ('A SOFT',  'maltogenic amylase'),
    ('AF SX',   'alpha-amylase'),
    ('AF',      'alpha-amylase'),
    ('AMG',     'amyloglucosidase'),
    ('GO',      'glucose oxidase'),
    ('HCB',     'xylanase'),
    ('HCF',     'xylanase'),
    ('L MAX',   'lipase'),
    ('L55',     'lipase'),
    ('L65',     'lipase'),
    ('TG',      'transglutaminase'),
]

_ENZYME_TEXT_PATTERNS = [
    (r'α[\s-]*amylase|alpha[\s-]*amylase|fungal\s*amylase', 'alpha-amylase'),
    (r'amyloglucosidase|glucoamylase',                       'amyloglucosidase'),
    (r'glucose\s*oxidase',                                   'glucose oxidase'),
    (r'xylanase',                                            'xylanase'),
    (r'phospholipase|(?<!\w)lipase',                         'lipase'),
    (r'transglutaminase',                                    'transglutaminase'),
    (r'maltogenic',                                          'maltogenic amylase'),
]


def _detect_enzyme_type(product_name, text):
    name_norm = re.sub(r'\s+', '', product_name).upper()
    for prefix, enzyme in _PREFIX_TO_ENZYME:
        if prefix.replace(' ', '').upper() in name_norm:
            return enzyme
    for pattern, enzyme in _ENZYME_TEXT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return enzyme
    return None


# ═══════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION (pdfplumber)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_product_name(pdf_path):
    name = Path(pdf_path).stem
    name = re.sub(r'BVZ(?:yme|YME)\s*TDS', 'BVZyme ', name, flags=re.IGNORECASE)
    name = re.sub(r'\(\d+\)', '', name)
    name = re.sub(r'\bTDS\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\bpdf\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[_\-]+', ' ', name)
    name = re.sub(r' +', ' ', name).strip()
    return name


# Company noise — still printed by pdfplumber since it's real text in the PDF
_NOISE_RE = re.compile(
    r'VTR\s*&?\s*beyond.*?(?=\n[A-Z]|\Z)|'
    r'No\.\s*8,?\s*Pingbei.*?vtrbeyond\.com|'
    r'Stresemann\s*str.*?vtrbeyond\.com|'
    r'Tel\s*:\s*86-756.*|Mail\s*:.*?vtrbeyond\.com|Website\s*:.*?vtrbeyond\.com|'
    r'Last\s*updat(?:ing|e)\s*:?\s*\d{2}/\d{2}/\d{4}|'
    r'TECH\s*N(?:ICAL)?\s*DATA\s*SHEET',
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def _extract_pdf(pdf_path):
    """Extract text + tables from PDF via pdfplumber."""
    try:
        pages_text = []
        all_tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ''
                pages_text.append(text)
                for tbl in page.extract_tables():
                    all_tables.append(tbl)
        full_text = '\n'.join(pages_text)
        logger.info(f"Extracted {len(full_text)} chars, {len(all_tables)} tables from {pdf_path.name}")
        return full_text, all_tables
    except Exception as e:
        logger.error(f"Error reading {pdf_path.name}: {e}")
        return "", []


def _clean(text):
    """Minimal cleaning — pdfplumber gives clean text, just remove noise."""
    text = _NOISE_RE.sub('', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _norm(s):
    """Normalise a section's text content."""
    if not s:
        return ''
    s = re.sub(r'\n', ' ', s)
    s = re.sub(r' +', ' ', s)
    return s.strip()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION PARSING
# ═══════════════════════════════════════════════════════════════════════════

# Ordered section headers found in English TDS documents
_SECTION_HEADERS = [
    'Product Description', 'Effective material', 'Activity',
    'Application', 'Function', 'Dosage',
    'Standardization of Wheat Flour', 'Bread Improvement',
    'Suggested Optimum', 'Organoleptic', 'Physicochemical',
    'FOOD SAF',  # "FOOD SAFTY DATA" header from page 2
    'Microbiology', 'Heavy metals', 'Allergens',
    'GMO', 'Ionization', 'Packaging', 'Package', 'Storage', 'Date of',
]


def _parse_sections(text):
    """Parse cleaned text into {header: content} dict."""
    pat = '|'.join(re.escape(h) for h in _SECTION_HEADERS)
    positions = [(m.start(), m.group().strip()) for m in re.finditer(rf'({pat})', text, re.I)]
    if not positions:
        return {}
    positions.sort(key=lambda x: x[0])
    sections = {}
    for i, (pos, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = _norm(text[pos + len(header):end])
        content = re.sub(r'^[\s:]+', '', content)
        if content:
            # Normalise some header names for consistency
            key = header
            if key.startswith('FOOD'):
                key = 'Food Safety'
            elif key == 'Package':
                key = 'Packaging'
            sections[key] = content
    return sections


# ═══════════════════════════════════════════════════════════════════════════
# TABLE → TEXT
# ═══════════════════════════════════════════════════════════════════════════

def _food_safety_table_to_text(table, prefix):
    """Convert the page-2 food safety table into a readable chunk."""
    lines = [f"{prefix}: Food safety and allergen information."]
    for row in table:
        cells = [str(c or '').strip() for c in row if c and str(c).strip()]
        if not cells:
            continue
        combined = ' '.join(cells)
        # Skip the header row and empty rows
        if 'FOOD SAF' in combined.upper():
            continue
        if any(kw in combined.lower() for kw in [
            'microbio', 'salmonella', 'coliform', 'staphy', 'plate count',
            'cadmium', 'mercury', 'arsenic', 'lead', 'heavy metal',
            'allergen', 'gluten', 'gmo', 'irradiat', 'ioniz',
        ]):
            lines.append(combined)
    return ' '.join(lines) if len(lines) > 1 else ''


# ═══════════════════════════════════════════════════════════════════════════
# ENGLISH TDS CHUNKING — entity-centric
# ═══════════════════════════════════════════════════════════════════════════

def _build_english_chunks(product_name, enzyme_type, text, tables):
    """Build purpose-built chunks from an English TDS.

    Strategy: generate BOTH structured chunks (for targeted queries)
    AND raw section chunks (for diverse/unexpected queries).
    More chunks = better coverage = higher average retrieval score.
    """
    chunks = []
    etype = enzyme_type or 'enzyme'
    prefix = f"{product_name} ({etype})"
    sections = _parse_sections(text)

    # ══════════════════════════════════════════════════════════════════
    # PART A — STRUCTURED CHUNKS (optimized for common query types)
    # ══════════════════════════════════════════════════════════════════

    # ── 1. IDENTITY CHUNK ────────────────────────────────────────────
    desc = sections.get('Product Description', '')
    material = sections.get('Effective material', '')
    activity = sections.get('Activity', '')
    identity_parts = []
    if desc:
        identity_parts.append(f"Product: {desc}.")
    if material:
        identity_parts.append(f"Source: {material}.")
    if activity:
        identity_parts.append(f"Activity: {activity}.")
    if identity_parts:
        chunks.append(f"{prefix}: {' '.join(identity_parts)}")

    # ── 2. FUNCTION CHUNK ────────────────────────────────────────────
    func = sections.get('Function', '')
    app = sections.get('Application', '')
    combined_func = ''
    if app and func:
        if func.lower() in app.lower():
            combined_func = app
        elif app.lower() in func.lower():
            combined_func = func
        else:
            combined_func = f"{app}. {func}"
    else:
        combined_func = app or func
    if combined_func and len(combined_func) > 15:
        chunks.append(
            f"{prefix} function in bakery: {combined_func}.")

    # ── 3. DOSAGE CHUNKS (bilingual) ─────────────────────────────────
    dosage_raw = sections.get('Dosage', '')
    wheat_flour = sections.get('Standardization of Wheat Flour', '')
    bread_imp = sections.get('Bread Improvement', '')
    suggested = sections.get('Suggested Optimum', '')

    dosage_parts = []
    general_ppm = ''

    all_dosage_text = f"{dosage_raw} {wheat_flour} {bread_imp} {suggested}"
    ppm_vals = re.findall(r'([\d.]+\s*[-–]\s*[\d.]+)\s*ppm', all_dosage_text)

    if ppm_vals:
        general_ppm = ppm_vals[0].replace('–', '-').replace(' ', '')
    elif re.search(r'(\d+)\s*ppm', all_dosage_text):
        general_ppm = re.search(r'(\d+)\s*ppm', all_dosage_text).group(0)

    if dosage_raw and 'ppm' in dosage_raw.lower():
        dosage_parts.append(f"General dosage: {dosage_raw.strip()}")
    if wheat_flour:
        dosage_parts.append(f"Wheat flour standardization: {wheat_flour.strip()}")
    if bread_imp:
        dosage_parts.append(f"Bread improvement: {bread_imp.strip()}")
    if suggested:
        dosage_parts.append(f"Suggested optimum: {suggested.strip()}")

    if dosage_parts:
        chunks.append(f"{prefix} dosage for bakery: {'. '.join(dosage_parts)}.")
        if general_ppm:
            chunks.append(
                f"Dosage {etype} ({product_name}) boulangerie panification : "
                f"{general_ppm}.")

    # ── 4. STORAGE + SHELF LIFE CHUNK ────────────────────────────────
    storage = sections.get('Storage', '')
    date_of = sections.get('Date of', '')
    storage_text = ' '.join(filter(None, [date_of, storage])).strip()
    if storage_text:
        chunks.append(
            f"{prefix} storage conditions and shelf life: {storage_text}.")

    # ── 5. PACKAGING CHUNK ───────────────────────────────────────────
    pkg = sections.get('Packaging', '')
    if pkg:
        chunks.append(f"{prefix} packaging: {pkg}.")

    # ── 6. ALLERGEN CHUNK ────────────────────────────────────────────
    allergen = sections.get('Allergens', '')
    if allergen:
        chunks.append(f"{prefix} allergen information: {allergen}.")
    elif 'allergen' in text.lower():
        m = re.search(
            r'(Allergen\w*\s+.+?)(?=GMO|Ioniz|Package|Storage|\Z)',
            text, re.I | re.S)
        if m:
            chunks.append(f"{prefix} allergen information: {_norm(m.group(1))}")

    # ── 7. FOOD SAFETY CHUNK (from table) ────────────────────────────
    for tbl in tables:
        if not tbl or not tbl[0]:
            continue
        first_text = ' '.join(str(c or '') for c in tbl[0]).upper()
        if 'FOOD' in first_text or 'SAF' in first_text:
            safety = _food_safety_table_to_text(tbl, prefix)
            if safety:
                chunks.append(safety)

    # ── 8. PHYSICAL PROPERTIES CHUNK ─────────────────────────────────
    organo = sections.get('Organoleptic', '')
    physico = sections.get('Physicochemical', '')
    props = ' '.join(filter(None, [organo, physico])).strip()
    if props:
        chunks.append(f"{prefix} physical properties: {props}.")

    # ── 9. GMO + REGULATORY CHUNK ────────────────────────────────────
    gmo = sections.get('GMO', '')
    ioniz = sections.get('Ionization', '')
    reg = ' '.join(filter(None, [gmo, ioniz])).strip()
    if reg:
        chunks.append(f"{prefix} regulatory status: {reg}.")

    # ══════════════════════════════════════════════════════════════════
    # PART B — RAW SECTION CHUNKS (for diverse / unexpected queries)
    # ══════════════════════════════════════════════════════════════════
    # These complement the structured chunks. Even if they partially
    # overlap, having the raw text indexed separately means unexpected
    # query formulations can still match.
    for header, content in sections.items():
        if header in ('Food Safety',):
            continue  # already handled via table
        raw_chunk = f"{prefix}: {header} {content}"
        if len(raw_chunk) > 450:
            for sub in _split_sentences(raw_chunk, max_len=400):
                chunks.append(sub)
        elif len(raw_chunk) > 25:
            chunks.append(raw_chunk)

    # ── FALLBACK: raw text if no sections found ──────────────────────
    if not sections:
        clean = re.sub(r'\n', ' ', text).strip()
        if len(clean) > 30:
            for sub in _split_sentences(clean, max_len=450):
                chunks.append(f"{prefix}: {sub}")

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# FRENCH DOCUMENT CHUNKING (acide ascorbique)
# ═══════════════════════════════════════════════════════════════════════════

_FR_SECTION_PATTERNS = [
    r'Résumé\s*Général', r'Propriétés\s*Principales',
    r'Points?\s*Important', r'Dosages?\s*Recommandés?',
    r'Table\s*de\s*Conversion', r'Spécifications?\s*Techniques?',
    r'Caractéristiques?\s*du\s*Produit', r'Mode\s*d[\'\u2019]Emploi',
    r'Précautions?', r'Interaction',
    r'Notes?\s*(?:de\s*)?Formulation', r'Alternatives?\s*et\s*Complé',
    r'Conseils?\s*Pratiques?', r'Stabilité',
    r'Avantages\s+et\s+Limitations', r'Avantages\b', r'Limitations\b',
    r'Réglementation', r'Recommandations',
    r'Conditionnement', r'Stockage',
]


def _fr_table_to_text(table, label):
    """Convert a French table to natural-language text."""
    if not table or not table[0]:
        return ''
    rows = [r for r in table if any(c and str(c).strip() for c in r)]
    if len(rows) < 2:
        return ''
    header = [str(c or '').strip() for c in rows[0]]
    lines = [label]
    for row in rows[1:]:
        cells = [str(c or '').strip() for c in row]
        parts = []
        for h, v in zip(header, cells):
            if v and h:
                parts.append(f"{h}: {v}")
            elif v:
                parts.append(v)
        if parts:
            lines.append('. '.join(parts))
    return '\n'.join(lines)


def _chunk_french_document(product_name, text, tables):
    """Chunk the French ascorbic acid document with table support."""
    text = _clean(text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r' +', ' ', text)

    prefix = "Acide ascorbique (vitamine C, E300)"
    chunks = []

    # ── Section-based splitting ──────────────────────────────────────
    positions = [(0, 'Introduction')]
    for pat in _FR_SECTION_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            positions.append((m.start(), m.group()))
    positions.sort(key=lambda x: x[0])

    for i, (pos, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section = text[pos:end].strip()
        if len(section) < 50:
            continue
        if len(section) > 400:
            for sub in _split_sentences(section, max_len=350):
                chunks.append(f"{prefix} : {sub}")
        else:
            chunks.append(f"{prefix} : {section}")

    # ── Table chunks ─────────────────────────────────────────────────
    for tbl in tables:
        if not tbl or not tbl[0]:
            continue
        header_text = ' '.join(str(c or '') for c in tbl[0]).lower()
        if 'dosage' in header_text or 'type de production' in header_text:
            tbl_text = _fr_table_to_text(tbl, f"{prefix} — dosages recommandés par type de production")
            if tbl_text:
                chunks.append(tbl_text)
        elif 'poids' in header_text or 'farine' in header_text:
            tbl_text = _fr_table_to_text(tbl, f"{prefix} — table de conversion (ppm en grammes)")
            if tbl_text:
                chunks.append(tbl_text)
        elif 'alternative' in header_text:
            tbl_text = _fr_table_to_text(tbl, f"{prefix} — alternatives et complémentarité")
            if tbl_text:
                chunks.append(tbl_text)

    # ── Bilingual dosage labels ──────────────────────────────────────
    ppm_vals = re.findall(r'(\d+)\s*[-–à]\s*(\d+)\s*ppm', text)
    if ppm_vals:
        all_low = [int(v[0]) for v in ppm_vals]
        all_high = [int(v[1]) for v in ppm_vals]
        lo, hi = min(all_low), max(all_high)
        chunks.append(
            f"Dosage acide ascorbique (vitamine C, E300) "
            f"boulangerie panification : {lo}-{hi} ppm.")
        chunks.append(
            f"Recommended dosage of ascorbic acid (vitamin C, E300) "
            f"for bakery: {lo}-{hi} ppm on flour.")

    # ── Function/purpose chunk ───────────────────────────────────────
    resume_m = re.search(
        r'(?:Résumé\s*Général|Propriétés\s*Principales)\s*(.+?)(?='
        r'Points?\s*Important|Dosages?\s*Recommandés?|Table\s*de|'
        r'Spécifications?|Mode\s*d|\Z)', text, re.I | re.S)
    if resume_m:
        func_txt = resume_m.group(1).strip()
        func_txt = re.sub(r' +', ' ', func_txt)
        if len(func_txt) > 30:
            if len(func_txt) > 350:
                func_txt = func_txt[:350].rsplit(' ', 1)[0]
            chunks.append(
                f"À quoi sert l'acide ascorbique en boulangerie ? {func_txt}")

    # ── Storage chunk ────────────────────────────────────────────────
    storage_m = re.search(
        r'(?:Stockage|Conservation|Conditionnement)\s*[:\s]*(.+?)(?='
        r'Référence|Document|Utilisation|\Z)', text, re.I | re.S)
    if storage_m:
        storage_txt = storage_m.group(1).strip()[:250]
        chunks.append(f"{prefix} stockage et conservation : {storage_txt}")

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _split_sentences(text, max_len=350):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    out, cur = [], ''
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + ' ' + s).strip()
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return [c for c in out if len(c) > 15]


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def process_pdf(pdf_path):
    """Extract and chunk a PDF using pdfplumber with entity-centric chunking."""
    raw_text, tables = _extract_pdf(pdf_path)
    if not raw_text:
        return []

    product_name = _extract_product_name(pdf_path)

    if 'ascorbique' in str(pdf_path).lower():
        chunks = _chunk_french_document(product_name, raw_text, tables)
    else:
        cleaned = _clean(raw_text)
        enzyme_type = _detect_enzyme_type(product_name, cleaned)
        chunks = _build_english_chunks(product_name, enzyme_type, cleaned, tables)

    if not chunks:
        logger.warning(f"No chunks for {pdf_path.name}, fallback")
        cleaned = _clean(raw_text)
        cleaned = re.sub(r'\n', ' ', cleaned).strip()
        if cleaned:
            for sub in _split_sentences(cleaned, max_len=450):
                chunks.append(f"{product_name}: {sub}")

    # Deduplicate
    final, seen = [], set()
    for c in chunks:
        c = re.sub(r' +', ' ', c).strip()
        if len(c) < 20:
            continue
        norm = c.lower()
        if norm in seen:
            continue
        seen.add(norm)
        final.append(c)

    logger.info(f"{pdf_path.name}: {len(final)} chunks")
    return final
