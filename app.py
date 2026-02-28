"""
Streamlit front-end for the RAG semantic search system.
Launch:  streamlit run app.py
"""
import streamlit as st
from services.search_service import search
from services.ingestion_data import ingest_pdfs
from config.settings import config

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG – Recherche Sémantique",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .score-badge {
        display: inline-block;
        background: #4CAF50;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .score-badge.medium { background: #FF9800; }
    .score-badge.low    { background: #f44336; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search-in-cloud.png", width=64)
    st.title("RAG Search")
    st.caption("Module de Recherche Sémantique\npour la Formulation en Boulangerie")
    st.divider()

    st.markdown("**Paramètres du système**")
    st.markdown(f"- Modèle : `all-MiniLM-L6-v2`")
    st.markdown(f"- Dimension : `{config.EMBEDDING_DIMENSION}`")
    st.markdown(f"- Top K : `{config.TOP_K}`")
    st.markdown(f"- Chunk size : `{config.CHUNK_SIZE}`")
    st.divider()

    # Ingestion button
    st.markdown("**Administration**")
    if st.button("🔄 Ré-ingérer les PDFs", use_container_width=True):
        with st.spinner("Ingestion en cours…"):
            ok = ingest_pdfs(config.PDF_FOLDER)
        if ok:
            st.success("Ingestion terminée ✓")
        else:
            st.error("Erreur lors de l'ingestion")

    st.divider()
    st.markdown(
        "<small>ARSII RAG Challenge — Rose Blanche Group</small>",
        unsafe_allow_html=True,
    )

# ── Main area ────────────────────────────────────────────────────────────
st.title("🔍 Recherche Sémantique")
st.markdown(
    "Posez une question en **français** ou en **anglais** sur les fiches techniques "
    "BVZyme (enzymes, améliorants, acide ascorbique…)."
)

# Example queries
EXAMPLES = [
    "Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?",
    "What is the recommended dosage of lipase?",
    "Quel est l'effet de la xylanase sur le volume du pain ?",
    "What is the function of ascorbic acid in bread making?",
    "Quelles sont les conditions de stockage de l'alpha-amylase ?",
]

# Query input
query = st.text_area(
    "Votre question :",
    height=80,
    placeholder="Ex: Quelles sont les quantités recommandées d'alpha-amylase ?",
)

col1, col2 = st.columns([1, 4])
with col1:
    search_btn = st.button("🔎 Rechercher", type="primary", use_container_width=True)
with col2:
    example_pick = st.selectbox(
        "Ou essayez un exemple :",
        [""] + EXAMPLES,
        label_visibility="collapsed",
    )

# Use example if picked and no manual query
active_query = query.strip() if query.strip() else example_pick

if search_btn or (example_pick and not query.strip()):
    if not active_query:
        st.warning("Veuillez entrer une question.")
    else:
        with st.spinner("Recherche en cours…"):
            results = search(active_query)

        if not results:
            st.info("Aucun résultat trouvé.")
        else:
            st.markdown(f"### Résultats pour : *{active_query}*")
            st.markdown("---")

            for res in results:
                score = res["score"]
                # Color coding
                if score >= 0.70:
                    badge_cls = ""
                elif score >= 0.50:
                    badge_cls = " medium"
                else:
                    badge_cls = " low"

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <strong>Résultat {res['rank']}</strong>
                            <span class="score-badge{badge_cls}">Score : {score:.4f}</span>
                        </div>
                        <div style="font-size:0.95rem; line-height:1.6;">
                            {res['texte']}
                        </div>
                        <div style="margin-top:0.4rem; font-size:0.75rem; color:#888;">
                            Document ID : {res['id_document']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Summary metrics
            scores = [r["score"] for r in results]
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Top Score", f"{max(scores):.4f}")
            col_b.metric("Avg Score", f"{sum(scores)/len(scores):.4f}")
            col_c.metric("Fragments", len(results))
