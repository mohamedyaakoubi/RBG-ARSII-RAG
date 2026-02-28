from services.ingestion_data import ingest_pdfs
from services.search_service import search, display_results
from config.settings import config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Point d'entrée principal du projet RAG"""
    
    print("\n" + "="*80)
    print(" "*20 + "🔍 RAG - Système de Recherche Sémantique")
    print("="*80 + "\n")
    
    while True:
        print("\nMenu Principal:")
        print("1. Ingérer les documents PDFs")
        print("2. Rechercher dans les documents")
        print("3. Quitter")
        print("-"*80)
        
        choice = input("Choisissez une option (1/2/3): ").strip()
        
        if choice == "1":
            ingest_documents()
        elif choice == "2":
            search_query()
        elif choice == "3":
            print("\n✓ Au revoir!")
            break
        else:
            print("❌ Option invalide. Veuillez réessayer.")

def ingest_documents():
    """Ingérer les PDFs dans la base de données"""
    print("\n" + "-"*80)
    print("📥 Ingestion des Documents")
    print("-"*80)
    
    pdf_folder = config.PDF_FOLDER
    print(f"📂 Dossier source: {pdf_folder}")
    
    confirm = input("\nÊtes-vous sûr? (o/n): ").strip().lower()
    if confirm != 'o':
        print("❌ Opération annulée.")
        return
    
    print("\n⏳ Traitement en cours...\n")
    success = ingest_pdfs(pdf_folder)
    
    if success:
        print("\n✅ Ingestion terminée avec succès!")
    else:
        print("\n❌ Erreur lors de l'ingestion.")

def search_query():
    """Rechercher une question dans les documents"""
    print("\n" + "-"*80)
    print("🔎 Recherche Sémantique")
    print("-"*80)
    
    question = input("\nEntrez votre question: ").strip()
    
    if not question:
        print("❌ Question vide. Veuillez réessayer.")
        return
    
    print("\n⏳ Recherche en cours...\n")
    results = search(question)
    
    if results:
        display_results(question, results)
    else:
        print("❌ Aucun résultat trouvé.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Programme interrompu par l'utilisateur.")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        print(f"\n❌ Erreur: {e}")