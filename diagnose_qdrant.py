import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Carica le variabili d'ambiente da .env se presente
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION") # Legge dal .env

if not QDRANT_URL or not QDRANT_API_KEY:
    print("ERRORE: QDRANT_URL o QDRANT_API_KEY non trovate nelle variabili d'ambiente.")
    print("Assicurati di avere un file .env configurato o di averle esportate.")
    exit(1)

if not QDRANT_COLLECTION:
    print("ERRORE: QDRANT_COLLECTION non trovata nelle variabili d'ambiente o nel file .env.")
    exit(1)


print(f"Tentativo di connessione a Qdrant URL: {QDRANT_URL}")
print(f"Usando la collezione: {QDRANT_COLLECTION}")

try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=20) # Timeout aumentato per sicurezza
    
    print("\n--- Informazioni sulla Collezione ---")
    collection_info = client.get_collection(collection_name=QDRANT_COLLECTION)
    
    print(f"Nome Collezione (usato per la query): {QDRANT_COLLECTION}") # Stampiamo il nome che abbiamo usato
    print(f"Stato Collezione: {collection_info.status}")
    print(f"Numero Punti (circa): {collection_info.points_count}")
    
    # Gestione di vectors_count che può essere un int o un dict
    if isinstance(collection_info.vectors_count, int):
        print(f"Numero Vettori (circa): {collection_info.vectors_count}")
    elif isinstance(collection_info.vectors_count, dict):
        print(f"Numero Vettori (per nome): {collection_info.vectors_count}")
    else:
        print(f"Numero Vettori (formato sconosciuto): {collection_info.vectors_count}")

    print(f"Configurazione Vettori (Params): {collection_info.config.params}")
    print(f"Configurazione Vettori (HNSW): {collection_info.config.hnsw_config}")
    print(f"Configurazione Vettori (Optimizers): {collection_info.config.optimizer_config}")
    print(f"Configurazione Vettori (Quantization): {collection_info.config.quantization_config}")
    
    print(f"\n--- Schema Payload Rilevato da Qdrant ---")
    if collection_info.payload_schema:
        for field_name, field_info in collection_info.payload_schema.items():
            print(f"  Campo: '{field_name}', Tipo Indice: {type(field_info)}, Dettagli: {field_info}")
    else:
        print("  Nessuno schema payload esplicito rilevato (potrebbe essere dinamico).")


    print(f"\n--- Esempio di alcuni Punti (max 3) dalla collezione '{QDRANT_COLLECTION}' ---")
    # Usiamo il metodo scroll per recuperare i punti
    # Se la collezione è molto grande e non ci sono filtri, scroll può essere lento
    # Per collezioni grandi, potresti voler aggiungere un filtro o usare client.get_points con ID specifici se li conosci
    
    # Tentativo di recuperare i primi punti disponibili
    points_response, next_offset = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=3, # Recuperiamo solo 3 punti per l'esempio
        with_payload=True, # Vogliamo il payload per ispezionarlo
        with_vectors=False # Non ci servono i vettori per questa diagnosi
    )

    if not points_response:
        print(f"Nessun punto trovato nella collezione '{QDRANT_COLLECTION}' usando scroll (limit=3). La collezione potrebbe essere vuota o lo scroll non ha restituito risultati immediatamente.")
    else:
        for i, point in enumerate(points_response):
            print(f"\n--- Punto Esempio #{i+1} (ID: {point.id}) ---")
            print("Payload:")
            if point.payload:
                for key, value in point.payload.items():
                    # Stampa solo un'anteprima per valori di testo lunghi
                    if isinstance(value, str) and len(value) > 150:
                        print(f"   Chiave: '{key}', Valore (anteprima): '{value[:150]}...'")
                    else:
                        print(f"  Chiave: '{key}', Valore: {value}")
            else:
                print("  (Nessun payload per questo punto)")
    
    print("\n--- Diagnosi Completata ---")

except Exception as e:
    print(f"\nERRORE DURANTE LA DIAGNOSI QDRANT: {e}")
    import traceback
    traceback.print_exc()