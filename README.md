# CRI Assistente

Un assistente virtuale per la Croce Rossa Italiana basato su un sistema RAG (Retrieval-Augmented Generation) per fornire risposte accurate basate sulla documentazione CRI.

## Struttura del Progetto

```
cri/
├── app/
│   ├── api/
│   │   ├── router.py       # Endpoints FastAPI
│   │   └── models.py       # Modelli Pydantic
│   ├── core/
│   │   ├── config.py       # Gestione configurazioni
│   │   └── logging.py      # Setup logging
│   ├── rag/
│   │   ├── engine.py       # Pipeline RAG
│   │   ├── memory.py       # Gestione memoria conversazioni
│   │   └── prompts.py      # Template dei prompt
│   └── utils/
│       └── helpers.py      # Funzioni di utilità
├── main.py                 # Entry point applicazione
├── .env.example            # Esempio variabili d'ambiente
├── requirements.txt        # Dipendenze del progetto
├── README.md               # Documentazione
└── .gitignore
```

## Pipeline RAG

Il sistema utilizza una pipeline RAG avanzata con le seguenti componenti:

1. **Retrieval**: Recupero dei documenti rilevanti da Qdrant Cloud
2. **Reranking**: Riordinamento dei risultati con Cohere Rerank per migliorare la pertinenza
3. **Generation**: Generazione di risposte contestuali utilizzando GPT-4

### Cohere Reranker

Il sistema implementa Cohere Rerank per migliorare significativamente la qualità delle risposte:

- Recupera inizialmente un set ampio di documenti (configurabile via `RETRIEVAL_TOP_K`)
- Applica il reranking di Cohere per identificare i documenti più pertinenti
- Utilizza solo i migliori documenti (configurabile via `RERANK_TOP_K`) per generare la risposta

Questo approccio permette di:
- Migliorare la pertinenza delle risposte
- Ridurre il "rumore" da documenti meno rilevanti
- Ottimizzare l'uso del contesto nel prompt per il modello LLM

## Configurazione

### Variabili d'Ambiente

Crea un file `.env` nella directory principale con le seguenti variabili:

```
# API Keys
OPENAI_API_KEY=sk-...
QDRANT_URL=https://...
QDRANT_API_KEY=...
QDRANT_COLLECTION=cri_docs
COHERE_API_KEY=...

# RAG Configuration
RETRIEVAL_TOP_K=70
RERANK_TOP_K=10
MEMORY_WINDOW_SIZE=4

# LLM Configuration
LLM_MODEL=gpt-4.1
EMBEDDING_MODEL=text-embedding-3-large

# Miscellaneous
LOG_LEVEL=INFO
ENVIRONMENT=development

```


## Avvio dell'Applicazione

```bash
uvicorn main:app --reload
```

L'applicazione sarà disponibile all'indirizzo `http://localhost:8000`.

## API Endpoints

- `POST /api/query`: Processa una query e restituisce una risposta contestuale
- `POST /api/reset`: Resetta la memoria della conversazione
- `GET /api/transcript`: Ottiene il transcript della conversazione
- `GET /api/contact`: Ottiene le informazioni di contatto della CRI
- `GET /health`: Endpoint di health check