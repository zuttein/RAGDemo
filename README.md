# RAG Demo

Detta är en enkel demo av ett **RAG-system** (Retrieval-Augmented Generation) byggt med:

- Streamlit
- LangGraph
- ChromaDB
- HuggingFace Embeddings
- Groq LLM

## Vad applikationen gör

Applikationen kan:

- omvandla dokument till embeddings
- lagra embeddings i en vector-databas
- göra semantisk sökning
- använda hybrid retrieval
- generera svar baserat på relevanta dokument
- route:a irrelevanta frågor med LangGraph

## Hur det fungerar

1. Dokument delas upp i chunks
2. Chunks omvandlas till embeddings
3. Embeddings lagras i ChromaDB
4. Användaren ställer en fråga
5. Systemet hämtar relevanta dokument
6. LLM genererar ett svar baserat på context

## Kör lokalt

Installera beroenden:

```bash
pip install -r requirements.txt

streamlit run app.py
