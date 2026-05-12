import streamlit as st

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompts import RAG_SYSTEM_PROMPT
from text import texts, metadatas


# Cache:a vector database
@st.cache_resource
def chroma_setup():

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    # Skapar dokument
    docs = splitter.create_documents(
        texts=texts,
        metadatas=metadatas
    )

    # Embedding-modell
    embeddings = HuggingFaceEmbeddings()

    # Skapar Chroma vector DB
    return Chroma.from_documents(
        docs,
        embeddings
    )


# Retrieval + generation pipeline
def rag_pipeline(client, db, query):

    # Retrieval med similarity score
    results = db.similarity_search_with_score(
        query,
        k=4
    )

    relevant_results = []

    # Filtrera relevanta dokument
    for doc, score in results:

        print(f"Score: {score}")

        if score < 0.9:
            relevant_results.append(doc)

    # Begränsa antal dokument
    relevant_results = relevant_results[:2]

    # Om inget relevant hittas
    if not relevant_results:

        return {
            "answer": (
                "Efterfrågad information "
                "finns inte i dokumenten"
            ),
            "sources": []
        }

    # Bygg context
    context = "\n\n".join([
        f"""
        Källa: {doc.metadata.get("source_id")}
        Titel: {doc.metadata.get("title")}
        Kategori: {doc.metadata.get("category")}
        Ämne: {doc.metadata.get("topic")}

        Text: {doc.page_content}
        """
        for doc in relevant_results
    ])

    # User prompt
    prompt = f"""
    Context:
    {context}

    Fråga:
    {query}

    Ge ditt svar:
    """

    # LLM generation
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": RAG_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": relevant_results
    }