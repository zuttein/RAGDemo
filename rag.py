import re
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from prompts import RAG_SYSTEM_PROMPT
#from text import texts, metadatas


# -------------------------
# Cache:a vector database
# -------------------------

# @st.cache_resource
# def chroma_setup():
#
#     splitter = CharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=80
#     )
#
#     docs = splitter.create_documents(
#         texts=texts,
#         metadatas=metadatas
#     )
#
#     embeddings = HuggingFaceEmbeddings()
#
#     db = Chroma.from_documents(
#         docs,
#         embeddings
#     )
#
#     return db
@st.cache_resource
def chroma_setup():

    embeddings = HuggingFaceEmbeddings()

    db = Chroma(
        embedding_function=embeddings
    )

    return db


# -------------------------
# Retrieval + generation pipeline
# -------------------------

def rag_pipeline(client, db, query):

    # -------------------------
    # VECTOR SEARCH
    # -------------------------

    results = db.similarity_search_with_score(
        query,
        k=4
    )

    relevant_results = []

    # Filtrera relevanta dokument
    for doc, score in results:

        print(f"""
        VECTOR MATCH
        Score: {score}
        Content:
        {doc.page_content[:500]}
        """)

        if score < 1.2:

            relevant_results.append(doc)

# Keyword search borttagen tills vidare.
# Skrev över relevanta vector-träffar och gav sämre resultat.

    # -------------------------
    # HYBRID / KEYWORD SEARCH
    # -------------------------

    # uery_words = re.findall(
    #  r"\w+",
    #   query.lower()
    # )

    # # Filtrering av vanliga ord
    # stopwords = {
    # "hur",
    # "vad",
    # "är",
    # "med",
    # "och",
    # "kan",
    # "jag",
    # "om",
    # "för",
    # "att"
    # }

    # query_words = [
    # word
    # for word in query_words
    # if word not in stopwords
    # ]

    # all_docs = db.get()

    # for i, text in enumerate(all_docs["documents"]):

    #     metadata = all_docs["metadatas"][i]

    #     searchable_text = f"""
    #     {text}
    #     {metadata.get("title", "")}
    #     {metadata.get("category", "")}
    #     {metadata.get("topic", "")}
    #     """.lower()

    #     # Keyword match
    #     if any(
    #         word in searchable_text
    #         for word in query_words
    #     ):

    #         print(f"""
    #         KEYWORD MATCH
    #         Title: {metadata.get("title")}
    #         """)

    #         already_exists = False

    #         for existing_doc in relevant_results:
    #             existing_key = (
    #                 existing_doc.metadata.get("chunk_id")
    #                 or existing_doc.metadata.get("source_id")
    #             )
    #             keyword_key = (
    #                 metadata.get("chunk_id")
    #                 or metadata.get("source_id")
    #             )

    #             if existing_key == keyword_key:

    #                 already_exists = True
    #                 break

    #         # Lägg till om dokumentet inte redan finns
    #         if not already_exists:

    #             keyword_doc = Document(
    #                 page_content=text,
    #                 metadata=metadata
    #             )

    #             relevant_results.insert(0, keyword_doc)

    # -------------------------
    # Begränsa antal dokument
    # -------------------------

    relevant_results = relevant_results[:4]

    # -------------------------
    # Om inget relevant hittas
    # -------------------------

    if not relevant_results:

        return {
            "answer": (
                "Efterfrågad information "
                "finns inte i dokumenten"
            ),
            "sources": []
        }

    # -------------------------
    # Bygg context
    # -------------------------

    context = "\n\n".join([

        f"""
        Källa: {doc.metadata.get("source_id")}
        Titel: {doc.metadata.get("title")}
        Kategori: {doc.metadata.get("category")}
        Ämne: {doc.metadata.get("topic")}

        Text:
        {doc.page_content}
        """

        for doc in relevant_results

    ])

    # -------------------------
    # User prompt
    # -------------------------

    prompt = f"""
    Context:
    {context}

    Fråga:
    {query}

    Ge ditt svar:
    """

    # -------------------------
    # LLM generation
    # -------------------------
    
    print("\nCONTEXT SENT TO LLM:\n")
    print(context[:2000])

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
    
def ingest(db, text, source):
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    print(text[:500])
    chunks = splitter.split_text(text)

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "title": source,
                "category": "website",
                "topic": "scraped",
                "source_id": source,
                "source_url": source,
                "chunk_index": i,
                "chunk_id": f"{source}#chunk-{i}",
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    print(f"\nTEXT LENGTH: {len(text)}")
    print(f"ADDED {len(docs)} CHUNKS\n")

    for i, doc in enumerate(docs):
        print(f"Chunk {i}: {len(doc.page_content)} chars")
    
    db.add_documents(docs)
