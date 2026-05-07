import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq
from dotenv import load_dotenv
import os
from text import texts, metadatas

@st.cache_resource # Spara db i cache
def chroma_setup():
    # Skapar en text-splitter
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    
    #Omvandlar text till dokument object (Varje text blir ett dokument)
    docs = splitter.create_documents(texts=texts, metadatas=metadatas)
    #st.write("Antal dokument:", len(docs))
    
    # Skapar en embedding-modell gör om mina dokument till vektorer
    embeddings = HuggingFaceEmbeddings()

    return Chroma.from_documents(docs, embeddings)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Alla "dokument" blir till embeddings och sparas
db = chroma_setup()

# UI
st.title("Retrieval Demo")

# UI
query = st.text_input("Ställ en fråga")

# Retrieval - R 
#När användaren skriver något körs retrieval
if query:
    #Frågan blir embedding och jämförs med alla "dokument", räknar likheter och returnerar den bästa matchen
    results = db.similarity_search(query, k=4)

    #  Context - A 

    # Sätt ihop alla dokument till en text som ska till AI
    #context = "\n\n".join([doc.page_content for doc in results ])
    context = "\n\n".join([
        f"""
            Källa: {doc.metadata.get("source_id")}
            Titel: {doc.metadata.get("title")}
            Kategori: {doc.metadata.get("category")}
            Ämne: {doc.metadata.get("topic")}

            Text: {doc.page_content}
            
        """
        for doc in results
    ])

    #  LLM - G 

    prompt = f"""
        Context: {context}

        Fråga: {query}

        Ge ditt svar:
    """

    response = client.chat.completions.create( # Skapa en klient från Groq. Specifiera modell och instruktioner till AI samt prompt
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
                            Du är en RAG-assistent.
                            Regler:
                            1. Svara enbart baserat på informationen i context.
                            2. Om svaret inte finns i context svara då: "Efterfrågad information finns inte i dokumenten"
                            3. Hitta inte på information.
                            4. Svara tydligt och kort.
                            5. Källhänvisa alltid till de dokument du svarar utifrån, visa då source_id och titel.
                            6. Om frågan ej är relaterad till Stonebeach svara : "Jag svarar enbart på frågor gällande StoneBeach"
                            """
            },
            { 
                "role": "user",
                "content": prompt
            }
            ],
            temperature=0.3
    )

    answer = response.choices[0].message.content
    st.write(answer)




