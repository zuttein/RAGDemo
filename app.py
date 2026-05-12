import streamlit as st

from groq import Groq
from dotenv import load_dotenv
from planner import planner_llm

import os

from rag import (
    chroma_setup,
    rag_pipeline
)

# Ladda env
load_dotenv()

# Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Chroma DB
db = chroma_setup()

# UI
st.title("RAG Demo")

query = st.text_input(
    "Ställ en fråga"
)

if query:

    with st.spinner(
        "Analyserar dokument..."
    ):
         #=============================================================
         # Detta är en tillfällig lösning för att testa planner logiken
         #=============================================================
        # Planner avgör route
        route = planner_llm(
        client,
        query
        )
        #Visar vald route i UI för debug
        st.caption(
            f"Planner route: {route}"
        )
        
        #Tillfällig lösning via if-sats
        if route == "irrelevant":

           st.write(
                "Jag svarar enbart på frågor gällande StoneBeach"
           )
           #Stoppar resten av applikationen, alltså körs aldrig retrieval
           st.stop()

        # Kör RAG pipeline
        result = rag_pipeline(
            client,
            db,
            query
        )

        # User message
        with st.chat_message("user"):
            st.write(query)

        # Assistant message
        with st.chat_message("assistant"):

            st.write(
                result["answer"]
            )

            # Visa källor
            if result["sources"]:

                st.subheader("Källor")

                for doc in result["sources"]:

                    st.markdown(
                        f"""
                        📄 **{doc.metadata.get("title")}**

                        ID:
                        `{doc.metadata.get("source_id")}`

                        Kategori:
                        {doc.metadata.get("category")}
                        """
                    )
   



