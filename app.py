import streamlit as st

from groq import Groq
from dotenv import load_dotenv
from graph import (
    build_graph,
    build_ingestion_graph
)

import os

from rag import (
    chroma_setup,
)

# Ladda env
load_dotenv()

# Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Chroma DB
db = chroma_setup()

# Initialisera graph
rag_graph = build_graph(client, db)

ingestion_graph = build_ingestion_graph(db)

# UI
st.title("RAG Demo")

st.subheader("Index Website")

url = st.text_input("Website URL")

if st.button("Scrape Website"):

    with st.spinner("Scraping website..."):

        ingestion_graph.invoke({
            "url": url
        })

    st.success("Website indexed!")

query = st.text_input(
    "Ställ en fråga"
)
all_docs = db.get()

print(all_docs)

if query:

    with st.spinner(
        "Analyserar dokument..."
    ):
         
        result = rag_graph.invoke({
            "query": query,
            "route": "",
            "answer": "",
            "sources": []
        })

        st.caption(f"Planner route: {result['route']}")
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
   



