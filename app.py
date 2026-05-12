import streamlit as st

from groq import Groq
from dotenv import load_dotenv
from planner import planner_llm
from graph import build_graph

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

# UI
st.title("RAG Demo")

query = st.text_input(
    "Ställ en fråga"
)

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
   



