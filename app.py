import streamlit as st

from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from graph import (
    build_graph,
    build_ingestion_graph
)

import os

from rag import (
    chroma_setup,
)

import asyncio

from graph_mcp import build_graph as build_mcp_graph

from langchain_core.messages import HumanMessage

from scrape_tool import create_scrape_tool


# Ladda env
load_dotenv()

# Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

scrape_tool = create_scrape_tool()

# Exponera endast vår lokala scraper-tool för modellen i MCP-testflödet.
tools = [scrape_tool]

llm_with_tools = llm.bind_tools(tools)

mcp_graph = build_mcp_graph(
    llm_with_tools,
    tools
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



if st.button("Test MCP Graph"):

    # Testa att modellen väljer scraper-toolen och att resultatet skickas tillbaka i grafen.
    result = asyncio.run(
        mcp_graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Scrape https://stonebeach.se"
                )
            ]
        }
    ))

    print("\nRESULT:")
    print(result)

    print("\nMESSAGES:")
    for msg in result["messages"]:
        print(type(msg))
        print(msg)
        print("----------------")
    

    print("\nFINAL RESULT:")
    st.write(result)
    
    
    
if st.button("Scrape Website"):

    with st.spinner("Scraping website..."):

        ingestion_graph.invoke({
            "url": url
        })

    st.success("Website indexed!")
    
if st.button("Show DB"):

    all_docs = db.get()

    st.write(all_docs)
    
if st.button("Clear database"):

    all_docs = db.get()

    if all_docs["ids"]:
        db.delete(ids=all_docs["ids"])
   

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

                    chunk_index = doc.metadata.get("chunk_index")
                    chunk_id = doc.metadata.get("chunk_id")
                    preview = doc.page_content[:500]

                    if chunk_index is not None:
                        st.markdown(f"Chunk: `{chunk_index}`")

                    if chunk_id:
                        st.markdown(f"Chunk ID: `{chunk_id}`")

                    with st.expander("Visa hämtad text"):
                        st.write(preview)
   


