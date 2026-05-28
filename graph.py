from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, START, END

from planner import planner_llm
from rag import rag_pipeline
from mcp_tools import scrape_website
from rag import ingest

class State(TypedDict): # Skapa state som skickas mellan noder
    query: str
    route: str
    answer: str
    sources: List[Any]
    
    url: str

def planner_node(state: State, client):
    query = state["query"]

    route = planner_llm(client, query)

    return {
        "route": route
    }

def route_decision(state: State):
    if state["route"] == "rag":
        return "rag"

    return "irrelevant"

def irrelevant_node(state: State):
    return {
        "answer": (
            "Irellevant fråga. Efterfrågad information finns inte i dokumenten."
        ),
        "sources": []
    }
    
def rag_node(state: State, client, db):
    result = rag_pipeline(
        client=client,
        db=db,
        query=state["query"]
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }
    
def scrape_node(state: State, db):

    url = state["url"]

    print(f"\nSCRAPING URL: {url}\n")

    website_text = scrape_website(url)

    print(f"\nSCRAPED TEXT LENGTH: {len(website_text)}\n")

    ingest(
        db=db,
        text=website_text,
        source=url
    )

    print("\nINGEST COMPLETE\n")

    return {}

# Planner borttagen just nu.
# Den användes tidigare för att routa frågor till:
# - rag
# - irrelevant
#
# Fungerade bättre när prompten var StoneBeach-specifik.
# Nu används en mer generell prompt eftersom olika hemsidor kan indexeras.

def build_graph(client, db):

    graph = StateGraph(State)

    # graph.add_node(
    #     "planner", lambda state: planner_node(state, client)
    # )

    graph.add_node(
        "rag", lambda state: rag_node(state, client, db)
    )

    # graph.add_node(
    #     "irrelevant", irrelevant_node
    # )

    # graph.add_edge(START, "planner")

    graph.add_edge(START, "rag")

    # graph.add_conditional_edges(
    #     "planner",
    #     route_decision,
    #     {
    #         "rag": "rag",
    #         "irrelevant": "irrelevant"
    #     }
    # )

    graph.add_edge("rag", END)

    # graph.add_edge("irrelevant", END)

    return graph.compile()

def build_ingestion_graph(db):
    
    graph = StateGraph(State)
    
    graph.add_node(
        "scrape",
        lambda state: scrape_node(state, db)
    )
    graph.add_edge(START, "scrape")
    graph.add_edge("scrape", END)
    return graph.compile()
