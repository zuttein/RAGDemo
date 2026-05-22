from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, START, END

from planner import planner_llm
from rag import rag_pipeline

class State(TypedDict): # Skapa state som skickas mellan noder
    query: str
    route: str
    answer: str
    sources: List[Any]

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
            "Jag svarar enbart på frågor gällande StoneBeach"
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

def build_graph(client, db):
    graph = StateGraph(State)

    graph.add_node(
        "planner", lambda state: planner_node(state, client)
    )
    graph.add_node(
        "rag", lambda state: rag_node(state, client, db)
    )
    graph.add_node(
        "irrelevant", irrelevant_node
    )
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_decision,
        {
            "rag": "rag",
            "irrelevant": "irrelevant"
        }
    )

    graph.add_edge("rag", END) #test
    graph.add_edge("irrelevant", END)

    return graph.compile()
