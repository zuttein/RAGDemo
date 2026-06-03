from typing import TypedDict, Annotated

from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langchain_core.messages import BaseMessage

# Delat state för grafen
# Messages byggs på automatiskt mellan noderna
class State(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

# Planner-noden använder LLM:en för att avgöra
# om ett tool ska anropas eller om flödet ska avslutas
def planner(state, llm):

    print("\n===================")
    print("PLANNER CALLED")
    print("===================")

    print(state["messages"])

    response = llm.invoke(
        state["messages"]
    )

    print(response)
    print("\nTOOL CALLS:")
    print(response.tool_calls)

    return {
        "messages": [response]
    }

def build_graph(llm, allowed_tools):
    
    # Asvarar för att verkställa tools
    # som LLM:en har valt.
    tool_node = ToolNode(allowed_tools)

    graph = StateGraph(State)

    graph.add_node(
        "planner",
        lambda state: planner(state, llm)
    )

    graph.add_node(
        "tools",
        tool_node
    )

    graph.add_edge(
        START,
        "planner"
    )
    # Om LLM:en returnerar ett tool_call skickas
    # flödet vidare till ToolNode, annars avslutas grafen
    graph.add_conditional_edges(
        "planner",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )

    # Nuvarande implementation tillåter endast
    # ett tool-anrop innan workflowet avslutas
    graph.add_edge(
        "tools",
        END
    )

    return graph.compile()
