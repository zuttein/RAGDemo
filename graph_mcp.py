from typing import TypedDict, Annotated

from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.graph.message import add_messages

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)

from langchain_core.messages import BaseMessage


class State(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


def planner(state, llm):

    print("\n===================")
    print("PLANNER CALLED")
    print("===================")

    print(state["messages"])

    response = llm.invoke(
        state["messages"]
    )

    print(response)

    return {
        "messages": [response]
    }

def build_graph(llm, allowed_tools):

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

    graph.add_conditional_edges(
        "planner",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    print("TOOLS -> PLANNER EDGE CREATED")

    graph.add_edge(
        "tools",
        "planner"
    )

    return graph.compile()