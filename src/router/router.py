from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.router.states import RouterState
from src.expert.sentiment_agent import sentiment_label_node
from src.expert.highlight_agent import highlight_extract_node
from src.expert.expert_agent import parenting_advice_node


def build_question_router():
    graph = StateGraph(RouterState)

    graph.add_node("sentiment_labeler", sentiment_label_node)
    graph.add_node("highlight_extractor", highlight_extract_node)
    graph.add_node("parenting_advice", parenting_advice_node)

    # parallel: both start from START
    graph.add_edge(START, "sentiment_labeler")
    graph.add_edge(START, "highlight_extractor")

    # converge into parenting_advice; it runs after both predecessors complete
    graph.add_edge("sentiment_labeler", "parenting_advice")
    graph.add_edge("highlight_extractor", "parenting_advice")
    graph.add_edge("parenting_advice", END)

    return graph.compile()
