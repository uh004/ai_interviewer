from langgraph.graph import StateGraph, END
from src.utils.state import State
from src.nodes.evaluate import evaluate_answer
from src.nodes.router import decide_next_step, route_next, change_strategy
from src.nodes.gen_question import generate_question
from src.nodes.summarize import summarize_interview

def build_graph():
    builder = StateGraph(State)

    builder.add_node("evaluate", evaluate_answer)
    builder.add_node("decide", decide_next_step)
    builder.add_node("generate", generate_question)
    builder.add_node("change_strategy", change_strategy)
    builder.add_node("summarize", summarize_interview)

    builder.set_entry_point("evaluate")
    builder.add_edge("evaluate", "decide")

    builder.add_conditional_edges(
        "decide",
        route_next,
        {"generate": "generate", "change_strategy": "change_strategy", "summarize": "summarize"}
    )

    builder.add_edge("generate", END)
    builder.add_edge("change_strategy", END)
    builder.add_edge("summarize", END)

    return builder.compile()