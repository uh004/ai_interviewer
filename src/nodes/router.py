from typing import Literal
import random
from src.utils.state import State

def decide_next_step(state: State) -> State:
    threshold = 0.75

    qs = state.get("question_strategy", {})
    seq = list(qs.keys())
    cur = state.get("current_strategy", seq[0] if seq else "")
    idx = seq.index(cur) if cur in seq else 0

    ev = state.get("evaluation", {})
    cur_ev = ev.get(cur, {})

    if cur_ev.get("_n", 0) > 0:
        n = int(cur_ev["_n"])
        score = (
            cur_ev.get("_sum_구체성", 0)/n +
            cur_ev.get("_sum_일관성", 0)/n +
            cur_ev.get("_sum_적합성", 0)/n +
            cur_ev.get("_sum_논리성", 0)/n
        ) / 4.0
    else:
        score = 0.0

    deep_counts = state.get("deep_counts", {})
    cur_deep = int(deep_counts.get(cur, 0))

    if (score >= threshold) or (cur_deep >= 2):
        if seq and idx >= len(seq) - 1:
            return {**state, "next_step": "summarize", "deep_counts": deep_counts}
        next_strategy = seq[idx + 1]
        return {**state, "next_step": "change_strategy", "current_strategy": next_strategy, "deep_counts": deep_counts}
    else:
        cur_deep += 1
        deep_counts[cur] = cur_deep
        return {**state, "next_step": "generate", "deep_counts": deep_counts}

def change_strategy(state: State) -> State:
    qs = state.get("question_strategy", {})
    cur = state.get("current_strategy", "")
    block = qs.get(cur, {})

    questions = [v for v in block.values() if isinstance(v, str) and v.strip()]
    selected = random.choice(questions) if questions else "다음 질문을 준비 중입니다."

    return {**state, "current_question": selected, "current_answer": "", "next_step": ""}

def route_next(state: State) -> Literal["generate", "change_strategy", "summarize"]:
    step = state.get("next_step", "")
    if step in ("summarize", "end"):
        return "summarize"
    if step == "change_strategy":
        return "change_strategy"
    return "generate"