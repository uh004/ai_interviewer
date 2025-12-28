from src.utils.state import InterviewState


def update_current_answer(state: InterviewState, user_answer: str) -> InterviewState:

    return {**state, "current_answer": (user_answer or "").strip()}