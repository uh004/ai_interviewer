from langchain_core.prompts import ChatPromptTemplate
from src.utils.state import State, DeepQuestion
from src.utils.llm import get_llm

llm = get_llm()

def generate_question(state: State) -> State:
    question = state.get("current_question", "")
    answer = state.get("current_answer", "")
    strategy = state.get("current_strategy", "")
    resume_summary = state.get("resume_summary", "")
    keywords = ", ".join(state.get("resume_keywords", []))

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 인사담당 면접관입니다... (네 기존 프롬프트 그대로)"),
        ("human",
         "이력서 요약: {resume_summary}\n"
         "키워드: {keywords}\n"
         "현재 전략 영역: {strategy}\n"
         "이전 질문: {question}\n"
         "지원자 답변: {answer}\n"
         "→ 심화 질문 1개 생성")
    ])

    chain = prompt | llm.with_structured_output(DeepQuestion)
    resp: DeepQuestion = chain.invoke({
        "resume_summary": resume_summary,
        "keywords": keywords,
        "strategy": strategy,
        "question": question,
        "answer": answer,
    })

    return {**state, "current_question": resp.question.strip(), "current_answer": ""}