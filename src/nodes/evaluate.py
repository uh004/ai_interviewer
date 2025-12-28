from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from src.utils.state import State, FourCriteriaEval
from src.utils.llm import get_llm

llm = get_llm()


def _ensure_state_schema(state: Dict[str, Any]) -> Dict[str, Any]:
    ev = state.get("evaluation")
    if ev is None or not isinstance(ev, dict):
        ev = {}
    conv = state.get("conversation")
    if not isinstance(conv, list):
        conv = []
    state["evaluation"] = ev
    state["conversation"] = conv
    return state


def evaluate_answer(state: State) -> State:
    state = _ensure_state_schema(dict(state))

    question = state.get("current_question", "")
    answer = state.get("current_answer", "")
    strategy = state.get("current_strategy", "기본")

    resume_summary = state.get("resume_summary", "")
    resume_keywords = ", ".join(state.get("resume_keywords", []))

    evaluation: Dict[str, Dict[str, Any]] = state.get("evaluation", {})
    if strategy not in evaluation:
        evaluation[strategy] = {
            "구체성": 0, "일관성": 1, "적합성": 0, "논리성": 0,
            "question": "", "answer": "",
            "_n": 0,
            "_sum_구체성": 0, "_sum_일관성": 0, "_sum_적합성": 0, "_sum_논리성": 0,
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 면접관이다. 아래 4개 항목을 서로 독립적으로 0/1로 채점한다.\n"
         "- 구체성: 수치/사실/역할/방법/결과가 명확하면 1\n"
         "- 일관성: 주장-근거-사례 흐름이 자연스러우면 1\n"
         "- 적합성: 질문 의도/직무/경험과 직접 연결되면 1\n"
         "- 논리성: 원인→행동→결과가 인과적으로 이어지면 1\n"
         "각 항목은 score(0/1)와 rationale(1~2문장)만 출력한다."
         ),

        # good shot
        ("human",
         "컨텍스트 요약: CV 프로젝트 경험\n키워드: OpenCV, 전처리\n"
         "전략 영역: 경험\n질문: 성능을 어떻게 개선했나요?\n"
         "답변: 노이즈 제거와 기울기 보정을 추가했고, 전처리 단계 개선으로 정확도를 86%→94%로 올렸습니다. "
         "저는 전처리 파이프라인을 설계/적용했습니다."
         ),
        ("ai",
         '{"specificity":{"score":1,"rationale":"수치/역할/방법이 구체적"},'
         '"consistency":{"score":1,"rationale":"개선 방법과 결과가 연결됨"},'
         '"fit":{"score":1,"rationale":"질문 의도에 직접 답함"},'
         '"logic":{"score":1,"rationale":"원인-행동-결과 구조가 명확"}}'
         ),

        # bad shot
        ("human",
         "컨텍스트 요약: 프로젝트 경험\n키워드: 협업\n"
         "전략 영역: 경험\n질문: 성능을 어떻게 개선했나요?\n"
         "답변: 열심히 했고 팀워크가 좋았습니다."
         ),
        ("ai",
         '{"specificity":{"score":0,"rationale":"수치/방법/역할이 없음"},'
         '"consistency":{"score":0,"rationale":"흐름이 없음"},'
         '"fit":{"score":0,"rationale":"질문과 직접 연결이 약함"},'
         '"logic":{"score":0,"rationale":"원인/과정/결과가 없음"}}'
         ),

        # real
        ("human",
         "컨텍스트 요약: {resume_summary}\n키워드: {resume_keywords}\n"
         "전략 영역: {strategy}\n질문: {question}\n답변: {answer}\n")
    ])

    chain = prompt | llm.with_structured_output(FourCriteriaEval)
    result: FourCriteriaEval = chain.invoke({
        "resume_summary": resume_summary,
        "resume_keywords": resume_keywords,
        "strategy": strategy,
        "question": question,
        "answer": answer
    })

    sp = int(result.specificity.score)
    co = int(result.consistency.score)
    fi = int(result.fit.score)
    lo = int(result.logic.score)

    ev = evaluation[strategy]
    ev["_n"] += 1
    ev["_sum_구체성"] += sp
    ev["_sum_일관성"] += co
    ev["_sum_적합성"] += fi
    ev["_sum_논리성"] += lo

    ev["구체성"] = sp
    ev["일관성"] = co
    ev["적합성"] = fi
    ev["논리성"] = lo
    ev["question"] = question
    ev["answer"] = answer

    evaluation[strategy] = ev

    conversation = state.get("conversation", [])
    conversation.append({"question": question, "answer": answer})

    return {**state, "evaluation": evaluation, "conversation": conversation}