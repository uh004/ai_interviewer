from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.state import State
from src.utils.llm import get_llm

llm = get_llm()

final_eval_prompt = ChatPromptTemplate.from_template("""
당신은 기술 면접관입니다.
다음은 면접 질문별 요약과 평균 점수입니다.

평균 점수: {avg_score}/10
질문별 평가 요약:
{summary_for_llm}

이를 바탕으로 전체 인터뷰에 대한 종합평가를 3문장 이내로 작성하세요.
- 강점과 개선점을 모두 포함
- 객관적이고 자연스러운 평가 문체
""")
final_eval_chain = final_eval_prompt | llm | StrOutputParser()

def summarize_interview(state: State) -> State:
    evaluations = state.get("evaluation", {})
    order = list(state.get("question_strategy", {}).keys()) or ["경험","동기","논리"]

    report = ""
    report += "============================================================\n"
    report += " [AI 면접 피드백 보고서]\n"
    report += "============================================================\n"

    if not evaluations:
        report += " 평가 내역이 없습니다.\n"
        state["next_step"] = "end"
        state["final_report"] = report
        return state

    crits = ["구체성", "일관성", "적합성", "논리성"]
    total_scores = []
    summary_for_llm_lines = []

    for s in order:
        ev = evaluations.get(s, {})
        n = int(ev.get("_n", 0))
        if n <= 0:
            continue

        avg = {
            "구체성": ev.get("_sum_구체성", 0) / n,
            "일관성": ev.get("_sum_일관성", 0) / n,
            "적합성": ev.get("_sum_적합성", 0) / n,
            "논리성": ev.get("_sum_논리성", 0) / n,
        }

        overall = round((sum(avg.values()) / 4) * 10)
        total_scores.append(overall)

        report += f"\n[{s}] 종합 평가 (질문 {n}개 기반)\n"
        report += "------------------------------------------------------------\n"
        report += " 항목별 평균(0~1): " + ", ".join([f"{k} {avg[k]:.2f}" for k in crits]) + "\n"
        report += f" ▶ 종합 점수: {overall}/10\n"

        summary_for_llm_lines.append(
            f"- {s}: 종합 {overall}/10 | 평균(구체성 {avg['구체성']:.2f}, 일관성 {avg['일관성']:.2f}, 적합성 {avg['적합성']:.2f}, 논리성 {avg['논리성']:.2f})"
        )

    if total_scores:
        avg_score = round(sum(total_scores) / len(total_scores))
        report += "\n [인터뷰 전체 종합 평가]\n"
        report += "============================================================\n"
        report += f" 평균 점수: {avg_score}/10\n\n"
        final_feedback = final_eval_chain.invoke({
            "avg_score": avg_score,
            "summary_for_llm": "\n".join(summary_for_llm_lines)
        })
        report += f" {final_feedback}\n"

    report += "============================================================\n"
    report += " 인터뷰가 종료되었습니다. 수고하셨습니다!\n"

    state["next_step"] = "end"
    state["final_report"] = report
    return state