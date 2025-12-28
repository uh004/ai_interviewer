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
    order = list(state.get("question_strategy", {}).keys()) if isinstance(state.get("question_strategy", {}), dict) else ["경험","동기","논리"]

    print("\n" + "=" * 60)
    print(" [AI 면접 피드백 보고서]")
    print("=" * 60)

    if not evaluations:
        print(" 평가 내역이 없습니다.")
        print("=" * 60)
        state["next_step"] = "end"
        return state

    crits = ["구체성", "일관성", "적합성", "논리성"]
    total_scores = []
    summary_for_llm_lines = []  # 최종평 프롬프트에 넣을 간단 요약

    for s in order:
        ev = evaluations.get(s, {})
        n = int(ev.get("_n", 0))

        # 누적값이 없는(=이번 키워드에 질문이 없거나, 이전 방식으로만 저장된) 경우 fallback
        if n > 0:
            avg = {
                "구체성": ev.get("_sum_구체성", 0) / n,
                "일관성": ev.get("_sum_일관성", 0) / n,
                "적합성": ev.get("_sum_적합성", 0) / n,
                "논리성": ev.get("_sum_논리성", 0) / n,
            }
        else:
            # 평면 구조(마지막 1건만)로 저장되어 있으면 그것으로라도 계산
            avg = {k: float(ev.get(k, 0)) for k in crits}
            # 전부 0이고 실제로 질문이 없을 수 있으니, 출력 스킵 처리
            if sum(avg.values()) == 0:
                continue
            n = 1  # 마지막 1건 기준

        overall = round((sum(avg.values()) / 4) * 10)  # 10점 환산
        total_scores.append(overall)

        strengths  = [k for k, v in avg.items() if v >= 0.75]
        weaknesses = [k for k, v in avg.items() if v < 0.5]

        print(f"\n[{s}] 종합 평가 (질문 {n}개 기반)")
        print("-" * 60)
        print(" 항목별 평균(0~1): " + ", ".join([f"{k} {avg[k]:.2f}" for k in crits]))
        print(f" ▶ 종합 점수: {overall}/10")
        if strengths:
            print(f" ▶ 강점: {', '.join(strengths)}")
        if weaknesses:
            print(f" ▶ 개선 필요: {', '.join(weaknesses)}")

        # 간단 액션 가이드
        if s == "경험":
            print(" ▸ 조언: 수치·역할·방법을 STAR(상황-과제-행동-결과) 구조로 일관되게 제시하세요.")
        elif s == "동기":
            print(" ▸ 조언: ‘왜 우리/왜 지금/왜 나’를 2~3문장 핵심 연결로 선명하게 설명하세요.")
        elif s == "논리":
            print(" ▸ 조언: 원인→대안→선택 근거→결과의 인과 뼈대를 먼저 말하고 사례를 덧붙이세요.")

        # 최종평 LLM 요약용 한 줄
        summary_for_llm_lines.append(
            f"- {s}: 종합 {overall}/10 | 평균(구체성 {avg['구체성']:.2f}, 일관성 {avg['일관성']:.2f}, 적합성 {avg['적합성']:.2f}, 논리성 {avg['논리성']:.2f})"
        )

    # 인터뷰 전체 종합 평가
    if total_scores:
        avg_score = round(sum(total_scores) / len(total_scores))
        print("\n [인터뷰 전체 종합 평가]")
        print("=" * 60)
        print(f" 평균 점수: {avg_score}/10\n")

        # 기존 final_eval_chain(LLM)을 그대로 활용해 한 단락 총평 생성
        final_feedback = final_eval_chain.invoke({
            "avg_score": avg_score,
            "summary_for_llm": "\n".join(summary_for_llm_lines)
        })
        print(f" {final_feedback}")
    else:
        print("\n 종합 점수를 계산할 수 있는 평가 데이터가 부족합니다.")

    print("=" * 60)
    print(" 인터뷰가 종료되었습니다. 수고하셨습니다!\n")

    state["next_step"] = "end"
    return state