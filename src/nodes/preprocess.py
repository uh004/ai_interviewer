import random
from langchain_core.prompts import ChatPromptTemplate
from src.utils.state import State, ResumeAnalysis, QSMultiOutput
from src.utils.llm import get_llm
from src.utils.io import extract_text_from_file

llm = get_llm()


def analyze_resume(state: State) -> State:
    resume_text = state["resume_text"]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 인사 당담자입니다. 다음 이력서 텍스트를 분석하여 핵심 요약과 주요 키워드를 도출하세요. "
         "결과는 JSON 형태로 반환하세요. summary(3~5문장), keywords(목록)"),
        ("human", "분석할 이력서 텍스트:\n---\n{resume_text}")
    ])
    chain = prompt | llm.with_structured_output(ResumeAnalysis)
    result: ResumeAnalysis = chain.invoke({"resume_text": resume_text})

    return {**state, "resume_summary": result.summary, "resume_keywords": result.keywords}


def generate_question_strategy(state: State) -> State:
    summary = state.get("resume_summary", "")
    keywords = state.get("resume_keywords", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 시니어 인사담당 면접관입니다.\n"
         "아래 이력서를 기반으로 **3명의 면접관(A/B/C)**에 대해 면접 질문 전략을 만듭니다.\n\n"

         "면접관 역할:\n"
         "A = 잠재력 평가 (도전, 문제 해결, 성장 가능성)\n"
         "B = 조직 적합도 평가 (협업, 소통, 조직문화)\n"
         "C = 직무 역량 평가 (기술/업무 수행능력, 성과)\n\n"

         "각 면접관은 아래 3개 항목에 대해 질문 생성:\n"
         "1) 경력 및 경험\n"
         "2) 동기 및 커뮤니케이션\n"
         "3) 논리적 사고\n\n"

         "각 항목에 대해 반드시 다음 정보를 포함:\n"
         "- direction: 평가 의도/목적 (1~2문장)\n"
         "- examples: 실제 면접 질문 2~3개 (구체적, 맥락 기반, 정중체)\n\n"

         "출력 형식은 JSON ONLY이며 다음 스키마를 따르세요.\n"
         "{{\n"
         "  \"potential\": {{ \"experience\": ..., \"motivation\": ..., \"logic\": ... }},\n"
         "  \"organization\": {{ \"experience\": ..., \"motivation\": ..., \"logic\": ... }},\n"
         "  \"job\": {{ \"experience\": ..., \"motivation\": ..., \"logic\": ... }}\n"
         "}}\n\n"

         "예시 질문 스타일 가이드:\n"
         "- '~했던 경험이 있나요?' '~어떤 기여를 했나요?' '~어떻게 해결했나요?' 형식\n"
         "- 숫자/성과/구체적 사례 포함\n"
         "- 협업, 난관 해결, 역할 분명히 질문\n"
         "- 모호한 질문 금지\n"
         "- JSON 외 텍스트 출력 금지"
        ),
        ("human",
         "이력서 요약:\n{summary}\n\n"
         "주요 키워드:\n{keywords}\n\n"
         "**반드시 JSON만 출력하세요.**")
    ])

    chain = prompt | llm.with_structured_output(QSMultiOutput)
    result: QSMultiOutput = chain.invoke({
        "summary": summary,
        "keywords": ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
    })

    strategy_dict = {
        "경험": {"A": result.potential.experience.examples[0],
               "B": result.organization.experience.examples[0],
               "C": result.job.experience.examples[0]},
        "동기": {"A": result.potential.motivation.examples[0],
               "B": result.organization.motivation.examples[0],
               "C": result.job.motivation.examples[0]},
        "논리": {"A": result.potential.logic.examples[0],
               "B": result.organization.logic.examples[0],
               "C": result.job.logic.examples[0]},
    }

    return {**state, "question_strategy": strategy_dict}


# ✅ 이름을 import되는 이름과 동일하게
def pre_processing_interview(file_path: str) -> State:
    resume_text = extract_text_from_file(file_path)

    # ✅ State로 통일
    state: State = {
        "resume_text": resume_text,
        "resume_summary": "",
        "resume_keywords": [],
        "question_strategy": {},

        "current_question": "",
        "current_answer": "",
        "current_strategy": "",

        "conversation": [],
        "evaluation": {},      # dict
        "next_step": "",
        "deep_counts": {},     # {분야:int}
        "final_report": "",
    }

    state = analyze_resume(state)
    state = generate_question_strategy(state)

    cat = "경험"
    iv = random.choice(["A", "B", "C"])
    selected_question = state["question_strategy"][cat][iv]

    return {**state, "current_question": selected_question, "current_strategy": cat}