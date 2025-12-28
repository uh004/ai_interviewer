from typing import TypedDict, List, Dict, Literal, Any
from pydantic import BaseModel, Field

class State(TypedDict, total=False):
    resume_text: str
    resume_summary: str
    resume_keywords: List[str]
    question_strategy: Dict[str, Dict[str, str]]

    current_question: str
    current_answer: str
    current_strategy: str

    conversation: List[Dict[str, str]]
    evaluation: Dict[str, Dict[str, Any]]
    next_step: str
    deep_counts: Dict[str, int]

    final_report: str

# ---- Pydantic schema들 ----
class ResumeAnalysis(BaseModel):
    summary: str = Field(..., description="이력서 주요 내용을 3~5문장으로 요약한 문장")
    keywords: List[str] = Field(..., description="이력서의 핵심 역량 및 키워드 목록")

class QSItem(BaseModel):
    direction: str = Field(..., description="질문 방향")
    examples: List[str] = Field(..., description="예시 질문 목록(2~3개)")

class QSOutput(BaseModel):
    experience: QSItem
    motivation: QSItem
    logic: QSItem

class QSMultiOutput(BaseModel):
    potential: QSOutput
    organization: QSOutput
    job: QSOutput

Binary = Literal[0, 1]

class BinCriterion(BaseModel):
    score: Binary = Field(..., description="0=부족, 1=우수")
    rationale: str = Field(..., description="간단 근거 1~2문장")

class FourCriteriaEval(BaseModel):
    specificity: BinCriterion
    consistency: BinCriterion
    fit: BinCriterion
    logic: BinCriterion

class DeepQuestion(BaseModel):
    question: str = Field(..., description="지원자의 사고력, 문제해결력, 기술적 깊이를 검증할 수 있는 심화 질문")