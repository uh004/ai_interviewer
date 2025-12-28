import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def get_llm(model: str | None = None):
    load_dotenv()  # .env 읽기
    model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
    return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))