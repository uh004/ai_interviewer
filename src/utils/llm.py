import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def get_llm():
    load_dotenv()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model)