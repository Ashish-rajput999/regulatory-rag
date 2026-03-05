import os
from dotenv import load_dotenv

load_dotenv()
from src.llm import call_groq
from src.config import GROQ_FAST_MODEL

response = call_groq("Say hi in one sentence", GROQ_FAST_MODEL)
print("✅ Groq is working:", response)
