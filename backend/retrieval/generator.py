import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Call Gemini to generate a response given a system and user prompt."""
    model = genai.GenerativeModel(
        model_name=_LLM_MODEL,
        system_instruction=system_prompt,
    )
    return model.generate_content(user_prompt).text