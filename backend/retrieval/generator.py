import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    response = _CLIENT.chat.completions.create(
        model=_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content