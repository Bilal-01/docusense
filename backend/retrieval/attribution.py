import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a document assistant. You will be given context chunks, each labeled with a chunk ID.\n"
    "Answer the user's question using ONLY the provided chunks.\n"
    "After every sentence in your answer, add the chunk ID(s) you used in square brackets.\n"
    "Format: sentence text [chunk_id_1] or sentence text [chunk_id_1, chunk_id_2]\n"
    "Never answer from your own knowledge. Only use the provided chunks."
)


def build_attribution_prompt(
    chunks: List[Dict[str, Any]], question: str
) -> Dict[str, str]:
    ctx_lines = [
        f'[{c["chunk_id"]}]: "{c["text"].replace(chr(34), chr(92) + chr(34))}"'
        for c in chunks
    ]
    user_msg = "Context:\n" + "\n".join(ctx_lines) + "\n\nQuestion: " + question
    return {"system": SYSTEM_PROMPT, "user": user_msg}


def parse_attributed_response(text: str) -> List[Dict[str, Any]]:
    """
    Parse an LLM response that contains inline chunk citations.

    Expected format per sentence: "Sentence text [chunk_id_1, chunk_id_2]"

    Trailing sentences after the last citation are attributed to the last
    seen chunk IDs rather than left unattributed.
    """
    results: List[Dict[str, Any]] = []
    pattern = re.compile(r"(.*?)\s*\[([^\]]+)\]", re.S)
    last_ids: List[str] = []
    pos = 0

    for m in pattern.finditer(text):
        pre = m.group(1).strip()
        ids = [s.strip() for s in m.group(2).split(",") if s.strip()]

        for sentence in re.split(r"(?<=[.?!])\s+", pre):
            sentence = sentence.strip()
            if sentence:
                results.append({"sentence": sentence, "chunk_ids": ids})

        last_ids = ids
        pos = m.end()

    # Trailing text after the final citation — attribute to last seen chunk IDs
    for sentence in re.split(r"(?<=[.?!])\s+", text[pos:].strip()):
        sentence = sentence.strip()
        if sentence:
            results.append({"sentence": sentence, "chunk_ids": last_ids})

    return results


def compute_ragas_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Evaluate faithfulness and answer relevancy using RAGAS with Gemini as judge.
    Returns (faithfulness, answer_relevancy). Both are None if evaluation fails.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash"),
            google_api_key=os.environ["GEMINI_API_KEY"],
        )

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm)
        return float(result["faithfulness"][0]), float(result["answer_relevancy"][0])

    except Exception as e:
        _LOG.warning(f"RAGAS evaluation failed: {e}")
        return None, None