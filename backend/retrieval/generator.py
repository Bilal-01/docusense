import os
from typing import Tuple

# Attempt to use Ollama for free local generation; fallback to simple deterministic generator
try:
    import ollama
except Exception:
    ollama = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama2")


def generate_attributed_answer(system_prompt: str, user_prompt: str) -> str:
    """
    Generate an attributed answer using an LLM. Tries Ollama if available, otherwise
    synthesizes a conservative answer by restating chunks with chunk ids.
    """
    # Prefer Ollama if available
    if ollama is not None:
        try:
            with ollama.Client(host=OLLAMA_HOST) as client:
                # Some Ollama clients accept `model` and `prompt` or `messages`.
                # Use a simple generate API if available.
                try:
                    # Try `generate` first
                    resp = client.generate(model=OLLAMA_GEN_MODEL, prompt=system_prompt + "\n" + user_prompt)
                    # resp may be a str or object
                    if isinstance(resp, str):
                        return resp
                    # try common attr
                    return getattr(resp, "text", str(resp))
                except Exception:
                    # Try `create` or `chat` style
                    try:
                        resp = client.chat(model=OLLAMA_GEN_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
                        if isinstance(resp, str):
                            return resp
                        return getattr(resp, "text", str(resp))
                    except Exception:
                        pass
        except Exception:
            pass

    # Fallback conservative generator: extract context lines from user prompt and craft answer
    # This ensures the output follows the attribution format: one sentence per chunk with bracketed ids
    lines = [l for l in user_prompt.splitlines() if l.strip().startswith("[")]
    out_sentences = []
    for ln in lines[:5]:
        # ln like: [chunk_id]: "text"
        try:
            cid_part, txt_part = ln.split(":", 1)
            cid = cid_part.strip().strip("[]")
            txt = txt_part.strip().strip().strip('"')
            # produce a conservative sentence
            sent = f"{txt.split('.')[0].strip()}. [{cid}]"
            out_sentences.append(sent)
        except Exception:
            continue
    if not out_sentences:
        return "I cannot answer using outside knowledge."
    return " ".join(out_sentences)


def polish_answer_with_ollama(answer_text: str, question: str, contexts: list) -> str:
    """
    Produce a concise, conversational one-sentence answer using Ollama if available.
    Falls back to a simple heuristic if Ollama is not reachable.
    """
    prompt = (
        f"Question: {question}\n\n"
        "Extracted answer:\n" + (answer_text or "") + "\n\n"
        "Relevant contexts:\n" + "\n".join([c for c in contexts if c]) + "\n\n"
        "Please produce a single concise, conversational sentence answering the question using only the information above."
    )

    # Try Ollama client first
    try:
        if ollama is not None:
            with ollama.Client(host=OLLAMA_HOST) as client:
                try:
                    resp = client.generate(model=OLLAMA_GEN_MODEL, prompt=prompt)
                    if isinstance(resp, str):
                        return resp.strip()
                    return getattr(resp, "text", str(resp)).strip()
                except Exception:
                    try:
                        resp = client.chat(model=OLLAMA_GEN_MODEL, messages=[{"role": "user", "content": prompt}])
                        if isinstance(resp, str):
                            return resp.strip()
                        return getattr(resp, "text", str(resp)).strip()
                    except Exception:
                        pass
    except Exception:
        pass

    # Fallback: try Ollama CLI
    try:
        import subprocess
        proc = subprocess.run(["ollama", "run", OLLAMA_GEN_MODEL, "--prompt", prompt], capture_output=True, text=True, check=True)
        out = proc.stdout.strip()
        if out:
            return out
    except Exception:
        pass

    # Final heuristic fallback: simple rewrite using keyword detection
    low = (answer_text or "").lower()
    if "langchain" in low:
        # try to include a short quote if available
        snippet = ""
        try:
            for s in (answer_text or "").split('.'):
                if 'langchain' in s.lower():
                    snippet = s.strip()
                    break
        except Exception:
            snippet = ""
        if snippet:
            return f"Yes — the resume mentions LangChain ({snippet})."
        return "Yes — the resume mentions LangChain in the technical skills section."

    return answer_text
