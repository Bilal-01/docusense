import re
from typing import List, Dict, Any, Optional


SYSTEM_PROMPT = (
    "You are a document assistant. You will be given context chunks, each labeled with a chunk ID.\n"
    "Answer the user's question using ONLY the provided chunks.\n"
    "After every sentence in your answer, add the chunk ID(s) you used in square brackets.\n"
    "Format: [chunk_id_1] or [chunk_id_1, chunk_id_2]\n"
    "If a sentence uses no chunk, do not include a bracket.\n"
    "Never answer from your own knowledge. Only use the provided chunks."
)


def build_attribution_prompt(chunks: List[Dict[str, Any]], question: str) -> Dict[str, str]:
    # Format context lines as: [chunk_id]: "text"
    ctx_lines = []
    for c in chunks:
        cid = c.get("chunk_id") or c.get("id") or c.get("chunkId")
        text = c.get("text") or c.get("document") or c.get("content") or ""
        safe_text = text.replace('"', '\\"')
        ctx_lines.append(f'[{cid}]: "{safe_text}"')

    user_msg = "Context:\n" + "\n".join(ctx_lines) + "\n\nQuestion: " + question
    return {"system": SYSTEM_PROMPT, "user": user_msg}


def parse_attributed_response(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    pattern = re.compile(r"(.*?)(\s*\[([^\]]+)\])", re.S)
    pos = 0
    for m in pattern.finditer(text):
        pre = m.group(1).strip()
        ids_raw = m.group(3).strip()
        ids = [s.strip() for s in ids_raw.split(",") if s.strip()]
        sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', pre) if s.strip()]
        for s in sents:
            results.append({"sentence": s, "chunk_ids": ids})
        pos = m.end()

    tail = text[pos:].strip()
    if tail:
        sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', tail) if s.strip()]
        for s in sents:
            results.append({"sentence": s, "chunk_ids": []})

    return results


def compute_ragas_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> (Optional[float], Optional[float]):
    """
    Compute faithfulness and answer relevancy using Ragas if available.
    Returns a tuple: (faithfulness, relevancy) where each may be None.
    """

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
    except Exception:
        return None, None

    try:
        data = {"question": [question], "answer": [answer], "contexts": [contexts]}
        if ground_truth is not None:
            data["ground_truth"] = [ground_truth]
        dataset = Dataset.from_dict(data)

        # attempt to use a local Ollama LLM if available
        llm = None
        try:
            from langchain_ollama import OllamaLLM
            import subprocess

            model_name = None
            try:
                out = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL, text=True)
                for ln in out.splitlines()[1:]:
                    ln = ln.strip()
                    if not ln:
                        continue
                    model_name = ln.split()[0]
                    break
            except Exception:
                model_name = None

            if model_name:
                try:
                    llm = OllamaLLM(model=model_name)
                except Exception:
                    llm = None
        except Exception:
            llm = None

        # shim when no LLM available
        if llm is None:
            class _LocalLLMShim:
                def generate(self, prompts, **kwargs):
                    from types import SimpleNamespace
                    gens = []
                    for p in prompts:
                        gens.append([SimpleNamespace(text=str(p))])
                    return SimpleNamespace(generations=gens)

            llm = _LocalLLMShim()

        try:
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm)
            faith = None
            rel = None
            try:
                faith = float(result["faithfulness"][0])
            except Exception:
                faith = None
            try:
                rel = float(result["answer_relevancy"][0])
            except Exception:
                rel = None
            return faith, rel

        except Exception:
            # fallbacks
            # 1) try prompt-based scoring with local llm (if it's not the shim)
            try:
                if not isinstance(llm, type(_LocalLLMShim())):
                    def _try_local_llm_scores(llm_obj):
                        import re
                        rel_prompt = (
                            "Rate how relevant the answer is to the question on a scale from 0 to 1"
                            " (0 = not relevant, 1 = fully relevant). Provide only a number.\n"
                            f"Question: {question}\nAnswer: {answer}"
                        )
                        faith_prompt = (
                            "For each sentence in the answer, return 1 if that sentence is fully supported by the provided contexts,"
                            " otherwise return 0. Provide a comma-separated list of 1s and 0s with no extra text.\n"
                            f"Contexts:\n{chr(10).join(contexts)}\nAnswer: {answer}"
                        )
                        rel_res = None
                        faith_res = None
                        try:
                            gen_r = llm_obj.generate([rel_prompt])
                            rel_text = str(gen_r.generations[0][0].text)
                            m = re.search(r"0(?:\.\d+)?|1(?:\.0+)?|0?\.\d+", rel_text)
                            if m:
                                rel_res = float(m.group(0))
                        except Exception:
                            rel_res = None
                        try:
                            gen_f = llm_obj.generate([faith_prompt])
                            faith_text = str(gen_f.generations[0][0].text)
                            nums = re.findall(r"[01]", faith_text)
                            if nums:
                                nums = [int(n) for n in nums]
                                faith_res = float(sum(nums)) / float(len(nums))
                        except Exception:
                            faith_res = None
                        return faith_res, rel_res

                    faith_llm, rel_llm = _try_local_llm_scores(llm)
                    if faith_llm is not None or rel_llm is not None:
                        return faith_llm, rel_llm
            except Exception:
                pass

            # 2) try embeddings-based similarity via ollama embed
            try:
                import subprocess
                import ollama
                model_name = None
                try:
                    out = subprocess.check_output(["ollama", "list"], stderr=subprocess.DEVNULL, text=True)
                    for ln in out.splitlines()[1:]:
                        ln = ln.strip()
                        if not ln:
                            continue
                        nm = ln.split()[0]
                        if "embed" in nm.lower() or "nomic" in nm.lower():
                            model_name = nm
                            break
                except Exception:
                    model_name = None

                if model_name:
                    sents = [s.strip() for s in re.split(r'(?<=[\.\!?])\s+', answer) if s.strip()]
                    inputs = [question, answer] + contexts + sents
                    resp = ollama.embed(model_name, inputs)
                    embs = resp.embeddings

                    def cos(a, b):
                        da = sum(x * x for x in a) ** 0.5
                        db = sum(x * x for x in b) ** 0.5
                        if da == 0 or db == 0:
                            return 0.0
                        return sum(x * y for x, y in zip(a, b)) / (da * db)

                    q_emb = embs[0]
                    a_emb = embs[1]
                    relevancy = cos(q_emb, a_emb)

                    ctx_embs = embs[2:2 + len(contexts)]
                    sent_embs = embs[2 + len(contexts):]
                    if not sent_embs:
                        faith = None
                    else:
                        match = 0
                        for se in sent_embs:
                            sims = [cos(se, ce) for ce in ctx_embs] if ctx_embs else [0.0]
                            if max(sims) > 0.25:
                                match += 1
                        faith = float(match) / float(len(sent_embs))
                    return faith, relevancy
            except Exception:
                pass

            # 3) final heuristic fallback
            def _heuristic_scores(question, answer, contexts, ground_truth=None):
                def toks(s: str):
                    return set([w for w in re.findall(r"\w+", s.lower()) if len(w) > 2])

                qt = toks(question)
                at = toks(answer)
                relevancy = None
                if qt and at:
                    inter = qt & at
                    union = qt | at
                    relevancy = float(len(inter)) / float(len(union)) if union else None

                ctx_tokens = set()
                for c in contexts:
                    ctx_tokens |= toks(c)

                sents = [s.strip() for s in re.split(r'(?<=[\.\!?])\s+', answer) if s.strip()]
                faith = None
                if sents:
                    match = 0
                    for s in sents:
                        st = toks(s)
                        if st & ctx_tokens:
                            match += 1
                    faith = float(match) / float(len(sents))
                return faith, relevancy

            return _heuristic_scores(question, answer, contexts, ground_truth)

    except Exception:
        return None, None
