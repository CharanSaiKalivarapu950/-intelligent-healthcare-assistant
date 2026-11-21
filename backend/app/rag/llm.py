import requests
from backend.app.core.config import settings

def generate_grounded_answer(question: str, contexts: list[dict]):
    if not settings.llm_api_key or not settings.llm_model:
        return None
    context_text = "\n\n".join(
        f"[Source {i+1}: {c['source']}, page {c.get('page', '?')}]\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    system = (
        "You are an educational medical knowledge assistant. You are not a clinician and "
        "must not diagnose, recommend treatment, or provide emergency triage. Answer only "
        "from the provided context. If context is insufficient, say so. Cite sources using "
        "[Source 1] labels. Avoid overclaiming and include a brief educational disclaimer."
    )
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}
    response = requests.post(
        settings.llm_base_url.rstrip("/") + "/chat/completions",
        headers=headers, json=payload, timeout=90
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
