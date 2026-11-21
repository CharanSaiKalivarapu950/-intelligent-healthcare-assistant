from backend.app.rag.documents import load_pdf_chunks
from backend.app.rag.embeddings import Embeddings
from backend.app.rag.vector_store import VectorStore
from backend.app.rag.llm import generate_grounded_answer


class RAGService:
    def __init__(self):
        self.embeddings = Embeddings()
        self.store = VectorStore()
        self.store.load()

    def ingest_directory(self):
        records = load_pdf_chunks()

        if not records:
            return {
                "ingested_chunks": 0,
                "documents": [],
                "message": "No extractable PDF text found in medical_documents/raw."
            }

        texts = [record["text"] for record in records]

        vectors = self.embeddings.encode(texts)

        self.store.build(vectors, records)
        self.store.save()

        documents = sorted(set(record["source"] for record in records))

        return {
            "ingested_chunks": len(records),
            "documents": documents,
            "message": "Documents successfully ingested and indexed."
        }

    def answer(self, question, top_k=4):

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if self.store.index is None:
            loaded = self.store.load()

            if not loaded:
                raise RuntimeError(
                    "No documents indexed. Add PDFs to medical_documents/raw "
                    "and ingest first."
                )

        query_vector = self.embeddings.encode([question])[0]

        retrieved = self.store.search(query_vector, top_k)

        if not retrieved:
            return {
                "answer": "I couldn't find relevant information in the uploaded medical documents.",
                "sources": [],
                "disclaimer": "Educational information only; not medical advice or a diagnosis."
            }

        answer = generate_grounded_answer(question, retrieved)

        if answer is None:
            answer = (
                "The LLM is not configured. Below are the relevant passages "
                "retrieved from your medical documents."
            )

        sources = []

        for item in retrieved:
            sources.append({
                "source": item["source"],
                "page": item.get("page"),
                "chunk": item.get("chunk"),
                "similarity": round(item["score"], 4),
                "excerpt": item["text"]
            })

        return {
            "answer": answer,
            "sources": sources,
            "disclaimer": (
                "Educational information only; not medical advice or a diagnosis."
            )
        }


rag_service = RAGService()