# Architecture

Streamlit frontend -> FastAPI backend.

Image workflow:
upload -> validation -> RGB conversion -> resize/normalize -> ResNet18 -> probabilities -> Grad-CAM overlay.

RAG workflow:
PDF extraction -> page-aware chunks -> Sentence Transformer embeddings -> normalized vectors -> FAISS inner-product search -> context-constrained LLM generation -> citations.

Index records store source filename, page number, chunk number, and text. FAISS index and records JSON are persisted separately.
