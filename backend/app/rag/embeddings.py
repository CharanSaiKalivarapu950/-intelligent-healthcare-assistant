from sentence_transformers import SentenceTransformer
from backend.app.core.config import settings

class Embeddings:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = SentenceTransformer(settings.embedding_model)
        return self.model

    def encode(self, texts):
        return self.load().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=(len(texts) > 20)
        )
