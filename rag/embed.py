"""Sentence-transformer embeddings.

`normalize_embeddings=True` makes vectors unit-length, so cosine similarity
collapses to a dot product downstream.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .config import CONFIG
from .embed_remote import RemoteEmbedder

# local selector so SentenceTransformer doesn't choke on device="ollama"
def get_embedder():
    """Return the embedder selected by [embed].device."""
    device = CONFIG["embed"]["device"]
    if device == "ollama":
        from .embed_remote import RemoteEmbedder
        return RemoteEmbedder()
    return Embedder()  # cpu / gpu -> local sentence-transformers

class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(
            model_name or CONFIG["embed"]["model"],
            device=CONFIG["embed"]["device"],
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
