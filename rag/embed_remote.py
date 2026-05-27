"""Embeddings via a remote Ollama server over HTTP (port 11434).

Drop-in replacement for the local sentence-transformers `Embedder`: same two
methods, same return shapes (unit-normalized vectors). `index.py` and the
graphs need no changes — `build_index(df, embedder=RemoteEmbedder())` and
`Index(..., embedder=RemoteEmbedder())` just work.

Ollama's /api/embed returns one vector per input string. We normalize
client-side so cosine space behaves identically to the local path, regardless
of whether the server/model normalized for us.
"""
from __future__ import annotations

import math

import requests

from .config import CONFIG, ollama_host


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class RemoteEmbedder:
    """Same interface as embed.Embedder, but computes vectors via Ollama's API over HTTP."""

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        batch_size: int | None = None,
        timeout: float| None = None,
    ):
        # Reuse the same model name and host helpers the rest of the app uses,
        # so OLLAMA_HOST / [embed].model still drive behaviour.
        self.model = model_name or CONFIG["embed"]["model"]
        self.host = (host or ollama_host()).rstrip("/")
        self.batch_size = batch_size or CONFIG["embed"]["batch_size"]
        self.timeout = timeout or CONFIG["embed"]["timeout"]
        self._session = requests.Session()

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        """POST one batch to /api/embed and return normalized vectors."""
        resp = self._session.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": inputs},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        # /api/embed returns {"embeddings": [[...], [...], ...]} aligned to input order.
        embeddings = resp.json()["embeddings"]
        return [_normalize(v) for v in embeddings]

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i+self.batch_size]
            try:
                out.extend(self._embed(chunk))
            except Exception:
                for t in chunk:                       # isolate the offender
                    try:
                        out.extend(self._embed([t]))
                    except Exception as e:
                        #print(e)
                        out.append(None)              # or skip
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]