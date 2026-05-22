"""Chroma-backed vector index over the `jobs` DataFrame.

Each row's `key` column becomes the Chroma document id. A small subset of
columns is stored as Chroma metadata to support `where`-filtering. Hits come
back with key, similarity, and metadata; joining back to the full row is the
caller's job — keeping the DataFrame off the Index keeps the object
lightweight and avoids accidental reference cycles.

Using `key` (a non-colliding hash column) rather than the DataFrame's
positional index means re-ordering or re-indexing the frame doesn't
invalidate the collection.

Set `CHROMA_DIR` in .env (or pass `persist_dir`) to keep the collection on
disk — `build_index` skips re-embedding rows whose key is already present, so
subsequent calls in a new session are near-instant. The persistent collection
accumulates across sessions; the `Index` returned by `build_index` is scoped
to the current df's keys via a Chroma `where` filter, so search results
never reference rows the caller doesn't have.
"""
from __future__ import annotations

from typing import Any

import chromadb
import pandas as pd

from .config import CONFIG, chroma_dir
from .embed import Embedder


def _row_to_text(row: pd.Series) -> str:
    """Concatenate the fields that carry retrieval signal."""
    parts = [row.get("title"), row.get("employer_name"),
             row.get("city"), row.get("description")]
    return "\n".join(str(p) for p in parts if p)


def _row_to_metadata(row: pd.Series) -> dict[str, Any]:
    """Chroma metadata must be flat scalars; coerce to str and skip nulls.

    `key` is always included (regardless of `filterable_cols`) because
    scoped search filters on it.
    """
    filterable = CONFIG["index"]["filterable_cols"]
    meta = {col: str(row[col]) for col in filterable
            if col in row and pd.notna(row[col])}
    meta["key"] = str(row["key"])
    return meta


class Index:
    """Lightweight handle on a Chroma collection.

    `scope_keys`, if set, restricts every search to those keys via a Chroma
    `where` filter on the `key` metadata field. The persistent collection
    can accumulate across sessions while each session only sees its own df.
    """

    def __init__(self, collection, embedder: Embedder,
                 scope_keys: set[str] | None = None):
        self.collection = collection
        self.embedder = embedder
        self.scope_keys = scope_keys

    def search(self, query: str, k: int = 5,
               where: dict | None = None) -> list[dict]:
        """Return top-k matches as dicts: {key (str), score, metadata}."""
        # Compose the caller's `where` with the session scope, if any.
        scope_where = (
            {"key": {"$in": list(self.scope_keys)}}
            if self.scope_keys is not None else None
        )
        if where and scope_where:
            effective_where = {"$and": [where, scope_where]}
        else:
            effective_where = where or scope_where

        results = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=k,
            where=effective_where or None,   # Chroma rejects empty dicts
        )
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        # Cosine distance is 1 - similarity; convert back for readability.
        return [
            {"key": id_, "score": 1.0 - dist, "metadata": meta}
            for id_, dist, meta in zip(ids, distances, metadatas)
        ]


def build_index(df: pd.DataFrame, persist_dir: str | None = None,
                embedder: Embedder | None = None) -> Index:
    """Build (or load) the Chroma collection for `df`.

    Rows whose `key` is already in the collection are skipped, so repeated
    calls only embed new rows. The persistent collection accumulates across
    sessions; the returned `Index` is scoped to the current df's keys, so
    search results never reference rows the caller doesn't have. With
    `persist_dir` set (or CHROMA_DIR in the environment), the collection
    survives across sessions.
    """
    embedder = embedder or Embedder()
    persist_dir = persist_dir or chroma_dir()
    client = (chromadb.PersistentClient(path=persist_dir) if persist_dir
              else chromadb.EphemeralClient())
    collection = client.get_or_create_collection(
        name=CONFIG["index"]["collection_name"],
        metadata={"hnsw:space": CONFIG["index"]["hnsw_space"]},
    )

    existing_keys = set(collection.get()["ids"])
    df_keys = set(df["key"].astype(str))

    new_rows = df.loc[~df["key"].astype(str).isin(existing_keys)]
    if not new_rows.empty:
        ids = [str(k) for k in new_rows["key"]]
        texts = [_row_to_text(r) for _, r in new_rows.iterrows()]
        collection.add(
            ids=ids,
            embeddings=embedder.embed_documents(texts),
            documents=texts,
            metadatas=[_row_to_metadata(r) for _, r in new_rows.iterrows()],
        )

    return Index(collection, embedder, scope_keys=df_keys)