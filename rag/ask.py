"""Notebook entrypoint for the ask graph."""
from __future__ import annotations

import pandas as pd

from .ask_graph import graph
from .config import CONFIG
from .index import build_index
from .llm import LLM


def _build_where(city: str | None, country: str | None) -> dict | None:
    """Translate keyword filters into a Chroma where dict."""
    clauses = {k: v for k, v in {"city": city, "country": country}.items() if v}
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses
    return {"$and": [{k: v} for k, v in clauses.items()]}


def ask(query: str, df: pd.DataFrame, k: int | None = None,
        city: str | None = None, country: str | None = None,
        return_sources: bool = True):
    """Answer `query` using jobs from `df`.

    With `return_sources=True`, returns `(answer, sources_df)` where the
    sources DataFrame is the subset of rows the model actually saw.
    """
    if k is None:
        k = CONFIG["ask"]["top_k"]
    index = build_index(df)
    state = graph.invoke({
        "query": query,
        "where": _build_where(city, country),
        "k": k,
        "df": df,
        "index": index,
        "llm": LLM(),
    })

    if not return_sources:
        return state["answer"]

    source_keys = [hit["key"] for hit in state["hits"]]
    return state["answer"], df[df["key"].isin(source_keys)]