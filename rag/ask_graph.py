"""LangGraph: retrieve → format → generate.

State threads through three pure functions; each writes one new field. The
DataFrame lives in state so `format` can resolve hit keys back to full rows.
"""
from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .config import CONFIG
from .index import Index
from .llm import LLM


class AskState(TypedDict):
    query: str
    where: dict | None
    k: int
    df: pd.DataFrame
    index: Index
    llm: LLM
    hits: list[dict[str, Any]]
    prompt: str
    answer: str


def _retrieve(state: AskState) -> dict:
    return {"hits": state["index"].search(
        state["query"], k=state["k"], where=state["where"]
    )}


def _format_one(row: pd.Series) -> str:
    truncate = CONFIG["ask"]["description_truncate"]
    desc = str(row.get("description", ""))[:truncate]
    return (
        f"Title: {row.get('title')}\n"
        f"Company: {row.get('employer_name')}\n"
        f"Location: {row.get('city')}, {row.get('country')}\n"
        f"Posted: {row.get('posted_at')}\n"
        f"Description: {desc}"
    )


def _format(state: AskState) -> dict:
    df = state["df"]
    rows = [df[df["key"] == hit["key"]].iloc[0] for hit in state["hits"]]
    context = "\n\n---\n\n".join(_format_one(r) for r in rows)
    return {"prompt": f"Context:\n\n{context}\n\nQuestion: {state['query']}"}


def _generate(state: AskState) -> dict:
    system_prompt = CONFIG["prompts"]["ask_system"]
    return {"answer": state["llm"].chat(system_prompt, state["prompt"])}


def build_graph():
    g = StateGraph(AskState)
    g.add_node("retrieve", _retrieve)
    g.add_node("format", _format)
    g.add_node("generate", _generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "format")
    g.add_edge("format", "generate")
    g.add_edge("generate", END)
    return g.compile()


graph = build_graph()