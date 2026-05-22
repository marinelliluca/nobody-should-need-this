"""Thematic coding of a CV into an editable structured profile.

The output is the highest-leverage decision in the matching pipeline: every
downstream score is computed against these themes. Surfacing it as a plain
dict means the user can inspect and tweak it in the notebook before any
jobs get scored.
"""
from __future__ import annotations

from typing import TypedDict

from .config import CONFIG
from .llm import LLM


class CVProfile(TypedDict):
    role_themes: list[str]
    must_have_themes: list[str]
    nice_to_have_themes: list[str]
    disqualifiers: list[str]
    search_query: str


def code_cv(cv_text: str, llm: LLM | None = None) -> CVProfile:
    """Run thematic coding on a CV. Returns an editable profile dict."""
    llm = llm or LLM()
    system_prompt = CONFIG["prompts"]["code_cv_system"]
    return llm.chat_json(system_prompt, cv_text)  # type: ignore[return-value]
