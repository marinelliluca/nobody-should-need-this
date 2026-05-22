"""Ollama wrapper.

Two methods: `chat` for free-form text, `chat_json` for structured output.
JSON mode is the cheapest defense against parser breakage in scoring code.
"""
from __future__ import annotations

import json
import re
import json5  # permissive JSON parser; accepts trailing commas, etc.

from typing import Any

from ollama import Client

from .config import ollama_host, ollama_model

# Strip a trailing comma immediately before `}` or `]`, allowing whitespace
# in between. Used only as a last-ditch fallback after json5 also fails.
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

def _parse_loose_json(text: str) -> dict:
    """Parse JSON tolerantly. Three layers, cheapest first.
 
    1. Strict `json.loads` — if the model got it right, no rewriting.
    2. `json5.loads` — handles trailing commas, single quotes, unquoted
       keys, comments. Crucially still a real parser, so commas inside
       strings are safe.
    3. Regex strip of trailing commas + strict retry. Last resort; can
       in principle corrupt strings containing literal `,}`, hence last.
 
    Raises `ValueError` with the original strict-parse error if all three
    fail, so callers get a meaningful message.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as strict_err:
        try:
            return json5.loads(text)
        except Exception:
            try:
                return json.loads(_TRAILING_COMMA.sub(r"\1", text))
            except json.JSONDecodeError:
                raise ValueError(
                    f"could not parse LLM JSON output: {strict_err.msg} "
                    f"at line {strict_err.lineno} col {strict_err.colno}"
                ) from strict_err


class LLM:
    def __init__(self, host: str | None = None, model: str | None = None):
        self.client = Client(
            host=host or ollama_host(), 
            timeout=300, # fail after 5 minutes
        ) 
        self.model = model or ollama_model()

    def chat(self, system: str, user: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            think=False,  # stop thinking models from wasting compute
            options={"temperature": 0}, # reproducible queries, otherwise good luck debugging
        )
        return response["message"]["content"]

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            think=False,  # stop thinking models from wasting compute
            options={"temperature": 0},
        )
        return _parse_loose_json(response["message"]["content"])
