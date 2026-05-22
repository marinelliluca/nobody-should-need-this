"""Notebook entrypoint for the match graph."""
from __future__ import annotations

import pandas as pd

from .code_cv import CVProfile
from .config import CONFIG
from .index import build_index
from .llm import LLM
from .match_graph import _prepare_dump_dir, _previous_run_dir, get_graph


def match_cv(profile: CVProfile, cv_text: str, df: pd.DataFrame,
             top_n_retrieve: int | None = None,
             dump: bool = True,
             prescreen_only: bool = False,
             previous_run_dir: str | Path | None = None) -> pd.DataFrame:
    """Score `df` against a CV, return rows sorted by `candidate_score`.

    Both inputs are needed: the recruiter pass reads the raw `cv_text` so
    it can form an independent view, while the candidate pass reads the
    candidate-coded `profile` (themes, must-haves, disqualifiers).

    With `dump=True` (default), per-hit verdicts plus a snapshot of the
    config, jobs slice, and CV profile land under `[match.dump].root` in
    a slug-named folder. Pass `dump=False` for ad-hoc runs you don't want
    on disk.

    With `prescreen_only=True`, the graph stops after the prescreen node
    and the returned frame contains the rows that survived prescreen,
    unsorted (no `candidate_score` / `recruiter_score` columns exist yet).
    Use `prescreen_only=True` exclusively with a dump_dir, since it's purpose
    is to run prescreen on a smaller model, and then load from cache in the next
    run with a bigger model for the rest of the passes.

    `previous_run_dir` is the resume-from-cache source: per-record JSONs in
    that folder are reused when their slugified-key filename matches, so a
    prior (e.g. prescreen-only) run can be resumed without re-querying the
    LLM. When None (the default), the value falls back to
    `[match.dump].previous_run_dir` in config, preserving the prior
    config-only behavior. A non-existent path is ignored (treated as no
    cache).
    """
    if top_n_retrieve is None:
        top_n_retrieve = CONFIG["match"]["top_n_retrieve"]
    index = build_index(df)
    dump_dir = _prepare_dump_dir(top_n_retrieve) if dump else None
    # Explicit arg wins; otherwise fall back to the config value. Resolved to
    # a validated Path (or None) here so the graph nodes just read state.
    resolved_prev_dir = _previous_run_dir(previous_run_dir)
    state = get_graph(prescreen_only).invoke({
        "profile": profile,
        "cv_text": cv_text,
        "df": df.set_index("key", drop=False), # index by "key" or evertything breaks downstream
        "index": index,
        "llm": LLM(),
        "top_n_retrieve": top_n_retrieve,
        "dump_dir": dump_dir,
        "previous_run_dir": resolved_prev_dir,
    })
    if dump_dir is not None:
        print(f"scores dumped to: {dump_dir}")

    if prescreen_only:
        # Graph stopped after prescreen; `_select` never ran, so there's no
        # `result` key. Return survivor rows in their original df order.
        survivor_keys = state["survivor_keys"]
        return df[df["key"].isin(survivor_keys)].reset_index(drop=True)

    return state["result"]
