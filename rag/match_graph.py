"""LangGraph: retrieve → recruiter_score → candidate_score → select.

Two scoring passes:

1. `recruiter_score` — a recruiter persona reads the *raw CV* (not the
   candidate-coded profile) and fills a fixed rubric (five themes, 0–10
   each). Reading the raw CV matters: the candidate-coded profile is
   already filtered through the candidate's own framing, and the recruiter
   is supposed to form an independent view from the company's side. The
   score is a weighted sum normalized to 0–100.

2. `candidate_score` — the existing CV-vs-posting extraction, driven by
   the candidate-coded profile. Only the top `keep_pct` of the recruiter
   pass reach this node.

The rubric is a fixed schema (not free-form themes like the candidate side)
because the recruiter view is meant to generalize across postings. A stable
schema is what makes the arithmetic clean and the dumps comparable.

Both scoring passes share `_extract_concurrent`: same threadpool, same
streaming dump, different prompt and reducer.

Rows are addressed throughout by the `key` column (assume non-colliding)
rather than DataFrame positional index, so reordering the frame between
runs doesn't invalidate cached embeddings or dumps.

Resume-from-cache: if `[match.dump].previous_run_dir` is set, per-record
JSONs in that folder are reused when their slugified-key filename matches.
The current run still gets its own (timestamped) dump folder, and the previous
folder is read-only. The user is responsible for pointing at a relevant folder 
since there's no automatic compatibility check.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from tqdm.auto import tqdm

from .code_cv import CVProfile
from .config import CONFIG, config_path, ollama_model
from .index import Index
from .llm import LLM

from functools import lru_cache

# --- scoring -----------------------------------------------------------------

def _candidate_score(verdict: dict,
                     weights: dict = CONFIG["match"]["candidate_scoring"]) -> int:
    """Pure scoring function for the candidate pass. Range is unbounded.

    Weight keys match the verdict's list-column names exactly
    (must_haves_met, must_haves_missing, themes_matched,
    disqualifiers_triggered). The config block [match.candidate_scoring]
    must use these same key names.
    """
    return sum(w * len(verdict.get(key, ())) for key, w in weights.items())


def _recruiter_score(rubric: dict, 
                     weights: dict = CONFIG["match"]["recruiter_scoring"]) -> int:
    """Weighted average of the rubric, normalized to 0–100.

    Each rubric theme is rated 0–10 by the LLM. We multiply by per-theme
    weights from config (so technical_match can outweigh, say, soft_skills),
    sum, then divide by the max achievable (10 × sum-of-weights) and scale.
    Hardcoded arithmetic — the LLM only fills in the integers.
    """
    weighted = sum(weights[theme] * rubric.get(theme, 0) for theme in weights)
    max_weighted = 10 * sum(weights.values())
    return round(100 * weighted / max_weighted) if max_weighted else 0


# --- dump bookkeeping --------------------------------------------------------

def _slugify(s: str) -> str:
    """Filesystem-safe slug: lowercase, alnum + dash, collapsed."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s or "x"


def _run_slug(top_n: int) -> str:
    """Encode result-affecting config + a timestamp.

    Two runs with identical config produce identical leading slugs; the
    timestamp suffix keeps them as separate folders. Both prompt hashes
    appear so a tweak to either system prompt is visible in the path.
    """
    cw = CONFIG["match"]["candidate_scoring"]
    keep = CONFIG["match"]["recruiter_keep_pct"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{_slugify(ollama_model())}"
        f"__n{top_n}__keep{keep}"
        f"__{timestamp}"
    )


def _prepare_dump_dir(top_n: int) -> Path | None:
    """Create the per-run dump directory and copy in static inputs.

    Returns None if dumping is disabled (empty `[match.dump].root`).
    """
    root_str = CONFIG["match"].get("dump", {}).get("root", "")
    if not root_str:
        return None
    run_dir = Path(root_str) / _run_slug(top_n)
    (run_dir / "prescreen").mkdir(parents=True, exist_ok=True)
    (run_dir / "recruiter").mkdir(parents=True, exist_ok=True)
    (run_dir / "candidate").mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(config_path(), run_dir / "config.toml")
    except (FileNotFoundError, OSError):
        pass
    return run_dir


def _previous_run_dir(override: str | Path | None = None) -> Path | None:
    """Resolve the resume-from-cache directory, or None.

    Precedence: an explicit `override` (e.g. chosen in the UI dropdown or
    passed to `match_cv`) wins; otherwise fall back to
    `[match.dump].previous_run_dir` in config. Empty string and missing key
    both mean "no cache". A non-existent path is treated as no cache too —
    we don't want a typo (or a stale UI selection) to silently fail loud
    later in the pipeline.
    """
    raw = override if override else CONFIG["match"].get("dump", {}).get(
        "previous_run_dir", ""
    )
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        print(f"warning: previous_run_dir {path} does not exist; ignoring")
        return None
    return path


def recent_dump_dirs(n: int = 5) -> list[Path]:
    """Return the `n` most recently modified dump folders under the dump root.

    Reads `[match.dump].root` from config, lists its immediate
    subdirectories, and returns the newest `n` by mtime (most recent first).
    Returns `[]` if the root is unset or doesn't exist. Used to populate the
    UI's "resume from previous run" dropdown; the backend logic for what a
    valid run folder looks like lives here, not in the interface layer.
    """
    root_str = CONFIG["match"].get("dump", {}).get("root", "")
    if not root_str:
        return []
    root = Path(root_str)
    if not root.exists():
        return []
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[:n]


def _record_path(subdir: Path, key: str) -> Path:
    """Canonical on-disk path for a record. 
    
    The only place slugification happens is inside _record_path. 
    Everywhere else passes raw keys around. 
    """
    return subdir / f"{_slugify(str(key))}.json"

def _nan2none(v):
    """Convert NaNs to None before dumping JSONs"""
    return None if isinstance(v, float) and pd.isna(v) else v

def _dump_record(subdir: Path | None, record: dict, row: pd.Series) -> None:
    """Atomic per-record JSON dump. No-op if `subdir` is None."""
    if subdir is None:
        return
    payload = {
        **record,
        "posting_preview": {
            "title": _nan2none(row.get("title")),
            "employer_name": _nan2none(row.get("employer_name")),
            "city": _nan2none(row.get("city")),
            "country": _nan2none(row.get("country")),
        },
    }
    path = _record_path(subdir, record["key"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _load_cached(cache_subdir: Path | None, key: str) -> dict | None:
    """Return a previously-dumped record for `key`, or None.

    Strips the dump-only `posting_preview` field so the returned dict has
    the same shape an extractor would have produced. Errored records are
    treated as cache misses so transient failures get retried.
    """
    if cache_subdir is None:
        return None
    path = _record_path(cache_subdir, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("error"):
        return None
    payload.pop("posting_preview", None)
    return payload


# --- the three extractors ------------------------------------------------------

RECRUITER_THEMES = [
    "technical_match",    # stack/tools proximity to what the role needs
    "seniority_match",    # responsibility level vs. what's asked
    "transferability",    # how well adjacent experience (incl. different
                          # vertical) carries over to this context
    "trajectory",         # promotions, growing scope, learning velocity
    "soft_skills",        # communication, collaboration, stakeholder evidence
]

def _empty_decision() -> dict:
    return {"decision": ""}

def _empty_rubric() -> dict:
    return {theme: 0 for theme in RECRUITER_THEMES}


def _empty_candidate_verdict() -> dict:
    return {"themes_matched": [], "must_haves_met": [],
            "must_haves_missing": [], "disqualifiers_triggered": []}


def _posting_text(row: pd.Series) -> str:
    return (f"Title: {row.get('title')}\n"
            f"Company: {row.get('employer_name')}\n"
            f"Location: {row.get('city')}, {row.get('country')}\n"
            f"Description: {row.get('description')}")

def _prescreen(llm: LLM, cv_text: str, hit_key: str, row: pd.Series) -> dict:
    """Apply prescreen criteria (defined in config [prompts])
    and receieve a binary reject/pass decision.
    """

    user_msg = (
        "Your knowledge cutoff date is outdated, the current month is actually "
        f"{datetime.now().strftime('%B %Y')}.\n\n"
        f"CANDIDATE CV:\n{cv_text}\n\n"
        f"JOB POSTING:\n{_posting_text(row)}"
    )
    try:
        decision = llm.chat_json(
            CONFIG["prompts"]["prescreen_system"], user_msg
        )
        if not isinstance(decision["decision"], str):
            raise Exception("Malformed prescreen: 'decision' is not a str.")
        if decision["decision"].lower() not in ["pass", "reject"]:
            raise Exception("Malformed prescreen: 'decision' is not 'pass' or 'reject'.")
        error = None
    except Exception as e:
        decision, error = _empty_decision(), f"prescreen failed at key '{hit_key}' with error: {e}"

    return {"key": hit_key, **decision, "error": error}


def _extract_recruiter(llm: LLM, cv_text: str, hit_key: str,
                       row: pd.Series) -> dict:
    """Recruiter persona scores a posting against the raw CV.

    Takes the CV text directly (not the candidate-coded profile) so the
    recruiter forms their own view of the candidate rather than inheriting
    the candidate's self-framing of must-haves and disqualifiers.
    """
    user_msg = (
        "Your knowledge cutoff date is outdated, the current month is actually "
        f"{datetime.now().strftime('%B %Y')}.\n\n"
        f"CANDIDATE CV:\n{cv_text}\n\n"
        f"JOB POSTING:\n{_posting_text(row)}"
    )
    try:
        rubric = llm.chat_json(
            CONFIG["prompts"]["match_recruiter_system"], user_msg
        )
        # Defensive: coerce missing/non-int values into the 0–10 range.
        for theme in RECRUITER_THEMES:
            v = rubric.get(theme, 0)
            rubric[theme] = max(0, min(10, int(v))) if isinstance(v, (int, float)) else 0
        rubric_clean = {t: rubric[t] for t in RECRUITER_THEMES}
        notes = rubric.get("recruiter_notes", "")  
        error = None
    except Exception as e:
        rubric_clean, notes, error = _empty_rubric(), "", f"recruiter failed at key '{hit_key}' with error: {e}"

    return {"key": hit_key, **rubric_clean, "recruiter_notes": notes,
            "recruiter_score": _recruiter_score(rubric_clean), "error": error}


def _extract_candidate(llm: LLM, profile: CVProfile, hit_key: str,
                       row: pd.Series) -> dict:
    user_msg = (f"CANDIDATE PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
                f"JOB POSTING:\n{_posting_text(row)}")
    try:
        verdict = llm.chat_json(
            CONFIG["prompts"]["match_extract_system"], user_msg
        )
        for key in _empty_candidate_verdict():
            verdict.setdefault(key, [])
        error = None
    except Exception as e:
        verdict = _empty_candidate_verdict()
        error = f"candidate extraction failed at key '{hit_key}' with error: {e}"

    return {"key": hit_key, **verdict,
            "candidate_score": _candidate_score(verdict), "error": error}


# --- shared concurrent runner ------------------------------------------------

def _extract_concurrent(
    rows: list[tuple[str, pd.Series]],
    extractor: Callable[[str, pd.Series], dict],
    dump_subdir: Path | None,
    cache_subdir: Path | None,
    desc: str,
) -> list[dict]:
    """Run `extractor` over `rows` in parallel; stream-dump and preserve order.

    For each row, first check `cache_subdir` for a previously-dumped record
    matching the slugified key. Cache hits skip the threadpool and skip the
    dump (the new dump_subdir gets a copy too, so a successful resumed run
    leaves a self-contained folder for the *next* resume).
    """
    parallelism = CONFIG["match"]["score_parallelism"]
    out: list[dict | None] = [None] * len(rows)

    # assign indices to each row, load cached results in the output
    # and pass the indexed pending rows to the ThreadPoolExecutor
    pending: list[tuple[int, str, pd.Series]] = []
    for i, (hit_key, row) in enumerate(rows):
        cached = _load_cached(cache_subdir, hit_key)
        if cached is not None:
            out[i] = cached
            # Mirror into the new dump dir so this run's folder is also a
            # complete cache for any future resume.
            _dump_record(dump_subdir, cached, row)
        else:
            pending.append((i, hit_key, row))

    if cache_subdir is not None:
        n_hits = len(rows) - len(pending)
        print(f"[{desc}] cache: {n_hits}/{len(rows)} reused from {cache_subdir}")

    if pending:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            # map pool submissions to the indices, so that we don't need
            # to wait for them sequentially. That is, so we can loop 
            # over as_completed(futures)) instead of looping over 
            # `futures` itself, and we still keep the right rows order
            futures = {
                pool.submit(extractor, hit_key, row): i
                for i, hit_key, row in pending
            }
            with tqdm(total=len(pending), desc=desc, unit="post") as bar:
                for fut in as_completed(futures):
                    i = futures[fut]
                    record = fut.result()
                    out[i] = record
                    _dump_record(dump_subdir, record, rows[i][1])
                    bar.update(1)

    return out  # type: ignore[return-value]


# --- graph nodes -------------------------------------------------------------

class MatchState(TypedDict):
    profile: CVProfile        # candidate-coded themes (used by candidate pass)
    cv_text: str              # raw CV (used by recruiter pass)
    df: pd.DataFrame
    index: Index
    llm: LLM
    top_n_retrieve: int
    dump_dir: Path | None
    previous_run_dir: Path | None
    hits: list[dict[str, Any]]
    recruiter: list[dict[str, Any]]
    survivor_keys: list[str]
    scored: list[dict[str, Any]]
    result: pd.DataFrame


def _retrieve(state: MatchState) -> dict:
    return {"hits": state["index"].search(
        state["profile"]["search_query"], k=state["top_n_retrieve"]
    )}

def _prescreen_node(state: MatchState) -> dict:
    """Prescreen node: just skip the postings outside of criteria
    defined in the prescreen config prompt."""

    df, hits, llm, cv_text = state["df"], state["hits"], state["llm"], state["cv_text"]
    rows = [
        (
            hit["key"],
            df.loc[hit["key"]]
        ) for hit in hits
    ]
    dump_dir = state.get("dump_dir")
    prev_dir = state.get("previous_run_dir")
    decisions = _extract_concurrent(
        rows,
        lambda hit_key, row: _prescreen(llm, cv_text, hit_key, row),
        dump_subdir=dump_dir / "prescreen" if dump_dir else None,
        cache_subdir=prev_dir / "prescreen" if prev_dir else None,
        desc="prescreen",
    )

    # Fail-open: anything that isn't a clean "reject" survives (malformed
    # / errored prescreens shouldn't silently drop postings).
    return {
        "survivor_keys": [d["key"] for d in decisions
                          if d["decision"].lower() != "reject"]
    }

def _recruiter_node(state: MatchState) -> dict:
    """Second pass: recruiter persona scores the postings 0–100."""
    
    df, llm, cv_text = state["df"], state["llm"], state["cv_text"]
    rows = [(k, df.loc[k]) for k in state["survivor_keys"]]
    dump_dir = state.get("dump_dir")
    prev_dir = state.get("previous_run_dir")
    records = _extract_concurrent(
        rows,
        lambda hit_key, row: _extract_recruiter(llm, cv_text, hit_key, row),
        dump_subdir=dump_dir / "recruiter" if dump_dir else None,
        cache_subdir=prev_dir / "recruiter" if prev_dir else None,
        desc="recruiter",
    )

    # Keep the top `keep_pct` for the candidate pass. Always keep at least
    # one row so an aggressive cut on tiny result sets doesn't return empty.
    keep_pct = CONFIG["match"]["recruiter_keep_pct"]
    n_keep = max(1, round(len(records) * keep_pct / 100))
    survivors = sorted(records, key=lambda r: r["recruiter_score"],
                       reverse=True)[:n_keep]
    return {"recruiter": records,
            "survivor_keys": [r["key"] for r in survivors]}


def _candidate_node(state: MatchState) -> dict:
    """Third pass: score postings based on candidate-oriented profile."""

    df, llm, profile = state["df"], state["llm"], state["profile"]
    rows = [(k, df.loc[k]) for k in state["survivor_keys"]]
    dump_dir = state.get("dump_dir")
    prev_dir = state.get("previous_run_dir")
    scored = _extract_concurrent(
        rows,
        lambda hit_key, row: _extract_candidate(llm, profile, hit_key, row),
        dump_subdir=dump_dir / "candidate" if dump_dir else None,
        cache_subdir=prev_dir / "candidate" if prev_dir else None,
        desc="candidate",
    )
    return {"scored": scored}


def _select(state: MatchState) -> dict:
    """Join both score sets back to the DataFrame and dump summaries.

    The result frame contains only survivor rows, and the full recruiter 
    pass still lands on disk under `recruiter/` so nothing is lost.

    We index on `key` for the join, then reset so the column survives in
    the returned frame.
    """

    candidate_df = pd.DataFrame(state["scored"]).set_index("key")
    
    # select which columns to keep from the recruiter pass
    recruiter_cols = RECRUITER_THEMES + ["recruiter_score", "recruiter_notes"]
    recruiter_df = pd.DataFrame(state["recruiter"]).set_index("key")[recruiter_cols]
    
    survivors = state["df"][state["df"]["key"].isin(candidate_df.index)]
    result = (survivors.set_index("key")
              .join(candidate_df)
              .join(recruiter_df)
              .sort_values(["candidate_score", "recruiter_score"], ascending=False)
              .reset_index())

    dump_dir = state.get("dump_dir")
    if dump_dir is not None:
        result.to_parquet(dump_dir / "selected_jobs.parquet")
        summary_cols = [c for c in
                        ["key", "title", "employer_name", "city", "country",
                         "recruiter_score", "candidate_score", "error"]
                        if c in result.columns]
        result[summary_cols].to_csv(dump_dir / "summary.csv", index=False)
        (dump_dir / "profile.json").write_text(
            json.dumps(state["profile"], indent=2)
        )

    return {"result": result.reset_index(drop=True)}


def build_graph(prescreen_only: bool = False):
    g = StateGraph(MatchState)

    g.add_node("retrieve", _retrieve)
    g.add_node("prescreen", _prescreen_node)

    if not prescreen_only:
        g.add_node("recruiter_score", _recruiter_node)
        g.add_node("candidate_score", _candidate_node)
        g.add_node("select", _select)
    
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "prescreen")

    if prescreen_only:
        g.add_edge("prescreen", END)

        return g.compile()

    g.add_edge("prescreen", "recruiter_score")
    g.add_edge("recruiter_score", "candidate_score")
    g.add_edge("candidate_score", "select")
    g.add_edge("select", END)
    return g.compile()


@lru_cache(maxsize=2)
def get_graph(prescreen_only: bool = False):
    """Return a compiled graph, cached per variant.

    Two variants exist (full / prescreen-only); each is compiled at most
    once per process.
    """
    return build_graph(prescreen_only=prescreen_only)
