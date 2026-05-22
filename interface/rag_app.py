"""Gradio interface for the `rag` package.

Run from root folder with:
    python -m interface.rag_app.py

The UI supports the following workflow:
  1. Load a jobs parquet and validate its key index.
  2. Set up a CV profile (auto-code from CV or upload existing JSON),
     edit it in-place, and save to disk.
  3. Run match_cv(), then interactively tune scorer weights and threshold
     to filter the preview. Export the full scored top-N when happy.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import requests

# this import applies GRADIO_ANALYTICS_ENABLED="False" 
# (i.e., it's in the __init__.py)
import interface  # noqa: F401

import gradio as gr
import pandas as pd

from rag import code_cv, match_cv
from rag.match_graph import _recruiter_score, _candidate_score, recent_dump_dirs

from interface.common import (
    DynamicList,
    build_dynamic_list,
    clean_list,
    env_warning,
    hydrate_dynamic_list,
    run_in_thread_with_log,
    stringify_found_on,
    wire_dynamic_list,
    write_table_outputs,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREVIEW_COLS = [
    "title",
    "employer_name",
    "city",
    "recruiter_score",
    "candidate_score",
    "found_on",
    "job_url",
]

# Environment variables surfaced as a startup warning (see interface.common).
_ENV_VARS = {
    "CHROMA_DIR":      "ChromaDB persistence directory for the vector index.",
    "RAG_CONFIG_PATH": "Path to the RAG config file (defaults to ./config.toml).",
    "HF_TOKEN":        "Hugging Face token for embedding model access.",
}

_SCORE_RENAMES = {
    "recruiter_score": "rec_sc",
    "candidate_score": "cand_sc",
}

# The four CV-profile keys surfaced as editable theme columns. Each maps to a
# list of free-text theme strings in the profile JSON. `key -> column title`.
THEME_KEYS: dict[str, str] = {
    "role_themes":         "Role themes",
    "must_have_themes":    "Must-have themes",
    "nice_to_have_themes": "Nice to have themes",
    "disqualifiers":       "Disqualifiers",
}

# Pre-allocated slots per theme column (see build_dynamic_list). Generous
# headroom; only filled rows are read back.
THEME_POOL_SIZE = 30

# Dropdown sentinel for "no resume" in the resume-from-cache control. Shared
# by _build_match (to set the choice) and _run_match (to map it back to None).
_NO_RESUME = "— none (fresh run) —"

# Candidate scorer weight keys, in the canonical order used across the UI
# widgets, _compute_scores, and the scoring_inputs wiring. These match the
# verdict list-columns produced by the candidate pass (and, post-fix, the
# weight keys read by rag.match_graph._candidate_score).
CANDIDATE_WEIGHT_KEYS = [
    "must_haves_met",
    "must_haves_missing",
    "themes_matched",
    "disqualifiers_triggered",
]

# Populated by _build_theme_columns() at UI-build time: {key: DynamicList}.
# Lets the hydrate callback resolve each column's output handles from the
# THEME_KEYS alone.
_THEME_LISTS: dict[str, DynamicList] = {}


# ---------------------------------------------------------------------------
# CV theme columns  (editable lists for the four THEME_KEYS profile keys)
# ---------------------------------------------------------------------------
#
# These four keys are flat lists of free-text strings in the profile JSON.
# We surface each as its own dynamic list-of-textboxes column (same machinery
# as the scraper's query list) so they can be added/removed/edited without
# hand-editing JSON. The raw JSON editor remains the source of truth: columns
# hydrate *from* it (on auto-code / upload) and write *back into* it (on Run
# and Save). Sync is one-directional and only at those explicit moments.

def _build_theme_columns() -> dict[str, DynamicList]:
    """Render the four theme keys as a 2x2 grid of editable list columns.

    Returns ``{key: DynamicList}`` for the four THEME_KEYS, in declaration
    order. Laid out two-per-row (a single row of four wraps awkwardly to 3+1
    at typical widths). Defaults start empty -- the columns are populated by
    hydrating from a loaded/auto-coded profile. Also records the handles in
    the module-level ``_THEME_LISTS`` so the hydrate callback (which only
    receives the JSON text) can address each column's outputs.
    """
    lists: dict[str, DynamicList] = {}
    items = list(THEME_KEYS.items())
    for pair in (items[:2], items[2:]):
        with gr.Row():
            for key, title in pair:
                with gr.Column():
                    lists[key] = build_dynamic_list(
                        title=title,
                        defaults=[],
                        pool_size=THEME_POOL_SIZE,
                        placeholder="theme…",
                        add_label="+ Add",
                    )
    _THEME_LISTS.update(lists)
    return lists


def _parse_profile(profile_json: str | None) -> dict:
    """Parse the editor JSON into a dict, or raise a friendly gr.Error.

    A blank editor yields ``{}`` so hydration of an empty profile is a no-op
    rather than an error.
    """
    if not profile_json or not profile_json.strip():
        return {}
    try:
        obj = json.loads(profile_json)
    except json.JSONDecodeError as e:
        raise gr.Error(f"Invalid JSON: {e}")
    if not isinstance(obj, dict):
        raise gr.Error("Profile JSON must be an object at the top level.")
    return obj


def _themes_from_profile(profile_json: str | None) -> list:
    """Return flattened hydrate updates to fill all four theme columns.

    Reads the four THEME_KEYS off the parsed profile (tolerating missing keys
    and non-list values) and produces, concatenated in THEME_KEYS order, each
    column's ``hydrate_dynamic_list`` output. The caller must lay out
    ``outputs=`` as ``[next_slot, *rows, *boxes]`` per column, same order.
    """
    profile = _parse_profile(profile_json)
    updates: list = []
    for key in THEME_KEYS:
        raw = profile.get(key, [])
        # Tolerate a scalar or null where a list was expected.
        if raw is None:
            values = []
        elif isinstance(raw, (list, tuple)):
            values = [str(x) for x in raw]
        else:
            values = [str(raw)]
        updates.extend(hydrate_dynamic_list(_THEME_LISTS[key], values))
    return updates


def _themes_into_profile(profile_json: str | None, *all_boxes: str) -> str:
    """Splice the four columns' values back into the profile JSON.

    ``all_boxes`` is the flattened textbox values of all four columns, in
    THEME_KEYS order, each contributing THEME_POOL_SIZE entries. Empty/blank
    slots are dropped (clean_list). The four keys are overwritten on a parsed
    copy of the existing profile -- all other keys are preserved -- and the
    result re-serialized with the same indent the rest of the app uses.
    """
    profile = _parse_profile(profile_json)
    for i, key in enumerate(THEME_KEYS):
        start = i * THEME_POOL_SIZE
        chunk = all_boxes[start:start + THEME_POOL_SIZE]
        profile[key] = clean_list(chunk)
    return json.dumps(profile, indent=4)


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def _check_ollama() -> None:
    """Raise gr.Error if the Ollama server is not reachable."""
    from rag.config import CONFIG
    host = os.getenv("OLLAMA_HOST") or CONFIG["llm"]["default_host"]
    try:
        requests.get(f"{host}/api/tags", timeout=5)
    except requests.exceptions.RequestException as e:
        gr.Warning(
            f"Ollama server is not reachable at {host}: {e}\n"
            "Start the server and try again."
        )


# ---------------------------------------------------------------------------
# Section 1 — Data
# ---------------------------------------------------------------------------

def _build_data() -> tuple[gr.File, gr.File, gr.Markdown]:
    """Upload jobs parquet and CV file. Combined status line."""
    with gr.Row():
        parquet_upload = gr.File(
            label="Jobs parquet",
            file_types=[".parquet"],
        )
        cv_upload = gr.File(
            label="CV file (markdown or text)",
            file_types=[".md", ".txt"],
        )
    data_status = gr.Markdown()

    return parquet_upload, cv_upload, data_status


def _update_data_status(
    jobs_df: pd.DataFrame | None,
    cv_text: str | None,
) -> str:
    """Render a combined status line for dataset and CV."""
    parts = []
    if jobs_df is not None:
        parts.append(f"✅ **{len(jobs_df)} jobs loaded**")
    else:
        parts.append("⬜ No dataset loaded")
    if cv_text:
        parts.append("✅ **CV loaded**")
    else:
        parts.append("⬜ No CV loaded")
    return " · ".join(parts)


def _load_dataset(file, cv_text: str | None) -> tuple[pd.DataFrame, str]:
    """Load the parquet, verify key uniqueness, return (df, status_md)."""
    if file is None:
        raise gr.Error("Upload a parquet file first.")

    jobs_df = pd.read_parquet(file.name)

    if "key" not in jobs_df.columns:
        raise gr.Error("Parquet has no 'key' column.")

    if not jobs_df.set_index("key").index.is_unique:
        dupes = jobs_df[jobs_df["key"].duplicated(keep=False)]
        raise gr.Error(
            f"Found {len(dupes)} duplicate keys. "
            "Manually remove them before proceeding."
        )

    return jobs_df, _update_data_status(jobs_df, cv_text)


def _load_cv(file, jobs_df: pd.DataFrame | None) -> tuple[str, str]:
    """Read the CV file into cv_text_state, update combined status."""
    if file is None:
        raise gr.Error("Upload a CV file first.")
    cv_text = Path(file.name).read_text(encoding="utf-8")
    return cv_text, _update_data_status(jobs_df, cv_text)


# ---------------------------------------------------------------------------
# Section 2 — CV Profile
# ---------------------------------------------------------------------------

def _build_profile() -> tuple:
    """CV profile section: auto-code or upload, edit themes, raw-JSON, save."""

    source_radio = gr.Radio(
        choices=["Auto-code from CV", "Upload existing JSON"],
        value="Auto-code from CV",
        label="Profile source",
    )

    # -- Auto-code path ------------------------------------------------------
    with gr.Group(visible=True) as autocode_group:
        code_btn = gr.Button("Auto-code CV", variant="secondary")

    # -- Upload path ---------------------------------------------------------
    with gr.Group(visible=False) as upload_group:
        json_upload = gr.File(
            label="Profile JSON",
            file_types=[".json"],
        )

    # -- Editable theme columns (collapsed by default) -----------------------
    with gr.Accordion("Theme lists", open=False):
        gr.Markdown(
            "Edit the four theme lists below. They populate when you auto-code "
            "or upload a profile, and are written back into the raw JSON when "
            "you save or run a match. \n\nClick two times to add a new row."
        )
        theme_lists = _build_theme_columns()

    # -- Raw JSON editor (advanced; collapsed by default) --------------------
    # The JSON is the authoritative document: the columns above hydrate from
    # it and write back into it. Tucked in a collapsed accordion so it only
    # takes over the page if the user deliberately opens it to tinker.
    with gr.Accordion("Raw profile JSON (advanced)", open=False):
        profile_editor = gr.Textbox(
            label="Profile JSON (editable)",
            lines=20,
            max_lines=40,
            placeholder="Profile will appear here once loaded or auto-coded.",
        )

    # -- Save ----------------------------------------------------------------
    with gr.Row():
        save_btn = gr.Button("Save profile", variant="secondary")
        profile_download = gr.File(label="Download profile")

    profile_status = gr.Markdown()

    return (
        source_radio,
        autocode_group, code_btn,
        upload_group, json_upload,
        theme_lists,
        profile_editor,
        save_btn, profile_download,
        profile_status,
    )


def _toggle_profile_source(choice: str):
    """Show/hide the auto-code or upload group."""
    is_auto = choice == "Auto-code from CV"
    return gr.update(visible=is_auto), gr.update(visible=not is_auto)


def _autocode_cv(cv_text: str | None):
    """Generator: streams terminal output, yields profile JSON at end."""
    _check_ollama()
    if not cv_text:
        raise gr.Error("Load a CV file in Section 1 first.")

    # The helper streams the log; map each log string into our 3-tuple
    # (editor, status, log) with skips for the two slots that only update
    # at the end. The helper's return value is surfaced via StopIteration.
    log = ""
    gen = run_in_thread_with_log(code_cv, cv_text)
    try:
        while True:
            log = next(gen)
            yield gr.skip(), gr.skip(), log
    except StopIteration as stop:
        profile = stop.value

    yield (
        json.dumps(profile, indent=4),
        "✅ **Profile auto-coded.**",
        log,
    )


def _load_profile_json(file) -> tuple[str, str]:
    """Load and validate an uploaded profile JSON."""
    if file is None:
        raise gr.Error("Upload a JSON file first.")
    raw = Path(file.name).read_text(encoding="utf-8")
    try:
        profile = json.loads(raw)
    except json.JSONDecodeError as e:
        raise gr.Error(f"Invalid JSON: {e}")
    return json.dumps(profile, indent=4), "✅ **Profile loaded.**"


def _save_profile(editor_text: str) -> tuple[str, str]:
    """Validate the editor contents and write to a temp file for download."""
    try:
        json.loads(editor_text)
    except json.JSONDecodeError as e:
        raise gr.Error(f"Invalid JSON: {e}")

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", prefix="profile_"
    )
    tmp.close()
    Path(tmp.name).write_text(editor_text, encoding="utf-8")
    return tmp.name, "✅ **Profile saved.**"


# ---------------------------------------------------------------------------
# Section 3 — Match & Score
# ---------------------------------------------------------------------------

def _build_match() -> tuple:
    """Match & score section: advanced params, run button, results."""

    gr.Markdown("Click \"Run match\" only once per session and write down "
                "somewhere the final dump folder from the logs window above.")

    top_n_retrieve = gr.Number(
        label="top_n_retrieve",
        value=1000, minimum=1, maximum=10000, step=10,
        info=(
            "Number of jobs to retrieve for prescreening. "
            "Leave this number bigger than the total amount of jobs for maximum recall."
        ),
    )

    # Resume-from-cache: pick a previous run's dump folder to reuse its
    # per-record verdicts (e.g. resume a prescreen-only run on a bigger
    # model). Populated at build time from the 5 most recent folders under
    # [match.dump].root; the empty default means "fresh run, no cache".
    # Created during this session won't appear until the app reloads.
    previous_run_dir = gr.Dropdown(
        label="Resume from previous run",
        choices=[_NO_RESUME] + [str(p) for p in recent_dump_dirs(5)],
        value=_NO_RESUME,
        info=(
            "Reuse cached per-record verdicts from a prior run's dump folder "
            "(matched by job key). Lets you resume an interrupted run or run "
            "prescreen on a small model first, then the rest on a bigger one. "
            "Leave as 'none' for a fresh run."
        ),
    )

    run_btn = gr.Button("Run match", variant="primary", size="lg")

    # Default candidate weights pulled from config so the UI matches the
    # backend's actual scoring on first render. Read lazily here at build
    # time; falls back to 0 for any missing key.
    from rag.config import CONFIG
    _cand_defaults = CONFIG["match"].get("candidate_scoring", {})

    with gr.Accordion("Rescore", open=False):

        gr.Markdown(
            "Change any weight to recompute the recruiter and candidate "
            "scores live. The recruiter-score threshold below is the only "
            "filter; sort the results table by any column(s) to rank.")

        gr.Markdown("#### Recruiter score threshold")
        recruiter_thr = gr.Number(
            label="Recruiter score threshold",
            value=65, minimum=0, maximum=100, step=1,
        )

        gr.Markdown("#### Recruiter scorer weights")
        with gr.Row():
            w_technical = gr.Number(
                label="technical_match",
                value=0.40, minimum=0.0, maximum=1.0, step=0.05,
            )
            w_seniority = gr.Number(
                label="seniority_match",
                value=0.60, minimum=0.0, maximum=1.0, step=0.05,
            )
            w_transferability = gr.Number(
                label="transferability",
                value=0.0, minimum=0.0, maximum=1.0, step=0.05,
            )
            w_trajectory = gr.Number(
                label="trajectory",
                value=0.0, minimum=0.0, maximum=1.0, step=0.05,
            )
            w_soft_skills = gr.Number(
                label="soft_skills",
                value=0.0, minimum=0.0, maximum=1.0, step=0.05,
            )

        gr.Markdown("#### Candidate scorer weights")
        with gr.Row():
            w_must_haves_met = gr.Number(
                label="must_haves_met",
                value=_cand_defaults.get("must_haves_met", 0),
                minimum=-20, maximum=20, step=5,
            )
            w_must_haves_missing = gr.Number(
                label="must_haves_missing",
                value=_cand_defaults.get("must_haves_missing", 0),
                minimum=-20, maximum=20, step=5,
            )
            w_themes_matched = gr.Number(
                label="themes_matched",
                value=_cand_defaults.get("themes_matched", 0),
                minimum=-20, maximum=20, step=5,
            )
            w_disqualifiers_triggered = gr.Number(
                label="disqualifiers_triggered",
                value=_cand_defaults.get("disqualifiers_triggered", 0),
                minimum=-20, maximum=20, step=5,
            )

    match_status = gr.Markdown()

    results = gr.Dataframe(
        label="Results preview (survivors only, full scored data in download below)",
        wrap=True,
        interactive=False,
        row_count=(15, "dynamic"),
        max_height=600,
    )

    export_btn = gr.Button("Export", variant="secondary")
    with gr.Row():
        parquet_out = gr.File(label="parquet (full data)")
        csv_out = gr.File(label="csv (no job descriptions)")

    return (
        recruiter_thr,
        w_technical, w_seniority, w_transferability, w_trajectory, w_soft_skills,
        w_must_haves_met, w_must_haves_missing, w_themes_matched,
        w_disqualifiers_triggered,
        top_n_retrieve,
        previous_run_dir,
        run_btn, export_btn,
        match_status,
        results,
        parquet_out, csv_out,
    )


def _run_match(
    jobs_df:        pd.DataFrame | None,
    profile_json:   str,
    cv_text:        str | None,
    top_n_retrieve: float, # gr.Number hands back a float
    previous_run_dir: str | None,
):
    """Generator: streams terminal output into match_log, yields top+status at end."""
    _check_ollama()
    if jobs_df is None:
        raise gr.Error("Load a dataset first.")
    if not cv_text:
        raise gr.Error("Load a CV file in Section 1 first.")
    try:
        profile = json.loads(profile_json)
    except (json.JSONDecodeError, TypeError):
        raise gr.Error("Profile is missing or invalid. Load or auto-code a profile first.")

    # Map the dropdown sentinel back to None (fresh run); any real folder
    # path is forwarded to match_cv, which validates it and falls back to
    # config when None.
    prev_dir = None if not previous_run_dir or previous_run_dir == _NO_RESUME \
        else previous_run_dir

    log = ""
    gen = run_in_thread_with_log(
        match_cv, profile, cv_text, jobs_df,
        top_n_retrieve=int(top_n_retrieve),
        previous_run_dir=prev_dir,
    )
    try:
        while True:
            log = next(gen)
            yield gr.skip(), gr.skip(), log
    except StopIteration as stop:
        top = stop.value

    status = f"⏳ **Scoring {len(top)} candidates…**"
    yield top, status, log


def _compute_scores(
    top: pd.DataFrame,
    w_technical: float,
    w_seniority: float,
    w_transferability: float,
    w_trajectory: float,
    w_soft_skills: float,
    w_must_haves_met: float,
    w_must_haves_missing: float,
    w_themes_matched: float,
    w_disqualifiers_triggered: float,
) -> pd.DataFrame:
    """Pure function: apply recruiter + candidate scorer weights, return a
    scored copy.

    Both scores are recomputed per row from the columns that survive into
    ``top``: the recruiter rubric themes and the candidate verdict
    list-columns. No combined score -- the user ranks via the results table's
    own multi-column sort.
    """
    top = top.copy()

    recruiter_weights = {
        "technical_match":  w_technical,
        "seniority_match":  w_seniority,
        "transferability":  w_transferability,
        "trajectory":       w_trajectory,
        "soft_skills":      w_soft_skills,
    }
    top["recruiter_score"] = top.apply(
        lambda row: _recruiter_score(row, weights=recruiter_weights), axis=1
    )

    candidate_weights = {
        "must_haves_met":          w_must_haves_met,
        "must_haves_missing":      w_must_haves_missing,
        "themes_matched":          w_themes_matched,
        "disqualifiers_triggered": w_disqualifiers_triggered,
    }
    top["candidate_score"] = top.apply(
        lambda row: _candidate_score(row, weights=candidate_weights), axis=1
    )

    return top


def _apply_scoring(
    top_raw: pd.DataFrame | None,
    w_technical: float,
    w_seniority: float,
    w_transferability: float,
    w_trajectory: float,
    w_soft_skills: float,
    w_must_haves_met: float,
    w_must_haves_missing: float,
    w_themes_matched: float,
    w_disqualifiers_triggered: float,
    recruiter_thr: float,
) -> tuple[str, pd.DataFrame]:
    """Recompute scores, filter to survivors, update preview and status."""
    if top_raw is None:
        raise gr.Error("Run match first.")

    top = _compute_scores(
        top_raw,
        w_technical, w_seniority, w_transferability, w_trajectory, w_soft_skills,
        w_must_haves_met, w_must_haves_missing, w_themes_matched,
        w_disqualifiers_triggered,
    )

    survivors = (
        top[top["recruiter_score"] >= recruiter_thr]
        .sort_values(["recruiter_score", "candidate_score"], ascending=False)
        .reset_index(drop=True)
    )

    if survivors.empty:
        return (
            "**No jobs passed the threshold.** "
            "Try lowering `Recruiter score threshold`.",
            pd.DataFrame(),
        )

    preview_cols = [c for c in PREVIEW_COLS if c in survivors.columns]
    preview = survivors[preview_cols].copy()
    if "found_on" in preview.columns:
        preview["found_on"] = preview["found_on"].apply(stringify_found_on)

    preview.rename(columns=_SCORE_RENAMES, inplace=True)

    status = (
        f"✅ **{len(survivors)} jobs** passed the threshold (recruiter ≥ {recruiter_thr})."
    )

    return status, preview


def _export(
    top_raw: pd.DataFrame | None,
    w_technical: float,
    w_seniority: float,
    w_transferability: float,
    w_trajectory: float,
    w_soft_skills: float,
    w_must_haves_met: float,
    w_must_haves_missing: float,
    w_themes_matched: float,
    w_disqualifiers_triggered: float,
) -> tuple[str, str]:
    """Score the full top df and write parquet + csv for download."""
    if top_raw is None:
        raise gr.Error("Run match first.")

    top = _compute_scores(
        top_raw,
        w_technical, w_seniority, w_transferability, w_trajectory, w_soft_skills,
        w_must_haves_met, w_must_haves_missing, w_themes_matched,
        w_disqualifiers_triggered,
    )

    out_dir, stamp = write_table_outputs(
        top, prefix="matches", csv_drop_cols=["description", "raw"]
    )
    parquet_path = out_dir / f"matches_{stamp}.parquet"
    csv_path = out_dir / f"matches_{stamp}.csv"

    return str(parquet_path), str(csv_path)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAG job matcher", analytics_enabled=False) as demo:

    gr.Markdown(
        "# RAG job matcher\n"
        "Load a job dataset, set up your CV profile, then match and rank jobs."
    )

    warning_text = env_warning(_ENV_VARS)
    if warning_text:
        gr.Markdown(warning_text)

    # shared state
    jobs_state    = gr.State()
    top_state     = gr.State()
    cv_text_state = gr.State()

    # --- Section 1: Data ----------------------------------------------------
    gr.Markdown("## 1 · Data")
    parquet_upload, cv_upload, data_status = _build_data()

    parquet_upload.upload(
        _load_dataset,
        inputs=[parquet_upload, cv_text_state],
        outputs=[jobs_state, data_status],
    )

    cv_upload.upload(
        _load_cv,
        inputs=[cv_upload, jobs_state],
        outputs=[cv_text_state, data_status],
    )

    # --- Section 2: CV Profile ----------------------------------------------
    gr.Markdown("## 2 · CV Profile")
    (
        source_radio,
        autocode_group, code_btn,
        upload_group, json_upload,
        theme_lists,
        profile_editor,
        save_btn, profile_download,
        profile_status,
    ) = _build_profile()

    # Flattened theme-column handles, in THEME_KEYS order, for event wiring.
    #   _theme_boxes:   every textbox across all four columns (read-back input)
    #   _theme_hydrate_outputs: the [next_slot, *rows, *boxes] target list per
    #       column, concatenated -- matches _themes_from_profile's output order.
    _theme_boxes = [
        box for key in THEME_KEYS for box in theme_lists[key].boxes
    ]
    _theme_hydrate_outputs = [
        comp
        for key in THEME_KEYS
        for comp in (
            theme_lists[key].next_slot,
            *theme_lists[key].rows,
            *theme_lists[key].boxes,
        )
    ]

    # shared log box
    match_log = gr.Textbox(
        label="Log (last 10 lines)",
        lines=10,
        max_lines=10,
        interactive=False,
        autoscroll=True,
    )

    source_radio.change(
        _toggle_profile_source,
        inputs=[source_radio],
        outputs=[autocode_group, upload_group],
    )

    # Add/remove/clear behaviour for each theme column.
    for _dl in theme_lists.values():
        wire_dynamic_list(_dl)

    # Auto-code → fill the JSON editor → hydrate the four columns from it.
    code_btn.click(
        _autocode_cv,
        inputs=[cv_text_state],
        outputs=[profile_editor, profile_status, match_log],
    ).then(
        _themes_from_profile,
        inputs=[profile_editor],
        outputs=_theme_hydrate_outputs,
    )

    # Upload JSON → fill the editor → hydrate the columns from it.
    json_upload.upload(
        _load_profile_json,
        inputs=[json_upload],
        outputs=[profile_editor, profile_status],
    ).then(
        _themes_from_profile,
        inputs=[profile_editor],
        outputs=_theme_hydrate_outputs,
    )

    # Save: write the columns back into the JSON first, then save that.
    save_btn.click(
        _themes_into_profile,
        inputs=[profile_editor, *_theme_boxes],
        outputs=[profile_editor],
    ).then(
        _save_profile,
        inputs=[profile_editor],
        outputs=[profile_download, profile_status],
    )

    # --- Section 3: Match & Score -------------------------------------------
    gr.Markdown("## 3 · Match & Score")
    (
        recruiter_thr,
        w_technical, w_seniority, w_transferability, w_trajectory, w_soft_skills,
        w_must_haves_met, w_must_haves_missing, w_themes_matched,
        w_disqualifiers_triggered,
        top_n_retrieve,
        previous_run_dir,
        run_btn, export_btn,
        match_status,
        results,
        parquet_out, csv_out,
    ) = _build_match()

    # inputs reused across all three scoring calls. Order must match the
    # parameter order of _compute_scores / _apply_scoring / _export:
    # top_state, the 5 recruiter weights, then the 4 candidate weights.
    scoring_inputs = [
        top_state,
        w_technical, w_seniority, w_transferability, w_trajectory, w_soft_skills,
        w_must_haves_met, w_must_haves_missing, w_themes_matched,
        w_disqualifiers_triggered,
    ]

    # run match → store raw top → apply scoring → export.
    # First fold the theme columns back into the JSON so _run_match sees the
    # user's latest edits without them having to open the raw editor.
    run_btn.click(
        _themes_into_profile,
        inputs=[profile_editor, *_theme_boxes],
        outputs=[profile_editor],
    ).then(
        _run_match,
        inputs=[jobs_state, profile_editor, cv_text_state, top_n_retrieve, previous_run_dir],
        outputs=[top_state, match_status, match_log],
    ).then(
        _apply_scoring,
        inputs=scoring_inputs + [recruiter_thr],
        outputs=[match_status, results],
    ).then(
        _export,
        inputs=scoring_inputs,
        outputs=[parquet_out, csv_out],
    )

    # weight/threshold changes → preview only, no export
    for component in [
        w_technical, w_seniority, w_transferability, w_trajectory, w_soft_skills,
        w_must_haves_met, w_must_haves_missing, w_themes_matched,
        w_disqualifiers_triggered, recruiter_thr,
    ]:
        component.change(
            _apply_scoring,
            inputs=scoring_inputs + [recruiter_thr],
            outputs=[match_status, results],
        )

    # explicit export
    export_btn.click(
        _export,
        inputs=scoring_inputs,
        outputs=[parquet_out, csv_out],
    )


if __name__ == "__main__":
    demo.launch()
