"""Gradio interface for the `scraper` package.

Run from root folder with:
    python -m interface.scraper_app.py

The UI supports a portfolio of queries: the form preloads a default list 
of searches; you can edit, remove, or append rows. Each query runs 
independently with its own try/except so one failure doesn't kill the rest. 
Results are concatenated and duplicate jobs (same employer + title + city 
across queries) are dropped.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field

# this import applies GRADIO_ANALYTICS_ENABLED="False" 
# (i.e., it's in the __init__.py)
import interface  # noqa: F401

import gradio as gr
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from scraper.main import run, safe_name
from scraper.sources import ACTOR_NAMES

from interface.common import (
    DynamicList,
    build_dynamic_list,
    clean_list,
    coerce_str,
    stringify_found_on,
    wire_dynamic_list,
    write_table_outputs,
)


# Columns shown in the on-screen results table. We drop `raw` (huge nested
# dicts) and `description` (multi-paragraph free text) from the downloads;
# the preview is further narrowed to the human-facing subset below. Edit
# PREVIEW_COLS to change what shows in the UI without affecting the
# parquet/csv contents.
PREVIEW_DROP_COLS = ["raw", "description"]
PREVIEW_COLS = [
    "title",
    "employer_name",
    "city",
    "country",
    "posted_at",
    "employment_type",
    "found_on",
    "job_url",
]

JOB_TYPE_CHOICES = ["", "all", "fulltime", "parttime", "internship", "contract"]

# Default query portfolio
DEFAULT_QUERIES: list[str] = [
    "data scientist",
    "data analyst",
    "quantitative analyst",
    "applied researcher",
]

# Pool size for query slots. Gradio Blocks don't easily support truly
# dynamic component lists, so we pre-allocate this many rows and just
# toggle visibility. Must be >= len(DEFAULT_QUERIES) with headroom for
# the "+ Add query" button.
MAX_QUERIES = max(30, len(DEFAULT_QUERIES) + 10)

# Defaults for numeric filter fields.
DEFAULT_RESULT_LIMIT = 20
DEFAULT_MAX_PAGES = 5
DEFAULT_DISTANCE_MILES = 50

# Names of the shared-filter components, in the order they are passed to
# scrape_jobs() after the query textboxes. Single source of truth used both
# to unpack all_inputs and to build the ScrapeArgs in the callback.
FILTER_NAMES: tuple[str, ...] = (
    "location", "limit", "actors",
    "country", "date_posted", "remote_only", "distance", "job_type", "currency",
    "discipline", "max_pages", "start_url",
)


@dataclass
class ScrapeArgs:
    """All parameters needed for a single scrape call.

    Mirrors the attribute names that `run()` reads off its args Namespace so
    that run() works without any modification; we just pass a ScrapeArgs
    instance in place of the Namespace.
    """
    title: str
    location: str = ""
    limit: int = DEFAULT_RESULT_LIMIT
    actors: list[str] = field(default_factory=list)
    country: str = ""
    date_posted: str = ""
    remote_only: bool = False
    distance: int | None = None
    job_type: str = ""
    currency: str = ""
    discipline: str = ""
    max_pages: int = DEFAULT_MAX_PAGES
    start_url: str = ""


# --- helpers ----------------------------------------------------------------

# `_s` is the local spelling of the shared coercion helper so the many call 
# sites below (within _build_args in particular) read better.
_s = coerce_str


def _int(v, default: int | None) -> int | None:
    """Coerce a form field to int, returning `default` when blank or None."""
    s = _s(v)
    return int(s) if s else default


def _label_for(query: str) -> str:
    """Derive the progress-log label for a query.

    Lowercased, collapse whitespace into underscore.
    """
    q = _s(query).lower()
    q = re.sub(r"\s+", "_", q)
    return q or "query"


def _build_args(title: str, filters: dict) -> ScrapeArgs:
    """Pack the form fields into a ScrapeArgs instance that `run()` expects."""
    return ScrapeArgs(
        title=_s(title),
        location=_s(filters["location"]),
        limit=_int(filters["limit"], DEFAULT_RESULT_LIMIT),
        actors=filters["actors"] or list(ACTOR_NAMES),
        country=_s(filters["country"]),
        date_posted=_s(filters["date_posted"]),
        remote_only=bool(filters["remote_only"]),
        distance=_int(filters["distance"], None),
        job_type=_s(filters["job_type"]),
        currency=_s(filters["currency"]),
        discipline=_s(filters["discipline"]),
        max_pages=_int(filters["max_pages"], DEFAULT_MAX_PAGES),
        start_url=_s(filters["start_url"]),
    )


def _preview_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    """Strip heavy columns and stringify list cells for gr.Dataframe."""
    preview = jobs.drop(columns=PREVIEW_DROP_COLS, errors="ignore").copy()
    # gr.Dataframe doesn't render Python lists cleanly; join `found_on`.
    if "found_on" in PREVIEW_COLS:
        preview["found_on"] = preview["found_on"].apply(stringify_found_on)
    # Narrow to PREVIEW_COLS
    # (defensive so it never crashes for pulling the wrong column)
    cols = [c for c in PREVIEW_COLS if c in preview.columns]
    return preview[cols] if cols else preview


def _write_outputs(jobs: pd.DataFrame) -> tuple[str, str, str]:
    """Write parquet + csv + a zipped raw-json bundle to a temp dir.

    Mirrors the layout `main.py` produces on disk, just under a fresh
    tmpdir per run so concurrent Gradio sessions don't stomp on each other.
    Returns (parquet_path, csv_path, raw_zip_path).
    """
    # The shared writer handles the parquet/csv pair. We hand it the frame
    # already minus `raw` (the parquet should never carry the huge nested
    # dicts), so the csv only needs to additionally drop `description`.
    parquet_jobs = jobs.drop(columns=["raw"], errors="ignore")
    csv_drop = [c for c in PREVIEW_DROP_COLS if c != "raw"]
    out_dir, stamp = write_table_outputs(
        parquet_jobs, prefix="jobs", csv_drop_cols=csv_drop
    )
    parquet_path = out_dir / f"jobs_{stamp}.parquet"
    csv_path = out_dir / f"jobs_{stamp}.csv"

    # Raw-JSON bundle: scraper-specific, written into the same temp dir.
    raw_dir = out_dir / "raw_results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i, row in jobs.iterrows():
        fname = (
            f"{i:04d}_{safe_name(row.get('employer_name'))}_"
            f"{safe_name(row.get('title'))}.json"
        )
        with open(raw_dir / fname, "w", encoding="utf-8") as f:
            json.dump(row["raw"], f, ensure_ascii=False, indent=2, default=str)

    raw_zip_base = out_dir / f"raw_{stamp}"
    raw_zip_path = shutil.make_archive(str(raw_zip_base), "zip", raw_dir)

    return str(parquet_path), str(csv_path), str(raw_zip_path)


def _dedupe_across_queries(all_jobs: pd.DataFrame) -> pd.DataFrame:
    """Collapse cross-query duplicates on (employer_name, title, city).

    Same approach as the notebook: a job that surfaces under several
    search labels is one job. Keep the first occurrence (so the
    earlier-listed query "wins" the search_label/search_query
    attribution) and drop the search-portfolio-tracking columns.
    """
    if all_jobs.empty:
        return all_jobs
    dedupe_key = [c for c in ("employer_name", "title", "city")
                  if c in all_jobs.columns]
    drop_cols = ["search_label", "search_query"]
    if not dedupe_key:
        return (all_jobs.drop(columns=drop_cols, errors="ignore")
                .reset_index(drop=True))
    return (
        all_jobs.drop_duplicates(subset=dedupe_key, keep="first")
        .drop(columns=drop_cols, errors="ignore")
        .reset_index(drop=True)
    )


# --- UI event callbacks -----------------------------------------------------
# Add/remove/clear behaviour for the query list now comes from
# interface.common.wire_dynamic_list. Only the filter-accordion toggle, which
# is scraper-specific, remains local.


def _toggle_groups(selected: list[str]):
    """Show/hide actor-specific filter accordions based on actor selection."""
    return (
        gr.update(visible=("alljobs" in selected)),
        gr.update(visible=("xing" in selected)),
    )


# --- main callback ----------------------------------------------------------

def scrape_jobs(
    *all_inputs,
    progress=gr.Progress(track_tqdm=False),
):
    """Gradio callback.

    Inputs are flattened: MAX_QUERIES textbox values followed by the shared
    filter fields in FILTER_NAMES order. Gradio doesn't pass list-typed
    inputs to a regular function, so we unpack positionally and immediately
    zip the filters into a dict for self-documenting access downstream.

    Returns (status_md, preview_df, parquet, csv, zip).
    """
    query_values = all_inputs[:MAX_QUERIES]
    filters = dict(zip(FILTER_NAMES, all_inputs[MAX_QUERIES:]))

    # Filter to real, non-empty queries (hidden slots arrive as None/"") and
    # dedupe identical queries case-insensitively, in case the user left a
    # duplicate in the list by accident.
    real_queries = clean_list(query_values, dedupe=True)

    if not real_queries:
        raise gr.Error("Enter at least one query.")
    if not filters["actors"]:
        raise gr.Error("Select at least one actor.")
    if "APIFY_TOKEN" not in os.environ:
        raise gr.Error(
            "APIFY_TOKEN env var is not set. Set it (e.g. in a .env file) "
            "before launching the app."
        )

    n = len(real_queries)
    frames: list[pd.DataFrame] = []
    log_lines: list[str] = []

    # # OLD, not parallelised
    # for i, query in enumerate(real_queries):
    #     label = _label_for(query)
    #     args = _build_args(title=query, filters=filters)
    #     try:
    #         df = run(args)
    #         df["search_label"] = label
    #         df["search_query"] = query
    #         frames.append(df)
    #         log_lines.append(f"`[{label}]` {len(df)} jobs")
    #         print(f"[{label}] {len(df)} jobs")
    #     except Exception as e:
    #         # Per-query isolation: log and move on
    #         log_lines.append(f"`[{label}]` FAILED: {e}")
    #         print(f"[{label}] FAILED: {e}")
    #     progress(i / max(n, 1), desc=f"[{i+1}/{n}] {label}…")

    # Precompute the cheap, pure prep OUTSIDE the threads
    queries_to_run = [(q, _label_for(q), _build_args(title=q, filters=filters)) 
                      for q in real_queries]

    # n == len(real_queries) : one worker per query, no throttling
    with ThreadPoolExecutor(max_workers=n) as pool: 
        # Submit all queries at once and map futures to their (query, label) 
        # so we know what came back when it finishes
        future_to_query = {
            pool.submit(run, args): (query, label)
            for query, label, args in queries_to_run
        }

        # Collect in finishing order (over as_completed(...))
        for i, future in enumerate(as_completed(future_to_query)):
            # retrieve query, label tuple from futures map
            query, label = future_to_query[future]
            try:
                df = future.result() # exceptions are re-raised here
                df["search_label"] = label
                df["search_query"] = query
                frames.append(df)
                log_lines.append(f"`[{label}]` {len(df)} jobs")
                print(f"[{label}] {len(df)} jobs")
            except Exception as e:
                # Per-query isolation preserved even in threads
                log_lines.append(f"`[{label}]` FAILED: {e}")
                print(f"[{label}] FAILED: {e}")
            progress((i + 1) / n, desc=f"[{i + 1}/{n}] {label} done…")

    progress(0.98, desc="Combining and deduping…")

    if not frames:
        return (
            "**No results.** All queries failed.\n\n"
            + "\n".join(f"- {l}" for l in log_lines),
            pd.DataFrame(),
            None, None, None,
        )

    all_jobs = pd.concat(frames, ignore_index=True)
    raw_count = len(all_jobs)
    jobs = _dedupe_across_queries(all_jobs)

    if jobs.empty:
        return (
            "**No results.** Try different queries and filters.\n\n"
            + "\n".join(f"- {l}" for l in log_lines),
            pd.DataFrame(),
            None, None, None,
        )

    parquet_path, csv_path, zip_path = _write_outputs(jobs)
    preview = _preview_frame(jobs)

    summary = (
        f"**{len(jobs)} unique jobs** "
        f"(from {raw_count} raw rows across {n} "
        f"{'query' if n == 1 else 'queries'}) "
        f"using actors: `{', '.join(filters['actors'])}`.\n\n"
        + "\n".join(f"- {l}" for l in log_lines)
    )
    return summary, preview, parquet_path, csv_path, zip_path


# --- UI section builders ----------------------------------------------------
# Each returns the component handles its parent needs to wire events later.
# These exist so the top-level `with gr.Blocks` reads as page outline rather
# than as a wall of widget definitions.

def _build_query_list() -> DynamicList:
    """Build the dynamic-query-list section via the shared list builder.

    Pre-allocates MAX_QUERIES rows -- the first len(DEFAULT_QUERIES) are
    visible and prefilled, the rest hidden and revealed one at a time by the
    "+ Add query" button. Returns the :class:`DynamicList` of handles the
    top-level wiring needs.
    """
    gr.Markdown("### Queries")
    return build_dynamic_list(
        title="",  # the section already has its own "### Queries" header
        defaults=DEFAULT_QUERIES,
        pool_size=MAX_QUERIES,
        placeholder="e.g. data scientist",
        add_label="+ Add query",
    )


def _build_filters() -> dict:
    """Build the shared-filter section (location, limit, actors, accordions)."""
    with gr.Row():
        with gr.Column(scale=2):
            location = gr.Textbox(
                label="Location",
                placeholder="Berlin",
                info="\n"
            )
        with gr.Column(scale=1):
            limit = gr.Number(
                label="Per query/sub-actor results limit",
                value=DEFAULT_RESULT_LIMIT, precision=0, minimum=1,
                info="Suggested maximum value 50 (redundant and **wasted Apify usage**)."
            )

    actors = gr.CheckboxGroup(
        choices=list(ACTOR_NAMES),
        value=list(ACTOR_NAMES),
        label="Actors",
        info=(
            "Alljobs via Apify actor `agentx/all-jobs-scraper` covers "
            "LinkedIn, Indeed, Glassdoor, Stepstone and other various "
            "regional boards in one call. \nXing via Apify actor "
            "`shahidirfan/xing-jobs-scraper` is only for the German-speaking DACH region."
        )
    )

    with gr.Accordion("alljobs filters", open=True) as alljobs_group:
        with gr.Row():
            country = gr.Textbox(
                label="Country",
                value="de",
                info="2-letter code ('de') or full name ('Germany'). Required.",
            )
            date_posted = gr.Textbox(
                label="Date posted",
                value="7",
                info='Days ("1", "7", "14"), freeform ("6 months"), or empty for any.',
            )
        with gr.Row():
            remote_only = gr.Checkbox(label="Remote only", value=False)
            distance = gr.Number(
                label="Distance (miles)",
                value=DEFAULT_DISTANCE_MILES, precision=0,
                minimum=50, maximum=10000,
                info="Search radius. Leave blank for none.",
            )
            job_type = gr.Dropdown(
                label="Job type",
                choices=JOB_TYPE_CHOICES, value="",
            )
            currency = gr.Textbox(
                label="Currency",
                placeholder="EUR",
                info="ISO code for salary FX normalization. Optional.",
            )

    with gr.Accordion("Xing filters", open=True) as xing_group:
        with gr.Row():
            discipline = gr.Textbox(
                label="Discipline",
                placeholder="IT and Software Development",
                info="Xing professional-field filter. Optional.",
            )
            max_pages = gr.Number(
                label="Max pages",
                value=DEFAULT_MAX_PAGES,
                precision=0, minimum=1,
                info="Add more pages to get older results (in which case increase the per-query limit)"
            )
        with gr.Row():
            start_url = gr.Textbox(
                label="Start URL",
                placeholder=(
                    "https://www.xing.com/jobs/search/ki?keywords=data%20scientist"
                    "&location=Berlin&cityId=2950159.e2912c&sincePeriod=LAST_WEEK&radius=50"
                ),
                info=(
                    "Optional Xing search URL. When set, the platform derives its "
                    "filters from it and the query/location/discipline from above "
                    "are ignored. Only use this if you set a **single** query above."
                ),
            )

    return dict(
        location=location, limit=limit, actors=actors,
        country=country, date_posted=date_posted, remote_only=remote_only,
        distance=distance, job_type=job_type, currency=currency,
        discipline=discipline, max_pages=max_pages, start_url=start_url,
        alljobs_group=alljobs_group, xing_group=xing_group,
    )


def _build_outputs() -> tuple[gr.Markdown, gr.Dataframe, gr.File, gr.File, gr.File]:
    """Build the status/preview/downloads section below the Scrape button."""
    status = gr.Markdown()
    results = gr.Dataframe(
        label="Results preview (full data in the downloads below)",
        wrap=True,
        interactive=False,
        row_count=(15, "dynamic"),
        max_height=600,
    )

    with gr.Row():
        parquet_out = gr.File(label="parquet (full data, no raw)")
        csv_out = gr.File(label="csv (no raw, no job descriptions)")
        raw_zip_out = gr.File(label="raw JSONs (zip)")

    return status, results, parquet_out, csv_out, raw_zip_out


# --- UI ---------------------------------------------------------------------

with gr.Blocks(title="Job scraper", analytics_enabled=False) as demo:
    gr.Markdown(
        "# Job scraper\n"
        "Run the Apify-backed job scrapers and download the results. "
        "Edit the query list below. Note that each runs independently "
        "and duplicates (same employer + title + city) are collapsed. "
        "Requires `APIFY_TOKEN` in the environment (see `README_scraper.md`)."
    )

    query_list = _build_query_list()
    filters = _build_filters()

    run_btn = gr.Button("Scrape", variant="primary", size="lg")

    status, results, parquet_out, csv_out, raw_zip_out = _build_outputs()

    # --- Event wiring -----------------------------------------------------

    # Add/remove/clear for the query list. The cursor is a high-water mark for
    # Add (not a live count): after a remove, Add reveals a fresh slot rather
    # than re-using the gap -- simpler than tracking holes, and empty rows are
    # dropped by clean_list() at read time anyway.
    wire_dynamic_list(query_list)

    filters["actors"].change(
        _toggle_groups,
        inputs=filters["actors"],
        outputs=[filters["alljobs_group"], filters["xing_group"]],
    )

    run_btn.click(
        scrape_jobs,
        inputs=[*query_list.boxes, *[filters[name] for name in FILTER_NAMES]],
        outputs=[status, results, parquet_out, csv_out, raw_zip_out],
    )


if __name__ == "__main__":
    demo.launch()
