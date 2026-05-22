"""Orchestrator: run all actor scrapers and combine into one DataFrame.

Each actor module (alljobs.py, xing.py, ...) exposes a `scrape()` that
returns a DataFrame with the unified schema:

    key, platform_url, job_url, title, employer_name,
    employer_ratings_count, employer_rating, country, city,
    posted_at, employment_type, description, raw

`alljobs` is a single Apify actor that covers LinkedIn + Indeed + Glassdoor +
ZipRecruiter (and a few regional boards) in one call.

Note on terminology: an *actor* is one of our scraper modules (alljobs,
xing) -- i.e. which Apify actor produced the row. A *platform* is the real
job board the listing came from (LinkedIn, Indeed, Glassdoor, Xing, ...).
The unified `alljobs` actor spans several platforms, so the two are not
1:1 and the code keeps them distinct. Real-platform provenance lives in
`source_platform` (set by each scraper's `_normalize`) and is aggregated
into `found_on` during dedupe.

This script tags each row with `actor`, concats, dedupes, and writes
parquet + csv outputs.

Usage (if run from the parent directory of `scraper/`):
    python -m scraper.main "data scientist" --location Berlin --limit 50
    python -m scraper.main "ML engineer" --location Berlin --limit 50 --actors alljobs
"""
from __future__ import annotations
import re
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .sources import alljobs, xing, ACTOR_NAMES
from .sources._utils import _make_key


def _scrape_one(actor: str, args) -> pd.DataFrame:
    """Dispatch to the right actor scraper with its specific kwargs."""
    if actor == "alljobs":
        return alljobs.scrape(
            title=args.title, location=args.location,
            country=args.country, limit=args.limit,
            date_posted=args.date_posted,
            remote_only=getattr(args, "remote_only", False),
            distance=getattr(args, "distance", None),
            job_type=getattr(args, "job_type", ""),
            currency=getattr(args, "currency", ""),
        )
    if actor == "xing":
        return xing.scrape(
            title=args.title, location=args.location,
            limit=args.limit,
            discipline=getattr(args, "discipline", ""),
            max_pages=getattr(args, "max_pages", ""),
            start_url=getattr(args, "start_url", ""),
        )
    raise ValueError(f"unknown actor: {actor}")

def _dedupe(combined: pd.DataFrame) -> pd.DataFrame:
    """---- Dedupe across platforms -------------------------------------
    The same role is often cross-posted (e.g. an Indeed listing that also
    shows up on LinkedIn). We collapse those into one row and record every
    platform it was found on in a `found_on` list column.

    The match key is (employer_name, title, city), but the raw values are
    noisy across sources: "Acme GmbH" vs "acme gmbh", trailing whitespace,
    double spaces, etc. We build normalized `_dedup_*` shadow columns
    (lowercased, whitespace-collapsed) and match on those, then drop them
    before returning.

    For `found_on` we use the per-row `source_platform` column, which every
    scraper's `_normalize` now sets to the real job board name ("LinkedIn",
    "Indeed", "Glassdoor", "Xing", ...). The orchestrator-level `actor`
    column (which just says "alljobs" or "xing") is intentionally not used
    here -- it would hide the original source for unified-actor rows."""

    key_cols = ["employer_name", "title", "city"]
    dedup_cols = [f"_dedup_{c}" for c in key_cols]

    for src, dst in zip(key_cols, dedup_cols):
        combined[dst] = (
            combined[src]
            .astype("string")                          # tolerate None / NaN
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)      # collapse internal whitespace
        )

    # Effective per-row source: the actor-reported `source_platform`. Every
    # scraper's `_normalize` sets this, so there shouldn't be empty values
    # in practice; fall back to the orchestrator `actor` defensively so a
    # forgetful future _normalize doesn't silently produce empty found_on.
    combined["_source"] = combined["source_platform"].fillna("").astype("string")
    combined["_source"] = combined["_source"].where(
        combined["_source"].str.len() > 0, combined["actor"]
    )

    # A row can only be cross-matched if all three key parts are present.
    # Rows missing any part (common on LinkedIn, which often omits city)
    # would otherwise collapse against each other via pandas' NaN-equals-NaN
    # behavior in drop_duplicates. Split them out and rejoin at the end.
    has_key = (combined[dedup_cols].fillna("") != "").all(axis=1)
    keyed = combined[has_key]
    unkeyed = combined[~has_key]

    # For keyed rows: group by the normalized key to collect which platforms
    # saw each job, then merge that list back onto the deduped frame.
    found_on = (
        keyed.groupby(dedup_cols)["_source"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={"_source": "found_on"})
    )
    deduped_keyed = (
        keyed.drop_duplicates(subset=dedup_cols, keep="first")
        .merge(found_on, on=dedup_cols, how="left")
    )

    # For unkeyed rows we can't cross-match, so each one is only "found on"
    # its own platform. Wrap in a list so the column has a consistent type.
    unkeyed = unkeyed.assign(found_on=unkeyed["_source"].map(lambda p: [p]))

    # Recombine and clean up. We drop `actor` and `source_platform` because
    # after dedup they're just the arbitrary first-seen values (misleading
    # for cross-posted jobs). Downstream filters should use
    # `"LinkedIn" in row["found_on"]`.
    drop_cols = dedup_cols + ["actor", "_source", "source_platform"]
    deduped_all = (
        pd.concat([deduped_keyed, unkeyed], ignore_index=True)
        .drop(columns=drop_cols)
        .reset_index(drop=True)
    )

    return deduped_all

def run(args) -> pd.DataFrame:
    """Run every selected actor and return the combined, deduped DataFrame."""
    frames = []
    for actor in args.actors:
        try:
            df = _scrape_one(actor, args)
            df["actor"] = actor
            frames.append(df)
            print(f"[{actor}] {len(df)} jobs")
        except Exception as e:
            print(f"[{actor}] FAILED: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"raw rows across actors: {len(combined)}")

    deduped = _dedupe(combined)
    deduped["key"] = deduped.apply(_make_key, axis=1)
    print(f"unique jobs: {len(deduped)}")

    return deduped


def safe_name(s, n: int = 60) -> str:
    """Filesystem-safe slug. Handles NaN, None, and non-string types."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "untitled"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")[:n] or "untitled"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("title", help="search keyword / role title")
    p.add_argument("--location", default="")
    p.add_argument("--limit", type=int, default=50, help="per-actor max results")
    p.add_argument("--actors", nargs="+", default=ACTOR_NAMES,
                   choices=ACTOR_NAMES,
                   help="which actors to run (default: all)")
    # alljobs-specific (also used as the global country/recency filter)
    p.add_argument("--country", default="de",
                   help="alljobs: 2-letter code ('de') or full name ('Germany')")
    p.add_argument("--date-posted", default="14",
                   help='alljobs: "1", "3", "7", "14" days, freeform like '
                        '"6 months", or "" for any time')
    p.add_argument("--remote-only", action="store_true",
                   help="alljobs: restrict to remote-friendly listings")
    p.add_argument("--distance", type=int, default=None,
                   help="alljobs: search radius in miles (50-10000)")
    p.add_argument("--job-type", default="",
                   choices=["", "all", "fulltime", "parttime",
                            "internship", "contract"],
                   help="alljobs: employment type filter")
    p.add_argument("--currency", default="",
                   help="alljobs: ISO currency code for salary FX normalization")
    # Xing-specific
    p.add_argument("--discipline", default="", help="Xing: professional field filter")
    p.add_argument("--max-pages", type=int, default=20, help="Xing: max result pages")
    p.add_argument("--start-url", default="",
                   help="Xing: direct search URL; when set, overrides "
                        "keyword/location/discipline")

    # Output
    p.add_argument("--out-dir", default=".", help="directory for output files")
    p.add_argument("--no-save", action="store_true", help="don't write files")
    args = p.parse_args()

    jobs = run(args)
    if jobs.empty:
        print("No results.")
        return

    if args.no_save:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_results" / stamp
    raw_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"jobs_{stamp}.parquet"
    csv_path = out_dir / f"jobs_{stamp}.csv"

    # dump entire raw column as individual json
    for i, row in jobs.iterrows():
        fname = f"{i:04d}_{safe_name(row['employer_name'])}_{safe_name(row['title'])}.json"
        with open(raw_dir / fname, "w", encoding="utf-8") as f:
            json.dump(row["raw"], f, ensure_ascii=False, indent=2, default=str)

    jobs.drop(columns=["raw"]).to_parquet(parquet_path)
    jobs.drop(columns=["raw", "description"], errors="ignore").to_csv(csv_path, index=False)
    print(f"Saved {parquet_path} and {csv_path}")

if __name__ == "__main__":
    main()
