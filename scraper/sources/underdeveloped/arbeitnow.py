"""Arbeitnow remote-jobs scraper via the public JSON API.

NB: this is deliberately not integrated in main.py

Arbeitnow is a DACH-focused job board specializing in English-speaking,
visa-sponsorship, and remote roles. It exposes a free, key-less JSON API
at https://www.arbeitnow.com/api/job-board-api with page-based pagination.

This module pulls every listing Arbeitnow flags as remote, with no further
filtering — the assumption is that title/location/seniority filtering
happens downstream on the combined DataFrame across all sources.

Usage:
    from arbeitnow import scrape
    df = scrape(limit=500)
    df.to_parquet("jobs_arbeitnow.parquet")
"""
from __future__ import annotations
import time
import requests
import pandas as pd

from ._utils import parse_posted_at, strip_html

API_URL = "https://www.arbeitnow.com/api/job-board-api"

# Tag values Arbeitnow uses to mark fully-remote roles. The API returns a
# `remote` boolean on most listings, but some legitimately-remote roles
# only carry the tag, so we check both.
_REMOTE_TAGS = {"remote", "fully-remote", "100-remote", "work-from-home"}


def _is_remote(item: dict) -> bool:
    """True if Arbeitnow flags this listing as remote, via flag or tags."""
    if item.get("remote") is True: # overtly defensive, but better safe ...
        return True
    tags = {t.lower() for t in (item.get("tags") or [])}
    return bool(tags & _REMOTE_TAGS) # true if at least one element in common


def _normalize(item: dict) -> dict:
    """Map raw Arbeitnow API output to the unified schema."""
    # Arbeitnow returns one free-text location field ('Berlin', 'Remote',
    # 'Munich, Germany'). We can't reliably split city/country without a
    # gazetteer, so the full string goes in `city` and `country` is empty.
    # Downstream consumers that care about country should look at `raw`
    # or apply a shared gazetteer pass across all sources.
    location = item.get("location") or ""

    # `job_types` is a list of strings like ['Full Time', 'Contract'].
    # Join them so it fits the single-string `employment_type` column.
    job_types = item.get("job_types") or []
    employment_type = ", ".join(job_types) if job_types else None

    return {
        "key": item.get("slug"),
        "platform_url": item.get("url"),
        "job_url": item.get("url"),       # Arbeitnow doesn't split apply vs listing
        "title": item.get("title"),
        "employer_name": item.get("company_name"),
        "employer_ratings_count": None,    # not exposed
        "employer_rating": None,           # not exposed
        "country": "",                     # see comment above
        "city": location,
        "posted_at": parse_posted_at(item.get("created_at")),
        "employment_type": employment_type,
        "description": strip_html(item.get("description")),
        "raw": item,
    }


def _fetch_pages(max_pages: int, sleep_s: float = 0.4) -> list[dict]:
    """Pull pages from the API until we hit max_pages or run out.

    The API paginates via `?page=N` and returns ~100 items per page. We
    stop early when a page returns no data. A small sleep is polite to
    the free endpoint and avoids any soft rate-limit.

    Note: The API accepts `remote=true` and `search=...` query params
    without error but silently ignores them. The response is identical
    to the unfiltered feed. The only documented server-side filter is
    `visa_sponsorship`. All other filtering must be done client-side
    on the returned items.
    """
    all_items: list[dict] = []
    for page in range(1, max_pages + 1):
        resp = requests.get(API_URL, params={"page": page}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") or []
        if not items:
            break
        all_items.extend(items)
        time.sleep(sleep_s)
    return all_items


def scrape(limit: int = 500, max_pages: int = 20) -> pd.DataFrame:
    """Fetch all remote listings from Arbeitnow, return a DataFrame.

    Args:
        limit: max rows to return after the remote-only filter. Set high
            (e.g. 2000) to capture the full remote feed; downstream dedupe
            and filtering will trim it.
        max_pages: cap on pages fetched. Each page is ~100 rows, so the
            default 20 covers ~2000 raw listings before remote filtering.
            Raise if you find `limit` is being hit before pagination ends.
    """
    raw_items = _fetch_pages(max_pages=max_pages)
    remote_items = [it for it in raw_items if _is_remote(it)][:limit]
    return pd.DataFrame(_normalize(it) for it in remote_items)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--max-pages", type=int, default=20)
    p.add_argument("--out", default="jobs_arbeitnow.parquet")
    args = p.parse_args()

    df = scrape(limit=args.limit, max_pages=args.max_pages)
    df.to_parquet(args.out)
    print(f"\nSaved {len(df)} rows to {args.out}")