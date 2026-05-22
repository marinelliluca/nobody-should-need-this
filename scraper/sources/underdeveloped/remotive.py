"""Remotive jobs scraper via the public Remotive Remote Jobs API.

NB: this is deliberately not integrated in main.py

Endpoint: https://remotive.com/api/remote-jobs
Free, no auth. Server-side filters: `search` (ilike on title+description),
`category`, `company_name`, `limit`. No pagination — the API returns all
matches up to `limit`, sorted by publication date.

Quirks worth knowing about:
- Data is delayed by 24 hours by design (per Remotive's ToS — gives them
  attribution time before syndication).
- They explicitly cap polling at ~2 requests per minute. Don't loop.
- Every listing is fully remote, but `candidate_required_location` can be
  region-locked, e.g. "USA Only", "Europe", "EMEA", "Worldwide". We surface
  that on `country` and let the caller post-filter via `location_filter`.
- Attribution: if you display this data publicly, link back to remotive.com.

API reference: https://github.com/remotive-com/remote-jobs-api

Usage:
    from remotive import scrape
    df = scrape("data scientist", limit=50, location_filter="europe")
    df.to_parquet("jobs_remotive.parquet")
"""
from __future__ import annotations
import requests
import pandas as pd

from ._utils import parse_posted_at, strip_html

API_URL = "https://remotive.com/api/remote-jobs"


def _normalize(item: dict) -> dict:
    """Map Remotive response shape to the unified schema.

    Field mapping notes:
    - `key`: Remotive's numeric `id`, cast to str for schema consistency
      (other sources use string ids).
    - `country`: we put `candidate_required_location` here. It's the only
      geo info Remotive exposes and it's free-text — values like "USA Only",
      "Europe", "EMEA", "Worldwide", "Anywhere", or specific countries.
      Empty string when missing. `city` is always empty (remote-only).
    - `description`: always HTML on Remotive, so always strip.
    - `posted_at`: ISO 8601, handled by parse_posted_at's fall-through branch.
    """
    description = strip_html(item.get("description") or "")

    return {
        "key": str(item.get("id")) if item.get("id") is not None else None,
        "platform_url": item.get("url"),
        "job_url": item.get("url"),  # Remotive doesn't expose a separate apply URL
        "title": item.get("title"),
        "employer_name": item.get("company_name"),
        "employer_ratings_count": None,  # not exposed
        "employer_rating": None,
        "country": item.get("candidate_required_location") or "",
        "city": "",  # remote-only board
        "posted_at": parse_posted_at(item.get("publication_date")),
        "employment_type": item.get("job_type"),
        "description": description,
        "raw": item,
    }


def _matches_location(loc_string: str, needle: str) -> bool:
    """Case-insensitive substring match on candidate_required_location.

    Worldwide/Anywhere always pass — those are eligible from anywhere
    including the user's location, regardless of what they typed."""
    if not needle:
        return True
    if not loc_string:
        return False
    haystack = loc_string.lower()
    if "worldwide" in haystack or "anywhere" in haystack:
        return True
    return needle.lower() in haystack


def scrape(title: str, location: str = "", country: str = "",
           limit: int = 100, category: str = "",
           company_name: str = "",
           location_filter: str = "") -> pd.DataFrame:
    """Run a Remotive search, return a DataFrame.

    Args:
        title: free-text query against title+description (maps to `search`)
        location: ignored at the API level — Remotive has no location filter
            server-side. Use `location_filter` for client-side filtering.
            Kept in the signature for parity with other source modules.
        country: ignored, same reason.
        limit: max results (server caps response to this number)
        category: optional category slug, e.g. 'software-dev', 'data',
            'design'. Full list at
            https://remotive.com/api/remote-jobs/categories
        company_name: filter by company name (server-side ilike match)
        location_filter: case-insensitive substring matched against
            `candidate_required_location` after fetch. Listings tagged
            "Worldwide" or "Anywhere" always pass.
            Examples: 'europe', 'emea', 'germany', 'usa'.
    """
    params: dict[str, str | int] = {"limit": limit}
    if title:
        params["search"] = title
    if category:
        params["category"] = category
    if company_name:
        params["company_name"] = company_name

    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    items = data.get("jobs") or []

    if location_filter:
        items = [
            it for it in items
            if _matches_location(
                it.get("candidate_required_location") or "",
                location_filter,
            )
        ]

    return pd.DataFrame(_normalize(it) for it in items)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("title")
    p.add_argument("--category", default="")
    p.add_argument("--company-name", default="")
    p.add_argument("--location-filter", default="",
                   help="client-side substring filter on required location, "
                        "e.g. 'europe', 'germany'")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--out", default="jobs_remotive.parquet")
    args = p.parse_args()

    df = scrape(args.title, limit=args.limit, category=args.category,
                company_name=args.company_name,
                location_filter=args.location_filter)
    df.to_parquet(args.out)
    print(f"\nSaved {len(df)} rows to {args.out}")
