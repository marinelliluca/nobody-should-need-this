"""Himalayas jobs scraper via the public Himalayas Remote Jobs API.

NB: this is deliberately not integrated in main.py

Uses the /jobs/api/search endpoint, which supports keyword + country + seniority
+ employment-type filters. All listings are 100% remote by definition (it's
what Himalayas does), but listings can carry a `locationRestrictions` array
limiting them to specific countries — we surface that via `country`.

Free, no auth. Rate limit is server-side; data refreshes every 24h, so don't
poll faster than that. Attribution required if displayed publicly (see ToS).

Usage:
    from himalayas import scrape
    df = scrape("data scientist", country="DE", limit=50)
    df.to_parquet("jobs_himalayas.parquet")

API reference: https://himalayas.app/api
"""
from __future__ import annotations
import time
import requests
import pandas as pd

from ._utils import parse_posted_at, strip_html

SEARCH_URL = "https://himalayas.app/jobs/api/search"
PAGE_SIZE = 20  # Hard server cap as of 2025-03-24; not configurable.


def _join_locations(locs) -> tuple[str, str]:
    """`locationRestrictions` is a list of {alpha2, name, slug} dicts.

    An empty list means worldwide-friendly. We collapse to two strings:
    country (semicolon-joined names) and country_codes (joined alpha2s).
    Returning empty strings for worldwide keeps the schema flat and lets
    downstream filters treat 'no restriction' uniformly."""
    if not locs:
        return "", ""
    names = [l.get("name", "") for l in locs if l.get("name")]
    codes = [l.get("alpha2", "") for l in locs if l.get("alpha2")]
    return ";".join(names), ";".join(codes)


def _normalize(item: dict) -> dict:
    """Map Himalayas response shape to the unified schema.

    Notes on field mapping:
    - `key`: Himalayas uses `guid` as the stable unique id.
    - `platform_url` vs `job_url`: Himalayas only exposes `applicationLink`
      (which points to the job page on himalayas.app and then redirects out).
      We put the same URL in both slots — there's no separate apply URL.
    - `city`: Himalayas is remote-only, so there is no city. Left empty.
    - `country`: collapsed from the `locationRestrictions` array; empty for
      worldwide-friendly roles.
    - `posted_at`: `pubDate` is a Unix ms timestamp; parse_posted_at handles
      that via its >1e12 branch.
    """
    locs = item.get("locationRestrictions") or []
    country, _country_codes = _join_locations(locs)

    description = item.get("description") or ""
    if "<" in description:
        description = strip_html(description)

    cats = item.get("categories") or item.get("category") or []
    employment_type = item.get("employmentType")

    return {
        "key": item.get("guid"),
        "platform_url": item.get("applicationLink"),
        "job_url": item.get("applicationLink"),
        "title": item.get("title"),
        "employer_name": item.get("companyName"),
        "employer_ratings_count": None,  # Himalayas doesn't expose ratings
        "employer_rating": None,
        "country": country,
        "city": "",  # remote-only board, no city
        "posted_at": parse_posted_at(item.get("pubDate")),
        "employment_type": employment_type,
        "description": description,
        "raw": item,
    }


def _fetch_page(params: dict, session: requests.Session) -> dict:
    """One HTTP call with basic error handling. Returns parsed JSON or {}."""
    r = session.get(SEARCH_URL, params=params, timeout=30)
    if r.status_code == 429:
        # Rate-limited. Back off once; if it persists, give up on this page
        # rather than hammering — Himalayas refreshes data only every 24h
        # so retrying aggressively gains nothing.
        time.sleep(5)
        r = session.get(SEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def scrape(title: str, location: str = "", country: str = "",
           limit: int = 100, seniority: str = "",
           employment_type: str = "",
           worldwide_only: bool = False) -> pd.DataFrame:
    """Run a Himalayas search, return a DataFrame.

    Args:
        title: free-text query (maps to `q`)
        location: ignored — Himalayas has no city/region filter, only country.
            Kept in the signature for parity with the other source modules.
        country: ISO alpha-2 code (e.g. 'DE', 'US'). Names and slugs also
            accepted by the API, but stick to alpha-2 for consistency.
        limit: max results across pages
        seniority: one of Entry-level, Mid-level, Senior, Manager,
            Director, Executive
        employment_type: one of Full Time, Part Time, Contractor, Temporary,
            Intern, Volunteer, Other
        worldwide_only: if True, restrict to worldwide-friendly listings
            (i.e. no country restrictions at all)
    """
    base_params: dict[str, str | int] = {"sort": "recent"}
    if title:
        base_params["q"] = title
    if country:
        base_params["country"] = country
    if seniority:
        base_params["seniority"] = seniority
    if employment_type:
        base_params["employment_type"] = employment_type
    if worldwide_only:
        base_params["worldwide"] = "true"

    items: list[dict] = []
    page = 1
    session = requests.Session()

    while len(items) < limit:
        params = {**base_params, "page": page}
        data = _fetch_page(params, session)
        batch = data.get("jobs") or []
        if not batch:
            break
        items.extend(batch)
        # The search endpoint doesn't return totalCount reliably, so we stop
        # when a page comes back smaller than PAGE_SIZE — that's the last page.
        if len(batch) < PAGE_SIZE:
            break
        page += 1

    items = items[:limit]
    return pd.DataFrame(_normalize(it) for it in items)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("title")
    p.add_argument("--country", default="", help="ISO alpha-2, e.g. DE")
    p.add_argument("--seniority", default="")
    p.add_argument("--employment-type", default="")
    p.add_argument("--worldwide-only", action="store_true")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--out", default="jobs_himalayas.parquet")
    args = p.parse_args()

    df = scrape(args.title, country=args.country, limit=args.limit,
                seniority=args.seniority,
                employment_type=args.employment_type,
                worldwide_only=args.worldwide_only)
    df.to_parquet(args.out)
    print(f"\nSaved {len(df)} rows to {args.out}")
