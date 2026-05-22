"""Xing jobs scraper via Apify (shahidirfan/xing-jobs-scraper).

Usage:
    from xing import scrape
    df = scrape("data scientist", location="Berlin", limit=50)
    df.to_parquet("jobs_xing.parquet")
"""
from __future__ import annotations
import os
import re
from apify_client import ApifyClient
from dotenv import load_dotenv
import pandas as pd

from ._utils import parse_posted_at, strip_html

load_dotenv()

ACTOR_ID = "shahidirfan/xing-jobs-scraper"


def _normalize(item: dict) -> dict:
    """Map raw Apify output to a flat unified dict matching indeed.py's schema.

    Column names overlap with indeed._normalize where the concept matches:
    key, title, employer_name, country, city, posted_at, employment_type,
    description, raw. URLs are platform_url (the Xing listing) and job_url
    (the apply URL, which often points to the employer's ATS).

    `source_platform` is hardcoded to "Xing": this actor only scrapes Xing,
    but the orchestrator's dedupe step expects every row to carry the real
    job-board name in this column (alljobs sets it per-row from the actor
    output, single-board actors set it as a constant). Keeping the
    convention symmetric means dedupe can rely on `source_platform` alone
    without per-actor fallbacks.
    """
    description_text = item.get("description_text")
    if not description_text:
        description_text = strip_html(item.get("description_html"))

    return {
        "platform_url": item.get("url"),
        "job_url": item.get("apply_url") or item.get("url"),
        "title": item.get("title"),
        "employer_name": item.get("company"),
        "employer_ratings_count": None,  # Xing doesn't expose this
        "employer_rating": None,         # Xing doesn't expose this
        "country": item.get("location_country") or item.get("location_country_code") or "",
        "city": item.get("location") or item.get("company_city") or "",
        "posted_at": parse_posted_at(item.get("date_posted")),
        "employment_type": item.get("job_type") or item.get("employment_type_id"),
        "description": description_text,
        "source_platform": "Xing",
        "raw": item,
    }


def scrape(title: str, location: str = "", discipline: str = "",
           limit: int = 100, max_pages: int = 20,
           start_url: str = "") -> pd.DataFrame:
    """Run the actor, return a DataFrame.

    Args:
        title: search keyword / role title (maps to actor's `keyword`)
        location: city or region
        discipline: professional field filter (e.g. "IT and Software Development")
        limit: max results (maps to actor's `results_wanted`)
        max_pages: max result pages to process
        start_url: optional direct Xing search URL; when set, the actor
            derives filters from it and keyword/location/discipline are ignored
    """
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    run_input: dict = {
        "results_wanted": limit,
        "max_pages": max_pages,
    }
    if start_url:
        run_input["startUrl"] = start_url
    else:
        if title:
            run_input["keyword"] = title
        if location:
            run_input["location"] = location
        if discipline:
            run_input["discipline"] = discipline

    run = client.actor(ACTOR_ID).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    return pd.DataFrame(_normalize(it) for it in items)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("title")
    p.add_argument("--location", default="")
    p.add_argument("--discipline", default="")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-pages", type=int, default=20)
    p.add_argument("--start-url", default="")
    p.add_argument("--out", default="jobs_xing.parquet")
    args = p.parse_args()

    df = scrape(args.title, location=args.location, discipline=args.discipline,
                limit=args.limit, max_pages=args.max_pages, start_url=args.start_url)
    df.to_parquet(args.out)
    print(f"\nSaved {len(df)} rows to {args.out}")