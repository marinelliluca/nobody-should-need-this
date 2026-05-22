"""Indeed jobs scraper via Apify (valig/indeed-jobs-scraper).

Usage:
    from indeed import scrape
    df = scrape("data scientist", location="Berlin", country="de", limit=50)
    df.to_parquet("jobs.parquet")
"""
from __future__ import annotations
import os
from apify_client import ApifyClient
from dotenv import load_dotenv
import pandas as pd

from ._utils import parse_posted_at, strip_html

load_dotenv()

ACTOR_ID = "valig/indeed-jobs-scraper"


def _normalize(item: dict) -> dict:
    """Map raw Apify output to a flat unified dict. `description` is the full body."""
    employer = item.get("employer") or {}
    location = item.get("location") or {}
    job_types = item.get("jobTypes") or {}
    description = item.get("description") or {}
    return {
        "key": item.get("key"),
        "platform_url": item.get("url"),
        "job_url": item.get("jobUrl"),
        "title": item.get("title"),
        "employer_name": employer.get("name"),
        "employer_ratings_count": employer.get("ratingsCount"),
        "employer_rating": employer.get("ratingsValue"),
        "country": location.get("countryName") or "",
        "city": location.get("city") or "",
        "posted_at": parse_posted_at(item.get("dateOnIndeed")),
        "employment_type": job_types.get("CF3CP"), # internal tag for employment type on indeed
        "description": description.get("text") or strip_html(description.get("html")),
        "raw": item,
    }

def scrape(title: str, location: str = "", country: str = "de",
           limit: int = 100, date_posted: str = "") -> pd.DataFrame:
    """Run the actor, return a DataFrame.

    Args:
        title: search keywords
        location: city, region, or "remote"
        country: 2-letter code (de, us, uk, ...)
        limit: max results (1-1000)
        date_posted: "1", "3", "7", "14" days, or "" for any
    """
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    run_input = {"title": title, "location": location, "country": country, "limit": limit}
    if date_posted:
        run_input["datePosted"] = date_posted

    run = client.actor(ACTOR_ID).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    return pd.DataFrame(_normalize(it) for it in items)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("title")
    p.add_argument("--location", default="")
    p.add_argument("--country", default="de")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--out", default="jobs_indeed.parquet")
    args = p.parse_args()

    df = scrape(args.title, args.location, args.country, args.limit)
    #rint(df[["title", "company", "location"]].to_string())
    df.to_parquet(args.out)
    print(f"\nSaved {len(df)} rows to {args.out}")
