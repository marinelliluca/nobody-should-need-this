"""LinkedIn jobs scraper via Apify (curious_coder/linkedin-jobs-scraper).

The actor is URL-driven: you give it a LinkedIn jobs search URL, it walks the
results. To keep parity with indeed.py / xing.py (which take a `title`
keyword and a `location`), this module builds the search URL for you.

If you need a more advanced search (seniority filter, work-mode, function,
boolean operators, etc.), construct the URL by hand on linkedin.com/jobs/
and pass it via `start_url=`.

Usage:
    from linkedin import scrape
    df = scrape("data scientist", location="Berlin", limit=50)
    df.to_parquet("jobs_linkedin.parquet")
"""
from __future__ import annotations
import os
from urllib.parse import urlencode
from apify_client import ApifyClient
from dotenv import load_dotenv
import pandas as pd

from ._utils import parse_posted_at, strip_html, split_location

load_dotenv()

ACTOR_ID = "curious_coder/linkedin-jobs-scraper"

# LinkedIn's "date posted" filter values (f_TPR query param):
#   r86400    = last 24 hours
#   r604800   = last week
#   r2592000  = last month
#   ""        = any time
_DATE_POSTED_MAP = {
    "1":  "r86400",
    "3":  "r259200",     # 3 days
    "7":  "r604800",
    "14": "r1209600",    # 14 days
    "30": "r2592000",
    "":   "",
}


def _build_search_url(title: str, location: str = "", date_posted: str = "") -> str:
    """Construct a LinkedIn jobs search URL from keyword + location + recency."""
    params: dict[str, str] = {}
    if title:
        params["keywords"] = title
    if location:
        params["location"] = location
    tpr = _DATE_POSTED_MAP.get(date_posted, "")
    if tpr:
        params["f_TPR"] = tpr
    return "https://www.linkedin.com/jobs/search/?" + urlencode(params)


def _normalize(item: dict) -> dict:
    """Map raw Apify output to the unified schema used by indeed.py / xing.py."""
    description_text = item.get("descriptionText")
    if not description_text:
        description_text = strip_html(item.get("descriptionHtml"))

    city, country = _split_location(item.get("location") or "")

    return {
        "key": item.get("id"),
        "platform_url": item.get("link"),
        "job_url": item.get("applyUrl") or item.get("link"),
        "title": item.get("title"),
        "employer_name": item.get("companyName"),
        "employer_ratings_count": None,    # not exposed on LinkedIn jobs API
        "employer_rating": None,           # not exposed
        "country": country,
        "city": city,
        "posted_at": parse_posted_at(item.get("postedAt")),
        "employment_type": item.get("employmentType"),
        "description": description_text,
        "raw": item,
    }


def scrape(title: str, location: str = "", country: str = "",
           limit: int = 100, date_posted: str = "",
           start_url: str = "", split_by_location: bool = False) -> pd.DataFrame:
    """Run the actor, return a DataFrame.

    Args:
        title: search keywords (ignored if start_url is set)
        location: city or region (ignored if start_url is set)
        country: 2-letter code, used only when split_by_location=True to
            generate per-city URLs and bypass LinkedIn's 1000-job cap
        limit: max results
        date_posted: "1", "3", "7", "14", "30", or "" for any time
        start_url: optional hand-crafted LinkedIn jobs search URL; when set,
            title/location/date_posted are ignored
        split_by_location: enable the actor's per-country URL splitting to
            scrape >1000 results in a single country
    """
    client = ApifyClient(os.environ["APIFY_TOKEN"])

    if not start_url:
        start_url = _build_search_url(title, location, date_posted)

    run_input: dict = {
        "urls": [start_url],
        "count": limit,
    }
    if split_by_location and country:
        run_input["splitByLocation"] = True
        run_input["country"] = country

    run = client.actor(ACTOR_ID).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    return pd.DataFrame(_normalize(it) for it in items)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("title")
    p.add_argument("--location", default="")
    p.add_argument("--country", default="")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--date-posted", default="")
    p.add_argument("--start-url", default="")
    p.add_argument("--split-by-location", action="store_true")
    p.add_argument("--out", default="jobs_linkedin.parquet")
    args = p.parse_args()

    df = scrape(args.title, location=args.location, country=args.country,
                limit=args.limit, date_posted=args.date_posted,
                start_url=args.start_url,
                split_by_location=args.split_by_location)
    df.to_parquet(args.out)
    print(f"\nSaved {len(df)} rows to {args.out}")
