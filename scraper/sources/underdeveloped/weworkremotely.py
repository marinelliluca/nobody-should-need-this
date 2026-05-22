"""WeWorkRemotely jobs scraper via Apify (crawlerbros/weworkremotely-job-scraper).

NB: this is deliberately not integrated in main.py

WWR is remote-only, so a few unified-schema fields are awkward:
  - `city`: WWR has no posting city. We use `companyHq` (where the company
    is based) as the closest meaningful value, falling back to "".
  - `country`: WWR exposes `applicantCountries` (a list of eligible regions
    for the role). We take the first entry, since most listings are scoped
    to one region.

The actor is either category-URL driven OR search-keyword driven. To stay
in parity with indeed.py / xing.py / linkedin.py (which take a `title`
keyword), this module passes `title` through as `searchTerm`. If you need
to pull a specific category (e.g. all "remote programming jobs"), pass
`category_urls=` directly.

Usage:
    from weworkremotely import scrape
    df = scrape("data scientist", limit=50)
    df.to_parquet("jobs_wwr.parquet")
"""
from __future__ import annotations
import os
from apify_client import ApifyClient
from dotenv import load_dotenv
import pandas as pd

from ._utils import parse_posted_at, strip_html

load_dotenv()

ACTOR_ID = "crawlerbros/weworkremotely-job-scraper"


def _first_or_empty(value) -> str:
    """Return the first element of a list-like, or the value itself if scalar,
    or "" if missing. WWR returns some fields as either str or list[str]
    depending on the JSON-LD shape of the source listing."""
    if value is None:
        return ""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _normalize(item: dict) -> dict:
    """Map raw Apify output to the unified schema used by the other sources."""
    description_text = item.get("descriptionText")
    if not description_text:
        description_text = strip_html(item.get("descriptionHtml"))

    # `applicantCountries` is the eligible-region list (e.g. ["USA", "Europe"]).
    # Take the first entry as our `country` value. If empty, fall back to
    # whatever's after the last comma in companyHq (e.g. "Berlin, Germany" -> "Germany").
    country = _first_or_empty(item.get("applicantCountries"))
    if not country:
        hq = item.get("companyHq") or ""
        if "," in hq:
            country = hq.rsplit(",", 1)[-1].strip()

    # WWR is remote-only: there's no posting "city". companyHq is the closest
    # meaningful thing (where the employer is based). It often contains a
    # full "City, Country" string, so strip to just the city part when possible.
    hq = item.get("companyHq") or ""
    city = hq.split(",", 1)[0].strip() if hq else ""

    return {
        "key": item.get("id"),
        "platform_url": item.get("url"),
        "job_url": item.get("applyUrl") or item.get("url"),
        "title": item.get("title"),
        "employer_name": item.get("company"),
        "employer_ratings_count": None,  # WWR doesn't expose this
        "employer_rating": None,         # WWR doesn't expose this
        "country": country,
        "city": city,
        "posted_at": parse_posted_at(item.get("postedAt")),
        "employment_type": item.get("employmentTypeNormalized") or item.get("employmentType"),
        "description": description_text,
        "raw": item,
    }


def scrape(title: str = "", limit: int = 100,
           category_urls: list[str] | None = None,
           regions: list[str] | None = None,
           job_types: list[str] | None = None,
           min_salary: int = 0,
           include_description: bool = True,
           clean_html: bool = True) -> pd.DataFrame:
    """Run the actor, return a DataFrame.

    Args:
        title: search keyword (maps to actor's `searchTerm`). Ignored when
            `category_urls` is provided.
        limit: max results (maps to actor's `maxItems`)
        category_urls: list of WWR category or search URLs (overrides title)
        regions: optional region substrings, case-insensitive, e.g. ["USA", "Europe"]
        job_types: optional employment-type filter, e.g. ["full-time", "contract"]
        min_salary: min USD salary; only applied to listings with numeric salary
        include_description: fetch full job-detail pages (default True)
        clean_html: strip scripts/tracking pixels from descriptionHtml
    """
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    run_input: dict = {
        "maxItems": limit,
        "includeDescription": include_description,
        "cleanHtml": clean_html,
    }
    if category_urls:
        run_input["categoryUrls"] = category_urls
    elif title:
        run_input["searchTerm"] = title
    if regions:
        run_input["regions"] = regions
    if job_types:
        run_input["jobTypes"] = job_types
    if min_salary:
        run_input["minSalary"] = min_salary

    run = client.actor(ACTOR_ID).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

    # The actor emits a single `job_wwr_blocked` sentinel row when a search
    # returns zero matches, so downstream pipelines never see an empty dataset.
    # Drop it here so our DataFrame is genuinely empty in that case.
    items = [it for it in items if it.get("type") != "job_wwr_blocked"]

    return pd.DataFrame(_normalize(it) for it in items)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("title", nargs="?", default="",
                   help="search keyword (ignored if --category-url given)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--category-url", action="append", default=[],
                   help="WWR category URL (repeatable; overrides title)")
    p.add_argument("--region", action="append", default=[],
                   help="region substring filter (repeatable), e.g. --region USA")
    p.add_argument("--job-type", action="append", default=[],
                   help="employment-type filter (repeatable): full-time, contract, part-time")
    p.add_argument("--min-salary", type=int, default=0)
    p.add_argument("--no-description", action="store_true",
                   help="skip per-job detail fetch (faster, less data)")
    p.add_argument("--out", default="jobs_weworkremotely.parquet")
    args = p.parse_args()

    df = scrape(args.title, limit=args.limit,
                category_urls=args.category_url or None,
                regions=args.region or None,
                job_types=args.job_type or None,
                min_salary=args.min_salary,
                include_description=not args.no_description)
    df.to_parquet(args.out)
    print(f"\nSaved {len(df)} rows to {args.out}")
