"""All-jobs scraper via Apify (agentx/all-jobs-scraper).

A single Apify actor that unifies LinkedIn, Indeed, Glassdoor, ZipRecruiter,
and a few regional boards (Naukri, Bayt, Bdjobs). Replaces the per-platform
indeed.py and linkedin.py wiring in main.py.

The actor itself emits a `platform` field per row identifying which board the
listing came from (e.g. "LinkedIn", "Indeed"). We preserve that as
`source_platform` so downstream dedupe / `found_on` logic still distinguishes
cross-posted jobs across LinkedIn vs Indeed vs Glassdoor.

Usage:
    from alljobs import scrape
    df = scrape("data scientist", location="Berlin", country="de", limit=50)
    df.to_parquet("jobs_alljobs.parquet")
"""
from __future__ import annotations
import hashlib
import os
from apify_client import ApifyClient
from dotenv import load_dotenv
import pandas as pd

from ._utils import parse_posted_at, split_location, strip_html

load_dotenv()

ACTOR_ID = "agentx/all-jobs-scraper"

# The actor wants full country names ("Germany"), not the 2-letter codes the
# rest of the codebase uses. Keep the CLI surface stable by mapping here.
# Codes for the countries the actor supports; extend as needed.
_COUNTRY_CODE_TO_NAME = {
    "ar": "Argentina", "au": "Australia", "at": "Austria",
    "bh": "Bahrain", "bd": "Bangladesh", "be": "Belgium",
    "bg": "Bulgaria", "br": "Brazil", "ca": "Canada",
    "cl": "Chile", "cn": "China", "co": "Colombia",
    "cr": "Costa Rica", "hr": "Croatia", "cy": "Cyprus",
    "cz": "Czech Republic", "dk": "Denmark", "ec": "Ecuador",
    "eg": "Egypt", "ee": "Estonia", "fi": "Finland",
    "fr": "France", "de": "Germany", "gr": "Greece",
    "hk": "Hong Kong", "hu": "Hungary", "in": "India",
    "id": "Indonesia", "ie": "Ireland", "il": "Israel",
    "it": "Italy", "jp": "Japan", "kw": "Kuwait",
    "lv": "Latvia", "lt": "Lithuania", "lu": "Luxembourg",
    "my": "Malaysia", "mt": "Malta", "mx": "Mexico",
    "ma": "Morocco", "nl": "Netherlands", "nz": "New Zealand",
    "ng": "Nigeria", "no": "Norway", "om": "Oman",
    "pk": "Pakistan", "pa": "Panama", "pe": "Peru",
    "ph": "Philippines", "pl": "Poland", "pt": "Portugal",
    "qa": "Qatar", "ro": "Romania", "sa": "Saudi Arabia",
    "sg": "Singapore", "sk": "Slovakia", "si": "Slovenia",
    "za": "South Africa", "kr": "South Korea", "es": "Spain",
    "se": "Sweden", "ch": "Switzerland", "tw": "Taiwan",
    "th": "Thailand", "tr": "Turkey", "ua": "Ukraine",
    "ae": "United Arab Emirates", "gb": "United Kingdom",
    "uk": "United Kingdom", "us": "United States",
    "uy": "Uruguay", "ve": "Venezuela", "vn": "Vietnam",
}


def _resolve_country(country: str) -> str:
    """Accept either a 2-letter code ('de') or a full name ('Germany')."""
    if not country:
        return ""
    key = country.strip().lower()
    return _COUNTRY_CODE_TO_NAME.get(key, country)


def _days_to_posted_since(date_posted: str) -> str:
    """Translate the existing '--date-posted 14' day-count convention into the
    actor's freeform `posted_since` string. Empty input means no filter."""
    if not date_posted:
        return ""
    s = str(date_posted).strip()
    if not s:
        return ""
    # If the caller already passed something like '6 months', forward as-is.
    if not s.isdigit():
        return s
    n = int(s)
    return f"{n} day{'s' if n != 1 else ''}"


def _normalize(item: dict) -> dict:
    """Map raw actor output to the unified schema used by xing.py et al.

    The actor's own `platform` field (LinkedIn, Indeed, ...) is kept as
    `source_platform`, distinct from the orchestrator-level `platform`
    column that main.py adds ("alljobs"). This lets the dedupe step still
    track per-source provenance via `found_on`.
    """
    description = item.get("description")
    if description:
        # Some sources return HTML in `description`; strip defensively.
        description = strip_html(description) if "<" in description else description

    city, country = split_location(item.get("location") or "")

    return {
        "platform_url": item.get("platform_url"),
        "job_url": item.get("official_url") or item.get("platform_url"),
        "title": item.get("title"),
        "employer_name": item.get("company_name"),
        "employer_ratings_count": item.get("review_count"),
        "employer_rating": item.get("company_rating"),
        "country": country,
        "city": city,
        "posted_at": parse_posted_at(item.get("posted_date")),
        "employment_type": item.get("job_type"),
        "description": description,
        "source_platform": item.get("platform"),
        "raw": item,
    }


def scrape(title: str, location: str = "", country: str = "",
           limit: int = 100, date_posted: str = "",
           remote_only: bool = False, distance: int | None = None,
           job_type: str = "", currency: str = "") -> pd.DataFrame:
    """Run the actor, return a DataFrame.

    Args:
        title: search keyword (maps to actor's `keyword`, required)
        location: city or region
        country: country name ('Germany') or 2-letter code ('de'); required
            by the actor, so an empty string will raise
        limit: max results (10-10000; the actor enforces a minimum of 10)
        date_posted: either a day-count string like "14" (matching the existing
            Indeed/LinkedIn convention) or the actor's native form like
            "6 months". Empty means no filter.
        remote_only: restrict to remote-friendly listings
        distance: search radius in miles (50-10000)
        job_type: one of "all", "fulltime", "parttime", "internship",
            "contract"; empty means no filter
        currency: ISO code for salary FX normalization (e.g. "USD")
    """
    resolved_country = _resolve_country(country)
    if not resolved_country:
        raise ValueError("alljobs scraper requires a country (e.g. 'de' or 'Germany')")
    if not title:
        raise ValueError("alljobs scraper requires a non-empty title")

    client = ApifyClient(os.environ["APIFY_TOKEN"])
    run_input: dict = {
        "keyword": title,
        "country": resolved_country,
        # Actor minimum is 10; bump up silently rather than failing the run.
        "max_results": max(int(limit), 10),
    }
    if location:
        run_input["location"] = location
    if remote_only:
        run_input["remote_only"] = True
    if distance is not None:
        run_input["distance"] = distance
    if job_type:
        run_input["job_type"] = job_type
    if currency:
        run_input["currency"] = currency

    posted_since = _days_to_posted_since(date_posted)
    if posted_since:
        run_input["posted_since"] = posted_since

    run = client.actor(ACTOR_ID).call(run_input=run_input)
    if run is None:                      # .call() returns None on a failed run
        raise RuntimeError(f"{ACTOR_ID} run failed (returned None)")

    # apify-client switched dict -> typed Run object; support both.
    dataset_id = run["defaultDatasetId"] if isinstance(run, dict) else run.default_dataset_id
    items = list(client.dataset(dataset_id).iterate_items())
    return pd.DataFrame(_normalize(it) for it in items)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("title")
    p.add_argument("--location", default="")
    p.add_argument("--country", default="de")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--date-posted", default="")
    p.add_argument("--remote-only", action="store_true")
    p.add_argument("--distance", type=int, default=None)
    p.add_argument("--job-type", default="")
    p.add_argument("--currency", default="")
    p.add_argument("--out", default="jobs_alljobs.parquet")
    args = p.parse_args()

    df = scrape(args.title, location=args.location, country=args.country,
                limit=args.limit, date_posted=args.date_posted,
                remote_only=args.remote_only, distance=args.distance,
                job_type=args.job_type, currency=args.currency)
    df.to_parquet(args.out)
    print(f"\nSaved {len(df)} rows to {args.out}")