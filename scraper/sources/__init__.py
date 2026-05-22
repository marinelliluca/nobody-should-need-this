"""Job-board scrapers. Each module exposes a `scrape()` returning a DataFrame
with the unified schema:

    key, platform_url, job_url, title, employer_name,
    employer_ratings_count, employer_rating, country, city,
    posted_at, employment_type, description, raw
"""
from . import alljobs, xing

ACTOR_NAMES = ["alljobs", "xing"]

__all__ = ["alljobs", "xing", "ACTOR_NAMES"]
