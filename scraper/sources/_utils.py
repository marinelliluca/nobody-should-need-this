"""Shared helpers for source normalizers."""
from __future__ import annotations
import re
from datetime import datetime, timedelta, timezone
from html import unescape
import hashlib
import pandas as pd

def _make_key(row):
    """Compute hash for df indexing.

    `found_on` can be either a numpy array (when pandas promoted the list
    column during groupby/merge) or a plain Python list (for unkeyed rows
    that took the `.assign(found_on=...map(lambda p: [p]))` path). Handle
    both -- `.tolist()` only exists on the ndarray.
    """
    found_on = row.get("found_on")
    if hasattr(found_on, "tolist"):
        found_on_list = found_on.tolist()
    elif found_on is None:
        found_on_list = []
    else:
        found_on_list = list(found_on)

    parts = [
        " ".join(found_on_list),
        row.get("platform_url"),
        row.get("title"),
        row.get("employer_name") or "",
    ]
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:32]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    """Strip HTML tags, unescape entities, and collapse whitespace."""
    if not text:
        return ""
    return _WS_RE.sub(" ", unescape(_TAG_RE.sub(" ", text))).strip()


# Country normalization: 'Deutschland' / 'DE' / 'germany' / 'Germany' should
# all resolve to a single canonical form so cross-locale duplicates collapse
# during dedupe. We use Babel's CLDR territory data as the backbone.
#
# In the followin, a *locale* is a language (de = German, ja = Japanese), a
# *country* is a territory (DE = Germany, JP = Japan). They are independent
# axes -- one locale renders every country's name in that language. We walk
# a set of source languages we expect to see in incoming job data, and from
# each one we harvest the localized name for every country. So including
# 'de' adds the German names of ALL countries ('Deutschland', 'Frankreich',
# 'Spanien', ...), not just Germany. CLDR doesn't list informal
# abbreviations like 'UK' or 'USA' though, so we layer a small supplement
# on top.
from babel import Locale
 
# Source languages we recognize country names from. Every locale here adds
# ~250 country-name aliases to the table, built once at import time, and
# each one (very slightly) raises the risk of an ambiguous alias collision.
# This tool is DACH/English-focused, so the list is scoped to the languages 
# we actually see in incoming job data rather than all of CLDR.
_CLDR_LOCALES = (
    "de",  # primary: German country names (Deutschland, Frankreich, ...)
    "en",  # primary: English canonical names + ISO codes
    # DACH neighbours / commonly cross-posted European languages
    "fr", "it", "nl", "es", "pl",
)
# The hand-curated `supplement` below covers the high-frequency informal 
# cases (UK / USA / etc.) that CLDR omits
 
def _build_alias_map() -> dict[str, str]:
    """Build {lowercased alias -> canonical English country name} from CLDR
    plus a hand-curated supplement for abbreviations CLDR doesn't list."""
    en = Locale("en")
    en_names = {iso: name for iso, name in en.territories.items()
                if len(iso) == 2 and iso.isalpha()}
 
    # ISO code (de, DE) -> English name.
    aliases: dict[str, str] = {iso.lower(): name for iso, name in en_names.items()}
 
    # Localized name (Deutschland, España, 日本) -> English name.
    for loc_code in _CLDR_LOCALES:
        try:
            loc = Locale(loc_code)
        except Exception:
            continue
        for iso, name in loc.territories.items():
            if len(iso) == 2 and iso.isalpha() and iso in en_names:
                aliases[name.lower()] = en_names[iso]
 
    # Hand-curated supplement: informal abbreviations and common alt-spellings
    # that CLDR doesn't include. Maps to canonical English name.
    supplement = {
        "uk": "United Kingdom", "u.k.": "United Kingdom",
        "great britain": "United Kingdom", "england": "United Kingdom",
        "scotland": "United Kingdom", "wales": "United Kingdom",
        "u.s.": "United States", "u.s.a.": "United States",
        "us": "United States", "america": "United States",
        "uae": "United Arab Emirates",
        "holland": "Netherlands",
        "czechia": "Czech Republic",
        "south korea": "South Korea", "korea": "South Korea",
    }
    aliases.update(supplement)
    return aliases
 
 
_COUNTRY_ALIASES: dict[str, str] = _build_alias_map()
 
 
def normalize_country(name: str | None) -> str:
    """Return the canonical English country name for any alias we recognize;
    otherwise return the input stripped (so unknown countries pass through)."""
    if not name:
        return ""
    return _COUNTRY_ALIASES.get(
        name.strip().lower(), 
        name.strip() # unknown key default: pass through stripped
    )
 
 
def split_location(loc: str) -> tuple[str, str]:
    """Split a free-text location like 'Berlin, Germany' or 'Greater Berlin
    Area' into (city, country). Used by sources that return location as one
    string (LinkedIn, the unified all-jobs actor).
 
    Handles three shapes:
      - 'City, Country' / 'City, Region, Country' -> last comma-piece is
        the country, first is the city
      - 'Country' alone -> ('', 'Country') (was previously misclassified
        as a city, causing cross-locale dedupe collisions like
        'Deutschland' vs 'Germany')
      - 'City' alone -> ('City', '')
 
    Country names are normalized to canonical English form (Deutschland ->
    Germany, USA -> United States, etc.) via `_COUNTRY_ALIASES`.
    """
    if not loc:
        return "", ""
    if isinstance(loc, dict):
        loc = loc["raw"]
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    if not parts:
        return "", ""
    if len(parts) >= 2:
        return parts[0], normalize_country(parts[-1])
    # Single token: country-only if we recognize it, else assume city.
    only = parts[0]
    if only.lower() in _COUNTRY_ALIASES:
        return "", normalize_country(only)
    return only, ""
 
 
_REL_RE = re.compile(
    r"(\d+)\s*(minute|hour|day|week|month|year)s?\s*ago",
    re.IGNORECASE,
)
_REL_UNITS = {
    "minute": "minutes",
    "hour":   "hours",
    "day":    "days",
    "week":   "weeks",
    "month":  "days",   # approximated as 30 days
    "year":   "days",   # approximated as 365 days
}
_REL_MULT = {"month": 30, "year": 365}
 
 
def parse_posted_at(value, now: datetime | None = None) -> str | None:
    """Normalize a posted-at value from any source to 'YY/MM/DD', or None.
 
    Accepts:
      - epoch seconds or milliseconds (int / float / numeric string)
      - ISO 8601 string ('2026-05-10', '2026-05-10T14:23:00Z', ...)
      - English relative string ('3 days ago', 'just now', 'yesterday')
      - None / NaN / empty
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
 
    now = now or datetime.now(timezone.utc)
 
    # Numeric: epoch seconds or milliseconds.
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:        # milliseconds
            ts /= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%y/%m/%d")
 
    s = str(value).strip()
    if not s:
        return None
 
    # Relative string.
    low = s.lower()
    if low in {"just now", "today"}:
        return now.strftime("%y/%m/%d")
    if low == "yesterday":
        return (now - timedelta(days=1)).strftime("%y/%m/%d")
 
    m = _REL_RE.search(low)
    if m:
        n = int(m.group(1)) * _REL_MULT.get(m.group(2).lower(), 1)
        delta = timedelta(**{_REL_UNITS[m.group(2).lower()]: n})
        return (now - delta).strftime("%y/%m/%d")
 
    # Numeric string (epoch).
    if s.lstrip("-").isdigit():
        return parse_posted_at(int(s), now=now)
 
    # Fall through to pandas' ISO/loose parser.
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.strftime("%y/%m/%d")
