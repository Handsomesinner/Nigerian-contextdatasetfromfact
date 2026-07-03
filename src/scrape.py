"""
Scaffold for collecting live fact-check articles to *extend* the curated corpus.

This is intentionally a scaffold, not an automatic crawler:

  * Always read and respect each site's robots.txt and terms of use.
  * Fact-check article text is copyrighted; store only what you are permitted to
    (typically the claim, the verdict, the URL and the publication date), and
    attribute the source.
  * Rate-limit requests and identify your client honestly.

Each fact-checker publishes a machine-readable verdict using the ClaimReview
schema (https://schema.org/ClaimReview) embedded as JSON-LD in the page, which is
the cleanest field to parse for a label. `parse_claimreview` extracts it.

Usage (pseudo):
    urls = discover_urls("https://africacheck.org/fact-checks")
    rows = [parse_claimreview(fetch(u)) for u in urls]
    -> normalise verdicts to {false, misleading, true} and append to the CSV.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Dict

# Mapping of common textualRating strings -> our label scheme.
VERDICT_MAP = {
    "false": "false", "incorrect": "false", "fake": "false", "hoax": "false",
    "misleading": "misleading", "mixture": "misleading", "partly false": "misleading",
    "exaggerated": "misleading", "unproven": "misleading", "misattributed": "misleading",
    "true": "true", "correct": "true", "mostly true": "true", "accurate": "true",
}

SOURCES = {
    "africacheck": "https://africacheck.org",
    "dubawa": "https://dubawa.org",
    "factcheckhub": "https://factcheckhub.com",
    "pesacheck": "https://pesacheck.org",
}

JSONLD_RE = re.compile(
    r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE)


def normalise_verdict(text_rating: str) -> Optional[str]:
    t = (text_rating or "").strip().lower()
    for key, val in VERDICT_MAP.items():
        if key in t:
            return val
    return None


def parse_claimreview(html: str) -> Optional[Dict]:
    """Extract a ClaimReview JSON-LD block from an article's HTML."""
    for block in JSONLD_RE.findall(html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        candidates = data if isinstance(data, list) else [data]
        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            if obj.get("@type") == "ClaimReview":
                rating = (obj.get("reviewRating") or {}).get("alternateName") \
                    or (obj.get("reviewRating") or {}).get("ratingValue")
                return {
                    "claim": obj.get("claimReviewed"),
                    "verdict": normalise_verdict(str(rating)),
                    "raw_rating": rating,
                    "url": obj.get("url"),
                    "author": (obj.get("author") or {}).get("name"),
                }
    return None


def fetch(url: str, timeout: int = 20) -> str:  # pragma: no cover - network
    """Fetch a URL honestly. Requires network access and requests installed."""
    import requests
    headers = {"User-Agent": "nigerian-misinfo-research/1.0 (+contact)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


if __name__ == "__main__":
    print("Scaffold only. See module docstring; respect robots.txt and terms of use.")
    print("Known ClaimReview sources:", ", ".join(SOURCES))
