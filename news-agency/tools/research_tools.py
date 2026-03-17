import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from state import SourceItem

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")


class ResearchToolError(RuntimeError):
    pass


def _iso_date_days_ago(days_ago: int) -> str:
    now = datetime.now(timezone.utc)
    target = now - timedelta(days=days_ago)
    return target.strftime("%Y-%m-%d")


def _normalize_source_item(item: dict[str, Any], source_type: str) -> SourceItem:
    return {
        "title": str(item.get("title", "")).strip(),
        "url": str(item.get("url", "")).strip(),
        "snippet": str(
            item.get("content")
            or item.get("snippet")
            or item.get("description")
            or ""
        ).strip(),
        "source_type": source_type,
    }


def search_news(query: str, max_results: int = 5) -> list[SourceItem]:
    """
    Search the web for relevant sources using Tavily Search.
    """
    if not TAVILY_API_KEY:
        raise ResearchToolError("Missing TAVILY_API_KEY")

    payload = {
        "query": query,
        "topic": "news",
        "max_results": max_results,
        "search_depth": "advanced",
        "include_raw_content": False,
        "include_answer": False,
    }

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.post(
            "https://api.tavily.com/search",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    normalized: list[SourceItem] = []

    for item in results:
        normalized.append(
            _normalize_source_item(
                item,
                source_type="official" if _looks_official(item.get("url", "")) else "news",
            )
        )

    return normalized


def search_headlines(
    query: str,
    days_back: int = 3,
    page_size: int = 5,
) -> list[SourceItem]:
    """
    Search recent news articles using NewsAPI Everything endpoint.
    """
    if not NEWSAPI_API_KEY:
        raise ResearchToolError("Missing NEWSAPI_API_KEY")

    params = {
        "q": query,
        "from": _iso_date_days_ago(days_back),
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": NEWSAPI_API_KEY,
    }

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        data = response.json()

    if data.get("status") != "ok":
        raise ResearchToolError(f"NewsAPI error: {data}")

    articles = data.get("articles", [])
    normalized: list[SourceItem] = []

    for article in articles:
        item = {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "description": article.get("description", ""),
        }
        normalized.append(_normalize_source_item(item, source_type="news"))

    return normalized


def fetch_page(url: str, max_chars: int = 8000) -> str:
    """
    Fetch a page and extract readable text content with httpx + BeautifulSoup.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        )
    }

    with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        html = response.text

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "iframe"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    text = " ".join(text.split())

    if not text:
        raise ResearchToolError(f"No readable content extracted from {url}")

    return text[:max_chars]


def _looks_official(url: str) -> bool:
    url = url.lower()
    official_markers = [
        ".gov/",
        ".gov",
        ".edu/",
        ".edu",
        "/blog",
        "openai.com",
        "anthropic.com",
        "google.com",
        "microsoft.com",
        "meta.com",
    ]
    return any(marker in url for marker in official_markers)