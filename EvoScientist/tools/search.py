"""Web search tools.

Provides ``tavily_search`` for the research agent. Despite the legacy tool
name, the implementation prefers Exa when ``EXA_API_KEY`` is configured and
falls back to Tavily when only ``TAVILY_API_KEY`` is available.
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Literal

import httpx
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - optional dependency
    TavilyClient = None

_tavily_client = None


def _get_tavily_client() -> TavilyClient:
    """Get or create the Tavily client (lazy initialization)."""
    global _tavily_client
    if TavilyClient is None:
        raise RuntimeError(
            "Tavily support is not installed. Set EXA_API_KEY or install tavily."
        )
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


def _get_active_search_provider() -> Literal["exa", "tavily"] | None:
    """Return the active search provider based on configured API keys."""
    if os.environ.get("EXA_API_KEY"):
        return "exa"
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    return None


def _exa_category_for_topic(topic: Literal["general", "news", "finance"]) -> str | None:
    """Map the legacy topic argument onto Exa's documented categories."""
    if topic == "news":
        return "news"
    if topic == "finance":
        return "financial report"
    return None


async def _search_with_exa(
    query: str,
    max_results: int,
    topic: Literal["general", "news", "finance"],
) -> dict:
    """Search via Exa's official REST API."""
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not configured.")

    payload: dict[str, object] = {
        "query": query,
        "type": "auto",
        "numResults": max_results,
        "contents": {
            "highlights": {
                "maxCharacters": 4000,
            }
        },
    }
    category = _exa_category_for_topic(topic)
    if category:
        payload["category"] = category

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            "https://api.exa.ai/search",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {e!s}"


def _format_exa_result(result: dict) -> str:
    """Format a single Exa result into markdown-ish output."""
    lines = [
        f"## {result.get('title', 'Untitled')}",
        f"**URL:** {result.get('url', '')}",
    ]

    if published_date := result.get("publishedDate"):
        lines.append(f"**Published:** {published_date}")
    if author := result.get("author"):
        lines.append(f"**Author:** {author}")
    if summary := result.get("summary"):
        lines.extend(["", f"**Summary:** {summary}"])
    highlights = result.get("highlights") or []
    if highlights:
        lines.extend(["", "**Highlights:**"])
        lines.extend(f"- {highlight}" for highlight in highlights)
    if text := result.get("text"):
        lines.extend(["", text])
    lines.extend(["", "---", ""])
    return "\n".join(lines)


async def _search_with_tavily(
    query: str,
    max_results: int,
    topic: Literal["general", "news", "finance"],
) -> str:
    """Run the legacy Tavily-backed search path."""

    def _sync_search() -> dict:
        return _get_tavily_client().search(
            query,
            max_results=max_results,
            topic=topic,
        )

    search_results = await asyncio.to_thread(_sync_search)
    results = search_results.get("results", [])
    if not results:
        return f"No results found for '{query}'"

    fetch_tasks = [fetch_webpage_content(r["url"]) for r in results]
    contents = await asyncio.gather(*fetch_tasks)

    result_texts = []
    for result, content in zip(results, contents, strict=False):
        result_text = f"""## {result["title"]}
**URL:** {result["url"]}

{content}

---
"""
        result_texts.append(result_text)

    return f"""Found {len(result_texts)} result(s) for '{query}':

{"".join(result_texts)}"""


async def _run_search(
    query: str,
    max_results: int,
    topic: Literal["general", "news", "finance"],
) -> str:
    """Dispatch search to the configured provider."""
    provider = _get_active_search_provider()

    if provider == "exa":
        search_results = await _search_with_exa(query, max_results, topic)
        results = search_results.get("results", [])
        if not results:
            return f"No results found for '{query}'"
        formatted = "".join(_format_exa_result(result) for result in results)
        return f"Found {len(results)} result(s) for '{query}':\n\n{formatted}"

    if provider == "tavily":
        return await _search_with_tavily(query, max_results, topic)

    return "Search failed: no search provider configured. Set EXA_API_KEY or TAVILY_API_KEY."


@tool(parse_docstring=True)
async def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Exa when available, otherwise falls back to Tavily.

    Args:
        query: Search query to execute

    Returns:
        Formatted search results with extracted webpage content
    """
    try:
        return await _run_search(query, max_results=max_results, topic=topic)
    except Exception as e:
        return f"Search failed: {e!s}"
